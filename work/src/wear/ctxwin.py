"""Context window model (ours, GPU): classifies a tile from the RAW inputs of the tile and its neighbourhood - up to
+-4 tiles along our OOF links (L4) and 4 video-kNN tiles of the same subject. Each tile goes through Briano's window
encoder (IMU dilated CNN + pooled video MLP, with a limb embedding); a small transformer over the 13 tokens (tile +
neighbours, each with a relative-position / kNN-rank embedding and its link confidence) predicts the centre label.
Neighbours carry other limbs of the same activity, so the model sees a 'virtual multi-limb' window.
Trained per held-out fold on the other folds' OOF tiles (their links are OOF too); test = mean of the 5 fold models."""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as C
from .common import FOLDS, N_CLS, T_START, macro_f1, stage
from .window_pre import ConvBlock

OFFS = [-4, -3, -2, -1, 1, 2, 3, 4]


class TileEnc(nn.Module):
    """same sub-module names/shapes as window_pre.Net (d=256) so a pretrained fold model can initialise it"""

    def __init__(self, d=256):
        super().__init__()
        self.sens = nn.Embedding(4, 32)
        self.imu = nn.Sequential(ConvBlock(4 + 32, 64, 7), ConvBlock(64, 128, 5), ConvBlock(128, 128, 5, 2),
                                 ConvBlock(128, d, 3, 4), ConvBlock(d, d, 3, 8))
        self.imu_out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.vin = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, d), nn.GELU(), nn.Dropout(0.1))
        self.vid_out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.out = nn.Linear(2 * d + 32, d)

    def forward(self, a, v, sensor):
        e = self.sens(sensor)
        x = torch.cat([a.transpose(1, 2), e[:, :, None].expand(-1, -1, a.shape[1])], 1)
        h = self.imu(x)
        hi = self.imu_out(torch.cat([h.mean(-1), h.amax(-1)], 1))
        z = self.vin(v)
        hv = self.vid_out(torch.cat([z.mean(1), z.std(1)], 1))
        return self.out(torch.cat([hi, hv, e], 1))


class CtxNet(nn.Module):
    def __init__(self, d=192, n_tok=13, layers=2):
        super().__init__()
        self.enc = TileEnc(d)
        self.pos = nn.Embedding(n_tok, d)
        self.conf = nn.Linear(2, d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_CLS))
        self.aux = nn.Linear(d, N_CLS)

    def forward(self, a, v, s, mask, conf):
        """a (B,T,50,4) v (B,T,15,768) s (B,T) mask (B,T) True=missing, conf (B,T,2)"""
        B, T = s.shape
        h = self.enc(a.reshape(B * T, *a.shape[2:]), v.reshape(B * T, *v.shape[2:]), s.reshape(B * T)).view(B, T, -1)
        h = h + self.pos.weight[None, :T] + self.conf(conf)
        z = self.tf(h, src_key_padding_mask=mask)
        return self.head(z[:, 0]), self.aux(h[:, 0])


class ResCtxNet(nn.Module):
    """residual context model: the pretrained window model (Net, incl. its fusion head) gives the centre tile's
    logits; a transformer over all tokens adds a correction whose output layer starts at zero, so training starts
    exactly at the window model and can only move away from it if the neighbours help"""

    def __init__(self, d=256, n_tok=13, layers=2, dt=192):
        super().__init__()
        from .window_pre import Net
        self.net = Net(d)
        self.proj = nn.Linear(2 * d + 32, dt)
        self.pos = nn.Embedding(n_tok, dt)
        self.conf = nn.Linear(2, dt)
        layer = nn.TransformerEncoderLayer(dt, 4, 2 * dt, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(dt), nn.Linear(dt, N_CLS))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def feats(self, a, v, s):
        n = self.net
        e = n.sens(s)
        x = torch.cat([a.transpose(1, 2), e[:, :, None].expand(-1, -1, a.shape[1])], 1)
        h = n.imu(x)
        hi = n.imu_out(torch.cat([h.mean(-1), h.amax(-1)], 1))
        z = n.vin(v)
        hv = n.vid_out(torch.cat([z.mean(1), z.std(1)], 1))
        f = torch.cat([hi, hv, e], 1)
        return f, n.fuse(f)

    def forward(self, a, v, s, mask, conf):
        B, T = s.shape
        f, logits = self.feats(a.reshape(B * T, *a.shape[2:]), v.reshape(B * T, *v.shape[2:]), s.reshape(B * T))
        f = f.view(B, T, -1)
        base = logits.view(B, T, -1)[:, 0]
        h = self.proj(f) + self.pos.weight[None, :T] + self.conf(conf)
        z = self.tf(h, src_key_padding_mask=mask)
        return base + self.head(z[:, 0]), base


def neighbour_index(succ, score, emb, sbj, Pk, k_knn=4):
    """(n, 13) neighbour rows (-1 = missing) and (n, 13, 2) confidences: [self, chain offsets, kNN ranks]"""
    n = len(succ)
    pred = np.full(n, -1, np.int64)
    m = succ >= 0
    pred[succ[m]] = np.flatnonzero(m)
    lp = 1 / (1 + np.exp(-np.clip(score, -20, 20)))
    lpp = np.where(pred >= 0, lp[np.maximum(pred, 0)], 0.0)
    idx = np.full((n, 1 + len(OFFS) + k_knn), -1, np.int64)
    conf = np.zeros((n, idx.shape[1], 2), np.float32)
    idx[:, 0] = np.arange(n)
    conf[:, 0] = 1.0
    for direction, nxt, prob in ((1, succ, lp), (-1, pred, lpp)):
        cur, acc = np.arange(n), np.ones(n)
        for k in range(1, 5):
            acc = acc * np.where(cur >= 0, prob[np.maximum(cur, 0)], 0)
            cur = np.where(cur >= 0, nxt[np.maximum(cur, 0)], -1)
            col = 1 + OFFS.index(direction * k)
            idx[:, col] = cur
            conf[:, col, 0] = acc
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    Q = np.sqrt(Pk).astype(np.float32)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        S = (E[ii] @ E[ii].T + 0.5 * (Q[ii] @ Q[ii].T)) / 1.5
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k_knn, axis=1)[:, :k_knn]
        sv = np.take_along_axis(S, nb, 1)
        o = np.argsort(-sv, 1)
        idx[ii, 1 + len(OFFS):] = ii[np.take_along_axis(nb, o, 1)]
        conf[ii, 1 + len(OFFS):, 1] = np.take_along_axis(sv, o, 1)
    return idx, conf


def main(cfg):
    from .common import Paths
    from .decode import load_art
    from .base import write_sub
    from . import data
    from .window_pre import acc_stats, vid_stats, norm_stats
    from .gsweep import finish_soft, run_cfg
    P = Paths()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = torch.cuda.get_device_capability() if dev.type == "cuda" else (0, 0)
    use_amp = cap >= (7, 0)
    print("ctxwin device", dev, cap, flush=True)
    recs, sim, d, t = data.build(P, smoke=bool(cfg.get("smoke")))

    def find(name, must=None):
        hits = sorted(p for p in Path("/kaggle/input").rglob(name) if must is None or must in str(p))
        assert hits, (name, must)
        return hits[0]
    a, b = load_art(find("oof_B.npz")), load_art(find("test_B.npz"))
    lo, lt = load_art(find("oof_L3.npz", cfg.get("links_from", "wear26-l4"))), \
        load_art(find("test_L3.npz", cfg.get("links_from", "wear26-l4")))
    so, sco, st, sct = lo["L3_succ"], lo["L3_score_qn"], lt["L3_succ"], lt["L3_score_qn"]
    y, sbj = d["y"], d["sbj"]
    # normalisation exactly like the window models (per subject; test: own tiles)
    t_acc, t_vid = t["acc"](), np.asarray(t["vid"]())
    stats, test_grp = norm_stats(P, recs, t_acc, t_vid, t["sensor"], t["sbj"])
    AM, AS, VM, VS = (torch.tensor(x, dtype=torch.float32) for x in stats)

    def prep(acc, vid, sensor, grp):
        acc = torch.from_numpy(np.asarray(acc, np.float32))
        mag = acc.norm(dim=-1, keepdim=True) - 1.0
        g, s_ = torch.from_numpy(grp), torch.from_numpy(sensor)
        a_ = torch.cat([(acc - AM[g, s_][:, None]) / AS[g, s_][:, None], mag], -1)
        v_ = ((torch.from_numpy(np.asarray(vid)).float() - VM[g][:, None]) / VS[g][:, None]).half()
        return a_.float(), v_
    with stage("prepare tensors"):
        A_o, V_o = prep(d["acc"](), d["vid"](), d["sensor"], sbj.astype(np.int64))
        A_t, V_t = prep(t_acc, t_vid, t["sensor"], test_grp.astype(np.int64))
        S_o, S_t = torch.from_numpy(d["sensor"]), torch.from_numpy(t["sensor"])
        idx_o, conf_o = neighbour_index(so, sco, a["emb"].astype(np.float32), sbj, np.exp(a["logp_b"]))
        idx_t, conf_t = neighbour_index(st, sct, b["emb"].astype(np.float32), t["sbj"], np.exp(b["logp_b"]))
        print("neighbour coverage OOF", (idx_o >= 0).mean(0).round(2).tolist(), flush=True)
        # label agreement of the neighbours (diagnostic)
        agree = [(y[np.maximum(idx_o[:, c], 0)] == y)[idx_o[:, c] >= 0].mean() for c in range(idx_o.shape[1])]
        print("neighbour same-label rate", np.round(agree, 3).tolist(), flush=True)
    A_o, V_o, A_t, V_t = A_o.to(dev), V_o.to(dev), A_t.to(dev), V_t.to(dev)
    S_o, S_t = S_o.to(dev), S_t.to(dev)
    y_t = torch.as_tensor(y, device=dev)
    freq = np.bincount(y, minlength=N_CLS).astype(np.float64)
    cls_w = (1.0 / np.sqrt(freq))[y]

    def batch(idx, conf, rows, A, V, S, train):
        nb = torch.as_tensor(idx[rows], device=dev)
        miss = nb < 0
        if train:   # neighbour dropout
            miss = miss | ((torch.rand(miss.shape, device=dev) < cfg.get("nb_drop", 0.15)) &
                           (torch.arange(miss.shape[1], device=dev)[None] > 0))
        nbc = nb.clamp(min=0)
        a_ = A[nbc]
        v_ = V[nbc].float()
        if train:
            v_ = v_ + 0.1 * torch.randn_like(v_)
        cf = torch.as_tensor(conf[rows], device=dev)
        a_ = a_ * (~miss)[..., None, None]
        v_ = v_ * (~miss)[..., None, None]
        return a_, v_, S[nbc], miss, cf

    epochs, bs, lr = int(cfg.get("epochs", 8)), int(cfg.get("bs", 256)), float(cfg.get("lr", 1e-3))
    oof = np.zeros((len(y), N_CLS), np.float32)
    test = np.zeros((len(t["sbj"]), N_CLS), np.float32)
    for f in range(FOLDS):
        torch.manual_seed(f)
        rng = np.random.default_rng(f)
        tr = np.flatnonzero(d["fold"] != f)
        va = np.flatnonzero(d["fold"] == f)
        pw = cls_w[tr] / cls_w[tr].sum()
        cdf = pw.cumsum()
        cdf /= cdf[-1]
        enc_lr = lr
        if cfg.get("arch") == "res":
            model = ResCtxNet(256, dt=cfg.get("dim", 192)).to(dev)
            from .common import find_file
            root = find_file(f"{cfg['init_from']}_model_f0.pt").parent
            sd = torch.load(root / f"{cfg['init_from']}_model_f{f}.pt", map_location=dev, weights_only=True)
            model.net.load_state_dict(sd, strict=True)
            print(f"[fold {f}] residual model on top of {cfg['init_from']}_model_f{f}.pt", flush=True)
            enc_lr = float(cfg.get("enc_lr", 5e-5))
            enc_params = list(model.net.parameters())
            ids = {id(p) for p in enc_params}
            other = [p for p in model.parameters() if id(p) not in ids]
        else:
            model = CtxNet(cfg.get("dim", 256)).to(dev)
            if cfg.get("init_from"):   # pretrained fold-f window model: never saw fold f's subjects (honest OOF)
                from .common import find_file
                root = find_file(f"{cfg['init_from']}_model_f0.pt").parent
                sd = torch.load(root / f"{cfg['init_from']}_model_f{f}.pt", map_location=dev, weights_only=True)
                missing, unexpected = model.enc.load_state_dict(sd, strict=False)
                print(f"[fold {f}] encoder initialised from {cfg['init_from']}_model_f{f}.pt (new: {missing})", flush=True)
                enc_lr = float(cfg.get("enc_lr", lr / 3))
            enc_params = [p for n_, p in model.named_parameters() if n_.startswith("enc.") and not n_.startswith("enc.out")]
            other = [p for n_, p in model.named_parameters() if not (n_.startswith("enc.") and not n_.startswith("enc.out"))]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": enc_lr}, {"params": other, "lr": lr}], weight_decay=0.05)
        steps = epochs * (len(tr) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[enc_lr, lr], total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        t0 = time.time()
        for step in range(steps):
            model.train()
            rows = tr[cdf.searchsorted(rng.random(bs), side="right")]
            a_, v_, s_, m_, c_ = batch(idx_o, conf_o, rows, A_o, V_o, S_o, True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                out, aux = model(a_, v_, s_, m_, c_)
            yy = y_t[torch.as_tensor(rows, device=dev)]
            loss = F.cross_entropy(out.float(), yy, label_smoothing=0.1) + \
                0.3 * F.cross_entropy(aux.float(), yy, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if (step + 1) % 500 == 0:
                print(f"[fold {f}] step {step + 1}/{steps} loss {loss.item():.3f} [{time.time() - t0:.0f}s]", flush=True)
        model.eval()
        base_pred = np.zeros(len(va), np.int64)
        with torch.no_grad():
            for lo_, hi_ in [(i, min(i + 512, len(va))) for i in range(0, len(va), 512)]:
                rows = va[lo_:hi_]
                a_, v_, s_, m_, c_ = batch(idx_o, conf_o, rows, A_o, V_o, S_o, False)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                    out, base_ = model(a_, v_, s_, m_, c_)
                oof[rows] = F.log_softmax(out.float(), 1).cpu().numpy()
                base_pred[lo_:hi_] = base_.float().argmax(1).cpu().numpy()
        print(f"[fold {f}] centre-tile base head held-out F1 {macro_f1(y[va], base_pred):.4f}", flush=True)
        with torch.no_grad():
            for lo_ in range(0, len(t["sbj"]), 512):
                rows = np.arange(lo_, min(lo_ + 512, len(t["sbj"])))
                a_, v_, s_, m_, c_ = batch(idx_t, conf_t, rows, A_t, V_t, S_t, False)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                    out, _ = model(a_, v_, s_, m_, c_)
                test[rows] += F.softmax(out.float(), 1).cpu().numpy() / FOLDS
        print(f"[fold {f}] held-out argmax F1 {macro_f1(y[va], oof[va].argmax(1)):.4f} [{time.time() - T_START:.0f}s]",
              flush=True)
        del model, opt
    test_lp = np.log(np.clip(test, 1e-7, 1)).astype(np.float32)
    res = {"ctx_argmax": macro_f1(y, oof.argmax(1))}
    print(f"context model OOF argmax {res['ctx_argmax']:.4f} (window 0.7256, S3 0.8142, S5 0.8607)", flush=True)
    np.savez_compressed(P.ART / f"ctx_{cfg.get('tag', 'c1')}.npz", oof=oof, test=test_lp)
    # quick downstream check: tab blend with the context logp in place of S3, graph on the same links
    import json
    from .tab import tab_blend
    from .common import norm_rows
    gs = sorted(Path("/kaggle/input").rglob("gsweep.json"))
    if gs:
        top = json.loads(gs[0].read_text())["top"]
        for w in (0.4, 0.5):
            lb_o = tab_blend(a["logp"], [(oof, w), (a["tab_T"], 0.3)])
            lb_t = tab_blend(b["logp"], [(test_lp, w), (b["tab_T"], 0.3)])
            do = dict(logp=lb_o, emb=a["emb"].astype(np.float32), grp=sbj, sbj=sbj, succ=so, score=sco,
                      sets=C.TRAIN_SETS)
            dt = dict(logp=lb_t, emb=b["emb"].astype(np.float32), grp=t["sbj"], sbj=t["sbj"], succ=st, score=sct,
                      sets={})
            cache = {}
            Po = norm_rows(np.mean([run_cfg(do, c, cache) for c in top], 0))
            fsoft = macro_f1(y, finish_soft(Po, do))
            res[f"graph_w{w}"] = fsoft
            print(f"tab blend (window, ctx@{w}, T@0.3) + graph: OOF soft {fsoft:.4f}", flush=True)
            cache = {}
            Pt = norm_rows(np.mean([run_cfg(dt, c, cache) for c in top], 0))
            np.savez_compressed(P.ART / f"oof_ctx{cfg.get('tag', 'c1')}_w{w}.npz", P_graph=Po, y=y, sbj=sbj)
            np.savez_compressed(P.ART / f"test_ctx{cfg.get('tag', 'c1')}_w{w}.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"],
                                oof_f1=fsoft)
    C.save_json(P.ART / f"ctx_{cfg.get('tag', 'c1')}.json", res)
