"""Neural successor model (ours): a contrastive bi-encoder that embeds each 1-s tile twice - as a predecessor ("tail")
and as a successor ("head") - so that tail(i) . head(j) is high when j is the second right after i.

Training: per training recording, all its test-like tiles (fresh random limb per tile every epoch) form one step; the
loss is a symmetric InfoNCE over the full n x n score matrix of that recording (target: the next / previous tile).
OOF embeddings come from the fold model that never saw the subject; test embeddings from every fold model + a model on
all subjects (score matrices can be averaged later).
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FOLDS, T_START, VLEN, WIN, macro_f1, stage


def rec_tiles(P, r):
    """all test-like tiles of one training recording: video (n,15,768) f16, acc of all limbs (n,50,4,3), ok (n,4)"""
    acc = np.load(P.PROC / f"{r['stem']}_acc.npy").reshape(-1, 4, 3)
    val = np.load(P.PROC / f"{r['stem']}_valid.npy")
    vid = np.load(P.PROC / f"{r['stem']}_vid.npy", mmap_mode="r")
    s = np.arange(0, r["T"] - WIN + 1, WIN)
    s = s[s * 3 // 5 + 8 + VLEN <= r["F"]]
    miss = np.zeros((len(val) + 1, 4), np.int32)
    np.cumsum(~val, axis=0, out=miss[1:])
    ok = (miss[s + WIN] - miss[s]) == 0
    keep = ok.any(1)
    s, ok = s[keep], ok[keep]
    v = np.asarray(vid[(s * 3 // 5)[:, None] + 8 + np.arange(VLEN)], np.float16)
    a = acc[s[:, None] + np.arange(WIN)]                      # (n,50,4,3)
    # true successor inside this recording (tiles are 50 samples apart; a dropped tile breaks the chain)
    nxt = np.full(len(s), -1)
    pos = {int(x): i for i, x in enumerate(s)}
    for i, x in enumerate(s):
        nxt[i] = pos.get(int(x) + WIN, -1)
    return v, a.astype(np.float32), ok, nxt


def vid_norm_stats(v):
    """per subject: unit-normalised frames, centred and scaled by that subject's own frame statistics"""
    u = v.astype(np.float32).reshape(-1, 768)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-6
    return u.mean(0), u.std(0) + 1e-3


def imu_norm_stats(a, limbs):
    """a (n,50,3) with its limb ids -> per-limb mean/std (4,3)"""
    m, sd = np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32)
    for l in range(4):
        x = a[limbs == l].reshape(-1, 3)
        if len(x) > 500:
            m[l], sd[l] = x.mean(0), x.std(0) + 1e-3
    return m, sd


class LinkNet(nn.Module):
    def __init__(self, dv=256, de=128, dropout=0.1):
        super().__init__()
        self.vproj = nn.Sequential(nn.Linear(768, dv), nn.GELU(), nn.Dropout(dropout))
        self.vtemp = nn.Conv1d(dv, dv, 3, padding=1)
        self.limb = nn.Embedding(4, 16)
        self.imu = nn.Sequential(nn.Conv1d(7 + 16, 64, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(64, 64, 5, padding=2, dilation=1), nn.GELU())
        fin = 5 * dv + 2 * 64 + 23 + 16
        self.tail = nn.Sequential(nn.Linear(fin, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, de))
        self.head = nn.Sequential(nn.Linear(fin, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, de))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.05)))

    def feats(self, v, a, az, limb):
        """v (B,15,768) normalised video; a (B,50,3) raw g; az (B,50,3) z-scored per subject x limb; limb (B,)"""
        h = self.vproj(v)                                   # (B,15,dv)
        h = h + F.gelu(self.vtemp(h.transpose(1, 2))).transpose(1, 2)
        mag = a.norm(dim=-1, keepdim=True) - 1.0
        le = self.limb(limb)
        x = torch.cat([a, az, mag, le[:, None].expand(-1, a.shape[1], -1)], -1).transpose(1, 2)
        g = self.imu(x)                                     # (B,64,50)
        tail_v = torch.cat([h[:, -1], h[:, -3:].mean(1), h.mean(1), h[:, -1] - h[:, -4], h.std(1)], -1)
        head_v = torch.cat([h[:, 0], h[:, :3].mean(1), h.mean(1), h[:, 3] - h[:, 0], h.std(1)], -1)
        tail_i = torch.cat([g[:, :, -10:].mean(-1), g.mean(-1), a[:, -1], a[:, -3:].mean(1), a[:, -1] - a[:, -4],
                            az[:, -1], az[:, -3:].mean(1), az[:, -1] - az[:, -4], mag[:, -5:, 0]], -1)
        head_i = torch.cat([g[:, :, :10].mean(-1), g.mean(-1), a[:, 0], a[:, :3].mean(1), a[:, 3] - a[:, 0],
                            az[:, 0], az[:, :3].mean(1), az[:, 3] - az[:, 0], mag[:, :5, 0]], -1)
        t = F.normalize(self.tail(torch.cat([tail_v, tail_i, le], -1)), dim=-1)
        hd = F.normalize(self.head(torch.cat([head_v, head_i, le], -1)), dim=-1)
        return t, hd

    def tau(self):
        return self.log_tau.exp().clamp(0.01, 0.5)


class LinkNetBlocks(nn.Module):
    """cheaper variant: frame blocks are averaged in input space, then one shared projection per block"""

    def __init__(self, dv=192, de=128, dropout=0.1, ci=32):
        super().__init__()
        self.vproj = nn.Sequential(nn.Linear(768, dv), nn.GELU(), nn.Dropout(dropout))
        self.limb = nn.Embedding(4, 16)
        self.imu = nn.Sequential(nn.Conv1d(7 + 16, ci, 5, padding=2), nn.GELU(), nn.Conv1d(ci, ci, 5, padding=2),
                                 nn.GELU())
        fin = 6 * dv + 2 * ci + 23 + 16
        self.tail = nn.Sequential(nn.Linear(fin, 384), nn.GELU(), nn.Dropout(dropout), nn.Linear(384, de))
        self.head = nn.Sequential(nn.Linear(fin, 384), nn.GELU(), nn.Dropout(dropout), nn.Linear(384, de))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.05)))

    def feats(self, v, a, az, limb):
        B = v.shape[0]
        blocks_t = torch.stack([v[:, -1], v[:, -3:].mean(1), v[:, -6:-3].mean(1), v.mean(1), v[:, -1] - v[:, -4],
                                v.std(1)], 1)                     # (B,6,768)
        blocks_h = torch.stack([v[:, 0], v[:, :3].mean(1), v[:, 3:6].mean(1), v.mean(1), v[:, 3] - v[:, 0],
                                v.std(1)], 1)
        pt = self.vproj(blocks_t).reshape(B, -1)
        ph = self.vproj(blocks_h).reshape(B, -1)
        mag = a.norm(dim=-1, keepdim=True) - 1.0
        le = self.limb(limb)
        x = torch.cat([a, az, mag, le[:, None].expand(-1, a.shape[1], -1)], -1).transpose(1, 2)
        g = self.imu(x)
        tail_i = torch.cat([g[:, :, -10:].mean(-1), g.mean(-1), a[:, -1], a[:, -3:].mean(1), a[:, -1] - a[:, -4],
                            az[:, -1], az[:, -3:].mean(1), az[:, -1] - az[:, -4], mag[:, -5:, 0]], -1)
        head_i = torch.cat([g[:, :, :10].mean(-1), g.mean(-1), a[:, 0], a[:, :3].mean(1), a[:, 3] - a[:, 0],
                            az[:, 0], az[:, :3].mean(1), az[:, 3] - az[:, 0], mag[:, :5, 0]], -1)
        t = F.normalize(self.tail(torch.cat([pt, tail_i, le], -1)), dim=-1)
        hd = F.normalize(self.head(torch.cat([ph, head_i, le], -1)), dim=-1)
        return t, hd

    def tau(self):
        return self.log_tau.exp().clamp(0.01, 0.5)


def make_model(cfg):
    if cfg.get("arch", "full") == "blocks":
        return LinkNetBlocks(cfg.get("dv", 192), cfg.get("de", 128), cfg.get("dropout", 0.1), cfg.get("ci", 32))
    return LinkNet(cfg.get("dv", 256), cfg.get("de", 128), cfg.get("dropout", 0.1))


class SubjectData:
    """normalised tensors of one subject (or one training recording) for the bi-encoder"""

    def __init__(self, v, a, limbs, vstats, istats, dev):
        vm, vs = vstats
        u = v.astype(np.float32)
        u /= np.linalg.norm(u, axis=2, keepdims=True) + 1e-6
        self.v = torch.from_numpy(((u - vm) / vs).astype(np.float16)).to(dev)
        self.set_imu(a, limbs, istats, dev)

    def set_imu(self, a, limbs, istats, dev):
        m, sd = istats
        self.a = torch.from_numpy(a.astype(np.float32)).to(dev)
        self.az = torch.from_numpy(((a - m[limbs][:, None]) / sd[limbs][:, None]).astype(np.float32)).to(dev)
        self.limb = torch.from_numpy(limbs.astype(np.int64)).to(dev)


def embed(model, sd, bs=4096):
    model.eval()
    ts, hs = [], []
    with torch.no_grad():
        for i in range(0, len(sd.limb), bs):
            t, h = model.feats(sd.v[i:i + bs].float(), sd.a[i:i + bs], sd.az[i:i + bs], sd.limb[i:i + bs])
            ts.append(t.float())
            hs.append(h.float())
    return torch.cat(ts), torch.cat(hs)


def scores(model, sd):
    t, h = embed(model, sd)
    S = (t @ h.T) / model.tau()
    S.fill_diagonal_(-1e4)
    return S


def eval_subject(S, true_succ):
    """top-1 / recall@20 of row argmax and of the dual softmax, given the (n,n) score matrix (torch)"""
    m = true_succ >= 0
    ts = torch.as_tensor(true_succ, device=S.device)
    top = S.topk(20, dim=1).indices
    r1 = (top[:, 0] == ts)[torch.as_tensor(m, device=S.device)].float().mean().item()
    r20 = (top == ts[:, None]).any(1)[torch.as_tensor(m, device=S.device)].float().mean().item()
    D = torch.log_softmax(S, 1) + torch.log_softmax(S, 0)
    d1 = (D.argmax(1) == ts)[torch.as_tensor(m, device=S.device)].float().mean().item()
    return r1, r20, d1


def train_model(train_items, cfg, dev, seed, log_prefix="", evals=None):
    """train_items: list of dicts {v, a4, ok, nxt, vstats, istats} for training recordings"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(cfg).to(dev)
    epochs = cfg.get("epochs", 40)
    block = cfg.get("block", 0)          # >0: each step uses a random contiguous block of this many tiles
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-4))
    total = epochs * len(train_items)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 1e-3), total_steps=total, pct_start=0.1)
    vnoise = cfg.get("vnoise", 0.1)
    for ep in range(epochs):
        model.train()
        tl, nb = 0.0, 0
        for k in rng.permutation(len(train_items)):
            it = train_items[k]
            n_all = len(it["ok"])
            if block and n_all > block:
                b0 = int(rng.integers(0, n_all - block + 1))
                sl = np.arange(b0, b0 + block)
            else:
                sl = np.arange(n_all)
            n = len(sl)
            ok = it["ok"][sl]
            cp = np.cumsum(ok / ok.sum(1, keepdims=True), 1)
            limbs = np.minimum((rng.random(n)[:, None] > cp).sum(1), 3)
            a = it["a4"][sl, :, limbs]                      # (n,50,3)
            m_, s_ = it["istats"]
            sl_t = torch.as_tensor(sl, device=dev)
            v = it["sd"].v[sl_t].float()
            at = torch.from_numpy(a.astype(np.float32)).to(dev)
            az = torch.from_numpy(((a - m_[limbs][:, None]) / s_[limbs][:, None]).astype(np.float32)).to(dev)
            lt = torch.from_numpy(limbs.astype(np.int64)).to(dev)
            if vnoise:
                v = v + vnoise * torch.randn_like(v)
            t, h = model.feats(v, at, az, lt)
            S = (t @ h.T) / model.tau()
            S = S - 1e4 * torch.eye(n, device=dev)
            nx = it["nxt"][sl] - sl[0]
            nx[(nx < 0) | (nx >= n) | (it["nxt"][sl] < 0)] = -1
            nxt = torch.as_tensor(nx, device=dev)
            src = torch.nonzero(nxt >= 0).squeeze(1)
            loss = F.cross_entropy(S[src], nxt[src]) + F.cross_entropy(S[:, nxt[src]].T, src)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
        if evals is not None and ((ep + 1) % cfg.get("eval_every", 10) == 0 or ep == epochs - 1):
            res = []
            for sd, ts in evals:
                res.append(eval_subject(scores(model, sd), ts))
            r = np.mean(res, 0)
            print(f"{log_prefix} ep {ep + 1}/{epochs} loss {tl / nb:.3f} tau {model.tau().item():.3f} "
                  f"held-out top1 {r[0]:.3f} rec@20 {r[1]:.3f} dual-top1 {r[2]:.3f} [{time.time() - T_START:.0f}s]",
                  flush=True)
        elif ep % 10 == 0:
            print(f"{log_prefix} ep {ep + 1}/{epochs} loss {tl / nb:.3f} [{time.time() - T_START:.0f}s]", flush=True)
    return model


def subject_tensors(v, a, limbs, dev):
    return SubjectData(v, a, limbs, vid_norm_stats(v), imu_norm_stats(a, limbs), dev)


def run(P, recs, d, t, fold_of, cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("nlink device", dev, flush=True)
    seed = cfg.get("seed", 0)
    # training recordings: per-recording tiles, normalised with the subject's own statistics
    items = {}
    for i, r in enumerate(recs):
        v, a4, ok, nxt = rec_tiles(P, r)
        items[i] = dict(v=v, a4=a4, ok=ok, nxt=nxt, sbj=r["sbj"])
    by_sbj = {}
    for i, it in items.items():
        by_sbj.setdefault(it["sbj"], []).append(i)
    for s, ii in by_sbj.items():
        vstats = vid_norm_stats(np.concatenate([items[i]["v"] for i in ii]))
        a_all = np.concatenate([items[i]["a4"] for i in ii])           # (N,50,4,3)
        ok_all = np.concatenate([items[i]["ok"] for i in ii])
        m, sd = np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32)
        for l in range(4):
            x = a_all[ok_all[:, l], :, l].reshape(-1, 3)
            m[l], sd[l] = x.mean(0), x.std(0) + 1e-3
        for i in ii:
            it = items[i]
            it["vstats"], it["istats"] = vstats, (m, sd)
            n = len(it["ok"])
            it["sd"] = SubjectData(it["v"], it["a4"][:, :, 0], np.zeros(n, np.int64), vstats, (m, sd), dev)
            del it["v"]
        del a_all
    # OOF subjects with their fixed test-like limbs (d rows), test subjects
    V, A = d["vid"](), d["acc"]()
    oof_sd = {}
    for s in np.unique(d["sbj"]):
        ii = np.flatnonzero(d["sbj"] == s)
        loc = np.full(len(d["y"]), -1)
        loc[ii] = np.arange(len(ii))
        ts = d["true_succ"][ii]
        oof_sd[int(s)] = (ii, subject_tensors(np.asarray(V[ii]), A[ii], d["sensor"][ii], dev),
                          np.where(ts >= 0, loc[np.maximum(ts, 0)], -1))
    del V, A
    TV, TA = t["vid"](), t["acc"]()
    test_sd = {}
    for s in np.unique(t["sbj"]):
        ii = np.flatnonzero(t["sbj"] == s)
        test_sd[int(s)] = (ii, subject_tensors(np.asarray(TV[ii]), TA[ii], t["sensor"][ii], dev))
    del TV, TA

    de = cfg.get("de", 128)
    oof_t = np.zeros((len(d["y"]), de), np.float16)
    oof_h = np.zeros((len(d["y"]), de), np.float16)
    test_t = np.zeros((FOLDS + 1, len(t["sbj"]), de), np.float16)
    test_h = np.zeros((FOLDS + 1, len(t["sbj"]), de), np.float16)
    taus = []
    report = {}
    for f in list(range(FOLDS)) + ["all"]:
        held = [s for s in fold_of if fold_of[s] == f] if f != "all" else []
        tr = [items[i] for i in items if items[i]["sbj"] not in held]
        ev = [(oof_sd[s][1], oof_sd[s][2]) for s in held]
        with stage(f"nlink fold {f}"):
            model = train_model(tr, cfg, dev, seed + (99 if f == "all" else f), f"[fold {f}]", ev if ev else None)
        taus.append(model.tau().item())
        fi = FOLDS if f == "all" else f
        if f != "all":
            res = []
            for s in held:
                ii, sd, ts = oof_sd[s]
                tt, hh = embed(model, sd)
                oof_t[ii], oof_h[ii] = tt.cpu().numpy(), hh.cpu().numpy()
                res.append(eval_subject(scores(model, sd), ts))
            report[f"fold{f}"] = np.round(np.mean(res, 0), 4).tolist()
        for s, (ii, sd) in test_sd.items():
            tt, hh = embed(model, sd)
            test_t[fi, ii], test_h[fi, ii] = tt.cpu().numpy(), hh.cpu().numpy()
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    report["taus"] = taus
    print("nlink report", report, flush=True)
    return dict(oof_t=oof_t, oof_h=oof_h, test_t=test_t, test_h=test_h, taus=np.array(taus)), report


def main(cfg):
    from .common import Paths, save_json
    from . import data
    P = Paths()
    recs, sim, d, t = data.build(P, smoke=bool(cfg.get("smoke")))
    fold_of = data.fold_split(d["sbj"])
    out, rep = run(P, recs, d, t, fold_of, cfg)
    np.savez_compressed(P.ART / f"nlink_{cfg.get('tag', 'v1')}.npz", **out)
    save_json(P.ART / f"nlink_{cfg.get('tag', 'v1')}_report.json", rep)
