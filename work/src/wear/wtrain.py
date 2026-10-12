"""Window classifier training (Briano's train_window.py, Apache 2.0, ported into the package) - 5 subject folds,
OOF log-probs of the test-like tiles + test log-probs (mean of the fold models). Used for extra seeds / variants
(GPU strongly recommended). Optional pseudo-labelled test tiles (transductive) via cfg["pseudo"] (npz with P_graph)."""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FOLDS, N_CLS, T_START, VLEN, WIN, macro_f1, stage
from .window_pre import Net, acc_stats, vid_stats

STRIDE = 5


def train_runs(P, recs, d, t, cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = torch.cuda.get_device_capability() if dev.type == "cuda" else (0, 0)
    use_amp = cap >= (7, 0)
    amp_dtype = torch.float16 if (7, 0) <= cap < (8, 0) else torch.bfloat16
    print(f"wtrain device {dev} cap {cap} amp {use_amp}", flush=True)
    acc_l, lab_l, vid_l, val_l = [], [], [], []
    for r in recs:
        acc_l.append(np.load(P.PROC / f"{r['stem']}_acc.npy"))
        val_l.append(np.load(P.PROC / f"{r['stem']}_valid.npy"))
        lab_l.append(np.load(P.PROC / f"{r['stem']}_lab.npy").astype(np.int64))
        vid_l.append(np.load(P.PROC / f"{r['stem']}_vid.npy", mmap_mode="r"))
    acc_off = np.cumsum([0] + [len(a) for a in acc_l])[:-1]
    vid_off = np.cumsum([0] + [len(v) for v in vid_l])[:-1]
    acc_all = np.concatenate(acc_l)
    vid_all = np.concatenate(vid_l)
    t_acc, t_vid = t["acc"](), np.asarray(t["vid"]())
    t_sensor, t_sbj = t["sensor"], t["sbj"]
    SB = sorted({r["sbj"] for r in recs})
    TS = sorted(set(t_sbj.tolist()))
    G = 22 + len(TS)
    AM, AS = np.zeros((G, 4, 3)), np.ones((G, 4, 3))
    VM, VS = np.zeros((G, 768)), np.ones((G, 768))
    for sb in SB:
        ii = [i for i, r in enumerate(recs) if r["sbj"] == sb]
        AM[sb], AS[sb] = acc_stats(np.concatenate([acc_l[i] for i in ii]).reshape(-1, 4, 3),
                                   np.concatenate([val_l[i] for i in ii]))
        VM[sb], VS[sb] = vid_stats(np.concatenate([vid_l[i][::3] for i in ii]))
    for ti, s in enumerate(TS):
        m = t_sbj == s
        x = np.zeros((m.sum(), WIN, 4, 3), np.float32)
        ok = np.zeros((m.sum(), WIN, 4), bool)
        x[np.arange(m.sum()), :, t_sensor[m]] = t_acc[m]
        ok[np.arange(m.sum()), :, t_sensor[m]] = True
        AM[22 + ti], AS[22 + ti] = acc_stats(x.reshape(-1, 4, 3), ok.reshape(-1, 4))
        VM[22 + ti], VS[22 + ti] = vid_stats(t_vid[m].reshape(-1, 768))
    t_grp = np.array([22 + TS.index(s) for s in t_sbj])
    AM, AS, VM, VS = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (AM, AS, VM, VS))
    ACC = torch.from_numpy(acc_all).to(dev)
    VID = torch.from_numpy(np.ascontiguousarray(vid_all)).to(dev)
    del acc_all, vid_all
    # all 0.1-s windows of the training recordings
    W_rec, W_start, W_y, W_sbj, W_ok = [], [], [], [], []
    for i, r in enumerate(recs):
        T_, Fr = len(acc_l[i]), len(vid_l[i])
        starts = np.arange(0, T_ - WIN + 1, STRIDE)
        starts = starts[starts * 3 // 5 + 8 + VLEN <= Fr]
        cs = np.zeros((T_ + 1, N_CLS), np.int32)
        np.cumsum(np.eye(N_CLS, dtype=np.int32)[lab_l[i]], axis=0, out=cs[1:])
        cnt = cs[starts + WIN] - cs[starts]
        miss = np.zeros((T_ + 1, 4), np.int32)
        np.cumsum(~val_l[i], axis=0, out=miss[1:])
        W_ok.append((miss[starts + WIN] - miss[starts]) == 0)
        W_rec.append(np.full(len(starts), i))
        W_start.append(starts)
        W_y.append(cnt.argmax(1))
        W_sbj.append(np.full(len(starts), r["sbj"]))
    W_rec, W_start, W_y, W_sbj, W_ok = map(np.concatenate, (W_rec, W_start, W_y, W_sbj, W_ok))
    keep = W_ok.any(1)
    W_rec, W_start, W_y, W_sbj, W_ok = (x[keep] for x in (W_rec, W_start, W_y, W_sbj, W_ok))
    W_ok_t = torch.as_tensor(W_ok, device=dev)
    W_grp = W_sbj.copy()
    acc_off_t, vid_off_t = torch.tensor(acc_off, device=dev), torch.tensor(vid_off, device=dev)
    ar50, ar15 = torch.arange(WIN, device=dev), torch.arange(VLEN, device=dev)
    from .data import fold_split
    fold_of = fold_split(W_sbj)
    W_fold = np.array([fold_of[int(s)] for s in W_sbj])
    freq = np.bincount(W_y, minlength=N_CLS).astype(np.float64)
    cls_w = (1.0 / np.sqrt(freq))[W_y]
    test_acc = torch.from_numpy(t_acc).to(dev)
    test_vid = torch.from_numpy(t_vid).to(dev)
    test_sensor = torch.from_numpy(t_sensor).to(dev)
    test_grp = torch.from_numpy(t_grp).to(dev)
    # optional pseudo-labelled test tiles
    pseudo = None
    if cfg.get("pseudo"):
        z = np.load(cfg["pseudo"], allow_pickle=True)
        Pp = z["P_graph"]
        conf = Pp.max(1)
        sel = np.flatnonzero(conf >= cfg.get("pseudo_conf", 0.9))
        pseudo = (torch.as_tensor(sel, device=dev), torch.as_tensor(Pp[sel].argmax(1), device=dev))
        print(f"pseudo-labelled test tiles: {len(sel)} of {len(Pp)}", flush=True)

    def rand_rot(n, max_deg=20.0):
        axis = F.normalize(torch.randn(n, 3, device=dev), dim=1)
        ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(max_deg)
        K = torch.zeros(n, 3, 3, device=dev)
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
        K = K - K.transpose(1, 2)
        s, c = ang.sin()[:, None, None], ang.cos()[:, None, None]
        return torch.eye(3, device=dev) + s * K + (1 - c) * (K @ K)

    def prep(a_raw, v_raw, sensor, g, train, only):
        B = a_raw.shape[0]
        a_raw = a_raw.float()
        if train:
            a_raw = a_raw @ rand_rot(B)
            a_raw = a_raw * (0.9 + 0.2 * torch.rand(B, 1, 1, device=dev))
            a_raw = a_raw + 0.02 * torch.randn_like(a_raw)
        mag = a_raw.norm(dim=-1, keepdim=True) - 1.0
        a = torch.cat([(a_raw - AM[g, sensor][:, None]) / AS[g, sensor][:, None], mag], -1)
        v = (v_raw.float() - VM[g][:, None]) / VS[g][:, None]
        has_a, has_v = torch.ones(B, device=dev), torch.ones(B, device=dev)
        if only == "imu":
            return a, torch.zeros_like(v), has_a, has_v * 0
        if train:
            v = v + 0.1 * torch.randn_like(v)
            v = v * (torch.rand(B, VLEN, 1, device=dev) > 0.1)
            r_ = torch.rand(B, device=dev)
            has_v = (r_ >= 0.2).float()
            has_a = ((r_ < 0.2) | (r_ >= 0.3)).float()
            v = v * has_v[:, None, None]
            a = a * has_a[:, None, None]
        return a, v, has_a, has_v

    def gather(idx, sensor):
        rec = torch.as_tensor(W_rec[idx], device=dev)
        st = torch.as_tensor(W_start[idx], device=dev)
        B = len(idx)
        a = ACC[acc_off_t[rec][:, None] + st[:, None] + ar50].view(B, WIN, 4, 3)
        a = a[torch.arange(B, device=dev), :, sensor]
        v = VID[vid_off_t[rec][:, None] + (st * 3 // 5)[:, None] + 8 + ar15]
        return a, v

    @torch.no_grad()
    def predict(model, a_raw, v_raw, sensor, g, only, bs=2048):
        model.eval()
        outs = []
        for i in range(0, len(sensor), bs):
            a, v, _, _ = prep(a_raw[i:i + bs], v_raw[i:i + bs], sensor[i:i + bs], g[i:i + bs], False, only)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                lo = model(a, v, sensor[i:i + bs])
            outs.append(F.log_softmax(lo[0].float(), 1))
        return torch.cat(outs).cpu().numpy()

    # OOF tiles (exact same rows as d)
    d_acc = torch.from_numpy(d["acc"]()).to(dev)
    d_vid = torch.from_numpy(np.asarray(d["vid"]())).to(dev)
    d_sensor = torch.from_numpy(d["sensor"]).to(dev)
    d_grp = torch.from_numpy(d["sbj"]).to(dev)
    results = {}
    for run in cfg["runs"]:
        name, seed, epochs, only = run["name"], run.get("seed", 0), run.get("epochs", 12), run.get("only", "none")
        steps, bs, lr = run.get("steps", 300), run.get("bs", 512), run.get("lr", 1e-3)
        oof = np.zeros((len(d["y"]), N_CLS), np.float32)
        tests = []
        t0 = time.time()
        for fold in range(FOLDS):
            torch.manual_seed(seed * 100 + fold)
            rng = np.random.default_rng(seed * 100 + fold)
            tr_idx = np.flatnonzero(W_fold != fold)
            pw = cls_w[tr_idx] / cls_w[tr_idx].sum()
            cdf = pw.cumsum()
            cdf /= cdf[-1]
            model = Net(run.get("dim", 256), run.get("vdrop", 0.5)).to(dev)
            ema = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999), use_buffers=True)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
            for ep in range(epochs):
                model.train()
                for _ in range(steps):
                    idx = tr_idx[cdf.searchsorted(rng.random(bs), side="right")]
                    sensor = torch.multinomial(W_ok_t[torch.as_tensor(idx, device=dev)].float(), 1).squeeze(1)
                    a_raw, v_raw = gather(idx, sensor)
                    g = torch.as_tensor(W_grp[idx], device=dev)
                    y = torch.as_tensor(W_y[idx], device=dev)
                    if pseudo is not None and run.get("pseudo_frac", 0) > 0:
                        k = int(bs * run["pseudo_frac"])
                        pos = torch.randint(len(pseudo[0]), (k,), device=dev)
                        pi, pl = pseudo[0][pos], pseudo[1][pos]
                        a_raw = torch.cat([a_raw, test_acc[pi]])
                        v_raw = torch.cat([v_raw, test_vid[pi]])
                        sensor = torch.cat([sensor, test_sensor[pi]])
                        g = torch.cat([g, test_grp[pi]])
                        y = torch.cat([y, pl])
                    a, v, has_a, has_v = prep(a_raw, v_raw, sensor, g, True, only)
                    with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                        out, oi, ov = model(a, v, sensor)
                    loss = F.cross_entropy(out.float(), y, label_smoothing=0.1)
                    li = F.cross_entropy(oi.float(), y, label_smoothing=0.1, reduction="none")
                    lv = F.cross_entropy(ov.float(), y, label_smoothing=0.1, reduction="none")
                    loss = loss + 0.3 * (li * has_a).sum() / has_a.sum().clamp(min=1) + \
                        0.3 * (lv * has_v).sum() / has_v.sum().clamp(min=1)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    ema.update_parameters(model)
            va = np.flatnonzero(d["fold"] == fold)
            vt = torch.as_tensor(va, device=dev)
            oof[va] = predict(ema.module, d_acc[vt], d_vid[vt], d_sensor[vt], d_grp[vt], only)
            tests.append(predict(ema.module, test_acc, test_vid, test_sensor, test_grp, only))
            print(f"[{name}] fold {fold} val macro F1 {macro_f1(d['y'][va], oof[va].argmax(1)):.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            del model, ema, opt
        x = np.mean(tests, 0)
        results[name] = (oof, x - np.log(np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)) -
                         x.max(1, keepdims=True))
        print(f"[{name}] OOF macro F1 {macro_f1(d['y'], oof.argmax(1)):.4f}", flush=True)
    return results


def main(cfg):
    from .common import Paths
    from . import data
    P = Paths()
    recs, sim, d, t = data.build(P)
    res = train_runs(P, recs, d, t, cfg)
    out = {}
    for k, (o, te) in res.items():
        out[f"oof_{k}"] = o
        out[f"test_{k}"] = te.astype(np.float32)
    np.savez_compressed(P.ART / f"wtrain_{cfg.get('tag', 'v1')}.npz", **out)
