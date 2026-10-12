"""Learned decoder (ours): a small edge-type-aware attention GNN over each subject's graph (L2 links fwd/bwd +
video-kNN edges). Node inputs are every OOF evidence we have (window / IMU / tabular / blended log-probs, the graph
propagation output, IMU activity, limb, video PCA). Trained full-batch per subject on the other folds' subjects
(nested), so OOF predictions are honest; the test uses the mean of the 5 fold models. Then soft calibration."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as C
from .common import FOLDS, N_CLS, T_START, calibrate, macro_f1, norm_rows


def soft_finish(P, sbj, sets, T=0.5, iters=3):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, 97, 0.05, iters=iters)


class Layer(nn.Module):
    def __init__(self, h, n_types=3, ef=2):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(h, h) for _ in range(n_types)])
        self.att = nn.ModuleList([nn.Linear(2 * h + ef, 1) for _ in range(n_types)])
        self.upd = nn.Sequential(nn.Linear(h * (n_types + 1), h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h))
        self.norm = nn.LayerNorm(h)

    def forward(self, x, edges):
        msgs = [x]
        for k, (src, dst, ef) in enumerate(edges):
            a = self.att[k](torch.cat([x[dst], x[src], ef], 1)).squeeze(1)
            a = a - a.max()
            w = torch.exp(a)
            den = torch.zeros(len(x), device=x.device).index_add_(0, dst, w) + 1e-9
            m = torch.zeros_like(x).index_add_(0, dst, w[:, None] * self.W[k](x[src]))
            msgs.append(m / den[:, None])
        return self.norm(x + self.upd(torch.cat(msgs, 1)))


class GNN(nn.Module):
    def __init__(self, fin, h=96, layers=3, n_types=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(fin, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h))
        self.layers = nn.ModuleList([Layer(h, n_types) for _ in range(layers)])
        self.out = nn.Linear(h, N_CLS)

    def forward(self, x, edges):
        h = self.inp(x)
        for L in self.layers:
            h = L(h, edges)
        return self.out(h)


def subject_graph(ii, succ, score, emb, P0, k=10, use_p=0.5):
    """local edge lists for one subject: (src, dst, edge feats) for link-forward, link-backward, kNN"""
    n = len(ii)
    g = {int(x): r for r, x in enumerate(ii)}
    s_loc = np.array([g.get(int(succ[x]), -1) if succ[x] >= 0 else -1 for x in ii])
    m = s_loc >= 0
    src = np.flatnonzero(m)
    dst = s_loc[m]
    sc = score[ii][m]
    pr = 1 / (1 + np.exp(-np.clip(sc, -20, 20)))
    e_fwd = (src, dst, np.stack([sc / 5, pr], 1))
    e_bwd = (dst, src, np.stack([sc / 5, pr], 1))
    E = emb[ii] / (np.linalg.norm(emb[ii], axis=1, keepdims=True) + 1e-6)
    Q = np.sqrt(P0[ii])
    S = E @ E.T + use_p * (Q @ Q.T)
    np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, k, axis=1)[:, :k]
    sv = np.take_along_axis(S, nb, 1)
    ks = np.repeat(np.arange(n), k)
    kd = nb.ravel()
    e_knn = (kd, ks, np.stack([sv.ravel(), (E[ks] * E[kd]).sum(1)], 1))   # message from neighbour to node
    return [e_fwd, e_bwd, e_knn]


def to_t(edges, dev):
    return [(torch.as_tensor(s, device=dev), torch.as_tensor(d, device=dev),
             torch.as_tensor(np.asarray(f, np.float32), device=dev)) for s, d, f in edges]


def build_inputs(dd, act, pca_m, pca_c):
    lp = [dd[k] for k in ("logp_b", "logp", "imu_logp", "fusion_logp", "tab_S3", "tab_T")]
    pg = np.log(np.clip(dd["P_graph"], 1e-7, 1))
    sens = np.eye(4)[dd["sensor"]]
    e = dd["emb"].astype(np.float32)
    ev = (e - pca_m) @ pca_c
    X = np.concatenate(lp + [pg, act, sens, ev / (np.abs(ev).std(0) + 1e-6)], 1).astype(np.float32)
    return np.clip(X, -20, 20)


def main(cfg):
    from .common import Paths
    from .decode import load_art
    from .pgs import activity
    from .base import write_sub
    from . import data
    torch.set_num_threads(int(cfg.get("threads", 4)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = Paths()
    recs, sim, dB, tB = data.build(P)
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    run = cfg.get("graph_run")
    if run:
        zo = sorted(Path("/kaggle/input").rglob(f"oof_{run}.npz"))
        zt = sorted(Path("/kaggle/input").rglob(f"test_{run}.npz"))
        if zo and zt:
            d["P_graph"], t["P_graph"] = np.load(zo[0])["P_graph"], np.load(zt[0])["P_graph"]
            print("P_graph from", run, flush=True)
    y, sbj = d["y"], d["sbj"]
    act_o = activity(dB["acc"](), dB["sbj"], dB["sensor"])
    act_t = activity(tB["acc"](), tB["sbj"], tB["sensor"])
    e = d["emb"].astype(np.float32)
    pm = e.mean(0)
    _, _, Vt = np.linalg.svd(e[::5] - pm, full_matrices=False)
    pc = Vt[:24].T.astype(np.float32)
    Xo = build_inputs(d, act_o, pm, pc)
    Xt = build_inputs(t, act_t, pm, pc)
    print("node features", Xo.shape, flush=True)
    P0o, P0t = np.exp(d["logp_b"]), np.exp(t["logp_b"])
    graphs_o = {int(s): np.flatnonzero(sbj == s) for s in np.unique(sbj)}
    graphs_t = {int(s): np.flatnonzero(t["sbj"] == s) for s in np.unique(t["sbj"])}
    E_o = {s: to_t(subject_graph(ii, d["l2_succ"], d["l2_score_qn"], e, P0o), dev) for s, ii in graphs_o.items()}
    E_t = {s: to_t(subject_graph(ii, t["l2_succ"], t["l2_score_qn"], t["emb"].astype(np.float32), P0t), dev)
           for s, ii in graphs_t.items()}
    Xo_t, Xt_t = torch.as_tensor(Xo, device=dev), torch.as_tensor(Xt, device=dev)
    y_t = torch.as_tensor(y, device=dev)
    freq = np.bincount(y, minlength=N_CLS).astype(np.float64)
    cw = torch.as_tensor((freq.mean() / freq) ** 0.5, dtype=torch.float32, device=dev)
    fold_of_s = {int(s): int(d["fold"][sbj == s][0]) for s in np.unique(sbj)}
    epochs, lr = int(cfg.get("epochs", 150)), float(cfg.get("lr", 2e-3))
    oof = np.zeros((len(y), N_CLS), np.float32)
    test_acc = np.zeros((len(t["sbj"]), N_CLS), np.float32)
    for f in range(FOLDS):
        torch.manual_seed(f)
        model = GNN(Xo.shape[1], cfg.get("hidden", 96), cfg.get("layers", 3)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        tr_s = [s for s in graphs_o if fold_of_s[s] != f]
        va_s = [s for s in graphs_o if fold_of_s[s] == f]
        rng = np.random.default_rng(f)
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for s in rng.permutation(tr_s):
                ii = torch.as_tensor(graphs_o[s], device=dev)
                x = Xo_t[ii]
                if cfg.get("feat_drop", 0.1):
                    x = F.dropout(x, cfg.get("feat_drop", 0.1))
                logits = model(x, E_o[s])
                loss = F.cross_entropy(logits, y_t[ii], weight=cw, label_smoothing=0.05)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                tot += loss.item()
            sched.step()
            if (ep + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    pv = {s: torch.softmax(model(Xo_t[torch.as_tensor(graphs_o[s], device=dev)], E_o[s]), 1).cpu().numpy()
                          for s in va_s}
                yy = np.concatenate([y[graphs_o[s]] for s in va_s])
                pp = np.concatenate([pv[s] for s in va_s])
                print(f"[fold {f}] ep {ep + 1} loss {tot / len(tr_s):.3f} held-out argmax F1 {macro_f1(yy, pp.argmax(1)):.4f} "
                      f"[{time.time() - T_START:.0f}s]", flush=True)
        model.eval()
        with torch.no_grad():
            for s in va_s:
                ii = graphs_o[s]
                oof[ii] = torch.softmax(model(Xo_t[torch.as_tensor(ii, device=dev)], E_o[s]), 1).cpu().numpy()
            for s, ii in graphs_t.items():
                test_acc[ii] += torch.softmax(model(Xt_t[torch.as_tensor(ii, device=dev)], E_t[s]), 1).cpu().numpy() / FOLDS
    res = {"gnn_argmax": macro_f1(y, oof.argmax(1))}
    print(f"GNN OOF argmax F1 {res['gnn_argmax']:.4f}", flush=True)
    Pg = d["P_graph"].astype(np.float64)
    res["graph_soft"] = macro_f1(y, soft_finish(Pg, sbj, C.TRAIN_SETS).argmax(1))
    best = (res["graph_soft"], None)
    for T in (0.5, 1.0):
        for it in (1, 3):
            k = f"gnn_T{T}_it{it}"
            res[k] = macro_f1(y, soft_finish(oof, sbj, C.TRAIN_SETS, T, it).argmax(1))
            print(f"{k}: {res[k]:.4f}", flush=True)
            if res[k] > best[0]:
                best = (res[k], ("gnn", T, it))
    for wg in (0.3, 0.5, 0.7):
        Bo = norm_rows(np.exp(wg * np.log(oof + 1e-9) + (1 - wg) * np.log(Pg + 1e-9)))
        k = f"blend_w{wg}"
        res[k] = macro_f1(y, soft_finish(Bo, sbj, C.TRAIN_SETS).argmax(1))
        print(f"gnn x graph geo w={wg}: {res[k]:.4f}", flush=True)
        if res[k] > best[0]:
            best = (res[k], ("blend", wg))
    print("best", best, "vs graph", res["graph_soft"], flush=True)
    np.savez_compressed(P.ART / "oof_gnn.npz", P_graph=norm_rows(oof.astype(np.float64)), y=y, sbj=sbj)
    np.savez_compressed(P.ART / "test_gnn.npz", P_graph=norm_rows(test_acc.astype(np.float64)), id=t["id"], sbj=t["sbj"])
    if best[1] is not None:
        if best[1][0] == "gnn":
            lab = soft_finish(test_acc, t["sbj"], {}, best[1][1], best[1][2]).argmax(1)
        else:
            wg = best[1][1]
            Bt = norm_rows(np.exp(wg * np.log(test_acc + 1e-9) + (1 - wg) * np.log(t["P_graph"] + 1e-9)))
            lab = soft_finish(Bt, t["sbj"], {}).argmax(1)
        write_sub(P, t, lab, "submission_gnn.csv")
    res["best"] = best
    C.save_json(P.ART / "gnn.json", res)
