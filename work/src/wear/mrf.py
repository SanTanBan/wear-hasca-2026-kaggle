"""Mean-field Potts MRF decoder (ours): unary = tile log-probs; pairwise = link edges (weighted by link probability)
+ video-kNN edges; per-subject class biases are updated during inference so the class counts drift towards the
count prior (soft, like a few Sinkhorn steps). Compared on the OOF tiles against the graph propagation."""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import common as C
from .common import N_CLS, macro_f1, norm_rows, calibrate


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def link_matrix(succ, score, n, b=-2.0):
    m = succ >= 0
    src, dst = np.flatnonzero(m), succ[m]
    w = 1 / (1 + np.exp(-np.clip(score[m] - b, -20, 20)))
    W = sp.csr_matrix((np.r_[w, w], (np.r_[src, dst], np.r_[dst, src])), shape=(n, n))
    return W


def knn_matrix(emb, sbj, k=10, tau=0.1):
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    rows, cols, vals = [], [], []
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        S = E[ii] @ E[ii].T
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k, axis=1)[:, :k]
        sv = np.take_along_axis(S, nb, 1)
        w = np.exp((sv - sv.max(1, keepdims=True)) / tau)
        rows.append(np.repeat(ii, k))
        cols.append(ii[nb].ravel())
        vals.append(w.ravel())
    n = len(emb)
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    W = W.maximum(W.T)
    d = np.asarray(W.sum(1)).ravel()
    return sp.diags(1 / np.maximum(d, 1e-9)) @ W      # row-normalised


def targets(sbj, sets, per_ex=97, null_min=0.05):
    T = {}
    for s in np.unique(sbj):
        n = (sbj == s).sum()
        null = max(n - 18 * per_ex * sets.get(int(s), 1), null_min * n)
        t_ = np.full(N_CLS, (n - null) / 18)
        t_[0] = null
        T[int(s)] = t_
    return T


def meanfield(U, Wl, Wk, sbj, sets, lam_l=1.0, lam_k=1.0, iters=10, damp=0.5, bias_lr=0.0, temp=1.0):
    q = softmax(U / temp)
    bias = np.zeros((len(np.unique(sbj)), N_CLS))
    subs = {int(s): i for i, s in enumerate(np.unique(sbj))}
    sidx = np.array([subs[int(s)] for s in sbj])
    tg = targets(sbj, sets)
    for it in range(iters):
        msg = lam_l * (Wl @ q) + lam_k * (Wk @ q)
        qn = softmax((U + msg + bias[sidx]) / temp)
        q = damp * q + (1 - damp) * qn
        if bias_lr:
            for s, i in subs.items():
                m = sbj == s
                cnt = q[m].sum(0)
                bias[i] += bias_lr * (np.log(tg[s] + 1) - np.log(cnt + 1))
    return q


def soft_cal(P, sbj, sets, T=0.5, iters=3):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, C.CFG["per_ex"], C.CFG["null_min"], iters=iters)


def main(cfg):
    from .common import Paths
    from .decode import load_art
    from .base import write_sub
    P = Paths()
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    y, sbj = d["y"], d["sbj"]
    n = len(y)
    emb = d["emb"].astype(np.float32)
    Wk = knn_matrix(emb, sbj)
    res = {}
    for lk_name, succ, score in (("L2", d["l2_succ"], d["l2_score_qn"]),
                                 ("ORACLE", d["true_succ"], np.where(d["true_succ"] >= 0, 5.0, -50.0))):
        Wl = link_matrix(succ, score, n)
        for unary_name, U in (("logp_b", d["logp_b"]), ("graph", np.log(np.clip(d["P_graph"], 1e-9, 1)))):
            for lam_l, lam_k, iters, bias_lr in ((1, 1, 10, 0), (2, 1, 10, 0), (2, 2, 10, 0), (3, 1, 20, 0),
                                                 (2, 1, 20, 0.3), (1, 0.5, 10, 0), (4, 2, 20, 0)):
                t0 = time.time()
                q = meanfield(U, Wl, Wk, sbj, C.TRAIN_SETS, lam_l, lam_k, iters, 0.5, bias_lr)
                f_raw = macro_f1(y, q.argmax(1))
                f_cal = macro_f1(y, soft_cal(q, sbj, C.TRAIN_SETS).argmax(1))
                name = f"{lk_name} unary={unary_name} lam_l={lam_l} lam_k={lam_k} it={iters} bias={bias_lr}"
                res[name] = (f_raw, f_cal)
                print(f"{name:70s} raw {f_raw:.4f} softcal {f_cal:.4f} [{time.time() - t0:.0f}s]", flush=True)
    C.save_json(P.ART / "mrf.json", res)
