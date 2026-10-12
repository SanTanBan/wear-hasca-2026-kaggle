"""High-precision episode clusters (ours). The diagnostics showed that connecting same-label tiles globally is what
the decoder needs (random same-label chains -> 0.994). Build per subject a strict graph (confident links + mutual
kNN pairs with high combined similarity + same predicted class), take connected components, report purity / size,
and smooth the graph probabilities towards their cluster mean."""
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import common as C
from .common import N_CLS, macro_f1, norm_rows


def soft_finish(P, sbj, sets, T=0.5, iters=3):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return C.calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, 97, 0.05, iters=iters).argmax(1)


def strict_edges(ii, succ, score, emb, Pg, thr_l, thr_k, k=5, same_arg=True, wp=1.0):
    n = len(ii)
    g = {int(x): r for r, x in enumerate(ii)}
    rows, cols = [], []
    am = Pg[ii].argmax(1)
    for r, x in enumerate(ii):
        s = succ[x]
        if s >= 0 and score[x] >= thr_l and int(s) in g:
            c = g[int(s)]
            if not same_arg or am[r] == am[c]:
                rows.append(r)
                cols.append(c)
    E = emb[ii] / (np.linalg.norm(emb[ii], axis=1, keepdims=True) + 1e-6)
    Q = np.sqrt(Pg[ii])
    S = (E @ E.T + wp * (Q @ Q.T)) / (1 + wp)
    np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, k, axis=1)[:, :k]
    isnb = np.zeros((n, n), bool)
    isnb[np.repeat(np.arange(n), k), nb.ravel()] = True
    mutual = isnb & isnb.T
    rr, cc = np.nonzero(np.triu(mutual & (S >= thr_k), 1))
    if same_arg:
        keep = am[rr] == am[cc]
        rr, cc = rr[keep], cc[keep]
    rows += rr.tolist()
    cols += cc.tolist()
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def cluster_stats(lab_c, y):
    pur, sizes = [], []
    for c in np.unique(lab_c):
        m = lab_c == c
        cnt = np.bincount(y[m], minlength=N_CLS)
        pur.append(cnt.max())
        sizes.append(m.sum())
    sizes = np.array(sizes)
    return np.sum(pur) / len(y), sizes


def main(cfg):
    from .common import Paths
    from .decode import load_art
    from .base import write_sub
    P = Paths()
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    run = cfg.get("graph_run", "gsweep")
    zo = sorted(Path("/kaggle/input").rglob(f"oof_{run}.npz"))
    zt = sorted(Path("/kaggle/input").rglob(f"test_{run}.npz"))
    Pg, Pt = np.load(zo[0])["P_graph"].astype(np.float64), np.load(zt[0])["P_graph"].astype(np.float64)
    y, sbj = d["y"], d["sbj"]
    emb_o, emb_t = d["emb"].astype(np.float32), t["emb"].astype(np.float32)
    f0 = macro_f1(y, soft_finish(Pg, sbj, C.TRAIN_SETS))
    print(f"input graph probs ({run}) soft-cal OOF {f0:.4f}", flush=True)
    res = {"input": f0}
    best = (f0, None)
    for thr_l in (0.5, 1.5):
        for thr_k in (0.80, 0.85, 0.90):
            for same_arg in (True,):
                lab_c = np.zeros(len(y), np.int64)
                off = 0
                for s in np.unique(sbj):
                    ii = np.flatnonzero(sbj == s)
                    A = strict_edges(ii, d["l2_succ"], d["l2_score_qn"], emb_o, Pg, thr_l, thr_k, same_arg=same_arg)
                    nc, lc = connected_components(A, directed=False)
                    lab_c[ii] = lc + off
                    off += nc
                pur, sizes = cluster_stats(lab_c, y)
                big = sizes[sizes >= 20].sum() / len(y)
                # cluster-mean smoothing
                M = np.zeros((lab_c.max() + 1, N_CLS))
                np.add.at(M, lab_c, Pg)
                cnt = np.bincount(lab_c)
                Mean = M[lab_c] / cnt[lab_c][:, None]
                line = f"thr_l={thr_l} thr_k={thr_k}: {len(sizes)} clusters, purity {pur:.4f}, tiles in clusters>=20 {big:.3f}"
                for lam in (0.3, 0.5, 0.8):
                    Pn = norm_rows((1 - lam) * Pg + lam * Mean)
                    f = macro_f1(y, soft_finish(Pn, sbj, C.TRAIN_SETS))
                    res[f"l{thr_l}_k{thr_k}_lam{lam}"] = f
                    line += f" | lam {lam}: {f:.4f}"
                    if f > best[0]:
                        best = (f, (thr_l, thr_k, lam))
                print(line, flush=True)
    print("best", best, flush=True)
    res["best"] = best
    if best[1] is not None:
        thr_l, thr_k, lam = best[1]
        lab_c = np.zeros(len(t["sbj"]), np.int64)
        off = 0
        for s in np.unique(t["sbj"]):
            ii = np.flatnonzero(t["sbj"] == s)
            A = strict_edges(ii, t["l2_succ"], t["l2_score_qn"], emb_t, Pt, thr_l, thr_k)
            nc, lc = connected_components(A, directed=False)
            lab_c[ii] = lc + off
            off += nc
        M = np.zeros((lab_c.max() + 1, N_CLS))
        np.add.at(M, lab_c, Pt)
        Mean = M[lab_c] / np.bincount(lab_c)[lab_c][:, None]
        Ptn = norm_rows((1 - lam) * Pt + lam * Mean)
        # OOF version for blending
        lab_c = np.zeros(len(y), np.int64)
        off = 0
        for s in np.unique(sbj):
            ii = np.flatnonzero(sbj == s)
            A = strict_edges(ii, d["l2_succ"], d["l2_score_qn"], emb_o, Pg, thr_l, thr_k)
            nc, lc = connected_components(A, directed=False)
            lab_c[ii] = lc + off
            off += nc
        M = np.zeros((lab_c.max() + 1, N_CLS))
        np.add.at(M, lab_c, Pg)
        Pon = norm_rows((1 - lam) * Pg + lam * M[lab_c] / np.bincount(lab_c)[lab_c][:, None])
        np.savez_compressed(P.ART / "oof_clusters.npz", P_graph=Pon, y=y, sbj=sbj)
        np.savez_compressed(P.ART / "test_clusters.npz", P_graph=Ptn, id=t["id"], sbj=t["sbj"], oof_f1=best[0])
        write_sub(P, t, soft_finish(Ptn, t["sbj"], {}), "submission_clusters.csv")
    C.save_json(P.ART / "clusters.json", res)
