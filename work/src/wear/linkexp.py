"""What about the links matters for the graph decoder? Synthetic link sets on the OOF tiles (uses true labels, so
these are diagnostics only): oracle, L2 with cross-label links cut, L2 with cross-label links replaced by the true
successor, random same-label permutation links, and L2 with links cut below a score threshold."""
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import macro_f1, calibrate


def soft_finish(P, dd, iters=3):
    Q = np.clip(P, 1e-12, None) ** 2.0
    return calibrate(Q / Q.sum(1, keepdims=True), dd["sbj"], dd["sets"], 97, 0.05, iters=iters).argmax(1)


def main(cfg):
    from .common import Paths
    from .decode import load_art
    from . import graph
    P = Paths()
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d = load_art(src / "oof_B.npz")
    y, sbj = d["y"], d["sbj"]
    n = len(y)
    dd = dict(logp=d["logp_b"], emb=d["emb"].astype(np.float32), grp=sbj, sbj=sbj, sets=C.TRAIN_SETS)
    ts = d["true_succ"]
    l2, sc = d["l2_succ"], d["l2_score_qn"]
    rng = np.random.default_rng(0)
    variants = {}
    variants["L2"] = (l2, sc)
    variants["ORACLE"] = (ts, np.where(ts >= 0, 5.0, -50.0).astype(np.float32))
    cut = l2.copy()
    cross = (l2 >= 0) & (y[np.maximum(l2, 0)] != y)
    cut[cross] = -1
    variants["L2 cross-label cut"] = (cut, np.where(cut >= 0, sc, -50.0).astype(np.float32))
    rep = l2.copy()
    rep[cross] = ts[cross]
    variants["L2 cross-label -> true succ"] = (rep, np.where(cross, 5.0, sc).astype(np.float32))
    # random same-label permutation links (1:1 within subject x label)
    rl = np.full(n, -1)
    for s in np.unique(sbj):
        for c in np.unique(y[sbj == s]):
            ii = np.flatnonzero((sbj == s) & (y == c))
            p = rng.permutation(ii)
            rl[p[:-1]] = p[1:]
    variants["random same-label chain"] = (rl, np.where(rl >= 0, 5.0, -50.0).astype(np.float32))
    for thr in (0.0, 1.0, 2.0):
        c2 = np.where(sc >= thr, l2, -1)
        variants[f"L2 cut below qn score {thr}"] = (c2, np.where(c2 >= 0, sc, -50.0).astype(np.float32))
    # fraction of tiles kept by each threshold and their same-label rate
    for thr in (0.0, 1.0, 2.0):
        m = (l2 >= 0) & (sc >= thr)
        print(f"qn score >= {thr}: kept {m.mean():.3f} of tiles, same-label {(y[l2[m]] == y[m]).mean():.3f}", flush=True)
    res = {}
    for name, (succ, score) in variants.items():
        t0 = time.time()
        de = dict(dd, succ=succ, score=score)
        Pg = graph.graph_P(de)
        f_full = macro_f1(y, C.finish(Pg, de))
        f_soft = macro_f1(y, soft_finish(Pg, de))
        res[name] = (f_full, f_soft)
        print(f"{name:35s} full-cal {f_full:.4f} soft-cal {f_soft:.4f} [{time.time() - t0:.0f}s]", flush=True)
    C.save_json(P.ART / "linkexp.json", res)
