"""Label purity of within-subject kNN graphs built from different embeddings (OOF tiles; diagnostics)."""
from pathlib import Path

import numpy as np

from . import common as C


def unit(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def purity(E, y, sbj, k=10, extra=None, w=0.0):
    out = []
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        S = E[ii] @ E[ii].T
        if extra is not None and w:
            S = S + w * (extra[ii] @ extra[ii].T)
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k, axis=1)[:, :k]
        out.append((y[ii][nb] == y[ii][:, None]).mean(1))
    return float(np.concatenate(out).mean())


def main(cfg):
    from .common import Paths
    from .decode import load_art
    P = Paths()
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d = load_art(src / "oof_B.npz")
    y, sbj = d["y"], d["sbj"]
    emb = unit(d["emb"])
    pb = np.sqrt(np.exp(d["logp_b"]))
    pg = np.sqrt(d["P_graph"])
    res = {}
    cands = {"video emb": (emb, None, 0), "video + 0.5 sqrt(P_b) (graph default)": (emb, pb, 0.5),
             "video + 1.0 sqrt(P_b)": (emb, pb, 1.0), "sqrt(P_b) only": (pb, None, 0),
             "sqrt(P_graph) only": (pg, None, 0), "video + 0.5 sqrt(P_graph)": (emb, pg, 0.5),
             "video + 1.0 sqrt(P_graph)": (emb, pg, 1.0)}
    nl = sorted(Path("/kaggle/input").rglob("nlink_v*.npz"))
    if nl:
        z = np.load(nl[0])
        en = unit(np.concatenate([unit(z["oof_t"]), unit(z["oof_h"])], 1))
        cands["nlink tail+head"] = (en, None, 0)
        cands["video + nlink"] = (unit(np.concatenate([emb, en], 1)), None, 0)
        cands["video + nlink + 0.5 sqrt(P_b)"] = (unit(np.concatenate([emb, en], 1)), pb, 0.5)
    for k in (5, 10):
        for name, (E, ex, w) in cands.items():
            p = purity(E, y, sbj, k, ex, w)
            res[f"{name} @k={k}"] = p
            print(f"purity@{k:2d} {p:.4f}  {name}", flush=True)
    C.save_json(P.ART / "knnpur.json", res)
