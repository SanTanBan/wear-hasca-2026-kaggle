"""Graph-propagation config sweep on the OOF tiles with the soft final calibration (Sinkhorn 3 iterations).
Inputs: base artifacts (oof_B / test_B). Prints single-config OOF F1, greedy config ensembles, writes the best test CSV."""
import itertools
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import macro_f1, calibrate, norm_rows


def finish_soft(P, dd, T=0.5, iters=3):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), dd["sbj"], dd["sets"], C.CFG["per_ex"], C.CFG["null_min"],
                     iters=iters).argmax(1)


def run_cfg(dd, c, cache):
    from .graph import Runner
    key = (c["link_b"], c.get("cal_iters", 50))
    if key not in cache:
        cache.clear()
        cache[key] = Runner(dd, b=c["link_b"], cal_iters=c.get("cal_iters", 50))
    cc = {k: v for k, v in c.items() if k not in ("link_b", "cal_iters")}
    return cache[key](cc).astype(np.float32)


def main(cfg):
    from .common import Paths
    from .base import write_sub
    from .decode import load_art
    P = Paths()
    hits = sorted(Path("/kaggle/input").rglob("oof_B.npz"))
    src = hits[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    d["emb"], t["emb"] = d["emb"].astype(np.float32), t["emb"].astype(np.float32)
    y = d["y"]
    dd = dict(logp=d["logp_b"], emb=d["emb"], grp=d["sbj"], sbj=d["sbj"], succ=d["l2_succ"], score=d["l2_score_qn"],
              sets=C.TRAIN_SETS)
    base = dict(emb_g=0.5, link_mode="abs", beta=0.8, link_b=-2.0, alpha=0.95, rounds=3, T=2.0, w=0.5)
    grid = [dict(base)]
    for k, vals in (("alpha", [0.9, 0.93, 0.97, 0.98]), ("beta", [0.4, 0.6, 1.0]), ("link_b", [-3.0, -1.0, 0.0]),
                    ("emb_g", [0.0, 1.0]), ("rounds", [1, 2, 4, 5]), ("w", [0.3, 0.7, 0.9]), ("T", [1.0, 3.0]),
                    ("k", [5, 15, 20]), ("tau", [0.05, 0.2]), ("use_p", [0.0, 1.0]), ("cal_iters", [3, 10]),
                    ("link_mode", ["row"])):
        for v in vals:
            c = dict(base)
            c[k] = v
            grid.append(c)
    cache = {}
    res = []
    t0 = time.time()
    for c in grid:
        Pc = run_cfg(dd, c, cache)
        f50 = macro_f1(y, C.finish(Pc, dd))
        f3 = macro_f1(y, finish_soft(Pc, dd))
        res.append((f3, f50, c))
        print(f"soft {f3:.4f} full {f50:.4f}  {c}  [{time.time() - t0:.0f}s]", flush=True)
    res.sort(key=lambda r: -r[0])
    print("TOP 10:", flush=True)
    for f3, f50, c in res[:10]:
        print(f"  soft {f3:.4f} full {f50:.4f} {c}", flush=True)
    # second round: combine the best single changes
    best = dict(res[0][2])
    improved = True
    rounds = 0
    while improved and rounds < 3:
        improved = False
        rounds += 1
        for f3, f50, c in res[1:8]:
            diff = {k: v for k, v in c.items() if base.get(k) != v}
            cand = dict(best)
            cand.update(diff)
            if cand == best:
                continue
            Pc = run_cfg(dd, cand, cache)
            f = macro_f1(y, finish_soft(Pc, dd))
            print(f"combo soft {f:.4f} {cand}", flush=True)
            if f > res[0][0] + 1e-4:
                res.insert(0, (f, 0, cand))
                best = cand
                improved = True
    # ensemble of the top-5 distinct configs
    top = [r[2] for r in sorted(res, key=lambda r: -r[0])[:5]]
    Pe = norm_rows(np.mean([run_cfg(dd, c, cache) for c in top], 0))
    fe = macro_f1(y, finish_soft(Pe, dd))
    print(f"ENSEMBLE top-5 soft {fe:.4f}", flush=True)
    dt = dict(logp=t["logp_b"], emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=t["l2_succ"], score=t["l2_score_qn"],
              sets={})
    tcache = {}
    Pt = norm_rows(np.mean([run_cfg(dt, c, tcache) for c in top], 0))
    lab = finish_soft(Pt, dt)
    write_sub(P, t, lab, "submission_gsweep_top5.csv")
    np.savez_compressed(P.ART / "oof_gsweep.npz", P_graph=Pe, y=y, sbj=d["sbj"])
    np.savez_compressed(P.ART / "test_gsweep.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=fe)
    C.save_json(P.ART / "gsweep.json", {"top": top, "ensemble_soft": fe,
                                        "results": [(r[0], r[1], r[2]) for r in res]})
