"""Graph iteration: rebuild the prediction-similarity part of the kNN graph from refined predictions (purity 0.908 ->
0.92-0.94) and re-run the propagation; also kNN updates inside the self-training rounds."""
import json
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import macro_f1, norm_rows
from .gsweep import finish_soft, run_cfg


def main(cfg):
    from .common import Paths
    from .base import write_sub
    from .decode import load_art
    P = Paths()
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    d["emb"], t["emb"] = d["emb"].astype(np.float32), t["emb"].astype(np.float32)
    y = d["y"]
    gs = sorted(Path("/kaggle/input").rglob("gsweep.json"))
    top = json.loads(gs[0].read_text())["top"] if gs else [dict(emb_g=0.5, link_mode="abs", beta=0.8, link_b=-2.0,
                                                                  alpha=0.95, rounds=5, T=2.0, w=0.5, k=5, tau=0.2)]
    best_cfg = top[0]
    print("config", best_cfg, flush=True)
    mk = lambda dd_src, which: dict(logp=dd_src["logp_b"], emb=dd_src["emb"], grp=dd_src["sbj"], sbj=dd_src["sbj"],
                                    succ=dd_src["l2_succ"], score=dd_src["l2_score_qn"],
                                    sets=C.TRAIN_SETS if which == "oof" else {})
    do, dt = mk(d, "oof"), mk(t, "test")
    res = {}

    def ens(dd, cfgs, extra=None):
        cache = {}
        Ps = []
        for c in cfgs:
            dd2 = dict(dd, **(extra or {}))
            Ps.append(run_cfg(dd2, c, cache))
        return norm_rows(np.mean(Ps, 0))

    t0 = time.time()
    P1 = ens(do, top)
    f1 = macro_f1(y, finish_soft(P1, do))
    print(f"iteration 0 (gsweep top-5): {f1:.4f} [{time.time() - t0:.0f}s]", flush=True)
    res["it0"] = f1
    best = (f1, P1, None)
    Pcur_o = P1
    Pcur_t = ens(dt, top)
    for it in range(1, 4):
        for up in (0.5, 1.0):
            cfgs = [dict(c, use_p=up) for c in top]
            Po = ens(do, cfgs, {"knn_P": Pcur_o})
            f = macro_f1(y, finish_soft(Po, do))
            res[f"it{it}_usep{up}"] = f
            print(f"iteration {it} knn from refined P, use_p={up}: {f:.4f} [{time.time() - t0:.0f}s]", flush=True)
            if f > best[0]:
                best = (f, Po, (it, up))
        if best[2] is None or best[2][0] != it:
            break
        up = best[2][1]
        Pt = ens(dt, [dict(c, use_p=up) for c in top], {"knn_P": Pcur_t})
        Pcur_o, Pcur_t = best[1], Pt
    # kNN update inside self-training rounds
    Po = ens(do, [dict(c, knn_update=True) for c in top])
    f = macro_f1(y, finish_soft(Po, do))
    res["knn_update_rounds"] = f
    print(f"kNN update inside rounds: {f:.4f}", flush=True)
    print("best", best[0], best[2], flush=True)
    np.savez_compressed(P.ART / "oof_giter.npz", P_graph=Pcur_o, y=y, sbj=d["sbj"])
    np.savez_compressed(P.ART / "test_giter.npz", P_graph=Pcur_t, id=t["id"], sbj=t["sbj"], oof_f1=best[0])
    write_sub(P, t, finish_soft(Pcur_t, dt), "submission_giter.csv")
    C.save_json(P.ART / "giter.json", res)
