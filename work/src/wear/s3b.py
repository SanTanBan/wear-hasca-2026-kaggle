"""S3 stacker with the stronger L2 links in its neighbour block (Briano's S3 used L0 links: 33% exact / 88.8%
same-label; L2: 56% / 93.7%), optionally longer neighbourhood. Nested OOF, then blend + graph (gsweep configs)."""
import json
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import FOLDS, N_CLS, T_START, macro_f1, norm_rows


def main(cfg):
    import lightgbm as lgb
    from .common import Paths
    from .base import write_sub
    from .decode import load_art
    from . import data, tab
    from .gsweep import finish_soft, run_cfg
    P = Paths()
    threads = int(cfg.get("threads", 4))
    recs, sim, d, t = data.build(P)
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    a, b = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    for k in ("logp", "fusion_logp", "imu_logp", "emb", "l0_succ", "l0_score", "l2_succ", "l2_score_qn", "logp_b",
              "tab_S3", "tab_T", "P_graph"):
        d[k] = a[k].astype(np.float32) if k == "emb" else a[k]
        t[k] = b[k].astype(np.float32) if k == "emb" else b[k]
    feat, names = tab.features(d, t)
    params = dict(tab.TAB_PARAMS, num_threads=threads)
    res = {}
    outs = {}
    for lk in cfg.get("links", ["l2"]):
        for dd in (d, t):
            if lk == "l2":
                dd["l0_succ_bak"], dd["l0_score_bak"] = dd["l0_succ"], dd["l0_score"]
                dd["l0_succ"], dd["l0_score"] = dd["l2_succ"], dd["l2_score_qn"]
        X = tab.tab_build("S3", d, feat["oof"], names)
        Xt = tab.tab_build("S3", t, feat["test"], names)
        oof = np.zeros((len(X), N_CLS), np.float32)
        for f in range(FOLDS):
            tr, va = d["fold"] != f, d["fold"] == f
            bst = lgb.train(params, lgb.Dataset(X[tr], d["y"][tr], params={"max_bin": 63}), tab.TAB_ROUNDS["S3"])
            oof[va] = np.log(np.clip(bst.predict(X[va]), 1e-7, 1))
        bst = lgb.train(params, lgb.Dataset(X, d["y"], params={"max_bin": 63}), tab.TAB_ROUNDS["S3"])
        tst = np.log(np.clip(bst.predict(Xt), 1e-7, 1)).astype(np.float32)
        f_s3 = macro_f1(d["y"], oof.argmax(1))
        print(f"S3 with {lk} neighbours: nested OOF argmax {f_s3:.4f} (L0 version 0.8142) [{time.time() - T_START:.0f}s]",
              flush=True)
        res[f"S3_{lk}"] = f_s3
        outs[lk] = (oof, tst)
        for dd in (d, t):
            if lk == "l2":
                dd["l0_succ"], dd["l0_score"] = dd["l0_succ_bak"], dd["l0_score_bak"]
    gs = sorted(Path("/kaggle/input").rglob("gsweep.json"))
    top = json.loads(gs[0].read_text())["top"]
    best = (0, None)
    for lk, (oof, tst) in outs.items():
        for w3 in (0.5, 0.6):
            lb_o = tab.tab_blend(d["logp"], [(oof, w3), (d["tab_T"], 0.3)])
            lb_t = tab.tab_blend(t["logp"], [(tst, w3), (t["tab_T"], 0.3)])
            do = dict(logp=lb_o, emb=d["emb"], grp=d["sbj"], sbj=d["sbj"], succ=d["l2_succ"], score=d["l2_score_qn"],
                      sets=C.TRAIN_SETS)
            cache = {}
            Po = norm_rows(np.mean([run_cfg(do, c, cache) for c in top], 0))
            f = macro_f1(d["y"], finish_soft(Po, do))
            res[f"graph_{lk}_w{w3}"] = f
            print(f"blend(window, S3[{lk}]@{w3}, T@0.3) + graph(gsweep top-5): OOF {f:.4f} (gsweep 0.9124)", flush=True)
            if f > best[0]:
                dt = dict(logp=lb_t, emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=t["l2_succ"],
                          score=t["l2_score_qn"], sets={})
                cache = {}
                Pt = norm_rows(np.mean([run_cfg(dt, c, cache) for c in top], 0))
                best = (f, (lk, w3))
                np.savez_compressed(P.ART / "oof_s3b.npz", P_graph=Po, y=d["y"], sbj=d["sbj"])
                np.savez_compressed(P.ART / "test_s3b.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
                np.savez_compressed(P.ART / "logp_s3b.npz", oof=lb_o, test=lb_t)
                write_sub(P, t, finish_soft(Pt, dt), "submission_s3b.csv")
    res["best"] = best
    print("best", best, flush=True)
    C.save_json(P.ART / "s3b.json", res)
