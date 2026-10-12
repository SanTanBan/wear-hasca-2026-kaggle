"""Subject-specific self-training (ours): for every subject, train a LightGBM on that subject's own tiles with the
pipeline's calibrated pseudo-labels (5 inner folds, so no tile sees its own pseudo-label), predict the subject's tiles
and blend with the graph probabilities. Captures each subject's own way of moving / filming."""
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import N_CLS, T_START, calibrate, macro_f1, norm_rows

ST_PARAMS = dict(objective="multiclass", num_class=N_CLS, learning_rate=0.1, num_leaves=15, min_data_in_leaf=20,
                 feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63,
                 num_threads=4, verbose=-1, seed=0)


def soft_cal(P, sbj, sets, T=0.5, iters=3):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, C.CFG["per_ex"], C.CFG["null_min"], iters=iters)


def subject_models(X, Q, sbj, rounds=80, inner=5, conf_pow=1.0, seed=0, threads=4):
    """X features, Q calibrated pseudo-label distribution -> out-of-inner-fold probabilities per subject"""
    import lightgbm as lgb
    params = dict(ST_PARAMS, num_threads=threads, seed=seed)
    out = np.zeros_like(Q, dtype=np.float32)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        yl = Q[ii].argmax(1)
        w = Q[ii].max(1) ** conf_pow
        rng = np.random.default_rng(seed + int(s))
        fold = rng.integers(0, inner, len(ii))
        for f in range(inner):
            tr, va = fold != f, fold == f
            present = np.unique(yl[tr])
            b = lgb.train(params, lgb.Dataset(X[ii[tr]], yl[tr], weight=w[tr],
                                              params={"max_bin": params["max_bin"]}), rounds)
            p = b.predict(X[ii[va]])
            out[ii[va]] = p
        print(f"self-train subject {s}: n={len(ii)} [{time.time() - T_START:.0f}s]", flush=True)
    return norm_rows(out)


def main(cfg):
    from .common import Paths
    from . import data, tab
    from .base import write_sub
    from .decode import load_art
    P = Paths()
    recs, sim, d, t = data.build(P)
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    a, b = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    for k in ("logp", "fusion_logp", "imu_logp", "emb", "l0_succ", "l0_score", "logp_b", "P_graph"):
        d[k] = a[k].astype(np.float32) if k == "emb" else a[k]
        t[k] = b[k].astype(np.float32) if k == "emb" else b[k]
    feat, names = tab.features(d, t)
    Xo = np.concatenate([feat["oof"]["imu"], feat["oof"]["vmot"], feat["oof"]["vpca"]], 1)
    Xt = np.concatenate([feat["test"]["imu"], feat["test"]["vmot"], feat["test"]["vpca"]], 1)
    # optional extra probability sources (e.g. better graph outputs from other kernels)
    Pg_o, Pg_t = d["P_graph"].astype(np.float64), t["P_graph"].astype(np.float64)
    for extra in cfg.get("extra_runs", []):
        z = sorted(Path("/kaggle/input").rglob(f"oof_{extra}.npz"))
        zt = sorted(Path("/kaggle/input").rglob(f"test_{extra}.npz"))
        if z and zt:
            Pg_o = Pg_o + np.load(z[0])["P_graph"]
            Pg_t = Pg_t + np.load(zt[0])["P_graph"]
            print("added run", extra, flush=True)
    Pg_o, Pg_t = norm_rows(Pg_o), norm_rows(Pg_t)
    y, sbj = d["y"], d["sbj"]
    f0 = macro_f1(y, soft_cal(Pg_o, sbj, C.TRAIN_SETS).argmax(1))
    print(f"graph only (soft cal): OOF macro F1 {f0:.4f}", flush=True)
    res = {"graph_soft": f0}
    Qo = soft_cal(Pg_o, sbj, C.TRAIN_SETS)
    Qt = soft_cal(Pg_t, t["sbj"], {})
    best = (f0, None)
    for rounds in cfg.get("rounds", [60]):
        So = subject_models(Xo, Qo, sbj, rounds=rounds)
        St = subject_models(Xt, Qt, t["sbj"], rounds=rounds)
        fs = macro_f1(y, soft_cal(So, sbj, C.TRAIN_SETS).argmax(1))
        print(f"self-train alone (rounds {rounds}): OOF macro F1 {fs:.4f}", flush=True)
        for wgt in (0.2, 0.35, 0.5):
            Bo = norm_rows(np.exp((1 - wgt) * np.log(Pg_o + 1e-9) + wgt * np.log(So + 1e-9)))
            f = macro_f1(y, soft_cal(Bo, sbj, C.TRAIN_SETS).argmax(1))
            res[f"r{rounds}_w{wgt}"] = f
            print(f"graph x self-train (geo w={wgt}, rounds {rounds}): OOF macro F1 {f:.4f}", flush=True)
            if f > best[0]:
                Bt = norm_rows(np.exp((1 - wgt) * np.log(Pg_t + 1e-9) + wgt * np.log(St + 1e-9)))
                best = (f, (rounds, wgt))
                np.savez_compressed(P.ART / "oof_selftrain.npz", P_graph=Bo, y=y, sbj=sbj)
                np.savez_compressed(P.ART / "test_selftrain.npz", P_graph=Bt, id=t["id"], sbj=t["sbj"], oof_f1=f)
                write_sub(P, t, soft_cal(Bt, t["sbj"], {}).argmax(1), "submission_selftrain.csv")
    res["best"] = best
    C.save_json(P.ART / "selftrain.json", res)
    print("best", best, flush=True)
