"""Post-graph stacker (ours): a LightGBM that reads the graph-propagated probabilities together with the *local*
evidence (window / tabular log-probs, IMU activity) of the tile and of its chain / video-kNN neighbourhood, plus
subject-level predicted class mass. Learns where the graph over-smooths (exercise labels bleeding into rests).
Nested 5-fold on the OOF tiles; the final model on all OOF tiles predicts the test tiles."""
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import FOLDS, N_CLS, T_START, calibrate, macro_f1, norm_rows
from .chainstack import walk

PGS_PARAMS = dict(objective="multiclass", num_class=N_CLS, learning_rate=0.05, num_leaves=31, min_data_in_leaf=80,
                  feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63,
                  num_threads=4, verbose=-1, seed=0)


def soft_cal(P, sbj, sets, T=0.5, iters=3):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, C.CFG["per_ex"], C.CFG["null_min"], iters=iters)


def window_mean(P, idx_rows, w_rows):
    num = np.zeros_like(P)
    den = np.zeros(len(P))
    for idx, w in zip(idx_rows, w_rows):
        ww = np.where(idx >= 0, w, 0.0)
        num += ww[:, None] * P[np.maximum(idx, 0)]
        den += ww
    return num / np.maximum(den, 1e-9)[:, None], den


def features(dd, Pg, act):
    """dd: dict with logp_b, logp, imu_logp, succ, score, emb, sbj, sets; Pg graph probs; act (n,k) IMU activity"""
    n = len(Pg)
    succ, score = dd["succ"], dd["score"]
    pred = np.full(n, -1, np.int64)
    m = succ >= 0
    pred[succ[m]] = np.flatnonzero(m)
    lp = 1 / (1 + np.exp(-np.clip(score, -20, 20)))
    fw, bw = walk(succ, 8), walk(pred, 8)
    pb = np.exp(dd["logp_b"])
    feats = [np.log(np.clip(Pg, 1e-7, 1)), dd["logp_b"], dd["logp"], dd["imu_logp"], act]
    Q = soft_cal(Pg, dd["sbj"], dd["sets"])
    feats.append(np.log(np.clip(Q, 1e-7, 1)))
    for idx in (fw, bw):
        for a, b in ((0, 1), (1, 3), (3, 8)):
            rows = [idx[k] for k in range(a, b)]
            ws = [np.ones(n)] * len(rows)
            mp, den = window_mean(Pg, rows, ws)
            ml, _ = window_mean(pb, rows, ws)
            ma, _ = window_mean(act, rows, ws)
            feats += [np.log(np.clip(mp, 1e-7, 1)), np.log(np.clip(ml, 1e-7, 1)), ma, den[:, None]]
    feats += [score[:, None], np.where(pred >= 0, score[np.maximum(pred, 0)], -50)[:, None]]
    # video kNN neighbourhood (within subject)
    E = dd["emb"] / (np.linalg.norm(dd["emb"], axis=1, keepdims=True) + 1e-6)
    kn_p, kn_l = np.zeros((n, N_CLS)), np.zeros((n, N_CLS))
    for s in np.unique(dd["sbj"]):
        ii = np.flatnonzero(dd["sbj"] == s)
        S = E[ii] @ E[ii].T
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, 10, axis=1)[:, :10]
        kn_p[ii] = Pg[ii][nb].mean(1)
        kn_l[ii] = pb[ii][nb].mean(1)
    feats += [np.log(kn_p + 1e-7), np.log(kn_l + 1e-7)]
    # subject-level predicted class mass vs prior
    sm = np.zeros((n, N_CLS))
    for s in np.unique(dd["sbj"]):
        ii = np.flatnonzero(dd["sbj"] == s)
        k = dd["sets"].get(int(s), 1)
        mass = Q[ii].sum(0) / k
        sm[ii] = mass / 97.0
        sm[ii, 0] = Q[ii, 0].mean()
    feats.append(sm)
    return np.concatenate(feats, 1).astype(np.float32)


def activity(acc, sbj, sensor):
    """per tile: std of |a|, mean |diff|, and those z-scored within subject x limb"""
    mag = np.linalg.norm(acc, axis=2)
    f = np.stack([mag.std(1), np.abs(np.diff(acc, axis=1)).mean((1, 2)), mag.mean(1)], 1)
    from .tab import standardize
    return np.concatenate([np.log(f + 1e-3), standardize(np.log(f + 1e-3), sbj, sensor)], 1).astype(np.float32)


def main(cfg):
    import lightgbm as lgb
    from .common import Paths
    from .base import write_sub
    from .decode import load_art
    from . import data
    P = Paths()
    recs, sim, dB, tB = data.build(P)
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    run = cfg.get("graph_run")         # optional better graph probabilities, e.g. 'gsweep'
    Pg_o, Pg_t = d["P_graph"].astype(np.float64), t["P_graph"].astype(np.float64)
    if run:
        zo = sorted(Path("/kaggle/input").rglob(f"oof_{run}.npz"))
        zt = sorted(Path("/kaggle/input").rglob(f"test_{run}.npz"))
        if zo and zt:
            Pg_o, Pg_t = np.load(zo[0])["P_graph"].astype(np.float64), np.load(zt[0])["P_graph"].astype(np.float64)
            print("graph probabilities from", run, flush=True)
    y, sbj = d["y"], d["sbj"]
    print(f"graph soft-cal OOF F1 {macro_f1(y, soft_cal(Pg_o, sbj, C.TRAIN_SETS).argmax(1)):.4f}", flush=True)
    act_o = activity(dB["acc"](), dB["sbj"], dB["sensor"])
    act_t = activity(tB["acc"](), tB["sbj"], tB["sensor"])
    do = dict(logp_b=d["logp_b"], logp=d["logp"], imu_logp=d["imu_logp"], succ=d["l2_succ"], score=d["l2_score_qn"],
              emb=d["emb"].astype(np.float32), sbj=sbj, sets=C.TRAIN_SETS)
    dt = dict(logp_b=t["logp_b"], logp=t["logp"], imu_logp=t["imu_logp"], succ=t["l2_succ"], score=t["l2_score_qn"],
              emb=t["emb"].astype(np.float32), sbj=t["sbj"], sets={})
    X = features(do, Pg_o, act_o)
    Xt = features(dt, Pg_t, act_t)
    print(f"pgs features {X.shape} [{time.time() - T_START:.0f}s]", flush=True)
    params = dict(PGS_PARAMS, num_threads=int(cfg.get("threads", 4)))
    rounds = int(cfg.get("rounds", 200))
    oof = np.zeros((len(X), N_CLS))
    for f in range(FOLDS):
        tr, va = d["fold"] != f, d["fold"] == f
        b = lgb.train(params, lgb.Dataset(X[tr], y[tr], params={"max_bin": 63}), rounds)
        oof[va] = b.predict(X[va])
        print(f"pgs fold {f} [{time.time() - T_START:.0f}s]", flush=True)
    res = {}
    res["pgs_argmax"] = macro_f1(y, oof.argmax(1))
    for T in (0.5, 0.7, 1.0):
        for it in (0, 1, 3):
            lab = oof.argmax(1) if it == 0 else soft_cal(oof, sbj, C.TRAIN_SETS, T, it).argmax(1)
            res[f"pgs_T{T}_it{it}"] = macro_f1(y, lab)
            print(f"pgs T={T} sinkhorn iters={it}: OOF macro F1 {res[f'pgs_T{T}_it{it}']:.4f}", flush=True)
    for wg in (0.3, 0.5, 0.7):
        Bo = norm_rows(np.exp(wg * np.log(oof + 1e-9) + (1 - wg) * np.log(Pg_o + 1e-9)))
        res[f"blend_w{wg}"] = macro_f1(y, soft_cal(Bo, sbj, C.TRAIN_SETS).argmax(1))
        print(f"pgs x graph geo w={wg}: OOF macro F1 {res[f'blend_w{wg}']:.4f}", flush=True)
    b = lgb.train(params, lgb.Dataset(X, y, params={"max_bin": 63}), rounds)
    pt = b.predict(Xt)
    np.savez_compressed(P.ART / "oof_pgs.npz", P_graph=norm_rows(oof), y=y, sbj=sbj)
    np.savez_compressed(P.ART / "test_pgs.npz", P_graph=norm_rows(pt), id=t["id"], sbj=t["sbj"])
    best_T, best_it = max(((T, it) for T in (0.5, 0.7, 1.0) for it in (1, 3)), key=lambda k: res[f"pgs_T{k[0]}_it{k[1]}"])
    write_sub(P, t, soft_cal(pt, t["sbj"], {}, best_T, best_it).argmax(1), "submission_pgs.csv")
    res["best"] = (best_T, best_it)
    C.save_json(P.ART / "pgs.json", res)
