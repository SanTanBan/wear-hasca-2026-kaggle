"""Chain stacker (ours): a LightGBM that reads the class probabilities of the tiles along the reconstructed chain
(several forward/backward windows, weighted by the cumulative link confidence) plus video-kNN context, trained with
nested folds on the OOF tiles. Its output replaces/blends the tile log-probs that enter the graph propagation."""
import time

import numpy as np

from .common import FOLDS, N_CLS, T_START, log_softmax, macro_f1

WINDOWS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 30)]
CS_PARAMS = dict(objective="multiclass", num_class=N_CLS, learning_rate=0.06, num_leaves=31, min_data_in_leaf=60,
                 feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63,
                 num_threads=4, verbose=-1, seed=0)


def walk(succ, steps):
    """idx[k] = tile k steps along succ (-1 when the chain ends); also the min link prob along the way"""
    n = len(succ)
    out = np.full((steps, n), -1, np.int64)
    cur = np.arange(n)
    for k in range(steps):
        cur = np.where(cur >= 0, succ[np.maximum(cur, 0)], -1)
        out[k] = cur
    return out


def chain_features(P, succ, score, emb, sbj, max_off=30, knn_k=10):
    n = len(succ)
    pred = np.full(n, -1, np.int64)
    m = succ >= 0
    pred[succ[m]] = np.flatnonzero(m)
    lp = 1 / (1 + np.exp(-np.clip(score, -20, 20)))               # prob of the link i -> succ[i]
    lp_pred = np.where(pred >= 0, lp[np.maximum(pred, 0)], 0.0)     # prob of the link pred[i] -> i
    fw, bw = walk(succ, max_off), walk(pred, max_off)
    # cumulative confidence along the walk (product of link probs)
    cf = np.ones((max_off, n))
    cb = np.ones((max_off, n))
    cur_f, cur_b = np.arange(n), np.arange(n)
    acc_f, acc_b = np.ones(n), np.ones(n)
    for k in range(max_off):
        acc_f = acc_f * np.where(cur_f >= 0, lp[np.maximum(cur_f, 0)], 0)
        acc_b = acc_b * np.where(cur_b >= 0, lp_pred[np.maximum(cur_b, 0)], 0)
        cf[k], cb[k] = acc_f, acc_b
        cur_f, cur_b = fw[k], bw[k]
    feats = [np.log(np.clip(P, 1e-7, 1))]
    for idx, conf in ((fw, cf), (bw, cb)):
        for a, b in WINDOWS:
            ks = np.arange(a - 1, b)
            num = np.zeros((n, N_CLS))
            den = np.zeros(n)
            for k in ks:
                w = np.where(idx[k] >= 0, conf[k], 0.0)
                num += w[:, None] * P[np.maximum(idx[k], 0)]
                den += w
            feats.append(np.log(num / np.maximum(den, 1e-9)[:, None] + 1e-6) * (den > 1e-6)[:, None])
            feats.append(den[:, None])
    # chain length context
    feats.append(((fw >= 0).sum(0))[:, None].astype(float))
    feats.append(((bw >= 0).sum(0))[:, None].astype(float))
    feats.append(score[:, None])
    feats.append(np.where(pred >= 0, score[np.maximum(pred, 0)], -50)[:, None])
    # video kNN context within the subject
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    kn = np.zeros((n, N_CLS))
    sims = np.zeros((n, 2))
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        S = E[ii] @ E[ii].T
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, knn_k, axis=1)[:, :knn_k]
        kn[ii] = P[ii][nb].mean(1)
        sv = np.take_along_axis(S, nb, 1)
        sims[ii, 0], sims[ii, 1] = sv.max(1), sv.mean(1)
    feats.append(np.log(kn + 1e-6))
    feats.append(sims)
    return np.concatenate(feats, 1).astype(np.float32)


ML_FEATS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z", "std_m", "energy_m", "jerk_m", "domf_m",
            "domfrac_m", "logpow_m", "acmax_m", "aclag_m", "band1-1_m", "band2-2_m", "band3-4_m", "band5-7_m",
            "spent_m", "grav_x", "grav_y", "grav_z", "corr_xy", "corr_xz", "corr_yz", "range_m", "kurt_m", "zcr_m"]


def imu_matrix(acc, sbj, sensor):
    """selected IMU statistics of each tile, z-scored within (subject x limb)"""
    from .tab import imu_feats, standardize
    X, names = imu_feats(acc)
    cols = [names.index(k) for k in ML_FEATS]
    return standardize(X[:, cols].astype(np.float32), sbj, sensor)


def multilimb_features(Z, sensor, idx_list, w_list):
    """per limb: weighted mean of the neighbours' IMU statistics (neighbours = idx_list with weights w_list), plus the
    weight mass per limb. Gives every tile a 'virtual 4-limb' view of its activity."""
    n, F_ = Z.shape
    num = np.zeros((4, n, F_))
    den = np.zeros((4, n))
    for idx, w in zip(idx_list, w_list):
        ok = idx >= 0
        j = np.maximum(idx, 0)
        lj = sensor[j]
        ww = np.where(ok, w, 0.0)
        for l in range(4):
            m = ww * (lj == l)
            num[l] += m[:, None] * Z[j]
            den[l] += m
    out = []
    for l in range(4):
        out.append(num[l] / np.maximum(den[l], 1e-9)[:, None] * (den[l] > 1e-6)[:, None])
        out.append(den[l][:, None])
    return np.concatenate(out, 1).astype(np.float32)


def neighbour_lists(succ, score, emb, sbj, max_off=15, knn_k=20, Pk=None, use_p=0.0):
    """chain neighbours (+-max_off along the links, weighted by cumulative link probability) and video-kNN neighbours
    (optionally with prediction similarity use_p * sqrt(Pk) . sqrt(Pk), as in the graph's kNN)"""
    n = len(succ)
    pred = np.full(n, -1, np.int64)
    m = succ >= 0
    pred[succ[m]] = np.flatnonzero(m)
    lp = 1 / (1 + np.exp(-np.clip(score, -20, 20)))
    lp_pred = np.where(pred >= 0, lp[np.maximum(pred, 0)], 0.0)
    idx, ws = [np.arange(n)], [np.ones(n)]
    for nxt, prob in ((succ, lp), (pred, lp_pred)):
        cur, acc = np.arange(n), np.ones(n)
        for k in range(max_off):
            acc = acc * np.where(cur >= 0, prob[np.maximum(cur, 0)], 0)
            cur = np.where(cur >= 0, nxt[np.maximum(cur, 0)], -1)
            idx.append(cur.copy())
            ws.append(acc.copy())
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    kidx = np.full((knn_k, n), -1, np.int64)
    kw = np.zeros((knn_k, n))
    Q = np.sqrt(Pk).astype(np.float32) if (Pk is not None and use_p) else None
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        S = E[ii] @ E[ii].T
        if Q is not None:
            S = (S + use_p * (Q[ii] @ Q[ii].T)) / (1 + use_p)
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, knn_k, axis=1)[:, :knn_k]
        kidx[:, ii] = ii[nb].T
        kw[:, ii] = np.clip(np.take_along_axis(S, nb, 1), 0, None).T
    return (idx, ws), (list(kidx), list(kw))


def run(d, t, P_oof, P_test, succ_o, score_o, succ_t, score_t, rounds=150, threads=4, tag="cs", multilimb=False,
        ml2=False, seed=0):
    """ml2: also per-limb aggregates of the IMU-only / fusion model probabilities of the neighbours, and a kNN that
    includes prediction similarity (purer neighbours)"""
    import lightgbm as lgb
    params = dict(CS_PARAMS, num_threads=threads, seed=seed)
    X = chain_features(P_oof, succ_o, score_o, d["emb"], d["sbj"])
    Xt = chain_features(P_test, succ_t, score_t, t["emb"], t["sbj"])
    if multilimb:
        for dd, succ, score, which, Pin in ((d, succ_o, score_o, "oof", P_oof), (t, succ_t, score_t, "test", P_test)):
            Z = imu_matrix(dd["acc"](), dd["sbj"], dd["sensor"])
            (ci, cw), (ki, kw) = neighbour_lists(succ, score, dd["emb"], dd["sbj"], Pk=Pin,
                                                 use_p=0.5 if ml2 else 0.0)
            parts = [Z, multilimb_features(Z, dd["sensor"], ci, cw), multilimb_features(Z, dd["sensor"], ki, kw)]
            if ml2:
                for key in ("imu_logp", "fusion_logp"):
                    Pm = np.exp(dd[key]).astype(np.float32)
                    parts += [multilimb_features(Pm, dd["sensor"], ci, cw), multilimb_features(Pm, dd["sensor"], ki, kw)]
            extra = np.concatenate(parts, 1)
            if which == "oof":
                X = np.concatenate([X, extra], 1)
            else:
                Xt = np.concatenate([Xt, extra], 1)
    print(f"{tag} features {X.shape} [{time.time() - T_START:.0f}s]", flush=True)
    oof = np.zeros((len(X), N_CLS), np.float32)
    for f in range(FOLDS):
        tr, va = d["fold"] != f, d["fold"] == f
        b = lgb.train(params, lgb.Dataset(X[tr], d["y"][tr], params={"max_bin": params["max_bin"]}), rounds)
        oof[va] = np.log(np.clip(b.predict(X[va]), 1e-7, 1))
    print(f"{tag}: nested OOF argmax macro F1 {macro_f1(d['y'], oof.argmax(1)):.4f} [{time.time() - T_START:.0f}s]",
          flush=True)
    b = lgb.train(params, lgb.Dataset(X, d["y"], params={"max_bin": params["max_bin"]}), rounds)
    test = np.log(np.clip(b.predict(Xt), 1e-7, 1)).astype(np.float32)
    return oof, test
