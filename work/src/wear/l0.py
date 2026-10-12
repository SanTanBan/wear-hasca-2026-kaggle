"""L0 timeline links (honghanhhh's Hungarian chain, via Briano's notebook, Apache 2.0): candidate successors from
video-boundary similarities, 12 pair features, a LightGBM scorer, Hungarian 1:1 assignment with cycle breaking."""
import json
import time

import lightgbm as lgb
import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import FOLDS, VLEN, WIN, T_START

_K_TH, _K_11, _K_MP = 40, 15, 15


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def _topk(sim_, k):
    k = min(k, sim_.shape[1] - 1)
    return np.argpartition(-sim_, k - 1, axis=1)[:, :k]


def pair_features(video, inertial, limbs):
    n = len(video)
    fr = np.asarray(video, dtype=np.float32).copy()
    fr /= np.maximum(np.linalg.norm(fr, axis=2, keepdims=True), 1e-9)
    tail, head = _unit(fr[:, -3:].mean(1)), _unit(fr[:, :3].mean(1))
    last, first, mp = fr[:, -1], fr[:, 0], _unit(fr.mean(1))
    th, lf, mm = tail @ head.T, last @ first.T, mp @ mp.T
    for m in (th, lf, mm):
        np.fill_diagonal(m, -1.0)
    merged = np.sort(np.concatenate([_topk(th, _K_TH), _topk(lf, _K_11), _topk(mm, _K_MP)], 1), 1)
    dup = np.zeros_like(merged, dtype=bool)
    dup[:, 1:] = merged[:, 1:] == merged[:, :-1]
    merged[dup] = -1
    cand = np.full_like(merged, -1)
    for r in range(n):
        v = merged[r, merged[r] >= 0]
        cand[r, :len(v)] = v
    valid = cand >= 0
    ch = np.where(valid, cand, 0)
    rows = np.arange(n)[:, None]
    fwd, frame, pooled, rev = th[rows, ch], lf[rows, ch], mm[rows, ch], th[ch, rows]
    dom_r = fwd - th.max(1)[:, None]
    dom_c = fwd - th.max(0)[ch]
    srt = np.sort(th, 1)
    margin = np.broadcast_to((srt[:, -1] - srt[:, -2])[:, None], fwd.shape)
    rank_row = np.empty((n, n), np.int32)
    rank_row[rows, np.argsort(-th, 1)] = np.arange(n, dtype=np.int32)[None]
    rank_col = np.empty((n, n), np.int32)
    rank_col[np.argsort(-th, 0), np.arange(n)[None]] = np.arange(n, dtype=np.int32)[:, None]
    same = (limbs[:, None] == limbs[ch]).astype(np.float32)
    a = np.asarray(inertial, dtype=np.float32)
    gap = np.linalg.norm(a[:, -1][:, None] - a[ch][:, :, 0], axis=2)
    ext = np.linalg.norm((2 * a[:, -1] - a[:, -2])[:, None] - a[ch][:, :, 0], axis=2)
    feats = np.stack([fwd, frame, pooled, rev, dom_r, dom_c, margin,
                      np.log1p(rank_row[rows, ch]), np.log1p(rank_col[rows, ch]), same,
                      np.where(same > 0, np.log1p(gap), np.nan), np.where(same > 0, np.log1p(ext), np.nan)],
                     2).astype(np.float32)
    feats[~valid] = np.nan
    return cand, feats


def train_tiles(P, recs, rec_ids, seed):
    out = []
    for i in rec_ids:
        st = recs[i]["stem"]
        acc = np.load(P.PROC / f"{st}_acc.npy").reshape(-1, 4, 3)
        ok = np.load(P.PROC / f"{st}_valid.npy")
        vid = np.load(P.PROC / f"{st}_vid.npy", mmap_mode="r")
        s = np.arange(0, len(acc) - WIN + 1, WIN)
        s = s[s * 3 // 5 + 8 + VLEN <= len(vid)]
        n = len(s)
        miss = np.zeros((len(ok) + 1, 4), np.int32)
        np.cumsum(~ok, axis=0, out=miss[1:])
        okw = (miss[s + WIN] - miss[s]) == 0
        rng = np.random.default_rng(seed + i * 17)
        cp = np.cumsum(okw / np.maximum(okw.sum(1, keepdims=True), 1), 1)
        limbs = np.minimum((rng.random(n)[:, None] > cp).sum(1), 3)
        a = acc[s[:, None] + np.arange(WIN), limbs[:, None]]
        v = np.asarray(vid[(s * 3 // 5)[:, None] + 8 + np.arange(VLEN)], dtype=np.float32)
        out.append((v, a, limbs))
    return out


def pair_dataset(tiles, negatives, rng):
    X, Y = [], []
    for v, a, limbs in tiles:
        cand, feats = pair_features(v, a, limbs)
        n = len(cand)
        pos = cand == (np.arange(n)[:, None] + 1)
        pos[-1] = False
        for r in range(n):
            p_ = np.flatnonzero(pos[r])
            ng = np.flatnonzero((cand[r] >= 0) & ~pos[r])
            if len(ng) > negatives:
                ng = rng.choice(ng, negatives, replace=False)
            take = np.concatenate([p_, ng])
            X.append(feats[r, take])
            Y.append(np.r_[np.ones(len(p_)), np.zeros(len(ng))])
    return np.concatenate(X), np.concatenate(Y).astype(int)


def fit_scorer(P, recs, rec_ids, seed=42, negatives=4, rounds=200):
    X, Y = pair_dataset(train_tiles(P, recs, rec_ids, seed), negatives, np.random.default_rng(seed))
    model = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)
    model.fit(np.nan_to_num(X, nan=0.0), Y)
    return model


def logodds(model, feats):
    flat = feats.reshape(-1, feats.shape[-1])
    out = np.full(len(flat), -50.0, np.float32)
    ok = np.isfinite(flat[:, 0])
    p_ = np.clip(model.predict_proba(np.nan_to_num(flat[ok], nan=0.0))[:, 1], 1e-6, 1 - 1e-6)
    out[ok] = np.log(p_ / (1 - p_))
    return out.reshape(feats.shape[:2])


def assign(cost, min_logodds=-1.0):
    """dense cost (n,n) of log-odds (diag very low) -> Hungarian 1:1 -> drop weak links -> break cycles"""
    n = len(cost)
    r, c = linear_sum_assignment(-cost)
    succ = np.full(n, -1, np.int64)
    succ[r] = c
    edge = cost[np.arange(n), succ]
    succ[edge < min_logodds] = -1
    seen = np.zeros(n, bool)
    for s in range(n):
        if seen[s] or succ[s] < 0:
            continue
        path, pos, node = [], {}, s
        while node >= 0 and not seen[node]:
            seen[node] = True
            pos[node] = len(path)
            path.append(node)
            node = int(succ[node])
        if node >= 0 and node in pos:
            cyc = path[pos[node]:]
            succ[cyc[int(np.argmin([cost[x, succ[x]] for x in cyc]))]] = -1
    score = np.where(succ >= 0, cost[np.arange(n), np.maximum(succ, 0)], -50.0)
    return succ, score.astype(np.float32)


def l0_assign(cand, lo, min_logodds=-1.0):
    n = len(cand)
    cost = np.full((n, n), -50.0, np.float32)
    valid = cand >= 0
    rows = np.broadcast_to(np.arange(n)[:, None], cand.shape)
    cost[rows[valid], cand[valid]] = lo[valid]
    np.fill_diagonal(cost, -1e4)
    return assign(cost, min_logodds)


def link_group(model, ii, video, acc, limbs):
    cand, feats = pair_features(video, acc, limbs)
    lo = logodds(model, feats)
    succ, score = l0_assign(cand, lo)
    return np.where(succ >= 0, ii[np.maximum(succ, 0)], -1), score


def link_report(name, d, succ, mask=None):
    m = succ >= 0
    if mask is not None:
        m &= mask
    ts = d["true_succ"]
    has = ts >= 0
    exact = (succ[m] == ts[m]).mean()
    recall = (succ[has] == ts[has]).mean()
    same = (d["y"][succ[m]] == d["y"][m]).mean()
    print(f"{name}: linked {m.mean():.3f} exact(of linked) {exact:.3f} recall(of true) {recall:.3f} same-label {same:.3f}",
          flush=True)
    return dict(linked=float(m.mean()), exact=float(exact), recall=float(recall), same_label=float(same))


def run(P, recs, d, t, rounds=200):
    _V, _A = d["vid"](), d["acc"]()
    n_oof = len(d["y"])
    succ_all, score_all = np.full(n_oof, -1, np.int64), np.full(n_oof, -50.0, np.float32)
    for f in range(FOLDS):
        held = set(d["sbj"][d["fold"] == f].tolist())
        model = fit_scorer(P, recs, [i for i, r in enumerate(recs) if r["sbj"] not in held], rounds=rounds)
        for sb in np.unique(d["sbj"][d["fold"] == f]):
            ii = np.flatnonzero(d["sbj"] == sb)
            succ_all[ii], score_all[ii] = link_group(model, ii, np.asarray(_V[ii], np.float32), _A[ii], d["sensor"][ii])
        print(f"L0 fold {f} done [{time.time() - T_START:.0f}s]", flush=True)
    rep = link_report("L0 OOF", d, succ_all)
    d["l0_succ"], d["l0_score"] = succ_all, score_all
    del _V, _A
    model = fit_scorer(P, recs, list(range(len(recs))), rounds=rounds)
    _tv, _ta = t["vid"](), t["acc"]()
    t_succ, t_score = np.full(len(t["sbj"]), -1, np.int64), np.full(len(t["sbj"]), -50.0, np.float32)
    for sb in np.unique(t["sbj"]):
        ii = np.flatnonzero(t["sbj"] == sb)
        t_succ[ii], t_score[ii] = link_group(model, ii, np.asarray(_tv[ii], np.float32), _ta[ii], t["sensor"][ii])
        print(f"L0 test sbj {sb} n={len(ii)} linked {(t_succ[ii] >= 0).mean():.3f}", flush=True)
    t["l0_succ"], t["l0_score"] = t_succ, t_score
    return rep
