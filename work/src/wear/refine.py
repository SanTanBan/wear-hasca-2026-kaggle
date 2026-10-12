"""Boundary refiner along the link chains, applied to our final blend (idea and features from the public notebook
'Learned Links + Counts' by goodpjw2008, section 9).

For every matching (successor links: base L2, L3, L4, L6, L4s), walk the chains; at each place where the final labels
change (tile i labelled A, its successor j labelled B) the tiles up to two steps before and three after are re-judged:
given the experts (window log-probs, blend B, the blend probabilities P and their calibration Q), movement energy,
posture and video continuity with the chain neighbours and the position relative to the change, should the tile take
the other side's label? LightGBM nested by subject fold on the OOF tiles; a tile flips when its mean flip probability
over the K matchings exceeds the threshold.

config: {"probs": "eqgp65q", "links": [["L2"], ["L3", "wear26-l4"], ["L3", "wear26-l3"], ["L3", "wear26-l6"],
         ["links_L4s", "wear26-l4s"]], "thr": [0.4, 0.5, 0.6], "rounds": 300, "threads": 4, "tag": "ref1"}
"""
from pathlib import Path

import numpy as np

from . import common as C
from .common import FOLDS, N_CLS, macro_f1, stage

REF_H = 3


def find(name, must=None):
    hits = sorted(p for p in Path("/kaggle/input").rglob(name) if must is None or must in str(p))
    assert hits, (name, must)
    return hits[0]


def soft_q(P, sbj, sets, T=0.5, iters=3):
    Q = np.clip(P.astype(np.float64), 1e-12, None) ** (1 / T)
    return C.calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, 97, C.CFG["null_min"], iters=iters)


def tile_scalars(dd):
    """per-tile movement energy, mean posture, video motion and frame mean"""
    acc = np.asarray(dd["acc"](), np.float32)
    ener = np.log(np.linalg.norm(acc, axis=2).std(1) + 0.01)
    post = acc.mean(1)
    V = dd["vid"]()
    n = len(acc)
    vmot, vmean = np.zeros(n, np.float32), np.zeros((n, V.shape[2]), np.float32)
    for a in range(0, n, 4000):
        v = np.asarray(V[a:a + 4000], np.float32)
        vmot[a:a + 4000] = np.linalg.norm(np.diff(v, axis=1), axis=2).mean(1)
        vmean[a:a + 4000] = v.mean(1)
    return ener, post, vmot, vmean


def refiner_rows(fin, succ, B, P, Q, lw, sens, ener, post, vmot, vmean, extra=()):
    """for one matching: the windows around every label change along the chains -> (features, tile, other label).
    extra: further experts' log-probabilities (e.g. our own blend and the public pipeline separately); each adds the
    cur-vs-other log-ratio of the tile and of its chain neighbours -1/+1."""
    n = len(fin)
    prv = np.full(n, -1)
    ok = succ >= 0
    prv[succ[ok]] = np.flatnonzero(ok)
    lP, lQ = np.log(np.clip(P, 1e-9, None)), np.log(Q + 1e-9)

    def step(g, k):
        for _ in range(abs(k)):
            g = succ[g] if k > 0 else prv[g]
            if g < 0:
                return -1
        return g

    rows, tiles, others = [], [], []
    for i in np.flatnonzero(ok & (fin != fin[np.maximum(succ, 0)])):
        j = succ[i]
        A, Bc = fin[i], fin[j]
        for off in range(-REF_H + 1, REF_H + 1):
            g = step(i, off) if off <= 0 else step(j, off - 1)
            if g < 0:
                continue
            cur, oth = fin[g], (Bc if off <= 0 else A)
            if cur == oth:
                continue
            nxt, prv_ = succ[g], prv[g]
            f = [off, int(cur == 0), int(oth == 0), B[g, cur] - B[g, oth], lP[g, cur] - lP[g, oth],
                 lw[g, cur] - lw[g, oth], lQ[g, cur] - lQ[g, oth], B[g, 0], lP[g, 0], ener[g], vmot[g],
                 float(np.linalg.norm(vmean[g] - vmean[prv_])) if prv_ >= 0 else -1.0,
                 float(np.linalg.norm(vmean[g] - vmean[nxt])) if nxt >= 0 else -1.0]
            for nb in (-2, -1, 1, 2):
                gj = step(g, nb)
                if gj >= 0:
                    f += [B[gj, cur] - B[gj, oth], lP[gj, cur] - lP[gj, oth], ener[gj] - ener[g], int(sens[gj] == sens[g]),
                          float(np.linalg.norm(post[gj] - post[g])) if sens[gj] == sens[g] else -1.0, int(fin[gj] == cur)]
                else:
                    f += [0, 0, 0, 0, -1.0, -1]
            for X_ in extra:
                f.append(X_[g, cur] - X_[g, oth])
                for nb in (-1, 1):
                    gj = step(g, nb)
                    f.append(X_[gj, cur] - X_[gj, oth] if gj >= 0 else 0.0)
            rows.append(f)
            tiles.append(g)
            others.append(oth)
    return np.array(rows, np.float32), np.array(tiles, np.int64), np.array(others, np.int64)


def refiner_flips(fin, tiles, others, prob, K, thr):
    """mean flip probability per (tile, other label) over the K matchings (absent = no flip); flip above thr"""
    acc = {}
    for g, o, p_ in zip(tiles.tolist(), others.tolist(), prob.tolist()):
        acc[(g, o)] = acc.get((g, o), 0.0) + p_ / K
    best = {}
    for (g, o), m in acc.items():
        if m > thr and m > best.get(g, (0, 0))[0]:
            best[g] = (m, o)
    new = fin.copy()
    for g, (m, o) in best.items():
        new[g] = o
    return new, len(best)


_DUMPS = {}


def load_links(spec, a, b):
    kind = spec[0]
    if kind == "L2":
        return (a["l2_succ"], a["l2_score_qn"]), (b["l2_succ"], b["l2_score_qn"])
    if kind == "dump":       # bagged L3 matchings of our run of the goodpjw notebook: ["dump", "wear26-gpjw/", k]
        if spec[1] not in _DUMPS:
            _DUMPS[spec[1]] = dict(np.load(find("dump_stageB.npz", spec[1])))
        z, k = _DUMPS[spec[1]], int(spec[2])
        return (z["succ_oof"][k], z["score_oof"][k]), (z["succ_test"][k], z["score_test"][k])
    from .decode import load_art
    if kind.startswith("links_"):
        z = load_art(find(f"{kind}.npz", spec[1]))
        return (z["oof_succ"], z["oof_score_qn"]), (z["test_succ"], z["test_score_qn"])
    lo, lt = load_art(find(f"oof_{kind}.npz", spec[1])), load_art(find(f"test_{kind}.npz", spec[1]))
    return (lo[f"{kind}_succ"], lo[f"{kind}_score_qn"]), (lt[f"{kind}_succ"], lt[f"{kind}_score_qn"])


def main(cfg):
    """config: probs (blend tag), links, thr (list), rounds, passes (default 1), extra (list of further blend tags
    whose log-probabilities become features), tag"""
    import lightgbm as lgb
    from .common import Paths
    from .base import write_sub
    from .decode import load_art
    from . import data
    P_ = Paths()
    threads = int(cfg.get("threads", 4))
    tag = cfg.get("tag", "ref")
    recs, sim, d, t = data.build(P_)
    a, b = load_art(find("oof_B.npz")), load_art(find("test_B.npz"))
    po = load_art(find(f"oof_{cfg['probs']}.npz"))["P_graph"].astype(np.float64)
    zt = load_art(find(f"test_{cfg['probs']}.npz"))
    pt = zt["P_graph"].astype(np.float64)
    assert (zt["id"] == t["id"]).all(), "test order of the blend differs from data.build"
    ex_o, ex_t = [], []
    for tg in cfg.get("extra", []):
        eo, et = load_art(find(f"oof_{tg}.npz")), load_art(find(f"test_{tg}.npz"))
        assert (et["id"] == t["id"]).all(), tg
        ex_o.append(np.log(np.clip(eo["P_graph"].astype(np.float64), 1e-9, None)))
        ex_t.append(np.log(np.clip(et["P_graph"].astype(np.float64), 1e-9, None)))
    Qo, Qt = soft_q(po, d["sbj"], C.TRAIN_SETS), soft_q(pt, t["sbj"], {})
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    base = macro_f1(d["y"], fin_o)
    print(f"input {cfg['probs']} (+ extra experts {cfg.get('extra', [])}): OOF macro F1 (soft decode) {base:.4f}",
          flush=True)
    links = [load_links(s, a, b) for s in cfg.get("links", [["L2"]])]
    K = len(links)
    for s, ((so, _), _) in zip(cfg.get("links", [["L2"]]), links):
        m = so >= 0
        print(f"links {s}: linked {m.mean():.3f} exact {(so[m] == d['true_succ'][m]).mean():.3f} "
              f"same-label {(d['y'][so[m]] == d['y'][m]).mean():.3f}", flush=True)
    with stage("tile scalars"):
        sc_o, sc_t = tile_scalars(d), tile_scalars(t)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=threads)
    rounds = int(cfg.get("rounds", 300))
    thrs = cfg.get("thr", [0.5])

    def rows(fin, which):
        Xs, tls, ots = [], [], []
        for (so, _), (st, _) in links:
            if which == "oof":
                X, tl_, ot_ = refiner_rows(fin, so, a["logp_b"], po, Qo, a["logp"], d["sensor"], *sc_o, extra=ex_o)
            else:
                X, tl_, ot_ = refiner_rows(fin, st, b["logp_b"], pt, Qt, b["logp"], t["sensor"], *sc_t, extra=ex_t)
            Xs.append(X)
            tls.append(tl_)
            ots.append(ot_)
        return np.concatenate(Xs), np.concatenate(tls), np.concatenate(ots)

    res = {"base": base}
    cur_o, cur_t = fin_o, fin_t
    for p in range(int(cfg.get("passes", 1))):
        with stage(f"refiner pass {p + 1}"):
            X, tl, ot = rows(cur_o, "oof")
            T = ((d["y"][tl] == ot) & (d["y"][tl] != cur_o[tl])).astype(int)
            Fr = d["fold"][tl]
            err = np.flatnonzero(cur_o != d["y"])
            print(f"pass {p + 1}: {len(T)} candidate rows ({X.shape[1]} features), {T.mean():.3f} should flip; the "
                  f"windows hold {np.isin(err, tl).sum()} of the {len(err)} remaining errors", flush=True)
            pr = np.zeros(len(T))
            for k in range(FOLDS):
                tr = Fr != k
                pr[~tr] = lgb.train(params, lgb.Dataset(X[tr], T[tr]), rounds).predict(X[~tr])
            f0 = macro_f1(d["y"], cur_o)
            best = (f0, None)
            for thr in thrs:
                ref_o, nfl = refiner_flips(cur_o, tl, ot, pr, K, thr)
                f = macro_f1(d["y"], ref_o)
                res[f"p{p + 1}_thr{thr}"] = f
                print(f"pass {p + 1} thr {thr}: {nfl} OOF tiles flipped, macro F1 {f0:.4f} -> {f:.4f} | per fold " +
                      " ".join(f"{macro_f1(d['y'][d['fold'] == k], ref_o[d['fold'] == k]):.4f}" for k in range(FOLDS)),
                      flush=True)
                if f > best[0]:
                    best = (f, thr)
            model = lgb.train(params, lgb.Dataset(X, T), rounds)
            Xt, tlt, ott = rows(cur_t, "test")
            prt = model.predict(Xt)
            for th in thrs:
                lab, nfl = refiner_flips(cur_t, tlt, ott, prt, K, th)
                print(f"pass {p + 1} test thr {th}: {nfl} tiles flipped", flush=True)
                write_sub(P_, t, lab, f"submission_{tag}_p{p + 1}_thr{th}.csv")
            if best[1] is None:
                print(f"pass {p + 1}: no threshold improves OOF - stop", flush=True)
                break
            res[f"p{p + 1}_best"] = best
            cur_o, _ = refiner_flips(cur_o, tl, ot, pr, K, best[1])
            cur_t, _ = refiner_flips(cur_t, tlt, ott, prt, K, best[1])
    write_sub(P_, t, cur_t, f"submission_{tag}.csv")
    np.savez_compressed(P_.ART / f"{tag}_labels.npz", oof=cur_o, test=cur_t, id=t["id"], sbj=t["sbj"],
                        oof_base=fin_o, test_base=fin_t)
    res["final"] = macro_f1(d["y"], cur_o)
    print(f"final OOF {base:.4f} -> {res['final']:.4f}", flush=True)
    C.save_json(P_.ART / f"{tag}.json", res)
