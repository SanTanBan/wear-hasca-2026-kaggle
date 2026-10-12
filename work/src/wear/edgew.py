"""Learned link reweighting (ours). Diagnostics showed that cutting the links that cross a label boundary lifts the
graph decoder from 0.910 to 0.926 OOF, while exact-successor accuracy barely matters. So: a LightGBM predicts
P(same label | link i -> succ[i]) from both endpoints' evidence (tile log-probs, activity, video similarity, link
score, neighbourhood); the graph's link weights are multiplied by it. Nested subject folds on the OOF links."""
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import FOLDS, N_CLS, T_START, calibrate, macro_f1, norm_rows

EW_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=0)


def soft_finish(P, dd, iters=3):
    Q = np.clip(P, 1e-12, None) ** 2.0
    return calibrate(Q / Q.sum(1, keepdims=True), dd["sbj"], dd["sets"], 97, 0.05, iters=iters).argmax(1)


def bc(p, q):
    return np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)).sum(1)


def link_edge_features(dd, act, Pg=None):
    succ, score = dd["succ"], dd["score"]
    n = len(succ)
    i = np.flatnonzero(succ >= 0)
    j = succ[i]
    pred = np.full(n, -1, np.int64)
    pred[j] = i
    pb = np.exp(dd["logp_b"])
    pw = np.exp(dd["logp"])
    pi_ = np.exp(dd["imu_logp"])
    E = dd["emb"] / (np.linalg.norm(dd["emb"], axis=1, keepdims=True) + 1e-6)
    pi_pred = np.where(pred[i] >= 0, pred[i], i)
    sj_succ = np.where(succ[j] >= 0, succ[j], j)
    f = [dd["logp_b"][i], dd["logp_b"][j], bc(pb[i], pb[j])[:, None], (pb[i].argmax(1) == pb[j].argmax(1))[:, None],
         bc(pw[i], pw[j])[:, None], bc(pi_[i], pi_[j])[:, None], act[i], act[j], act[i] - act[j],
         (E[i] * E[j]).sum(1)[:, None], score[i][:, None],
         bc(pb[pi_pred], pb[i])[:, None], bc(pb[j], pb[sj_succ])[:, None], bc(pb[pi_pred], pb[sj_succ])[:, None],
         dd["sensor"][i][:, None], dd["sensor"][j][:, None], (dd["sensor"][i] == dd["sensor"][j])[:, None]]
    if Pg is not None:
        f += [np.log(np.clip(Pg[i], 1e-7, 1)), np.log(np.clip(Pg[j], 1e-7, 1)), bc(Pg[i], Pg[j])[:, None],
              (Pg[i].argmax(1) == Pg[j].argmax(1))[:, None]]
    return i, j, np.concatenate([np.asarray(x, np.float32) for x in f], 1)


def chain_window_features(dd, act, Pg, i, j, wins=((1, 2), (3, 5), (6, 10))):
    """change-point style features: evidence aggregated over windows before i and after j along the chain"""
    from .chainstack import walk
    succ = dd["succ"]
    n = len(succ)
    pred = np.full(n, -1, np.int64)
    m = succ >= 0
    pred[succ[m]] = np.flatnonzero(m)
    maxo = max(b for _, b in wins)
    bw = walk(pred, maxo)                      # bw[k][x] = k+1 steps before x
    fw = walk(succ, maxo)
    pb = np.exp(dd["logp_b"])
    feats = []
    for a, b in wins:
        def agg(idx_rows, base):
            num = np.zeros((len(base), N_CLS))
            na = np.zeros((len(base), act.shape[1]))
            cnt = np.zeros(len(base))
            num_g = np.zeros((len(base), N_CLS))
            for k in range(a - 1, b):
                ix = idx_rows[k][base]
                ok = ix >= 0
                num[ok] += pb[ix[ok]]
                num_g[ok] += Pg[ix[ok]]
                na[ok] += act[ix[ok]]
                cnt += ok
            c = np.maximum(cnt, 1)[:, None]
            return num / c, num_g / c, na / c, cnt
        # before i (walking back from i), after j (walking forward from j)
        pb_b, pg_b, a_b, c_b = agg(bw, i)
        pb_f, pg_f, a_f, c_f = agg(fw, j)
        feats += [bc(pb_b, pb_f)[:, None], bc(pg_b, pg_f)[:, None], bc(pb_b, pb[j])[:, None],
                  bc(pb[i], pb_f)[:, None], bc(pg_b, Pg[j])[:, None], bc(Pg[i], pg_f)[:, None],
                  (pb_b.argmax(1) == pb_f.argmax(1))[:, None], a_b - a_f, c_b[:, None], c_f[:, None],
                  pb_b[:, :1], pb_f[:, :1]]
    return np.concatenate([np.asarray(x, np.float32) for x in feats], 1)


def fit_predict(Xo, yo, fold_o, Xt, rounds=300, threads=4):
    import lightgbm as lgb
    params = dict(EW_PARAMS, num_threads=threads)
    oof = np.zeros(len(Xo))
    for f in range(FOLDS):
        tr, va = fold_o != f, fold_o == f
        b = lgb.train(params, lgb.Dataset(Xo[tr], yo[tr]), rounds)
        oof[va] = b.predict(Xo[va])
    b = lgb.train(params, lgb.Dataset(Xo, yo), rounds)
    return oof, b.predict(Xt)


def main(cfg):
    from .common import Paths
    from .decode import load_art
    from .pgs import activity
    from .base import write_sub
    from . import data, graph
    P = Paths()
    recs, sim, dB, tB = data.build(P)
    src = sorted(Path("/kaggle/input").rglob("oof_B.npz"))[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    y, sbj = d["y"], d["sbj"]
    act_o = activity(dB["acc"](), dB["sbj"], dB["sensor"])
    act_t = activity(tB["acc"](), tB["sbj"], tB["sensor"])
    do = dict(logp_b=d["logp_b"], logp=d["logp"], imu_logp=d["imu_logp"], succ=d["l2_succ"], score=d["l2_score_qn"],
              emb=d["emb"].astype(np.float32), sbj=sbj, sensor=d["sensor"], sets=C.TRAIN_SETS, grp=sbj)
    dt = dict(logp_b=t["logp_b"], logp=t["logp"], imu_logp=t["imu_logp"], succ=t["l2_succ"], score=t["l2_score_qn"],
              emb=t["emb"].astype(np.float32), sbj=t["sbj"], sensor=t["sensor"], sets={}, grp=t["sbj"])
    res = {}
    Pg_o, Pg_t = d["P_graph"].astype(np.float64), t["P_graph"].astype(np.float64)
    best = (macro_f1(y, soft_finish(Pg_o, do)), "base", Pg_o, Pg_t)
    print(f"base graph soft-cal OOF {best[0]:.4f}", flush=True)
    for use_pg in cfg.get("use_pg", [False, True]):
        io, jo, Xo = link_edge_features(do, act_o, Pg_o if use_pg else None)
        it, jt, Xt = link_edge_features(dt, act_t, Pg_t if use_pg else None)
        if cfg.get("chain_windows", False):
            Xo = np.concatenate([Xo, chain_window_features(do, act_o, Pg_o, io, jo)], 1)
            Xt = np.concatenate([Xt, chain_window_features(dt, act_t, Pg_t, it, jt)], 1)
        same = (y[io] == y[jo]).astype(np.float32)
        ps_o, ps_t = fit_predict(Xo, same, d["fold"][io], Xt, threads=int(cfg.get("threads", 4)))
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(same, ps_o)
        print(f"edge model (use P_graph={use_pg}): {len(io)} links, same-label rate {same.mean():.3f}, OOF AUC {auc:.4f} "
              f"[{time.time() - T_START:.0f}s]", flush=True)
        modes = [("pow", g) for g in cfg.get("gammas", [1.0, 2.0, 4.0])] + \
                [("cut", c) for c in cfg.get("cuts", [])]
        for mode, val in modes:
            mo = np.ones(len(y))
            mt = np.ones(len(t["sbj"]))
            if mode == "pow":
                mo[io] = ps_o ** val
                mt[it] = ps_t ** val
            else:
                mo[io] = (ps_o >= val).astype(float)
                mt[it] = (ps_t >= val).astype(float)
            de = dict(do, logp=d["logp_b"], link_mult=mo)
            Pe = graph.graph_P(de)
            f = macro_f1(y, soft_finish(Pe, de))
            gamma = f"{mode}{val}"
            name = f"edgew pg={use_pg} {gamma} cw={cfg.get('chain_windows', False)}"
            res[name] = f
            print(f"{name:35s} soft-cal OOF {f:.4f} (full-cal {macro_f1(y, C.finish(Pe, de)):.4f})", flush=True)
            if f > best[0]:
                dte = dict(dt, logp=t["logp_b"], link_mult=mt)
                Pt = graph.graph_P(dte)
                best = (f, name, Pe, Pt)
    f, name, Pe, Pt = best
    print("best:", name, round(f, 4), flush=True)
    np.savez_compressed(P.ART / "oof_edgew.npz", P_graph=Pe, y=y, sbj=sbj)
    np.savez_compressed(P.ART / "test_edgew.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
    write_sub(P, t, soft_finish(Pt, dict(dt)), "submission_edgew.csv")
    res["best"] = (name, f)
    C.save_json(P.ART / "edgew.json", res)
