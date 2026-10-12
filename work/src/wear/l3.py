"""L3 links (ours): Briano's L2 candidate sources + 65 pair features, plus the neural successor scores (nlink bi-encoder)
as extra candidate sources and features. Fold-held-out LightGBM scorers give honest OOF links; then the same graph
propagation + calibration as the base pipeline. Inputs: base artifacts (oof_A/test_A or *_B) and nlink artifacts."""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import common as C
from . import l2
from .common import FOLDS, T_START, macro_f1, class_f1, stage


def find_art(name):
    hits = sorted(Path("/kaggle/input").rglob(name))
    assert hits, f"{name} not found under /kaggle/input"
    return hits[0]


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def log_softmax_np(S, axis):
    m = S.max(axis, keepdims=True)
    return S - m - np.log(np.exp(S - m).sum(axis, keepdims=True))


def nl_extra(S, kr=20, kc=10, suffix=""):
    S = S.astype(np.float32)
    np.fill_diagonal(S, -1e4)
    D = (log_softmax_np(S, 1) + log_softmax_np(S, 0)).astype(np.float32)
    np.fill_diagonal(D, -1e4)
    return {"nl" + suffix: (S, kr, kc), "nld" + suffix: (D, kr, kc)}


def nl_extras(nls, ii, g=None, kr=20, kc=10):
    """nls: {tag: nlink artifacts} -> extra matrices for every model (suffix '' for a single model)"""
    ex = {}
    for k, (tag, nl) in enumerate(sorted(nls.items())):
        S = nl_oof(nl, ii, g) if g is not None else nl_test(nl, ii)
        ex.update(nl_extra(S, kr, kc, "" if len(nls) == 1 else f"_{tag}"))
    return ex


def nl_oof(nl, ii, g):
    t, h = nl["oof_t"][ii].astype(np.float32), nl["oof_h"][ii].astype(np.float32)
    return (t @ h.T) / float(nl["taus"][g])


def nl_test(nl, ii, models=None):
    models = models if models is not None else range(len(nl["taus"]))
    S = 0
    for m in models:
        t, h = nl["test_t"][m, ii].astype(np.float32), nl["test_h"][m, ii].astype(np.float32)
        S = S + (t @ h.T) / float(nl["taus"][m])
    return S / len(list(models))


def quantile_ref(score, succ):
    v = np.sort(score[succ >= 0].astype(np.float64))
    return np.quantile(v, np.linspace(0, 1, 1001))


def run(P, recs, d, t, nl, fold_of, threads=4, rounds=l2.LK_ROUNDS, kr=20, kc=10, tag="L3", soft=0.0, soft_win=10):
    """soft > 0: training target = 1 for the true successor, `soft` for a same-label tile of the same recording
    within +-soft_win seconds of the true successor (label-purity objective), 0 otherwise"""
    nls = nl if isinstance(nl, dict) and "taus" not in nl else {"v": nl}
    params = dict(l2.LK_PARAMS, num_threads=threads)
    RIDGES = l2.ridges(P, recs, fold_of)
    _V, _A = d["vid"](), d["acc"]()
    per_sbj, Xs, Ts, Gs = {}, [], [], []
    ex_keys = None
    for s in np.unique(d["sbj"]):
        ii = np.flatnonzero(d["sbj"] == s)
        n = len(ii)
        g = fold_of[int(s)]
        ex = nl_extras(nls, ii, g, kr, kc)
        ex_keys = sorted(ex)
        diag = {"nl": ex[[k for k in ex_keys if k.startswith("nl") and not k.startswith("nld")][0]],
                "nld": ex[[k for k in ex_keys if k.startswith("nld")][0]]}
        cand, valid, base, new, ext = l2.subject_all(_V[ii], _A[ii], d["sensor"][ii], d["logp"][ii], d["plp"][ii],
                                                     [RIDGES[g]], extra=ex)
        X_all = np.concatenate([base.astype(np.float32), new[0], ext], 1)
        tl_ = np.full(len(d["y"]), -1)
        tl_[ii] = np.arange(n)
        ts = d["true_succ"][ii]
        true_loc = np.where(ts >= 0, tl_[np.maximum(ts, 0)], -1)
        rp = np.repeat(np.arange(n), valid.sum(1))
        cj = cand[valid]
        pos = cj == true_loc[rp]
        rng = np.random.default_rng(7000 + int(s))
        key = rng.random(len(rp))
        key[pos] = -1.0
        o = np.lexsort((key, rp))
        start = np.r_[0, np.cumsum(np.bincount(rp, minlength=n))]
        rank = np.arange(len(o)) - start[rp[o]]
        npos = np.bincount(rp, weights=pos.astype(float), minlength=n).astype(np.int64)
        keep = o[rank < l2.LK_NEG + npos[rp[o]]]
        Xs.append(X_all[keep])
        target = pos.astype(np.float32)
        if soft > 0:
            ysub, rsub, ssub = d["y"][ii], d["rec"][ii], d["start"][ii]
            near = (ysub[cj] == ysub[rp]) & (rsub[cj] == rsub[rp]) & \
                (np.abs(ssub[cj] - ssub[rp] - 50) <= soft_win * 50) & ~pos
            target = np.where(pos, 1.0, np.where(near, soft, 0.0)).astype(np.float32)
        Ts.append(target[keep])
        Gs.append(np.full(len(keep), g))
        recall = ((cand == true_loc[:, None]) & valid).any(1)[true_loc >= 0].mean()
        nl_top1 = (diag["nl"][0].argmax(1) == true_loc)[true_loc >= 0].mean()
        nld_arg = diag["nld"][0].argmax(1)
        nld_top1 = (nld_arg == true_loc)[true_loc >= 0].mean()
        ys = d["y"][ii]
        nld_same = (ys[nld_arg] == ys)[true_loc >= 0].mean()
        per_sbj[int(s)] = (ii, cand, valid, X_all)
        print(f"{tag} sbj {s} fold {g} n={n} K={cand.shape[1]} recall {recall:.3f} nl-top1 {nl_top1:.3f} "
              f"nl-dual-top1 {nld_top1:.3f} nl-dual same-label {nld_same:.3f} [{time.time() - T_START:.0f}s]",
              flush=True)
        del base, new, ext, ex, diag
    del _V, _A
    names = l2.FEAT_NAMES + l2.extra_names({k: 0 for k in ex_keys})
    X, T, G = np.concatenate(Xs), np.concatenate(Ts), np.concatenate(Gs)
    del Xs, Ts
    print(f"{tag} scorer rows {len(T)} positive rate {T.mean():.4f}", flush=True)
    n_oof = len(d["y"])
    succ_o, score_o = np.full(n_oof, -1, np.int64), np.full(n_oof, -50.0, np.float32)
    for g in range(FOLDS):
        tr = G != g
        model = lgb.train(params, lgb.Dataset(X[tr], T[tr]), num_boost_round=rounds)
        for s, (ii, cand, valid, X_all) in per_sbj.items():
            if fold_of[s] != g:
                continue
            p_ = np.clip(model.predict(X_all, num_threads=threads), 1e-6, 1 - 1e-6)
            L = np.full(cand.shape, -50.0, np.float32)
            L[valid] = np.log(p_ / (1 - p_))
            s_loc, sc = l2.lk_assign(cand, L)
            succ_o[ii] = np.where(s_loc >= 0, ii[np.maximum(s_loc, 0)], -1)
            score_o[ii] = sc
        print(f"{tag} OOF fold {g} done [{time.time() - T_START:.0f}s]", flush=True)
        del model
    del per_sbj
    model = lgb.train(params, lgb.Dataset(X, T), num_boost_round=rounds)
    imp = model.feature_importance("gain")
    top = np.argsort(-imp)[:20]
    print(f"{tag} top features by gain:", [(names[i], round(float(imp[i] / imp.sum()), 3)) for i in top], flush=True)
    del X, T
    ref = quantile_ref(score_o, succ_o)
    d[f"{tag}_succ"], d[f"{tag}_score"] = succ_o, score_o
    d[f"{tag}_score_qn"] = l2.qnorm(succ_o, score_o, d["sbj"], ref)
    _tv, _ta = t["vid"](), t["acc"]()
    NT = len(t["sbj"])
    succ_t, score_t = np.full(NT, -1, np.int64), np.full(NT, -50.0, np.float32)
    for s in np.unique(t["sbj"]):
        ii = np.flatnonzero(t["sbj"] == s)
        ex = nl_extras(nls, ii, None, kr, kc)
        cand, valid, base, new, ext = l2.subject_all(_tv[ii], _ta[ii], t["sensor"][ii], t["logp"][ii], t["plp"][ii],
                                                     RIDGES, extra=ex)
        b = base.astype(np.float32)
        acc_ = np.zeros(valid.sum(), np.float64)
        for v in range(len(RIDGES)):
            p_ = np.clip(model.predict(np.concatenate([b, new[v], ext], 1), num_threads=threads), 1e-6, 1 - 1e-6)
            acc_ += np.log(p_ / (1 - p_))
        L = np.full(cand.shape, -50.0, np.float32)
        L[valid] = acc_ / len(RIDGES)
        s_loc, sc = l2.lk_assign(cand, L)
        succ_t[ii] = np.where(s_loc >= 0, ii[np.maximum(s_loc, 0)], -1)
        score_t[ii] = sc
        m = succ_t[ii] >= 0
        print(f"{tag} test sbj {s} n={len(ii)} K={cand.shape[1]} linked {m.mean():.3f} median score {np.median(sc[m]):.2f}",
              flush=True)
        del cand, valid, base, new, b, ext, ex
    t[f"{tag}_succ"], t[f"{tag}_score"] = succ_t, score_t
    t[f"{tag}_score_qn"] = l2.qnorm(succ_t, score_t, t["sbj"], ref)
    return ref


def main(cfg):
    from .common import Paths
    from . import data, graph, l0
    from .base import write_sub
    P = Paths()
    threads = int(cfg.get("threads", 4))
    recs, sim, d, t = data.build(P)
    fold_of = data.fold_split(d["sbj"])
    for name, dd in (("oof", d), ("test", t)):
        z = load_npz(find_art(f"{name}_B.npz"))
        for k in ("logp", "fusion_logp", "imu_logp", "emb", "plp", "logp_b", "l2_succ", "l2_score", "l2_score_qn",
                  "P_graph"):
            dd[k] = z[k].astype(np.float32) if k == "emb" else z[k]
    tags = cfg.get("nl_tags") or [cfg.get("nl_tag", "v1")]
    nls = {tg: load_npz(find_art(f"nlink_{tg}.npz")) for tg in tags}
    nl = nls[tags[0]]                                  # used by the optional kNN-embedding experiment
    nl_arg = nls if len(nls) > 1 else nl
    report = {"cfg": cfg}
    report["L2"] = l0.link_report("L2 OOF", d, d["l2_succ"])
    with stage("L3 links"):
        ref = run(P, recs, d, t, nl_arg, fold_of, threads, kr=cfg.get("kr", 20), kc=cfg.get("kc", 10))
        report["L3"] = l0.link_report("L3 OOF", d, d["L3_succ"])
    with stage("graph with L3 links"):
        dd = dict(logp=d["logp_b"], emb=d["emb"], grp=d["sbj"], sbj=d["sbj"], succ=d["L3_succ"],
                  score=d["L3_score_qn"], sets=C.TRAIN_SETS)
        d["P_graph3"] = graph.graph_P(dd)
        lab = C.finish(d["P_graph3"], dd)
        report["graph_L3_oof_f1"] = macro_f1(d["y"], lab)
        base_lab = C.finish(d["P_graph"], dict(dd, succ=d["l2_succ"], score=d["l2_score_qn"]))
        report["graph_L2_oof_f1"] = macro_f1(d["y"], base_lab)
        print(f"OOF macro F1: graph with L2 {report['graph_L2_oof_f1']:.4f} -> with L3 {report['graph_L3_oof_f1']:.4f}",
              flush=True)
        report["class_f1_L3"] = np.round(class_f1(d["y"], lab), 4).tolist()
        dt = dict(logp=t["logp_b"], emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=t["L3_succ"],
                  score=t["L3_score_qn"], sets={})
        t["P_graph3"] = graph.graph_P(dt)
        write_sub(P, t, C.finish(t["P_graph3"], dt), "submission_L3.csv")
    if cfg.get("nl_emb_graph", True):
        with stage("graph with nlink embeddings in the kNN"):
            def unit(x):
                x = x.astype(np.float32)
                return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
            for mode in ("cat", "nl"):
                eo = np.concatenate([unit(nl["oof_t"]), unit(nl["oof_h"])], 1)
                et = np.concatenate([unit(nl["test_t"][-1]), unit(nl["test_h"][-1])], 1)
                if mode == "cat":
                    eo = np.concatenate([unit(d["emb"]), eo / np.sqrt(2)], 1)
                    et = np.concatenate([unit(t["emb"]), et / np.sqrt(2)], 1)
                de = dict(dd, emb=eo)
                Pe = graph.graph_P(de)
                f = macro_f1(d["y"], C.finish(Pe, de))
                report[f"graph_L3_emb_{mode}_oof_f1"] = f
                print(f"graph(L3) with kNN embedding '{mode}': OOF macro F1 {f:.4f}", flush=True)
                dte = dict(dt, emb=et)
                Pt = graph.graph_P(dte)
                write_sub(P, t, C.finish(Pt, dte), f"submission_L3_emb_{mode}.csv")
                np.savez_compressed(P.ART / f"test_L3_emb_{mode}.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
                np.savez_compressed(P.ART / f"oof_L3_emb_{mode}.npz", P_graph=Pe, y=d["y"], sbj=d["sbj"])
    keys = ["L3_succ", "L3_score", "L3_score_qn", "P_graph3"]
    np.savez_compressed(P.ART / "oof_L3.npz", **{k: d[k] for k in keys}, y=d["y"], sbj=d["sbj"])
    np.savez_compressed(P.ART / "test_L3.npz", **{k: t[k] for k in keys}, id=t["id"], sbj=t["sbj"], qn_ref=ref)
    C.save_json(P.ART / "l3_report.json", report)
    if cfg.get("cs_on_l2", False):
        from . import chainstack
        with stage("chain stacker (multi-limb) on L2 links + graph(L2)"):
            cs_o, cs_t = chainstack.run(d, t, np.exp(d["logp_b"]), np.exp(t["logp_b"]), d["l2_succ"],
                                        d["l2_score_qn"], t["l2_succ"], t["l2_score_qn"], threads=threads,
                                        multilimb=True)
            mo = C.log_softmax(cs_o)
            de = dict(dd, logp=mo, succ=d["l2_succ"], score=d["l2_score_qn"])
            Pe = graph.graph_P(de)
            f = macro_f1(d["y"], C.finish(Pe, de))
            report["cs_ml_w1.0_graph_L2_oof_f1"] = f
            print(f"chain stacker (multi-limb) w=1.0 + graph(L2): OOF macro F1 {f:.4f}", flush=True)
            dt2 = dict(dt, logp=C.log_softmax(cs_t), succ=t["l2_succ"], score=t["l2_score_qn"])
            Pt = graph.graph_P(dt2)
            write_sub(P, t, C.finish(Pt, dt2), "submission_L2_csml_w1.0.csv")
            np.savez_compressed(P.ART / "test_L2_csml_w1.0.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
            np.savez_compressed(P.ART / "oof_L2_csml_w1.0.npz", P_graph=Pe, y=d["y"], sbj=d["sbj"])
    if cfg.get("chainstack", True):
        from . import chainstack
        with stage("chain stacker on L3 links + graph"):
            cs_o, cs_t = chainstack.run(d, t, np.exp(d["logp_b"]), np.exp(t["logp_b"]), d["L3_succ"],
                                        d["L3_score_qn"], t["L3_succ"], t["L3_score_qn"], threads=threads,
                                        multilimb=cfg.get("multilimb", True))
            for w in cfg.get("cs_weights", [1.0, 0.5]):
                mo = C.log_softmax(w * cs_o + (1 - w) * d["logp_b"])
                mt = C.log_softmax(w * cs_t + (1 - w) * t["logp_b"])
                de = dict(dd, logp=mo)
                Pe = graph.graph_P(de)
                f = macro_f1(d["y"], C.finish(Pe, de))
                report[f"cs_w{w}_graph_L3_oof_f1"] = f
                print(f"chain stacker w={w} + graph(L3): OOF macro F1 {f:.4f}", flush=True)
                dt2 = dict(dt, logp=mt)
                Pt = graph.graph_P(dt2)
                write_sub(P, t, C.finish(Pt, dt2), f"submission_L3_cs_w{w}.csv")
                np.savez_compressed(P.ART / f"test_L3_cs_w{w}.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
                np.savez_compressed(P.ART / f"oof_L3_cs_w{w}.npz", P_graph=Pe, y=d["y"], sbj=d["sbj"])
        C.save_json(P.ART / "l3_report.json", report)
