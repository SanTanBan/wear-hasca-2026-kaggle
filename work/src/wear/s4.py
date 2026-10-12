"""S4 (ours): Briano's S3 tabular stacker with (a) the neighbour block taken from our best links (L4: L2 features +
two nlink models) and (b) multi-limb neighbour IMU aggregates (chain + video-kNN). Nested OOF, tab blend with the
window log-probs and T, then graph propagation (gsweep top-5 configs) on the same links. Saved for blending."""
import json
from pathlib import Path

import numpy as np

from . import common as C
from .common import FOLDS, N_CLS, T_START, macro_f1, norm_rows, stage


def find(name, must=None):
    hits = sorted(p for p in Path("/kaggle/input").rglob(name) if must is None or must in str(p))
    assert hits, (name, must)
    return hits[0]


def main(cfg):
    import lightgbm as lgb
    from .common import Paths
    from .base import write_sub
    from .decode import load_art
    from . import data, tab, chainstack
    from .gsweep import finish_soft, run_cfg
    P = Paths()
    threads = int(cfg.get("threads", 4))
    recs, sim, d, t = data.build(P)
    a, b = load_art(find("oof_B.npz")), load_art(find("test_B.npz"))
    for k in ("logp", "fusion_logp", "imu_logp", "emb", "logp_b", "tab_T"):
        d[k] = a[k].astype(np.float32) if k == "emb" else a[k]
        t[k] = b[k].astype(np.float32) if k == "emb" else b[k]
    if cfg.get("links_npz"):                       # links_<tag>.npz written by l5.py
        z = load_art(find(f"{cfg['links_npz']}.npz", cfg.get("links_from")))
        so, sco, st, sct = z["oof_succ"], z["oof_score_qn"], z["test_succ"], z["test_score_qn"]
    elif cfg.get("links") == "L2":                 # Briano's L2 links from the base artifacts
        so, sco, st, sct = a["l2_succ"], a["l2_score_qn"], b["l2_succ"], b["l2_score_qn"]
    else:
        lo, lt = load_art(find("oof_L3.npz", cfg.get("links_from", "wear26-l4"))), \
            load_art(find("test_L3.npz", cfg.get("links_from", "wear26-l4")))
        so, sco, st, sct = lo["L3_succ"], lo["L3_score_qn"], lt["L3_succ"], lt["L3_score_qn"]
    d["l0_succ"], d["l0_score"], t["l0_succ"], t["l0_score"] = so, sco, st, sct
    with stage("tabular features"):
        feat, names = tab.features(d, t)
    if cfg.get("adapt_T"):
        # pseudo-label adaptation of expert T (idea from the public notebook 'Learned Links + Counts' by goodpjw2008):
        # T is re-fitted with the subject's own tiles labelled by our best blend, cross-fitted halves so no tile is
        # predicted by a model that saw its own pseudo-label. OOF subjects use OOF pseudo-labels (honest).
        ac = cfg["adapt_T"]
        with stage("pseudo-label adapted T"):
            po = load_art(find(f"oof_{ac['pseudo']}.npz"))["P_graph"].astype(np.float64)
            pt = load_art(find(f"test_{ac['pseudo']}.npz"))["P_graph"].astype(np.float64)

            def soft_lab(P_, sb, sets):
                Q = np.clip(P_, 1e-12, None) ** 2.0
                return C.calibrate(Q / Q.sum(1, keepdims=True), sb, sets, 97, 0.05, iters=3).argmax(1)
            pl_o, pl_t = soft_lab(po, d["sbj"], C.TRAIN_SETS), soft_lab(pt, t["sbj"], {})
            print(f"pseudo-label OOF accuracy {np.mean(pl_o == d['y']):.4f}, macro F1 {macro_f1(d['y'], pl_o):.4f}",
                  flush=True)
            XT = tab.tab_build("T", d, feat["oof"], names)
            XTt = tab.tab_build("T", t, feat["test"], names)
            ps_params = dict(tab.TAB_PARAMS, num_threads=threads, min_data_in_leaf=40)
            rounds_ps, wps = int(ac.get("rounds", 150)), float(ac.get("weight", 3.0))

            def adapt(X_lab, y_lab, X_new, pseudo, seed):
                half = np.random.default_rng(seed).integers(0, 2, len(X_new))
                out = np.zeros((len(X_new), N_CLS), np.float32)
                for h in (0, 1):
                    ps_, te_ = half == h, half != h
                    bst = lgb.train(ps_params, lgb.Dataset(np.concatenate([X_lab, X_new[ps_]]),
                                                           np.concatenate([y_lab, pseudo[ps_]]),
                                                           weight=np.concatenate([np.ones(len(y_lab)), np.full(ps_.sum(), wps)]),
                                                           params={"max_bin": 63}), rounds_ps)
                    out[te_] = np.log(np.clip(bst.predict(X_new[te_]), 1e-7, 1))
                return out
            TA = np.zeros((len(d["y"]), N_CLS), np.float32)
            for k in range(FOLDS):
                tr, va = d["fold"] != k, d["fold"] == k
                TA[va] = adapt(XT[tr], d["y"][tr], XT[va], pl_o[va], k)
            TAt = adapt(XT, d["y"], XTt, pl_t, 99)
            print(f"tile-level OOF macro F1: T {macro_f1(d['y'], d['tab_T'].argmax(1)):.4f} -> adapted T "
                  f"{macro_f1(d['y'], TA.argmax(1)):.4f}", flush=True)
            d["tab_T"], t["tab_T"] = TA, TAt
    extra = {}
    if cfg.get("multilimb", True):
        with stage("multi-limb neighbour features"):
            for dd, succ, score, which in ((d, so, sco, "oof"), (t, st, sct, "test")):
                Z = chainstack.imu_matrix(dd["acc"](), dd["sbj"], dd["sensor"])
                Pb = np.exp(dd["logp_b"])
                (ci, cw), (ki, kw) = chainstack.neighbour_lists(succ, score, dd["emb"], dd["sbj"], Pk=Pb,
                                                                use_p=0.5 if cfg.get("ml2") else 0.0)
                parts = [chainstack.multilimb_features(Z, dd["sensor"], ci, cw),
                         chainstack.multilimb_features(Z, dd["sensor"], ki, kw)]
                if cfg.get("ml2"):           # per-limb aggregates of the neighbours' model probabilities
                    for key in ("imu_logp", "fusion_logp"):
                        Pm = np.exp(dd[key]).astype(np.float32)
                        parts += [chainstack.multilimb_features(Pm, dd["sensor"], ci, cw),
                                  chainstack.multilimb_features(Pm, dd["sensor"], ki, kw)]
                if cfg.get("chain_windows"):  # chain-context probabilities (as in the chain stacker)
                    parts.append(chainstack.chain_features(Pb, succ, score, dd["emb"], dd["sbj"])[:, N_CLS:])
                extra[which] = np.concatenate(parts, 1)
    X = tab.tab_build("S3", d, feat["oof"], names)
    Xt = tab.tab_build("S3", t, feat["test"], names)
    if extra:
        X, Xt = np.concatenate([X, extra["oof"]], 1), np.concatenate([Xt, extra["test"]], 1)
    print("S4 features", X.shape, flush=True)
    params = dict(tab.TAB_PARAMS, num_threads=threads, **cfg.get("lgb", {}))
    rounds = int(cfg.get("rounds", 80))
    oof = np.zeros((len(X), N_CLS), np.float32)
    with stage("S4 nested OOF"):
        for f in range(FOLDS):
            tr, va = d["fold"] != f, d["fold"] == f
            bst = lgb.train(params, lgb.Dataset(X[tr], d["y"][tr], params={"max_bin": 63}), rounds)
            oof[va] = np.log(np.clip(bst.predict(X[va]), 1e-7, 1))
        bst = lgb.train(params, lgb.Dataset(X, d["y"], params={"max_bin": 63}), rounds)
        tst = np.log(np.clip(bst.predict(Xt), 1e-7, 1)).astype(np.float32)
    res = {"S4_argmax": macro_f1(d["y"], oof.argmax(1))}
    print(f"S4 nested OOF argmax {res['S4_argmax']:.4f} (S3 0.8142, S3b 0.8216)", flush=True)
    top = json.loads(find("gsweep.json").read_text())["top"]
    best = (0, None)
    for w4 in cfg.get("weights", [0.5, 0.6]):
        lb_o = tab.tab_blend(d["logp"], [(oof, w4), (d["tab_T"], 0.3)])
        lb_t = tab.tab_blend(t["logp"], [(tst, w4), (t["tab_T"], 0.3)])
        do = dict(logp=lb_o, emb=d["emb"], grp=d["sbj"], sbj=d["sbj"], succ=so, score=sco, sets=C.TRAIN_SETS)
        dt = dict(logp=lb_t, emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=st, score=sct, sets={})
        cache = {}
        Po = norm_rows(np.mean([run_cfg(do, c, cache) for c in top], 0))
        f = macro_f1(d["y"], finish_soft(Po, do))
        res[f"graph_w{w4}"] = f
        print(f"tab blend (window, S4@{w4}, T@0.3) + graph(gsweep, L4 links): OOF soft {f:.4f}", flush=True)
        cache = {}
        Pt = norm_rows(np.mean([run_cfg(dt, c, cache) for c in top], 0))
        np.savez_compressed(P.ART / f"oof_{cfg.get('tag', 's4')}_w{w4}.npz", P_graph=Po, y=d["y"], sbj=d["sbj"])
        np.savez_compressed(P.ART / f"test_{cfg.get('tag', 's4')}_w{w4}.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
        if f > best[0]:
            best = (f, w4)
            write_sub(P, t, finish_soft(Pt, dt), f"submission_{cfg.get('tag', 's4')}.csv")
    np.savez_compressed(P.ART / f"{cfg.get('tag', 's4')}_logp.npz", oof=oof, test=tst)
    res["best"] = best
    C.save_json(P.ART / f"{cfg.get('tag', 's4')}.json", res)
