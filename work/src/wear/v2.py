"""v2 decoder stack on L3 links (ours): multi-limb chain stacker (cs) mixed into the tile log-probs, graph propagation
with the swept configs (gsweep top-5), then the post-graph stacker. Saves every probability set for blending."""
import json
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import macro_f1, norm_rows, stage


def load_npz(name, must=None):
    hits = sorted(p for p in Path("/kaggle/input").rglob(name) if must is None or must in str(p))
    assert hits, (name, must)
    z = np.load(hits[0], allow_pickle=True)
    return {k: z[k] for k in z.files}


def main(cfg):
    from .common import Paths
    from .base import write_sub
    from . import data, chainstack, pgs
    from .gsweep import finish_soft, run_cfg
    P = Paths()
    threads = int(cfg.get("threads", 4))
    recs, sim, d, t = data.build(P)
    a, b = load_npz("oof_B.npz"), load_npz("test_B.npz")
    for k in ("logp", "fusion_logp", "imu_logp", "emb", "logp_b", "tab_S3", "tab_T", "l2_succ", "l2_score_qn", "P_graph"):
        d[k] = a[k].astype(np.float32) if k == "emb" else a[k]
        t[k] = b[k].astype(np.float32) if k == "emb" else b[k]
    links = cfg.get("links", "L3")
    if links == "L3":
        l3o = load_npz("oof_L3.npz", cfg.get("links_from"))
        l3t = load_npz("test_L3.npz", cfg.get("links_from"))
        so, sco, st, sct = l3o["L3_succ"], l3o["L3_score_qn"], l3t["L3_succ"], l3t["L3_score_qn"]
    elif links.startswith("links_"):          # links_<tag>.npz written by l5.py
        z = load_npz(f"{links}.npz", cfg.get("links_from"))
        so, sco, st, sct = z["oof_succ"], z["oof_score_qn"], z["test_succ"], z["test_score_qn"]
    else:
        so, sco, st, sct = d["l2_succ"], d["l2_score_qn"], t["l2_succ"], t["l2_score_qn"]
    tag = cfg.get("tag", "v2")
    top = json.loads(sorted(Path("/kaggle/input").rglob("gsweep.json"))[0].read_text())["top"]
    y, sbj = d["y"], d["sbj"]
    res = {}
    with stage("multi-limb chain stacker"):
        Pin_o, Pin_t = np.exp(d["logp_b"]), np.exp(t["logp_b"])
        if cfg.get("cs_input"):                 # second-stage: a saved blend as the chain stacker's input
            Pin_o = load_npz(f"oof_{cfg['cs_input']}.npz")["P_graph"].astype(np.float64)
            Pin_t = load_npz(f"test_{cfg['cs_input']}.npz")["P_graph"].astype(np.float64)
            print(f"chain stacker input: {cfg['cs_input']} (OOF argmax {macro_f1(y, Pin_o.argmax(1)):.4f})", flush=True)
        cs_o, cs_t = chainstack.run(d, t, Pin_o, Pin_t, so, sco, st, sct, threads=threads,
                                    rounds=int(cfg.get("cs_rounds", 150)), multilimb=True,
                                    ml2=bool(cfg.get("ml2", False)), seed=int(cfg.get("seed", 0)))
    best = (0, None)
    for w in cfg.get("cs_weights", [0.4, 0.5, 0.6]):
        mo = C.log_softmax(w * cs_o + (1 - w) * d["logp_b"])
        mt = C.log_softmax(w * cs_t + (1 - w) * t["logp_b"])
        do = dict(logp=mo, emb=d["emb"], grp=sbj, sbj=sbj, succ=so, score=sco, sets=C.TRAIN_SETS)
        dt = dict(logp=mt, emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=st, score=sct, sets={})
        cache = {}
        Po = norm_rows(np.mean([run_cfg(do, c, cache) for c in top], 0))
        f = macro_f1(y, finish_soft(Po, do))
        res[f"cs_w{w}_gsweep"] = f
        print(f"cs w={w} + graph(gsweep top-5, {links}): OOF soft {f:.4f}", flush=True)
        cache = {}
        Pt = norm_rows(np.mean([run_cfg(dt, c, cache) for c in top], 0))
        np.savez_compressed(P.ART / f"oof_{tag}_cs{w}.npz", P_graph=Po, y=y, sbj=sbj)
        np.savez_compressed(P.ART / f"test_{tag}_cs{w}.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
        if f > best[0]:
            best = (f, w, Po, Pt)
    f, w, Po, Pt = best
    write_sub(P, t, finish_soft(Pt, dict(sbj=t["sbj"], sets={})), f"submission_{tag}_cs.csv")
    if cfg.get("pgs", True):
        import lightgbm as lgb
        with stage("post-graph stacker on the v2 graph"):
            act_o = pgs.activity(d["acc"](), sbj, d["sensor"])
            act_t = pgs.activity(t["acc"](), t["sbj"], t["sensor"])
            do = dict(logp_b=d["logp_b"], logp=d["logp"], imu_logp=d["imu_logp"], succ=so, score=sco,
                      emb=d["emb"], sbj=sbj, sets=C.TRAIN_SETS)
            dt = dict(logp_b=t["logp_b"], logp=t["logp"], imu_logp=t["imu_logp"], succ=st, score=sct,
                      emb=t["emb"], sbj=t["sbj"], sets={})
            X = np.concatenate([pgs.features(do, Po, act_o), cs_o], 1)
            Xt = np.concatenate([pgs.features(dt, Pt, act_t), cs_t], 1)
            params = dict(pgs.PGS_PARAMS, num_threads=threads)
            oof = np.zeros((len(X), C.N_CLS))
            for fo in range(C.FOLDS):
                tr, va = d["fold"] != fo, d["fold"] == fo
                bst = lgb.train(params, lgb.Dataset(X[tr], y[tr], params={"max_bin": 63}), 200)
                oof[va] = bst.predict(X[va])
            bst = lgb.train(params, lgb.Dataset(X, y, params={"max_bin": 63}), 200)
            pt = bst.predict(Xt)
            for T in (0.5, 1.0):
                res[f"pgs_T{T}"] = macro_f1(y, pgs.soft_cal(oof, sbj, C.TRAIN_SETS, T, 3).argmax(1))
                print(f"pgs(v2) T={T}: {res[f'pgs_T{T}']:.4f}", flush=True)
            np.savez_compressed(P.ART / f"oof_{tag}_pgs.npz", P_graph=norm_rows(oof), y=y, sbj=sbj)
            np.savez_compressed(P.ART / f"test_{tag}_pgs.npz", P_graph=norm_rows(pt), id=t["id"], sbj=t["sbj"])
    res["best_cs"] = (best[0], best[1])
    C.save_json(P.ART / f"{tag}.json", res)
