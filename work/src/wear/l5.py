"""L5 (ours): the L3 link scorer (L2 features + neural successor features of several nlink models) whose prediction-
similarity features use our best blended probabilities instead of the window log-probs / label propagation, so links
prefer same-activity successors. Then the v2 decoder: multi-limb chain stacker mixed into the tile log-probs and the
swept graph configs (gsweep top-5). Every probability set is saved for blending."""
import json
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import macro_f1, norm_rows, stage


def find(name, must=None):
    hits = sorted(p for p in Path("/kaggle/input").rglob(name) if must is None or must in str(p))
    assert hits, (name, must)
    return hits[0]


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main(cfg):
    from .common import Paths
    from .base import write_sub
    from . import data, chainstack, l0, l3
    from .gsweep import finish_soft, run_cfg
    P = Paths()
    threads = int(cfg.get("threads", 4))
    recs, sim, d, t = data.build(P)
    fold_of = data.fold_split(d["sbj"])
    a, b = load_npz(find("oof_B.npz")), load_npz(find("test_B.npz"))
    for k in ("logp", "fusion_logp", "imu_logp", "emb", "plp", "logp_b", "tab_S3", "tab_T", "l2_succ", "l2_score_qn"):
        d[k] = a[k].astype(np.float32) if k == "emb" else a[k]
        t[k] = b[k].astype(np.float32) if k == "emb" else b[k]
    tag = cfg.get("tag", "L5")
    res = {"cfg": cfg}
    # probabilities used inside the link features
    src = cfg.get("p_link")
    if src:
        po = load_npz(find(f"oof_{src}.npz"))["P_graph"].astype(np.float32)
        pt = load_npz(find(f"test_{src}.npz"))["P_graph"].astype(np.float32)
        keep = {k: (d[k], t[k]) for k in ("logp", "plp")}
        d["logp"], d["plp"] = np.log(np.clip(po, 1e-7, 1)), po
        t["logp"], t["plp"] = np.log(np.clip(pt, 1e-7, 1)), pt
        print(f"link features use '{src}' probabilities (OOF F1 {macro_f1(d['y'], po.argmax(1)):.4f})", flush=True)
    nls = {tg: load_npz(find(f"nlink_{tg}.npz")) for tg in cfg.get("nl_tags", ["v2", "v3"])}
    with stage(f"{tag} links"):
        l3.run(P, recs, d, t, nls if len(nls) > 1 else list(nls.values())[0], fold_of, threads,
               kr=cfg.get("kr", 20), kc=cfg.get("kc", 10), tag=tag, soft=float(cfg.get("soft", 0.0)),
               soft_win=int(cfg.get("soft_win", 10)))
        res["links"] = l0.link_report(f"{tag} OOF", d, d[f"{tag}_succ"])
    if src:
        for k, (vo, vt) in keep.items():
            d[k], t[k] = vo, vt
    so, sco, st, sct = d[f"{tag}_succ"], d[f"{tag}_score_qn"], t[f"{tag}_succ"], t[f"{tag}_score_qn"]
    np.savez_compressed(P.ART / f"links_{tag}.npz", oof_succ=so, oof_score_qn=sco, test_succ=st, test_score_qn=sct)
    top = json.loads(find("gsweep.json").read_text())["top"]
    y, sbj = d["y"], d["sbj"]

    def graph_ens(logp_o, logp_t, name):
        do = dict(logp=logp_o, emb=d["emb"], grp=sbj, sbj=sbj, succ=so, score=sco, sets=C.TRAIN_SETS)
        dt = dict(logp=logp_t, emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=st, score=sct, sets={})
        cache = {}
        Po = norm_rows(np.mean([run_cfg(do, c, cache) for c in top], 0))
        f = macro_f1(y, finish_soft(Po, do))
        cache = {}
        Pt = norm_rows(np.mean([run_cfg(dt, c, cache) for c in top], 0))
        np.savez_compressed(P.ART / f"oof_{name}.npz", P_graph=Po, y=y, sbj=sbj)
        np.savez_compressed(P.ART / f"test_{name}.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
        res[name] = f
        print(f"{name}: OOF soft {f:.4f}", flush=True)
        return f, Pt, dt

    with stage("graph(gsweep) on the new links"):
        graph_ens(d["logp_b"], t["logp_b"], f"{tag}_g")
    with stage("multi-limb chain stacker + graph"):
        cs_o, cs_t = chainstack.run(d, t, np.exp(d["logp_b"]), np.exp(t["logp_b"]), so, sco, st, sct, threads=threads,
                                    multilimb=True, ml2=bool(cfg.get("ml2", False)))
        best = (0, None)
        for w in cfg.get("cs_weights", [0.3, 0.4, 0.5]):
            f, Pt, dt = graph_ens(C.log_softmax(w * cs_o + (1 - w) * d["logp_b"]),
                                  C.log_softmax(w * cs_t + (1 - w) * t["logp_b"]), f"{tag}_cs{w}")
            if f > best[0]:
                best = (f, w)
                write_sub(P, t, finish_soft(Pt, dt), f"submission_{tag}_cs.csv")
        res["best_cs"] = best
    C.save_json(P.ART / f"{tag}.json", res)
