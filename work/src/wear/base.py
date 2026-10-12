"""Base pipeline with honest out-of-fold estimates for every stage.

window logp (pretrained 5-fold models) -> L0 links -> tabular experts (nested OOF) -> L2 links (fold scorers)
-> graph propagation + count calibration. Saves every intermediate array to /kaggle/working/art so later kernels can
build on it without recomputation.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .common import Paths, macro_f1, class_f1, stage, label_prop, norm_rows


def plp_of(logp, emb, grp):
    P = label_prop(np.exp(logp), emb, grp, *C.CFG["lp"])
    return (P / P.sum(1, keepdims=True)).astype(np.float32)


KEYS = ["logp", "fusion_logp", "imu_logp", "emb", "l0_succ", "l0_score", "plp", "tab_S3", "tab_T", "logp_b",
        "l2_succ", "l2_score", "l2_score_qn", "P_graph"]


def save(P, d, t, tag):
    od = {k: d[k] for k in ["y", "sbj", "fold", "sensor", "rec", "start", "true_succ"] + KEYS if k in d}
    ot = {k: t[k] for k in ["id", "sbj", "sensor"] + KEYS if k in t}
    for o in (od, ot):
        if "emb" in o:
            o["emb"] = o["emb"].astype(np.float16)
    np.savez_compressed(P.ART / f"oof_{tag}.npz", **od)
    np.savez_compressed(P.ART / f"test_{tag}.npz", **ot)
    print(f"saved artifacts {tag}: oof {sorted(od)} test {sorted(ot)}", flush=True)


def load(P, tag, d, t, src=None):
    root = Path(src) if src else P.ART
    for name, dd in (("oof", d), ("test", t)):
        z = np.load(root / f"{name}_{tag}.npz", allow_pickle=True)
        for k in z.files:
            if k in KEYS:
                dd[k] = z[k].astype(np.float32) if k == "emb" else z[k]
    print(f"loaded artifacts {tag} from {root}", flush=True)


def write_sub(P, t, labels, name="submission.csv"):
    sample = pd.read_csv(P.INPUT / "sample_submission.csv")
    id_col, target_col = sample.columns[:2]
    sub = sample[[id_col]].merge(pd.DataFrame({id_col: t["id"], target_col: labels}), on=id_col, how="left")
    assert len(sub) == len(sample) and sub[target_col].notna().all()
    sub[target_col] = sub[target_col].astype(int)
    sub.to_csv(P.WORK / name, index=False)
    print(f"wrote {name}: counts {np.bincount(labels, minlength=C.N_CLS).tolist()}", flush=True)


def main(cfg):
    P = Paths()
    smoke = bool(cfg.get("smoke", False))
    threads = int(cfg.get("threads", os.cpu_count() or 4))
    report = {"cfg": cfg}
    from . import data
    recs, sim, d, t = data.build(P, smoke=smoke)
    fold_of = data.fold_split(d["sbj"])
    qn_ref = json.loads((Path(__file__).parent / "qn_ref.json").read_text())

    src = cfg.get("load_from")   # e.g. a previous kernel's art dir with stage A
    if src and Path(src).exists() and (Path(src) / "oof_A.npz").exists():
        load(P, "A", d, t, src)
    else:
        from . import window_pre, l0, tab
        with stage("W window inference"):
            runs_oof, runs_test = window_pre.run(P, recs, d, t)
            extra = cfg.get("extra_window")        # e.g. ["wtrain_g1.npz"] from a wtrain kernel output
            if extra:
                for name in extra:
                    hits = sorted(Path("/kaggle/input").rglob(name))
                    assert hits, name
                    z = np.load(hits[0])
                    for k in z.files:
                        if k.startswith("oof_"):
                            runs_oof[k[4:]] = z[k]
                            runs_test[k[4:]] = z["test_" + k[4:]]
                            print(f"extra window run {k[4:]}: OOF macro F1 {macro_f1(d['y'], z[k].argmax(1)):.4f}",
                                  flush=True)
                fus = [r for r in runs_oof if not r.startswith("imu")]
                imu = [r for r in runs_oof if r.startswith("imu")]
                d["fusion_logp"] = C.mean_logp([runs_oof[r] for r in fus])
                d["imu_logp"] = C.mean_logp([runs_oof[r] for r in imu])
                d["logp"] = C.blend(d["fusion_logp"], d["imu_logp"], C.CFG["imu_w"])
                t["fusion_logp"] = C.mean_logp([runs_test[r] for r in fus])
                t["imu_logp"] = C.mean_logp([runs_test[r] for r in imu])
                t["logp"] = C.blend(t["fusion_logp"], t["imu_logp"], C.CFG["imu_w"])
                print(f"window ensemble with {len(fus)} fusion + {len(imu)} IMU runs: fusion "
                      f"{macro_f1(d['y'], d['fusion_logp'].argmax(1)):.4f} imu {macro_f1(d['y'], d['imu_logp'].argmax(1)):.4f} "
                      f"blend {macro_f1(d['y'], d['logp'].argmax(1)):.4f}", flush=True)
            report["window_oof_f1"] = macro_f1(d["y"], d["logp"].argmax(1))
        with stage("L0 links"):
            report["l0"] = l0.run(P, recs, d, t, rounds=20 if smoke else 200)
        with stage("Plp"):
            d["plp"] = plp_of(d["logp"], d["emb"], d["sbj"])
            t["plp"] = plp_of(t["logp"], t["emb"], t["sbj"])
        with stage("tabular experts (nested OOF)"):
            if smoke:
                from . import tab as _t
                _t.TAB_ROUNDS.update(S3=5, T=5)
            tab.run(d, t, nested=True, threads=threads)
            report["tab_blend_oof_f1"] = macro_f1(d["y"], d["logp_b"].argmax(1))
        save(P, d, t, "A")
        C.save_json(P.ART / "report.json", report)

    if cfg.get("stop_after_A"):
        return
    from . import l2, graph, l0
    with stage("L2 links (OOF fold scorers + test)"):
        l2.run(P, recs, d, t, d["plp"], t["plp"], qn_ref, fold_of, threads=threads, rounds=20 if smoke else l2.LK_ROUNDS)
        report["l2"] = l0.link_report("L2 OOF", d, d["l2_succ"])
    with stage("graph + calibration"):
        dd = dict(logp=d["logp_b"], emb=d["emb"], grp=d["sbj"], sbj=d["sbj"], succ=d["l2_succ"],
                  score=d["l2_score_qn"], sets=C.TRAIN_SETS)
        d["P_graph"] = graph.graph_P(dd)
        lab = C.finish(d["P_graph"], dd)
        report["graph_oof_f1"] = macro_f1(d["y"], lab)
        report["graph_oof_class_f1"] = np.round(class_f1(d["y"], lab), 4).tolist()
        print(f"OOF macro F1 after graph + calibration: {report['graph_oof_f1']:.4f}  (Briano nested: 0.9053)", flush=True)
        dt = dict(logp=t["logp_b"], emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=t["l2_succ"], score=t["l2_score_qn"],
                  sets={})
        t["P_graph"] = graph.graph_P(dt)
        tl = C.finish(t["P_graph"], dt)
        write_sub(P, t, tl)
        report["test_null_share"] = {int(s): round(float((tl[t["sbj"] == s] == 0).mean()), 3) for s in np.unique(t["sbj"])}
    save(P, d, t, "B")
    C.save_json(P.ART / "report.json", report)
    print("stage times (min):", {k: round(v / 60, 1) for k, v in C.TIMES.items()}, flush=True)
