"""Decoding experiments on saved artifacts (ours): oracle links, Viterbi along reconstructed chains, calibration sweeps,
error analysis. Everything is measured on the 22 training subjects' OOF tiles; selected configs also write test CSVs."""
import json
import time
from pathlib import Path

import numpy as np

from . import common as C
from .common import N_CLS, class_f1, macro_f1, norm_rows, calibrate


def chains_from_succ(succ):
    n = len(succ)
    pred = np.full(n, -1)
    m = succ >= 0
    pred[succ[m]] = np.flatnonzero(m)
    seen = np.zeros(n, bool)
    chains = []
    for s in np.flatnonzero(pred < 0):
        c, x = [], s
        while x >= 0 and not seen[x]:
            seen[x] = True
            c.append(x)
            x = succ[x]
        chains.append(np.array(c, np.int64))
    for s in np.flatnonzero(~seen):   # leftover cycles (should not happen after cycle breaking)
        if seen[s]:
            continue
        c, x = [], s
        while x >= 0 and not seen[x]:
            seen[x] = True
            c.append(x)
            x = succ[x]
        chains.append(np.array(c, np.int64))
    return chains


def viterbi(logE, pen):
    """logE (L,C) emission log-scores; pen (L-1,) cost of switching label between k and k+1 -> best path (L,)"""
    L, Cn = logE.shape
    if L == 1:
        return logE.argmax(1)
    back = np.zeros((L, Cn), np.int64)
    cur = logE[0].copy()
    for k in range(1, L):
        best_prev = cur.argmax()
        stay = cur
        switch = cur[best_prev] - pen[k - 1]
        use_switch = switch > stay
        back[k] = np.where(use_switch, best_prev, np.arange(Cn))
        cur = np.where(use_switch, switch, stay) + logE[k]
    path = np.zeros(L, np.int64)
    path[-1] = cur.argmax()
    for k in range(L - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


def chain_viterbi(Q, succ, score, base_pen=4.0, conf_scale=True):
    """per-chain Viterbi on log Q with a switch penalty scaled by the link probability"""
    out = Q.argmax(1).copy()
    logQ = np.log(np.clip(Q, 1e-9, None))
    for ch in chains_from_succ(succ):
        if len(ch) < 2:
            continue
        if conf_scale:
            p = 1 / (1 + np.exp(-np.clip(score[ch[:-1]], -20, 20)))
            pen = base_pen * p
        else:
            pen = np.full(len(ch) - 1, base_pen)
        out[ch] = viterbi(logQ[ch], pen)
    return out


def cal_Q(P, sbj, sets, T=0.5, per_ex=97, null_min=0.05):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, per_ex, null_min)


def class_targets(y, sbj, fold, sets):
    """per held-out fold: mean tiles per exercise per circuit, from the other folds' subjects"""
    out = {}
    for f in np.unique(fold):
        tr = fold != f
        cnt = []
        for s in np.unique(sbj[tr]):
            c = np.bincount(y[sbj == s], minlength=N_CLS).astype(float) / sets.get(int(s), 1)
            cnt.append(c)
        out[int(f)] = np.mean(cnt, 0)
    return out


def calibrate_targets(P, sbj, sets, ex_counts, null_min=0.05, iters=50):
    """Sinkhorn per subject with class-specific exercise targets ex_counts (19,) (index 0 ignored)"""
    out = np.empty_like(P)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        n = len(ii)
        k = sets.get(int(s), 1)
        ex = ex_counts[s][1:] * k if isinstance(ex_counts, dict) else ex_counts[1:] * k
        tot = ex.sum()
        if tot > (1 - null_min) * n:
            ex = ex * (1 - null_min) * n / tot
        t_ = np.r_[n - ex.sum(), ex]
        Q = P[ii].copy()
        for _ in range(iters):
            Q *= (t_ / np.maximum(Q.sum(0), 1e-9))[None]
            Q /= Q.sum(1, keepdims=True)
        out[ii] = Q
    return out


def boundary_mask(d, w=2):
    """tiles within w tiles of a label change in their recording (true timeline)"""
    y, rec, st = d["y"], d["rec"], d["start"]
    o = np.lexsort((st, rec))
    yo, ro = y[o], rec[o]
    ch = np.flatnonzero((yo[1:] != yo[:-1]) & (ro[1:] == ro[:-1]))
    near = np.zeros(len(y), bool)
    for c in ch:
        near[o[max(0, c - w + 1):c + w + 1]] = True
    return near


def report(name, y, lab, extra=None, store=None):
    f = macro_f1(y, lab)
    msg = f"{name:55s} OOF macro F1 {f:.4f}"
    if extra:
        msg += "  " + extra
    print(msg, flush=True)
    if store is not None:
        store[name] = f
    return f


def load_art(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main(cfg):
    from .common import Paths
    from . import graph
    P = Paths()
    src = Path(cfg.get("art", "/kaggle/input/wear26-base/art"))
    if not src.exists():
        hits = sorted(Path("/kaggle/input").rglob("oof_B.npz"))
        src = hits[0].parent
    d, t = load_art(src / "oof_B.npz"), load_art(src / "test_B.npz")
    d["emb"], t["emb"] = d["emb"].astype(np.float32), t["emb"].astype(np.float32)
    y, sbj = d["y"], d["sbj"]
    res = {}
    dd = dict(logp=d["logp_b"], emb=d["emb"], grp=sbj, sbj=sbj, succ=d["l2_succ"], score=d["l2_score_qn"],
              sets=C.TRAIN_SETS)
    Pg = d["P_graph"]
    report("base graph+calibration (saved)", y, C.finish(Pg, dd), store=res)
    cf = class_f1(y, C.finish(Pg, dd))
    print("per-class F1:", {C.CLASSES[i][:14]: round(float(v), 3) for i, v in enumerate(cf)}, flush=True)
    near = boundary_mask(d)
    lab = C.finish(Pg, dd)
    print(f"error rate near label boundaries {(lab[near] != y[near]).mean():.3f} ({near.mean():.3f} of tiles), "
          f"elsewhere {(lab[~near] != y[~near]).mean():.3f}", flush=True)
    per_s = {int(s): round(macro_f1(y[sbj == s], lab[sbj == s]), 3) for s in np.unique(sbj)}
    print("per-subject macro F1:", per_s, flush=True)
    cm = np.bincount(y * N_CLS + lab, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    off = [(int(cm[i, j]), C.CLASSES[i], C.CLASSES[j]) for i in range(N_CLS) for j in range(N_CLS) if i != j]
    print("top confusions (true -> pred):", sorted(off, reverse=True)[:15], flush=True)

    # chain statistics of the L2 links
    for nm, succ in (("L0", d["l0_succ"]), ("L2", d["l2_succ"])):
        chs = chains_from_succ(succ)
        lens = np.array([len(c) for c in chs])
        pure = np.array([len(np.unique(y[c])) == 1 for c in chs])
        print(f"{nm} chains: {len(chs)} chains, median len {np.median(lens):.0f}, mean {lens.mean():.1f}, "
              f"tiles in chains >=20: {lens[lens >= 20].sum() / lens.sum():.3f}, label-pure chains {pure.mean():.3f}, "
              f"tiles in pure chains {lens[pure].sum() / lens.sum():.3f}", flush=True)
    # links: L2 vs oracle (true successor) inside the same graph pipeline
    ts = d["true_succ"]
    for nm, succ, score in (("L2 links", d["l2_succ"], d["l2_score_qn"]),
                            ("ORACLE true links (score 5)", ts, np.where(ts >= 0, 5.0, -50.0).astype(np.float32))):
        de = dict(dd, succ=succ, score=score)
        t0 = time.time()
        Pe = graph.graph_P(de)
        lab_e = C.finish(Pe, de)
        report(f"graph with {nm}", y, lab_e, f"[{time.time() - t0:.0f}s]", res)
        Q = cal_Q(Pe, sbj, C.TRAIN_SETS)
        for pen in (2.0, 4.0, 8.0):
            report(f"  + chain Viterbi pen {pen}", y, chain_viterbi(Q, succ, score, pen), store=res)
    # calibration sweeps on the saved graph probabilities
    for T in (0.35, 0.5, 0.7):
        for pe in (92, 97, 102):
            report(f"calib T={T} per_ex={pe}", y, cal_Q(Pg, sbj, C.TRAIN_SETS, T, pe).argmax(1), store=res)
    # class-specific exercise counts (from the other folds' subjects)
    tg = class_targets(y, sbj, d["fold"], C.TRAIN_SETS)
    print("mean tiles per exercise (fold 0 targets):", np.round(tg[0][1:], 0).tolist(), flush=True)
    allc = np.array([np.bincount(y[sbj == s], minlength=N_CLS) / C.TRAIN_SETS.get(int(s), 1) for s in np.unique(sbj)])
    print("per-class count CV across subjects:", np.round(allc[:, 1:].std(0) / allc[:, 1:].mean(0), 2).tolist(),
          flush=True)
    ex_by_s = {int(s): tg[int(d["fold"][sbj == s][0])] for s in np.unique(sbj)}
    for T in (0.5,):
        Q = np.clip(Pg, 1e-12, None) ** (1 / T)
        Q = Q / Q.sum(1, keepdims=True)
        report(f"calib T={T} class-specific targets", y, calibrate_targets(Q, sbj, C.TRAIN_SETS, ex_by_s).argmax(1),
               store=res)
        for it in (3, 10):
            report(f"calib T={T} per_ex=97 sinkhorn iters={it}", y,
                   calibrate(Q, sbj, C.TRAIN_SETS, 97, 0.05, iters=it).argmax(1), store=res)
    report("no calibration (argmax of graph P)", y, Pg.argmax(1), store=res)

    # chain stacker on L2 links: longer chain context than S3's L0 neighbours
    from . import chainstack
    from .base import write_sub
    for src_name, lo, lt in (("logp_b", d["logp_b"], t["logp_b"]),):
        cs_o, cs_t = chainstack.run(d, t, np.exp(lo), np.exp(lt), d["l2_succ"], d["l2_score_qn"], t["l2_succ"],
                                    t["l2_score_qn"], threads=int(cfg.get("threads", 4)))
        for w in (1.0, 0.5):
            mix_o = C.log_softmax(w * cs_o + (1 - w) * lo)
            mix_t = C.log_softmax(w * cs_t + (1 - w) * lt)
            de = dict(dd, logp=mix_o)
            Pe = graph.graph_P(de)
            lab_e = C.finish(Pe, de)
            f = report(f"chainstack({src_name}) w={w} + graph(L2)", y, lab_e, store=res)
            if cfg.get("write_test", True):
                dt = dict(logp=mix_t, emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=t["l2_succ"],
                          score=t["l2_score_qn"], sets={})
                Pt = graph.graph_P(dt)
                write_sub(P, t, C.finish(Pt, dt), f"submission_cs_w{w}.csv")
                np.savez_compressed(P.ART / f"test_cs_w{w}.npz", P_graph=Pt, id=t["id"], sbj=t["sbj"], oof_f1=f)
                np.savez_compressed(P.ART / f"oof_cs_w{w}.npz", P_graph=Pe, y=y, sbj=sbj)
    C.save_json(P.ART / "decode_report.json", res)
