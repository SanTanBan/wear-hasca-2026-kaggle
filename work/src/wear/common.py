"""Shared constants, paths, timing and the context helpers (label propagation, link smoothing, count calibration).

Large parts of this package are adapted from the public notebook "WEAR@HASCA 2026 | Timeline Reconstruction + Graph"
by Briano (woominyo), which itself credits honghanhhh for the first link stage (both Apache 2.0).
"""
import gc
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_CLS, WIN, VLEN = 19, 50, 15
FOLDS = 5
SLUG = "3rd-wear-dataset-challenge-hasca-2026"
ON_KAGGLE = Path("/kaggle/input").exists()
TRAIN_SETS = {0: 2, 14: 2}   # train subjects 0 and 14 recorded the exercise circuit twice
CFG = {"imu_w": 0.3, "lp": (10, 0.1, 0.5, 0.9), "link_alpha": 0.5, "link_iters": 10, "sharpen_T": 0.5, "per_ex": 97,
       "null_min": 0.05}
CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)",
           "jogging (butt-kicks)", "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)",
           "stretching (hamstrings)", "stretching (lumbar rotation)", "push-ups", "push-ups (complex)",
           "sit-ups", "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LABEL_IDX = {c: i for i, c in enumerate(CLASSES)}
SENSORS = ["right_arm", "right_leg", "left_leg", "left_arm"]
ACC_COLS = [f"{s}_acc_{a}" for s in SENSORS for a in "xyz"]

T_START = time.time()
TIMES = {}


def find_input():
    if os.environ.get("WEAR_DATA"):
        return Path(os.environ["WEAR_DATA"])
    for p in (Path("/kaggle/input/competitions") / SLUG, Path("/kaggle/input") / SLUG):
        if (p / "test" / "test_meta_data.csv").exists():
            return p
    hits = sorted(Path("/kaggle/input").glob("*/test/test_meta_data.csv")) + \
        sorted(Path("/kaggle/input").glob("*/*/test/test_meta_data.csv"))
    assert hits, "competition data not found"
    return hits[0].parent.parent


def find_file(name, must_contain=None):
    """first file under /kaggle/input with this name (optionally with a path fragment)"""
    roots = [Path("/kaggle/input")] if ON_KAGGLE else [Path(os.environ.get("WEAR_INPUTS", "."))]
    for r in roots:
        hits = sorted(p for p in r.rglob(name) if must_contain is None or must_contain in str(p))
        if hits:
            return hits[0]
    return None


class Paths:
    def __init__(self):
        self.INPUT = find_input()
        self.WORK = Path(os.environ.get("WEAR_WORK", "/kaggle/working" if ON_KAGGLE else "wear_run")).resolve()
        self.WORK.mkdir(parents=True, exist_ok=True)
        if os.environ.get("WEAR_ROOT"):
            self.ROOT = Path(os.environ["WEAR_ROOT"])
        elif ON_KAGGLE and shutil.disk_usage("/tmp").free > 12e9:
            self.ROOT = Path("/tmp/wear")
        else:
            self.ROOT = self.WORK / "_wear"
        self.PROC = self.ROOT / "proc"
        self.ART = self.WORK / "art"
        for p in (self.ROOT, self.PROC, self.ART):
            p.mkdir(parents=True, exist_ok=True)


def rss_gb():
    try:
        s = Path("/proc/self/status").read_text()
        return tuple(int(re.search(k + r":\s+(\d+)", s).group(1)) / 2 ** 20 for k in ("VmRSS", "VmHWM"))
    except Exception:
        return float("nan"), float("nan")


class stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print(f"\n===== {self.name} | start at {(self.t - T_START) / 60:.1f} min", flush=True)
        return self

    def __exit__(self, *exc):
        gc.collect()
        TIMES[self.name] = time.time() - self.t
        r, h = rss_gb()
        print(f"===== {self.name} | {TIMES[self.name] / 60:.1f} min (total {(time.time() - T_START) / 60:.1f} min) "
              f"| RSS {r:.1f} GB, peak {h:.1f} GB", flush=True)


def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def class_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)


def mean_logp(logps):
    return np.log(np.mean([np.exp(x) for x in logps], 0) + 1e-8)


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def blend(fus, imu, wi):
    return log_softmax(fus + wi * imu if imu is not None else fus)


def norm_rows(P):
    P = np.clip(P, 1e-12, None)
    return P / P.sum(1, keepdims=True)


def label_prop(P0, emb, grp, k=10, tau=0.1, use_p=0.5, alpha=0.9, iters=20):
    """per subject: kNN graph on (video similarity + prediction similarity), then probability propagation"""
    E = emb.astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-6
    F = np.empty_like(P0)
    for g in np.unique(grp):
        ii = np.flatnonzero(grp == g)
        n = len(ii)
        Q = np.sqrt(P0[ii]).astype(np.float32)
        S = E[ii] @ E[ii].T + use_p * (Q @ Q.T)
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        vals = S[rows, nb.reshape(-1)]
        W = sp.csr_matrix((np.exp((vals - vals.max()) / tau), (rows, nb.reshape(-1))), shape=(n, n))
        W = W.maximum(W.T)
        d_ = np.asarray(W.sum(1)).ravel()
        d_[d_ == 0] = 1
        Dm = sp.diags(1 / np.sqrt(d_))
        Wn = Dm @ W @ Dm
        f0 = P0[ii]
        f = f0.copy()
        for _ in range(iters):
            f = alpha * (Wn @ f) + (1 - alpha) * f0
        F[ii] = f
    return F


def link_smooth(P, succ, score, alpha, iters):
    m = succ >= 0
    src, dst = np.flatnonzero(m), succ[m]
    wt = 1 / (1 + np.exp(-np.clip(score[m], -20, 20)))
    W = sp.csr_matrix((np.r_[wt, wt], (np.r_[src, dst], np.r_[dst, src])), shape=(len(P), len(P)))
    d_ = np.asarray(W.sum(1)).ravel()
    cur = P.copy()
    for _ in range(iters):
        nb = (W @ cur) / np.maximum(d_[:, None], 1e-8)
        mix = np.where(d_[:, None] > 0, (1 - alpha) * P + alpha * nb, P)
        cur = mix / mix.sum(1, keepdims=True)
    return cur


def calibrate(P, sbj, sets, per_ex, null_min, iters=50):
    """Sinkhorn per subject: each exercise column sums to per_ex x sets, null takes the rest"""
    out = np.empty_like(P)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        n = len(ii)
        null = max(n - 18 * per_ex * sets.get(int(s), 1), null_min * n)
        t_ = np.full(P.shape[1], (n - null) / 18)
        t_[0] = null
        Q = P[ii].copy()
        for _ in range(iters):
            Q *= (t_ / np.maximum(Q.sum(0), 1e-9))[None]
            Q /= Q.sum(1, keepdims=True)
        out[ii] = Q
    return out


def finish(P, d, T=None, per_ex=None):
    Q = np.clip(P, 1e-12, None) ** (1 / (T or CFG["sharpen_T"]))
    Q = calibrate(Q / Q.sum(1, keepdims=True), d["sbj"], d["sets"], per_ex or CFG["per_ex"], CFG["null_min"])
    return Q.argmax(1)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
