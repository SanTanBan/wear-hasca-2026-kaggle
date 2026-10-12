"""Raw files -> compact arrays, and test-like 1-s tiles of the training recordings (the OOF rows of every stage)."""
import hashlib
import json

import numpy as np
import pandas as pd

from .common import ACC_COLS, CLASSES, FOLDS, LABEL_IDX, N_CLS, SENSORS, TRAIN_SETS, VLEN, WIN, stage

STRIDE = 5


def prep(P, smoke_subjects=None):
    """per recording: acc (T,12) float32 [g], lab (T,) int8, valid (T,4) bool, vid (F,768) float16; test arrays"""
    done = P.ROOT / ".done_prep"
    if done.exists():
        return json.loads((P.PROC / "recordings.json").read_text())
    recs = []
    csvs = sorted((P.INPUT / "train" / "inertial_feat").glob("*.csv"))   # this order defines the recording index
    if smoke_subjects:
        csvs = [c for c in csvs if int(c.stem.split("_")[1]) in smoke_subjects]
    for csv in csvs:
        stem = csv.stem
        df = pd.read_csv(csv, usecols=["sbj_id", *ACC_COLS, "label"], dtype={"label": str})
        acc = df[ACC_COLS].to_numpy(np.float32)
        valid = ~np.isnan(acc).reshape(-1, 4, 3).any(2)
        if np.isnan(acc).any():
            acc = pd.DataFrame(acc).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)
        labels = df["label"].fillna("null").astype(str).str.strip()
        unknown = set(labels.unique()) - set(CLASSES)
        assert not unknown, f"{stem}: unknown labels {unknown}"
        lab = labels.map(LABEL_IDX).to_numpy(np.int8)
        vid = np.load(P.INPUT / "train" / "videomae_feat" / f"{stem}.npy").astype(np.float16)
        np.save(P.PROC / f"{stem}_acc.npy", acc)
        np.save(P.PROC / f"{stem}_lab.npy", lab)
        np.save(P.PROC / f"{stem}_valid.npy", valid)
        np.save(P.PROC / f"{stem}_vid.npy", vid)
        recs.append({"stem": stem, "sbj": int(df["sbj_id"].iloc[0]), "T": len(acc), "F": len(vid)})
        print(f"{stem}: T={len(acc)} ({len(acc) / 50:.0f}s) F={len(vid)}", flush=True)
        del df, acc, vid
    (P.PROC / "recordings.json").write_text(json.dumps(recs, indent=1))

    meta = pd.read_csv(P.INPUT / "test" / "test_meta_data.csv")
    t_acc = np.load(P.INPUT / "test" / "test_inertial_data.npy").astype(np.float32)
    t_vid = np.load(P.INPUT / "test" / "test_videomae_data.npy")
    if t_vid.shape[1] == 768:
        t_vid = t_vid.transpose(0, 2, 1)
    loc_col = "sensor_location" if "sensor_location" in meta else "inertial_sensor_location"
    np.save(P.PROC / "test_acc.npy", t_acc)
    np.save(P.PROC / "test_vid.npy", np.ascontiguousarray(t_vid).astype(np.float16))
    np.save(P.PROC / "test_sensor.npy", meta[loc_col].map({s: i for i, s in enumerate(SENSORS)}).to_numpy(np.int8))
    np.save(P.PROC / "test_id.npy", meta["id"].to_numpy())
    np.save(P.PROC / "test_sbj.npy", meta["sbj_id"].to_numpy(np.int16))
    del t_vid
    done.touch()
    return recs


def all_windows(P, recs):
    """every 0.1-s window start of every recording with its majority label and per-limb validity"""
    W_rec, W_start, W_y, W_sbj, W_ok = [], [], [], [], []
    for i, r in enumerate(recs):
        lab = np.load(P.PROC / f"{r['stem']}_lab.npy").astype(np.int64)
        val = np.load(P.PROC / f"{r['stem']}_valid.npy")
        T, Fr = r["T"], r["F"]
        starts = np.arange(0, T - WIN + 1, STRIDE)
        starts = starts[starts * 3 // 5 + 8 + VLEN <= Fr]
        cs = np.zeros((T + 1, N_CLS), np.int32)
        np.cumsum(np.eye(N_CLS, dtype=np.int32)[lab], axis=0, out=cs[1:])
        cnt = cs[starts + WIN] - cs[starts]
        miss = np.zeros((T + 1, 4), np.int32)
        np.cumsum(~val, axis=0, out=miss[1:])
        W_ok.append((miss[starts + WIN] - miss[starts]) == 0)
        W_rec.append(np.full(len(starts), i))
        W_start.append(starts)
        W_y.append(cnt.argmax(1))
        W_sbj.append(np.full(len(starts), r["sbj"]))
    W_rec, W_start, W_y, W_sbj, W_ok = map(np.concatenate, (W_rec, W_start, W_y, W_sbj, W_ok))
    keep = W_ok.any(1)
    return tuple(x[keep] for x in (W_rec, W_start, W_y, W_sbj, W_ok))


def fold_split(W_sbj):
    sbjs, sbj_n = np.unique(W_sbj, return_counts=True)
    fold_of, load = {}, np.zeros(FOLDS)
    for s in sbjs[np.argsort(-sbj_n)]:
        f = int(load.argmin())
        fold_of[int(s)] = f
        load[f] += sbj_n[sbjs == s][0]
    return fold_of


def oof_windows(P, recs):
    """test-like tiles (start % 50 == 0) with one random valid limb (fixed RNG per fold), split into 5 subject folds"""
    W_rec, W_start, W_y, W_sbj, W_ok = all_windows(P, recs)
    fold_of = fold_split(W_sbj)
    W_fold = np.array([fold_of[int(s)] for s in W_sbj])
    out = {k: [] for k in ("rec", "start", "sensor", "y", "sbj", "fold")}
    for fold in range(FOLDS):
        idx = np.flatnonzero(W_fold == fold)
        idx = idx[W_start[idx] % WIN == 0]
        rng = np.random.default_rng(1234 + fold)
        cp = np.cumsum(W_ok[idx] / W_ok[idx].sum(1, keepdims=True), 1)
        sensor = (rng.random(len(idx))[:, None] > cp).sum(1)
        for k, v in (("rec", W_rec[idx]), ("start", W_start[idx]), ("sensor", sensor), ("y", W_y[idx]),
                     ("sbj", W_sbj[idx]), ("fold", np.full(len(idx), fold))):
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def load_inputs(P, recs, w):
    """per-window IMU of the chosen limb (N,50,3) float32 and video (N,15,768) float16"""
    n = len(w["rec"])
    acc = np.zeros((n, WIN, 3), np.float32)
    vid = np.zeros((n, VLEN, 768), np.float16)
    for i, r in enumerate(recs):
        ii = np.flatnonzero(w["rec"] == i)
        if not len(ii):
            continue
        a = np.load(P.PROC / f"{r['stem']}_acc.npy").reshape(-1, 4, 3)
        v = np.load(P.PROC / f"{r['stem']}_vid.npy", mmap_mode="r")
        st = w["start"][ii]
        acc[ii] = a[st[:, None] + np.arange(WIN), w["sensor"][ii][:, None]]
        vid[ii] = v[(st * 3 // 5)[:, None] + 8 + np.arange(VLEN)]
    return acc, vid


def true_successor(rec, start):
    """row of the tile that starts 1 s later in the same recording, else -1 (training target / diagnostics only)"""
    key = {(r, s): n for n, (r, s) in enumerate(zip(rec.tolist(), start.tolist()))}
    return np.array([key.get((r, s + 50), -1) for r, s in zip(rec.tolist(), start.tolist())], np.int64)


def build(P, smoke=False):
    """prep + simulated OOF tiles; returns (recs, sim, d, t) where d/t are dicts of OOF / test rows"""
    with stage("W1 preprocessing"):
        recs = prep(P, {0, 1, 2, 3, 4} if smoke else None)
        if not smoke:
            assert len(recs) == 24 and sum(r["T"] for r in recs) == 3466400 and sum(r["F"] for r in recs) == 2079840
    with stage("W2 test-like OOF windows"):
        if not (P.ROOT / ".done_simwin").exists():
            w = oof_windows(P, recs)
            acc, vid = load_inputs(P, recs, w)
            np.save(P.ROOT / "sim_acc.npy", acc)
            np.save(P.ROOT / "sim_vid.npy", vid)
            np.savez(P.ROOT / "sim_meta.npz", **w)
            del acc, vid, w
            (P.ROOT / ".done_simwin").touch()
        sim = {k: v.astype(np.int64) for k, v in dict(np.load(P.ROOT / "sim_meta.npz")).items()}
        if not smoke:
            assert len(sim["y"]) == 69326
            assert np.bincount(sim["fold"]).tolist() == [14353, 12995, 14625, 12768, 14585]
            assert hashlib.sha1(sim["sensor"].tobytes()).hexdigest()[:12] == "f81d90085315"
    d = {"y": sim["y"], "sbj": sim["sbj"], "grp": sim["sbj"], "fold": sim["fold"], "sensor": sim["sensor"],
         "rec": sim["rec"], "start": sim["start"], "sets": TRAIN_SETS,
         "vid": lambda: np.load(P.ROOT / "sim_vid.npy", mmap_mode="r"),
         "acc": lambda: np.load(P.ROOT / "sim_acc.npy")}
    tsbj = np.load(P.PROC / "test_sbj.npy").astype(np.int64)
    t = {"sbj": tsbj, "grp": tsbj, "sensor": np.load(P.PROC / "test_sensor.npy").astype(np.int64), "sets": {},
         "id": np.load(P.PROC / "test_id.npy", allow_pickle=True),
         "vid": lambda: np.load(P.PROC / "test_vid.npy", mmap_mode="r"),
         "acc": lambda: np.load(P.PROC / "test_acc.npy")}
    d["true_succ"] = true_successor(sim["rec"], sim["start"])
    return recs, sim, d, t
