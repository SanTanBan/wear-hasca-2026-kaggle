"""Tabular experts (Briano): LightGBM on hand-crafted IMU/video features (T) and a stacker (S3) adding window
log-probs, L0-neighbour and video-kNN blocks. Here with nested out-of-fold predictions for the training rows."""
import time

import lightgbm as lgb
import numpy as np

from .common import FOLDS, N_CLS, T_START, log_softmax, macro_f1

BANDS = [(1, 1), (2, 2), (3, 4), (5, 7), (8, 12), (13, 25)]
K_PCA = 48
TAB_PARAMS = dict(objective="multiclass", num_class=19, learning_rate=0.08, num_leaves=31, min_data_in_leaf=80,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63,
                  num_threads=4, verbose=-1, seed=0)
TAB_ROUNDS = {"S3": 60, "T": 113}
TAB_TEST_CHOICE = (("S3", 0.5), ("T", 0.3))
NB_IMU = ["mean_x", "mean_y", "mean_z", "std_m", "energy_m", "mad_x", "mad_y", "mad_z", "jerk_m", "domf_m",
          "acmax_m", "aclag_m", "logpow_m", "grav_x", "grav_y", "grav_z"]


def imu_feats(acc):
    acc = acc.astype(np.float32)
    mag = np.linalg.norm(acc, axis=2, keepdims=True)
    X = np.concatenate([acc, mag], 2)
    chn = ["x", "y", "z", "m"]
    feats, names = [], []

    def add(v, nm):
        feats.append(v.astype(np.float32))
        names.append(nm)

    mu = X.mean(1)
    Xd = X - mu[:, None]
    sd = X.std(1) + 1e-6
    q = np.percentile(X, [10, 25, 50, 75, 90], axis=1)
    dif = np.diff(X, axis=1)
    F = np.abs(np.fft.rfft(Xd, axis=1)) ** 2
    Ftot = F[:, 1:].sum(1) + 1e-9
    pn = F[:, 1:] / Ftot[:, None]
    ac_full = np.stack([(Xd[:, :-L] * Xd[:, L:]).mean(1) for L in range(3, 41)], 1) / (sd[:, None] ** 2)
    for c in range(4):
        n = chn[c]
        add(mu[:, c], f"mean_{n}")
        add(sd[:, c], f"std_{n}")
        add(X[:, :, c].min(1), f"min_{n}")
        add(X[:, :, c].max(1), f"max_{n}")
        for k, p_ in enumerate([10, 25, 50, 75, 90]):
            add(q[k, :, c], f"p{p_}_{n}")
        add(q[3, :, c] - q[1, :, c], f"iqr_{n}")
        add(X[:, :, c].max(1) - X[:, :, c].min(1), f"range_{n}")
        add((Xd[:, :, c] ** 2).mean(1), f"energy_{n}")
        add(np.abs(Xd[:, :, c]).mean(1), f"mad_{n}")
        z = Xd[:, :, c] / sd[:, c:c + 1]
        add((z ** 3).mean(1), f"skew_{n}")
        add((z ** 4).mean(1), f"kurt_{n}")
        s = np.sign(Xd[:, :, c])
        add((s[:, 1:] * s[:, :-1] < 0).mean(1), f"zcr_{n}")
        add(np.abs(dif[:, :, c]).mean(1), f"jerk_{n}")
        add(dif[:, :, c].std(1), f"jerkstd_{n}")
        add(np.argmax(F[:, 1:, c], 1) + 1.0, f"domf_{n}")
        add(F[:, 1:, c].max(1) / Ftot[:, c], f"domfrac_{n}")
        add(np.log(Ftot[:, c]), f"logpow_{n}")
        for lo, hi in BANDS:
            add(F[:, lo:hi + 1, c].sum(1) / Ftot[:, c], f"band{lo}-{hi}_{n}")
        add(-(pn[:, :, c] * np.log(pn[:, :, c] + 1e-12)).sum(1), f"spent_{n}")
        add(ac_full[:, :, c].max(1), f"acmax_{n}")
        add(np.argmax(ac_full[:, :, c], 1) + 3.0, f"aclag_{n}")
        add(X[:, -10:, c].mean(1) - X[:, :10, c].mean(1), f"trend_{n}")
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        add((Xd[:, :, a] * Xd[:, :, b]).mean(1) / (sd[:, a] * sd[:, b]), f"corr_{chn[a]}{chn[b]}")
    g = mu[:, :3] / (np.linalg.norm(mu[:, :3], axis=1, keepdims=True) + 1e-6)
    for k in range(3):
        add(g[:, k], f"grav_{chn[k]}")
    return np.stack(feats, 1), names


def standardize(X, sbj, sensor):
    Z = np.empty_like(X)
    for s in np.unique(sbj):
        for k in np.unique(sensor):
            ii = np.flatnonzero((sbj == s) & (sensor == k))
            if not len(ii):
                continue
            m, sd = X[ii].mean(0), X[ii].std(0) + 1e-6
            Z[ii] = (X[ii] - m) / sd
    return Z


def video_pass(V, n, chunk=3000):
    vm = np.zeros((n, 768), np.float32)
    mot = np.zeros((n, 4), np.float32)
    for a in range(0, n, chunk):
        v = np.asarray(V[a:a + chunk], np.float32)
        vm[a:a + chunk] = v.mean(1)
        u = v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-6)
        dd = np.linalg.norm(np.diff(v, axis=1), axis=2)
        mot[a:a + chunk, 0] = dd.mean(1)
        mot[a:a + chunk, 1] = (u[:, 0] * u[:, -1]).sum(1)
        mot[a:a + chunk, 2] = (u[:, :-1] * u[:, 1:]).sum(2).min(1)
        mot[a:a + chunk, 3] = v.std(1).mean(1)
    return vm, mot


def hand_features(dd):
    imu, names = imu_feats(dd["acc"]())
    z = standardize(imu, dd["sbj"], dd["sensor"])
    X = np.concatenate([imu, z, dd["sensor"][:, None].astype(np.float32)], 1).astype(np.float32)
    vm, mot = video_pass(dd["vid"](), len(dd["sbj"]))
    return {"imu": X, "vmean": vm, "vmot": mot}, names + [f"z_{n}" for n in names] + ["sensor"]


def sub_center(vm, sbj):
    mu = np.zeros((sbj.max() + 1, vm.shape[1]), np.float32)
    for s in np.unique(sbj):
        mu[s] = vm[sbj == s].mean(0)
    return vm - mu[sbj], mu


def pca(X):
    m = X.mean(0)
    C = np.cov((X - m).T)
    ev, evec = np.linalg.eigh(C)
    o = np.argsort(-ev)[:K_PCA]
    return m.astype(np.float32), evec[:, o].astype(np.float32), ev[o]


def tstd(V, comps, chunk=3000):
    n = V.shape[0]
    out = np.zeros((n, comps.shape[1]), np.float32)
    for a in range(0, n, chunk):
        v = np.asarray(V[a:a + chunk], np.float32)
        out[a:a + chunk] = (v @ comps).std(1)
    return out


def neighbor_block(dd, imu, names):
    succ = dd["l0_succ"]
    n = len(succ)
    prv = np.full(n, -1)
    prv[succ[succ >= 0]] = np.flatnonzero(succ >= 0)
    sc_n = np.where(succ >= 0, dd["l0_score"], np.nan)
    sc_p = np.where(prv >= 0, dd["l0_score"][np.maximum(prv, 0)], np.nan)

    def take(A_, idx):
        out = A_[np.maximum(idx, 0)].astype(np.float32)
        out[idx < 0] = np.nan
        return out

    nxt2 = np.where(succ >= 0, succ[np.maximum(succ, 0)], -1)
    prv2 = np.where(prv >= 0, prv[np.maximum(prv, 0)], -1)
    lp = dd["logp"]
    P = np.exp(lp)
    acc = np.zeros_like(P)
    cnt = np.zeros(n)
    for idx in (prv2, prv, succ, nxt2):
        acc += np.where(idx[:, None] >= 0, P[np.maximum(idx, 0)], 0)
        cnt += idx >= 0
    ctx = np.log(np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), np.nan) + 1e-6)
    zi = [names.index("z_" + k) for k in NB_IMU]
    sens = dd["sensor"].astype(np.float32)
    blocks = [take(lp, succ), take(lp, prv), ctx, sc_n[:, None], sc_p[:, None], cnt[:, None],
              take(sens[:, None], succ), take(sens[:, None], prv), take(imu[:, zi], succ), take(imu[:, zi], prv)]
    return np.concatenate(blocks, 1).astype(np.float32)


def knn_block(dd, k=10):
    E = dd["emb"] / (np.linalg.norm(dd["emb"], axis=1, keepdims=True) + 1e-6)
    P = np.exp(dd["logp"])
    out = np.zeros((len(P), 21), np.float32)
    for s in np.unique(dd["sbj"]):
        ii = np.flatnonzero(dd["sbj"] == s)
        S = E[ii] @ E[ii].T
        np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k, axis=1)[:, :k]
        sims = np.take_along_axis(S, nb, 1)
        out[ii, :19] = np.log(P[ii][nb].mean(1) + 1e-6)
        out[ii, 19], out[ii, 20] = sims.max(1), sims.mean(1)
    return out


def tab_build(variant, dd, feat, names):
    imu = feat["imu"]
    X = [imu, feat["vmot"], feat["vpca"]]
    if variant in ("S1", "S2", "S3"):
        X += [dd["logp"], dd["fusion_logp"], dd["imu_logp"]]
    if variant in ("S2", "S3"):
        X.append(neighbor_block(dd, imu, names))
    if variant == "S3":
        X.append(knn_block(dd))
    return np.concatenate(X, 1).astype(np.float32)


def tab_blend(lp, parts):
    x = (1 - sum(w for _, w in parts)) * lp
    for q, w in parts:
        x = x + w * q
    return log_softmax(x).astype(np.float32)


def features(d, t):
    feat = {}
    feat["oof"], names = hand_features(d)
    feat["test"], _ = hand_features(t)
    xc, _ = sub_center(feat["oof"].pop("vmean"), d["sbj"])
    tc, _ = sub_center(feat["test"].pop("vmean"), t["sbj"])
    m, c, _ = pca(xc)
    feat["oof"]["vpca"] = np.concatenate([(xc - m) @ c, tstd(d["vid"](), c)], 1).astype(np.float32)
    feat["test"]["vpca"] = np.concatenate([(tc - m) @ c, tstd(t["vid"](), c)], 1).astype(np.float32)
    return feat, names


def run(d, t, nested=True, threads=4):
    """d['tab_S3'], d['tab_T'] (nested OOF log-probs), t['tab_S3'], t['tab_T']; d/t['logp_b'] blended"""
    params = dict(TAB_PARAMS, num_threads=threads)
    feat, names = features(d, t)
    for v in ("S3", "T"):
        X = tab_build(v, d, feat["oof"], names)
        Xt = tab_build(v, t, feat["test"], names)
        oof = np.zeros((len(X), N_CLS), np.float32)
        if nested:
            for f in range(FOLDS):
                tr, va = d["fold"] != f, d["fold"] == f
                b = lgb.train(params, lgb.Dataset(X[tr], d["y"][tr], params={"max_bin": params["max_bin"]}),
                              TAB_ROUNDS[v])
                oof[va] = np.log(np.clip(b.predict(X[va], num_iteration=TAB_ROUNDS[v]), 1e-7, 1))
                print(f"tab {v} fold {f} [{time.time() - T_START:.0f}s]", flush=True)
            print(f"tab {v}: nested OOF macro F1 {macro_f1(d['y'], oof.argmax(1)):.4f}", flush=True)
        b = lgb.train(params, lgb.Dataset(X, d["y"], params={"max_bin": params["max_bin"]}), TAB_ROUNDS[v])
        d["tab_" + v] = oof
        t["tab_" + v] = np.log(np.clip(b.predict(Xt, num_iteration=TAB_ROUNDS[v]), 1e-7, 1)).astype(np.float32)
        del X, Xt, b
    t["logp_b"] = tab_blend(t["logp"], [(t["tab_" + v], w) for v, w in TAB_TEST_CHOICE])
    if nested:
        d["logp_b"] = tab_blend(d["logp"], [(d["tab_" + v], w) for v, w in TAB_TEST_CHOICE])
        print(f"tab blend OOF macro F1 {macro_f1(d['y'], d['logp_b'].argmax(1)):.4f} (window alone "
              f"{macro_f1(d['y'], d['logp'].argmax(1)):.4f})", flush=True)
    return feat
