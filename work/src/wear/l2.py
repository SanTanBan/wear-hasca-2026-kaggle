"""Combined links L2 (Briano, Apache 2.0): ridge boundary predictors, 11 candidate sources + ridge top-20, 65 pair
features and a LightGBM link scorer. Here the scorer is also trained per held-out fold, so the training subjects get
honest out-of-fold links (the public notebook only built test links)."""
import time

import lightgbm as lgb
import numpy as np

from .common import FOLDS, T_START, VLEN, WIN
from .l0 import assign

STRIDE_B, D_V = 10, 768
BF, BH, BT, BL, BM = range(5)
SHRINK, IMU_LAM, RIDGE_LAM = 0.1, 1e-3, 0.1
SOURCES = [("th", "r", 40), ("lf", "r", 15), ("mm", "r", 15), ("lfc", "r", 15),
           ("imu", "r", 8), ("imu", "c", 5), ("extlin", "r", 25), ("extlin", "c", 10),
           ("lfz", "c", 30), ("ext2", "r", 25), ("dual", "r", 25)]
KCAP, KNEW, LOWV = 160, 20, -1e4
OLD30 = ["fwd", "frame", "pooled", "rev", "dom_r", "dom_c", "margin", "rank_r", "rank_c", "same", "gap", "ext",
         "lfc", "thc", "mmc", "revc", "lfc_dom_r", "lfc_dom_c", "ext2", "dratio",
         "bc_w", "bc_lp", "maxp_i", "maxp_j", "magdiff", "stdratio", "limb_i", "limb_j", "gap3", "slope"]
NEW_DECODE = ["lfz", "lfz_dom_r", "lfz_dom_c", "lfz_rank_r", "lfz_rank_c", "extlin", "extlin_rank_r", "extlin_rank_c",
              "ext2_rank_r", "dual", "dual_rank_r", "imu_d", "imu_dom_r", "imu_dom_c", "imu_rank_r", "imu_rank_c",
              "lfc_rank_r", "lfc_rank_c", "n_src", "bc_ca", "maxca_i", "maxca_j", "l1_ca"]
CA_FEATS = ["bc_ca", "maxca_i", "maxca_j", "l1_ca"]
BASE49 = [f for f in OLD30 + NEW_DECODE if f not in CA_FEATS]
NEW16 = ["ll_f", "ll_b", "ll_f_domc", "ll_f_domr", "ll_b_domr", "ll_b_domc", "rs", "rs_domr", "rs_domc", "rs_rank_r",
         "rs_rank_c", "cos_f", "cos_b", "raw_f", "imu_f", "imu_b"]
FEAT_NAMES = BASE49 + NEW16
LK_NEG = 16
LK_ROUNDS = 300
LK_PARAMS = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                 bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                 num_threads=4, verbosity=-1, seed=0)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def bs_vid_blocks(fr):
    return np.concatenate([fr[:, 0], fr[:, :3].mean(1), fr[:, -3:].mean(1), fr[:, -1], fr.mean(1)], 1)


def bs_imu_summ(a):
    mag = np.linalg.norm(a, axis=2)
    lsd = lambda m: np.log(m.std(1) + 0.01)[:, None]
    tail_in = np.concatenate([a.mean(1), a.std(1), a[:, -10:].mean(1), a[:, -10:].std(1), a[:, -1], a[:, -3:].mean(1), lsd(mag)], 1)
    head_in = np.concatenate([a.mean(1), a.std(1), a[:, :10].mean(1), a[:, :10].std(1), a[:, 0], a[:, :3].mean(1), lsd(mag)], 1)
    head_out = np.concatenate([a[:, :10].mean(1), a[:, :3].mean(1), a.mean(1), lsd(mag[:, :10]), lsd(mag)], 1)
    tail_out = np.concatenate([a[:, -10:].mean(1), a[:, -3:].mean(1), a.mean(1), lsd(mag[:, -10:]), lsd(mag)], 1)
    return tail_in, head_out, head_in, tail_out


def add1(x):
    return np.concatenate([x, np.ones((len(x), 1), x.dtype)], 1)


class Gram:
    def __init__(self, dim, gpu=False):
        self.torch = None
        if gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    self.torch, self.dev = torch, torch.device("cuda")
                    self.S = torch.zeros((dim, dim), dtype=torch.float64, device=self.dev)
                    self.C = torch.zeros((dim, dim), dtype=torch.float64, device=self.dev)
            except Exception:
                self.torch = None
        if self.torch is None:
            self.S, self.C = np.zeros((dim, dim)), np.zeros((dim, dim))

    def add(self, zi, zj):
        if self.torch is None:
            self.S += zi.T @ zi
            self.C += zi.T @ zj
        else:
            a, b = self.torch.from_numpy(zi).to(self.dev), self.torch.from_numpy(zj).to(self.dev)
            self.S += (a.T @ a).double()
            self.C += (a.T @ b).double()

    def result(self):
        if self.torch is None:
            return self.S, self.C
        S, C = self.S.cpu().numpy(), self.C.cpu().numpy()
        del self.S, self.C
        self.torch.cuda.empty_cache()
        return S, C


def subject_stats(P, recs_):
    acc_sum, cnt = np.zeros(D_V), 0
    for r in recs_:
        v = np.load(P.PROC / f"{r['stem']}_vid.npy", mmap_mode="r")
        st = np.arange(0, r["T"] - WIN + 1, WIN)
        st = st[st * 3 // 5 + 8 + VLEN <= r["F"]]
        idx = ((st * 3 // 5)[:, None] + 8 + np.arange(VLEN)).ravel()
        fr = unit(np.asarray(v[idx], np.float32))
        acc_sum += fr.sum(0)
        cnt += len(fr)
    mu = (acc_sum / cnt).astype(np.float32)
    Dz = 5 * D_V + 1
    gram = Gram(Dz)
    n = 0
    IM = {k: np.zeros(s) for k, s in (("fS", (4, 4, 20, 20)), ("fC", (4, 4, 20, 11)), ("fY", (4, 4, 11, 11)), ("fn", (4, 4)),
                                      ("bS", (4, 4, 20, 20)), ("bC", (4, 4, 20, 11)), ("bY", (4, 4, 11, 11)))}
    for r in recs_:
        v = np.load(P.PROC / f"{r['stem']}_vid.npy", mmap_mode="r")
        a = np.load(P.PROC / f"{r['stem']}_acc.npy").reshape(-1, 4, 3)
        val = np.load(P.PROC / f"{r['stem']}_valid.npy")
        T, Fr = r["T"], r["F"]
        st = np.arange(0, T - 2 * WIN + 1, STRIDE_B)
        st = st[(st + WIN) * 3 // 5 + 8 + VLEN <= Fr]
        frames = unit(np.asarray(v[: Fr], np.float32)) - mu
        for c0 in range(0, len(st), 4000):
            s = st[c0:c0 + 4000]
            fi = (s * 3 // 5)[:, None] + 8 + np.arange(VLEN)
            zi = add1(bs_vid_blocks(frames[fi]))
            zj = add1(bs_vid_blocks(frames[fi + 30]))
            gram.add(zi, zj)
            n += len(s)
            del zi, zj
        del frames
        miss = np.zeros((T + 1, 4), np.int32)
        np.cumsum(~val, axis=0, out=miss[1:])
        ok_i = (miss[st + WIN] - miss[st]) == 0
        ok_j = (miss[st + 2 * WIN] - miss[st + WIN]) == 0
        su_i = [bs_imu_summ(a[st[:, None] + np.arange(WIN), l]) for l in range(4)]
        su_j = [bs_imu_summ(a[st[:, None] + WIN + np.arange(WIN), l]) for l in range(4)]
        for l in range(4):
            for l2 in range(4):
                m = ok_i[:, l] & ok_j[:, l2]
                if not m.any():
                    continue
                x, y = add1(su_i[l][0][m]), su_j[l2][1][m]
                IM["fS"][l, l2] += x.T @ x
                IM["fC"][l, l2] += x.T @ y
                IM["fY"][l, l2] += y.T @ y
                IM["fn"][l, l2] += m.sum()
                x, y = add1(su_j[l2][2][m]), su_i[l][3][m]
                IM["bS"][l, l2] += x.T @ x
                IM["bC"][l, l2] += x.T @ y
                IM["bY"][l, l2] += y.T @ y
    S, C = gram.result()
    return dict(S=S.astype(np.float32), C=C.astype(np.float32), n=n, **IM)


def blk(*bs):
    return np.concatenate([np.arange(b * D_V, (b + 1) * D_V) for b in bs] + [np.array([5 * D_V])])


XF, YF = blk(BL, BT, BH, BM), np.arange(BH * D_V, (BH + 1) * D_V)
XB, YB = blk(BF, BH, BT, BM), np.arange(BT * D_V, (BT + 1) * D_V)


def _dir_stats(st, fwd):
    S, C = st["S"], st["C"]
    if fwd:
        xx, xy, yy = S[np.ix_(XF, XF)], C[np.ix_(XF, YF)], S[np.ix_(YF, YF)]
    else:
        xx, xy, yy = S[np.ix_(XB, XB)], C[np.ix_(YB, XB)].T, S[np.ix_(YB, YB)]
    return xx, xy, yy, float(st["n"])


def _solve(xx, xy, n, lam):
    reg = lam * np.mean(np.diag(xx)[:-1])
    A_ = xx + reg * np.diag(np.r_[np.ones(len(xx) - 1), 0.0])
    return np.linalg.solve(A_, xy)


def _prec_sqrt(R):
    e, U = np.linalg.eigh((R + R.T) / 2)
    e = np.maximum(e, 0)
    e = e + SHRINK * e.mean()
    return ((U / np.sqrt(e)) @ U.T).astype(np.float32), float(np.sum(np.log(e)))


def fit_ridge(fold_sum, train_folds):
    st = None
    for g in train_folds:
        f = fold_sum[g]
        if st is None:
            st = {k: np.array(v, np.float64) for k, v in f.items()}
        else:
            for k in st:
                st[k] += f[k]
    m = {"lam_f": RIDGE_LAM, "lam_b": RIDGE_LAM}
    for fwd, tag in ((True, "f"), (False, "b")):
        xx, xy, yy, n = _dir_stats(st, fwd)
        W = _solve(xx, xy, n, RIDGE_LAM)
        R = (yy - xy.T @ W - W.T @ xy + W.T @ xx @ W) / n
        m["W" + tag] = W.astype(np.float32)
        m["P" + tag], _ = _prec_sqrt(R)
        m["r2" + tag] = 1 - np.trace(R) / (np.trace(yy) / n)
    for tag in ("f", "b"):
        Wl = np.zeros((4, 4, 20, 11), np.float32)
        Pl = np.zeros((4, 4, 11, 11), np.float32)
        Ld = np.zeros((4, 4), np.float32)
        for l in range(4):
            for l2 in range(4):
                xx, xy, yy, n = st[tag + "S"][l, l2], st[tag + "C"][l, l2], st[tag + "Y"][l, l2], st["fn"][l, l2]
                reg = IMU_LAM * np.mean(np.diag(xx)[:-1])
                W = np.linalg.solve(xx + reg * np.diag(np.r_[np.ones(19), 0.0]), xy)
                R = (yy - xy.T @ W - W.T @ xy + W.T @ xx @ W) / n
                R = (R + R.T) / 2 + 1e-4 * np.eye(11)
                Pr = np.linalg.inv(R)
                Wl[l, l2], Pl[l, l2], Ld[l, l2] = W, Pr, np.linalg.slogdet(Pr)[1]
        m["imuW" + tag], m["imuP" + tag], m["imuL" + tag] = Wl, Pl, Ld
    return m


def ranks_at(S, cand, axis):
    n, K = cand.shape
    ch = np.maximum(cand, 0)
    rows = np.arange(n)[:, None]
    M = S if axis == 1 else S.T
    Ss = np.sort(M, axis=1).astype(np.float64)
    lo, hi = float(Ss.min()), float(Ss.max())
    span = hi - lo + 1.0
    flat = (Ss - lo + (np.arange(n) * span)[:, None]).ravel()
    del Ss
    if axis == 1:
        v, base = S[rows, ch], rows
    else:
        v, base = S[rows, ch], ch
    q = (v.astype(np.float64) - lo + base * span)
    pos = np.searchsorted(flat, q.ravel(), side="right").reshape(n, K)
    return (base + 1) * n - pos


def topk_rows(S, k):
    k = min(k, S.shape[1] - 1)
    return np.argpartition(-S, k - 1, axis=1)[:, :k]


def topk(S, k):
    return np.argpartition(-S, k - 1, axis=1)[:, :k]


def vid_blocks(frc):
    return np.concatenate([frc[:, 0], frc[:, :3].mean(1), frc[:, -3:].mean(1), frc[:, -1], frc.mean(1),
                           np.ones((len(frc), 1), np.float32)], 1)


def imu_summ(a):
    mag = np.linalg.norm(a, axis=2)
    lsd = lambda x: np.log(x.std(1) + 0.01)[:, None]
    tail_in = np.concatenate([a.mean(1), a.std(1), a[:, -10:].mean(1), a[:, -10:].std(1), a[:, -1], a[:, -3:].mean(1), lsd(mag), np.ones((len(a), 1))], 1)
    head_in = np.concatenate([a.mean(1), a.std(1), a[:, :10].mean(1), a[:, :10].std(1), a[:, 0], a[:, :3].mean(1), lsd(mag), np.ones((len(a), 1))], 1)
    head_out = np.concatenate([a[:, :10].mean(1), a[:, :3].mean(1), a.mean(1), lsd(mag[:, :10]), lsd(mag)], 1)
    tail_out = np.concatenate([a[:, -10:].mean(1), a[:, -3:].mean(1), a.mean(1), lsd(mag[:, -10:]), lsd(mag)], 1)
    return [x.astype(np.float32) for x in (tail_in, head_out, head_in, tail_out)]


def imu_pair_ll(summ, limbs, I, J, m):
    tail_in, head_out, head_in, tail_out = summ
    lf = np.zeros(len(I), np.float32)
    lb = np.zeros(len(I), np.float32)
    li, lj = limbs[I], limbs[J]
    for l in range(4):
        for l2 in range(4):
            k = np.flatnonzero((li == l) & (lj == l2))
            if not len(k):
                continue
            i, j = I[k], J[k]
            r = head_out[j] - tail_in[i] @ m["imuWf"][l, l2]
            lf[k] = -0.5 * np.einsum("nd,de,ne->n", r, m["imuPf"][l, l2], r) + 0.5 * m["imuLf"][l, l2]
            r = tail_out[i] - head_in[j] @ m["imuWb"][l, l2]
            lb[k] = -0.5 * np.einsum("nd,de,ne->n", r, m["imuPb"][l, l2], r) + 0.5 * m["imuLb"][l, l2]
    return lf, lb


def ll_mats(z, m):
    p_ = z[:, XF] @ m["Wf"]
    q = z[:, XB] @ m["Wb"]
    out = []
    for pred, tgt, P, fwd in ((p_, z[:, YF], m["Pf"], True), (q, z[:, YB], m["Pb"], False)):
        a, b = pred @ P, tgt @ P
        L = (b @ a.T) if not fwd else (a @ b.T)
        L *= 2
        if fwd:
            L -= (a * a).sum(1)[:, None]
            L -= (b * b).sum(1)[None, :]
        else:
            L -= (b * b).sum(1)[:, None]
            L -= (a * a).sum(1)[None, :]
        L *= 0.5 / D_V
        np.fill_diagonal(L, -1e4)
        out.append(L)
    return out, p_, q


def rs_mat(ll_f, ll_b):
    fcm, bcm = ll_f.max(0), ll_b.max(0)
    rs = ll_b + (ll_b - bcm[None]) + (ll_f - fcm[None])
    np.fill_diagonal(rs, -1e5)
    return rs


def new_feats(z, mu, m, cand, summ, limbs):
    n, K = cand.shape
    valid = cand >= 0
    ch = np.where(valid, cand, 0)
    rows = np.arange(n)[:, None]
    (ll_f, ll_b), p_, q = ll_mats(z, m)
    F = {"ll_f": ll_f[rows, ch], "ll_b": ll_b[rows, ch]}
    fcm, frm, bcm, brm = ll_f.max(0), ll_f.max(1), ll_b.max(0), ll_b.max(1)
    F["ll_f_domc"] = F["ll_f"] - fcm[ch]
    F["ll_f_domr"] = F["ll_f"] - frm[:, None]
    F["ll_b_domr"] = F["ll_b"] - brm[:, None]
    F["ll_b_domc"] = F["ll_b"] - bcm[ch]
    rs = rs_mat(ll_f, ll_b)
    del ll_f, ll_b
    F["rs"] = rs[rows, ch]
    F["rs_domr"] = F["rs"] - rs.max(1)[:, None]
    F["rs_domc"] = F["rs"] - rs.max(0)[ch]
    rr = np.empty((n, n), np.int32)
    rr[rows, np.argsort(-rs, 1)] = np.arange(n, dtype=np.int32)[None]
    F["rs_rank_r"] = np.log1p(rr[rows, ch])
    rr[np.argsort(-rs, 0), np.arange(n)[None]] = np.arange(n, dtype=np.int32)[:, None]
    F["rs_rank_c"] = np.log1p(rr[rows, ch])
    del rr, rs
    H, T = z[:, YF], z[:, YB]
    S = unit(p_) @ unit(H).T
    F["cos_f"] = S[rows, ch]
    S = unit(T) @ unit(q).T
    F["cos_b"] = S[rows, ch]
    S = unit(p_ + mu) @ unit(H + mu).T
    F["raw_f"] = S[rows, ch]
    del S
    I = np.broadcast_to(rows, cand.shape)[valid]
    J = cand[valid]
    lf, lb = imu_pair_ll(summ, limbs, I, J, m)
    F["imu_f"] = np.zeros((n, K), np.float32)
    F["imu_b"] = np.zeros((n, K), np.float32)
    F["imu_f"][valid], F["imu_b"][valid] = lf, lb
    out = np.stack([F[k] for k in NEW16], 2).astype(np.float32)
    out[~valid] = np.nan
    return out


def extra_feats(extra, cand):
    """features of extra (n,n) score matrices at the candidate pairs: value, row/col dominance, row/col rank"""
    n, K = cand.shape
    ch = np.maximum(cand, 0)
    rows = np.arange(n)[:, None]
    out = []
    for name in sorted(extra):
        S = extra[name][0]
        v = S[rows, ch]
        out += [v, v - S.max(1)[:, None], v - S.max(0)[ch], np.log1p(ranks_at(S, cand, 1)),
                np.log1p(ranks_at(S, cand, 0))]
    return np.stack(out, 2).astype(np.float32)


def extra_names(extra):
    return [f"{nm}_{k}" for nm in sorted(extra) for k in ("v", "dom_r", "dom_c", "rank_r", "rank_c")]


def subject_all(V, A, limbs, logp, plp, models, extra=None):
    """extra: optional {name: (S (n,n) float32 with very low diagonal, k_rows, k_cols)} -> extra candidate sources
    and 5 features each (returned as a 5th value)"""
    n = len(V)
    fr = unit(np.asarray(V, np.float32))
    mu = fr.reshape(-1, fr.shape[-1]).mean(0)
    sd = fr.reshape(-1, fr.shape[-1]).std(0) + 1e-3
    frc_all = fr - mu
    z = vid_blocks(frc_all)
    del frc_all
    T, H = unit(fr[:, -3:].mean(1)), unit(fr[:, :3].mean(1))
    last, first, mp = fr[:, -1].copy(), fr[:, 0].copy(), unit(fr.mean(1))
    frc_last, frc_first = fr[:, -1] - mu, fr[:, 0] - mu
    frc_l4, frc_mean = fr[:, -4] - mu, fr.mean(1) - mu
    Tc, Hc = fr[:, -3:].mean(1) - mu, fr[:, :3].mean(1) - mu
    del fr
    a = np.asarray(A, np.float32)
    e = 2 * a[:, -1] - a[:, -2]
    sl = limbs[:, None] == limbs[None, :]

    def mat(name):
        if name == "th":
            S = T @ H.T
        elif name == "lf":
            S = last @ first.T
        elif name == "mm":
            S = mp @ mp.T
        elif name == "lfc":
            S = unit(frc_last) @ unit(frc_first).T
        elif name == "lfz":
            S = unit(frc_last / sd) @ unit(frc_first / sd).T
        elif name == "extlin":
            S = unit(frc_last + (frc_last - frc_l4) * 16 / 3) @ unit(frc_first).T
        elif name == "ext2":
            S = unit(Tc + 1.5 * (Tc - Hc)) @ unit(Hc).T
            S += unit(Tc) @ unit(Hc - 1.5 * (Tc - Hc)).T
        elif name == "thc":
            S = unit(Tc) @ unit(Hc).T
        elif name == "dual":
            L = (unit(frc_last) @ unit(frc_first).T) / 0.05
            np.fill_diagonal(L, LOWV)
            rm = L.max(1, keepdims=True)
            r = rm + np.log(np.exp(L - rm).sum(1, keepdims=True))
            cm = L.max(0, keepdims=True)
            c = cm + np.log(np.exp(L - cm).sum(0, keepdims=True))
            S = 2 * L - r - c
        elif name == "imu":
            S = np.empty((n, n), np.float32)
            for s0 in range(0, n, 1024):
                S[s0:s0 + 1024] = -np.linalg.norm(e[s0:s0 + 1024, None] - a[None, :, 0], axis=2)
            S[~sl] = LOWV
        S = np.asarray(S, np.float32)
        np.fill_diagonal(S, LOWV)
        return S

    tops = []
    for m in models:
        (ll_f, ll_b), _, _ = ll_mats(z, m)
        rs = rs_mat(ll_f, ll_b)
        del ll_f, ll_b
        tops.append(np.ascontiguousarray(topk(rs, KNEW)))
        del rs
    parts = []
    for name, dr, k in SOURCES:
        S = mat(name)
        if dr == "r":
            parts.append(np.ascontiguousarray(topk_rows(S, k)))
        else:
            tk = topk_rows(S.T, k)
            jj = np.repeat(np.arange(n), tk.shape[1])
            parts.append((tk.ravel(), jj))
        del S
    for name in sorted(extra or {}):
        S, kr, kc = extra[name]
        if kr:
            parts.append(np.ascontiguousarray(topk_rows(S, kr)))
        if kc:
            tk = topk_rows(S.T, kc)
            parts.append((tk.ravel(), np.repeat(np.arange(n), tk.shape[1])))
    pairs_i = [np.repeat(np.arange(n), p_.shape[1]) if not isinstance(p_, tuple) else p_[0] for p_ in parts]
    pairs_j = [p_.ravel() if not isinstance(p_, tuple) else p_[1] for p_ in parts]
    pi, pj = np.concatenate(pairs_i), np.concatenate(pairs_j)
    ok = pi != pj
    key = pi[ok].astype(np.int64) * n + pj[ok]
    uk, nsrc = np.unique(key, return_counts=True)
    ui, uj = uk // n, uk % n
    cntr = np.bincount(ui, minlength=n)
    Kd = int(min(cntr.max(), KCAP))
    start = np.concatenate([[0], np.cumsum(cntr)])
    lists, nss = [], []
    ext_all = np.concatenate(tops, 1)
    for r in range(n):
        js, ns = uj[start[r]:start[r + 1]], nsrc[start[r]:start[r + 1]]
        if len(js) > Kd:
            o = np.argsort(-ns, kind="stable")[:Kd]
            js, ns = js[o], ns[o]
        ex = np.unique(ext_all[r])
        ex = ex[(ex != r) & ~np.isin(ex, js)]
        lists.append(np.concatenate([js, ex]).astype(np.int32))
        nss.append(np.concatenate([ns, np.zeros(len(ex))]).astype(np.float32))
    K = max(len(x) for x in lists)
    cand = np.full((n, K), -1, np.int32)
    nsr = np.zeros((n, K), np.float32)
    for r in range(n):
        m_ = len(lists[r])
        cand[r, :m_], nsr[r, :m_] = lists[r], nss[r]
    del lists, nss
    valid = cand >= 0
    ch = np.maximum(cand, 0)
    rows = np.arange(n)[:, None]
    F = {"n_src": nsr}
    th = mat("th")
    F["fwd"], F["rev"] = th[rows, ch], th[ch, rows]
    F["dom_r"] = F["fwd"] - th.max(1)[:, None]
    F["dom_c"] = F["fwd"] - th.max(0)[ch]
    srt = np.partition(th, n - 2, axis=1)[:, -2:]
    F["margin"] = np.broadcast_to((srt[:, 1] - srt[:, 0])[:, None], ch.shape)
    F["rank_r"] = np.log1p(ranks_at(th, cand, 1))
    F["rank_c"] = np.log1p(ranks_at(th, cand, 0))
    del th
    F["frame"] = mat("lf")[rows, ch]
    F["pooled"] = mat("mm")[rows, ch]
    S = mat("lfc")
    F["lfc"] = S[rows, ch]
    F["lfc_dom_r"] = F["lfc"] - S.max(1)[:, None]
    F["lfc_dom_c"] = F["lfc"] - S.max(0)[ch]
    F["lfc_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    F["lfc_rank_c"] = np.log1p(ranks_at(S, cand, 0))
    S = mat("thc")
    F["thc"], F["revc"] = S[rows, ch], S[ch, rows]
    del S
    mpc = unit(frc_mean)
    S = mpc @ mpc.T
    F["mmc"] = S[rows, ch]
    del S
    S = mat("ext2")
    F["ext2"] = S[rows, ch]
    F["ext2_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    din = 1 - np.sum(unit(Tc) * unit(Hc), 1)
    F["dratio"] = np.log((1 - F["thc"] + 1e-4) / (0.5 * (din[:, None] + din[ch]) + 1e-4))
    S = mat("lfz")
    F["lfz"] = S[rows, ch]
    F["lfz_dom_r"] = F["lfz"] - S.max(1)[:, None]
    F["lfz_dom_c"] = F["lfz"] - S.max(0)[ch]
    F["lfz_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    F["lfz_rank_c"] = np.log1p(ranks_at(S, cand, 0))
    S = mat("extlin")
    F["extlin"] = S[rows, ch]
    F["extlin_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    F["extlin_rank_c"] = np.log1p(ranks_at(S, cand, 0))
    S = mat("dual")
    F["dual"] = S[rows, ch]
    F["dual_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    S = mat("imu")
    same = sl[rows, ch]
    v = S[rows, ch]
    F["imu_d"] = np.where(same, np.log1p(-v), np.nan)
    F["imu_dom_r"] = np.where(same, np.log1p(-v) - np.log1p(-S.max(1))[:, None], np.nan)
    F["imu_dom_c"] = np.where(same, np.log1p(-v) - np.log1p(-S.max(0))[ch], np.nan)
    F["imu_rank_r"] = np.where(same, np.log1p(ranks_at(S, cand, 1)), np.nan)
    F["imu_rank_c"] = np.where(same, np.log1p(ranks_at(S, cand, 0)), np.nan)
    del S
    pw = np.sqrt(np.exp(logp)).astype(np.float32)
    F["bc_w"] = np.einsum("nc,nkc->nk", pw, pw[ch])
    pl = np.sqrt(plp).astype(np.float32)
    F["bc_lp"] = np.einsum("nc,nkc->nk", pl, pl[ch])
    F["maxp_i"] = np.broadcast_to(plp.max(1)[:, None], ch.shape)
    F["maxp_j"] = plp.max(1)[ch]
    F["same"] = same.astype(np.float32)
    a0 = a[:, 0]
    gap = np.linalg.norm(a[:, -1][:, None] - a0[ch], axis=2)
    ext = np.linalg.norm(e[:, None] - a0[ch], axis=2)
    F["gap"] = np.where(same, np.log1p(gap), np.nan)
    F["ext"] = np.where(same, np.log1p(ext), np.nan)
    g3 = np.linalg.norm(a[:, -3:].mean(1)[:, None] - a[:, :3].mean(1)[ch], axis=2)
    F["gap3"] = np.where(same, np.log1p(g3), np.nan)
    sl_i, sl_j = a[:, -1] - a[:, -4], a[:, 3] - a[:, 0]
    F["slope"] = np.where(same, np.log1p(np.linalg.norm(sl_i[:, None] - sl_j[ch], axis=2)), np.nan)
    mag = np.linalg.norm(a, axis=2)
    F["magdiff"] = np.abs(mag[:, -10:].mean(1)[:, None] - mag[:, :10].mean(1)[ch])
    ls = np.log(mag.std(1) + 0.01)
    F["stdratio"] = np.abs(ls[:, None] - ls[ch])
    F["limb_i"] = np.broadcast_to(limbs[:, None].astype(np.float32), ch.shape)
    F["limb_j"] = limbs[ch].astype(np.float32)
    base = np.stack([np.asarray(F[k], np.float32)[valid] for k in BASE49], 1).astype(np.float16)
    del F
    summ = imu_summ(a)
    new = np.stack([new_feats(z, mu, m, cand, summ, limbs)[valid] for m in models])
    if extra:
        return cand, valid, base, new, extra_feats(extra, cand)[valid]
    return cand, valid, base, new


def lk_assign(cl, lo, floor=-50.0):
    n = len(cl)
    cost = np.full((n, n), floor, np.float32)
    valid = cl >= 0
    rows = np.broadcast_to(np.arange(n)[:, None], cl.shape)
    cost[rows[valid], cl[valid]] = lo[valid]
    np.fill_diagonal(cost, -1e4)
    succ, score = assign(cost, min_logodds=floor + 1)
    return succ, score


QN_REF = None  # filled lazily from qn_ref.npy shipped with the package


def qnorm(succ, score, sbj, ref):
    ref = np.sort(np.asarray(ref, np.float64))
    out = score.astype(np.float32).copy()
    for s in np.unique(sbj):
        m = np.flatnonzero((sbj == s) & (succ >= 0))
        if not len(m):
            continue
        r = np.argsort(np.argsort(score[m], kind="stable"), kind="stable")
        u = (r + 0.5) / len(m)
        out[m] = np.quantile(ref, u).astype(np.float32)
    return out


def ridges(P, recs, fold_of):
    fold_sum = {}
    for g in range(FOLDS):
        acc_ = None
        for sb in sorted(s for s in fold_of if fold_of[s] == g):
            st_ = subject_stats(P, [r for r in recs if r["sbj"] == sb])
            st_ = {k: np.array(v, np.float64) for k, v in st_.items()}
            if acc_ is None:
                acc_ = st_
            else:
                for k in acc_:
                    acc_[k] += st_[k]
            del st_
        fold_sum[g] = {k: (v.astype(np.float32) if k in ("S", "C") else v) for k, v in acc_.items()}
        del acc_
        print(f"ridge stats fold {g}: {int(fold_sum[g]['n'])} tile pairs [{time.time() - T_START:.0f}s]", flush=True)
    R = [fit_ridge(fold_sum, [h for h in range(FOLDS) if h != g]) for g in range(FOLDS)]
    print("ridge r2 (forward, backward):", [(round(float(m["r2f"]), 3), round(float(m["r2b"]), 3)) for m in R],
          "(Briano: about 0.58)", flush=True)
    return R


def run(P, recs, d, t, plp_oof, plp_test, qn_ref, fold_of, threads=4, rounds=LK_ROUNDS):
    """OOF links with fold-held-out scorers + test links with the all-subject scorer (5 ridge variants averaged)"""
    params = dict(LK_PARAMS, num_threads=threads)
    RIDGES = ridges(P, recs, fold_of)
    _V, _A = d["vid"](), d["acc"]()
    per_sbj = {}
    Xs, Ts, Gs = [], [], []
    for s in np.unique(d["sbj"]):
        ii = np.flatnonzero(d["sbj"] == s)
        n = len(ii)
        g = fold_of[int(s)]
        cand, valid, base, new = subject_all(_V[ii], _A[ii], d["sensor"][ii], d["logp"][ii], plp_oof[ii], [RIDGES[g]])
        X_all = np.concatenate([base.astype(np.float32), new[0]], 1)
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
        keep = o[rank < LK_NEG + npos[rp[o]]]
        Xs.append(X_all[keep])
        Ts.append(pos[keep].astype(np.float32))
        Gs.append(np.full(len(keep), g))
        recall = ((cand == true_loc[:, None]) & valid).any(1)[true_loc >= 0].mean()
        per_sbj[int(s)] = (ii, cand, valid, X_all)
        print(f"L2 sbj {s} fold {g} n={n} K={cand.shape[1]} recall {recall:.3f} [{time.time() - T_START:.0f}s]", flush=True)
        del base, new
    del _V, _A
    X, T, G = np.concatenate(Xs), np.concatenate(Ts), np.concatenate(Gs)
    del Xs, Ts
    print(f"scorer rows {len(T)} positive rate {T.mean():.4f}  (Briano: 1,173,696 / 0.0549)", flush=True)
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
            s_loc, sc = lk_assign(cand, L)
            succ_o[ii] = np.where(s_loc >= 0, ii[np.maximum(s_loc, 0)], -1)
            score_o[ii] = sc
        print(f"L2 OOF fold {g} done [{time.time() - T_START:.0f}s]", flush=True)
        del model
    del per_sbj
    model = lgb.train(params, lgb.Dataset(X, T), num_boost_round=rounds)
    del X, T
    d["l2_succ"], d["l2_score"] = succ_o, score_o
    d["l2_score_qn"] = qnorm(succ_o, score_o, d["sbj"], qn_ref)
    _tv, _ta = t["vid"](), t["acc"]()
    NT = len(t["sbj"])
    succ_t, score_t = np.full(NT, -1, np.int64), np.full(NT, -50.0, np.float32)
    for s in np.unique(t["sbj"]):
        ii = np.flatnonzero(t["sbj"] == s)
        cand, valid, base, new = subject_all(_tv[ii], _ta[ii], t["sensor"][ii], t["logp"][ii], plp_test[ii], RIDGES)
        b = base.astype(np.float32)
        acc_ = np.zeros(valid.sum(), np.float64)
        for v in range(len(RIDGES)):
            p_ = np.clip(model.predict(np.concatenate([b, new[v]], 1), num_threads=threads), 1e-6, 1 - 1e-6)
            acc_ += np.log(p_ / (1 - p_)).astype(np.float32)
        L = np.full(cand.shape, -50.0, np.float32)
        L[valid] = acc_ / len(RIDGES)
        s_loc, sc = lk_assign(cand, L)
        succ_t[ii] = np.where(s_loc >= 0, ii[np.maximum(s_loc, 0)], -1)
        score_t[ii] = sc
        m = succ_t[ii] >= 0
        print(f"L2 test sbj {s} n={len(ii)} K={cand.shape[1]} linked {m.mean():.3f} median score {np.median(sc[m]):.2f}",
              flush=True)
        del cand, valid, base, new, b
    t["l2_succ"], t["l2_score"] = succ_t, score_t
    t["l2_score_qn"] = qnorm(succ_t, score_t, t["sbj"], qn_ref)
    return RIDGES, model
