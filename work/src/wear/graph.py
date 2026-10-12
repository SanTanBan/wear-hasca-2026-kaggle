"""Graph propagation with self-training (Briano, Apache 2.0): per subject kNN graph on (link-smoothed video
embedding + prediction similarity) mixed with the link graph; count-calibrated pseudo-label rounds."""
import re

import numpy as np
import scipy.sparse as sp

from .common import CFG, calibrate, norm_rows

TEST_CFGS = ["g0.5/abs0.8/b-2.0/a0.95/r3w0.5", "g0.5/abs0.8/b-2.0/a0.93/r3w0.5", "g0.5/abs0.8/b-2.0/a0.93/r2w0.7",
             "g0.5/abs0.6/b-2.0/a0.93/r3w0.5", "g0.5/abs0.8/b-2.0/a0.95/r2w0.7"]


class SubjectKNN:
    def __init__(self, emb, grp, K=40):
        E = emb.astype(np.float32)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        self.grp = grp
        self.groups = [np.flatnonzero(grp == g) for g in np.unique(grp)]
        self.K = K
        self.cache = {}

    def topk(self, P0, use_p, key):
        ck = (key, use_p)
        if ck in self.cache:
            return self.cache[ck]
        out = []
        for ii in self.groups:
            S = self.E[ii] @ self.E[ii].T
            if use_p:
                Q = np.sqrt(P0[ii]).astype(np.float32)
                S += use_p * (Q @ Q.T)
            np.fill_diagonal(S, -np.inf)
            nb = np.argpartition(-S, self.K, axis=1)[:, :self.K]
            vals = np.take_along_axis(S, nb, 1)
            o = np.argsort(-vals, 1)
            out.append((ii, np.take_along_axis(nb, o, 1), np.take_along_axis(vals, o, 1)))
            del S
        self.cache[ck] = out
        return out


def knn_W(nb, vals, k, tau):
    n = len(nb)
    nbk, vk = nb[:, :k], vals[:, :k]
    rows = np.repeat(np.arange(n), k)
    w = np.exp((vk - vk.max()) / tau).reshape(-1)
    W = sp.csr_matrix((w, (rows, nbk.reshape(-1))), shape=(n, n))
    return W.maximum(W.T).tocsr()


def sym_norm(W):
    d_ = np.asarray(W.sum(1)).ravel()
    d_[d_ == 0] = 1
    Dm = sp.diags(1 / np.sqrt(d_))
    return (Dm @ W @ Dm).tocsr()


def row_norm(W):
    d_ = np.asarray(W.sum(1)).ravel()
    d_[d_ == 0] = 1
    return (sp.diags(1 / d_) @ W).tocsr()


def link_W(succ, score, n_total, T=1.0, b=0.0, power=1.0, mult=None):
    """mult: optional per-tile multiplier of the link i -> succ[i] (e.g. predicted P(same label))"""
    m = succ >= 0
    src, dst = np.flatnonzero(m), succ[m]
    wt = (1 / (1 + np.exp(-np.clip((score[m] - b) / T, -20, 20)))) ** power
    if mult is not None:
        wt = wt * mult[m]
    return sp.csr_matrix((np.r_[wt, wt], (np.r_[src, dst], np.r_[dst, src])), shape=(n_total, n_total))


def label_prop_generic(P0, knn, use_p=0.5, k=10, tau=0.1, alpha=0.9, iters=20, norm="sym", Qsrc=None, Lw=None,
                       beta=0.0, key="p0", link_mode="row"):
    tops = knn.topk(P0 if Qsrc is None else Qsrc, use_p, key)
    out = np.empty_like(P0)
    for ii, nb, vals in tops:
        W = knn_W(nb, vals, k, tau)
        A_ = sym_norm(W) if norm == "sym" else row_norm(W)
        f0 = P0[ii]
        if Lw is not None and beta > 0:
            Wl = Lw[ii][:, ii]
            dl = np.asarray(Wl.sum(1)).ravel()
            Al = row_norm(Wl)
            bb = beta * (dl > 0)[:, None] if link_mode == "row" else beta * np.minimum(dl / 2, 1)[:, None]
        f = f0.copy()
        for _ in range(iters):
            g = A_ @ f
            if Lw is not None and beta > 0:
                if norm == "sym":
                    gl = Al @ f
                    s_g = g.sum(1, keepdims=True) / np.maximum(f.sum(1, keepdims=True), 1e-12)
                    g = (1 - bb) * g + bb * gl * s_g
                else:
                    g = (1 - bb) * g + bb * (Al @ f)
            f = alpha * g + (1 - alpha) * f0
        out[ii] = f
    return norm_rows(out)


def smooth_emb(emb, Lw, g=0.5, h=1):
    E = emb.astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
    if g == 0:
        return E
    An = row_norm(Lw)
    Es = E.copy()
    for _ in range(h):
        Es = E + g * (An @ Es)
    return Es.astype(np.float32)


class Runner:
    def __init__(self, dd, K=40, b=0.0, cal_iters=50):
        self.d = dd
        self.P0 = np.exp(dd["logp"])
        self.Lw = link_W(dd["succ"], dd["score"], len(self.P0), b=b, mult=dd.get("link_mult"))
        self.knns, self.K, self.stage1 = {}, K, {}
        self.cal_iters = cal_iters

    def knn(self, emb_g):
        if emb_g not in self.knns:
            E = smooth_emb(self.d["emb"], self.Lw, emb_g, 1) if emb_g else self.d["emb"]
            self.knns[emb_g] = SubjectKNN(E, self.d["grp"], self.K)
        return self.knns[emb_g]

    def cal(self, P, T):
        return calibrate(norm_rows(P ** T), self.d["sbj"], self.d["sets"], CFG["per_ex"], CFG["null_min"],
                         iters=self.cal_iters)

    def __call__(self, cfg):
        c = dict(alpha=0.95, beta=0.5, tau=0.1, k=10, iters=20, use_p=0.5, link_mode="row", emb_g=0.0,
                 rounds=1, T=2.0, w=0.5, alpha2=None, knn_update=False)
        c.update(cfg)
        knn = self.knn(c["emb_g"])
        Pk = self.d.get("knn_P", None)
        Pk = self.P0 if Pk is None else Pk
        kw = dict(use_p=c["use_p"], k=c["k"], tau=c["tau"], iters=c["iters"], norm="row", Lw=self.Lw,
                  beta=c["beta"], link_mode=c["link_mode"], key="p0")
        k1 = (c["emb_g"], c["alpha"], repr(tuple(sorted(kw.items(), key=lambda x: x[0]))))
        if k1 not in self.stage1:
            self.stage1[k1] = label_prop_generic(self.P0, knn, alpha=c["alpha"], Qsrc=Pk, **kw)
        F = self.stage1[k1]
        a2 = c["alpha2"] if c["alpha2"] is not None else c["alpha"]
        for r in range(c["rounds"] - 1):
            Q = self.cal(F, c["T"])
            if c["knn_update"]:
                kw_r = dict(kw, key=f"upd{id(F)}")
                F = label_prop_generic(norm_rows(self.P0 ** (1 - c["w"]) * Q ** c["w"]), knn, Qsrc=F, alpha=a2, **kw_r)
            else:
                F = label_prop_generic(norm_rows(self.P0 ** (1 - c["w"]) * Q ** c["w"]), knn, Qsrc=Pk, alpha=a2, **kw)
        return F


def parse_cfg(name):
    m = re.fullmatch(r"g([\d.]+)/(abs|row)([\d.]+)/b(-?[\d.]+)/a([\d.]+)/r(\d)w([\d.]+)", name)
    g, lm, beta, b, a, r, w = m.groups()
    return dict(emb_g=float(g), link_mode=lm, beta=float(beta), link_b=float(b), alpha=float(a), rounds=int(r), T=2.0,
                w=float(w))


def graph_P(dd, cfg_names=TEST_CFGS):
    Ps, runners = [], {}
    for c in sorted((parse_cfg(nm) for nm in cfg_names), key=lambda c: c["link_b"]):
        b = c.pop("link_b")
        if b not in runners:
            runners.clear()
            runners[b] = Runner(dd, b=b)
        Ps.append(runners[b](c).astype(np.float32))
    return norm_rows(np.mean(Ps, 0))
