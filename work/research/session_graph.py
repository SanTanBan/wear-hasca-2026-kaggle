"""Can we recover the recording sessions of a subject from unordered tiles?

Truth on the training recordings: sessions = segments between the 1-2 strongest persistent jumps of the video
embedding along the true time order (approximate). Prediction (no time order): spectral clustering of a tile graph made
of successor links (true links never cross sessions) and video-embedding kNN edges. Then the session-aware suppression
(each activity kept only in the session holding most of its predicted mass) is scored on OOF.

  python research/session_graph.py [--knn 10] [--w-link 1.0] [--probs datasets/wear26-art:eqhh]
"""
import argparse, os, sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from blend import finish, load_P, macro_f1  # noqa: E402

SETS = {0: 2, 14: 2}


def time_sessions(E, rec, start, k=10, thr=-0.15, max_cuts=2, min_sep=300):
    sess = np.zeros(len(rec), int)
    for r in np.unique(rec):
        ii = np.flatnonzero(rec == r)
        ii = ii[np.argsort(start[ii])]
        n = len(ii)
        cs = np.cumsum(np.r_[np.zeros((1, E.shape[1])), E[ii]], 0)
        ls = np.ones(n - 1)
        for j in range(k, n - k):
            a, b = cs[j + 1] - cs[j + 1 - k], cs[j + 1 + k] - cs[j + 1]
            ls[j] = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        cuts = []
        for j in np.argsort(ls):
            if ls[j] > thr or len(cuts) >= max_cuts:
                break
            if all(abs(j - c) > min_sep for c in cuts):
                cuts.append(j)
        lab = np.zeros(n, int)
        for c in sorted(cuts):
            lab[c + 1:] += 1
        sess[ii] = lab + 10 * r
    return sess


def graph_sessions(idx, k, succ, score, E, knn=10, w_link=1.0, w_knn=1.0, seed=0):
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import NearestNeighbors
    n = len(idx)
    pos = np.full(len(succ), -1)
    pos[idx] = np.arange(n)
    s = succ[idx]
    ok = (s >= 0) & (pos[np.maximum(s, 0)] >= 0)
    w = np.clip(score[idx][ok], 0, None)
    A = sp.coo_matrix((w * w_link, (np.flatnonzero(ok), pos[s[ok]])), shape=(n, n)).tocsr()
    nn = NearestNeighbors(n_neighbors=knn + 1, metric='cosine').fit(E[idx])
    dist, ind = nn.kneighbors()
    rows = np.repeat(np.arange(n), knn)
    sim = np.clip(1 - dist[:, 1:].ravel(), 0, None)
    K = sp.coo_matrix((sim * w_knn, (rows, ind[:, 1:].ravel())), shape=(n, n)).tocsr()
    W = A + A.T + K + K.T
    W = W + sp.eye(n) * 1e-6
    return SpectralClustering(k, affinity='precomputed', random_state=seed, assign_labels='cluster_qr').fit_predict(W)


def match_acc(true, pred):
    from scipy.optimize import linear_sum_assignment
    t_u, p_u = np.unique(true), np.unique(pred)
    C = np.array([[np.sum((true == a) & (pred == b)) for b in p_u] for a in t_u])
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(true)


def suppress(P, sbj, sess, sets, eps=0.05):
    Q = P.copy()
    for s0 in np.unique(sbj):
        ms = sbj == s0
        ss = np.unique(sess[ms])
        if len(ss) < 2:
            continue
        M = np.array([P[ms & (sess == s), :].sum(0) for s in ss])
        reps = sets.get(int(s0), 1)
        for c in range(1, P.shape[1]):
            keep = np.argsort(-M[:, c])[:reps]
            for i, s in enumerate(ss):
                if i not in keep:
                    Q[ms & (sess == s), c] *= eps
    return Q / Q.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--w-link', type=float, default=1.0)
    ap.add_argument('--probs', default='datasets/wear26-art:eqhh')
    ap.add_argument('--links', default='artifacts/wear26-l4/art/oof_L3.npz')
    a = ap.parse_args()
    from sklearn.metrics import adjusted_rand_score
    z = np.load(os.path.join(HERE, 'artifacts', 'wear26-base', 'art', 'oof_B.npz'))
    rec, start, y, sbj = z['rec'], z['start'], z['y'], z['sbj']
    E = z['emb'].astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    L = np.load(os.path.join(HERE, a.links))
    succ, score = L['L3_succ'], L['L3_score_qn']
    truth = time_sessions(E, rec, start)
    pred = np.zeros(len(y), int)
    accs = []
    for r in np.unique(rec):
        ii = np.flatnonzero(rec == r)
        k = len(np.unique(truth[ii]))
        if k == 1:
            pred[ii] = 10 * r
            continue
        lab = graph_sessions(ii, k, succ, score, E, a.knn, a.w_link)
        pred[ii] = lab + 10 * r
        acc = match_acc(truth[ii], lab)
        accs.append(acc)
        print(f'rec {r}: k {k}, true sizes {sorted(np.bincount(truth[ii] - 10 * r).tolist(), reverse=True)}, '
              f'pred sizes {sorted(np.bincount(lab).tolist(), reverse=True)}, match acc {acc:.3f}, '
              f'ARI {adjusted_rand_score(truth[ii], lab):.3f}', flush=True)
    print(f'mean match accuracy {np.mean(accs):.3f}')
    d_, t_ = a.probs.rsplit(':', 1)
    P, _ = load_P(os.path.join(HERE, d_, f'oof_{t_}.npz'))
    # sessions are per recording; suppression groups by subject (sbj 0 / 14 hold two recordings)
    base = macro_f1(y, finish(P, sbj, SETS, 0.5, 97, 3))
    f_t = macro_f1(y, finish(suppress(P, rec, truth, {}), sbj, SETS, 0.5, 97, 3))
    f_p = macro_f1(y, finish(suppress(P, rec, pred, {}), sbj, SETS, 0.5, 97, 3))
    print(f'OOF {a.probs}: base {base:.4f} | time-cut sessions {f_t:.4f} | graph sessions {f_p:.4f}')


if __name__ == '__main__':
    main()
