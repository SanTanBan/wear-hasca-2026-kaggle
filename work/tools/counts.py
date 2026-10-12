"""Learned count prior (idea and feature design from the public notebook 'Learned Links + Counts' by goodpjw2008):
for every (subject, exercise) a ridge regression reads the margin profile of the class (how strongly the k-th best
tile still prefers it, k = 60..140) from the graph probabilities P and the tile log-probs B, and predicts the bout's
tile count. Nested by subject fold on the 22 training subjects; the test subjects use the model on all of them. The
predicted counts replace the fixed 97 per exercise in the per-subject Sinkhorn calibration.

  python tools/counts.py --src datasets/wear26-art:eq15 [--b artifacts/wear26-base/art/oof_B.npz] [--out subs/x.csv]
                         [--save dir:tag] [--queue "msg"]
"""
import argparse, csv, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from blend import macro_f1, load_P, finish  # noqa: E402

N_CLS = 19
TRAIN_SETS = {0: 2, 14: 2}
RK = [60, 70, 80, 90, 97, 105, 115, 125, 140]


def profile_features(mats, sbj, sets):
    X, key = [], []
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        ns = sets.get(int(s), 1)
        for c in range(1, N_CLS):
            f = []
            for M in mats:
                Mi = M[ii]
                q = Mi[:, c]
                other = np.delete(Mi, c, 1).max(1)
                sc = np.sort(np.log(q + 1e-6) - np.log(other + 1e-6))[::-1]
                f += [sc[min(int(r * ns), len(sc) - 1)] for r in RK]
                f += [q.sum() / ns, (Mi.argmax(1) == c).sum() / ns, (sc > 0).sum() / ns, (sc > -1).sum() / ns,
                      (sc > 1).sum() / ns]
            X.append(f + [len(ii) / ns])
            key.append((int(s), c))
    return np.array(X, np.float64), np.array(key)


def ridge_fit(X, y, alphas=(1, 3, 10, 30, 100, 300, 1000)):
    """ridge with intercept; alpha by leave-one-out error (closed form, like sklearn RidgeCV)"""
    Xa = np.c_[X, np.ones(len(X))]
    best = None
    for a in alphas:
        R = np.eye(Xa.shape[1]) * a
        R[-1, -1] = 0
        A = np.linalg.inv(Xa.T @ Xa + R)
        w = A @ Xa.T @ y
        H = np.einsum('ij,jk,ik->i', Xa, A, Xa)
        loo = np.mean(((y - Xa @ w) / np.maximum(1 - H, 1e-6)) ** 2)
        if best is None or loo < best[0]:
            best = (loo, w, a)
    return best[1], best[2]


def fit_counts(Xtr, ttr, Xte):
    reg = (ttr > 55) & (ttr < 150)     # bouts done twice or skipped are not predictable
    mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6
    w, a = ridge_fit((Xtr[reg] - mu) / sd, ttr[reg])
    pred = np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ w
    return np.clip(pred, 70, 135)


def count_targets(sbj, sets, key, cnt, null_min=0.05):
    out = {}
    for s in np.unique(sbj):
        n = int((sbj == s).sum())
        per = cnt[key[:, 0] == s] * sets.get(int(s), 1)
        null = max(n - per.sum(), null_min * n)
        out[int(s)] = np.r_[null, per * (n - null) / per.sum()]
    return out


def calibrate_targets(P, sbj, targets, iters=50, T=0.5):
    Q0 = np.clip(P, 1e-12, None) ** (1 / T)
    Q0 /= Q0.sum(1, keepdims=True)
    out = np.empty_like(Q0)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        t_ = targets[int(s)]
        Q = Q0[ii].copy()
        for _ in range(iters):
            Q *= (t_ / np.maximum(Q.sum(0), 1e-9))[None]
            Q /= Q.sum(1, keepdims=True)
        out[ii] = Q
    return out


def learned_targets(Po, Bo, y, sbj, fold, Pt, Bt, tsbj):
    X, key = profile_features([Po, Bo], sbj, TRAIN_SETS)
    true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    fold_of = {int(s): int(fold[sbj == s][0]) for s in np.unique(sbj)}
    kf = np.array([fold_of[int(s)] for s, _ in key])
    cnt = np.zeros(len(true))
    for f in np.unique(kf):
        cnt[kf == f] = fit_counts(X[kf != f], true[kf != f], X[kf == f])
    Xt, kt = profile_features([Pt, Bt], tsbj, {})
    ct = fit_counts(X, true, Xt)
    print(f'count error (OOF) {np.abs(cnt - true).mean():.2f} tiles vs fixed 97: {np.abs(97 - true).mean():.2f}')
    print('test predicted mean count per subject:',
          {int(s): round(float(ct[kt[:, 0] == s].mean()), 1) for s in np.unique(tsbj)})
    return count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dir:tag with oof_<tag>.npz / test_<tag>.npz (key P_graph)')
    ap.add_argument('--b', default='artifacts/wear26-base/art')
    ap.add_argument('--out')
    ap.add_argument('--save')
    ap.add_argument('--queue')
    ap.add_argument('--priority', type=float, default=6)
    a = ap.parse_args()
    bo = np.load(os.path.join(HERE, a.b, 'oof_B.npz'))
    bt = np.load(os.path.join(HERE, a.b, 'test_B.npz'))
    y, sbj, fold = bo['y'], bo['sbj'], bo['fold']
    Bo, Bt = np.exp(bo['logp_b'].astype(np.float64)), np.exp(bt['logp_b'].astype(np.float64))
    d_, tag = a.src.rsplit(':', 1)
    Po, _ = load_P(os.path.join(HERE, d_, f'oof_{tag}.npz'))
    Pt, zt = load_P(os.path.join(HERE, d_, f'test_{tag}.npz'))
    tsbj, tid = zt['sbj'], zt['id']
    print(f'input {a.src}: soft fixed-97 OOF {macro_f1(y, finish(Po, sbj, TRAIN_SETS, 0.5, 97, 3)):.4f}, '
          f'full fixed-97 {macro_f1(y, finish(Po, sbj, TRAIN_SETS, 0.5, 97, 50)):.4f}')
    tg, tgt = learned_targets(Po, Bo, y, sbj, fold, Pt, Bt, tsbj)
    best = None
    for it in (3, 10, 50):
        f = macro_f1(y, calibrate_targets(Po, sbj, tg, iters=it).argmax(1))
        print(f'learned counts, Sinkhorn iters {it}: OOF {f:.4f}')
        if best is None or f > best[0]:
            best = (f, it)
    f, it = best
    if a.save:
        sd, stag = a.save.rsplit(':', 1)
        sd = os.path.join(HERE, sd)
        os.makedirs(sd, exist_ok=True)
        np.savez_compressed(os.path.join(sd, f'oof_{stag}.npz'), P_graph=calibrate_targets(Po, sbj, tg, iters=it), y=y,
                            sbj=sbj)
        np.savez_compressed(os.path.join(sd, f'test_{stag}.npz'), P_graph=calibrate_targets(Pt, tsbj, tgt, iters=it),
                            id=tid, sbj=tsbj, oof_f1=f)
    if a.out:
        lab = calibrate_targets(Pt, tsbj, tgt, iters=it).argmax(1)
        order = np.argsort(tid)
        out = os.path.join(HERE, a.out)
        with open(out, 'w', encoding='utf-8', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['id', 'target_feature'])
            for i in order:
                wr.writerow([int(tid[i]), int(lab[i])])
        print('wrote', out, 'counts', np.bincount(lab, minlength=N_CLS).tolist())
        if a.queue:
            subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'wear.py'), 'add', out,
                            f'{a.queue} (OOF {f:.4f})', '--hypothesis',
                            f'learned per-(subject, exercise) counts instead of 97 on {a.src}, Sinkhorn {it} iters',
                            '--priority', str(a.priority)], check=False)


if __name__ == '__main__':
    main()
