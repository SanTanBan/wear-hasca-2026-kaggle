"""Local blending of saved pre-calibration probabilities (P_graph) from several runs, evaluated on the OOF tiles.

  python tools/blend.py --runs artifacts/wear26-base/art:B artifacts/wear26-l3/art:L3_cs_w1.0 ... [--weights 1 1]
                        [--T 0.5] [--per-ex 97] [--out subs/blend_x.csv] [--queue "message"]

Each run "<dir>:<tag>" needs <dir>/oof_<tag>.npz and <dir>/test_<tag>.npz with key P_graph (or P_graph3).
OOF labels/subjects come from the base artifacts (artifacts/wear26-base/art/oof_B.npz).
"""
import argparse, csv, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CLS = 19
TRAIN_SETS = {0: 2, 14: 2}


def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def calibrate(P, sbj, sets, per_ex, null_min=0.05, iters=50):
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


ITERS = 3   # Sinkhorn iterations: 3 beats full convergence (50) by +0.005 OOF on the base pipeline


def finish(P, sbj, sets, T=0.5, per_ex=97, iters=None):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, per_ex, iters=iters or ITERS).argmax(1)


def load_P(path):
    z = np.load(path, allow_pickle=True)
    for k in ("P_graph", "P_graph3"):
        if k in z.files:
            return z[k].astype(np.float64), z
    raise KeyError(f"no P_graph in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--weights', nargs='*', type=float)
    ap.add_argument('--T', type=float, default=0.5)
    ap.add_argument('--per-ex', type=float, default=97)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--geo', action='store_true', help='geometric instead of arithmetic mean')
    ap.add_argument('--out')
    ap.add_argument('--save', help='write the blended probabilities to <dir>/oof_<tag>.npz + test_<tag>.npz, as dir:tag')
    ap.add_argument('--queue')
    ap.add_argument('--base', default='artifacts/wear26-base/art/oof_B.npz')
    a = ap.parse_args()
    b = np.load(os.path.join(HERE, a.base), allow_pickle=True)
    y, sbj = b['y'], b['sbj']
    w = a.weights or [1.0] * len(a.runs)
    Po, Pt = [], []
    tsbj = tid = None
    for r in a.runs:
        d, tag = r.rsplit(':', 1)
        po, _ = load_P(os.path.join(HERE, d, f'oof_{tag}.npz'))
        pt, zt = load_P(os.path.join(HERE, d, f'test_{tag}.npz'))
        tsbj, tid = zt['sbj'], zt['id']
        print(f'{r:60s} OOF macro F1 {macro_f1(y, finish(po, sbj, TRAIN_SETS, a.T, a.per_ex, a.iters)):.4f}')
        Po.append(po)
        Pt.append(pt)
    ws = np.array(w) / np.sum(w)
    if a.geo:
        po = np.exp(sum(wi * np.log(np.clip(p, 1e-12, None)) for wi, p in zip(ws, Po)))
        pt = np.exp(sum(wi * np.log(np.clip(p, 1e-12, None)) for wi, p in zip(ws, Pt)))
    else:
        po = sum(wi * p for wi, p in zip(ws, Po))
        pt = sum(wi * p for wi, p in zip(ws, Pt))
    po /= po.sum(1, keepdims=True)
    pt /= pt.sum(1, keepdims=True)
    f = macro_f1(y, finish(po, sbj, TRAIN_SETS, a.T, a.per_ex, a.iters))
    print(f'BLEND weights {np.round(ws, 3).tolist()} T={a.T} per_ex={a.per_ex} iters={a.iters}: OOF macro F1 {f:.4f}')
    if a.save:
        sd, stag = a.save.rsplit(':', 1)
        sd = os.path.join(HERE, sd)
        os.makedirs(sd, exist_ok=True)
        np.savez_compressed(os.path.join(sd, f'oof_{stag}.npz'), P_graph=po.astype(np.float32), y=y, sbj=sbj)
        np.savez_compressed(os.path.join(sd, f'test_{stag}.npz'), P_graph=pt.astype(np.float32), id=tid, sbj=tsbj,
                            oof_f1=f)
        print('saved blend arrays to', sd, stag)
    if a.out:
        lab = finish(pt, tsbj, {}, a.T, a.per_ex, a.iters)
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
                            f'{a.queue} (OOF {f:.4f})', '--hypothesis', 'blend of ' + ' + '.join(a.runs)], check=False)


if __name__ == '__main__':
    main()
