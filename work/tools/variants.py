"""Post-processing variants from saved test probabilities (cheap, local): sharpen temperature, Sinkhorn iterations,
per-exercise count and null floor of the per-subject count calibration. Skips label vectors identical to anything
already in subs/. If the matching OOF file (oof_<tag>.npz next to the test file) exists, each variant's OOF F1 is
computed and put into the queue message.

  python tools/variants.py --src datasets/wear26-art/test_blend7.npz --n 3 [--queue] [--priority 0]
"""
import argparse, csv, glob, hashlib, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CLS = 19
TRAIN_SETS = {0: 2, 14: 2}
GRID = [  # (T, per_ex, null_min, iters) in priority order; (0.5, 97, 0.05, 3) is the reference decode
    (0.5, 97, 0.05, 3), (0.7, 97, 0.05, 3), (0.5, 97, 0.05, 5), (0.5, 94, 0.05, 3), (0.5, 100, 0.05, 3),
    (0.5, 97, 0.05, 2), (0.4, 97, 0.05, 3), (0.7, 97, 0.05, 5), (1.0, 97, 0.05, 3), (0.5, 97, 0.10, 3),
]


def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def calibrate(P, sbj, sets, per_ex, null_min, iters):
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


def decode(P, sbj, sets, T, per_ex, null_min, iters):
    Q = np.clip(P, 1e-12, None) ** (1 / T)
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, per_ex, null_min, iters).argmax(1)


def label_hash(labels):
    return hashlib.sha1(np.asarray(labels, np.int8).tobytes()).hexdigest()[:16]


def known_hashes():
    hs = set()
    for f in glob.glob(os.path.join(HERE, 'subs', '*.csv')):
        with open(f, encoding='utf-8') as fh:
            rows = list(csv.reader(fh))[1:]
        lab = np.zeros(len(rows), np.int64)
        for r in rows:
            lab[int(r[0])] = int(float(r[1]))
        hs.add(label_hash(lab))
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--key', default='P_graph')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--queue', action='store_true')
    ap.add_argument('--priority', type=float, default=0)
    a = ap.parse_args()
    path = a.src if os.path.isabs(a.src) else os.path.join(HERE, a.src)
    z = np.load(path, allow_pickle=True)
    P, sbj, ids = z[a.key].astype(np.float64), z['sbj'], z['id']
    oof_path = os.path.join(os.path.dirname(path), os.path.basename(path).replace('test_', 'oof_', 1))
    oof = np.load(oof_path, allow_pickle=True) if os.path.exists(oof_path) else None
    tag = a.tag or os.path.basename(path).replace('test_', '').replace('.npz', '')
    seen = known_hashes()
    made = 0
    for T, pe, nm, it in GRID:
        if made >= a.n:
            break
        lab = decode(P, sbj, {}, T, pe, nm, it)
        order = np.argsort(ids)
        h = label_hash(lab[order])
        if h in seen:
            continue
        seen.add(h)
        f_oof = ''
        if oof is not None:
            f_oof = f' (OOF {macro_f1(oof["y"], decode(oof[a.key].astype(np.float64), oof["sbj"], TRAIN_SETS, T, pe, nm, it)):.4f})'
        name = f"subs/v_{tag}_T{T}_pe{pe}_nm{nm}_it{it}.csv"
        with open(os.path.join(HERE, name), 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['id', 'target_feature'])
            for i in order:
                w.writerow([int(ids[i]), int(lab[i])])
        made += 1
        print('wrote', name, f_oof)
        if a.queue:
            subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'wear.py'), 'add', os.path.join(HERE, name),
                            f'{tag} decode T={T} per_ex={pe} null_min={nm} sinkhorn_iters={it}{f_oof}',
                            '--hypothesis', 'calibration variant of the best probabilities', '--priority',
                            str(a.priority)], check=False)
    print(f'{made} new variants')


if __name__ == '__main__':
    main()
