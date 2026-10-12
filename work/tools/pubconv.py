"""Convert a public notebook's final_probabilities.npz (oof = calibrated probabilities on the 69,326 test-like OOF tiles,
test = the same for the test tiles in the notebook's test order) into our oof_<tag>.npz / test_<tag>.npz (key P_graph),
after checking that the rows line up with ours: OOF argmax against our labels, test argmax against the notebook's own
submission.csv. The probabilities are already count-calibrated (sharp), so they are flattened with a power 1/temp
chosen on the OOF blend with --ref (one scalar, so little room to overfit).

  python tools/pubconv.py artifacts/pub-hh --tag hh [--ref datasets/wear26-art:eq17]
"""
import argparse, csv, os, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from blend import finish, macro_f1, load_P  # noqa: E402

SETS = {0: 2, 14: 2}


def norm(p):
    p = np.clip(p, 1e-12, None)
    return p / p.sum(1, keepdims=True)


def geo(Ps, w=None):
    w = w or [1 / len(Ps)] * len(Ps)
    x = sum(wi * np.log(np.clip(p, 1e-12, None)) for wi, p in zip(w, Ps))
    p = np.exp(x - x.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='folder with final_probabilities.npz and submission.csv')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ref', default='datasets/wear26-art:eq17', help='our best blend dir:tag for the temp choice')
    ap.add_argument('--temps', default='1,2,3,4,6')
    ap.add_argument('--swap', action='store_true', help="carry the notebook's refiner flips into its probabilities")
    a = ap.parse_args()
    src = os.path.join(HERE, a.src)
    b = np.load(os.path.join(HERE, 'artifacts', 'wear26-base', 'art', 'oof_B.npz'))
    y, sbj = b['y'], b['sbj']
    rd, rtag = a.ref.rsplit(':', 1)
    Ro, _ = load_P(os.path.join(HERE, rd, f'oof_{rtag}.npz'))
    Rt, zt = load_P(os.path.join(HERE, rd, f'test_{rtag}.npz'))
    tid, tsbj = zt['id'], zt['sbj']
    z = np.load(os.path.join(src, 'final_probabilities.npz'))
    Po, Pt = norm(z['oof'].astype(np.float64)), norm(z['test'].astype(np.float64))
    print('shapes', Po.shape, Pt.shape, 'ours', len(y), len(tid))
    assert Po.shape == (len(y), 19) and Pt.shape == (len(tid), 19), 'row counts differ'
    if a.swap and 'oof_labels' in z.files and 'test_labels' in z.files:
        # the notebook's boundary refiner flips labels after these probabilities were saved: swap the two entries of
        # every flipped tile so the blend keeps the flip (OOF: eq17 + gp 0.5 -> 0.9286 plain, 0.9293 swapped; but
        # public 0.92704 plain vs 0.92646 swapped, so plain is the default)
        for P_, R_ in ((Po, z['oof_labels']), (Pt, z['test_labels'])):
            a_ = P_.argmax(1)
            ii = np.flatnonzero(R_ != a_)
            qa, qr = P_[ii, a_[ii]].copy(), P_[ii, R_[ii]].copy()
            P_[ii, a_[ii]], P_[ii, R_[ii]] = qr, qa
            print(f'refiner flips carried into the probabilities: {len(ii)}')
    f_arg = macro_f1(y, Po.argmax(1))
    print(f'OOF alignment: argmax macro F1 vs our labels {f_arg:.4f} (misaligned rows would give ~0.05)')
    sub = {}
    with open(os.path.join(src, 'submission.csv'), encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            sub[int(row['id'])] = int(row['target_feature'])
    agree = np.mean([sub[int(i)] == l for i, l in zip(tid, Pt.argmax(1))])
    print(f'test alignment: argmax agrees with its submission.csv on {agree:.4f} of tiles')
    assert f_arg > 0.85 and agree > 0.95, 'rows do not line up with ours'
    lab_ref = finish(Rt, tsbj, {}, 0.5, 97, 3)
    print(f'test agreement with our {a.ref}: {np.mean(lab_ref == Pt.argmax(1)):.4f}')
    f_ref = macro_f1(y, finish(Ro, sbj, SETS, 0.5, 97, 3))
    print(f'ours alone OOF {f_ref:.4f}; theirs alone (soft-97) {macro_f1(y, finish(Po, sbj, SETS, 0.5, 97, 3)):.4f}')
    best = None
    for T in [float(x) for x in a.temps.split(',')]:
        Qo = norm(Po ** (1 / T))
        for w in (0.3, 0.5):
            f = macro_f1(y, finish(geo([Ro, Qo], [1 - w, w]), sbj, SETS, 0.5, 97, 3))
            print(f'  temp {T:g}, weight {w}: blend OOF {f:.4f}')
            if best is None or f > best[0]:
                best = (f, T, w)
    f, T, w = best
    print(f'chosen temp {T:g} (best blend OOF {f:.4f} at weight {w})')
    out = os.path.join(src, 'art')
    os.makedirs(out, exist_ok=True)
    Qo, Qt = norm(Po ** (1 / T)), norm(Pt ** (1 / T))
    fo = macro_f1(y, finish(Qo, sbj, SETS, 0.5, 97, 3))
    np.savez_compressed(os.path.join(out, f'oof_{a.tag}.npz'), P_graph=Qo.astype(np.float32), y=y, sbj=sbj)
    np.savez_compressed(os.path.join(out, f'test_{a.tag}.npz'), P_graph=Qt.astype(np.float32), id=tid, sbj=tsbj,
                        oof_f1=fo)
    print('saved', out, f'{a.tag} (own soft OOF {fo:.4f})')


if __name__ == '__main__':
    main()
