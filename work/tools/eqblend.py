"""Equal-weight geometric blend of every saved run whose own OOF F1 (soft calibration) is >= --min-f1.
Cross-validation showed tuned blend weights do not generalise (CV-tuned 0.9228 < equal 0.9234), so this is the
default way to combine runs.

  python tools/eqblend.py --min-f1 0.915 [--exclude wear26-l5,wear26-v4] [--out subs/x.csv] [--save dir:tag]
                          [--queue "msg"] [--priority 5]
"""
import argparse, csv, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from blend import finish, macro_f1, load_P  # noqa: E402
from blend_search import discover  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-f1', type=float, default=0.915)
    ap.add_argument('--exclude', default='')
    ap.add_argument('--out')
    ap.add_argument('--save')
    ap.add_argument('--queue')
    ap.add_argument('--priority', type=float, default=5)
    a = ap.parse_args()
    b = np.load(os.path.join(HERE, 'artifacts', 'wear26-base', 'art', 'oof_B.npz'))
    y, sbj = b['y'], b['sbj']
    ex = [x for x in a.exclude.split(',') if x]
    L, Lt, used = [], [], []
    tsbj = tid = None
    for r in discover():
        if any(x in r for x in ex):
            continue
        d, tag = r.rsplit(':', 1)
        po, _ = load_P(os.path.join(HERE, d, f'oof_{tag}.npz'))
        if po.shape[0] != len(y):
            continue
        f = macro_f1(y, finish(po, sbj, {0: 2, 14: 2}, 0.5, 97, 3))
        if f < a.min_f1:
            continue
        pt, zt = load_P(os.path.join(HERE, d, f'test_{tag}.npz'))
        L.append(np.log(np.clip(po, 1e-12, None)))
        Lt.append(np.log(np.clip(pt, 1e-12, None)))
        tsbj, tid = zt['sbj'], zt['id']
        used.append((r, f))
    for r, f in sorted(used, key=lambda x: -x[1]):
        print(f'{f:.4f}  {r}')

    def blend(LL):
        x = sum(LL) / len(LL)
        p = np.exp(x - x.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    po, pt = blend(L), blend(Lt)
    f = macro_f1(y, finish(po, sbj, {0: 2, 14: 2}, 0.5, 97, 3))
    print(f'equal-weight blend of {len(L)} runs (min single F1 {a.min_f1}): OOF {f:.4f}')
    if a.save:
        sd, stag = a.save.rsplit(':', 1)
        sd = os.path.join(HERE, sd)
        os.makedirs(sd, exist_ok=True)
        np.savez_compressed(os.path.join(sd, f'oof_{stag}.npz'), P_graph=po.astype(np.float32), y=y, sbj=sbj)
        np.savez_compressed(os.path.join(sd, f'test_{stag}.npz'), P_graph=pt.astype(np.float32), id=tid, sbj=tsbj,
                            oof_f1=f)
    if a.out:
        lab = finish(pt, tsbj, {}, 0.5, 97, 3)
        order = np.argsort(tid)
        out = os.path.join(HERE, a.out)
        with open(out, 'w', encoding='utf-8', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['id', 'target_feature'])
            for i in order:
                wr.writerow([int(tid[i]), int(lab[i])])
        print('wrote', out)
        if a.queue:
            subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'wear.py'), 'add', out,
                            f'{a.queue} (OOF {f:.4f})', '--hypothesis',
                            f'equal-weight geo blend of {len(L)} runs with single OOF >= {a.min_f1}',
                            '--priority', str(a.priority)], check=False)


if __name__ == '__main__':
    main()
