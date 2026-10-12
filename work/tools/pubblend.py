"""Blend our best saved probabilities with every converted public-notebook run (artifacts/pub-*/art, see pubsync.py):
geometric mean, our share --w-ours, the public runs share the rest equally; soft Sinkhorn decode (T 0.5, 97, 3 iters).
Prints the OOF of a few alternative shares for the record; writes/queues the --w-ours blend.

  python tools/pubblend.py --ours datasets/wear26-art:eq17 [--w-ours 0.5] [--only gp,hh] [--out subs/x.csv]
                           [--save dir:tag] [--queue "msg"] [--priority 6]

Evidence (2026-10-04): eq17 alone 0.92247 public; + hh 0.3 -> 0.92604; + hh 0.5 -> 0.92705 (OOF preferred 0.3).
"""
import argparse, csv, glob, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from blend import finish, macro_f1, load_P  # noqa: E402

SETS = {0: 2, 14: 2}


def geo(Ps, w):
    x = sum(wi * np.log(np.clip(p, 1e-12, None)) for wi, p in zip(w, Ps))
    p = np.exp(x - x.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', default='datasets/wear26-art:eq17')
    ap.add_argument('--w-ours', type=float, default=0.5)
    ap.add_argument('--only', default='', help='comma list of public tags to use (default: all converted)')
    ap.add_argument('--out')
    ap.add_argument('--save')
    ap.add_argument('--queue')
    ap.add_argument('--priority', type=float, default=6)
    a = ap.parse_args()
    b = np.load(os.path.join(HERE, 'artifacts', 'wear26-base', 'art', 'oof_B.npz'))
    y, sbj = b['y'], b['sbj']
    d_, t_ = a.ours.rsplit(':', 1)
    Oo, _ = load_P(os.path.join(HERE, d_, f'oof_{t_}.npz'))
    Ot, zt = load_P(os.path.join(HERE, d_, f'test_{t_}.npz'))
    tid, tsbj = zt['id'], zt['sbj']
    pubs = []
    for f in sorted(glob.glob(os.path.join(HERE, 'artifacts', 'pub-*', 'art', 'oof_*.npz'))):
        tag = os.path.basename(f)[4:-4]
        if a.only and tag not in a.only.split(','):
            continue
        po, _ = load_P(f)
        pt, z2 = load_P(f.replace('oof_', 'test_'))
        assert (z2['id'] == tid).all(), f'{tag}: test order differs from ours'
        pubs.append((tag, po, pt))
        print(f'{tag}: own OOF {macro_f1(y, finish(po, sbj, SETS, 0.5, 97, 3)):.4f}')
    assert pubs, 'no converted public runs (run tools/pubsync.py first)'
    print(f'ours {a.ours}: OOF {macro_f1(y, finish(Oo, sbj, SETS, 0.5, 97, 3)):.4f}')

    def mix(w0, split):
        n = len(pubs)
        w = [w0] + [(1 - w0) / n] * n
        Ps = [Oo] + [p[1] for p in pubs] if split == 'oof' else [Ot] + [p[2] for p in pubs]
        return geo(Ps, w)
    for w0 in sorted({0.4, 0.5, 0.6, 0.7, a.w_ours}):
        print(f'  ours {w0:.2f}, publics {(1 - w0) / len(pubs):.3f} each: OOF '
              f'{macro_f1(y, finish(mix(w0, "oof"), sbj, SETS, 0.5, 97, 3)):.4f}')
    po, pt = mix(a.w_ours, 'oof'), mix(a.w_ours, 'test')
    f = macro_f1(y, finish(po, sbj, SETS, 0.5, 97, 3))
    names = '+'.join(p[0] for p in pubs)
    print(f'chosen: ours {a.w_ours} + {names}: OOF {f:.4f}')
    if a.save:
        sd, stag = a.save.rsplit(':', 1)
        sd = os.path.join(HERE, sd)
        os.makedirs(sd, exist_ok=True)
        np.savez_compressed(os.path.join(sd, f'oof_{stag}.npz'), P_graph=po.astype(np.float32), y=y, sbj=sbj)
        np.savez_compressed(os.path.join(sd, f'test_{stag}.npz'), P_graph=pt.astype(np.float32), id=tid, sbj=tsbj,
                            oof_f1=f)
    if a.out:
        lab = finish(pt, tsbj, {}, 0.5, 97, 3)
        out = os.path.join(HERE, a.out)
        with open(out, 'w', encoding='utf-8', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['id', 'target_feature'])
            for i in np.argsort(tid):
                wr.writerow([int(tid[i]), int(lab[i])])
        print('wrote', out, 'counts', np.bincount(lab, minlength=19).tolist())
        if a.queue:
            subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'wear.py'), 'add', out,
                            f'{a.queue} (OOF {f:.4f})', '--hypothesis',
                            f'geo blend ours {a.w_ours} ({a.ours}) + public runs {names} equal share',
                            '--priority', str(a.priority)], check=False)


if __name__ == '__main__':
    main()
