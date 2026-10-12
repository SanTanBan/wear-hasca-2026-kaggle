"""Coordinate-ascent search of geometric blend weights over saved OOF probability runs (soft calibration), then
optionally write the test submission and queue it.

  python tools/blend_search.py --runs artifacts/wear26-l4/art:L3_cs_w0.5 artifacts/wear26-v2/art:v2_cs0.4 ...
                               [--all] [--min-gain 0.0005] [--out subs/sNNN_x.csv --queue "msg"] [--save dir:tag]

--all: use every artifacts/*/art/oof_*.npz that has a matching test_*.npz (except diagnostics).
Writes/queues only if the tuned blend beats the best OOF recorded in daily/BEST_OOF.txt by --min-gain.
"""
import argparse, glob, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from blend import finish, macro_f1, load_P  # noqa: E402

SKIP = ('oof_A', 'oof_B', 'oof_gnn', 'oof_edgew', 'oof_clusters', 'oof_selftrain', 'oof_giter')


def discover(public=False):
    """our runs under artifacts/*/art; public-notebook runs (artifacts/pub-*, see pubsync.py) only with public=True"""
    runs = []
    for f in sorted(glob.glob(os.path.join(HERE, 'artifacts', '*', 'art', 'oof_*.npz'))):
        base = os.path.basename(f)
        if base.startswith(SKIP):
            continue
        if not public and os.path.basename(os.path.dirname(os.path.dirname(f))).startswith('pub-'):
            continue
        tag = base[4:-4]
        if os.path.exists(os.path.join(os.path.dirname(f), f'test_{tag}.npz')):
            rel = os.path.relpath(os.path.dirname(f), HERE).replace('\\', '/')
            runs.append(f'{rel}:{tag}')
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--min-gain', type=float, default=0.0005)
    ap.add_argument('--out')
    ap.add_argument('--queue')
    ap.add_argument('--save')
    ap.add_argument('--priority', type=float, default=5)
    ap.add_argument('--cv', action='store_true', help='also report a fold-cross-validated estimate of the tuned blend')
    ap.add_argument('--exclude', default='', help='comma-separated substrings of runs to leave out')
    a = ap.parse_args()
    runs = discover() if a.all else a.runs
    ex = [x for x in a.exclude.split(',') if x]
    runs = [r for r in runs if not any(x in r for x in ex)]
    b = np.load(os.path.join(HERE, 'artifacts', 'wear26-base', 'art', 'oof_B.npz'))
    y, sbj = b['y'], b['sbj']
    L, Lt, ok_runs, single = [], [], [], {}
    tsbj = tid = None
    for r in runs:
        d, tag = r.rsplit(':', 1)
        try:
            po, _ = load_P(os.path.join(HERE, d, f'oof_{tag}.npz'))
            pt, zt = load_P(os.path.join(HERE, d, f'test_{tag}.npz'))
        except Exception as e:
            print('skip', r, e)
            continue
        if po.shape[0] != len(y):
            continue
        f = macro_f1(y, finish(po, sbj, {0: 2, 14: 2}, 0.5, 97, 3))
        single[r] = f
        if f < 0.9:          # weak runs only add noise
            continue
        L.append(np.log(np.clip(po, 1e-12, None)))
        Lt.append(np.log(np.clip(pt, 1e-12, None)))
        tsbj, tid = zt['sbj'], zt['id']
        ok_runs.append(r)
    for r in sorted(single, key=lambda k: -single[k]):
        print(f'{single[r]:.4f}  {r}')

    def blend(w, LL):
        w = np.array(w, float) / np.sum(w)
        x = sum(wi * li for wi, li in zip(w, LL))
        p = np.exp(x - x.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def score(w):
        return macro_f1(y, finish(blend(w, L), sbj, {0: 2, 14: 2}, 0.5, 97, 3))

    def tune(mask=None):
        def sc(w):
            p = blend(w, [l[mask] for l in L] if mask is not None else L)
            yy, ss = (y[mask], sbj[mask]) if mask is not None else (y, sbj)
            return macro_f1(yy, finish(p, ss, {0: 2, 14: 2}, 0.5, 97, 3))
        w = [1.0] * len(L)
        best = sc(w)
        for sweep in range(2):
            for i in range(len(L)):
                for v in (0.0, 0.5, 1.5, 2.0, 3.0):
                    w2 = list(w)
                    w2[i] = v
                    if sum(w2) == 0:
                        continue
                    s = sc(w2)
                    if s > best + 1e-5:
                        best, w = s, w2
        return w, best

    print(f'equal weights over {len(L)} runs: {score([1.0] * len(L)):.4f}')
    if a.cv:      # honest estimate of the tuned blend: weights tuned on 4 subject folds, applied to the 5th
        fold = b['fold']
        pred = np.zeros(len(y), int)
        for f in range(5):
            wf, _ = tune(fold != f)
            m = fold == f
            pred[m] = finish(blend(wf, [l[m] for l in L]), sbj[m], {0: 2, 14: 2}, 0.5, 97, 3)
        print(f'cross-validated tuned-blend OOF {macro_f1(y, pred):.4f}')
    w, best = tune()
    ws = np.array(w) / sum(w)
    print(f'tuned OOF {best:.4f}')
    for r, wi in zip(ok_runs, ws):
        if wi > 0:
            print(f'   {wi:.3f}  {r}')
    rec = os.path.join(HERE, 'daily', 'BEST_OOF.txt')
    prev = float(open(rec).read().split()[0]) if os.path.exists(rec) else 0.0
    print(f'previous best queued/submitted OOF {prev:.4f}')
    if best < prev + a.min_gain:
        print('no meaningful gain; nothing written')
        return
    pt = blend(w, Lt)
    if a.save:
        sd, stag = a.save.rsplit(':', 1)
        sd = os.path.join(HERE, sd)
        os.makedirs(sd, exist_ok=True)
        np.savez_compressed(os.path.join(sd, f'oof_{stag}.npz'), P_graph=blend(w, L).astype(np.float32), y=y, sbj=sbj)
        np.savez_compressed(os.path.join(sd, f'test_{stag}.npz'), P_graph=pt.astype(np.float32), id=tid, sbj=tsbj,
                            oof_f1=best)
    if a.out:
        import csv
        lab = finish(pt, tsbj, {}, 0.5, 97, 3)
        order = np.argsort(tid)
        out = os.path.join(HERE, a.out)
        with open(out, 'w', encoding='utf-8', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['id', 'target_feature'])
            for i in order:
                wr.writerow([int(tid[i]), int(lab[i])])
        print('wrote', out)
        with open(rec, 'w') as fh:
            fh.write(f'{best:.5f} {a.out}\n')
        if a.queue:
            desc = ' + '.join(f'{r.split("/")[1]}:{r.rsplit(":", 1)[1]}@{wi:.2f}' for r, wi in zip(ok_runs, ws) if wi > 0)
            subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'wear.py'), 'add', out,
                            f'{a.queue} (OOF {best:.4f})', '--hypothesis', 'tuned geo blend: ' + desc[:400],
                            '--priority', str(a.priority)], check=False)


if __name__ == '__main__':
    main()
