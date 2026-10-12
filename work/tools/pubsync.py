"""Watch the competition's public notebooks that save final_probabilities.npz (OOF + test probabilities, the
'Learned Links + Counts' family) and turn every new completed run into blendable arrays with tools/pubconv.py.

  python tools/pubsync.py           # check, download new completed runs, convert, print a summary
  python tools/pubsync.py --list    # only show the watched notebooks and their state

Watched: the refs in WATCH plus any other notebook of the competition by the same authors. A run is new when the
listing's lastRunTime differs from artifacts/pub-<tag>/RUN.json. Outputs are only downloaded when the status is
COMPLETE (a running notebook serves no files). Converted arrays: artifacts/pub-<tag>/art/{oof,test}_<tag>.npz.
"""
import argparse, csv, hashlib, io, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = '3rd-wear-dataset-challenge-hasca-2026'
ENV = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
WATCH = {'goodpjw2008/wear-hasca-learned-links-counts-lb-0-92942': 'gp', 'honghanhhh/wear-hasca-lb-0-9': 'hh',
         'sujanmajhisuzan/wear-hasca-sota-bagged-boundary-refiner': 'sj'}
PATTERN = r'final_probabilities|submission\.csv|\.log$'


def kg(*args, timeout=900):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True, env=ENV, encoding='utf-8', errors='replace',
                       timeout=timeout)
    return (r.stdout + r.stderr).strip()


def listing():
    out = kg('kernels', 'list', '--competition', COMP, '--sort-by', 'dateRun', '--page-size', '100', '--csv')
    rows = [r for r in csv.DictReader(io.StringIO('\n'.join(l for l in out.splitlines() if l.strip())))]
    return {r['ref']: r for r in rows if r.get('ref')}


def tag_of(ref):
    if ref in WATCH:
        return WATCH[ref]
    return 'x' + hashlib.md5(ref.encode()).hexdigest()[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--ref', default='datasets/wear26-art:eq17', help='our blend used by pubconv for the temp choice')
    a = ap.parse_args()
    rows = listing()
    authors = {r.split('/')[0] for r in WATCH}
    refs = sorted(set(WATCH) | {r for r in rows if r.split('/')[0] in authors})
    for ref in refs:
        tag = tag_of(ref)
        dst = os.path.join(HERE, 'artifacts', f'pub-{tag}')
        state_path = os.path.join(dst, 'RUN.json')
        state = json.load(open(state_path, encoding='utf-8')) if os.path.exists(state_path) else {}
        run = rows.get(ref, {}).get('lastRunTime', '?')
        if state.get('lastRunTime') == run:
            print(f'{ref} [{tag}]: run {run} already converted (own OOF {state.get("oof", "?")})')
            continue
        st = kg('kernels', 'status', ref)
        m = re.search(r'status "?([A-Za-z_.]+)"?', st)
        st = (m.group(1) if m else st).upper()
        print(f'{ref} [{tag}]: new run {run}, status {st}')
        if a.list or 'COMPLETE' not in st:
            continue
        os.makedirs(dst, exist_ok=True)
        print(kg('kernels', 'output', ref, '-p', dst, '-o', '--file-pattern', PATTERN)[-400:])
        if not os.path.exists(os.path.join(dst, 'final_probabilities.npz')):
            print('  no final_probabilities.npz in its output - not blendable')
            json.dump({'ref': ref, 'lastRunTime': run, 'oof': None}, open(state_path, 'w', encoding='utf-8'))
            continue
        r = subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'pubconv.py'), os.path.relpath(dst, HERE),
                            '--tag', tag, '--ref', a.ref], capture_output=True, text=True, env=ENV, encoding='utf-8',
                           errors='replace')
        print('  ' + (r.stdout + r.stderr).strip().replace('\n', '\n  '))
        m = re.search(r'own soft OOF ([0-9.]+)', r.stdout)
        if r.returncode == 0 and m:
            json.dump({'ref': ref, 'lastRunTime': run, 'oof': float(m.group(1))}, open(state_path, 'w', encoding='utf-8'))
        else:
            print('  conversion failed (rows may not line up with ours) - left unrecorded')


if __name__ == '__main__':
    main()
