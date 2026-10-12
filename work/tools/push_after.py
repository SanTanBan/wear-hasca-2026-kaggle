"""Wait until another wear26 kernel is no longer QUEUED/RUNNING, then run tools/kpush.py with the given arguments.
Keeps at most one of our GPU kernels on the account at a time (the account has 2 GPU slots shared with other
competitions).

  python tools/push_after.py --after wear26-wtrain-g1 --max-wait-hours 4 -- --slug wear26-ctx-c1 --entry ctxwin ...
"""
import argparse, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')


def status(slug):
    r = subprocess.run(['kaggle', 'kernels', 'status', f'santanubanerjee9/{slug}'], capture_output=True, text=True,
                       env=ENV, encoding='utf-8', errors='replace')
    m = re.search(r'status "?([A-Za-z_.]+)"?', r.stdout + r.stderr)
    return (m.group(1) if m else (r.stdout + r.stderr)).upper()


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit('usage: push_after.py --after SLUG [--max-wait-hours H] -- <kpush args>')
    i = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--after', required=True)
    ap.add_argument('--max-wait-hours', type=float, default=4)
    ap.add_argument('--interval', type=int, default=300)
    a = ap.parse_args(argv[:i])
    t0 = time.time()
    while True:
        st = status(a.after)
        if not any(x in st for x in ('RUNNING', 'QUEUED')):
            print(f'{a.after}: {st} -> pushing', flush=True)
            break
        if time.time() - t0 > a.max_wait_hours * 3600:
            raise SystemExit(f'{a.after} still {st} after {a.max_wait_hours} h; not pushing')
        print(f'{a.after}: {st}; waiting', flush=True)
        time.sleep(a.interval)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'tools', 'kpush.py')] + argv[i + 1:], env=ENV)
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()
