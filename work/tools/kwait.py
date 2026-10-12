"""Wait for a wear26-* kernel to finish, then download its output to work/artifacts/<slug>/ and show the log tail.

  python tools/kwait.py wear26-base [--interval 60] [--timeout-hours 12] [--no-download] [--tail 60]
  python tools/kwait.py wear26-base --status        # one status check, no waiting
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = 'santanubanerjee9'
ENV = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')


def kg(*args, timeout=900):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True, env=ENV, encoding='utf-8', errors='replace',
                       timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def status(ref):
    _, out = kg('kernels', 'status', ref)
    m = re.search(r'status "?([A-Za-z_.]+)"?', out)
    return (m.group(1) if m else out).upper(), out


def log_text(path):
    try:
        raw = open(path, encoding='utf-8', errors='replace').read()
        items = json.loads(raw)
        return ''.join(i.get('data', '') for i in items)
    except Exception:
        return open(path, encoding='utf-8', errors='replace').read() if os.path.exists(path) else ''


def download(slug, tail=60, ref=None, pattern=None, dst=None):
    dst = dst or os.path.join(HERE, 'artifacts', slug)
    os.makedirs(dst, exist_ok=True)
    extra = ['--file-pattern', pattern] if pattern else []
    code, out = kg('kernels', 'output', ref or f'{USER}/{slug}', '-p', dst, '-o', *extra)
    print(out[-800:])
    logs = [f for f in os.listdir(dst) if f.endswith('.log')]
    for f in logs:
        txt = log_text(os.path.join(dst, f))
        open(os.path.join(dst, 'log.txt'), 'w', encoding='utf-8').write(txt)
        lines = [l for l in txt.splitlines() if 'Warning' not in l and 'warn(' not in l]
        print('\n'.join(lines[-tail:]))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slug')
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--timeout-hours', type=float, default=12)
    ap.add_argument('--no-download', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--tail', type=int, default=60)
    ap.add_argument('--pattern', help='regex of output files to download (public notebooks: skip big files)')
    ap.add_argument('--dst', help='download folder (default artifacts/<slug>)')
    a = ap.parse_args()
    ref = a.slug if '/' in a.slug else f'{USER}/{a.slug}'
    slug = ref.split('/')[1]
    if a.status:
        print(status(ref)[1])
        return
    t0 = time.time()
    while True:
        st, raw = status(ref)
        if any(x in st for x in ('COMPLETE', 'ERROR', 'CANCEL')):
            print(f'{slug}: {st} after {(time.time() - t0) / 60:.1f} min', flush=True)
            break
        if time.time() - t0 > a.timeout_hours * 3600:
            print(f'{slug}: timeout, last status {st}')
            sys.exit(2)
        time.sleep(a.interval)
    if not a.no_download:
        download(slug, a.tail, ref, a.pattern, a.dst and os.path.join(HERE, a.dst))
    sys.exit(0 if 'COMPLETE' in st else 1)


if __name__ == '__main__':
    main()
