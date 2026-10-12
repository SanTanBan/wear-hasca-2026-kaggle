"""Submission bookkeeping for the 3rd WEAR Dataset Challenge (CSV upload, 10 submissions per UTC day).

Run from anywhere (paths are resolved relative to work/):
  python tools/wear.py status                 # limits, quota, queue, last ledger rows
  python tools/wear.py sync                   # pull public scores into daily/LEDGER.csv
  python tools/wear.py leaderboard [--top 20] # public LB top + our rank
  python tools/wear.py add <csv> "<message>" [--hypothesis "..."] [--front] [--priority N]
  python tools/wear.py submit-ready [--max N] [--dry-run]   # submit queued CSVs up to today's remaining slots
  python tools/wear.py submit <csv> "<message>" [--hypothesis "..."]   # submit one file now (also logged)
  python tools/wear.py check <csv>            # validate a submission file

Queue: daily/QUEUE.json -> list of {id, file, message, hypothesis, priority, status: ready|submitted|failed|skipped}
Ledger: daily/LEDGER.csv (one row per submission we made).
"""
import argparse, csv, datetime as dt, json, os, re, subprocess, sys, time

COMP = '3rd-wear-dataset-challenge-hasca-2026'
USER = 'santanubanerjee9'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE = os.path.join(HERE, 'daily', 'QUEUE.json')
LEDGER = os.path.join(HERE, 'daily', 'LEDGER.csv')
SAMPLE_IDS = 12234
LEDGER_COLS = ['date_utc', 'submitted_at_utc', 'queue_id', 'file', 'message', 'hypothesis', 'public_lb', 'status',
               'submission_ref']
ENV = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')


def kg(*args, check=True, timeout=600):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True, env=ENV, timeout=timeout,
                       encoding='utf-8', errors='replace')
    out = '\n'.join(l for l in (r.stdout + '\n' + r.stderr).splitlines()
                    if 'Warning' not in l and 'warnings.warn' not in l)
    if check and r.returncode != 0:
        raise RuntimeError(f'kaggle {" ".join(args)} failed ({r.returncode}):\n{out}')
    return out.strip()


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, HERE) if p.startswith(HERE) else p


def absf(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def load_queue():
    return json.load(open(QUEUE, encoding='utf-8')) if os.path.exists(QUEUE) else []


def save_queue(q):
    json.dump(q, open(QUEUE, 'w', encoding='utf-8'), indent=1)


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_ledger(rows):
    with open(LEDGER, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in LEDGER_COLS})


def limits():
    out = kg('competitions', 'submission-limits', COMP, check=False)
    used = re.search(r'Submissions today:\s*(\d+)', out)
    rem = re.search(r'Remaining today:\s*(\d+)', out)
    return (int(used.group(1)) if used else -1), (int(rem.group(1)) if rem else 0)


def check_file(path):
    """returns an error string or '' if the file is a valid submission"""
    path = absf(path)
    if not os.path.exists(path):
        return f'missing file {path}'
    with open(path, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if rows[0] != ['id', 'target_feature']:
        return f'bad header {rows[0]}'
    body = rows[1:]
    if len(body) != SAMPLE_IDS:
        return f'{len(body)} rows, expected {SAMPLE_IDS}'
    ids = sorted(int(r[0]) for r in body)
    if ids != list(range(SAMPLE_IDS)):
        return 'ids are not 0..N-1'
    try:
        labs = [int(float(r[1])) for r in body]
    except ValueError:
        return 'non-integer labels'
    if min(labs) < 0 or max(labs) > 18:
        return 'labels outside 0..18'
    return ''


def submissions_json():
    out = kg('competitions', 'submissions', COMP, '--format', 'json', check=False)
    m = re.search(r'\[.*\]', out, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


def cmd_status(a):
    used, rem = limits()
    print(f'UTC now {utc_now():%Y-%m-%d %H:%M}  submissions today used={used} remaining={rem}')
    print(kg('quota', check=False))
    q = load_queue()
    ready = sorted([e for e in q if e.get('status') == 'ready'], key=lambda e: -float(e.get('priority', 0)))
    print(f'queue: {len(ready)} ready / {len(q)} total')
    for e in ready[:15]:
        print(f"  [{e['id']}] p={e.get('priority', 0)} {e['file']} :: {e['message'][:100]}")
    rows = read_ledger()
    print(f'ledger: {len(rows)} rows; last 10:')
    for r in rows[-10:]:
        print(f"  {r['date_utc']} lb={r['public_lb'] or '?':8s} {r['queue_id']:5s} :: {r['message'][:100]}")


def cmd_sync(a):
    subs = submissions_json()
    rows = read_ledger()
    by_ref = {r.get('submission_ref'): r for r in rows if r.get('submission_ref')}
    changed = 0
    for s in subs:
        ref = str(s.get('ref') or s.get('id') or '')
        desc = (s.get('description') or '').strip()
        score = s.get('publicScore') or s.get('public_score') or ''
        status = str(s.get('status') or '')
        r = by_ref.get(ref)
        if r is None:
            for rr in rows:
                if not rr.get('submission_ref') and rr.get('message', '').strip() == desc:
                    r = rr
                    r['submission_ref'] = ref
                    break
        if r is None:  # a submission made outside this tool: record it
            r = dict(date_utc=str(s.get('date', ''))[:10], submitted_at_utc=str(s.get('date', '')), queue_id='ext',
                     file=str(s.get('fileName', '')), message=desc, hypothesis='', public_lb='', status=status,
                     submission_ref=ref)
            rows.append(r)
            changed += 1
        if score not in ('', None) and r.get('public_lb') != str(score):
            r['public_lb'] = str(score)
            changed += 1
        if status and r.get('status') != status:
            r['status'] = status
            changed += 1
    rows.sort(key=lambda r: r.get('submitted_at_utc', ''))
    write_ledger(rows)
    print(f'synced {len(subs)} submissions; {changed} ledger fields updated')
    best = max((float(r['public_lb']) for r in rows if r.get('public_lb')), default=None)
    print('best public LB so far:', best)
    for s in subs[:12]:
        print(f"  {s.get('date')} {s.get('status')} pub={s.get('publicScore')} :: {(s.get('description') or '')[:100]}")


def cmd_leaderboard(a):
    out = kg('competitions', 'leaderboard', COMP, '--show', check=False)
    lines = [l for l in out.splitlines() if not l.startswith('Next Page')]
    print('\n'.join(lines[:a.top + 2]))
    try:
        dl = os.path.join(HERE, 'daily', 'lb')
        os.makedirs(dl, exist_ok=True)
        kg('competitions', 'leaderboard', COMP, '--download', '-p', dl, check=False)
    except Exception:
        pass


def _new_id(q, rows):
    nums = [int(re.sub(r'\D', '', e['id']) or 0) for e in q] + \
           [int(re.sub(r'\D', '', r.get('queue_id', '')) or 0) for r in rows if r.get('queue_id', '').startswith('q')]
    return f'q{1 + max(nums + [0]):03d}'


def cmd_add(a):
    err = check_file(a.file)
    if err:
        raise SystemExit(f'not queued: {err}')
    q = load_queue()
    e = dict(id=_new_id(q, read_ledger()), file=rel(a.file), message=a.message, hypothesis=a.hypothesis or '',
             priority=a.priority, status='ready', added=utc_now().isoformat(timespec='minutes'))
    if a.front:
        e['priority'] = max([float(x.get('priority', 0)) for x in q] + [0]) + 1
    q.append(e)
    save_queue(q)
    print('queued', e)


def _submit(path, msg, hyp, qid, rows):
    out = kg('competitions', 'submit', COMP, '-f', absf(path), '-m', msg[:250])
    print('   ', out[-200:])
    now = utc_now()
    rows.append(dict(date_utc=f'{now:%Y-%m-%d}', submitted_at_utc=now.isoformat(timespec='seconds'), queue_id=qid,
                     file=rel(path), message=msg[:250], hypothesis=hyp, public_lb='', status='submitted',
                     submission_ref=''))


def cmd_submit_ready(a):
    used, rem = limits()
    cap = rem if a.max is None else min(rem, a.max)
    print(f'remaining today {rem}; will submit up to {cap}')
    if cap <= 0:
        return
    q = load_queue()
    rows = read_ledger()
    done = 0
    for e in sorted([e for e in q if e.get('status') == 'ready'], key=lambda e: -float(e.get('priority', 0))):
        if done >= cap:
            break
        err = check_file(e['file'])
        if err:
            print(f"  skip {e['id']}: {err}")
            e['status'] = 'failed'
            e['error'] = err
            continue
        msg = f"{e['id']} {e['message']}"
        print(f"  SUBMIT {e['id']} {e['file']} :: {msg[:120]}")
        if a.dry_run:
            continue
        try:
            _submit(e['file'], msg, e.get('hypothesis', ''), e['id'], rows)
            e['status'] = 'submitted'
            e['submitted_at'] = utc_now().isoformat(timespec='minutes')
            done += 1
        except Exception as ex:
            print('    FAILED:', ex)
            e['status'] = 'failed'
            e['error'] = str(ex)[-500:]
        save_queue(q)
        write_ledger(rows)
        time.sleep(2)
    print(f'submitted {done}')


def cmd_submit(a):
    err = check_file(a.file)
    if err:
        raise SystemExit(err)
    rows = read_ledger()
    q = load_queue()
    qid = _new_id(q, rows)
    _submit(a.file, f'{qid} {a.message}', a.hypothesis or '', qid, rows)
    write_ledger(rows)


def cmd_check(a):
    print(check_file(a.file) or 'OK')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('status')
    sp.add_parser('sync')
    p = sp.add_parser('leaderboard')
    p.add_argument('--top', type=int, default=20)
    p = sp.add_parser('add')
    p.add_argument('file')
    p.add_argument('message')
    p.add_argument('--hypothesis')
    p.add_argument('--front', action='store_true')
    p.add_argument('--priority', type=float, default=0)
    p = sp.add_parser('submit-ready')
    p.add_argument('--max', type=int)
    p.add_argument('--dry-run', action='store_true')
    p = sp.add_parser('submit')
    p.add_argument('file')
    p.add_argument('message')
    p.add_argument('--hypothesis')
    p = sp.add_parser('check')
    p.add_argument('file')
    a = ap.parse_args()
    {'status': cmd_status, 'sync': cmd_sync, 'leaderboard': cmd_leaderboard, 'add': cmd_add,
     'submit-ready': cmd_submit_ready, 'submit': cmd_submit, 'check': cmd_check}[a.cmd](a)


if __name__ == '__main__':
    main()
