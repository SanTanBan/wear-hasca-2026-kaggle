"""Build and push a private Kaggle script kernel that runs one entry module of src/wear with a JSON config.

  python tools/kpush.py --slug wear26-base --entry base --config '{"threads": 4}' [--gpu | --tpu]
                        [--datasets jiweiliu/wear-timeline-graph-models] [--kernels wear26-base] [--no-push]

The whole src/wear package is embedded in the script (base64 zip), so no code dataset has to be versioned.
Kernel slugs must start with "wear26-" (other competitions share this Kaggle account - never touch their kernels).
Pushing retries while Kaggle reports the account's session limit. Prints and records the pushed version.
"""
import argparse, base64, io, json, os, re, subprocess, sys, time, zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = 'santanubanerjee9'
COMP = '3rd-wear-dataset-challenge-hasca-2026'
ENV = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
TPU_SHAPE = 'Tpu1VmV38'
GPU_SHAPE = 'NvidiaTeslaT4'

BOOT = '''import base64, io, json, os, sys, zipfile, time, traceback
CODE = "{code}"
CONFIG = json.loads(r"""{config}""")
dst = "/tmp/wear_src"
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall(dst)
sys.path.insert(0, dst)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings; warnings.filterwarnings("ignore")
print("config:", json.dumps(CONFIG), flush=True)
import importlib
mod = importlib.import_module("wear." + CONFIG["entry"])
try:
    mod.main(CONFIG)
except Exception:
    traceback.print_exc()
    open("/kaggle/working/FAILED.txt", "w").write(traceback.format_exc())
    raise
'''


def package_b64():
    buf = io.BytesIO()
    src = os.path.join(HERE, 'src')
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(os.path.join(src, 'wear')):
            if '__pycache__' in root:
                continue
            for f in files:
                if f.endswith(('.py', '.json')):
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, src))
    return base64.b64encode(buf.getvalue()).decode()


def build(slug, entry, config, gpu=False, tpu=False, datasets=(), kernels=(), internet=False):
    assert slug.startswith('wear26-'), 'kernel slugs must start with wear26-'
    d = os.path.join(HERE, 'kernels', slug)
    os.makedirs(d, exist_ok=True)
    config = dict(config, entry=entry)
    code = BOOT.format(code=package_b64(), config=json.dumps(config, indent=1))
    open(os.path.join(d, f'{slug}.py'), 'w', encoding='utf-8').write(code)
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": bool(gpu and not tpu), "enable_tpu": bool(tpu),
            "enable_internet": bool(internet), "dataset_sources": list(datasets),
            "kernel_sources": [k if '/' in k else f'{USER}/{k}' for k in kernels],
            "competition_sources": [COMP], "model_sources": []}
    if tpu:
        meta["machine_shape"] = TPU_SHAPE
    elif gpu:
        meta["machine_shape"] = GPU_SHAPE
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
    return d


def quota_ok(kind, hours):
    """True when the job's estimated hours fit into 50% of the remaining Kaggle quota (user rule)"""
    r = subprocess.run(['kaggle', 'quota'], capture_output=True, text=True, env=ENV, encoding='utf-8', errors='replace')
    m = re.search(rf'^{kind}\s+\S+\s+([\d.]+)h', r.stdout, re.M)
    if not m:
        print('quota unreadable:', r.stdout[-200:], flush=True)
        return False
    rem = float(m.group(1))
    ok = hours <= 0.5 * rem
    print(f'{kind} remaining {rem:.2f}h; job estimate {hours:.2f}h -> {"OK" if ok else "too large, waiting"}', flush=True)
    return ok


def push(d, gpu=False, tpu=False, retry_hours=6, interval=300, est_hours=None):
    t0 = time.time()
    while True:
        if (gpu or tpu) and est_hours is not None and not quota_ok('TPU' if tpu else 'GPU', est_hours):
            if time.time() - t0 > retry_hours * 3600:
                raise SystemExit('quota never allowed the job')
            time.sleep(interval)
            continue
        args = ['kaggle', 'kernels', 'push', '-p', d]
        if tpu:
            args += ['--accelerator', TPU_SHAPE]
        elif gpu:
            args += ['--accelerator', GPU_SHAPE]
        r = subprocess.run(args, capture_output=True, text=True, env=ENV, encoding='utf-8', errors='replace')
        out = (r.stdout + r.stderr).strip()
        m = re.search(r'version (\d+) successfully', out)
        if m:
            print(out[-300:])
            return int(m.group(1))
        if ('Maximum' in out or 'maximum' in out) and 'session' in out.lower() and time.time() - t0 < retry_hours * 3600:
            print(f'session limit reached; retrying in {interval // 60} min', flush=True)
            time.sleep(interval)
            continue
        raise SystemExit(f'push failed: {out[-800:]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', required=True)
    ap.add_argument('--entry', required=True)
    ap.add_argument('--config', default='{}', help='inline JSON or path to a JSON file')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--tpu', action='store_true')
    ap.add_argument('--internet', action='store_true')
    ap.add_argument('--datasets', default='')
    ap.add_argument('--kernels', default='')
    ap.add_argument('--no-push', action='store_true')
    ap.add_argument('--est-hours', type=float, help='GPU/TPU runtime estimate; push only if <= 50% of remaining quota')
    ap.add_argument('--retry-hours', type=float, default=6)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding='utf-8')) if os.path.exists(a.config) else json.loads(a.config)
    ds = [x for x in a.datasets.split(',') if x]
    ks = [x for x in a.kernels.split(',') if x]
    d = build(a.slug, a.entry, cfg, a.gpu, a.tpu, ds, ks, a.internet)
    print('built', d, f'({os.path.getsize(os.path.join(d, a.slug + ".py")) / 1024:.0f} KB)')
    if (a.gpu or a.tpu) and a.est_hours is None:
        raise SystemExit('--est-hours is required for GPU/TPU kernels (50%-of-remaining-quota rule)')
    if not a.no_push:
        v = push(d, a.gpu, a.tpu, retry_hours=a.retry_hours, est_hours=a.est_hours)
        log = os.path.join(HERE, 'daily', 'KERNELS.csv')
        new = not os.path.exists(log)
        with open(log, 'a', encoding='utf-8') as f:
            if new:
                f.write('pushed_utc,slug,version,entry,accel,config\n')
            accel = 'tpu' if a.tpu else ('gpu' if a.gpu else 'cpu')
            f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())},{a.slug},{v},{a.entry},{accel},'
                    f'"{json.dumps(cfg).replace(chr(34), chr(39))}"\n')
        print('pushed version', v)


if __name__ == '__main__':
    main()
