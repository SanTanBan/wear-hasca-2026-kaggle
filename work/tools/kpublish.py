"""Make wear26-* Kaggle kernels public: pull the latest code + metadata and save a new version with is_private=False
and kernel_execution_type=QUICK_SAVE.

CAUTION (observed 2026-10-12): Kaggle still starts a run for a QUICK_SAVE version of a script kernel, on the account's
shared CPU sessions (5 max) - GPU/TPU kernels are switched to CPU with a 60 s cap below so they cannot use quota. When
all sessions are busy the save is refused ("Maximum batch CPU session count") but the visibility change still applies,
keeping the original run. The clean way to publish without a re-run is Kaggle's web UI (Share -> Public).

  python tools/kpublish.py wear26-smoke [wear26-base ...]     # or --all (every kernel of ours whose slug starts wear26-)
"""
import argparse, csv, io, json, os, subprocess, sys

USER = 'santanubanerjee9'
TMP = r'D:\DevCache\tmp\claude\wear2026\publish'
ENV = dict(os.environ, PYTHONWARNINGS='ignore', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')


def my_kernels():
    r = subprocess.run(['kaggle', 'kernels', 'list', '--mine', '--search', 'wear26', '--page-size', '200', '--csv'],
                       capture_output=True, text=True, env=ENV, encoding='utf-8', errors='replace')
    rows = csv.DictReader(io.StringIO('\n'.join(l for l in r.stdout.splitlines() if l.strip())))
    return sorted({x['ref'].split('/')[1] for x in rows if x.get('ref', '').startswith(USER + '/wear26')})


def publish(slug):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    assert slug.startswith('wear26'), slug
    d = os.path.join(TMP, slug)
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(['kaggle', 'kernels', 'pull', f'{USER}/{slug}', '-p', d, '-m'], capture_output=True, text=True,
                       env=ENV, encoding='utf-8', errors='replace')
    meta = json.load(open(os.path.join(d, 'kernel-metadata.json'), encoding='utf-8'))
    code = open(os.path.join(d, meta['code_file']), encoding='utf-8').read()
    if meta.get('kernel_type') == 'notebook':
        nb = json.loads(code)
        for c in nb.get('cells', []):
            if isinstance(c.get('source'), list):
                c['source'] = ''.join(c['source'])
        code = json.dumps(nb)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiSaveKernelRequest()
        req.slug = f'{USER}/{slug}'
        req.new_title = meta.get('title')
        req.text = code
        req.language = meta.get('language')
        req.kernel_type = meta.get('kernel_type')
        req.is_private = False
        req.enable_gpu = bool(meta.get('enable_gpu', False))
        req.enable_tpu = bool(meta.get('enable_tpu', False))
        req.enable_internet = bool(meta.get('enable_internet', False))
        req.dataset_data_sources = meta.get('dataset_sources', [])
        req.competition_data_sources = meta.get('competition_sources', [])
        req.kernel_data_sources = meta.get('kernel_sources', [])
        req.model_data_sources = meta.get('model_sources', [])
        req.kernel_execution_type = KernelExecutionType.QUICK_SAVE
        if req.enable_gpu or req.enable_tpu:
            # safety net: if Kaggle ever ran a quick-save version, it must not spend GPU/TPU quota
            req.enable_gpu = req.enable_tpu = False
            req.session_timeout_seconds = 60
        res = kaggle.kernels.kernels_api_client.save_kernel(req)
    err = getattr(res, 'error', None)
    print(f'{slug}: {"ERROR " + str(err) if err else "public, version " + str(res.version_number)}', flush=True)
    return not err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slugs', nargs='*')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    slugs = my_kernels() if a.all else a.slugs
    print(len(slugs), 'kernels', flush=True)
    bad = [s for s in slugs if not publish(s)]
    print('failed:', bad)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
