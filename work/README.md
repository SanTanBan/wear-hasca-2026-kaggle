# WEAR @HASCA 2026 — work directory

## Layout
- `src/wear/` — pipeline package (embedded into every Kaggle kernel by `tools/kpush.py`)
  - `data.py` raw -> compact arrays, test-like OOF tiles (22 train subjects, 5 subject folds)
  - `window_pre.py` window classifier OOF/test inference from jiweiliu/wear-timeline-graph-models (5-fold models)
  - `l0.py`, `tab.py`, `l2.py`, `graph.py` — Briano's public pipeline, re-implemented with honest nested OOF
  - `base.py` entry: window -> L0 -> tabular -> L2 -> graph; saves `art/oof_{A,B}.npz`, `art/test_{A,B}.npz`
  - `nlink.py` entry: contrastive successor bi-encoder (ours); saves `art/nlink_<tag>.npz`
  - `l3.py` entry: L2 features + nlink candidates/features -> L3 links -> graph (+ chain stacker); submissions
  - `chainstack.py` chain-context LightGBM stacker (ours); `decode.py` entry: oracle/Viterbi/calibration analysis
- `tools/wear.py` submissions + ledger + queue; `tools/kpush.py` / `tools/kwait.py` kernels; `tools/variants.py`
- `configs/*.json` kernel configs; `kernels/<slug>/` generated kernel folders; `artifacts/<slug>/` downloaded outputs
- `subs/` every CSV we submitted or queued; `daily/` runbook, strategy, backlog, feedback, ledger, queue, reports

## Kernels (all private, slug prefix `wear26-`)
| slug | entry | inputs | outputs |
|---|---|---|---|
| wear26-base | base | competition + jiweiliu/wear-timeline-graph-models | art/oof_B.npz, test_B.npz, submission.csv |
| wear26-nlink{,2,3} | nlink | competition | art/nlink_v{1,2,3}.npz |
| wear26-decode | decode | wear26-base | report + submission_cs_w*.csv |
| wear26-l3 | l3 | wear26-base + wear26-nlinkX | submission_L3*.csv |
