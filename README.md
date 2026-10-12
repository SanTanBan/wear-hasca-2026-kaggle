# 3rd WEAR Dataset Challenge @ HASCA 2026 — team "InSociEUP"

Our solution for the Kaggle community competition
[3rd WEAR Dataset Challenge @ HASCA 2026](https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026)
(macro F1 over 19 classes: null + 18 outdoor workout activities). Built over 2026-10-02 .. 10-12 by
Santanu Banerjee with Claude (Claude Code), entering late, without prize eligibility, playing for rank.

**Best public score 0.93371 (#11 of the public LB on the last day; #1 0.94538)**, up from 0.90770 for the public
starter notebook. How we got there, and why, is told in [CONVERSATION_SUMMARY.md](CONVERSATION_SUMMARY.md).

## The problem in one paragraph
The test set holds 12,234 shuffled, non-overlapping 1-second tiles from 4 unseen subjects. Each tile has one randomly
chosen limb's accelerometer signal (50 x 3 at 50 Hz) and 15 VideoMAEv2 frame features from a head-mounted camera. The
tiles cover whole recording sessions, and every subject performs each of the 18 exercises once (~97 s, often split
into sets by rests = null). Recovering the time order of the shuffled tiles (who follows whom) is the key to the task.

## Solution in layers
1. **Public pipeline, re-implemented with honest nested OOF** (`work/src/wear/`): window classifier (pretrained 5-fold
   models) -> L0 links -> tabular experts -> L2 successor links (ridge boundary predictors + LightGBM + Hungarian
   matching) -> graph propagation -> per-subject Sinkhorn count calibration. Validation = the 22 training subjects cut
   into test-like tiles, 5 subject folds (OOF 0.9050, public 0.9061).
2. **Our additions** (OOF 0.9256): soft Sinkhorn (3 iterations), graph-config ensembles, a contrastive successor
   bi-encoder (`nlink.py`) as link features (L3/L4/L6), a multi-limb chain stacker (`chainstack.py`, `s4.py`),
   pseudo-label adapted experts, equal-weight geometric blends of ~26 first-level runs.
3. **Blending independently trained public pipelines** (`tools/pubsync.py`, `pubconv.py`, `pubblend.py`): public
   notebooks that save out-of-fold + test probabilities in the same row order can be blended and validated honestly.
4. **Boundary refiner along the link chains** (`src/wear/refine.py`, idea from goodpjw2008's notebook): a LightGBM
   re-judges the tiles around every label change along 5 link sets; +0.002 public on every blend.
5. **More seeds of the strongest public pipeline** (our GPU re-runs with other seeds), averaged with our blend and
   refined: **0.93371**.

| Milestone | OOF | Public |
|---|---|---|
| Public starter output | – | 0.90770 |
| Re-implemented public pipeline | 0.9050 | 0.90610 |
| + soft Sinkhorn, graph sweep | 0.9124 | 0.91209 |
| + successor bi-encoder links, chain stacker, blends | 0.9235 | 0.92314 |
| + independently trained public pipeline (goodpjw2008) | 0.9277 | 0.92762 |
| + our boundary refiner, public-heavy weight 0.8 | 0.9285 | 0.93149 |
| + 4 public-pipeline variants averaged | 0.9306 | 0.93326 |
| + 4 seeds of the ranked-links pipeline, ours 0.4 (q057) | 0.9313 | **0.93371** |

## Repository layout
- `work/src/wear/` — the pipeline package (embedded into every Kaggle kernel by `work/tools/kpush.py`)
- `work/tools/` — submission ledger/queue (`wear.py`), kernel push/wait (`kpush.py`, `kwait.py`, `kpublish.py`),
  blending (`blend.py`, `eqblend.py`, `blend_search.py`, `pubblend.py`), public runs (`pubsync.py`, `pubconv.py`),
  count prior (`counts.py`), decode variants (`variants.py`)
- `work/configs/` — the JSON config of every kernel run; `work/daily/KERNELS.csv` lists them
- `work/daily/` — runbook of the daily submission routine, strategy log, dev log, backlog, user feedback, daily reports,
  `LEDGER.csv` (every submission with its hypothesis, OOF and public score), `QUEUE.json`
- `work/subs/` — every CSV we submitted or queued; `work/artifacts/wear26-*/log.txt` — logs of our Kaggle kernels
- `work/research/` — side experiments (session recovery)

Not included: the competition data (Kaggle license), the large prediction arrays (`work/datasets`,
`work/artifacts/*/art`: >1 GB, derived from the competition data), copies of other people's notebooks (linked below),
and saved Kaggle web pages.

## Kaggle notebooks
**[NOTEBOOKS.md](NOTEBOOKS.md) lists all 68 of our Kaggle kernels** (owner
[santanubanerjee9](https://www.kaggle.com/santanubanerjee9/code), slugs `wear26-*`; each one embeds `work/src/wear` and
runs one entry module with its config) with what each does and whether it is public yet.

> **Caveat:** some notebooks are still private and show "We can't find that page". If you need one that is
> inaccessible, open an issue in this repository or message santanubanerjee9 on Kaggle, and it will be made public.

Key ones:
- [wear26-base](https://www.kaggle.com/code/santanubanerjee9/wear26-base) — re-implemented public pipeline with nested OOF
- [wear26-l4](https://www.kaggle.com/code/santanubanerjee9/wear26-l4) — L4 links with two successor bi-encoders
  ([wear26-nlink2](https://www.kaggle.com/code/santanubanerjee9/wear26-nlink2),
  [wear26-nlink3](https://www.kaggle.com/code/santanubanerjee9/wear26-nlink3))
- [wear26-s11](https://www.kaggle.com/code/santanubanerjee9/wear26-s11) — multi-limb stacker with pseudo-label adapted T
- [wear26-ref22](https://www.kaggle.com/code/santanubanerjee9/wear26-ref22) — boundary refiner that produced the best
  public submission (q057); [wear26-ref14](https://www.kaggle.com/code/santanubanerjee9/wear26-ref14) — best-OOF blend (q043)
- [wear26-p2s3](https://www.kaggle.com/code/santanubanerjee9/wear26-p2s3),
  [wear26-p2s6](https://www.kaggle.com/code/santanubanerjee9/wear26-p2s6),
  [wear26-p2s9](https://www.kaggle.com/code/santanubanerjee9/wear26-p2s9) — goodpjw2008's ranked-links notebook
  re-run with other seeds (GPU); [wear26-gpjw](https://www.kaggle.com/code/santanubanerjee9/wear26-gpjw) — its
  "Learned Links + Counts" notebook with seed 1

## Credits (public work we built on)
- Briano (woominyo), [LB 0.890 Timeline + Graph](https://www.kaggle.com/code/woominyo/lb-0-890-wear-hasca-2026-timeline-graph) — the pipeline we re-implemented
- Jiwei Liu, [public fast GPU inference](https://www.kaggle.com/code/jiweiliu/public-fast-gpu-inference) and the
  pretrained window models (dataset `jiweiliu/wear-timeline-graph-models`)
- goodpjw2008, [Learned Links + Counts](https://www.kaggle.com/code/goodpjw2008/wear-hasca-learned-links-counts-lb-0-92942)
  (boundary refiner idea, learned counts, adapted T) and
  [Part 2: Ranked Links](https://www.kaggle.com/code/goodpjw2008/wear-hasca-part-2-ranked-links-lb-0-92990)
- honghanhhh, [WEAR@HASCA LB 0.9](https://www.kaggle.com/code/honghanhhh/wear-hasca-lb-0-9);
  sujanmajhisuzan, [SOTA bagged boundary refiner](https://www.kaggle.com/code/sujanmajhisuzan/wear-hasca-sota-bagged-boundary-refiner)

## Running it
Everything heavy runs as private/public Kaggle kernels built from this repo:
`python work/tools/kpush.py --slug wear26-<name> --entry <module> --config work/configs/<file>.json [--gpu]
[--datasets ...] [--kernels ...]`, then `python work/tools/kwait.py wear26-<name>`. Blending, refining and the
submission queue are described step by step in `work/daily/DAILY_RUNBOOK.md`.
