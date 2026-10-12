# WEAR daily routine (runbook)

Competition: `3rd-wear-dataset-challenge-hasca-2026` (CSV upload, macro F1). 10 submissions per UTC day
(reset 00:00 UTC = 05:30 IST). Final deadline **2026-10-12 21:59 UTC** (2026-10-13 03:29 IST).
Work dir: `D:\GitHub_Ideas\For_Profit\Kaggle\3rd WEAR Dataset Challenge @HASCA 2026\work` (all paths below relative to it).
Shell: PowerShell, set `$env:PYTHONUTF8='1'; $env:PYTHONWARNINGS='ignore'` before python/kaggle commands.

## 0. Guard
- If the UTC date is after 2026-10-12: submit nothing; make sure the two final submissions are selected (see
  STRATEGY.md "Final selection"), disable the scheduled task `wear-daily-submissions`, tell the user, stop.

## 1. Read state (in this order)
1. `daily/FEEDBACK.md` — user feedback, highest priority. Apply anything marked Open.
2. `daily/STRATEGY.md`, `daily/BACKLOG.md`, newest `daily/reports/*.md`.
3. `python tools/wear.py sync` then `python tools/wear.py status` (remaining slots, quota, queue, ledger tail).
4. `python tools/wear.py leaderboard --top 10`.

## 2. Make sure there are as many ready candidates as remaining slots
- Candidates live in `daily/QUEUE.json` (status `ready`, highest `priority` first). Each is a validated CSV in `subs/`.

### 2a. Public-notebook blends first (biggest lever since 2026-10-04)
Blending an independently trained public pipeline gave +0.0046 public (our eq17 alone 0.92247; + honghanhhh
probabilities at 0.3 -> 0.92604, at 0.5 -> 0.92705) although that notebook alone scores only 0.91082.
1. `python tools/pubsync.py` — downloads every new COMPLETE run of the watched public notebooks
   (goodpjw2008 "Learned Links + Counts", honghanhhh "WEAR@HASCA LB 0.9", and other notebooks by those authors)
   and converts it to `artifacts/pub-<tag>/art/` after an alignment check. A RUNNING notebook serves no files: skip it.
2. If a run was newly converted:
   `python tools/pubblend.py --ours datasets/wear26-art:eq17 --w-ours 0.2 --only gp --save datasets/wear26-art:pub<NNN>`
   (best recipe 2026-10-05: ours 0.2 + goodpjw 0.8, then the refiner -> 0.93149; refined weight curve 0.5 0.92937,
   0.65 0.92977, 0.8 0.93149, 0.9 0.92965, goodpjw alone 0.92912. honghanhhh hurts: leave it out. Plain blends without
   the refiner are ~0.002 worse, so always refine before queueing).
   Then refine it on Kaggle CPU (3 min): copy `datasets/wear26-art/{oof,test}_pub<NNN>.npz` into
   `D:\DevCache\tmp\claude\wear2026\art_upload_v2`, `kaggle datasets version -p D:\DevCache\tmp\claude\wear2026\art_upload_v2
   -m "pub<NNN>"`, wait until `kaggle datasets status santanubanerjee9/wear26-art` says ready, copy configs/ref1.json to
   configs/ref<NNN>.json with "probs": "pub<NNN>", "tag": "ref<NNN>", then `python tools/kpush.py --slug wear26-ref<NNN>
   --entry refine --config configs/ref<NNN>.json --datasets santanubanerjee9/wear26-art --kernels
   wear26-base,wear26-l3,wear26-l4,wear26-l6,wear26-l4s` and `python tools/kwait.py wear26-ref<NNN>`; queue
   `artifacts/wear26-ref<NNN>/submission_ref<NNN>_thr0.6.csv` (log.txt shows the OOF gain; refiner gave +0.001-0.002).
3. Our own independent run of the goodpjw2008 notebook with other seeds lives in `kernels/wear26-gpjw`
   (WEAR_SEED=1, WEAR_DUMP=1). If `python tools/kwait.py wear26-gpjw --status` says it does not exist / never ran:
   check `kaggle quota` (needs >= 6 h GPU left, the run takes ~2-3 h on T4 x2), then ONE push attempt:
   `kaggle kernels push -p kernels/wear26-gpjw --accelerator NvidiaTeslaT4`. "Maximum batch GPU session count"
   means other projects use both sessions: do not wait, try on the next run. When it is COMPLETE:
   `python tools/kwait.py wear26-gpjw`, then `python tools/pubconv.py artifacts/wear26-gpjw --tag gpjw1`, copy the
   `art` folder to `artifacts/pub-gpjw1/art` so pubblend picks it up, and build the pubblend candidates again.
- Finished kernels that are not harvested yet: `python tools/kwait.py <slug> --status`; if COMPLETE,
  `python tools/kwait.py <slug>` downloads to `artifacts/<slug>/`; queue its `submission*.csv` files with
  `python tools/wear.py add <csv> "<msg>" --hypothesis "<why>" --priority <p>`.
### 2b. Our own runs
- New probability runs from harvested kernels (e.g. wear26-s11 = S5 with pseudo-label adapted T, outputs
  oof/test_s11_w*.npz):
  prefer `python tools/eqblend.py --min-f1 0.915 --exclude "wear26-l5,wear26-v4,wear26-v5,wear26-s6"
  --out subs/s<NNN>_eq.csv --save datasets/wear26-art:eq<NNN>` (equal weights; tuned weights proved not to generalise;
  second-level runs looked optimistic on the LB). Queue it only if its OOF beats daily/BEST_OOF.txt by >= 0.001
  (the public LB cannot resolve smaller differences), then update BEST_OOF.txt and STRATEGY.md, and rebuild the
  public blends of 2a with `--ours` set to the new blend.
  `tools/blend_search.py --cv` gives an honest estimate if you want to compare tuned vs equal weights.
- If still short: `python tools/variants.py --src <best test_*.npz from STRATEGY.md> --n <missing> --queue` builds
  calibration variants of the best probabilities and queues them (each message carries its OOF estimate).
- Never queue a file that fails `python tools/wear.py check <csv>`.

## 3. Submit
- `python tools/wear.py submit-ready` until remaining is 0. Do not finish with unused slots unless every path
  failed; then say why with the failing command and its output.
- User rule (2026-10-05): split the day's submissions between low-risk candidates (refined blends near the best
  recipe, seeds, small threshold/weight changes) and high-risk ones (bigger departures: new pipelines, new decoders,
  research bets). Prefix high-risk ledger messages with "HIGH-RISK:" and report the split in the brief.
- GPU/TPU: the budget is 50% of what `kaggle quota` shows as remaining at that moment (re-evaluate every time).

## 4. Report (≤ 200 words) to `daily/reports/<UTC date>.md` and brief the user (≤ 8 lines)
- yesterday's public scores and what they taught; today's submissions and hypotheses; our best vs LB #1;
  next steps.
- **Guidance request** (1–3 concrete questions with options) when: best public score has not improved for 2+ days,
  or we are ≥ 0.01 behind #1 for 3+ days, or a costly decision is pending. The user explicitly wants to be
  consulted when we are stuck relative to the leaders.

## 5. Notify
- One short PushNotification line (load the tool with ToolSearch if needed).

## Hard rules
- Kaggle GPU/TPU: never ask how much; use at most 50% of the remaining `kaggle quota` at any time
  (other competitions share the account). CPU kernels are free.
- Only touch kernels/datasets whose slug starts with `wear26-`. Never stop/edit other competitions' kernels.
- Keep kernels private. No hand-labelling of test data, no external data that breaks the rules.
- Never write anything to C: (scratch: `D:\DevCache\tmp\claude\wear2026`).
