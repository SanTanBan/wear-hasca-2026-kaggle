# How this solution came about — a compressed record of the working conversation

A summary of the conversation between Santanu Banerjee (user) and Claude (Claude Code, agentic coding assistant)
that produced this repository, 2026-10-02 .. 2026-10-12. Details live in `work/daily/` (STRATEGY.md is the living
strategy log, DEVLOG.md the day log, LEDGER.csv every submission with its hypothesis and score).

## 1. The brief and the standing rules
- Goal: "help me win" the Kaggle community competition 3rd WEAR Dataset Challenge @ HASCA 2026 (entered on 2026-10-02,
  10 days before the deadline; prizes were already settled at the paper deadline, so we played for rank).
- Use **every** daily submission (10 per UTC day) through an automated routine that does not clash with the other
  competitions' routines sharing the same laptop and Kaggle account.
- Kaggle GPU/TPU: never ask how much — use at most **50% of whatever quota remains at that moment**, re-evaluated each
  time (10 h left -> up to 5 h; later 2 h left -> up to 1 h). On the last two days: "use as much GPU as necessary at
  the earliest opportunity" (still within the shared-account cap).
- Split submissions between **low-risk and high-risk** strategies.
- Ask for guidance when stuck relative to the leaders. Never touch other competitions' kernels; laptop rules: nothing
  on C:, check CPU/RAM first and push heavy work to Kaggle kernels.

## 2. Timeline
- **Day 1 (10-02)**: built the tooling (submission ledger/queue, kernel push/wait with quota checks, blend tools, the
  daily routine). Re-implemented the best public pipeline with honest nested out-of-fold validation on test-like
  tiles of the 22 training subjects (OOF 0.9050 vs the author's 0.9053; public 0.9061). Quick wins: soft Sinkhorn
  (+0.005), graph-config ensembles. Oracle diagnostics showed that label-pure connectivity of the link graph matters far
  more than exact successors. Added a contrastive successor bi-encoder as link features, a multi-limb chain stacker
  and blends: public 0.92314 (#8).
- **Day 2-3 (10-03/04)**: the LB plateaued (~0.922-0.923) and public differences of 0.001-0.002 turned out to be noise
  (submissions differed on only 47-162 of 12,234 tiles). GPU context model and extra window seeds did not help. Key
  discovery: public notebooks (goodpjw2008, honghanhhh) save OOF + test probabilities **in exactly our row order**, so
  independently trained pipelines can be blended and validated honestly -> 0.92762.
- **10-05**: a boundary refiner along the link chains (re-judging tiles around label changes, idea from goodpjw2008)
  on our blends: +0.002 public. Probes showed the public notebook's edge came from its refiner (+0.006), not its decode.
  Public-heavy weights were preferred by the LB -> 0.93149 (#6).
- **10-06 .. 10-10**: no submissions — the daily routine did not run (the app was closed / runs were skipped). The
  most expensive lesson of the project (~50 lost submissions).
- **10-11**: new public variants (goodpjw2008's "part 2 ranked links", sujanmajhisuzan's bagged refiner) and our own GPU
  re-runs of the public notebooks with other seeds (p2s3, p2s6). Averaging several public pipelines + ours, refined:
  0.93326.
- **10-12 (last day)**: a third seed (p2s9); 4 seeds of the ranked-links pipeline + ours 0.4, refined: **0.93371**
  (#11; #1 0.94538). Discovered from the saved Submissions page that this competition scores **only one** final
  submission ("0/1") — the earlier plan "tick the best-OOF blend, let Kaggle auto-fill the second slot with the best
  public" would have dropped the best public one. Recommended q057 (best public, 4-seed average) over q043 (best OOF,
  single seed); the user makes the final pick in the Kaggle UI.

## 3. Reasoning behind the main decisions
- **Honest validation first.** Every component was nested by subject fold, so any idea could be judged by OOF before
  spending a submission. Every queued CSV states its OOF.
- **Treat small public differences as noise.** Two submissions that differ on ~100 tiles move the public score by
  ~0.002 by chance; we only acted on consistent patterns across several submissions.
- **Diversity beats more of the same.** Our ~26 runs shared the same window models and links, so adding more of them
  did not help; an independently trained public pipeline that alone scored only 0.911 still added +0.0046 in a blend.
- **Our OOF overstated our own pipeline** (by ~0.003 vs public), while the public pipelines' OOF understated theirs ->
  the public LB repeatedly preferred more public-pipeline weight than OOF did; we followed the public trend only when it
  was consistent across several weights.
- **Refine after blending.** Boundary errors dominate (81% of OOF errors are null <-> exercise at bout edges), and the
  refiner fixes them along the reconstructed chains; it worked on every blend (+0.002) at threshold 0.6 (0.5 flips too
  many tiles; extra experts as features and a second pass did not help).
- **Seeds of the strongest pipeline > mixing families.** On the last day, 4 seeds of one pipeline beat a mix of 6-7
  different public runs.
- **Final pick by robustness.** q057 averages 4 seeds and leads on public by +0.0019; q043's +0.0009 OOF lead comes from
  one seed whose own OOF sat above our three re-runs of the same notebook (seed luck).

## 4. Ideas that did not work (and why we stopped)
Viterbi/HSMM on reconstructed chains, mean-field MRF, GNN decoder, learned link reweighting/cutting, cluster
smoothing, graph iteration, second-level stacking (optimistic OOF, worse public), learned per-(subject, exercise)
counts on our blends, pseudo-label adaptation inside our blend (helps single runs only), extra window-model seeds and
a GPU context model, session clustering from the participant metadata (an oracle says +0.003..+0.013, but sessions
cannot be recovered from unordered tiles: video features encode the activity, not the scene).

## 5. Operational lessons
- A routine that only runs while the desktop app is open is fragile: 5 days of submissions were lost. Missed runs fire
  together when the app reopens and compete for CPU/RAM.
- The shared Kaggle account had only 2 GPU sessions for many projects; GPU jobs often waited hours. A notebook patch
  (`WEAR_ABLATION=0`) broke a 97-minute GPU run at its summary cell — test patches on a smoke run first.
- Check the competition's final-selection limit early instead of assuming two slots.
- Publishing: the Kaggle API can only change a kernel's visibility by saving a new version, and even a QUICK_SAVE
  version starts a run on the account's shared CPU sessions (`work/tools/kpublish.py`; 5 notebooks got failed
  re-runs that way). Saves refused with "Maximum batch CPU session count" still flipped 9 notebooks to public with
  their original runs. The clean route is Kaggle's web UI (Share -> Public), which needs a signed-in browser; the
  remaining notebooks are listed in NOTEBOOKS.md and made public on request.
- Browser automation surfaced two macOS browsers connected to the user's Claude extension although the user has no
  Mac; they were not used further and the user was advised to disconnect them.

## 6. If we had more time
Better successor links (a cross-encoder re-ranker on raw frames; the refiner would gain +0.018 with the true order),
test-time adaptation of the window models on confident pseudo-labels, and a scene representation independent of the
activity to exploit the per-session activity counts given in the participant metadata.
