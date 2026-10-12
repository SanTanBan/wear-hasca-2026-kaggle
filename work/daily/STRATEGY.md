# Strategy (living document)

## Update 2026-10-12 15:40 UTC (final day)
- p2s9 harvested; **best public 0.93371 (q057)** = refiner thr 0.6 on x4w40 (ours 0.4 + xbbaf9, p2s3, p2s6, p2s9).
- **Final selection CORRECTED 15:55 UTC**: the Submissions page allows only ONE selection ("Select up to 1"; 0/1
  ticked). Ticking q043 would make it the only scored entry and drop q057. Recommendation: q057 (leave nothing
  ticked -> Kaggle auto-selects the best public, or tick q057). Why: q043's OOF lead (0.9322 vs 0.9313) comes from the
  0.8 weight on the public ranked-links run xbbaf9, whose OOF (soft-97 0.9261) sits above our 3 reruns of the same
  notebook (0.9216 / 0.9222 / 0.9240) -> seed-specific OOF luck; q057 averages 4 seeds and leads on public by +0.0019
  (public = ~50% of the same 4 test subjects). The user ticks it in the Kaggle UI (no CLI).

## Update 2026-10-11: averaging public pipelines
- **Best public 0.93326 (q044)** = refiner thr 0.6 on `datasets/wear26-art:g4w20` (ours eq17 0.2 + gp, gpjw1, sj,
  xbbaf9 [goodpjw part-2 ranked links] equally). q039 (g3w20 thr 0.55) 0.93323. Single ranked-links run with ours:
  0.93185-0.93188 despite the best OOF (0.9322) -> more public runs > one strong one. No-ours 0.93003.
- Last day 10-12: try g4w20 thr 0.55 (q046 queued), g4w35 (ref16, q047/q048), thr 0.55 on others.
- GPU (user 10-11: "as much as necessary, earliest opportunity"): wear26-p2s3 / wear26-p2s6 = goodpjw part-2
  ranked-links notebook with WEAR_SEED 3 / 6 (window seeds +30/+60, link seeds 3-5 / 6-8: independent of the public
  run), full training on T4 x2, pushed 2026-10-11 07:20 UTC (~3.5 h). When COMPLETE: kwait, `pubconv.py
  artifacts/wear26-p2s3 --tag p2s3` (same for p2s6), copy art/ to artifacts/pub-p2s3/art, then pubblend
  `--only gp,gpjw1,sj,xbbaf9,p2s3,p2s6` at ours 0.2 / 0.35 and `--only xbbaf9,p2s3,p2s6` at ours 0.2 / 0.4, upload,
  refine (thr 0.55/0.6/0.65), queue the 10-12 slots with these first.
  DONE 2026-10-11 12:30 UTC: both COMPLETE (~4.5 h each; own nested OOF 0.9259 / 0.9263, soft 0.9222 / 0.9216),
  converted to pub-p2s3 / pub-p2s6; refined blends g6w20 (ref17, OOF 0.9306), g6w35 (ref18, 0.9312), x3w20 (ref19,
  0.9308), x3w40 (ref20, 0.9313) queued as q049-q054 -> 10 ready for 10-12 (with q046-q048, q035).
  Highest refined OOF is still x1w20 (q043, 0.9322) -> it stays the one manual final selection.
- 2026-10-11 15:20 UTC: pushed wear26-p2s9 (same notebook, WEAR_SEED 9, GPU ~4.5 h; quota 16.5 h left). 10-12 run:
  if COMPLETE, kwait + `pubconv.py artifacts/wear26-p2s9 --tag p2s9`, copy art/ to artifacts/pub-p2s9/art, pubblend
  `--only gp,gpjw1,sj,xbbaf9,p2s3,p2s6,p2s9` (ours 0.2/0.35) and `--only xbbaf9,p2s3,p2s6,p2s9` (ours 0.2/0.4), refine,
  queue them at priority > 10.5 (they displace the lowest of q035/q054/q048). If not complete, use the queue as is.
- **Final selection (user 10-11)**: manually select ONE submission = the refined blend with the best OOF (today q043,
  OOF 0.9322, public 0.93185; replace it if a 10-12 blend has higher OOF). The other slot is left to Kaggle's
  best-public auto-selection (today q044 0.93326); Kaggle fills every unselected slot with the best public score
  (user confirmed), so select nothing else.

## Update 2026-10-05 (08:00 IST): the refiner + goodpjw-heavy blends
- **Best public 0.93149 (q031)** = our boundary refiner (src/wear/refine.py, thr 0.6, 5 link sets) on ours eq17 0.2 +
  goodpjw 0.8 (`datasets/wear26-art:eqgp80q`). #6 on the LB (#1 0.94285, #5 0.93208).
- Probes: goodpjw probabilities with our soft-97 decode 0.92449, its argmax (its learned-count decode, no refiner)
  0.92370, its own refined labels 0.92942 -> its refiner is worth ~+0.006 public; ours on the same probabilities
  0.92912 (same). Our refiner adds ~+0.002 on our blends (q028 0.92977 vs 0.92762; q030 0.92937 vs 0.92704).
- Refiner thresholds: 0.5 (171 flips) 0.92808 < 0.6 (105 flips) 0.92977; OOF optimum 0.6. Extra experts as features
  and a second pass do not help (OOF 0.9292 vs 0.9296). Adding goodpjw's own flips on top of ours hurts OOF.
- honghanhhh in a refined three-way blend: 0.92901 (hurts) -> drop it.
- The public LB prefers more goodpjw weight than OOF does (OOF optimum ~0.5): our OOF overstates our pipeline.
- Final selection (tentative): (1) best public refined goodpjw-heavy blend; (2) a refined 0.5-0.65 blend that OOF
  prefers (hedge in case the public split flatters goodpjw).

## Update 2026-10-04 (20:15 IST): public-pipeline blending is the lever
- LB top: 0.94201 (Anonym), 0.93849, 0.93661, 0.93593, 0.93019; public notebook goodpjw2008 0.92942.
  **We: 0.92705 (q020)**, previous best 0.92314.
- Public notebook honghanhhh/wear-hasca-lb-0-9 (fork of goodpjw2008: own window models, two-tower links, learned
  counts, adapted T) saves final_probabilities.npz with OOF + test probabilities in exactly our row order.
  Alone: OOF 0.9219, **public 0.91082 (q019)**. It disagrees with our eq17 on 5.8% of test tiles.
- Our 26-run equal blend eq17: OOF 0.9256, public 0.92247 (q014). eq17 0.7 + hh 0.3 (temp 2): OOF 0.9273, public
  0.92604 (q017); with learned counts 0.92599 (q018: counts do not matter); eq17 0.5 + hh 0.5: OOF 0.9269,
  **public 0.92705 (q020)**. Diversity of an independently trained pipeline is worth ~+0.004.
- Tools: `tools/pubsync.py` (watch + download + convert public runs), `tools/pubblend.py` (ours + public runs),
  `tools/pubconv.py` (alignment check + temp). Runbook 2a.
- Next: goodpjw2008's own run (0.92942 alone) when its current run completes; our own seed-1 run of that notebook
  (kernels/wear26-gpjw, GPU, blocked by other projects' sessions today); S5 with adapted T (wear26-s11, CPU).
- Best probabilities: `datasets/wear26-art:eq17` (ours), `artifacts/pub-hh/art:hh`, `datasets/wear26-art:eqhh`
  (eq17 0.7 + hh 0.3).
- Evening: goodpjw2008's run converted (`artifacts/pub-gp/art:gp`; its own LB 0.92942, OOF 0.9272 after its boundary
  refiner). Blends: eq17 0.5 + gp 0.5 -> 0.92704; **eq17 0.35 + gp 0.65 -> 0.92762 (best)**; + hh -> 0.92721;
  with gp's refiner flips swapped into its probabilities 0.9264x (OOF up, public down -> noise level).
- Our own boundary refiner (src/wear/refine.py, CPU 3 min, 5 link sets) on those blends: eqgp65q 0.9277 -> 0.9296,
  eqgp 0.9286 -> 0.9295 (thr 0.6). Queued for 2026-10-05 (q028-q030) with the decode probes q026/q027.
- Puzzle: goodpjw alone (0.92942) still beats every blend on public although OOF prefers the blends by ~0.002.
  q026/q027 probe whether our soft-97 decode or the missing refiner explains it. If gp's own decode wins, move our
  blends to its learned-count decode; final selection should then keep one gp-heavy candidate.

## Situation (2026-10-02 17:30 IST)
- LB top: 0.93849 (Nicolas Krusche), 0.93661, 0.93200, 0.92874, 0.92718, 0.92554, 0.92368; **we are #8 at 0.92276**
  (team name on the LB: "InSociEUP"). Public notebooks: 0.9077.
- No prize/points for late entrants (host, discussion 743256) — we play for rank.

## Best probabilities right now (use these for variants / blends)
- `datasets/wear26-art/{oof,test}_blend8.npz` (key P_graph) = tuned geo blend of 19 runs incl. S4, OOF 0.9235 ->
  **public 0.92314 (q008)**. Previous: blend7 (6 runs) OOF 0.9223 -> 0.92276 (q007).
- blend7 is on Kaggle as private dataset `santanubanerjee9/wear26-art` (version it when a new blend is needed there).
- `python tools/blend_search.py --all ...` reproduces/extends the blend from artifacts/*/art (see runbook).

## Problem structure we exploit
- Test = 4 subjects x ~2–5k shuffled non-overlapping 1-s tiles of continuous sessions, one random limb per tile.
- Each subject does each of 18 exercises once (~97 tiles, sometimes split by rests); the rest is null.
- Sessions per test subject (participant_meta_data.txt): sbj 22: 3 sessions (9/8/1 activities), 23: 3 (9/8/1),
  24: 2 (9/9), 25: 2 (9/9). Test ids are fully shuffled (checked: no temporal information in id order).

## Pipeline (public SOTA, re-implemented with honest OOF everywhere in src/wear)
window classifier (pretrained 5-fold) -> L0 links -> tabular experts -> L2 combined links -> graph propagation with
self-training -> sharpen + per-subject Sinkhorn count calibration. Briano's OOF: 0.725 -> 0.847 -> 0.877 -> 0.905.

## What worked (OOF -> public)
- Soft count calibration (Sinkhorn 3 iterations instead of 50): 0.9050 -> 0.9104 OOF, 0.9061 -> 0.91055 public.
- Graph config sweep (k 5, tau 0.2, rounds 5; top-5 ensemble): 0.9124 OOF -> 0.91209 public.
- L3 links = L2 features + neural successor (nlink bi-encoder) features: exact 56% -> 59% (L4 with two nlink models 60.5%),
  graph 0.9104 -> 0.9128 (L4 0.9150).
- Multi-limb chain stacker (LightGBM on chain-context probabilities + per-limb IMU aggregates of chain / video-kNN
  neighbours) mixed 0.4–0.5 into the tile log-probs before the graph: 0.9156–0.9178 single runs.
- Geometric blends of diverse decoders: 0.9197 OOF -> 0.92112 public; 0.9223 OOF -> 0.92276 public.

## Dead ends (do not retry without a new idea)
- Viterbi on real chains, mean-field Potts MRF, subject self-training LightGBM, GNN decoder (0.858), learned link
  reweighting / cutting (AUC 0.87–0.89 not enough), episode-cluster smoothing, graph iteration, nlink embeddings in kNN.
- Diagnostics: oracle links 0.936; cutting only cross-label links 0.926; random same-label chains 0.994 ->
  label-pure connectivity matters, exact successors barely do.

## LB plateau and noise (2026-10-02 evening)
- Public: q007 0.92276, **q008 0.92314 (best; OOF 0.9235)**, q009 0.92258 (OOF 0.9244, L5 second-level), q010 0.92189
  (equal-weight clean blend, OOF 0.9243), q011 0.92073 (equal-weight blend incl. second-level runs, OOF 0.9255).
- These submissions differ on only 47-162 of 12,234 test tiles -> public differences of 0.001-0.002 are noise.
  OOF gains < ~0.002 are not detectable on the public LB; blends with second-level runs (anything using blend7:
  wear26-l5, -v4, -v5, -s6) look optimistic -> prefer first-level runs for final selections.
- Tuned blend weights do not generalise (fold-CV tuned 0.9228 < equal weights 0.9234): use tools/eqblend.py.
- Next real gains need new information: GPU context model (wear26-ctx-c1, raw inputs of the tile + 8 chain + 4 kNN
  neighbours) and extra window-model seeds (wear26-wtrain-g1); both wait for the GPU quota reset / a free slot.

## Open levers
1. Link features from better probabilities (L5: blend7 inside the link scorer) — running.
2. Stronger window models (GPU; slots have been fully used by other competitions all day).
3. More diverse decoders for the blend (S4 with multi-limb features, chain-stacker seeds).

## Validation
- 22 train subjects, 5 subject folds, tiles sampled exactly like the test. OOF macro F1 tracks the public LB within ~0.002.
- Every candidate submission must state its OOF estimate.

## Final selection
- Kaggle uses the best public score unless we pick. Pick (1) best public LB and (2) best OOF-validated distinct variant.

