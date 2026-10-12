# Dev log

## 2026-10-02 (day 1)
- Set up tooling (tools/wear.py, kpush/kwait, blend, variants), daily routine `wear-daily-submissions` (20:47 IST).
- s000 public fast-inference output: 0.90770 public (sanity check).
- Re-implemented Briano's pipeline with honest nested OOF (wear26-base): window 0.7256, S3 0.8142, L0 33% exact,
  L2 56% exact / 93.7% same-label, graph+cal OOF 0.9050 (Briano 0.9053). Test output: 0.90610 public (q001).
- Soft Sinkhorn (3 iters): OOF 0.9104 -> **0.91055** public (q002).
- Graph config sweep (k=5, tau 0.2, rounds 5, top-5 ensemble): OOF 0.9124 -> **0.91209** public (q003).
- Diagnostics: oracle links 0.936 (soft cal); cutting cross-label links 0.926; random same-label chains 0.994 ->
  label-pure connectivity is what matters, exact successors do not. Learned link reweighting (AUC 0.87-0.89) did not
  help (0.9093). Mean-field MRF, subject self-training, Viterbi on L2 chains: no gain.
- Stackers: chain stacker 0.841 argmax but 0.911 after graph; post-graph stacker 0.909; geo blend
  gsweep+pgs+cs = OOF 0.9143 (q004 queued).
- nlink (contrastive successor bi-encoder, CPU): held-out dual-top1 ~0.27-0.30.
- L3 (L2 + nlink v3 features): exact 0.593, graph soft 0.9128; multi-limb chain stacker on L3 (w 0.5) 0.9156.
- Blend of 5 runs: OOF 0.9197 -> **0.92112** public (q005).
- (session glitch ~10:00-11:30 UTC; all in-flight kernels had finished fine)
- L4 (nlink v2+v3): exact 0.605, graph soft 0.9150, cs 0.9178; v2 (cs w0.4 + gsweep configs on L3) 0.9174.
- GNN decoder 0.858 (dead end); edge reweighting v2 with change-point features AUC 0.8925, no gain (dead end).
- Blend of 6 runs (L3cs, L4cs, v2cs0.4, pgs, s3b, gsweep): OOF 0.9223 -> **0.92276** public (q007), rank #8.
- Best blend arrays uploaded as private dataset santanubanerjee9/wear26-art (blend7).
- S4 (S3 + multi-limb IMU aggregates, L4 links): 0.9189 -> blend OOF 0.9235 -> **0.92314** public (q008).
- L5 (link scorer fed with blend7 probabilities; second-level): same-label links 94.8% (L4 93.9%), graph 0.9191,
  cs 0.9212; blend OOF 0.9244 but **0.92258** public (q009) -> treat second-level gains with caution.
- "ml2" features (per-limb aggregates of neighbours' IMU-only / fusion probabilities + prediction-aware kNN):
  v3 cs (L4) 0.9219, S5 (L4) 0.9221 [first-level]; S6 (L5) 0.9240, v5 cs (L5) 0.9224, v4 (cs on blend7) 0.9230
  [second-level]; v6 cs (L2) 0.9169.
- Dead ends today: asymmetric calibration, per-class bias tuning (CV 0.9205 < 0.9223), GNN, edge reweighting v2.
- GPU job (wtrain g1) waits: quota-checked loop (needs <= 50% of remaining; other routes drained it to ~1.9h).
- LB plateau: q010 (equal clean blend, OOF 0.9243) 0.92189; q011 (equal blend incl. second-level, OOF 0.9255)
  0.92073. Submissions differ on 47-162 of 12,234 tiles -> noise; second-level runs look optimistic.
- Late runs: L4s (soft-target links: same-label 94.5%, exact 59.3%) cs 0.9215; S7 (S5, slower LightGBM) 0.9223;
  S8 (S5 on L2) 0.9169. Clean equal-weight blend of 19 runs: OOF 0.9252 -> queued q012 for 2026-10-03.
- GPU context model (wear26-ctx-c1, encoder initialised from the pretrained fold models) + wtrain g1 queued; one-time
  task wear-gpu-morning-push launches them at 05:40 IST after the quota reset.

## 2026-10-03 (day 2)
- GPU quota is weekly (30 h for Oct 3-9). wtrain g1 (52 min T4): two fusion seeds x 20 epochs OOF 0.698/0.700, IMU
  x 40 epochs 0.591 -> adding them to the window ensemble does not help (0.7243 vs 0.7256). Dropped the base2 refresh.
- 01:00 UTC: pushed wear26-ctx-c1 (GPU context model) myself after wtrain finished (one GPU slot at a time);
  L6 links (nlink v1+v2+v3) running on CPU.

## 2026-10-04 (day 3)
- Oct 3 routine fired late (04:36 IST) and used only 1 slot. Disabled the obsolete wear-gpu-morning-push.
- tools/counts.py (learned per-(subject, exercise) counts, goodpjw-style ridge on margin profiles): no OOF gain on
  our blends (0.9253 vs 0.9252) and no public gain (q018 0.92599 vs q017 0.92604).
- wear26-s11 = S5 + pseudo-label adapted T: single run OOF 0.9237 (best single), but adds nothing to the blend.
- Public notebooks save final_probabilities.npz in our exact row order -> tools/pubsync.py, pubconv.py, pubblend.py.
  Blends with honghanhhh (0.92705) and goodpjw2008 (0.92762, best) — see reports/2026-10-04.md.
- Session-aware suppression: +0.0028 OOF with time-ordered sessions, oracle +0.013, but sessions cannot be recovered
  from unordered tiles (spectral clustering on links + video kNN: 63% match) -> parked.
- wear26-ref1 (CPU): boundary refiner along chains on our blend (eqgp65q) with 5 link sets.
- GPU/TPU sessions busy all evening (other projects). wear26-gpjw (seed-1 copy) retried by one-time task at 03:45 IST.
2026-10-04 16:37 UTC daily catch-up run: pushed wear26-gpjw version 1 (T4, WEAR_SEED=1) - night push task will see it running/complete and skip
2026-10-05 00:26 UTC night GPU push: sessions busy (Maximum batch GPU session count of 2 reached; prior version status was ERROR; GPU remaining 17.25h)
2026-10-05 00:27 UTC night GPU push: skipped (v1 status ERROR after ~97 min T4: KeyError 'one matching, public configs, fixed 97' in cell 23 SCORES summary - ablation keys missing with WEAR_ABLATION=0; notebook unchanged so a re-push would fail the same way; needs a fix before re-push; GPU left 17.25h)
