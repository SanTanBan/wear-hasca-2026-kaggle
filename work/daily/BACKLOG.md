# Backlog (updated 2026-10-04 21:00 IST)

## A. Ready-to-run
- Queue for 2026-10-05: refined blends (q028/q029, OOF 0.9296/0.9294), then the goodpjw decode probes (q026 our soft-97
  decode of its probabilities, q027 its argmax without refiner) — they tell whether our decode or its refiner explains
  why goodpjw alone (0.92942) beats our blends (0.92762).
- Refiner (src/wear/refine.py, CPU, ~3 min) on any blend saved in the wear26-art dataset: config probs=<tag>.
- Public runs: `tools/pubsync.py` then `tools/pubblend.py` (runbook 2a).

## B. Needs GPU
- wear26-gpjw = goodpjw notebook with WEAR_SEED=1 + WEAR_DUMP=1 (independent window models, two-tower links, 8 bagged
  L3 matchings). Blocked all evening by other projects; one-time task wear-gpu-night-push retries at 03:45 IST.
  When done: pubconv -> pub-gpjw1, blend with gp (seed averaging), and feed its 8 matchings to the refiner.

## C. Parked / dead ends
- Session-aware activity suppression: oracle +0.013, time-ordered sessions +0.0028, but unordered recovery fails
  (63% match). Would need a scene representation independent of the activity.
- Learned counts on our blends, adapted T inside our blend, second-level stacking, Viterbi/MRF/GNN decoders.

## D. Ideas
- Refiner with goodpjw's 8 bagged matchings + ours; second refiner pass.
- Cross-encoder re-ranker for top-K successor candidates (needs GPU).
