# Kaggle notebooks of this solution

All 68 Kaggle kernels we ran for the competition (owner `santanubanerjee9`, slugs `wear26-*`). Each of our own
kernels embeds the `work/src/wear` package of this repository and runs one entry module with the JSON config listed
(`work/tools/kpush.py` builds them); the readable code is in this repository. Key notebooks are in bold.

> **Caveat:** 14 of these notebooks are public; the others are still private and show "We can't find that page" to
> visitors. If you need a notebook that is inaccessible, please open an issue in this repository or message
> [santanubanerjee9 on Kaggle](https://www.kaggle.com/santanubanerjee9) and it will be made public.
> (Kaggle's API can only change visibility by saving a new version, which re-runs the notebook; the remaining ones
> will be switched in Kaggle's web UI instead, keeping their original outputs.)

| Notebook | What it does | Accelerator | Config | Visibility | Notes |
|---|---|---|---|---|---|
| **[wear26-base](https://www.kaggle.com/code/santanubanerjee9/wear26-base)** | Re-implemented public pipeline (window -> L0 links -> tabular experts -> L2 links -> graph -> Sinkhorn) with nested OOF | CPU | `work/configs/base.json` | public | latest version is a failed re-save from publishing - open Version 1 for the original run |
| [wear26-clusters](https://www.kaggle.com/code/santanubanerjee9/wear26-clusters) | Episode-cluster smoothing (dead end) | CPU |  | public | latest version is a failed re-save from publishing - open Version 1 for the original run |
| [wear26-ctx-c1](https://www.kaggle.com/code/santanubanerjee9/wear26-ctx-c1) | Context model over tile + chain + kNN neighbours (GPU, dead end) | GPU |  | public | latest version is a failed re-save from publishing - open Version 1 for the original run |
| [wear26-ctx-c2](https://www.kaggle.com/code/santanubanerjee9/wear26-ctx-c2) | Context model over tile + chain + kNN neighbours (GPU, dead end) | GPU |  | public | latest version is a failed re-save from publishing - open Version 1 for the original run |
| [wear26-decode](https://www.kaggle.com/code/santanubanerjee9/wear26-decode) | Decoding analysis: oracle links, Viterbi, calibration variants | CPU |  | public |  |
| [wear26-edgew](https://www.kaggle.com/code/santanubanerjee9/wear26-edgew) | Learned edge reweighting (dead end) | CPU |  | public |  |
| [wear26-edgew2](https://www.kaggle.com/code/santanubanerjee9/wear26-edgew2) | Learned edge reweighting (dead end) | CPU |  | public |  |
| [wear26-giter](https://www.kaggle.com/code/santanubanerjee9/wear26-giter) | Graph iteration (dead end) | CPU |  | public |  |
| [wear26-gnn](https://www.kaggle.com/code/santanubanerjee9/wear26-gnn) | GNN decoder (dead end) | CPU |  | public |  |
| **[wear26-gpjw](https://www.kaggle.com/code/santanubanerjee9/wear26-gpjw)** | goodpjw2008 "Learned Links + Counts" notebook re-run with WEAR_SEED=1 (blend member gpjw1) | GPU |  | public |  |
| [wear26-gsweep](https://www.kaggle.com/code/santanubanerjee9/wear26-gsweep) | Graph-propagation config sweep (top-5 ensemble) | CPU |  | public |  |
| [wear26-knnpur](https://www.kaggle.com/code/santanubanerjee9/wear26-knnpur) | kNN purity diagnostics | CPU |  | public |  |
| [wear26-l3](https://www.kaggle.com/code/santanubanerjee9/wear26-l3) | Link scorer: L2 pair features + successor bi-encoder features, graph + chain stacker | CPU |  | public |  |
| **[wear26-l4](https://www.kaggle.com/code/santanubanerjee9/wear26-l4)** | Link scorer: L2 pair features + successor bi-encoder features, graph + chain stacker | CPU |  | private (on request) |  |
| [wear26-l5](https://www.kaggle.com/code/santanubanerjee9/wear26-l5) | Link scorer trained with soft targets / blend probabilities | CPU |  | private (on request) |  |
| [wear26-l6](https://www.kaggle.com/code/santanubanerjee9/wear26-l6) | Link scorer: L2 pair features + successor bi-encoder features, graph + chain stacker | CPU |  | private (on request) |  |
| [wear26-l4s](https://www.kaggle.com/code/santanubanerjee9/wear26-l4s) | Link scorer trained with soft targets / blend probabilities | CPU |  | private (on request) |  |
| [wear26-linkexp](https://www.kaggle.com/code/santanubanerjee9/wear26-linkexp) | Link experiments | CPU |  | private (on request) |  |
| [wear26-mrf](https://www.kaggle.com/code/santanubanerjee9/wear26-mrf) | Mean-field Potts MRF decoder (dead end) | CPU |  | private (on request) |  |
| [wear26-nlink](https://www.kaggle.com/code/santanubanerjee9/wear26-nlink) | Contrastive successor bi-encoder (neural link features) | CPU |  | private (on request) |  |
| **[wear26-nlink2](https://www.kaggle.com/code/santanubanerjee9/wear26-nlink2)** | Contrastive successor bi-encoder (neural link features) | CPU |  | private (on request) |  |
| **[wear26-nlink3](https://www.kaggle.com/code/santanubanerjee9/wear26-nlink3)** | Contrastive successor bi-encoder (neural link features) | CPU |  | private (on request) |  |
| **[wear26-p2s3](https://www.kaggle.com/code/santanubanerjee9/wear26-p2s3)** | goodpjw2008 "Part 2: Ranked Links" notebook re-run with WEAR_SEED=3 (blend member) | GPU |  | private (on request) |  |
| **[wear26-p2s6](https://www.kaggle.com/code/santanubanerjee9/wear26-p2s6)** | goodpjw2008 "Part 2: Ranked Links" notebook re-run with WEAR_SEED=6 (blend member) | GPU |  | private (on request) |  |
| **[wear26-p2s9](https://www.kaggle.com/code/santanubanerjee9/wear26-p2s9)** | goodpjw2008 "Part 2: Ranked Links" notebook re-run with WEAR_SEED=9 (blend member) | GPU |  | private (on request) |  |
| [wear26-pgs](https://www.kaggle.com/code/santanubanerjee9/wear26-pgs) | Post-graph stacker | CPU |  | private (on request) |  |
| [wear26-probe-tpu](https://www.kaggle.com/code/santanubanerjee9/wear26-probe-tpu) | Hardware probe (CPU throughput on a TPU VM) | TPU |  | private (on request) |  |
| [wear26-ref1](https://www.kaggle.com/code/santanubanerjee9/wear26-ref1) | Boundary refiner along the link chains on blend `eqgp65q` | CPU | `work/configs/ref1.json` | private (on request) | q028 (0.92977) |
| [wear26-ref2](https://www.kaggle.com/code/santanubanerjee9/wear26-ref2) | Boundary refiner along the link chains on blend `eqgp` | CPU | `work/configs/ref2.json` | private (on request) |  |
| [wear26-ref3](https://www.kaggle.com/code/santanubanerjee9/wear26-ref3) | Boundary refiner along the link chains on blend `eqgp65q` (+ experts eq17, gp, hh) | CPU | `work/configs/ref3.json` | private (on request) |  |
| [wear26-ref4](https://www.kaggle.com/code/santanubanerjee9/wear26-ref4) | Boundary refiner along the link chains on blend `eqgp65q` | CPU | `work/configs/ref4.json` | private (on request) |  |
| [wear26-ref5](https://www.kaggle.com/code/santanubanerjee9/wear26-ref5) | Boundary refiner along the link chains on blend `eqgp80q` | CPU | `work/configs/ref5.json` | private (on request) | q031 (0.93149) |
| [wear26-ref6](https://www.kaggle.com/code/santanubanerjee9/wear26-ref6) | Boundary refiner along the link chains on blend `eqgphh` | CPU | `work/configs/ref6.json` | private (on request) |  |
| [wear26-ref7](https://www.kaggle.com/code/santanubanerjee9/wear26-ref7) | Boundary refiner along the link chains on blend `eqgp90q` | CPU | `work/configs/ref7.json` | private (on request) |  |
| [wear26-ref8](https://www.kaggle.com/code/santanubanerjee9/wear26-ref8) | Boundary refiner along the link chains on blend `gp` | CPU | `work/configs/ref8.json` | private (on request) |  |
| [wear26-ref9](https://www.kaggle.com/code/santanubanerjee9/wear26-ref9) | Boundary refiner along the link chains on blend `g3w20` | CPU | `work/configs/ref9.json` | private (on request) |  |
| [wear26-ref10](https://www.kaggle.com/code/santanubanerjee9/wear26-ref10) | Boundary refiner along the link chains on blend `g2w20` | CPU | `work/configs/ref10.json` | private (on request) |  |
| [wear26-ref11](https://www.kaggle.com/code/santanubanerjee9/wear26-ref11) | Boundary refiner along the link chains on blend `g3w0` | CPU | `work/configs/ref11.json` | private (on request) |  |
| [wear26-ref12](https://www.kaggle.com/code/santanubanerjee9/wear26-ref12) | Boundary refiner along the link chains on blend `g3w35` | CPU | `work/configs/ref12.json` | private (on request) |  |
| [wear26-ref13](https://www.kaggle.com/code/santanubanerjee9/wear26-ref13) | Boundary refiner along the link chains on blend `g4w20` | CPU | `work/configs/ref13.json` | private (on request) |  |
| **[wear26-ref14](https://www.kaggle.com/code/santanubanerjee9/wear26-ref14)** | Boundary refiner along the link chains on blend `x1w20` | CPU | `work/configs/ref14.json` | private (on request) | produced the best-OOF submission q043 (OOF 0.9322) |
| [wear26-ref15](https://www.kaggle.com/code/santanubanerjee9/wear26-ref15) | Boundary refiner along the link chains on blend `x1w40` | CPU | `work/configs/ref15.json` | private (on request) |  |
| [wear26-ref16](https://www.kaggle.com/code/santanubanerjee9/wear26-ref16) | Boundary refiner along the link chains on blend `g4w35` | CPU | `work/configs/ref16.json` | private (on request) |  |
| [wear26-ref17](https://www.kaggle.com/code/santanubanerjee9/wear26-ref17) | Boundary refiner along the link chains on blend `g6w20` | CPU | `work/configs/ref17.json` | private (on request) |  |
| [wear26-ref18](https://www.kaggle.com/code/santanubanerjee9/wear26-ref18) | Boundary refiner along the link chains on blend `g6w35` | CPU | `work/configs/ref18.json` | private (on request) |  |
| [wear26-ref19](https://www.kaggle.com/code/santanubanerjee9/wear26-ref19) | Boundary refiner along the link chains on blend `x3w20` | CPU | `work/configs/ref19.json` | private (on request) |  |
| [wear26-ref20](https://www.kaggle.com/code/santanubanerjee9/wear26-ref20) | Boundary refiner along the link chains on blend `x3w40` | CPU | `work/configs/ref20.json` | private (on request) |  |
| [wear26-ref21](https://www.kaggle.com/code/santanubanerjee9/wear26-ref21) | Boundary refiner along the link chains on blend `g7w20` | CPU | `work/configs/ref21.json` | private (on request) |  |
| **[wear26-ref22](https://www.kaggle.com/code/santanubanerjee9/wear26-ref22)** | Boundary refiner along the link chains on blend `x4w40` | CPU | `work/configs/ref22.json` | private (on request) | produced the best public submission q057 (0.93371) |
| [wear26-s4](https://www.kaggle.com/code/santanubanerjee9/wear26-s4) | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| [wear26-s5](https://www.kaggle.com/code/santanubanerjee9/wear26-s5) | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| [wear26-s6](https://www.kaggle.com/code/santanubanerjee9/wear26-s6) | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| [wear26-s7](https://www.kaggle.com/code/santanubanerjee9/wear26-s7) | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| [wear26-s8](https://www.kaggle.com/code/santanubanerjee9/wear26-s8) | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| [wear26-s9](https://www.kaggle.com/code/santanubanerjee9/wear26-s9) | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| [wear26-s10](https://www.kaggle.com/code/santanubanerjee9/wear26-s10) | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| **[wear26-s11](https://www.kaggle.com/code/santanubanerjee9/wear26-s11)** | Multi-limb stacker (S3 + per-limb neighbour aggregates) + graph | CPU |  | private (on request) |  |
| [wear26-s3b](https://www.kaggle.com/code/santanubanerjee9/wear26-s3b) | Tabular stacker S3 variant | CPU |  | private (on request) |  |
| [wear26-selftrain](https://www.kaggle.com/code/santanubanerjee9/wear26-selftrain) | Subject self-training LightGBM (dead end) | CPU |  | private (on request) |  |
| [wear26-smoke](https://www.kaggle.com/code/santanubanerjee9/wear26-smoke) | Smoke test of wear26-base on 5 subjects | CPU | `work/configs/smoke.json` | public | latest version is a failed re-save from publishing - open Version 1 for the original run |
| [wear26-v2](https://www.kaggle.com/code/santanubanerjee9/wear26-v2) | Chain stacker on reconstructed chains (+ multi-limb features) | CPU |  | private (on request) |  |
| [wear26-v3](https://www.kaggle.com/code/santanubanerjee9/wear26-v3) | Chain stacker on reconstructed chains (+ multi-limb features) | CPU |  | private (on request) |  |
| [wear26-v4](https://www.kaggle.com/code/santanubanerjee9/wear26-v4) | Chain stacker on reconstructed chains (+ multi-limb features) | CPU |  | private (on request) |  |
| [wear26-v5](https://www.kaggle.com/code/santanubanerjee9/wear26-v5) | Chain stacker on reconstructed chains (+ multi-limb features) | CPU |  | private (on request) |  |
| [wear26-v6](https://www.kaggle.com/code/santanubanerjee9/wear26-v6) | Chain stacker on reconstructed chains (+ multi-limb features) | CPU |  | private (on request) |  |
| [wear26-v7](https://www.kaggle.com/code/santanubanerjee9/wear26-v7) | Chain stacker on reconstructed chains (+ multi-limb features) | CPU |  | private (on request) |  |
| [wear26-wtrain-g1](https://www.kaggle.com/code/santanubanerjee9/wear26-wtrain-g1) | Window-model retraining with new seeds (GPU) | GPU |  | private (on request) |  |
| [wear26-wtrain-smoke](https://www.kaggle.com/code/santanubanerjee9/wear26-wtrain-smoke) | Smoke test of wear26-wtrain | CPU |  | private (on request) |  |
