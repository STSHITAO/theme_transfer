# Progress

Last updated: 2026-08-04

## Completed

- Audited the project and prevented stale package outputs from mixing across reruns.
- Created and verified Conda environment `theme_transfer` for CPU development/tests.
- Audited real Benchmark mappings and built `benchmark/evaluation_set_v1`.
- Validated 322 immutable assets, 158 original/theme pairs, 137 cross-theme pairs and a five-App four-theme core.
- Added Benchmark descriptions and protocol with image-only scoring boundaries.
- Implemented `benchmark/tools/evaluate_current_itte.py` without changing ITTE v1.2.
- Restored official DINOv3 and VGG16 weights and verified their SHA-256 hashes.
- Confirmed Conda environment `pytorch` uses PyTorch `2.5.0+cu118` on an NVIDIA GTX 1650.
- Installed missing ITTE dependencies into `pytorch` without replacing CUDA PyTorch.
- Completed one CPU smoke evaluation; it confirmed all model families run, while also confirming that a one-query identity run is intentionally marked unreliable.
- Added deterministic VGG Gram and DISTS/LPIPS pair caches; verified cache-hit scores are identical and repeated GPU runtime drops.
- Completed the unchanged ITTE v1.2 five-fold GPU baseline: 20 designer-positive plus 20 matched no-transfer-control runs.
- Measured real-label identity retrieval: DINOv3 AUC 0.827, DISTS-structure 0.543 and LPIPS 0.559.
- Implemented ITTE v1.3 identity-gallery calibration and removed unsupported absolute identity hard gates.
- Corrected dark-background quality false positives while retaining actual empty/exposure hard failures.
- Completed the matched v1.3 five-fold rerun: designer-positive acceptance 95%, designer/control separation 45.01 versus 40.12 in v1.2.
- Verified cache-miss CPU inference for DINOv3, VGG16, DISTS and LPIPS; full suite passes (57 tests).

## In progress

- Prepare a clean Git repository boundary and GitHub delivery.

## Pending

- Confirm the GitHub repository name, visibility and authenticated account, then initialize and push.
- Future optional validation: add independently sourced generated failures or human ratings only when such data becomes available; do not synthesize subjective labels merely to tune weights.

## Deployment status

No hosted service is deployed. The project currently runs locally through Python CLI entry points. No GitHub destination has yet been configured in this working copy.
