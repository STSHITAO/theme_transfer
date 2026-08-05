# Progress

Last updated: 2026-08-06

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
- Verified cache-miss CPU inference for DINOv3, VGG16, DISTS and LPIPS.
- Delivered the validated ITTE v1.3 project and Benchmark evidence to `STSHITAO/theme_transfer` on the `main` branch; required model weights remain versioned with Git LFS.
- Added and exercised `scripts/prepare_generation_data.py`: it converts the read-only normalized dataset into 91 generation targets and 158 matched theme examples, preserves image formats and reports 73 unmatched theme assets.
- Implemented the pre-generation per-App structure policy across theme analysis, identity strategy, transfer plan, package metadata and ITTE v1.4 conditional identity scoring; full suite passes (68 tests).
- Added resumable four-theme generation and batch evaluation entry points; the current validation run uses two candidates per App for multimodal QC selection.
- Added per-App Wan content-inspection fault isolation: rejected inputs are recorded and skipped, while ITTE evaluates successful outputs with explicit coverage and cross-theme common-intersection metadata.

## In progress

- Four-theme real generation/evaluation pipeline started on 2026-08-05. It runs themes sequentially, generates two candidates per App, applies multimodal QC, then launches ITTE v1.4 on `cuda:0` with feature caching.
- The run is paused by the external Wan API account entitlement: theme_001 produced 90 usable Apps and one recorded content-inspection rejection; theme_002 produced 48 Apps before `AccessDenied.Unpurchased`; themes 003 and 004 have not started. No cross-theme ITTE summary is claimed from this partial run.

## Pending

- Future optional validation: add independently sourced generated failures or human ratings only when such data becomes available; do not synthesize subjective labels merely to tune weights.
- Restore Wan API billing/entitlement, resume from `package_full_structure_v1_theme_002`, then complete and summarize the four-theme real generation/evaluation run before treating v1.4 as empirically accepted.

## Deployment status

No hosted service is deployed. The project runs locally through Python CLI entry points. The previously delivered ITTE v1.3 state is available at `https://github.com/STSHITAO/theme_transfer`; the current data-preparation changes have not been pushed.
