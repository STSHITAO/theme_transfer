# Progress

Last updated: 2026-08-24

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
- Reworked the final Wan prompt: identity is the hard priority, each image role is mapped explicitly, full transfer-plan reasoning is omitted, and reference App names are scrubbed and tested before the API call.
- Completed a controlled prompt-v2 run on 10 known high-pollution theme_001 Apps (20 candidates). Compared with the same Apps in the frozen package, candidates below target-recognition 60 decreased from 15 to 4, and selected failures decreased from five Apps to one. Nine selected outputs passed the recognition threshold; `crossfire_mobile` still copied reference identities in both candidates.
- Audited the complex-App metadata path and found that `target.json` was used by Qwen planning but `core_function` and the final `generation_brief` were absent from the Wan prompt. Added direct target metadata and executable-brief compilation.
- Removed the experimental target-level `generation_policy`. Existing App facts remain unchanged; the model now derives structure preservation versus function-semantic recomposition from all designer pairs, the target profile and the target original.
- Replaced first-five alphabetic theme learning with five-pair batched analysis over every valid pair plus aggregate conditional policy. Per-target style references now exclude the target, prefer the selected route and are deterministically distributed; Wan receives the target image first.
- Completed theme-learning-v2 real generation for theme_003 and theme_004 on the same 40 Apps, with two candidates per App. Both themes produced 80 candidates and 40 final outputs with no API skips. Qwen observed preserve/recompose pair counts of 61/25 and 20/16; generated route counts were 29/11 and 26/14.
- Completed cached GPU ITTE v1.4 on both new packages. theme_003 scored 89.03 and theme_004 87.27 with 100% coverage, but both correctly remain `failed_hard_gate` because of severe package outliers (`xiaohongshu` and `wechat`). Full evidence and the reconstruction/generalization boundary are recorded in `docs/THEME_LEARNING_V2_EXPERIMENT.md`.
- Isolated application-market metadata extraction under `tools/app_metadata_extractor/`. It owns its crawler-directory loader, stable-ID map, Chinese Prompt, Qwen client, checkpoints and tests; it imports no project pipeline code and never writes project datasets automatically.
- Unified all fixed Qwen/Wan natural-language prompts in Chinese while preserving machine-facing JSON fields, enum values and image-role identifiers. Corrected the legacy Wan template so its documented order matches the real request: `TARGET_IMAGE` first, then `STYLE_REFERENCE` images. The 86-test suite includes a regression check for active Prompt language and image order.

## In progress

- Four-theme real generation/evaluation pipeline started on 2026-08-05. It runs themes sequentially, generates two candidates per App, applies multimodal QC, then launches ITTE v1.4 on `cuda:0` with feature caching.
- The original 91-App `package_full_structure_v1` run is paused by the external Wan API account entitlement: theme_001 produced 90 usable Apps and one recorded content-inspection rejection; theme_002 produced 48 case outputs before `AccessDenied.Unpurchased`; its theme_003/theme_004 legs have not started. The separate 40-App theme-learning-v2 experiments for theme_003/theme_004 are complete, but they do not complete this four-theme 91-App run.

## Pending

- Future optional validation: add independently sourced generated failures or human ratings only when such data becomes available; do not synthesize subjective labels merely to tune weights.
- Restore Wan API billing/entitlement, resume from `package_full_structure_v1_theme_002`, then complete and summarize the four-theme real generation/evaluation run before treating v1.4 as empirically accepted.
- Add a hard final-publication gate: when every candidate fails identity QC, record and skip the App instead of publishing the highest-scoring fallback. Then rerun `crossfire_mobile` as the remaining prompt-v2 failure.
- Decide whether the next experiment should use target-wise leave-one-out theme evidence. The current run intentionally supports target/reference overlap and therefore measures in-package reconstruction, not unseen-App generalization.
- Investigate why target-compliant outputs can still become ITTE package outliers, starting with theme_003 `xiaohongshu` and theme_004 `wechat`; do not change ITTE thresholds from these two examples alone.

## Deployment status

No hosted service is deployed. The project runs locally through Python CLI entry points. The project state through commit `4f15f2f` is available at `https://github.com/STSHITAO/theme_transfer`; the current application-market metadata extraction changes are local until explicitly committed and pushed.
