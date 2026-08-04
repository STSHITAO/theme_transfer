# Technical design

## Architecture

```text
data/styles + data/targets
        |
        v
backend/package_workflow.py
  Qwen analysis -> Wan candidates -> Qwen QC -> final package
        |
        v
evaluation/tpqs_workflow.py
  preprocessing -> cached features -> ITTE v1.3 -> reports

benchmark/evaluation_set_v1
        |
        v
benchmark/tools/evaluate_current_itte.py
  frozen real-data baseline -> component evidence -> later candidate comparison
```

Generation and evaluation are deliberately separated. Prompt text and generation QC are diagnostic provenance only.

## ITTE data flow

- Appearance view: style, material, color and package comparisons.
- Structure view: identity comparisons with launcher labels/background reduced.
- Artifact view: original pixels, alpha and borders for deterministic integrity checks.
- DINOv3: dense appearance and structure features.
- VGG16 Gram: multi-layer style texture representation.
- DISTS: texture evidence in style; structure distance remains diagnostic for identity.
- LPIPS: content-distance identity diagnostic.
- Primary identity: DINOv3 dense same-label score ranked against reference-original and target-original identity galleries. DISTS/LPIPS identity weights are zero because real-label retrieval AUC was only 0.543/0.559 versus DINOv3 0.827.
- Handcrafted visual groups: color, background, stroke, texture/material, composition and complexity.

## Execution devices

- Official GPU environment: Conda `pytorch`, PyTorch `2.5.0+cu118`, `TPQS_DEVICE=cuda:0`.
- CPU compatibility: set `TPQS_DEVICE=cpu`; no data or API changes are required.
- The GTX 1650 has 4 GB VRAM. Default batch size is conservative and must fall back on smaller batches if out-of-memory occurs.

## Feature cache

Persistent caches live under `data/evaluations/_cache/`. A cache entry is determined by:

- resolved input path and file metadata/content identity;
- preprocessing view;
- model/backend ID;
- image size and feature-relevant configuration.

CPU and GPU may reuse the same deterministic feature cache. Device is execution metadata, not feature identity. Corrupt or incompatible entries must be rejected and recomputed atomically.

Cache coverage includes handcrafted style vectors, DINO dense features, VGG Gram vectors and DISTS/LPIPS pair distances. A full cache hit bypasses VGG and perceptual GPU model execution.

## Benchmark model

- `identity_gallery.csv`: one reliable original per canonical App identity.
- `theme_assets_manifest.csv`: every real themed asset and provenance.
- `pair_manifest.csv`: verified original/theme positives.
- `package_pair_manifest.csv`: same-App cross-theme relations.
- `four_theme_core.csv`: five identities present in all four themes.

Reference and query identities are disjoint within a Benchmark run. Splits are grouped by `canonical_app_id`.

## ITTE validation decision

- Frozen v1.2: five seeds, 20 designer-positive and 20 no-transfer-control runs.
- Identity retrieval: all 158 real pairs for DINOv3; 10 deterministic same-theme distractors per pair for DISTS/LPIPS.
- Final v1.3: identical five folds and dataset fingerprint.
- Designer-positive hard-gate acceptance improved from 35% to 95%.
- Mean designer/control ITTE separation improved from 40.12 to 45.01.
- Style and package scores are unchanged between matched runs.
- Absolute identity hard gates were removed because available data does not validate a defensible absolute cutoff. Low identity remains visible in continuous scores and decisions.

## Security

- `.env` is local-only and ignored by Git.
- API keys must never appear in logs, reports, prompts committed to Git or documentation.
- Model responses are sanitized before persistence when they may contain credentials.
- Benchmark assets and manifests are treated as immutable; reports go to separate result directories.
- GitHub push must exclude caches, incomplete downloads, secrets and machine-specific absolute paths.
