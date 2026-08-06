# Technical design

## Architecture

```text
dataset/apps.json -> data/targets/<app_id>/target.json
data/styles + data/targets
        |
        v
backend/package_workflow.py
  all-pair batched Qwen analysis -> per-theme/App route decision -> executable Wan prompt -> candidates -> Qwen QC -> final package
        |
        v
evaluation/tpqs_workflow.py
  preprocessing -> cached features -> ITTE v1.4 policy-gated scoring -> reports

benchmark/evaluation_set_v1
        |
        v
benchmark/tools/evaluate_current_itte.py
  frozen real-data baseline -> component evidence -> later candidate comparison
```

Generation and evaluation are deliberately separated. Prompt text and generation QC are diagnostic provenance only.

## Wan identity-isolation prompt contract

Wan still receives the same fixed real theme references, but the final prompt treats them as style-only evidence. The prompt is compiled from executable transfer fields rather than embedding the full Qwen reasoning JSON. `structure_policy_rationale` and reference App names are scrubbed before the request; the request is rejected locally if a forbidden reference identity remains.

The single Wan text content item includes an explicit image-role map. `IMAGE_1` is `TARGET_IMAGE` and is the only source of logo, text, subject, silhouette, geometry and layout. The following three `STYLE_REFERENCE` images may contribute only visual treatment. They are selected after the route decision from examples with the same observed structure mode when possible, are deterministically distributed across the pool and always exclude the current target App. Target identity is the highest priority; theme fidelity may not replace it. The original fixed-reference packages remain frozen as the baseline for comparison.

The executable prompt also receives `display_name`, `category`, `core_function`, the frozen target identity anchor, semantic cues, allowed recomposition and the final `generation_brief`. This closes the earlier gap where `target.json` reached Qwen planning but its description and the plan's executable brief were omitted from the Wan prompt.

## Theme learning and target semantics

`dataset/apps.json` is the maintained metadata source; `scripts/prepare_generation_data.py` copies each complete record to `data/targets/<app_id>/target.json` and into the matching entries of `theme.json`. The existing fields (`display_name`, `category`, `store_description`, `core_function`) remain neutral facts and are sufficient input for constrained multimodal reasoning; no target-level generation policy or hand-authored object list is used.

Theme learning uses every valid `original -> style_ref` pair, not the first alphabetic examples. Pairs are processed in batches of five to respect multimodal input size. Each batch records the observed transformation, preserved identity, redesigned parts, structure decision and image evidence. A text-only aggregation then derives the shared theme board and conditional policy for when the designer preserves structure versus represents software function. The complete compact pair-pattern list and coverage are retained in `theme_design_analysis.json`.

For each target, Qwen receives the aggregate theme evidence, existing target facts and target original. It freezes `preserve_major_structure` or `semantic_recompose` before generation. Functional objects may be inferred from `core_function` only when real theme examples support semantic recomposition; otherwise the original subject remains primary. Theme artifacts are reused on resume so an interrupted package does not repeat paid theme-analysis calls.

## Pre-generation structure policy

Each theme/App case freezes the following fields in `identity_strategy.json` and carries them unchanged into `transfer_plan.json` before Wan is called:

- `structure_preservation_mode`: `preserve_major_structure` or `semantic_recompose`;
- `structure_identity_metric_applicable`: derived deterministically from the mode;
- `structure_policy_rationale`: evidence from reference original/style pairs and the target identity.

`preserve_major_structure` means the main silhouette, geometry and spatial relationships should remain recognizable. `semantic_recompose` allows the main geometry to be replaced by a function symbol, prop or small scene while still preserving brand or semantic identity. Package metadata records the policy for every App. Evaluation rejects inconsistent mode/boolean pairs. Legacy packages without the fields default to structural evaluation for backward compatibility.

## ITTE data flow

- Appearance view: style, material, color and package comparisons.
- Structure view: identity comparisons with launcher labels/background reduced.
- Artifact view: original pixels, alpha and borders for deterministic integrity checks.
- DINOv3: dense appearance and structure features.
- VGG16 Gram: multi-layer style texture representation.
- DISTS: texture evidence in style; structure distance remains diagnostic for identity.
- LPIPS: content-distance identity diagnostic.
- Primary identity: DINOv3 dense same-label score ranked against reference-original and target-original identity galleries, but only for Apps frozen as `preserve_major_structure`. For `semantic_recompose`, the same value is retained as a diagnostic and excluded from the primary identity and total score. DISTS/LPIPS identity weights remain zero because real-label retrieval AUC was only 0.543/0.559 versus DINOv3 0.827.
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
- The frozen v1.3 Benchmark result remains the evidence for the image metrics themselves. ITTE v1.4 changes applicability rather than the DINO scoring formula; its four-theme generated-package validation is tracked separately and must not overwrite the frozen v1.3 baseline.

## Security

- `.env` is local-only and ignored by Git.
- API keys must never appear in logs, reports, prompts committed to Git or documentation.
- Model responses are sanitized before persistence when they may contain credentials.
- Benchmark assets and manifests are treated as immutable; reports go to separate result directories.
- GitHub push must exclude caches, incomplete downloads, secrets and machine-specific absolute paths.
