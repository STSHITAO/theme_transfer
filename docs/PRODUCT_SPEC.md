# Product specification

## Goal

Generate a coherent package of themed mobile App icons from real theme examples, and evaluate the generated images objectively with image-derived ITTE scores. The product separates generation-time assistance from evaluation-time evidence; a structure policy frozen before generation controls only whether geometric identity evidence is applicable for each App.

## Users and workflows

1. A researcher prepares paired theme examples (`original -> style_ref`) and target App originals.
2. The package workflow analyzes the theme, plans each target transfer, calls the image generator, selects candidates and publishes a package.
3. The evaluation workflow measures style fidelity, identity preservation, package coherence and observable technical defects.
4. The Benchmark workflow tests whether the ITTE design itself behaves correctly on real originals and designer-made themed icons before metric changes are accepted.

## Current functions

- Single-App and package-level theme transfer.
- Qwen-based theme/identity analysis and candidate QC.
- Wan multi-image generation.
- Deterministic ITTE v1.4 evaluation with verified-label DINOv3 identity retrieval gated per App by a pre-generation structure policy.
- A self-contained real-image Benchmark with 91 originals, 231 themed assets and 158 verified original/theme pairs.
- CPU/GPU inference selection and persistent feature caches.
- Two generation routes for target Apps: structure-preserving transfer and function-semantic recomposition. The route is decided per theme/App from designer pair evidence, the neutral `target.json` facts and the target original; it is not stored as a preset target field.

## Interfaces and pages

The current implementation is CLI and artifact based; no web UI is implemented yet.

Planned UI pages, if a frontend is added:

- **Projects**: theme, package and evaluation history.
- **Theme setup**: paired reference inspection and mapping validation.
- **Generation**: target selection, progress, candidates and final package.
- **Package review**: contact sheet, per-App provenance and generation QC.
- **ITTE report**: dimension scores, hard gates, outliers and component diagnostics.
- **Benchmark**: frozen split, baseline/candidate comparison and cache status.
- **Settings**: model IDs, device, batch size and API configuration without displaying secret values.

## Non-goals

- ITTE does not judge artistic intent, cultural meaning, copyright, spelling or subjective beauty.
- Generation-stage Qwen scores do not enter the ITTE primary score.
- Benchmark theme descriptions are metadata, not scoring inputs.
- Current validation establishes real-positive behavior and no-transfer separation; it does not claim human-opinion correlation or calibrated performance on synthetic degradation severity.
- The structure policy does not contribute a score and cannot be changed after seeing generated results. It only marks DINOv3 geometric identity as primary evidence or diagnostic-only for that App.
- `target.json` metadata is generation input, not ITTE scoring evidence. It describes what the App is and does; Qwen may make a constrained visual extension only when the observed theme-pair policy supports semantic recomposition.
