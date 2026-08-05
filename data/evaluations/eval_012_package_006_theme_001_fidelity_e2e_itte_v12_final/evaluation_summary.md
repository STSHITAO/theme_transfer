# ITTE v1.2 Evaluation Summary

## Run

- Theme: `theme_001`
- Package: `package_006_theme_001_fidelity_e2e`
- Framework: `ITTE v1.2-image-only`
- Confidence: `medium` (7 complete `original -> style_ref` reference pairs)
- Decision: `failed_hard_gate`

## Scores

| Dimension | Score |
|---|---:|
| ITTE total | 74.39 |
| Style Fidelity | 59.63 |
| Identity Preservation | 74.05 |
| Package Coherence | 86.65 |
| Visual Quality | 93.13 |

## Interpretation

The package is visually coherent and most generated icons retain recognizable image structure. The repeated dog motif, hand-drawn outlines, pastel palette, and rounded light backgrounds are present across the package. The style score remains moderate because VGG Gram and DISTS texture detect a visible rendering/detail gap between the low-resolution reference set and the more polished generated outputs.

The package does not receive a formal pass because hard gates are evaluated per icon and at the lower tail, not only by the mean:

- `hanglvzongheng`: the original green U/ring structure was replaced by a globe, causing a real visual-identity loss.
- `meituan`: the Chinese wordmark remains visible, but the added illustrated panel and dog substantially change dense structure; it is a borderline identity-risk item.
- `tieba`: the source output contains large pure-black corner regions, which is a real artifact in the original generated PNG.
- Package identity P10 is below the provisional threshold because the weakest icon cannot be hidden by stronger icons.

## Determinism And Scope

The main score uses image-only deterministic features: DINOv3 dense patches, VGG Gram, DISTS, LPIPS, visual attributes, robust package statistics, and artifact rules. OpenCLIP, prompts, OCR/text compliance, and Qwen QC do not affect any score or decision. The generation-stage Qwen report is retained only under `diagnostics`.

Thresholds and weights are provisional engineering priors and still require calibration on the proposed development/validation/blind benchmark before being used as paper-level universal claims.
