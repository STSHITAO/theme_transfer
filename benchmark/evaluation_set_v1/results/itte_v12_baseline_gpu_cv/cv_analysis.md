# v1.2-image-only five-fold analysis

- Folds: 5 (20260804, 20260805, 20260806, 20260807, 20260808)
- Designer-positive runs: 20
- Original-control runs: 20
- Dataset fingerprint count: 1
- Scoring-config hash count: 1

## Scenario summary

| Scenario | ITTE mean ± std | Style | Identity | Package | Quality | hard-gate acceptance |
|---|---:|---:|---:|---:|---:|---:|
| designer positive | 82.98 ± 6.96 | 90.98 | 64.20 | 84.96 | 99.22 | 35% |
| original control | 42.86 ± 0.84 | 0.00 | 100.00 | 0.00 | 85.72 | 60% |

## Paired designer-minus-control separation

| Dimension | mean | minimum | maximum |
|---|---:|---:|---:|
| itte_score | 40.12 | 22.77 | 52.10 |
| style_fidelity_score | 90.98 | 77.79 | 98.05 |
| identity_preservation_score | -35.80 | -74.54 | -8.59 |
| package_coherence_score | 84.96 | 7.23 | 100.00 |
| visual_quality_score | 13.50 | 5.62 | 26.88 |
| identity_p10 | -60.31 | -99.23 | -18.31 |

## Designer identity diagnostics

| Component | mean | std | min | max |
|---|---:|---:|---:|---:|
| dino_dense | 56.55 | 26.05 | 2.62 | 100.00 |
| dists_structure | 72.14 | 28.44 | 25.00 | 100.00 |
| lpips_content | 71.41 | 22.91 | 27.49 | 100.00 |

- Runs with at least 50 points of identity-component disagreement: 9/20.
- Per-App designer-positive identity scores below the current 35 gate: 13/80.
- Designer-positive identity p10 mean: 39.69; range: 0.77–81.69.
- Designer hard-failure types: `{"identity_below_35": 13, "identity_p10_below_45": 12, "severe_package_outliers": 1, "visual_quality": 1}`.

## Mechanical conclusion

The unchanged total score consistently separates real designer transfers from the no-transfer control, but the current decision gates do not accept real designer positives consistently. Identity component calibration and small-package tail gating require review before the decision label can be treated as validated.
