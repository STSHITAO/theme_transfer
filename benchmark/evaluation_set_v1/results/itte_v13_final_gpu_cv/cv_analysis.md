# v1.3-image-only five-fold analysis

- Folds: 5 (20260804, 20260805, 20260806, 20260807, 20260808)
- Designer-positive runs: 20
- Original-control runs: 20
- Dataset fingerprint count: 1
- Scoring-config hash count: 1

## Scenario summary

| Scenario | ITTE mean ± std | Style | Identity | Package | Quality | hard-gate acceptance |
|---|---:|---:|---:|---:|---:|---:|
| designer positive | 88.17 ± 6.97 | 90.98 | 81.25 | 84.96 | 99.72 | 95% |
| original control | 43.16 ± 0.67 | 0.00 | 100.00 | 0.00 | 87.72 | 70% |

## Paired designer-minus-control separation

| Dimension | mean | minimum | maximum |
|---|---:|---:|---:|
| itte_score | 45.01 | 28.78 | 53.18 |
| style_fidelity_score | 90.98 | 77.79 | 98.05 |
| identity_preservation_score | -18.75 | -47.73 | 0.00 |
| package_coherence_score | 84.96 | 7.23 | 100.00 |
| visual_quality_score | 12.00 | 5.62 | 20.62 |
| identity_p10 | -43.82 | -83.64 | 0.00 |

## Designer identity diagnostics

| Component | mean | std | min | max |
|---|---:|---:|---:|---:|
| dino_dense | 81.25 | 13.50 | 52.27 | 100.00 |
| dists_structure | 72.14 | 28.44 | 25.00 | 100.00 |
| lpips_content | 71.41 | 22.91 | 27.49 | 100.00 |

- Runs with at least 50 points of identity-component disagreement: 5/20.
- Per-App designer-positive identity scores below the current 35 gate: 9/80.
- Designer-positive identity p10 mean: 56.18; range: 16.36–100.00.
- Designer hard-failure types: `{"severe_package_outliers": 1}`.

## Mechanical conclusion

The unchanged total score consistently separates real designer transfers from the no-transfer control, but the current decision gates do not accept real designer positives consistently. Identity component calibration and small-package tail gating require review before the decision label can be treated as validated.
