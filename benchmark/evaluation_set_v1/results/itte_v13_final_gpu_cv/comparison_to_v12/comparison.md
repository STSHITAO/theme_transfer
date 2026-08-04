# ITTE v1.2 → v1.3 matched five-fold comparison

- Matched runs: 40
- Same seeds, themes, reference/query identities and dataset fingerprint: yes

| Measure | v1.2 | v1.3 | paired change |
|---|---:|---:|---:|
| designer hard-gate acceptance | 35% | 95% | +60% |
| designer itte_score | 82.98 | 88.17 | +5.19 |
| designer style_fidelity_score | 90.98 | 90.98 | +0.00 |
| designer identity_preservation_score | 64.20 | 81.25 | +17.05 |
| designer package_coherence_score | 84.96 | 84.96 | +0.00 |
| designer visual_quality_score | 99.22 | 99.72 | +0.50 |
| designer identity_p10 | 39.69 | 56.18 | +16.49 |
| control ITTE | 42.86 | 43.16 | +0.30 |
| designer-control ITTE separation | 40.12 | 45.01 | +4.89 |

## Decision

Retain v1.3. It removes identity metrics that were near random in verified-label retrieval from the primary identity score, replaces unstable absolute normalization with a same-run DINO identity-gallery percentile, and removes identity hard thresholds unsupported by the available data. The matched result improves real-positive acceptance and designer/control separation without changing style or package scores. The remaining real-positive rejection is a package-coherence outlier and remains visible.
