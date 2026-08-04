# DINOv3 dense identity retrieval on real pairs

Each designer-themed icon is a query. Candidate identities are the original icons from the same theme. The correct candidate is defined only by the verified canonical label; other labels are objective distractors.

- Device: `cuda:0`
- Model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Total verified pairs: 158

| Theme | pairs | top-1 | top-5 | MRR | pairwise AUC | mean top-1 margin |
|---|---:|---:|---:|---:|---:|---:|
| theme_001 | 24 | 70.8% | 75.0% | 0.735 | 0.842 | 0.0226 |
| theme_002 | 12 | 66.7% | 66.7% | 0.703 | 0.735 | -0.0121 |
| theme_003 | 86 | 48.8% | 66.3% | 0.578 | 0.846 | -0.0202 |
| theme_004 | 36 | 36.1% | 52.8% | 0.465 | 0.803 | -0.0274 |
| **macro/overall** | 158 | 50.6% | 64.6% | 0.586 | 0.827 | -0.0147 |

This test validates identity discrimination without generated degradations or human ratings. It does not by itself validate an absolute 0–100 calibration or a hard acceptance threshold.
