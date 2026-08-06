# Theme learning v2 experiment

## Purpose

Validate the corrected generation path in which existing `theme.json` metadata and every real `original -> style_ref` pair teach Qwen when the designer preserves icon structure versus re-expresses software function. Existing `target.json` fields remain neutral facts; no `generation_policy` or hand-authored visual-object field is used.

## Protocol

- Themes: `theme_003`, `theme_004`.
- Targets: the same deterministic evenly spaced 40 Apps from the 91-target list.
- Candidates: two Wan candidates per App, followed by Qwen multimodal selection.
- Theme learning: five pairs per Qwen batch, then text-only aggregate conditional policy.
- Wan image order: target first, then three style-only references.
- Per-App references: current target excluded; examples with the selected structure route preferred; deterministic spread within the eligible pool.
- Evaluation: unchanged ITTE v1.4 on Conda `pytorch`, `cuda:0`, batch size 2, persistent feature cache.
- API coverage: 40/40 successful Apps for each theme; no content-inspection or entitlement skip.

The 40-target set is identical across themes, so the cross-theme evaluated intersection is 40 Apps.

## Theme-learning evidence

| Theme | Real pairs analyzed | Qwen batches | Observed preserve pairs | Observed semantic-recompose pairs | Generated preserve cases | Generated semantic-recompose cases |
|---|---:|---:|---:|---:|---:|---:|
| theme_003 | 86 | 18 | 61 | 25 | 29 | 11 |
| theme_004 | 36 | 8 | 20 | 16 | 26 | 14 |

The result confirms that the model no longer treats a theme as one global binary mode. It learns a conditional policy and freezes the route per theme/App before generation.

## Generation QC

| Theme | Candidates | Final outputs | Cases marked `needs_retry` | Apps |
|---|---:|---:|---:|---|
| theme_003 | 80 | 40 | 4 | `golden_spatula`, `keep`, `tencent_meeting`, `toutiao_lite` |
| theme_004 | 80 | 40 | 7 | `bilibili`, `jianying`, `kugou`, `lol_mobile`, `qq`, `soul`, `youku` |

The lowest selected-QC identity failures were `tencent_meeting` and `toutiao_lite` in theme_003, and `youku`, `kugou`, and `bilibili` in theme_004. These artifacts remain published for objective evaluation and diagnosis; Qwen QC is not used in the ITTE score.

## ITTE results

| Theme | ITTE | Style fidelity | Identity preservation | Identity applicable / diagnostic-only | Package coherence | Visual quality | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| theme_003 | 89.03 | 90.19 | 94.34 | 29 / 11 | 70.96 | 99.81 | `failed_hard_gate` |
| theme_004 | 87.27 | 86.08 | 89.90 | 26 / 14 | 76.69 | 98.88 | `failed_hard_gate` |

Coverage is 40/40 for both reports. The hard-gate failures are:

- theme_003: `xiaohongshu` is a severe package outlier. Its identity and quality are both 100 and style fidelity is 91.09, but the package-membership distribution still marks it as an outlier.
- theme_004: `wechat` is a severe package outlier. Its identity and quality are both 100, but style fidelity is 61.55 and package membership is 62.89.

Qwen candidate QC scored the selected `xiaohongshu` at 97 and `wechat` at 96. ITTE still rejected the packages. This is useful disagreement: generation QC judged target-specific prompt compliance, while ITTE detected package-relative visual deviation. It confirms why Qwen QC must remain diagnostic and outside the objective score.

## Interpretation and validity boundary

The corrected path materially improves provenance and avoids the former fixed `alipay/bilibili/douban` reference concentration. It does not yet satisfy the package hard gate: one severe package outlier remains in each theme, and theme_004 has more candidate-level identity retries and lower style fidelity.

This is an in-package reconstruction experiment, not a held-out generalization benchmark. Because the normalized data intentionally permits the same App in `styles` and `targets`, 38/40 theme_003 targets and 16/40 theme_004 targets also occur among the pairs used to learn the aggregate theme policy. The current raw Wan references always exclude the target itself, but aggregate Qwen theme evidence may include that App's designer pair. Scores must therefore not be presented as unseen-App generalization. A future held-out test would need target-wise leave-one-out theme evidence while retaining the same target list for designer comparison.

## Artifacts

- `data/packages/package_theme_learning_v2_theme_003/`
- `data/packages/package_theme_learning_v2_theme_004/`
- `data/evaluations/eval_theme_learning_v2_theme_003/`
- `data/evaluations/eval_theme_learning_v2_theme_004/`
- `data/evaluations/eval_theme_learning_v2_summary.json`
