# Acceptance checklist

## Environment

- [x] Conda environment `pytorch` contains CUDA-enabled PyTorch.
- [x] `torch.cuda.is_available()` is true on the current machine.
- [x] DINOv3 and VGG16 official weights pass their recorded SHA-256 checks.
- [x] ITTE dependencies import without replacing CUDA PyTorch.
- [x] DINOv3, VGG16, DISTS and LPIPS complete a CUDA smoke forward pass.
- [x] `TPQS_DEVICE=cpu` completes cache-miss inference for all model families.

## Benchmark integrity

- [x] Independent validator reports `PASS`.
- [x] 91 originals, 231 themed assets, 158 positive pairs and 137 cross-theme pairs resolve.
- [x] Dataset fingerprint is reproducible.
- [x] Reference/query identities are disjoint in the baseline runner.
- [x] Five frozen seeds and every identity-disjoint split are recorded in the reports.

## ITTE baseline

- [x] A CPU smoke run exercised all ITTE components.
- [x] Four-theme designer-positive baseline completes on GPU across five folds.
- [x] Natural no-transfer original control completes on every matched split.
- [x] Every component records availability, reliability and model/device provenance.
- [x] Results are reported per theme, fold and macro average.

## Metric changes

- [x] No metric was changed before the frozen v1.2 five-fold baseline was saved.
- [x] Every retained change is linked to a measured baseline or retrieval failure.
- [x] Candidate and baseline use identical seeds, inputs, splits and dataset fingerprint.
- [x] v1.3 improves real-positive acceptance and total separation; matched style/package scores do not regress.
- [x] Tests and documentation are updated after the decision.

## Structure-policy generation and evaluation

- [x] The structure policy is decided per theme/App before Wan generation.
- [x] `transfer_plan.json` freezes mode, applicability and rationale; inconsistent fields are rejected by evaluation.
- [x] Recomposition cases retain a DINO structure diagnostic but do not contribute it to the primary score.
- [x] Legacy packages continue to use structural identity by default.
- [x] Full automated suite passes with conditional identity coverage (80 tests).
- [ ] Four real 91-App theme packages complete with non-mock API calls.
- [ ] ITTE v1.4 reports complete for all four generated packages and are summarized together.
- [x] A single Wan `DataInspectionFailed` response is recorded and skipped without stopping the remaining batch.
- [x] ITTE reports requested/evaluated/skipped App counts and does not score missing generated images.
- [x] Wan receives exactly one text content item with an explicit style-reference/target role map.
- [x] Final Wan prompts omit the full transfer-plan reasoning and contain no names of the three non-target reference Apps.
- [x] Prompt-v2 focused validation completed on 10 previously polluted Apps with the same references and two candidates each: low-recognition candidates decreased from 15/20 to 4/20; selected low-recognition Apps decreased from 5/10 to 1/10.
- [x] `target.json` identity/function metadata and the final `generation_brief` are compiled into the Wan prompt rather than remaining Qwen-only context.
- [x] No target-level `generation_policy` is used; existing target metadata remains factual and the route is decided from theme-pair evidence plus the target original.
- [x] Theme design analysis covers all valid pairs in deterministic five-pair batches and records coverage instead of learning only the first five alphabetic Apps.
- [x] Per-App Wan references exclude the target itself and prefer examples whose observed structure route matches the frozen target route.
- [x] Wan receives `TARGET_IMAGE` first, followed by three explicitly style-only references.
- [x] The new theme_003 and theme_004 40-App/two-candidate real runs complete with 100% generation/evaluation coverage and receive cached-GPU ITTE reports.
- [x] Experimental results preserve the ITTE hard-gate failures instead of treating high mean scores as acceptance (`xiaohongshu` for theme_003; `wechat` for theme_004).
- [x] The report states that target/reference overlap makes this an in-package reconstruction experiment rather than unseen-App generalization.
- [ ] Replace the current fallback behavior so an App with no identity-passing candidate is not published to `final/` (observed for `crossfire_mobile` in prompt-v2 validation).

## Repository and delivery

- [x] Full test suite passes in the selected environment (80 tests).
- [ ] `.env`, caches, temporary downloads and machine-specific artifacts are excluded.
- [ ] Large model weights are handled by Git LFS or excluded with reproducible download instructions.
- [ ] GitHub repository/branch and visibility are confirmed before the first push.
