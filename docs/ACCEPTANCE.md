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

## Repository and delivery

- [x] Full test suite passes in the selected environment (57 tests).
- [ ] `.env`, caches, temporary downloads and machine-specific artifacts are excluded.
- [ ] Large model weights are handled by Git LFS or excluded with reproducible download instructions.
- [ ] GitHub repository/branch and visibility are confirmed before the first push.
