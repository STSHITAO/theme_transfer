# Project rules

- Treat `benchmark/evaluation_set_v1/assets` and its labels as immutable test data; derived reports belong under `results/`.
- Run official ITTE work in Conda environment `pytorch`; prefer `TPQS_DEVICE=cuda:0`, while keeping `TPQS_DEVICE=cpu` functional.
- Reuse deterministic on-disk feature caches. Cache keys must include input identity, preprocessing view, model ID and relevant configuration; device must not change numerical cache identity.
- Never use prompt text, Qwen QC or OpenCLIP in the ITTE primary score unless a separately versioned protocol explicitly authorizes it.
- Do not commit `.env`, API keys, temporary downloads or incomplete model files. Large public weights require Git LFS or documented download steps.
- Preserve existing user data. Before changing metrics, run the frozen baseline on the same grouped Benchmark split; after changing them, rerun that split and report both results.
- Add or update tests for behavioral changes. Keep documentation and `docs/PROGRESS.md` synchronized with verified state only.
