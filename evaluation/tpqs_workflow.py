from __future__ import annotations

from pathlib import Path

from evaluation.services.embedding_service import TpqsConfig, embed_images
from evaluation.services.eval_path_service import resolve_eval_inputs, write_inputs_manifest
from evaluation.services.report_service import write_tpqs_outputs
from evaluation.services.style_feature_service import extract_style_feature_groups, extract_style_features
from evaluation.services.tpqs_feedback_service import write_tpqs_feedback_retry_prompt
from evaluation.services.tpqs_service import compute_tpqs_metrics


def run_tpqs(
    theme_id: str,
    package_id: str,
    eval_id: str,
    root_dir: Path | None = None,
    config: TpqsConfig | None = None,
) -> dict:
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1]
    resolved = resolve_eval_inputs(theme_id, package_id, root_dir=root)
    eval_dir = root / "data" / "evaluations" / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(eval_dir)

    manifest_path = write_inputs_manifest(resolved, eval_dir)
    tpqs_config = config or TpqsConfig.from_env()
    image_paths = _all_image_paths(resolved)
    style_features = extract_style_features(image_paths, tpqs_config, root_dir=root)
    style_feature_groups = extract_style_feature_groups(image_paths, tpqs_config, root_dir=root)
    dino_embeddings = embed_images(image_paths, tpqs_config, root_dir=root)
    metrics = compute_tpqs_metrics(
        resolved,
        style_features,
        dino_embeddings,
        tpqs_config,
        style_feature_groups,
        root_dir=root,
    )
    metrics.report["eval_id"] = eval_id
    output_paths = write_tpqs_outputs(
        eval_dir,
        metrics.report,
        metrics.per_app_rows,
        metrics.style_pairwise,
        metrics.style_delta,
        metrics.dino_pairwise,
    )
    retry_prompt_path = write_tpqs_feedback_retry_prompt(metrics.report, eval_dir)

    return {
        "eval_id": eval_id,
        "eval_dir": str(eval_dir),
        "manifest_path": str(manifest_path),
        **output_paths,
        "tpqs_feedback_retry_prompt_path": str(retry_prompt_path),
        "generation_feedback_prompt_path": str(eval_dir / "generation_feedback_prompt.md"),
        "report": metrics.report,
    }


def _all_image_paths(resolved) -> list[Path]:
    paths = list(resolved.theme_refs)
    paths.extend(item.reference_raw_path for item in resolved.theme_examples)
    paths.extend(item.path for item in resolved.generated_icons)
    paths.extend(resolved.target_originals[item.app] for item in resolved.generated_icons)
    return paths


def _remove_stale_outputs(eval_dir: Path) -> None:
    for name in ["embedding_pca.png", "pairwise_distances.json"]:
        path = eval_dir / name
        if path.exists():
            path.unlink()
