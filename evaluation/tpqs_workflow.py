from __future__ import annotations

from pathlib import Path

from evaluation.services.embedding_service import TpqsConfig, embed_images
from evaluation.services.eval_path_service import resolve_eval_inputs, write_inputs_manifest
from evaluation.services.pca_service import write_embedding_pca
from evaluation.services.report_service import write_tpqs_outputs
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

    manifest_path = write_inputs_manifest(resolved, eval_dir)
    tpqs_config = config or TpqsConfig.from_env()
    image_paths = _all_image_paths(resolved)
    embeddings = embed_images(image_paths, tpqs_config, root_dir=root)
    metrics = compute_tpqs_metrics(resolved, embeddings, tpqs_config)
    output_paths = write_tpqs_outputs(eval_dir, metrics.report, metrics.per_app_rows, metrics.pairwise)
    pca_path = _write_pca(eval_dir, resolved, embeddings)

    return {
        "eval_id": eval_id,
        "eval_dir": str(eval_dir),
        "manifest_path": str(manifest_path),
        "pca_path": str(pca_path),
        **output_paths,
        "report": metrics.report,
    }


def _all_image_paths(resolved) -> list[Path]:
    paths = list(resolved.theme_refs)
    paths.extend(item.path for item in resolved.generated_icons)
    paths.extend(resolved.target_originals[item.app] for item in resolved.generated_icons)
    return paths


def _write_pca(eval_dir: Path, resolved, embeddings: dict) -> Path:
    labels = []
    groups = []
    vectors = []
    for path in resolved.theme_refs:
        labels.append(path.parent.name)
        groups.append("theme")
        vectors.append(embeddings[str(path)])
    for item in resolved.generated_icons:
        labels.append(item.app)
        groups.append("generated")
        vectors.append(embeddings[str(item.path)])
    for item in resolved.generated_icons:
        target_path = resolved.target_originals[item.app]
        labels.append(item.app)
        groups.append("target")
        vectors.append(embeddings[str(target_path)])
    return write_embedding_pca(eval_dir / "embedding_pca.png", labels, groups, vectors)
