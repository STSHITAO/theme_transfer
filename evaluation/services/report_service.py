from __future__ import annotations

import csv
import json
from pathlib import Path


def write_tpqs_outputs(
    eval_dir: Path,
    report: dict,
    per_app_rows: list[dict],
    style_pairwise: dict,
    style_delta: dict,
    dino_pairwise: dict,
) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "tpqs_report.json"
    metrics_path = eval_dir / "metrics.csv"
    style_pairwise_path = eval_dir / "style_pairwise_distances.json"
    style_delta_path = eval_dir / "style_delta_distances.json"
    dino_pairwise_path = eval_dir / "dino_pairwise_distances.json"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    style_pairwise_path.write_text(json.dumps(style_pairwise, ensure_ascii=False, indent=2), encoding="utf-8")
    style_delta_path.write_text(json.dumps(style_delta, ensure_ascii=False, indent=2), encoding="utf-8")
    dino_pairwise_path.write_text(json.dumps(dino_pairwise, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "app",
        "itte_score",
        "style_transfer_score",
        "theme_style_image_transfer_score",
        "style_cue_profile_match_score",
        "theme_prompt_image_alignment_score",
        "style_attribute_transfer_score",
        "color_transfer_score",
        "background_transfer_score",
        "stroke_transfer_score",
        "texture_material_transfer_score",
        "composition_transfer_score",
        "complexity_transfer_score",
        "package_coherence_score",
        "package_unity_score",
        "app_identity_coherence_score",
        "target_structure_retention_score",
        "over_recomposition_penalty",
        "qwen_identity_score",
        "dino_identity_structure_risk_score",
        "visual_quality_score",
        "visual_artifact_quality_score",
        "strict_delta_transfer_score",
        "theme_style_image_fit_score",
        "theme_style_text_fit_score",
        "style_delta_transfer_score",
        "d_to_reference_delta_centroid",
        "theme_membership_score",
        "d_to_theme_style_centroid",
        "is_style_outlier",
        "generated_internal_distance",
        "generated_internal_outlier",
        "matched_target_app",
        "identity_match_correct",
        "generated_to_own_target_similarity",
        "max_target_similarity",
        "strict_delta_warning",
        "membership_warning",
        "qwen_problematic_app",
        "visual_quality_distance_to_theme",
        "semantic_fit_score",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_app_rows:
            writer.writerow(row)

    return {
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "style_pairwise_path": str(style_pairwise_path),
        "style_delta_path": str(style_delta_path),
        "dino_pairwise_path": str(dino_pairwise_path),
    }
