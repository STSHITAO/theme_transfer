from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.services.eval_path_service import ResolvedEvalInputs
from evaluation.services.style_cue_profile_service import compute_style_cue_profile_match
from evaluation.services.style_text_clip_service import (
    OpenClipBackend,
    compute_theme_style_text_fit,
    disabled_theme_style_text_fit,
)
from evaluation.services.vgg_gram_service import compute_vgg_gram_style_fit
from evaluation.services.visual_stats_service import visual_statistics_score


STYLE_ATTRIBUTE_NAMES = [
    "color",
    "background",
    "stroke",
    "texture_material",
    "composition",
    "complexity",
]
ATTRIBUTE_IMPORTANCE_WEIGHTS = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}


@dataclass(frozen=True)
class TpqsMetrics:
    report: dict
    per_app_rows: list[dict]
    style_pairwise: dict
    style_delta: dict
    dino_pairwise: dict


def compute_tpqs_metrics(
    resolved: ResolvedEvalInputs,
    style_features: dict[str, np.ndarray],
    dino_embeddings: dict[str, np.ndarray],
    config,
    style_feature_groups: dict[str, dict[str, np.ndarray]] | None = None,
    root_dir: Path | None = None,
) -> TpqsMetrics:
    theme_paths = resolved.theme_refs
    reference_raw_paths = [item.reference_raw_path for item in resolved.theme_examples]
    generated_paths = [item.path for item in resolved.generated_icons]
    target_paths = [resolved.target_originals[item.app] for item in resolved.generated_icons]

    theme_style = _stack_embeddings(theme_paths, style_features)
    reference_raw_style = _stack_embeddings(reference_raw_paths, style_features)
    generated_style = _stack_embeddings(generated_paths, style_features)
    target_style = _stack_embeddings(target_paths, style_features)
    theme_dino = _stack_embeddings(theme_paths, dino_embeddings)
    generated_dino = _stack_embeddings(generated_paths, dino_embeddings)
    target_dino = _stack_embeddings(target_paths, dino_embeddings)

    style_rr = pairwise_distances(theme_style)
    style_gg = pairwise_distances(generated_style)
    style_tt = pairwise_distances(target_style)
    style_gr = cross_distances(generated_style, theme_style)
    style_tr = cross_distances(target_style, theme_style)
    dino_rr = pairwise_distances(theme_dino)
    dino_gg = pairwise_distances(generated_dino)
    dino_tt = pairwise_distances(target_dino)
    dino_gr = cross_distances(generated_dino, theme_dino)
    dino_tr = cross_distances(target_dino, theme_dino)
    dino_gt = cross_distances(generated_dino, target_dino)
    grouped_style = style_feature_groups or _fallback_style_feature_groups(style_features)

    final_style_membership = _style_transfer_score(theme_style, generated_style, target_style)
    delta_transfer = _style_delta_transfer_score(
        reference_raw_style,
        theme_style,
        target_style,
        generated_style,
        [item.app for item in resolved.generated_icons],
        _group_matrices(reference_raw_paths, grouped_style),
        _group_matrices(theme_paths, grouped_style),
        _group_matrices(target_paths, grouped_style),
        _group_matrices(generated_paths, grouped_style),
    )
    internal_consistency = _package_consistency_score(style_gg)
    reference_match = _reference_style_distribution_match_score(style_gg, style_rr, style_tt)
    membership = _theme_membership_score(
        theme_style,
        generated_style,
        target_style,
        [item.app for item in resolved.generated_icons],
    )
    identity = _identity_separability_score(dino_gt, resolved.generated_icons)
    visual = visual_statistics_score(theme_paths, generated_paths, target_paths, config.image_size)
    package_dir = _package_dir_from_resolved(resolved)
    root = Path(root_dir) if root_dir else package_dir.parents[2]
    qwen_qc = _load_qwen_qc_scores(package_dir)
    theme_design_analysis = _load_theme_design_analysis(package_dir)
    qwen_instruction_text = _load_qwen_instruction_text(package_dir)
    theme_style_text_fit = _theme_style_text_fit_score(
        config,
        root,
        theme_paths,
        generated_paths,
        target_paths,
        theme_design_analysis,
        qwen_instruction_text,
    )
    style_cue_profile_match = compute_style_cue_profile_match(
        theme_paths,
        generated_paths,
        target_paths,
        config.image_size,
    )
    style_attribute_transfer = _style_attribute_transfer_scores(
        theme_paths,
        reference_raw_paths,
        generated_paths,
        target_paths,
        grouped_style,
        theme_design_analysis,
    )
    style_transfer_effectiveness = _style_transfer_effectiveness(
        theme_style_image_transfer_score=final_style_membership["score"],
        style_cue_profile_match_score=style_cue_profile_match["score"],
        attribute_transfer_score=style_attribute_transfer["score"],
        theme_style_text_fit=theme_style_text_fit,
    )

    strict_scores = [
        delta_transfer["score"],
        internal_consistency["score"],
        reference_match["score"],
        membership["score"],
        identity["score"],
        visual["score"],
        visual["artifact_quality_score"],
    ]
    strict_delta_tpqs_score = _geometric_mean(strict_scores)
    strict_delta_diagnostics = {
        "strict_delta_transfer_score": delta_transfer["score"],
        "color_delta_score": delta_transfer["group_scores"]["color"]["score"],
        "edge_delta_score": delta_transfer["group_scores"]["edge"]["score"],
        "composition_delta_score": delta_transfer["group_scores"]["composition"]["score"],
        "complexity_delta_score": delta_transfer["group_scores"]["complexity"]["score"],
        "visual_stats_transfer_score": visual["score"],
        "interpretation": "Only evaluates whether target->generated follows reference_raw->style_ref low-level statistical direction.",
    }
    package_coherence = {
        "score": _geometric_mean([internal_consistency["score"], membership["score"]]),
        "package_unity_score": internal_consistency["score"],
        "theme_membership_score": membership["score"],
        "generated_internal_outlier_apps": membership["generated_internal_outlier_apps"],
    }
    risk_scores = _risk_scores(identity)
    app_names = [item.app for item in resolved.generated_icons]
    app_identity_coherence = _app_identity_coherence_score(
        identity=identity,
        qwen_qc=qwen_qc,
        dino_gt=dino_gt,
        app_names=app_names,
        style_transfer_score=style_transfer_effectiveness["score"],
    )
    app_identity_coherence["dino_identity_structure_risk_score"] = risk_scores["dino_identity_structure_risk_score"]
    app_identity_coherence["dino_identity_warning_apps"] = risk_scores["dino_identity_warning_apps"]
    visual_quality = {
        "score": visual["artifact_quality_score"],
        "visual_artifact_quality_score": visual["artifact_quality_score"],
        "artifact_warnings": visual["artifact_warnings"],
    }
    total = _weighted_score(
        [
            (style_transfer_effectiveness["score"], 0.40),
            (package_coherence["score"], 0.20),
            (app_identity_coherence["score"], 0.25),
            (visual_quality["score"], 0.15),
        ]
    )
    decision = _itte_decision(
        style_transfer_score=style_transfer_effectiveness["score"],
        package_coherence_score=package_coherence["score"],
        app_identity_score=app_identity_coherence["score"],
        visual_quality_score=visual_quality["score"],
        strict_delta_score=strict_delta_diagnostics["strict_delta_transfer_score"],
    )
    warnings = _itte_warnings(
        style_transfer_effectiveness,
        package_coherence,
        app_identity_coherence,
        visual_quality,
        strict_delta_diagnostics,
        qwen_qc,
    )
    failed_reasons = _itte_failed_reasons(warnings)
    auxiliary_scores = _auxiliary_scores(config, root, theme_paths, generated_paths, target_paths)
    primary_scores = {
        "theme_style_image_fit_score": final_style_membership["score"],
        "theme_style_text_fit_score": theme_style_text_fit["score"],
        "style_cue_profile_match_score": style_cue_profile_match["score"],
        "theme_prompt_image_alignment_score": theme_style_text_fit["score"],
        "package_unity_score": internal_consistency["score"],
        "theme_membership_score": membership["score"],
        "visual_artifact_quality_score": visual["artifact_quality_score"],
    }
    diagnostic_scores = {
        "style_delta_transfer_score": delta_transfer["score"],
        "color_delta_score": delta_transfer["group_scores"]["color"]["score"],
        "edge_delta_score": delta_transfer["group_scores"]["edge"]["score"],
        "composition_delta_score": delta_transfer["group_scores"]["composition"]["score"],
        "complexity_delta_score": delta_transfer["group_scores"]["complexity"]["score"],
        "visual_stats_transfer_score": visual["score"],
    }
    tpqs_summary = _tpqs_summary(delta_transfer, internal_consistency, membership, identity, visual)
    diagnosis_summary = _itte_diagnosis_summary(
        decision,
        failed_reasons,
        warnings,
        style_transfer_effectiveness,
        package_coherence,
        app_identity_coherence,
        visual_quality,
        strict_delta_diagnostics,
        primary_scores,
        diagnostic_scores,
        risk_scores,
        qwen_qc,
        delta_transfer,
        internal_consistency,
        membership,
        identity,
        visual,
    )
    report = {
        "eval_id": None,
        "evaluation_framework": "ITTE",
        "itte_version": "v1.1",
        "theme_id": resolved.theme_id,
        "package_id": resolved.package_id,
        "itte_score": total,
        "tpqs": total,
        "tpqs_primary_score": total,
        "tpqs_total_score": total,
        "strict_delta_tpqs_score": strict_delta_tpqs_score,
        "legacy_tpqs_strict_delta_score": strict_delta_tpqs_score,
        "style_transfer_score": style_transfer_effectiveness["score"],
        "legacy_style_delta_transfer_score": delta_transfer["score"],
        "style_delta_transfer_score": delta_transfer["score"],
        "color_delta_score": delta_transfer["group_scores"]["color"]["score"],
        "edge_delta_score": delta_transfer["group_scores"]["edge"]["score"],
        "composition_delta_score": delta_transfer["group_scores"]["composition"]["score"],
        "complexity_delta_score": delta_transfer["group_scores"]["complexity"]["score"],
        "final_style_membership_score": final_style_membership["score"],
        "theme_style_image_transfer_score": final_style_membership["score"],
        "style_cue_profile_match_score": style_cue_profile_match["score"],
        "theme_prompt_image_alignment_score": theme_style_text_fit["score"],
        "style_attribute_transfer_score": style_attribute_transfer["score"],
        "package_internal_style_consistency_score": internal_consistency["score"],
        "reference_style_distribution_match_score": reference_match["score"],
        "package_consistency_score": internal_consistency["score"],
        "theme_membership_score": membership["score"],
        "identity_separability_score": identity["score"],
        "visual_quality_score": visual["artifact_quality_score"],
        "visual_stats_transfer_score": visual["score"],
        "visual_artifact_quality_score": visual["artifact_quality_score"],
        "visual_statistics_score": visual["score"],
        "embedding_backend": config.embedding_backend,
        "model_source": config.model_source,
        "model_id": config.model_id,
        "style_feature_backend": config.style_feature_backend,
        "is_official_tpqs": config.is_official_tpqs,
        "decision": decision,
        "failed_reasons": failed_reasons,
        "warnings": warnings,
        "diagnosis_summary": diagnosis_summary,
        "tpqs_summary": tpqs_summary,
        "style_transfer_effectiveness": style_transfer_effectiveness,
        "style_cue_profile_match": style_cue_profile_match,
        "style_attribute_transfer_scores": style_attribute_transfer["attributes"],
        "package_coherence": package_coherence,
        "app_identity_coherence": app_identity_coherence,
        "visual_quality": visual_quality,
        "strict_delta_diagnostics": strict_delta_diagnostics,
        "diagnostics": {
            "strict_delta_diagnostics": strict_delta_diagnostics,
            "generation_qwen_qc_prior": qwen_qc,
            "text_policy_diagnostics": {
                "enabled": False,
                "reason": "Text policy is diagnostic-only and does not affect ITTE v1.1 score.",
            },
        },
        "auxiliary_scores": auxiliary_scores,
        "primary_scores": primary_scores,
        "diagnostic_scores": diagnostic_scores,
        "risk_scores": risk_scores,
        "qwen_qc_scores": qwen_qc,
        "outlier_apps": membership["generated_internal_outlier_apps"],
        "identity_top1_accuracy": identity["top1_accuracy"],
        "identity_random_baseline": identity["random_baseline"],
        "flags": {
            "is_style_transfer_effective": delta_transfer["is_style_delta_transfer_effective"],
            "is_style_delta_transfer_effective": delta_transfer["is_style_delta_transfer_effective"],
            "is_package_style_consistency_improved": reference_match["is_package_style_consistency_improved"],
            "is_package_consistent": internal_consistency["is_package_consistent"],
            "has_style_outliers": membership["has_generated_internal_outliers"],
            "has_generated_internal_outliers": membership["has_generated_internal_outliers"],
            "identity_above_random": identity["identity_above_random"],
            "is_visual_stats_improved": visual["is_visual_stats_improved"],
            "is_visual_artifact_quality_ok": visual["is_artifact_quality_ok"],
        },
        "feature_backends": {
            "style_features": config.style_feature_backend,
            "dino_backend": config.embedding_backend,
            "dino_model_id": config.model_id,
            "dino_model_source": config.model_source,
            "openclip_enabled": bool(config.use_openclip),
            "openclip_model": config.openclip_model,
            "openclip_pretrained": config.openclip_pretrained,
            "device": config.device,
            "pooling": config.pooling,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
        },
        "details": {
            "style_transfer": delta_transfer,
            "style_delta_transfer": delta_transfer,
            "final_style_membership": final_style_membership,
            "style_cue_profile_match": style_cue_profile_match,
            "package_internal_style_consistency": internal_consistency,
            "reference_style_distribution_match": reference_match,
            "theme_style_text_fit": theme_style_text_fit,
            "theme_membership": {
                key: value for key, value in membership.items() if key != "per_app"
            },
            "identity_separability": {
                key: value for key, value in identity.items() if key != "per_app"
            },
            "visual_statistics": visual,
            "visual_quality": visual,
        },
    }

    per_app_rows = []
    identity_rows = {row["app"]: row for row in identity["per_app"]}
    delta_rows = {row["app"]: row for row in delta_transfer["per_app"]}
    style_cue_rows = {row["app"]: row for row in style_cue_profile_match["per_app"]}
    structure_rows = {row["app"]: row for row in app_identity_coherence["target_structure_per_app"]}
    qwen_problematic_apps = set(qwen_qc.get("problematic_apps") or [])
    strict_delta_warning = delta_transfer["score"] < 40.0
    for row in membership["per_app"]:
        identity_row = identity_rows[row["app"]]
        delta_row = delta_rows[row["app"]]
        style_cue_row = style_cue_rows.get(row["app"], {})
        structure_row = structure_rows.get(row["app"], {})
        per_app_rows.append(
            {
                "app": row["app"],
                "itte_score": round(total, 4),
                "style_transfer_score": round(style_transfer_effectiveness["score"], 4),
                "theme_style_image_transfer_score": round(final_style_membership["score"], 4),
                "style_cue_profile_match_score": round(
                    style_cue_row.get("style_cue_profile_match_score", style_cue_profile_match["score"]), 4
                ),
                "theme_prompt_image_alignment_score": ""
                if theme_style_text_fit["score"] is None
                else round(theme_style_text_fit["score"], 4),
                "style_attribute_transfer_score": ""
                if style_attribute_transfer["score"] is None
                else round(style_attribute_transfer["score"], 4),
                "color_transfer_score": round(style_attribute_transfer["attributes"]["color"]["score"], 4),
                "background_transfer_score": round(style_attribute_transfer["attributes"]["background"]["score"], 4),
                "stroke_transfer_score": round(style_attribute_transfer["attributes"]["stroke"]["score"], 4),
                "texture_material_transfer_score": round(
                    style_attribute_transfer["attributes"]["texture_material"]["score"], 4
                ),
                "composition_transfer_score": round(style_attribute_transfer["attributes"]["composition"]["score"], 4),
                "complexity_transfer_score": round(style_attribute_transfer["attributes"]["complexity"]["score"], 4),
                "package_coherence_score": round(package_coherence["score"], 4),
                "app_identity_coherence_score": round(app_identity_coherence["score"], 4),
                "target_structure_retention_score": round(
                    structure_row.get("target_structure_retention_score", app_identity_coherence["target_structure_retention_score"]),
                    4,
                ),
                "over_recomposition_penalty": round(app_identity_coherence["over_recomposition_penalty"], 4),
                "qwen_identity_score": ""
                if app_identity_coherence["qwen_identity_score"] is None
                else round(app_identity_coherence["qwen_identity_score"], 4),
                "visual_quality_score": round(visual_quality["score"], 4),
                "strict_delta_transfer_score": round(delta_row["style_delta_transfer_score"], 4),
                "theme_style_image_fit_score": round(final_style_membership["score"], 4),
                "theme_style_text_fit_score": "" if theme_style_text_fit["score"] is None else round(theme_style_text_fit["score"], 4),
                "package_unity_score": round(internal_consistency["score"], 4),
                "visual_artifact_quality_score": round(visual["artifact_quality_score"], 4),
                "dino_identity_structure_risk_score": round(risk_scores["dino_identity_structure_risk_score"], 4),
                "style_delta_transfer_score": round(delta_row["style_delta_transfer_score"], 4),
                "d_to_reference_delta_centroid": round(delta_row["d_to_reference_delta_centroid"], 6),
                "theme_membership_score": round(row["theme_membership_score"], 4),
                "d_to_theme_style_centroid": round(row["d_to_theme_style_centroid"], 6),
                "is_style_outlier": row["is_style_outlier"],
                "generated_internal_distance": round(row["generated_internal_distance"], 6),
                "generated_internal_outlier": row["generated_internal_outlier"],
                "matched_target_app": identity_row["matched_target_app"],
                "identity_match_correct": identity_row["identity_match_correct"],
                "generated_to_own_target_similarity": round(identity_row["generated_to_own_target_similarity"], 6),
                "max_target_similarity": round(identity_row["max_target_similarity"], 6),
                "strict_delta_warning": strict_delta_warning,
                "membership_warning": row["generated_internal_outlier"],
                "qwen_problematic_app": row["app"] in qwen_problematic_apps,
                "visual_quality_distance_to_theme": "",
                "semantic_fit_score": "",
            }
        )

    style_pairwise = {
        "theme_ref_pairwise": style_rr.tolist(),
        "reference_raw_pairwise": pairwise_distances(reference_raw_style).tolist(),
        "generated_pairwise": style_gg.tolist(),
        "target_pairwise": style_tt.tolist(),
        "generated_to_theme": style_gr.tolist(),
        "target_to_theme": style_tr.tolist(),
        "app_names": [item.app for item in resolved.generated_icons],
        "theme_example_names": [item.app for item in resolved.theme_examples],
    }
    style_delta = {
        "reference_delta_pairwise": delta_transfer["reference_delta_pairwise"],
        "generated_delta_pairwise": delta_transfer["generated_delta_pairwise"],
        "generated_delta_to_reference_delta_centroid": delta_transfer["generated_delta_to_reference_delta_centroid"],
        "no_change_delta_to_reference_delta_centroid": delta_transfer["no_change_delta_to_reference_delta_centroid"],
        "reference_delta_leave_one_out_distance": delta_transfer["D_R_delta"],
        "group_scores": delta_transfer["group_scores"],
        "app_names": [item.app for item in resolved.generated_icons],
        "theme_example_names": [item.app for item in resolved.theme_examples],
    }
    dino_pairwise = {
        "theme_ref_pairwise": dino_rr.tolist(),
        "generated_pairwise": dino_gg.tolist(),
        "target_pairwise": dino_tt.tolist(),
        "generated_to_theme": dino_gr.tolist(),
        "target_to_theme": dino_tr.tolist(),
        "generated_to_target": dino_gt.tolist(),
        "app_names": [item.app for item in resolved.generated_icons],
    }
    return TpqsMetrics(
        report=report,
        per_app_rows=per_app_rows,
        style_pairwise=style_pairwise,
        style_delta=style_delta,
        dino_pairwise=dino_pairwise,
    )


def _style_transfer_effectiveness(
    theme_style_image_transfer_score: float,
    style_cue_profile_match_score: float | None = None,
    attribute_transfer_score: float | None = None,
    theme_style_text_fit: dict | None = None,
) -> dict:
    if theme_style_text_fit is None:
        theme_style_text_fit = {}
    weighted_components = []
    included = []
    if style_cue_profile_match_score is not None:
        weighted_components.append((float(style_cue_profile_match_score), 0.20))
        included.append("style_cue_profile_match_score")
    weighted_components.append((float(theme_style_image_transfer_score), 0.45))
    included.append("theme_style_image_transfer_score")
    if attribute_transfer_score is not None:
        weighted_components.append((float(attribute_transfer_score), 0.15))
        included.append("style_attribute_transfer_score")
    text_score = theme_style_text_fit.get("score")
    text_reliable = bool(theme_style_text_fit.get("theme_style_text_fit_reliable"))
    if text_score is not None and text_reliable:
        weighted_components.append((float(text_score), 0.20))
        included.append("theme_prompt_image_alignment_score")
    score = _weighted_score(weighted_components)
    return {
        "itte_style_transfer_version": "v1.1",
        "score": score,
        "style_cue_profile_match_score": style_cue_profile_match_score,
        "theme_style_image_transfer_score": float(theme_style_image_transfer_score),
        "style_attribute_transfer_score": attribute_transfer_score,
        "theme_prompt_image_alignment_score": text_score,
        "theme_style_text_fit_score": text_score,
        "theme_style_text_fit_reliable": text_reliable,
        "included_components": included,
        "component_weights": {name: weight for name, (_, weight) in zip(included, weighted_components)},
    }


def _style_attribute_transfer_scores(
    theme_paths: list[Path],
    reference_raw_paths: list[Path],
    generated_paths: list[Path],
    target_paths: list[Path],
    grouped_style: dict[str, dict[str, np.ndarray]],
    theme_design_analysis: dict,
) -> dict:
    attributes = {}
    weighted_scores = []
    weights = []
    salient = _salient_attribute_importance(theme_design_analysis)
    for name in STYLE_ATTRIBUTE_NAMES:
        theme = _single_attribute_matrix(theme_paths, grouped_style, name)
        reference_raw = _single_attribute_matrix(reference_raw_paths, grouped_style, name)
        generated = _single_attribute_matrix(generated_paths, grouped_style, name)
        target = _single_attribute_matrix(target_paths, grouped_style, name)
        transfer = _single_attribute_transfer_score(theme, generated, target)
        reliability = _attribute_reliability(name, theme, reference_raw, target, salient)
        item = {
            "score": transfer["score"],
            "reliable": reliability["reliable"],
            "importance": reliability["importance"],
            "weight": reliability["importance_weight"],
            "reliability_score": reliability["reliability_score"],
            "reason": reliability["reason"],
            "D_TR_attr": transfer["D_TR_attr"],
            "D_GR_attr": transfer["D_GR_attr"],
            "D_RR_attr": transfer["D_RR_attr"],
        }
        attributes[name] = item
        if item["reliable"]:
            weight = item["weight"] * item["reliability_score"]
            weighted_scores.append(item["score"] * weight)
            weights.append(weight)
    aggregate = None if not weights else float(sum(weighted_scores) / max(sum(weights), 1e-8))
    return {"score": aggregate, "attributes": attributes}


def _single_attribute_transfer_score(theme: np.ndarray, generated: np.ndarray, targets: np.ndarray) -> dict:
    centroid = _normalized_centroid(theme)
    d_rr = _theme_leave_one_out_distance(theme)
    d_gr = float(np.mean([_cosine_distance(item, centroid) for item in generated]))
    d_tr = float(np.mean([_cosine_distance(item, centroid) for item in targets]))
    score = max(0.0, min(100.0, (d_tr - d_gr) / max(d_tr - d_rr, 1e-8) * 100.0))
    return {
        "score": score,
        "D_TR_attr": d_tr,
        "D_GR_attr": d_gr,
        "D_RR_attr": d_rr,
    }


def _attribute_reliability(
    name: str,
    theme: np.ndarray,
    reference_raw: np.ndarray,
    targets: np.ndarray,
    salient: dict[str, str],
) -> dict:
    d_rr = _theme_leave_one_out_distance(theme)
    centroid = _normalized_centroid(theme)
    d_tr = float(np.mean([_cosine_distance(item, centroid) for item in targets]))
    raw_to_theme = cross_distances(reference_raw, theme)
    raw_theme_distance = float(np.mean(raw_to_theme))
    theme_consistency = max(0.0, min(1.0, 1.0 - d_rr / 0.55))
    target_theme_separability = max(0.0, min(1.0, (d_tr - d_rr) / 0.45))
    raw_theme_salience = max(0.0, min(1.0, raw_theme_distance / 0.45))
    importance = salient.get(name) or _auto_importance(theme_consistency, target_theme_separability, raw_theme_salience)
    importance_weight = ATTRIBUTE_IMPORTANCE_WEIGHTS.get(importance, 0.6)
    reliability_score = theme_consistency * target_theme_separability * importance_weight
    reliable = reliability_score >= 0.02
    reason = (
        f"theme_consistency={theme_consistency:.2f}; "
        f"target_theme_separability={target_theme_separability:.2f}; "
        f"raw_theme_salience={raw_theme_salience:.2f}; importance={importance}"
    )
    if not reliable:
        reason += "; excluded from style_attribute_transfer_score due to low reliability"
    return {
        "reliable": reliable,
        "importance": importance,
        "importance_weight": importance_weight,
        "reliability_score": reliability_score,
        "reason": reason,
    }


def _auto_importance(theme_consistency: float, target_theme_separability: float, raw_theme_salience: float) -> str:
    value = 0.45 * theme_consistency + 0.35 * target_theme_separability + 0.20 * raw_theme_salience
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "medium"
    return "low"


def _salient_attribute_importance(theme_design_analysis: dict) -> dict[str, str]:
    result = {}
    items = theme_design_analysis.get("salient_style_attributes", [])
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        importance = str(item.get("importance", "")).strip().lower()
        if name in STYLE_ATTRIBUTE_NAMES and importance in ATTRIBUTE_IMPORTANCE_WEIGHTS:
            result[name] = importance
    return result


def _single_attribute_matrix(
    paths: list[Path],
    grouped_style: dict[str, dict[str, np.ndarray]],
    attribute: str,
) -> np.ndarray:
    matrix = np.vstack([grouped_style[str(path)][attribute] for path in paths]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def _itte_decision(
    style_transfer_score: float,
    package_coherence_score: float,
    app_identity_score: float,
    visual_quality_score: float,
    strict_delta_score: float | None = None,
) -> str:
    if visual_quality_score < 50.0:
        return "visual_quality_problem"
    if style_transfer_score >= 70.0 and package_coherence_score >= 70.0 and app_identity_score < 70.0:
        return "style_transfer_success_with_identity_risk"
    if style_transfer_score >= 70.0 and package_coherence_score >= 70.0 and app_identity_score >= 70.0:
        return "style_transfer_success"
    if package_coherence_score >= 70.0 and style_transfer_score < 60.0:
        return "package_coherent_but_theme_transfer_weak"
    if style_transfer_score >= 70.0 and package_coherence_score < 60.0:
        return "theme_transfer_effective_but_package_unity_weak"
    return "needs_review"


def _itte_warnings(
    style_transfer_effectiveness: dict,
    package_coherence: dict,
    app_identity_coherence: dict,
    visual_quality: dict,
    strict_delta_diagnostics: dict,
    qwen_qc: dict,
) -> dict:
    warnings = {
        "style_transfer_warnings": [],
        "package_coherence_warnings": [],
        "identity_warnings": [],
        "quality_warnings": [],
        "strict_delta_warnings": [],
        "low_confidence_warnings": [],
    }
    if style_transfer_effectiveness["score"] < 60.0:
        warnings["style_transfer_warnings"].append("style_transfer_effectiveness_weak")
    if style_transfer_effectiveness["style_attribute_transfer_score"] is None:
        warnings["low_confidence_warnings"].append("no_reliable_style_attributes")
    if style_transfer_effectiveness["theme_style_text_fit_score"] is not None and not style_transfer_effectiveness["theme_style_text_fit_reliable"]:
        warnings["low_confidence_warnings"].append("openclip_text_anchor_low_confidence")
    if package_coherence["score"] < 70.0:
        warnings["package_coherence_warnings"].append("package_coherence_weak")
    if package_coherence["generated_internal_outlier_apps"]:
        warnings["package_coherence_warnings"].append("generated_internal_outliers")
    if app_identity_coherence["score"] < 70.0:
        warnings["identity_warnings"].append("app_identity_coherence_weak")
    if app_identity_coherence["dino_identity_warning_apps"]:
        warnings["identity_warnings"].append("dino_identity_structure_risk")
    if qwen_qc.get("problematic_apps"):
        warnings["package_coherence_warnings"].append("qwen_qc_problematic_apps")
    if visual_quality["score"] < 70.0:
        warnings["quality_warnings"].append("visual_artifact_quality_weak")
    if strict_delta_diagnostics["strict_delta_transfer_score"] < 35.0:
        warnings["strict_delta_warnings"].append("strict_delta_diagnostic_weak")
    return warnings


def _itte_failed_reasons(warnings: dict) -> list[str]:
    reasons = []
    for key in [
        "style_transfer_warnings",
        "package_coherence_warnings",
        "identity_warnings",
        "quality_warnings",
    ]:
        reasons.extend(warnings.get(key, []))
    return reasons


def _auxiliary_scores(
    config,
    root: Path,
    theme_paths: list[Path],
    generated_paths: list[Path],
    target_paths: list[Path],
) -> dict:
    enabled = config.style_feature_backend == "vgg_gram_attribute"
    return {
        "vgg_gram_style_fit": compute_vgg_gram_style_fit(
            theme_paths=theme_paths,
            generated_paths=generated_paths,
            target_paths=target_paths,
            root_dir=root,
            enabled=enabled,
        )
    }


def _normalize_qwen_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 10.0:
        return score * 10.0
    return max(0.0, min(100.0, score))


def _app_identity_coherence_score(
    identity: dict,
    qwen_qc: dict,
    dino_gt: np.ndarray,
    app_names: list[str],
    style_transfer_score: float,
) -> dict:
    qwen_identity_score = _normalize_qwen_score(qwen_qc.get("target_identity_score"))
    structure = _target_structure_retention_score(dino_gt, app_names)
    identity_separability = float(identity.get("score", 0.0))
    recognition_prior = qwen_identity_score if qwen_identity_score is not None else identity_separability
    over_recomposition_penalty = max(
        0.0,
        min(100.0, float(style_transfer_score) * (100.0 - structure["score"]) / 100.0),
    )
    score = _weighted_score(
        [
            (structure["score"], 0.65),
            (recognition_prior, 0.25),
            (identity_separability, 0.10),
        ]
    )
    score = max(0.0, min(100.0, score - 0.15 * over_recomposition_penalty))
    return {
        "score": score,
        "qwen_identity_score": qwen_identity_score,
        "identity_recognition_prior_score": recognition_prior,
        "target_structure_retention_score": structure["score"],
        "identity_separability_score": identity_separability,
        "over_recomposition_penalty": over_recomposition_penalty,
        "target_structure_warning_apps": structure["warning_apps"],
        "target_structure_per_app": structure["per_app"],
    }


def _target_structure_retention_score(dino_gt: np.ndarray, app_names: list[str]) -> dict:
    matrix = np.asarray(dino_gt, dtype=np.float32)
    if matrix.size == 0 or not app_names:
        return {"score": 0.0, "warning_apps": [], "per_app": []}
    per_app = []
    scores = []
    for index, app in enumerate(app_names):
        own_distance = float(matrix[index, index])
        other_distances = np.delete(matrix[index], index)
        other_mean = float(np.mean(other_distances)) if len(other_distances) else own_distance
        nearest_index = int(np.argmin(matrix[index]))
        nearest_app = app_names[nearest_index]
        relative_gain = max(0.0, other_mean - own_distance)
        score = max(0.0, min(100.0, relative_gain / max(other_mean * 0.45, 1e-8) * 100.0))
        if nearest_app != app:
            score *= 0.75
        scores.append(score)
        per_app.append(
            {
                "app": app,
                "target_structure_retention_score": score,
                "generated_to_own_target_distance": own_distance,
                "generated_to_other_targets_mean_distance": other_mean,
                "nearest_target_app": nearest_app,
                "nearest_target_distance": float(matrix[index, nearest_index]),
                "structure_match_correct": nearest_app == app,
            }
        )
    warning_apps = [
        row["app"]
        for row in per_app
        if row["target_structure_retention_score"] < 60.0 or not row["structure_match_correct"]
    ]
    return {
        "score": float(np.mean(scores)),
        "scoring_method": "relative_to_other_target_distances",
        "relative_gain_denominator_ratio": 0.45,
        "warning_apps": warning_apps,
        "per_app": per_app,
    }


def _weighted_score(weighted_components: list[tuple[float | None, float]]) -> float:
    valid = [(float(score), float(weight)) for score, weight in weighted_components if score is not None and weight > 0.0]
    if not valid:
        return 0.0
    total_weight = sum(weight for _, weight in valid)
    return float(round(sum(score * weight for score, weight in valid) / max(total_weight, 1e-8), 10))


def pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        return np.array([0.0], dtype=np.float32)
    distances = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            distances.append(_cosine_distance(matrix[i], matrix[j]))
    return np.asarray(distances, dtype=np.float32)


def cross_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((len(left), len(right)), dtype=np.float32)
    for i in range(len(left)):
        for j in range(len(right)):
            result[i, j] = _cosine_distance(left[i], right[j])
    return result


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    left_sorted = np.sort(left.reshape(-1))
    right_sorted = np.sort(right.reshape(-1))
    n = max(len(left_sorted), len(right_sorted), 1)
    quantiles = np.linspace(0.0, 1.0, n)
    left_values = np.quantile(left_sorted, quantiles)
    right_values = np.quantile(right_sorted, quantiles)
    return float(np.mean(np.abs(left_values - right_values)))


def _stack_embeddings(paths: list[Path], embeddings: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.vstack([embeddings[str(path)] for path in paths]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8))


def _style_transfer_score(theme: np.ndarray, generated: np.ndarray, targets: np.ndarray) -> dict:
    centroid = _normalized_centroid(theme)
    d_rr = _theme_leave_one_out_distance(theme)
    d_gr = float(np.mean([_cosine_distance(item, centroid) for item in generated]))
    d_tr = float(np.mean([_cosine_distance(item, centroid) for item in targets]))
    score = max(0.0, min(100.0, (d_tr - d_gr) / max(d_tr - d_rr, 1e-8) * 100.0))
    return {
        "score": score,
        "D_TR_style": d_tr,
        "D_GR_style": d_gr,
        "D_RR_style": d_rr,
        "is_style_transfer_effective": d_gr < d_tr,
    }


def _style_delta_transfer_score(
    reference_raw: np.ndarray,
    reference_styled: np.ndarray,
    targets: np.ndarray,
    generated: np.ndarray,
    app_names: list[str],
    reference_raw_groups: dict[str, np.ndarray] | None = None,
    reference_styled_groups: dict[str, np.ndarray] | None = None,
    target_groups: dict[str, np.ndarray] | None = None,
    generated_groups: dict[str, np.ndarray] | None = None,
) -> dict:
    if not all([reference_raw_groups, reference_styled_groups, target_groups, generated_groups]):
        reference_raw_groups = _single_group_fallback(reference_raw)
        reference_styled_groups = _single_group_fallback(reference_styled)
        target_groups = _single_group_fallback(targets)
        generated_groups = _single_group_fallback(generated)

    group_scores = {}
    per_group_distances = {}
    for group in ["color", "edge", "composition", "complexity"]:
        group_scores[group] = _single_group_delta_score(
            reference_raw_groups[group],
            reference_styled_groups[group],
            target_groups[group],
            generated_groups[group],
        )
        per_group_distances[group] = group_scores[group]["generated_delta_to_reference_delta_centroid"]

    group_score_values = [group_scores[group]["score"] for group in ["color", "edge", "composition", "complexity"]]
    score = float(np.mean(group_score_values))
    reference_delta = np.hstack(
        [reference_styled_groups[group] - reference_raw_groups[group] for group in ["color", "edge", "composition", "complexity"]]
    )
    generated_delta = np.hstack(
        [generated_groups[group] - target_groups[group] for group in ["color", "edge", "composition", "complexity"]]
    )
    d_g_per_app = np.asarray(
        [
            float(np.mean([per_group_distances[group][index] for group in per_group_distances]))
            for index in range(len(generated_delta))
        ],
        dtype=np.float32,
    )
    d_g = float(np.mean(d_g_per_app))
    d_no_change = float(np.mean([group_scores[group]["D_no_change_delta"] for group in group_scores]))
    d_r = float(np.mean([group_scores[group]["D_R_delta"] for group in group_scores]))
    d_no_change_per_app = np.full(len(generated_delta), d_no_change, dtype=np.float32)
    per_app = []
    for index, app in enumerate(app_names):
        item_group_scores = [group_scores[group]["per_app_scores"][index] for group in group_scores]
        item_score = float(np.mean(item_group_scores))
        per_app.append(
            {
                "app": app,
                "d_to_reference_delta_centroid": float(d_g_per_app[index]),
                "no_change_d_to_reference_delta_centroid": float(d_no_change_per_app[index]),
                "group_delta_scores": {
                    group: float(group_scores[group]["per_app_scores"][index])
                    for group in group_scores
                },
                "style_delta_transfer_score": item_score,
                "is_delta_outlier": float(d_g_per_app[index]) > max(d_no_change, d_r),
            }
        )
    return {
        "score": score,
        "D_G_delta": d_g,
        "D_no_change_delta": d_no_change,
        "D_R_delta": d_r,
        "is_style_delta_transfer_effective": d_g < d_no_change,
        "group_scores": group_scores,
        "reference_delta_pairwise": pairwise_euclidean_distances(reference_delta).tolist(),
        "generated_delta_pairwise": pairwise_euclidean_distances(generated_delta).tolist(),
        "generated_delta_to_reference_delta_centroid": d_g_per_app.tolist(),
        "no_change_delta_to_reference_delta_centroid": d_no_change_per_app.tolist(),
        "per_app": per_app,
    }


def _single_group_delta_score(reference_raw: np.ndarray, reference_styled: np.ndarray, targets: np.ndarray, generated: np.ndarray) -> dict:
    reference_delta = reference_styled - reference_raw
    generated_delta = generated - targets
    reference_centroid = reference_delta.mean(axis=0)
    d_r = _leave_one_out_euclidean_distance(reference_delta)
    d_g_per_app = np.asarray([_euclidean_distance(item, reference_centroid) for item in generated_delta], dtype=np.float32)
    d_g = float(np.mean(d_g_per_app))
    no_change_delta = np.zeros_like(generated_delta)
    d_no_change_per_app = np.asarray(
        [_euclidean_distance(item, reference_centroid) for item in no_change_delta],
        dtype=np.float32,
    )
    d_no_change = float(np.mean(d_no_change_per_app))
    score = max(0.0, min(100.0, (d_no_change - d_g) / max(d_no_change - d_r, 1e-8) * 100.0))
    per_app_scores = []
    for index in range(len(generated_delta)):
        per_app_scores.append(
            max(
                0.0,
                min(
                    100.0,
                    (float(d_no_change_per_app[index]) - float(d_g_per_app[index]))
                    / max(float(d_no_change_per_app[index]) - d_r, 1e-8)
                    * 100.0,
                ),
            )
        )
    return {
        "score": score,
        "D_G_delta": d_g,
        "D_no_change_delta": d_no_change,
        "D_R_delta": d_r,
        "is_effective": d_g < d_no_change,
        "generated_delta_to_reference_delta_centroid": d_g_per_app.tolist(),
        "no_change_delta_to_reference_delta_centroid": d_no_change_per_app.tolist(),
        "per_app_scores": per_app_scores,
    }


def _package_consistency_score(gg: np.ndarray, rr: np.ndarray | None = None, tt: np.ndarray | None = None) -> dict:
    generated_pairwise = np.asarray(gg, dtype=np.float32).reshape(-1)
    mean_distance = float(np.mean(generated_pairwise))
    std_distance = float(np.std(generated_pairwise))
    max_distance = float(np.max(generated_pairwise))

    mean_score = max(0.0, min(1.0, 1.0 - mean_distance / 0.85))
    spread_score = max(0.0, min(1.0, 1.0 - std_distance / 0.25))
    max_score = max(0.0, min(1.0, 1.0 - max_distance / 1.10))
    score = 100.0 * (0.55 * mean_score + 0.30 * spread_score + 0.15 * max_score)
    return {
        "score": score,
        "generated_pairwise_mean_distance": mean_distance,
        "generated_pairwise_std_distance": std_distance,
        "generated_pairwise_max_distance": max_distance,
        "mean_distance_soft_limit": 0.85,
        "std_distance_soft_limit": 0.25,
        "max_distance_soft_limit": 1.10,
        "is_package_consistent": score >= 60.0 and max_distance <= 1.10,
    }


def _reference_style_distribution_match_score(gg: np.ndarray, rr: np.ndarray, tt: np.ndarray) -> dict:
    c_gr = wasserstein_1d(gg, rr)
    c_tr = wasserstein_1d(tt, rr)
    score = max(0.0, min(100.0, (c_tr - c_gr) / max(c_tr, 1e-8) * 100.0))
    return {
        "score": score,
        "C_GR_style": c_gr,
        "C_TR_style": c_tr,
        "is_package_style_consistency_improved": c_gr < c_tr,
    }


def _theme_membership_score(theme: np.ndarray, generated: np.ndarray, targets: np.ndarray, app_names: list[str]) -> dict:
    centroid = _normalized_centroid(theme)
    reference_distances = _theme_leave_one_out_distances(theme)
    reference_mean = float(np.mean(reference_distances))
    reference_max = float(np.max(reference_distances))
    d_tr = float(np.mean([_cosine_distance(item, centroid) for item in targets]))
    generated_centroid = _normalized_centroid(generated)
    generated_internal_distances = np.asarray(
        [_cosine_distance(item, generated_centroid) for item in generated],
        dtype=np.float32,
    )
    internal_mean = float(np.mean(generated_internal_distances))
    internal_std = float(np.std(generated_internal_distances))
    internal_threshold = internal_mean + max(2.0 * internal_std, 0.08)
    per_app = []
    scores = []
    for index, item in enumerate(generated):
        distance = _cosine_distance(item, centroid)
        score = max(0.0, min(100.0, (d_tr - distance) / max(d_tr - reference_mean, 1e-8) * 100.0))
        scores.append(score)
        per_app.append(
            {
                "app": app_names[index],
                "index": index,
                "d_to_theme_style_centroid": distance,
                "theme_membership_score": score,
                "is_style_outlier": distance > reference_max,
                "generated_internal_distance": float(generated_internal_distances[index]),
                "generated_internal_outlier": float(generated_internal_distances[index]) > internal_threshold,
            }
        )
    generated_internal_outlier_apps = [row["app"] for row in per_app if row["generated_internal_outlier"]]
    return {
        "score": float(np.mean(scores)),
        "reference_mean_style_distance": reference_mean,
        "reference_max_style_distance": reference_max,
        "legacy_theme_reference_outlier_apps": [row["app"] for row in per_app if row["is_style_outlier"]],
        "generated_internal_mean_distance": internal_mean,
        "generated_internal_std_distance": internal_std,
        "generated_internal_outlier_threshold": internal_threshold,
        "generated_internal_outlier_apps": generated_internal_outlier_apps,
        "has_style_outliers": bool(generated_internal_outlier_apps),
        "has_generated_internal_outliers": bool(generated_internal_outlier_apps),
        "per_app": per_app,
    }


def _identity_separability_score(gt: np.ndarray, generated_icons) -> dict:
    app_names = [item.app for item in generated_icons]
    correct = 0
    per_app = []
    for i, app in enumerate(app_names):
        order = np.argsort(gt[i])
        rank = int(np.where(order == i)[0][0]) + 1
        nearest = app_names[int(order[0])]
        is_match = nearest == app
        own_similarity = 1.0 - float(gt[i, i])
        max_similarity = 1.0 - float(gt[i, order[0]])
        correct += int(is_match)
        per_app.append(
            {
                "app": app,
                "identity_rank": rank,
                "matched_target_app": nearest,
                "identity_match_correct": is_match,
                "generated_to_own_target_similarity": own_similarity,
                "max_target_similarity": max_similarity,
            }
        )
    accuracy = correct / max(len(app_names), 1)
    random_baseline = 1.0 / max(len(app_names), 1)
    if random_baseline >= 1.0:
        score = 100.0 if accuracy >= 1.0 else 0.0
    else:
        score = max(0.0, min(100.0, 100.0 * (accuracy - random_baseline) / (1.0 - random_baseline)))
    return {
        "score": score,
        "top1_accuracy": accuracy,
        "random_baseline": random_baseline,
        "identity_above_random": accuracy > random_baseline,
        "per_app": per_app,
    }


def _geometric_mean(scores: list[float]) -> float:
    clipped = np.maximum(np.asarray(scores, dtype=np.float32), 1e-6)
    return float(np.exp(np.mean(np.log(clipped))))


def _risk_scores(identity: dict) -> dict:
    warning_apps = [
        row["app"]
        for row in identity["per_app"]
        if not row["identity_match_correct"]
    ]
    return {
        "dino_identity_structure_risk_score": max(0.0, 100.0 - float(identity["score"])),
        "dino_identity_top1_accuracy": identity["top1_accuracy"],
        "dino_identity_random_baseline": identity["random_baseline"],
        "dino_identity_warning_apps": warning_apps,
    }


def _tpqs_summary(
    delta_transfer: dict,
    internal_consistency: dict,
    membership: dict,
    identity: dict,
    visual: dict,
) -> dict:
    return {
        "style_transfer_diagnostics": {
            "score": delta_transfer["score"],
            "group_scores": {
                group: data["score"]
                for group, data in delta_transfer.get("group_scores", {}).items()
            },
            "is_effective": delta_transfer["is_style_delta_transfer_effective"],
            "interpretation": "Evaluates whether target->generated follows the reference_raw->style_ref transfer direction.",
        },
        "package_unity_diagnostics": {
            "score": internal_consistency["score"],
            "generated_pairwise_mean_distance": internal_consistency["generated_pairwise_mean_distance"],
            "is_package_consistent": internal_consistency["is_package_consistent"],
            "generated_internal_outlier_apps": membership["generated_internal_outlier_apps"],
        },
        "identity_diagnostics": {
            "score": identity["score"],
            "top1_accuracy": identity["top1_accuracy"],
            "random_baseline": identity["random_baseline"],
            "identity_above_random": identity["identity_above_random"],
        },
        "visual_quality_diagnostics": {
            "visual_stats_transfer_score": visual["score"],
            "visual_artifact_quality_score": visual["artifact_quality_score"],
            "artifact_warnings": visual["artifact_warnings"],
            "is_artifact_quality_ok": visual["is_artifact_quality_ok"],
        },
    }


def _itte_diagnosis_summary(
    decision: str,
    failed_reasons: list[str],
    warnings: dict,
    style_transfer_effectiveness: dict,
    package_coherence: dict,
    app_identity_coherence: dict,
    visual_quality: dict,
    strict_delta_diagnostics: dict,
    primary_scores: dict,
    diagnostic_scores: dict,
    risk_scores: dict,
    qwen_qc_scores: dict,
    delta_transfer: dict,
    internal_consistency: dict,
    membership: dict,
    identity: dict,
    visual: dict,
) -> dict:
    strong_points = []
    weak_points = []
    if style_transfer_effectiveness["score"] >= 70.0:
        strong_points.append("Style Transfer Effectiveness Score indicates generated icons moved from target originals toward the theme_refs/style_ref final style.")
    else:
        weak_points.append("Style Transfer Effectiveness Score is weak; generated icons may not have moved enough toward the theme_refs/style_ref final style.")
    if package_coherence["score"] >= 70.0:
        strong_points.append("Package Coherence Score indicates the generated package is internally consistent.")
    else:
        weak_points.append("Package Coherence Score is weak; generated icons may not look like one unified package.")
    if app_identity_coherence["qwen_identity_score"] is not None and app_identity_coherence["qwen_identity_score"] >= 70.0:
        strong_points.append("Qwen package QC judges App semantic identity as recognizable.")
    elif app_identity_coherence["score"] < 70.0:
        weak_points.append("App Identity Coherence Score is weak.")
    if app_identity_coherence["dino_identity_warning_apps"]:
        weak_points.append(
            "DINOv3 structure identity risk apps: "
            + ", ".join(app_identity_coherence["dino_identity_warning_apps"])
            + ". This is a structure-level risk signal, not direct semantic identity failure."
        )
    if visual_quality["score"] >= 70.0:
        strong_points.append("Visual Quality Score has no severe artifact warning.")
    else:
        weak_points.append("Visual Quality Score indicates artifact risk.")
    if strict_delta_diagnostics["strict_delta_transfer_score"] < 35.0:
        weak_points.append("Strict Delta Diagnostics are weak, meaning low-level target->generated statistics did not replicate the reference_raw->style_ref delta path.")
    if warnings.get("low_confidence_warnings"):
        weak_points.append("Low-confidence signals: " + ", ".join(warnings["low_confidence_warnings"]))

    conclusions = {
        "style_transfer_success": "ITTE judges the icon theme transfer as successful across style transfer, package coherence, identity coherence, and visual quality.",
        "package_coherent_but_theme_transfer_weak": "The generated package is internally coherent, but the core style transfer toward the specified theme is weak.",
        "theme_transfer_effective_but_package_unity_weak": "Generated icons moved toward the theme style, but package-level unity is weak.",
        "style_transfer_success_with_identity_risk": "Style transfer is effective, but App identity coherence has risk.",
        "visual_quality_problem": "Visual quality problems dominate this evaluation.",
        "needs_review": "ITTE found mixed signals that need human review.",
    }
    return {
        "main_conclusion": conclusions.get(decision, "ITTE found mixed signals that need human review."),
        "strong_points": strong_points,
        "weak_points": weak_points,
        "failed_reasons": failed_reasons,
        "metric_warning": [
            "ITTE is Icon Theme Transfer Evaluation, not the old single TPQS score.",
            "Style Transfer Effectiveness Score is the core score; it measures whether generated icons move from target_originals toward theme_refs/style_ref final style.",
            "Package Coherence Score only measures whether generated icons are internally unified; package coherence is not proof of successful transfer to the specified theme.",
            "Strict Delta Diagnostics only measure whether low-level target->generated statistics replicate the reference_raw->style_ref path; they do not drive the main decision.",
            "DINOv3 is used only as an identity structure risk signal.",
            "OpenCLIP theme text fit is only included when theme_style_text_fit_reliable=true.",
        ],
    }


def _package_dir_from_resolved(resolved: ResolvedEvalInputs) -> Path:
    first_icon = resolved.generated_icons[0].path
    return first_icon.parent.parent


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_theme_design_analysis(package_dir: Path) -> dict:
    return _load_json(package_dir / "theme_design_analysis.json")


def _load_qwen_instruction_text(package_dir: Path) -> str:
    path = package_dir / "generation_base_prompt.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_qwen_qc_scores(package_dir: Path) -> dict:
    report = _load_json(package_dir / "package_qc_report.json")
    problematic_raw = report.get("problematic_apps") or []
    problematic_names = _problematic_app_names(problematic_raw)
    return {
        "package_consistency_score": report.get("package_consistency_score"),
        "style_consistency_score": report.get("style_consistency_score"),
        "target_identity_score": report.get("target_identity_score"),
        "problematic_apps": problematic_names,
        "problematic_app_details": problematic_raw,
    }


def _problematic_app_names(items) -> list[str]:
    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("app"):
            names.append(str(item["app"]))
    return names


def _theme_style_text_fit_score(
    config,
    root: Path,
    theme_paths: list[Path],
    generated_paths: list[Path],
    target_paths: list[Path],
    theme_design_analysis: dict,
    qwen_instruction_text: str,
) -> dict:
    if not getattr(config, "use_openclip", False):
        return disabled_theme_style_text_fit()
    backend = OpenClipBackend(
        model_name=config.openclip_model,
        pretrained=config.openclip_pretrained,
        device=config.device,
        root_dir=root,
    )
    return compute_theme_style_text_fit(
        theme_ref_paths=theme_paths,
        generated_paths=generated_paths,
        target_paths=target_paths,
        theme_design_analysis=theme_design_analysis,
        qwen_instruction_text=qwen_instruction_text,
        backend=backend,
    )


def _fallback_style_feature_groups(style_features: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        path: _single_vector_groups(vector)
        for path, vector in style_features.items()
    }


def _single_vector_groups(vector: np.ndarray) -> dict[str, np.ndarray]:
    return {group: vector for group in [*STYLE_ATTRIBUTE_NAMES, "edge"]}


def _single_group_fallback(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {group: matrix for group in ["color", "edge", "composition", "complexity"]}


def _group_matrices(paths: list[Path], grouped_style: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        group: np.vstack([grouped_style[str(path)][group] for path in paths]).astype(np.float32)
        for group in ["color", "edge", "composition", "complexity"]
    }


def pairwise_euclidean_distances(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        return np.array([0.0], dtype=np.float32)
    distances = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            distances.append(_euclidean_distance(matrix[i], matrix[j]))
    return np.asarray(distances, dtype=np.float32)


def _leave_one_out_euclidean_distance(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 0.0
    distances = []
    for index, item in enumerate(matrix):
        others = np.delete(matrix, index, axis=0)
        distances.append(_euclidean_distance(item, others.mean(axis=0)))
    return float(np.mean(distances))


def _euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _normalized_centroid(matrix: np.ndarray) -> np.ndarray:
    centroid = matrix.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1e-8:
        return centroid
    return centroid / norm


def _theme_leave_one_out_distances(theme: np.ndarray) -> np.ndarray:
    if len(theme) < 2:
        return np.array([0.0], dtype=np.float32)
    distances = []
    for index, item in enumerate(theme):
        others = np.delete(theme, index, axis=0)
        distances.append(_cosine_distance(item, _normalized_centroid(others)))
    return np.asarray(distances, dtype=np.float32)


def _theme_leave_one_out_distance(theme: np.ndarray) -> float:
    return float(np.mean(_theme_leave_one_out_distances(theme)))
