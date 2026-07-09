from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.services.eval_path_service import ResolvedEvalInputs
from evaluation.services.visual_stats_service import visual_statistics_score


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

    scores = [
        delta_transfer["score"],
        internal_consistency["score"],
        reference_match["score"],
        membership["score"],
        identity["score"],
        visual["score"],
        visual["artifact_quality_score"],
    ]
    total = _geometric_mean(scores)

    decision = _decision(delta_transfer, reference_match, membership, identity, visual, internal_consistency)
    failed_reasons = _failed_reasons(delta_transfer, reference_match, membership, identity, visual, internal_consistency)
    tpqs_summary = _tpqs_summary(delta_transfer, internal_consistency, membership, identity, visual)
    diagnosis_summary = _diagnosis_summary(
        decision,
        failed_reasons,
        delta_transfer,
        internal_consistency,
        membership,
        identity,
        visual,
    )
    report = {
        "eval_id": None,
        "theme_id": resolved.theme_id,
        "package_id": resolved.package_id,
        "tpqs": total,
        "tpqs_total_score": total,
        "style_transfer_score": delta_transfer["score"],
        "style_delta_transfer_score": delta_transfer["score"],
        "color_delta_score": delta_transfer["group_scores"]["color"]["score"],
        "edge_delta_score": delta_transfer["group_scores"]["edge"]["score"],
        "composition_delta_score": delta_transfer["group_scores"]["composition"]["score"],
        "complexity_delta_score": delta_transfer["group_scores"]["complexity"]["score"],
        "final_style_membership_score": final_style_membership["score"],
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
        "diagnosis_summary": diagnosis_summary,
        "tpqs_summary": tpqs_summary,
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
            "device": config.device,
            "pooling": config.pooling,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
        },
        "details": {
            "style_transfer": delta_transfer,
            "style_delta_transfer": delta_transfer,
            "final_style_membership": final_style_membership,
            "package_internal_style_consistency": internal_consistency,
            "reference_style_distribution_match": reference_match,
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
    for row in membership["per_app"]:
        identity_row = identity_rows[row["app"]]
        delta_row = delta_rows[row["app"]]
        per_app_rows.append(
            {
                "app": row["app"],
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


def _decision(
    theme_transfer: dict,
    reference_match: dict,
    membership: dict,
    identity: dict,
    visual: dict,
    internal_consistency: dict | None = None,
) -> str:
    internal_score = 0.0 if internal_consistency is None else float(internal_consistency["score"])
    if internal_score >= 70.0 and not _is_transfer_effective(theme_transfer):
        return "metric_conflict_needs_review"
    if not _is_transfer_effective(theme_transfer):
        return "style_transfer_failed"
    if not identity["identity_above_random"]:
        return "identity_collapse_risk"
    if membership["has_generated_internal_outliers"]:
        return "local_retry_recommended"
    if reference_match["is_package_style_consistency_improved"] and visual.get("is_artifact_quality_ok", True):
        return "closed_loop_pass"
    return "needs_review_or_retry"


def _failed_reasons(
    theme_transfer: dict,
    reference_match: dict,
    membership: dict,
    identity: dict,
    visual: dict,
    internal_consistency: dict | None = None,
) -> list[str]:
    reasons = []
    internal_score = 0.0 if internal_consistency is None else float(internal_consistency["score"])
    if internal_score >= 70.0 and not _is_transfer_effective(theme_transfer):
        reasons.append("style_transfer_metric_conflicts_with_package_unity")
    elif not _is_transfer_effective(theme_transfer):
        reasons.append("style_transfer_not_effective")
    if not reference_match["is_package_style_consistency_improved"]:
        reasons.append("package_style_distribution_not_improved")
    if membership["has_generated_internal_outliers"]:
        reasons.append("generated_internal_outliers_detected")
    if not identity["identity_above_random"]:
        reasons.append("identity_not_above_random")
    if not visual["is_visual_stats_improved"]:
        reasons.append("visual_stats_transfer_not_improved")
    if not visual.get("is_artifact_quality_ok", True):
        reasons.append("visual_artifact_quality_risk")
    return reasons


def _is_transfer_effective(theme_transfer: dict) -> bool:
    if "is_style_delta_transfer_effective" in theme_transfer:
        return bool(theme_transfer["is_style_delta_transfer_effective"])
    return bool(theme_transfer["is_style_transfer_effective"])


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


def _diagnosis_summary(
    decision: str,
    failed_reasons: list[str],
    delta_transfer: dict,
    internal_consistency: dict,
    membership: dict,
    identity: dict,
    visual: dict,
) -> dict:
    strong_points = []
    weak_points = []
    if internal_consistency["score"] >= 70.0:
        strong_points.append("生成包内部风格一致性较好。")
    else:
        weak_points.append("生成包内部风格一致性不足。")
    if identity["identity_above_random"]:
        strong_points.append("DINOv3 身份可分辨度高于随机基线。")
    else:
        weak_points.append("生成图和目标 App 的身份匹配风险较高。")
    if delta_transfer["score"] <= 20.0:
        weak_points.append("style_delta_transfer_score 较低，说明 target->generated 的变化方向没有贴近 reference_raw->style_ref 的统计变化。")
    if visual["artifact_quality_score"] >= 70.0:
        strong_points.append("基础视觉 artifact 质量没有明显异常。")
    else:
        weak_points.append("存在清晰度、亮度、边缘复杂度或主体面积方面的 artifact 风险。")
    if membership["generated_internal_outlier_apps"]:
        weak_points.append("生成包内部存在局部离群图标：" + ", ".join(membership["generated_internal_outlier_apps"]))

    if decision == "metric_conflict_needs_review":
        main_conclusion = (
            "生成包内部一致性较好，但 style transfer 方向指标未通过，可能是当前 style_delta 指标与人眼主题风格判断不完全一致。"
        )
    elif decision == "closed_loop_pass":
        main_conclusion = "当前量化指标整体通过。"
    else:
        main_conclusion = "当前结果存在需要复核或重试的量化风险。"

    return {
        "main_conclusion": main_conclusion,
        "strong_points": strong_points,
        "weak_points": weak_points,
        "failed_reasons": failed_reasons,
        "metric_reliability_warning": (
            "TPQS 是业务诊断面板，不是绝对审美裁判。Style Features 是轻量统计特征，不能完全等同于人眼审美；"
            "theme_refs 只有 3 个时，membership/outlier 需要谨慎解释。"
        ),
    }


def _diagnosis_summary(
    decision: str,
    failed_reasons: list[str],
    delta_transfer: dict,
    internal_consistency: dict,
    membership: dict,
    identity: dict,
    visual: dict,
) -> dict:
    strong_points = []
    weak_points = []
    if internal_consistency["score"] >= 70.0:
        strong_points.append("Generated package has strong internal visual consistency.")
    else:
        weak_points.append("Generated package internal visual consistency is weak.")
    if identity["identity_above_random"]:
        strong_points.append("DINOv3 identity separability is above the random baseline.")
    else:
        weak_points.append("Generated icons have elevated target identity confusion risk.")
    if delta_transfer["score"] <= 20.0:
        weak_points.append("Low-level style delta transfer is weak; target->generated does not follow the reference_raw->style_ref statistical direction.")
    if visual["artifact_quality_score"] >= 70.0:
        strong_points.append("Basic visual artifact quality has no obvious risk.")
    else:
        weak_points.append("There may be artifact risk in clarity, brightness, edge complexity, or subject area.")
    if membership["generated_internal_outlier_apps"]:
        weak_points.append("Generated package has local outlier apps: " + ", ".join(membership["generated_internal_outlier_apps"]))

    if decision == "metric_conflict_needs_review":
        main_conclusion = (
            "Package unity is strong, but style transfer direction diagnostics did not pass; "
            "low-level metrics may conflict with human theme-style judgment."
        )
    elif decision == "closed_loop_pass":
        main_conclusion = "Current quantitative diagnostics pass overall."
    else:
        main_conclusion = "Current result has quantitative risks that need review or retry."

    return {
        "main_conclusion": main_conclusion,
        "strong_points": strong_points,
        "weak_points": weak_points,
        "failed_reasons": failed_reasons,
        "metric_reliability_warning": (
            "TPQS is a business diagnostics panel, not an absolute aesthetic judge. "
            "Style Features are lightweight statistical features and do not fully equal human visual judgment. "
            "When theme_refs are limited, membership/outlier results need cautious interpretation."
        ),
    }


def _fallback_style_feature_groups(style_features: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        path: _single_vector_groups(vector)
        for path, vector in style_features.items()
    }


def _single_vector_groups(vector: np.ndarray) -> dict[str, np.ndarray]:
    return {group: vector for group in ["color", "edge", "composition", "complexity"]}


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
