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
    pairwise: dict


def compute_tpqs_metrics(resolved: ResolvedEvalInputs, embeddings: dict[str, np.ndarray], config) -> TpqsMetrics:
    theme_paths = resolved.theme_refs
    generated_paths = [item.path for item in resolved.generated_icons]
    target_paths = [resolved.target_originals[item.app] for item in resolved.generated_icons]

    theme = _stack_embeddings(theme_paths, embeddings)
    generated = _stack_embeddings(generated_paths, embeddings)
    targets = _stack_embeddings(target_paths, embeddings)

    rr = pairwise_distances(theme)
    gg = pairwise_distances(generated)
    tt = pairwise_distances(targets)
    gr = cross_distances(generated, theme)
    tr = cross_distances(targets, theme)
    gt = cross_distances(generated, targets)

    theme_transfer = _theme_transfer_score(rr, gr, tr)
    consistency = _package_consistency_score(gg, rr, tt)
    membership = _theme_membership_score(gr, rr, [item.app for item in resolved.generated_icons])
    identity = _identity_separability_score(gt, resolved.generated_icons)
    visual = visual_statistics_score(theme_paths, generated_paths, target_paths, config.image_size)

    scores = [
        theme_transfer["score"],
        consistency["score"],
        membership["score"],
        identity["score"],
        visual["score"],
    ]
    total = _geometric_mean(scores)

    decision = _decision(theme_transfer, consistency, membership, identity, visual)
    report = {
        "theme_id": resolved.theme_id,
        "package_id": resolved.package_id,
        "embedding_backend": config.embedding_backend,
        "model_id": config.model_id,
        "is_official_tpqs": config.is_official_tpqs,
        "tpqs_total_score": total,
        "theme_transfer_score": theme_transfer["score"],
        "package_consistency_score": consistency["score"],
        "theme_membership_score": membership["score"],
        "identity_separability_score": identity["score"],
        "visual_statistics_score": visual["score"],
        "decision": decision,
        "flags": {
            "is_transfer_effective": theme_transfer["is_transfer_effective"],
            "is_consistency_improved": consistency["is_consistency_improved"],
            "has_style_outliers": membership["has_style_outliers"],
            "identity_above_random": identity["identity_above_random"],
            "is_visual_stats_improved": visual["is_visual_stats_improved"],
        },
        "details": {
            "theme_transfer": theme_transfer,
            "package_consistency": consistency,
            "theme_membership": {
                key: value for key, value in membership.items() if key != "per_app"
            },
            "identity_separability": {
                key: value for key, value in identity.items() if key != "per_app"
            },
            "visual_statistics": visual,
        },
    }

    per_app_rows = []
    identity_rows = {row["app"]: row for row in identity["per_app"]}
    for row in membership["per_app"]:
        identity_row = identity_rows[row["app"]]
        per_app_rows.append(
            {
                "app": row["app"],
                "theme_membership_score": round(row["theme_membership_score"], 4),
                "identity_rank": identity_row["identity_rank"],
                "identity_top1_match": identity_row["identity_top1_match"],
                "nearest_target_app": identity_row["nearest_target_app"],
                "generated_to_theme_distance": round(row["generated_to_theme_distance"], 6),
                "is_style_outlier": row["is_style_outlier"],
            }
        )

    pairwise = {
        "theme_ref_pairwise": rr.tolist(),
        "generated_pairwise": gg.tolist(),
        "target_pairwise": tt.tolist(),
        "generated_to_theme": gr.tolist(),
        "target_to_theme": tr.tolist(),
        "generated_to_target": gt.tolist(),
        "app_names": [item.app for item in resolved.generated_icons],
    }
    return TpqsMetrics(report=report, per_app_rows=per_app_rows, pairwise=pairwise)


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


def _theme_transfer_score(rr: np.ndarray, gr: np.ndarray, tr: np.ndarray) -> dict:
    rr_mean = float(np.mean(rr))
    gr_mean = float(np.mean(gr))
    tr_mean = float(np.mean(tr))
    denominator = max(tr_mean - rr_mean, 1e-8)
    transfer_pull = max(0.0, min(1.0, (tr_mean - gr_mean) / denominator))
    reference_fit = float(np.exp(-abs(gr_mean - rr_mean) / max(float(np.std(rr)) + 0.05, 1e-8)))
    score = max(0.0, min(100.0, 100.0 * np.sqrt(transfer_pull * reference_fit)))
    return {
        "score": score,
        "theme_ref_internal_mean_distance": rr_mean,
        "generated_to_theme_mean_distance": gr_mean,
        "target_to_theme_mean_distance": tr_mean,
        "is_transfer_effective": gr_mean < tr_mean,
    }


def _package_consistency_score(gg: np.ndarray, rr: np.ndarray, tt: np.ndarray) -> dict:
    generated_vs_ref = wasserstein_1d(gg, rr)
    target_vs_ref = wasserstein_1d(tt, rr)
    improvement = max(target_vs_ref - generated_vs_ref, 0.0) / max(target_vs_ref, 1e-8)
    score = max(0.0, min(100.0, improvement * 100.0))
    return {
        "score": score,
        "generated_vs_theme_wasserstein": generated_vs_ref,
        "target_vs_theme_wasserstein": target_vs_ref,
        "is_consistency_improved": generated_vs_ref < target_vs_ref,
    }


def _theme_membership_score(gr: np.ndarray, rr: np.ndarray, app_names: list[str]) -> dict:
    rr_mean = float(np.mean(rr))
    rr_std = float(np.std(rr))
    threshold = rr_mean + max(2.0 * rr_std, 0.08)
    per_app = []
    scores = []
    for index, row in enumerate(gr):
        distance = float(np.mean(row))
        score = max(0.0, min(100.0, 100.0 * (1.0 - max(distance - rr_mean, 0.0) / max(threshold, 1e-8))))
        scores.append(score)
        per_app.append(
            {
                "app": app_names[index],
                "index": index,
                "generated_to_theme_distance": distance,
                "theme_membership_score": score,
                "is_style_outlier": distance > threshold,
            }
        )
    return {
        "score": float(np.mean(scores)),
        "outlier_threshold": threshold,
        "has_style_outliers": any(row["is_style_outlier"] for row in per_app),
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
        correct += int(is_match)
        per_app.append(
            {
                "app": app,
                "identity_rank": rank,
                "nearest_target_app": nearest,
                "identity_top1_match": is_match,
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


def _decision(theme_transfer: dict, consistency: dict, membership: dict, identity: dict, visual: dict) -> str:
    if not theme_transfer["is_transfer_effective"]:
        return "transfer_failed"
    if not identity["identity_above_random"]:
        return "identity_collapse_risk"
    if membership["has_style_outliers"]:
        return "local_retry_recommended"
    if consistency["is_consistency_improved"] and visual["is_visual_stats_improved"]:
        return "closed_loop_pass"
    return "needs_review_or_retry"
