from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.services.dino_dense_service import DenseImageFeature, dense_correspondence
from evaluation.services.eval_path_service import ResolvedEvalInputs
from evaluation.services.perceptual_service import compute_perceptual_scores
from evaluation.services.quality_service import compute_visual_quality
from evaluation.services.vgg_gram_service import compute_vgg_gram_style_fit


STYLE_GROUPS = ("color", "background", "stroke", "texture_material", "composition", "complexity")
STYLE_COMPONENT_WEIGHTS = {
    "vgg_gram": 0.40,
    "dists_texture": 0.20,
    "visual_attributes": 0.25,
    "dino_motif": 0.15,
}
IDENTITY_COMPONENT_WEIGHTS = {
    "dino_dense": 0.50,
    "dists_structure": 0.30,
    "lpips_content": 0.20,
}


@dataclass(frozen=True)
class ItteMetrics:
    report: dict
    per_app_rows: list[dict]
    style_pairwise: dict
    style_delta: dict
    dino_pairwise: dict


def compute_itte_v12_metrics(
    resolved: ResolvedEvalInputs,
    style_features: dict[str, np.ndarray],
    style_feature_groups: dict[str, dict[str, np.ndarray]],
    dense_structure_features: dict[str, DenseImageFeature],
    dense_appearance_features: dict[str, DenseImageFeature],
    config,
    root_dir: Path,
) -> ItteMetrics:
    app_names = [item.app for item in resolved.generated_icons]
    reference_raw_paths = [item.reference_raw_path for item in resolved.theme_examples]
    theme_paths = list(resolved.theme_refs)
    generated_paths = [item.path for item in resolved.generated_icons]
    target_paths = [resolved.target_originals[app] for app in app_names]

    attributes = _attribute_style_scores(
        reference_raw_paths,
        theme_paths,
        target_paths,
        generated_paths,
        app_names,
        style_feature_groups,
    )
    vgg = compute_vgg_gram_style_fit(
        theme_paths,
        generated_paths,
        target_paths,
        root_dir,
        enabled=bool(config.use_vgg_gram),
    )
    perceptual = compute_perceptual_scores(
        reference_raw_paths,
        theme_paths,
        target_paths,
        generated_paths,
        app_names,
        root_dir,
        config.device,
        enabled=bool(config.use_perceptual),
        image_size=config.image_size,
    )
    motif = _motif_style_score(
        reference_raw_paths,
        theme_paths,
        target_paths,
        generated_paths,
        app_names,
        dense_appearance_features,
    )
    style_components = {
        "vgg_gram": _component(vgg.get("score"), bool(vgg.get("reliable")), STYLE_COMPONENT_WEIGHTS["vgg_gram"]),
        "dists_texture": _component(
            perceptual["dists_texture"].get("score"),
            bool(perceptual["dists_texture"].get("reliable")),
            STYLE_COMPONENT_WEIGHTS["dists_texture"],
        ),
        "visual_attributes": _component(attributes.get("score"), bool(attributes.get("reliable")), 0.25),
        "dino_motif": _component(motif.get("score"), bool(motif.get("reliable")), 0.15),
    }
    style_score, style_available_weight = _weighted_available(style_components)
    style_fidelity = {
        "score": style_score,
        "available_weight": style_available_weight,
        "components": style_components,
        "vgg_gram": vgg,
        "dists_texture": perceptual["dists_texture"],
        "visual_attributes": attributes,
        "dino_motif": motif,
    }

    dense_identity = _dense_identity_score(
        reference_raw_paths,
        theme_paths,
        target_paths,
        generated_paths,
        app_names,
        dense_structure_features,
    )
    identity_components = {
        "dino_dense": _component(dense_identity["score"], dense_identity["reliable"], 0.50),
        "dists_structure": _component(
            perceptual["dists_structure"].get("score"),
            bool(perceptual["dists_structure"].get("reliable")),
            0.30,
        ),
        "lpips_content": _component(
            perceptual["lpips_content"].get("score"),
            bool(perceptual["lpips_content"].get("reliable")),
            0.20,
        ),
    }
    identity_score, identity_available_weight = _weighted_available(identity_components)
    identity_per_app = _merge_identity_rows(
        app_names,
        dense_identity["per_app"],
        perceptual["dists_structure"].get("per_app", []),
        perceptual["lpips_content"].get("per_app", []),
        identity_components,
    )
    identity_scores = [item["score"] for item in identity_per_app]
    identity = {
        "score": identity_score,
        "p10_score": float(np.percentile(identity_scores, 10)) if identity_scores else 0.0,
        "available_weight": identity_available_weight,
        "components": identity_components,
        "dino_dense": dense_identity,
        "dists_structure": perceptual["dists_structure"],
        "lpips_content": perceptual["lpips_content"],
        "per_app": identity_per_app,
    }

    package = _package_coherence(
        theme_paths,
        target_paths,
        generated_paths,
        app_names,
        style_features,
    )
    quality = compute_visual_quality(theme_paths, generated_paths, app_names)
    total = _weighted_score(
        [
            (style_score, 0.35),
            (identity_score, 0.30),
            (package["score"], 0.20),
            (quality["score"], 0.15),
        ]
    )
    confidence = _evaluation_confidence(len(theme_paths), style_available_weight, identity_available_weight, attributes)
    hard_failures = _hard_failures(identity, package, quality, style_available_weight)
    decision = _decision(total, style_score, identity, package, quality, confidence, hard_failures)
    qwen_diagnostic = _load_generation_qwen_diagnostic(root_dir, resolved.package_id)
    provenance = _provenance(root_dir, [*reference_raw_paths, *theme_paths, *target_paths, *generated_paths], config)

    report = {
        "eval_id": None,
        "evaluation_framework": "ITTE",
        "itte_version": "v1.2-image-only",
        "evaluation_scope": "observable_image_transfer_only",
        "theme_id": resolved.theme_id,
        "package_id": resolved.package_id,
        "itte_score": total,
        "tpqs": total,
        "tpqs_primary_score": total,
        "style_fidelity_score": style_score,
        "identity_preservation_score": identity_score,
        "package_coherence_score": package["score"],
        "visual_quality_score": quality["score"],
        "style_fidelity": style_fidelity,
        "identity_preservation": identity,
        "package_coherence": package,
        "visual_quality": quality,
        "hard_failures": hard_failures,
        "evaluation_confidence": confidence,
        "decision": decision,
        "per_app": _per_app_report(app_names, style_fidelity, identity, package, quality, total),
        "diagnostics": {
            "generation_qwen_qc": qwen_diagnostic,
            "generation_qwen_qc_used_in_score": False,
            "openclip_used_in_score": False,
            "text_policy": "out_of_scope",
            "perceptual_backend": perceptual,
        },
        "provenance": provenance,
        "compatibility": {
            "tpqs_fields_are_aliases": True,
            "legacy_v11_scores_removed_from_main_decision": True,
        },
    }
    per_app_rows = report["per_app"]
    style_pairwise = _style_pairwise_payload(theme_paths, target_paths, generated_paths, app_names, style_features)
    style_delta = {
        "framework": "ITTE v1.2 image-only",
        "attribute_transfer": attributes,
        "vgg_gram": vgg,
        "dists_texture": perceptual["dists_texture"],
        "dino_motif": motif,
    }
    dino_pairwise = dense_identity["pairwise"]
    return ItteMetrics(report, per_app_rows, style_pairwise, style_delta, dino_pairwise)


def _attribute_style_scores(raw_paths, theme_paths, target_paths, generated_paths, app_names, grouped):
    groups = {}
    per_app_parts = {app: [] for app in app_names}
    valid_scores = []
    for name in STYLE_GROUPS:
        raw = _group_matrix(raw_paths, grouped, name)
        theme = _group_matrix(theme_paths, grouped, name)
        target = _group_matrix(target_paths, grouped, name)
        generated = _group_matrix(generated_paths, grouped, name)
        centroid = _normalized_centroid(theme)
        d_rr = _leave_one_out_distance(theme)
        d_or = float(np.mean([_cosine_distance(item, centroid) for item in raw]))
        d_tr = float(np.mean([_cosine_distance(item, centroid) for item in target]))
        d_gr = float(np.mean([_cosine_distance(item, centroid) for item in generated]))
        reliable = d_or > d_rr + 0.01 and d_tr > d_rr + 0.005
        per_app = []
        for app, target_item, generated_item in zip(app_names, target, generated):
            item_tr = _cosine_distance(target_item, centroid)
            item_gr = _cosine_distance(generated_item, centroid)
            item_score = _relative_transfer(item_tr, item_gr, d_rr) if reliable else None
            per_app.append({"app": app, "score": item_score})
            if item_score is not None:
                per_app_parts[app].append(item_score)
        item_scores = [item["score"] for item in per_app if item["score"] is not None]
        score = float(np.mean(item_scores)) if reliable and item_scores else None
        groups[name] = {
            "score": score,
            "reliable": reliable,
            "D_RR": d_rr,
            "D_OR": d_or,
            "D_TR": d_tr,
            "D_GR": d_gr,
            "per_app": per_app,
        }
        if score is not None:
            valid_scores.append(score)
    per_app = [
        {"app": app, "score": float(np.mean(values)) if values else None}
        for app, values in per_app_parts.items()
    ]
    return {
        "score": float(np.mean(valid_scores)) if valid_scores else None,
        "reliable": len(valid_scores) >= 2,
        "reliable_group_count": len(valid_scores),
        "groups": groups,
        "per_app": per_app,
    }


def _motif_style_score(raw_paths, theme_paths, target_paths, generated_paths, app_names, dense):
    if len(theme_paths) < 2:
        return {"score": None, "reliable": False, "reason": "At least two references are required.", "per_app": []}
    candidates = []
    for index, (raw_path, theme_path) in enumerate(zip(raw_paths, theme_paths)):
        theme_feature = dense[str(theme_path)]
        raw_feature = dense[str(raw_path)]
        other_patches = np.vstack(
            [dense[str(path)].selected_patches for other_index, path in enumerate(theme_paths) if other_index != index]
        )
        patches = theme_feature.selected_patches
        repeated = np.max(patches @ other_patches.T, axis=1)
        raw_match = np.max(patches @ raw_feature.selected_patches.T, axis=1)
        salience = repeated - raw_match
        top = np.argsort(salience)[::-1][:4]
        for patch_index in top:
            if salience[patch_index] > 0.015:
                candidates.append((float(salience[patch_index]), patches[patch_index]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    prototypes = np.vstack([item[1] for item in candidates[:32]]) if candidates else np.empty((0, 1), dtype=np.float32)
    if len(prototypes) < 4:
        return {"score": None, "reliable": False, "reason": "No repeated theme-specific dense motif was found.", "per_app": []}
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)

    def coverage(path):
        similarities = dense[str(path)].selected_patches @ prototypes.T
        return float(np.mean(np.max(similarities, axis=0)))

    reference_coverage = [coverage(path) for path in theme_paths]
    target_coverage = [coverage(path) for path in target_paths]
    generated_coverage = [coverage(path) for path in generated_paths]
    ref_baseline = float(np.median(reference_coverage))
    target_baseline = float(np.mean(target_coverage))
    generated_baseline = float(np.mean(generated_coverage))
    reliable = ref_baseline > target_baseline + 0.015
    score = _relative_gain(target_baseline, generated_baseline, ref_baseline) if reliable else None
    per_app = [
        {
            "app": app,
            "score": _relative_gain(target_value, generated_value, ref_baseline) if reliable else None,
            "target_coverage": target_value,
            "generated_coverage": generated_value,
        }
        for app, target_value, generated_value in zip(app_names, target_coverage, generated_coverage)
    ]
    return {
        "score": score,
        "reliable": reliable,
        "prototype_count": len(prototypes),
        "reference_coverage": ref_baseline,
        "target_coverage": target_baseline,
        "generated_coverage": generated_baseline,
        "per_app": per_app,
    }


def _dense_identity_score(raw_paths, theme_paths, target_paths, generated_paths, app_names, dense):
    positive = [dense_correspondence(dense[str(raw)], dense[str(theme)])["score"] for raw, theme in zip(raw_paths, theme_paths)]
    distractor = [
        dense_correspondence(dense[str(target_paths[left])], dense[str(target_paths[right])])["score"]
        for left in range(len(target_paths))
        for right in range(left + 1, len(target_paths))
    ]
    positive_baseline = float(np.median(positive)) if positive else 1.0
    distractor_baseline = float(np.median(distractor)) if distractor else positive_baseline - 0.20
    if positive_baseline <= distractor_baseline + 1e-6:
        positive_baseline = distractor_baseline + max(float(np.std(positive)) * 3.0, 0.08)

    per_app = []
    generated_to_target = []
    for app, generated_path, target_path in zip(app_names, generated_paths, target_paths):
        row = []
        for candidate_target in target_paths:
            row.append(dense_correspondence(dense[str(generated_path)], dense[str(candidate_target)])["score"])
        generated_to_target.append(row)
        own = dense_correspondence(dense[str(generated_path)], dense[str(target_path)])
        score = float(np.clip((own["score"] - distractor_baseline) / max(positive_baseline - distractor_baseline, 1e-8), 0.0, 1.0) * 100.0)
        per_app.append(
            {
                "app": app,
                "score": score,
                "raw_correspondence": own["score"],
                "appearance_similarity": own["appearance_similarity"],
                "spatial_consistency": own["spatial_consistency"],
            }
        )
    scores = [item["score"] for item in per_app]
    return {
        "score": float(np.mean(scores)) if scores else 0.0,
        "p10_score": float(np.percentile(scores, 10)) if scores else 0.0,
        "reliable": bool(positive and distractor),
        "positive_reference_similarity": positive_baseline,
        "distractor_similarity": distractor_baseline,
        "per_app": per_app,
        "pairwise": {
            "app_names": app_names,
            "generated_to_target_dense_similarity": generated_to_target,
            "reference_pair_identity_similarity": positive,
        },
    }


def _merge_identity_rows(app_names, dense_rows, dists_rows, lpips_rows, component_config):
    maps = [{item["app"]: item for item in rows} for rows in (dense_rows, dists_rows, lpips_rows)]
    output = []
    for app in app_names:
        values = {
            "dino_dense": maps[0].get(app, {}).get("score"),
            "dists_structure": maps[1].get(app, {}).get("score"),
            "lpips_content": maps[2].get(app, {}).get("score"),
        }
        weighted = {
            name: {"score": value, "reliable": component_config[name]["reliable"], "weight": component_config[name]["weight"]}
            for name, value in values.items()
        }
        score, _ = _weighted_available(weighted)
        output.append({"app": app, "score": score, "components": values})
    return output


def _package_coherence(theme_paths, target_paths, generated_paths, app_names, features):
    theme = _matrix(theme_paths, features)
    target = _matrix(target_paths, features)
    generated = _matrix(generated_paths, features)
    rr = _pairwise(theme)
    gg = _pairwise(generated)
    ref_p90 = float(np.percentile(rr, 90)) if len(rr) else 0.0
    gen_p90 = float(np.percentile(gg, 90)) if len(gg) else 0.0
    ref_mad = _mad(rr)
    unity = float(100.0 * np.exp(-max(gen_p90 - ref_p90, 0.0) / max(3.0 * ref_mad, 0.03)))

    ref_centroid = _normalized_centroid(theme)
    generated_to_ref = np.asarray([_cosine_distance(item, ref_centroid) for item in generated], dtype=np.float32)
    target_to_ref = np.asarray([_cosine_distance(item, ref_centroid) for item in target], dtype=np.float32)
    d_rr = _leave_one_out_distance(theme)
    membership_rows = []
    for app, d_tr, d_gr in zip(app_names, target_to_ref, generated_to_ref):
        membership_rows.append({"app": app, "score": _relative_transfer(float(d_tr), float(d_gr), d_rr), "distance_to_reference": float(d_gr)})
    membership_score = float(np.mean([item["score"] for item in membership_rows]))

    generated_centroid = _normalized_centroid(generated)
    internal = np.asarray([_cosine_distance(item, generated_centroid) for item in generated], dtype=np.float32)
    threshold = float(np.median(internal) + max(3.0 * _mad(internal), 0.02))
    reference_limit = float(max(ref_p90 + 3.0 * ref_mad, d_rr + 0.03))
    outliers = []
    for item, internal_distance in zip(membership_rows, internal):
        is_outlier = bool(internal_distance > threshold and item["distance_to_reference"] > reference_limit)
        item["internal_distance"] = float(internal_distance)
        item["is_outlier"] = is_outlier
        if is_outlier:
            outliers.append(item["app"])
    score = float(np.sqrt(max(unity, 0.0) * max(membership_score, 0.0)))
    return {
        "score": score,
        "package_unity_score": unity,
        "theme_membership_score": membership_score,
        "generated_p90_distance": gen_p90,
        "reference_p90_distance": ref_p90,
        "reference_mad": ref_mad,
        "outlier_apps": outliers,
        "per_app": membership_rows,
    }


def _per_app_report(app_names, style, identity, package, quality, total):
    attribute = {item["app"]: item for item in style["visual_attributes"]["per_app"]}
    motif = {item["app"]: item for item in style["dino_motif"].get("per_app", [])}
    identity_map = {item["app"]: item for item in identity["per_app"]}
    package_map = {item["app"]: item for item in package["per_app"]}
    quality_map = {item["app"]: item for item in quality["per_app"]}
    output = []
    for app in app_names:
        style_values = [attribute.get(app, {}).get("score"), motif.get(app, {}).get("score")]
        style_values = [value for value in style_values if value is not None]
        output.append(
            {
                "app": app,
                "itte_score": total,
                "style_fidelity_score": float(np.mean(style_values)) if style_values else style["score"],
                "identity_preservation_score": identity_map[app]["score"],
                "package_membership_score": package_map[app]["score"],
                "visual_quality_score": quality_map[app]["score"],
                "is_package_outlier": package_map[app]["is_outlier"],
                "quality_warnings": "|".join(quality_map[app]["warnings"]),
                "identity_components": json.dumps(identity_map[app]["components"], ensure_ascii=False),
            }
        )
    return output


def _hard_failures(identity, package, quality, style_available_weight):
    failures = []
    for item in identity["per_app"]:
        if item["score"] < 35.0:
            failures.append({"type": "identity_below_35", "app": item["app"], "score": item["score"]})
    if identity["p10_score"] < 45.0:
        failures.append({"type": "identity_p10_below_45", "score": identity["p10_score"]})
    failures.extend({"type": "visual_quality", **item} for item in quality["hard_failures"])
    if package["outlier_apps"]:
        failures.append({"type": "severe_package_outliers", "apps": package["outlier_apps"]})
    if style_available_weight < 0.60:
        failures.append({"type": "insufficient_style_evidence", "available_weight": style_available_weight})
    return failures


def _decision(total, style, identity, package, quality, confidence, hard_failures):
    if confidence == "low":
        return "insufficient_reference_evidence"
    if hard_failures:
        return "failed_hard_gate"
    if total >= 80.0:
        return "style_transfer_success_strong"
    if total >= 70.0:
        return "style_transfer_success"
    if style < 60.0:
        return "style_transfer_weak"
    if identity["score"] < 60.0:
        return "identity_preservation_weak"
    if package["score"] < 60.0:
        return "package_coherence_weak"
    if quality["score"] < 60.0:
        return "visual_quality_weak"
    return "needs_review"


def _evaluation_confidence(reference_count, style_weight, identity_weight, attributes):
    if reference_count >= 8 and style_weight >= 0.80 and identity_weight >= 0.80 and attributes["reliable"]:
        return "high"
    if reference_count >= 5 and style_weight >= 0.60 and identity_weight >= 0.50:
        return "medium"
    return "low"


def _component(score, reliable, weight):
    return {"score": score, "reliable": bool(reliable and score is not None), "weight": weight}


def _weighted_available(components):
    included = [item for item in components.values() if item.get("reliable") and item.get("score") is not None]
    available_weight = float(sum(item["weight"] for item in included))
    if available_weight <= 1e-8:
        return 0.0, 0.0
    score = sum(float(item["score"]) * float(item["weight"]) for item in included) / available_weight
    return float(np.clip(score, 0.0, 100.0)), available_weight


def _weighted_score(items):
    return float(sum(score * weight for score, weight in items) / sum(weight for _, weight in items))


def _relative_transfer(d_tr, d_gr, d_rr):
    if d_tr <= d_rr + 1e-8:
        return 0.0
    return float(np.clip((d_tr - d_gr) / (d_tr - d_rr), 0.0, 1.0) * 100.0)


def _relative_gain(target, generated, reference):
    if reference <= target + 1e-8:
        return 0.0
    return float(np.clip((generated - target) / (reference - target), 0.0, 1.0) * 100.0)


def _matrix(paths, features):
    return np.vstack([features[str(path)] for path in paths]).astype(np.float32)


def _group_matrix(paths, features, group):
    return np.vstack([features[str(path)][group] for path in paths]).astype(np.float32)


def _pairwise(matrix):
    if len(matrix) < 2:
        return np.asarray([], dtype=np.float32)
    return np.asarray([_cosine_distance(matrix[i], matrix[j]) for i in range(len(matrix)) for j in range(i + 1, len(matrix))], dtype=np.float32)


def _normalized_centroid(matrix):
    vector = matrix.mean(axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _leave_one_out_distance(matrix):
    if len(matrix) < 2:
        return 0.0
    values = []
    for index, item in enumerate(matrix):
        values.append(_cosine_distance(item, _normalized_centroid(np.delete(matrix, index, axis=0))))
    return float(np.median(values))


def _cosine_distance(left, right):
    return float(1.0 - np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8))


def _mad(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(array - np.median(array)))) if len(array) else 0.0


def _style_pairwise_payload(theme_paths, target_paths, generated_paths, app_names, features):
    theme = _matrix(theme_paths, features)
    target = _matrix(target_paths, features)
    generated = _matrix(generated_paths, features)
    return {
        "app_names": app_names,
        "theme_ref_pairwise": _distance_matrix(theme, theme).tolist(),
        "generated_pairwise": _distance_matrix(generated, generated).tolist(),
        "target_pairwise": _distance_matrix(target, target).tolist(),
        "generated_to_theme": _distance_matrix(generated, theme).tolist(),
        "target_to_theme": _distance_matrix(target, theme).tolist(),
    }


def _distance_matrix(left, right):
    return np.asarray([[_cosine_distance(a, b) for b in right] for a in left], dtype=np.float32)


def _load_generation_qwen_diagnostic(root, package_id):
    path = root / "data" / "packages" / package_id / "package_qc_report.json"
    if not path.exists():
        return {"available": False}
    try:
        return {"available": True, "report": json.loads(path.read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _provenance(root, paths, config):
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    config_payload = {
        "embedding_backend": config.embedding_backend,
        "model_source": config.model_source,
        "model_id": config.model_id,
        "image_size": config.image_size,
        "use_perceptual": config.use_perceptual,
        "use_vgg_gram": config.use_vgg_gram,
        "style_weights": STYLE_COMPONENT_WEIGHTS,
        "identity_weights": IDENTITY_COMPONENT_WEIGHTS,
    }
    return {
        "input_sha256": hashes,
        "model_revisions": {
            "dinov3": config.model_id,
            "vgg_gram": "torchvision-vgg16-default",
            "dists": "DISTS-pytorch-0.1",
            "lpips": "LPIPS-vgg-0.1",
        },
        "code_commit": commit,
        "scoring_config_hash": hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest(),
        "scoring_config": config_payload,
    }
