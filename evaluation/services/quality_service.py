from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


QUALITY_FIELDS = (
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "edge_density",
    "foreground_ratio",
    "border_dark_ratio",
    "border_foreground_contact",
    "border_edge_ratio",
    "semi_transparent_ratio",
)


def compute_visual_quality(
    theme_paths: list[Path],
    generated_paths: list[Path],
    app_names: list[str],
    image_size: int = 448,
) -> dict:
    reference = [_image_quality_metrics(path, image_size) for path in theme_paths]
    generated = [_image_quality_metrics(path, image_size) for path in generated_paths]
    envelopes = {
        field: _robust_envelope([item[field] for item in reference])
        for field in QUALITY_FIELDS
    }

    per_app = []
    hard_failures = []
    for app, metrics in zip(app_names, generated):
        score = 100.0
        warnings: list[str] = []
        z_scores = {
            field: _robust_z(metrics[field], envelopes[field])
            for field in QUALITY_FIELDS
        }
        if metrics["border_edge_ratio"] > 0.30:
            score -= 45.0
            warnings.append("severe_crop_or_subject_touches_border")
        elif metrics["border_edge_ratio"] > 0.18:
            score -= 20.0
            warnings.append("possible_crop")
        if metrics["border_dark_ratio"] > 0.12 and envelopes["border_dark_ratio"][0] < 0.05:
            score -= 40.0
            warnings.append("large_black_border")
        if metrics["semi_transparent_ratio"] > 0.35 and envelopes["semi_transparent_ratio"][0] < 0.08:
            score -= 35.0
            warnings.append("abnormal_transparency")
        if metrics["sharpness"] < 2e-4 and z_scores["sharpness"] < -4.0:
            score -= 30.0
            warnings.append("severe_blur_or_empty_image")
        if metrics["foreground_ratio"] < 0.03:
            score -= 40.0
            warnings.append("subject_area_too_small")
        if metrics["brightness"] < 0.04 or metrics["brightness"] > 0.97:
            score -= 30.0
            warnings.append("extreme_exposure")
        for field in ("contrast", "saturation", "edge_density", "foreground_ratio"):
            if abs(z_scores[field]) > 8.0:
                score -= 7.5
                warnings.append(f"{field}_outside_reference_envelope")

        score = float(np.clip(score, 0.0, 100.0))
        severe = [
            warning
            for warning in warnings
            if warning in {
                "severe_crop_or_subject_touches_border",
                "large_black_border",
                "abnormal_transparency",
                "severe_blur_or_empty_image",
            }
        ]
        if severe:
            hard_failures.append({"app": app, "reasons": severe})
        per_app.append(
            {
                "app": app,
                "score": score,
                "warnings": warnings,
                "metrics": metrics,
                "robust_z": z_scores,
            }
        )

    scores = [item["score"] for item in per_app]
    return {
        "score": float(np.mean(scores)) if scores else 0.0,
        "p10_score": float(np.percentile(scores, 10)) if scores else 0.0,
        "per_app": per_app,
        "hard_failures": hard_failures,
        "reference_envelopes": {
            key: {"median": value[0], "mad": value[1]}
            for key, value in envelopes.items()
        },
        "method": "deterministic_reference_normalized_artifact_checks",
    }


def _image_quality_metrics(path: Path, image_size: int) -> dict[str, float]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA").resize((image_size, image_size), Image.Resampling.BICUBIC)
        hsv = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC).convert("HSV")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    gray = rgb.mean(axis=2)
    saturation = np.asarray(hsv, dtype=np.float32)[:, :, 1] / 255.0

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    magnitude = np.sqrt(gx * gx + gy * gy)
    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center

    if float((alpha < 0.99).mean()) > 0.01:
        foreground = alpha > 0.05
    else:
        corners = np.vstack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
        background = np.median(corners, axis=0)
        foreground = np.linalg.norm(rgb - background, axis=2) > 0.10

    border_width = max(2, image_size // 32)
    border_gray = np.concatenate(
        [
            gray[:border_width].reshape(-1),
            gray[-border_width:].reshape(-1),
            gray[:, :border_width].reshape(-1),
            gray[:, -border_width:].reshape(-1),
        ]
    )
    border_foreground = np.concatenate(
        [
            foreground[:border_width].reshape(-1),
            foreground[-border_width:].reshape(-1),
            foreground[:, :border_width].reshape(-1),
            foreground[:, -border_width:].reshape(-1),
        ]
    )
    border_edges = np.concatenate(
        [
            magnitude[:border_width].reshape(-1),
            magnitude[-border_width:].reshape(-1),
            magnitude[:, :border_width].reshape(-1),
            magnitude[:, -border_width:].reshape(-1),
        ]
    )
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(saturation.mean()),
        "sharpness": float(laplacian.var()) if laplacian.size else 0.0,
        "edge_density": float((magnitude > 0.08).mean()),
        "foreground_ratio": float(foreground.mean()),
        "border_dark_ratio": float((border_gray < 0.06).mean()),
        "border_foreground_contact": float(border_foreground.mean()),
        "border_edge_ratio": float((border_edges > 0.08).mean()),
        "semi_transparent_ratio": float(((alpha > 0.02) & (alpha < 0.98)).mean()),
    }


def _robust_envelope(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr)) if len(arr) else 0.0
    mad = float(np.median(np.abs(arr - median))) if len(arr) else 0.0
    return median, max(mad, 1e-4)


def _robust_z(value: float, envelope: tuple[float, float]) -> float:
    median, mad = envelope
    return float(0.6745 * (value - median) / max(mad, 1e-8))
