from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def compute_style_cue_profile_match(
    theme_paths: list[Path],
    generated_paths: list[Path],
    target_paths: list[Path],
    image_size: int,
) -> dict:
    theme = _profile_matrix(theme_paths, image_size)
    generated = _profile_matrix(generated_paths, image_size)
    targets = _profile_matrix(target_paths, image_size)
    centroid = theme.mean(axis=0)
    d_rr = _leave_one_out_distance(theme)
    d_gr_per_app = np.asarray([_euclidean_distance(item, centroid) for item in generated], dtype=np.float32)
    d_tr_per_app = np.asarray([_euclidean_distance(item, centroid) for item in targets], dtype=np.float32)
    d_gr = float(np.mean(d_gr_per_app))
    d_tr = float(np.mean(d_tr_per_app))
    denominator = max(d_tr - d_rr, 0.05)
    score = max(0.0, min(100.0, (d_tr - d_gr) / denominator * 100.0))
    per_app = []
    for index, path in enumerate(generated_paths):
        if path.parent.name == "final":
            app = path.stem
        elif path.name == "best_output.png":
            app = path.parent.name
        elif path.parent.name == "cases":
            app = path.stem
        else:
            app = path.parent.name
        per_app_score = max(
            0.0,
            min(100.0, (float(d_tr_per_app[index]) - float(d_gr_per_app[index])) / denominator * 100.0),
        )
        per_app.append(
            {
                "app": app,
                "style_cue_profile_match_score": per_app_score,
                "d_to_style_cue_profile": float(d_gr_per_app[index]),
                "target_d_to_style_cue_profile": float(d_tr_per_app[index]),
            }
        )
    return {
        "score": score,
        "D_GR_style_cue_profile": d_gr,
        "D_TR_style_cue_profile": d_tr,
        "D_RR_style_cue_profile": d_rr,
        "per_app": per_app,
        "feature_names": _STYLE_CUE_FEATURE_NAMES,
    }


_STYLE_CUE_FEATURE_NAMES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_saturation",
    "std_saturation",
    "mean_value",
    "background_r",
    "background_g",
    "background_b",
    "background_saturation",
    "background_value",
    "border_std",
    "foreground_ratio",
    "foreground_center_x",
    "foreground_center_y",
    "foreground_bbox_w",
    "foreground_bbox_h",
    "foreground_margin",
    "edge_density",
    "edge_strength",
    "dark_edge_ratio",
    "black_pixel_ratio",
    "gray_entropy",
    "quantized_color_count",
    "texture_energy",
    "pastel_softness",
]


def _profile_matrix(paths: list[Path], image_size: int) -> np.ndarray:
    matrix = np.vstack([_extract_style_cue_profile(path, image_size) for path in paths]).astype(np.float32)
    return matrix


def _extract_style_cue_profile(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = _composite_on_light_background(image).resize((image_size, image_size))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    value = mx

    corner = max(4, image_size // 18)
    corner_samples = np.vstack(
        [
            arr[:corner, :corner].reshape(-1, 3),
            arr[:corner, -corner:].reshape(-1, 3),
            arr[-corner:, :corner].reshape(-1, 3),
            arr[-corner:, -corner:].reshape(-1, 3),
        ]
    )
    background = np.median(corner_samples, axis=0)
    bg_max = float(background.max())
    bg_min = float(background.min())
    bg_sat = (bg_max - bg_min) / max(bg_max, 1e-6)
    bg_val = bg_max

    border = max(4, image_size // 14)
    border_pixels = np.vstack(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ]
    )
    foreground_mask = np.linalg.norm(arr - background, axis=2) > 0.12
    if foreground_mask.any():
        ys, xs = np.nonzero(foreground_mask)
        center_x = float(xs.mean() / max(image_size - 1, 1))
        center_y = float(ys.mean() / max(image_size - 1, 1))
        bbox_w = float((xs.max() - xs.min() + 1) / image_size)
        bbox_h = float((ys.max() - ys.min() + 1) / image_size)
        margin = float(min(xs.min(), ys.min(), image_size - 1 - xs.max(), image_size - 1 - ys.max()) / image_size)
    else:
        center_x = center_y = 0.5
        bbox_w = bbox_h = 0.0
        margin = 0.5

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    magnitude = np.sqrt(gx * gx + gy * gy)
    edge_mask = magnitude > 0.06
    dark_mask = gray < 0.28
    dark_edge_ratio = float(np.logical_and(edge_mask, dark_mask).sum() / max(edge_mask.sum(), 1))

    hist_gray, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=False)
    probs = hist_gray.astype(np.float32) / max(float(hist_gray.sum()), 1.0)
    entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum() / 5.0)
    quantized = np.floor(arr * 8).astype(np.int16).reshape(-1, 3)
    color_count = float(len(np.unique(quantized, axis=0)) / 512.0)
    pastel_softness = float(np.mean(value * (1.0 - saturation)))

    return np.asarray(
        [
            *arr.mean(axis=(0, 1)).tolist(),
            float(saturation.mean()),
            float(saturation.std()),
            float(value.mean()),
            *background.tolist(),
            bg_sat,
            bg_val,
            float(border_pixels.std()),
            float(foreground_mask.mean()),
            center_x,
            center_y,
            bbox_w,
            bbox_h,
            margin,
            float(edge_mask.mean()),
            float(magnitude.mean()),
            dark_edge_ratio,
            float(dark_mask.mean()),
            entropy,
            color_count,
            float(magnitude.mean()),
            pastel_softness,
        ],
        dtype=np.float32,
    )


def _leave_one_out_distance(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 0.0
    distances = []
    for index in range(len(matrix)):
        others = np.delete(matrix, index, axis=0)
        distances.append(_euclidean_distance(matrix[index], others.mean(axis=0)))
    return float(np.mean(distances))


def _euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)))


def _composite_on_light_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")
