from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def image_statistics(path: Path, image_size: int = 224) -> tuple[np.ndarray, dict]:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0

    rgb_mean = arr.reshape(-1, 3).mean(axis=0)
    rgb_std = arr.reshape(-1, 3).std(axis=0)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float(hsv[:, :, 1].mean())
    colorfulness = float(np.sqrt(rgb_std[0] ** 2 + rgb_std[1] ** 2 + rgb_std[2] ** 2))
    edge_density = _edge_density(gray)

    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(hsv[:, :, channel], bins=16, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist_parts.append(hist / max(float(hist.sum()), 1.0))

    vector = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            np.array([brightness, saturation, contrast, edge_density, colorfulness], dtype=np.float32),
            *hist_parts,
        ]
    ).astype(np.float32)
    details = {
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "edge_density": edge_density,
        "colorfulness": colorfulness,
    }
    return vector, details


def visual_statistics_score(theme_paths: list[Path], generated_paths: list[Path], target_paths: list[Path], image_size: int) -> dict:
    theme = np.vstack([image_statistics(path, image_size=image_size)[0] for path in theme_paths])
    generated = np.vstack([image_statistics(path, image_size=image_size)[0] for path in generated_paths])
    targets = np.vstack([image_statistics(path, image_size=image_size)[0] for path in target_paths])

    theme_center = theme.mean(axis=0)
    generated_center = generated.mean(axis=0)
    target_center = targets.mean(axis=0)
    generated_distance = float(np.linalg.norm(generated_center - theme_center))
    target_distance = float(np.linalg.norm(target_center - theme_center))
    improvement = max(target_distance - generated_distance, 0.0) / max(target_distance, 1e-8)
    score = max(0.0, min(100.0, improvement * 100.0))
    return {
        "score": score,
        "generated_to_theme_distance": generated_distance,
        "target_to_theme_distance": target_distance,
        "is_visual_stats_improved": generated_distance < target_distance,
    }


def stats_embedding(path: Path, image_size: int) -> np.ndarray:
    vector, _ = image_statistics(path, image_size=image_size)
    return vector


def _edge_density(gray: np.ndarray) -> float:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gradient = np.zeros_like(gray)
    gradient[:, 1:] += gx
    gradient[1:, :] += gy
    return float((gradient > 0.08).mean())
