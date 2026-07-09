from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.services.visual_stats_service import image_statistics


STYLE_FEATURE_GROUPS = ("color", "edge", "composition", "complexity")


def extract_style_features(paths: list[Path], config, root_dir: Path | None = None) -> dict[str, np.ndarray]:
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
    cache_dir = root / "data" / "evaluations" / "_cache" / "style_features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    features: dict[str, np.ndarray] = {}
    for path in paths:
        cache_path = _style_cache_path(path, config, cache_dir)
        if cache_path.exists():
            features[str(path)] = np.load(cache_path)
            continue
        vector = _extract_single_style_vector(path, image_size=config.image_size)
        vector = _l2_normalize(vector.astype(np.float32))
        np.save(cache_path, vector)
        features[str(path)] = vector
    return features


def extract_style_feature_groups(paths: list[Path], config, root_dir: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
    cache_dir = root / "data" / "evaluations" / "_cache" / "style_feature_groups"
    cache_dir.mkdir(parents=True, exist_ok=True)

    grouped_features: dict[str, dict[str, np.ndarray]] = {}
    for path in paths:
        cache_path = _style_group_cache_path(path, config, cache_dir)
        if cache_path.exists():
            loaded = np.load(cache_path)
            grouped_features[str(path)] = {name: loaded[name] for name in STYLE_FEATURE_GROUPS}
            continue
        vector = _extract_single_style_vector(path, image_size=config.image_size).astype(np.float32)
        groups = _split_style_vector(vector)
        np.savez(cache_path, **groups)
        grouped_features[str(path)] = groups
    return grouped_features


def _split_style_vector(vector: np.ndarray) -> dict[str, np.ndarray]:
    if len(vector) < 80:
        normalized = _l2_normalize(vector.astype(np.float32))
        return {name: normalized for name in STYLE_FEATURE_GROUPS}

    color_indices = np.r_[0:8, 10:59]
    edge_indices = np.r_[8:10, 59:71]
    composition_indices = np.r_[71:77]
    complexity_indices = np.r_[8:10, 77:80]
    return {
        "color": _l2_normalize(vector[color_indices].astype(np.float32)),
        "edge": _l2_normalize(vector[edge_indices].astype(np.float32)),
        "composition": _l2_normalize(vector[composition_indices].astype(np.float32)),
        "complexity": _l2_normalize(vector[complexity_indices].astype(np.float32)),
    }


def _extract_single_style_vector(path: Path, image_size: int) -> np.ndarray:
    base_vector, _ = image_statistics(path, image_size=image_size)
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)
    edge_mask = magnitude > 0.06
    if edge_mask.any():
        orientation_hist, _ = np.histogram(angle[edge_mask], bins=12, range=(0.0, 1.0), weights=magnitude[edge_mask])
        orientation_hist = orientation_hist.astype(np.float32)
        orientation_hist = orientation_hist / max(float(orientation_hist.sum()), 1.0)
    else:
        orientation_hist = np.zeros(12, dtype=np.float32)

    corner_samples = np.vstack(
        [
            arr[:8, :8].reshape(-1, 3),
            arr[:8, -8:].reshape(-1, 3),
            arr[-8:, :8].reshape(-1, 3),
            arr[-8:, -8:].reshape(-1, 3),
        ]
    )
    background_color = np.median(corner_samples, axis=0)
    foreground_mask = np.linalg.norm(arr - background_color, axis=2) > 0.12
    foreground_ratio = float(foreground_mask.mean())
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

    hist_gray, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=False)
    probs = hist_gray.astype(np.float32) / max(float(hist_gray.sum()), 1.0)
    entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum() / 5.0)
    quantized = np.floor(arr * 8).astype(np.int16).reshape(-1, 3)
    color_count = float(len(np.unique(quantized, axis=0)) / 512.0)
    texture_energy = float(np.mean(magnitude))

    composition = np.asarray(
        [
            foreground_ratio,
            center_x,
            center_y,
            bbox_w,
            bbox_h,
            margin,
            entropy,
            color_count,
            texture_energy,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base_vector.astype(np.float32), orientation_hist, composition])


def _style_cache_path(path: Path, config, cache_dir: Path) -> Path:
    stat = path.stat()
    payload = {
        "abs_path": str(path.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "backend": config.style_feature_backend,
        "image_size": config.image_size,
        "version": 1,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def _style_group_cache_path(path: Path, config, cache_dir: Path) -> Path:
    stat = path.stat()
    payload = {
        "abs_path": str(path.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "backend": config.style_feature_backend,
        "image_size": config.image_size,
        "version": 1,
        "groups": STYLE_FEATURE_GROUPS,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npz"


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm
