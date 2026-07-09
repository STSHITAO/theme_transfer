from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


def compute_vgg_gram_style_fit(
    theme_paths: list[Path],
    generated_paths: list[Path],
    target_paths: list[Path],
    root_dir: Path,
    enabled: bool,
) -> dict:
    if not enabled:
        return {
            "score": None,
            "enabled": False,
            "reliable": False,
            "used_in_itte_score": False,
            "reason": "VGG Gram auxiliary signal is disabled.",
        }

    try:
        vectors = _extract_vgg_vectors([*theme_paths, *generated_paths, *target_paths], root_dir)
    except Exception as exc:
        return {
            "score": None,
            "enabled": True,
            "reliable": False,
            "used_in_itte_score": False,
            "reason": f"VGG Gram auxiliary signal unavailable; fallback to lightweight ITTE attributes. {type(exc).__name__}: {exc}",
        }

    theme = _stack([vectors[str(path)] for path in theme_paths])
    generated = _stack([vectors[str(path)] for path in generated_paths])
    targets = _stack([vectors[str(path)] for path in target_paths])
    centroid = _normalized_centroid(theme)
    d_rr = _leave_one_out_distance(theme)
    d_gr = float(np.mean([_cosine_distance(item, centroid) for item in generated]))
    d_tr = float(np.mean([_cosine_distance(item, centroid) for item in targets]))
    score = max(0.0, min(100.0, (d_tr - d_gr) / max(d_tr - d_rr, 1e-8) * 100.0))
    return {
        "score": score,
        "enabled": True,
        "reliable": d_tr > d_rr,
        "used_in_itte_score": False,
        "D_TR_vgg_gram": d_tr,
        "D_GR_vgg_gram": d_gr,
        "D_RR_vgg_gram": d_rr,
        "reason": "VGG Gram is an auxiliary texture/material style signal; it is not used in itte_score or decision.",
    }


def _extract_vgg_vectors(paths: list[Path], root_dir: Path) -> dict[str, np.ndarray]:
    os.environ["TORCH_HOME"] = str(root_dir / "models" / "torch")
    (root_dir / "models" / "torch").mkdir(parents=True, exist_ok=True)

    import torch
    from torchvision import models, transforms

    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights).features[:16].eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    vectors = {}
    with torch.no_grad():
        for path in paths:
            with Image.open(path) as image:
                tensor = preprocess(image.convert("RGB")).unsqueeze(0)
            features = model(tensor)
            gram = _gram_matrix(features).cpu().numpy().reshape(-1).astype(np.float32)
            vectors[str(path)] = _l2_normalize(gram)
    return vectors


def _gram_matrix(tensor):
    _, channels, height, width = tensor.shape
    features = tensor.reshape(channels, height * width)
    return features @ features.T / max(channels * height * width, 1)


def _stack(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def _normalized_centroid(matrix: np.ndarray) -> np.ndarray:
    centroid = matrix.mean(axis=0)
    return _l2_normalize(centroid.astype(np.float32))


def _leave_one_out_distance(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 0.0
    distances = []
    for index, item in enumerate(matrix):
        others = np.delete(matrix, index, axis=0)
        distances.append(_cosine_distance(item, _normalized_centroid(others)))
    return float(np.mean(distances))


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8))


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm
