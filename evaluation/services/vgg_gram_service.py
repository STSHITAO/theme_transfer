from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from evaluation.services.image_view_service import load_image_view


def compute_vgg_gram_style_fit(
    theme_paths: list[Path],
    generated_paths: list[Path],
    target_paths: list[Path],
    root_dir: Path,
    enabled: bool,
    device: str = "cpu",
    image_size: int = 224,
) -> dict:
    if not enabled:
        return {
            "score": None,
            "enabled": False,
            "reliable": False,
            "used_in_itte_score": False,
            "reason": "VGG Gram main style signal is disabled.",
        }

    try:
        vectors, cache_info = _extract_vgg_vectors(
            [*theme_paths, *generated_paths, *target_paths],
            root_dir,
            device=device,
            image_size=image_size,
        )
    except Exception as exc:
        return {
            "score": None,
            "enabled": True,
            "reliable": False,
            "used_in_itte_score": False,
            "reason": f"VGG Gram style signal unavailable; remaining reliable style components are renormalized. {type(exc).__name__}: {exc}",
        }

    theme = _stack([vectors[str(path)] for path in theme_paths])
    generated = _stack([vectors[str(path)] for path in generated_paths])
    targets = _stack([vectors[str(path)] for path in target_paths])
    centroid = _normalized_centroid(theme)
    d_rr = _leave_one_out_distance(theme)
    d_gr = float(np.mean([_cosine_distance(item, centroid) for item in generated]))
    d_tr = float(np.mean([_cosine_distance(item, centroid) for item in targets]))
    score = max(0.0, min(100.0, (d_tr - d_gr) / max(d_tr - d_rr, 1e-8) * 100.0))
    per_app = []
    for index, (generated_item, target_item) in enumerate(zip(generated, targets)):
        generated_distance = _cosine_distance(generated_item, centroid)
        target_distance = _cosine_distance(target_item, centroid)
        item_score = max(
            0.0,
            min(100.0, (target_distance - generated_distance) / max(target_distance - d_rr, 1e-8) * 100.0),
        )
        per_app.append(
            {
                "index": index,
                "score": item_score,
                "generated_to_theme_distance": generated_distance,
                "target_to_theme_distance": target_distance,
            }
        )
    return {
        "score": score,
        "enabled": True,
        "reliable": d_tr > d_rr,
        "used_in_itte_score": d_tr > d_rr,
        "D_TR_vgg_gram": d_tr,
        "D_GR_vgg_gram": d_gr,
        "D_RR_vgg_gram": d_rr,
        "per_app": per_app,
        "layers": ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
        "device": cache_info["device"],
        "cache": cache_info,
        "reason": "Multi-layer VGG Gram is the Gatys-derived primary texture/style representation in ITTE v1.3.",
    }


def _extract_vgg_vectors(
    paths: list[Path],
    root_dir: Path,
    device: str = "cpu",
    image_size: int = 224,
) -> tuple[dict[str, np.ndarray], dict]:
    os.environ["TORCH_HOME"] = str(root_dir / "models" / "torch")
    (root_dir / "models" / "torch").mkdir(parents=True, exist_ok=True)

    unique_paths = list(dict.fromkeys(paths))
    cache_dir = root_dir / "data" / "evaluations" / "_cache" / "vgg_gram"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors: dict[str, np.ndarray] = {}
    missing: list[tuple[Path, Path]] = []
    for path in unique_paths:
        cache_path = _vgg_cache_path(path, cache_dir, image_size)
        cached = _load_cached_vector(cache_path)
        if cached is None:
            missing.append((path, cache_path))
        else:
            vectors[str(path)] = cached

    import torch
    from torchvision import models, transforms

    actual_device = device
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        actual_device = "cpu"
    if not missing:
        return vectors, {
            "directory": str(cache_dir),
            "hits": len(unique_paths),
            "misses": 0,
            "device": actual_device,
        }

    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights).features[:23].to(actual_device).eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        for path, cache_path in missing:
            image = load_image_view(path, "appearance", image_size)
            tensor = preprocess(image).unsqueeze(0).to(actual_device)
            grams = []
            features = tensor
            for index, layer in enumerate(model):
                features = layer(features)
                if index in {3, 8, 15, 22}:
                    gram = _gram_matrix(features).cpu().numpy().reshape(-1).astype(np.float32)
                    grams.append(_l2_normalize(gram))
            vector = _l2_normalize(np.concatenate(grams).astype(np.float32))
            vectors[str(path)] = vector
            _save_cached_vector(cache_path, vector)
    return vectors, {
        "directory": str(cache_dir),
        "hits": len(unique_paths) - len(missing),
        "misses": len(missing),
        "device": actual_device,
    }


def _vgg_cache_path(path: Path, cache_dir: Path, image_size: int) -> Path:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "view": "appearance-v1",
        "image_size": image_size,
        "model": "torchvision-vgg16-imagenet1k-v1",
        "layers": [3, 8, 15, 22],
        "format": 1,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def _load_cached_vector(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        vector = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            return None
        return vector
    except Exception:
        return None


def _save_cached_vector(path: Path, vector: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, vector, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
