from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.services.embedding_service import (
    TpqsConfig,
    _resolve_dinov3_model_path,
    prepare_model_cache_dirs,
)
from evaluation.services.image_view_service import foreground_mask, load_image_view


@dataclass(frozen=True)
class DenseImageFeature:
    patches: np.ndarray
    coordinates: np.ndarray
    foreground: np.ndarray

    @property
    def selected_patches(self) -> np.ndarray:
        selected = self.patches[self.foreground]
        return selected if len(selected) >= 4 else self.patches

    @property
    def selected_coordinates(self) -> np.ndarray:
        selected = self.coordinates[self.foreground]
        return selected if len(selected) >= 4 else self.coordinates


def extract_dense_features(
    paths: list[Path],
    config: TpqsConfig,
    root_dir: Path | None = None,
    view: str = "structure",
) -> dict[str, DenseImageFeature]:
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
    cache_dir = root / "data" / "evaluations" / "_cache" / "dino_dense"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[str, DenseImageFeature] = {}
    missing: list[Path] = []
    for path in paths:
        cache_path = _cache_path(path, config, cache_dir, view)
        if cache_path.exists():
            cached[str(path)] = _load_feature(cache_path)
        else:
            missing.append(path)

    if not missing:
        return cached

    if config.embedding_backend == "stats":
        for path in missing:
            feature = _stats_dense_feature(path, view=view)
            _save_feature(_cache_path(path, config, cache_dir, view), feature)
            cached[str(path)] = feature
        return cached
    if config.embedding_backend != "dinov3":
        raise ValueError(f"Unsupported ITTE dense backend: {config.embedding_backend}")

    cache_dirs = prepare_model_cache_dirs(root)
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model_path = _resolve_dinov3_model_path(config, cache_dirs)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    patch_size = int(getattr(model.config, "patch_size", 16))

    batch_size = max(config.batch_size, 1)
    for start in range(0, len(missing), batch_size):
        batch_paths = missing[start : start + batch_size]
        images = [load_image_view(path, view, config.image_size) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        grid_h = int(pixel_values.shape[-2] // patch_size)
        grid_w = int(pixel_values.shape[-1] // patch_size)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        patch_count = grid_h * grid_w
        patch_tensor = hidden[:, 1 + register_tokens : 1 + register_tokens + patch_count]
        patch_tensor = torch.nn.functional.normalize(patch_tensor, dim=-1)
        patch_arrays = patch_tensor.detach().cpu().numpy().astype(np.float32)
        coordinates = _grid_coordinates(grid_h, grid_w)
        for path, image, patches in zip(batch_paths, images, patch_arrays):
            feature = DenseImageFeature(
                patches=patches,
                coordinates=coordinates,
                foreground=foreground_mask(image, grid_h, grid_w),
            )
            _save_feature(_cache_path(path, config, cache_dir, view), feature)
            cached[str(path)] = feature
    return cached


def dense_correspondence(left: DenseImageFeature, right: DenseImageFeature) -> dict[str, float]:
    left_patches = left.selected_patches
    right_patches = right.selected_patches
    left_coords = left.selected_coordinates
    right_coords = right.selected_coordinates
    similarities = np.clip(left_patches @ right_patches.T, -1.0, 1.0)

    left_matches = np.argmax(similarities, axis=1)
    right_matches = np.argmax(similarities, axis=0)
    left_similarity = similarities[np.arange(len(left_patches)), left_matches]
    right_similarity = similarities[right_matches, np.arange(len(right_patches))]
    appearance = float((left_similarity.mean() + right_similarity.mean()) / 2.0)

    left_spatial = np.linalg.norm(left_coords - right_coords[left_matches], axis=1)
    right_spatial = np.linalg.norm(right_coords - left_coords[right_matches], axis=1)
    spatial = float(
        (
            np.exp(-2.5 * left_spatial).mean()
            + np.exp(-2.5 * right_spatial).mean()
        )
        / 2.0
    )
    score = float(np.clip(0.75 * appearance + 0.25 * spatial, 0.0, 1.0))
    return {
        "score": score,
        "appearance_similarity": appearance,
        "spatial_consistency": spatial,
    }


def _stats_dense_feature(path: Path, grid_size: int = 7, view: str = "structure") -> DenseImageFeature:
    image = load_image_view(path, view, grid_size * 8)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            block = arr[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
            gray = block.mean(axis=2)
            patches.append(
                np.concatenate(
                    [
                        block.reshape(-1, 3).mean(axis=0),
                        block.reshape(-1, 3).std(axis=0),
                        np.asarray([gray.mean(), gray.std()], dtype=np.float32),
                    ]
                )
            )
    matrix = np.vstack(patches).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    return DenseImageFeature(
        patches=matrix,
        coordinates=_grid_coordinates(grid_size, grid_size),
        foreground=foreground_mask(image, grid_size, grid_size),
    )


def _grid_coordinates(height: int, width: int) -> np.ndarray:
    rows, cols = np.meshgrid(
        (np.arange(height, dtype=np.float32) + 0.5) / height,
        (np.arange(width, dtype=np.float32) + 0.5) / width,
        indexing="ij",
    )
    return np.stack([rows, cols], axis=-1).reshape(-1, 2).astype(np.float32)


def _cache_path(path: Path, config: TpqsConfig, cache_dir: Path, view: str) -> Path:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "backend": config.embedding_backend,
        "model": config.model_id,
        "source": config.model_source,
        "image_size": config.image_size,
        "view": view,
        "version": 5,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npz"


def _save_feature(path: Path, feature: DenseImageFeature) -> None:
    np.savez_compressed(
        path,
        patches=feature.patches,
        coordinates=feature.coordinates,
        foreground=feature.foreground,
    )


def _load_feature(path: Path) -> DenseImageFeature:
    loaded = np.load(path)
    return DenseImageFeature(
        patches=loaded["patches"].astype(np.float32),
        coordinates=loaded["coordinates"].astype(np.float32),
        foreground=loaded["foreground"].astype(bool),
    )
