from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.services.visual_stats_service import stats_embedding


DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@dataclass(frozen=True)
class TpqsConfig:
    embedding_backend: str = "dinov3"
    model_source: str = "modelscope"
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    pooling: str = "cls"
    image_size: int = 224
    batch_size: int = 1
    style_feature_backend: str = "color_edge_composition"

    @property
    def is_official_tpqs(self) -> bool:
        return self.embedding_backend == "dinov3"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "TpqsConfig":
        source = env if env is not None else os.environ
        return cls(
            embedding_backend=source.get("TPQS_EMBEDDING_BACKEND", "dinov3").lower(),
            model_source=source.get("TPQS_MODEL_SOURCE", "modelscope").lower(),
            model_id=source.get("TPQS_MODEL_ID", DEFAULT_MODEL_ID),
            device=source.get("TPQS_DEVICE", "cpu"),
            pooling=source.get("TPQS_POOLING", "cls").lower(),
            image_size=int(source.get("TPQS_IMAGE_SIZE", "224")),
            batch_size=int(source.get("TPQS_BATCH_SIZE", "1")),
            style_feature_backend=source.get("TPQS_STYLE_FEATURE_BACKEND", "color_edge_composition").lower(),
        )


@dataclass(frozen=True)
class ModelCacheDirs:
    hf_home: Path
    hf_hub_cache: Path
    modelscope_cache: Path
    torch_home: Path


def prepare_model_cache_dirs(root_dir: Path | None = None) -> ModelCacheDirs:
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
    hf_home = root / "models" / "huggingface"
    hf_hub_cache = hf_home / "hub"
    modelscope_cache = root / "models" / "modelscope"
    torch_home = root / "models" / "torch"
    for path in [hf_home, hf_hub_cache, modelscope_cache, torch_home]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["MODELSCOPE_CACHE"] = str(modelscope_cache)
    os.environ["TORCH_HOME"] = str(torch_home)
    return ModelCacheDirs(
        hf_home=hf_home,
        hf_hub_cache=hf_hub_cache,
        modelscope_cache=modelscope_cache,
        torch_home=torch_home,
    )


def embed_images(paths: list[Path], config: TpqsConfig, root_dir: Path | None = None) -> dict[str, np.ndarray]:
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
    cache_dir = root / "data" / "evaluations" / "_cache" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if config.embedding_backend == "stats":
        return {
            str(path): _cached_embedding(path, config, cache_dir, lambda item: stats_embedding(item, config.image_size))
            for path in paths
        }
    if config.embedding_backend == "dinov3":
        return _embed_dinov3(paths, config, root, cache_dir)
    raise ValueError(f"Unsupported TPQS embedding backend: {config.embedding_backend}")


def _embed_dinov3(paths: list[Path], config: TpqsConfig, root: Path, cache_dir: Path) -> dict[str, np.ndarray]:
    cache_dirs = prepare_model_cache_dirs(root)

    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model_path = _resolve_dinov3_model_path(config, cache_dirs)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    embeddings: dict[str, np.ndarray] = {}
    uncached = []
    for path in paths:
        cache_path = _embedding_cache_path(path, config, cache_dir)
        if cache_path.exists():
            embeddings[str(path)] = np.load(cache_path)
        else:
            uncached.append(path)

    for start in range(0, len(uncached), max(config.batch_size, 1)):
        batch_paths = uncached[start : start + max(config.batch_size, 1)]
        images = [_load_rgb(path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        if config.pooling == "mean":
            tensor = outputs.last_hidden_state.mean(dim=1)
        else:
            tensor = outputs.last_hidden_state[:, 0]
        batch_embeddings = tensor.detach().cpu().numpy().astype(np.float32)
        for path, embedding in zip(batch_paths, batch_embeddings):
            embedding = _l2_normalize_vector(embedding)
            np.save(_embedding_cache_path(path, config, cache_dir), embedding)
            embeddings[str(path)] = embedding

    return embeddings


def _cached_embedding(path: Path, config: TpqsConfig, cache_dir: Path, builder) -> np.ndarray:
    cache_path = _embedding_cache_path(path, config, cache_dir)
    if cache_path.exists():
        return np.load(cache_path)
    vector = _l2_normalize_vector(builder(path).astype(np.float32))
    np.save(cache_path, vector)
    return vector


def _embedding_cache_path(path: Path, config: TpqsConfig, cache_dir: Path) -> Path:
    stat = path.stat()
    payload = {
        "abs_path": str(path.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "backend": config.embedding_backend,
        "model_source": config.model_source,
        "model_id": config.model_id,
        "image_size": config.image_size,
        "pooling": config.pooling,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _resolve_dinov3_model_path(config: TpqsConfig, cache_dirs: ModelCacheDirs) -> str:
    if config.model_source == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "TPQS_MODEL_SOURCE=modelscope requires the `modelscope` package. "
                "Install it in the SEG environment with `pip install modelscope`."
            ) from exc
        return snapshot_download(config.model_id, cache_dir=str(cache_dirs.modelscope_cache))

    if config.model_source == "huggingface":
        try:
            from transformers import AutoConfig

            AutoConfig.from_pretrained(config.model_id, cache_dir=str(cache_dirs.hf_hub_cache))
        except Exception as exc:
            message = str(exc).lower()
            if "gated repo" in message or "401" in message or "403" in message:
                raise RuntimeError(
                    "Cannot download DINOv3 because the Hugging Face repo is gated. "
                    "Grant access to the model and authenticate in the SEG environment "
                    "with HF_TOKEN or `hf auth login`. "
                    f"Model: {config.model_id}. Cache directory: {cache_dirs.hf_hub_cache}"
                ) from None
            raise
        return config.model_id

    raise ValueError(f"Unsupported TPQS model source: {config.model_source}")


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm
