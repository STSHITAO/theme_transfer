from __future__ import annotations

from pathlib import Path
import os

import numpy as np
from PIL import Image, ImageDraw


def write_embedding_pca(
    output_path: Path,
    labels: list[str],
    groups: list[str],
    embeddings: list[np.ndarray],
) -> Path:
    matrix = np.vstack(embeddings).astype(np.float32)
    coords = _pca_2d(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if os.environ.get("TPQS_PCA_RENDERER", "pil").lower() != "matplotlib":
        return _write_pca_fallback_png(output_path, labels, groups, coords)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _write_pca_fallback_png(output_path, labels, groups, coords)

    colors = {"theme": "#d95f02", "generated": "#1b9e77", "target": "#7570b3"}

    plt.figure(figsize=(7, 5), dpi=140)
    for group in sorted(set(groups)):
        indexes = [index for index, value in enumerate(groups) if value == group]
        plt.scatter(
            coords[indexes, 0],
            coords[indexes, 1],
            label=group,
            s=36,
            color=colors.get(group, "#333333"),
            alpha=0.85,
        )
    for label, (x, y) in zip(labels, coords):
        plt.text(float(x), float(y), label, fontsize=7)
    plt.title("TPQS embedding PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords), dtype=np.float32)])
    return coords.astype(np.float32)


def _write_pca_fallback_png(output_path: Path, labels: list[str], groups: list[str], coords: np.ndarray) -> Path:
    colors = {"theme": (217, 95, 2), "generated": (27, 158, 119), "target": (117, 112, 179)}
    width, height = 980, 700
    margin = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 24), "TPQS embedding PCA", fill=(30, 30, 30))

    x = coords[:, 0]
    y = coords[:, 1]
    x_span = float(max(x.max() - x.min(), 1e-6))
    y_span = float(max(y.max() - y.min(), 1e-6))
    for label, group, (px, py) in zip(labels, groups, coords):
        sx = margin + int((float(px) - float(x.min())) / x_span * (width - margin * 2))
        sy = height - margin - int((float(py) - float(y.min())) / y_span * (height - margin * 2))
        color = colors.get(group, (50, 50, 50))
        draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=color)
        draw.text((sx + 7, sy - 7), label, fill=(40, 40, 40))

    image.save(output_path)
    return output_path
