from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


NEUTRAL_BACKGROUND = (238, 238, 238, 255)


def load_image_view(path: Path, view: str, image_size: int) -> Image.Image:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    rgba = _crop_launcher_screenshot(rgba)
    if view == "structure":
        rgba = _crop_to_foreground(rgba)
    elif view != "appearance":
        raise ValueError(f"Unsupported ITTE image view: {view}")
    canvas = Image.new("RGBA", rgba.size, NEUTRAL_BACKGROUND)
    canvas.alpha_composite(rgba)
    rgb = canvas.convert("RGB")
    return ImageOps.pad(
        rgb,
        (image_size, image_size),
        method=Image.Resampling.BICUBIC,
        color=NEUTRAL_BACKGROUND[:3],
        centering=(0.5, 0.5),
    )


def foreground_mask(image: Image.Image, grid_h: int, grid_w: int) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB").resize((grid_w, grid_h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    corners = np.vstack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    background = np.median(corners, axis=0)
    mask = np.linalg.norm(rgb - background, axis=2) > 0.08
    if int(mask.sum()) < 4:
        mask[:] = True

    detail_size = 8
    detailed = np.asarray(
        image.convert("RGB").resize((grid_w * detail_size, grid_h * detail_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    salience = np.zeros((grid_h, grid_w), dtype=np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            block = detailed[
                row * detail_size : (row + 1) * detail_size,
                col * detail_size : (col + 1) * detail_size,
            ]
            gray = block.mean(axis=2)
            gx = np.abs(np.diff(gray, axis=1)).mean()
            gy = np.abs(np.diff(gray, axis=0)).mean()
            salience[row, col] = float(gray.std() + gx + gy)
    threshold = float(np.quantile(salience[mask], 0.35)) if mask.any() else 0.0
    selected = mask & (salience >= threshold)
    if int(selected.sum()) < 4:
        selected = mask
    return selected.reshape(-1)


def _crop_launcher_screenshot(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width < 64 or height < 64:
        return image
    rgb = image.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    saturation = hsv[:, :, 1]
    bottom_color = float(np.median(saturation[104:]))
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2) / 255.0
    bottom_edges = float(np.mean(np.abs(np.diff(gray[96:], axis=1)) > 0.03))
    screenshot_like = bottom_color < 0.22 and bottom_edges > 0.05
    if not screenshot_like:
        return image

    crop_size = max(32, min(width, int(round(height * 0.72))))
    left = max(0, (width - crop_size) // 2)
    top = 0
    return image.crop((left, top, min(left + crop_size, width), min(top + crop_size, height)))


def _crop_to_foreground(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image, dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    if float((alpha < 0.99).mean()) > 0.01:
        mask = alpha > 0.05
    else:
        height, width = rgb.shape[:2]
        border = max(2, min(height, width) // 24)
        samples = np.vstack(
            [
                rgb[:border].reshape(-1, 3),
                rgb[-border:].reshape(-1, 3),
                rgb[:, :border].reshape(-1, 3),
                rgb[:, -border:].reshape(-1, 3),
            ]
        )
        background = np.median(samples, axis=0)
        mask = np.linalg.norm(rgb - background, axis=2) > 0.09
    if int(mask.sum()) < 16:
        return image
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    padding = max(2, int(round(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.08)))
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(width, int(xs.max()) + 1 + padding)
    bottom = min(height, int(ys.max()) + 1 + padding)
    if right - left < width * 0.20 or bottom - top < height * 0.20:
        return image
    return image.crop((left, top, right, bottom))
