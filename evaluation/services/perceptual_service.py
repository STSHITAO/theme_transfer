from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from evaluation.services.image_view_service import load_image_view


def compute_perceptual_scores(
    reference_raw_paths: list[Path],
    theme_paths: list[Path],
    target_paths: list[Path],
    generated_paths: list[Path],
    app_names: list[str],
    root_dir: Path,
    device: str,
    enabled: bool,
    image_size: int = 224,
) -> dict:
    if not enabled:
        return _unavailable("Perceptual backends disabled by ITTE_USE_PERCEPTUAL=false.")

    torch_home = root_dir / "models" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    try:
        import torch
        import lpips
        from DISTS_pytorch import DISTS
    except Exception as exc:
        return _unavailable(f"Perceptual dependencies unavailable: {type(exc).__name__}: {exc}")

    actual_device = device
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        actual_device = "cpu"

    unique_paths = list(dict.fromkeys([*reference_raw_paths, *theme_paths, *target_paths, *generated_paths]))
    appearance_tensors = {
        str(path): _load_tensor(path, image_size, torch, "appearance")
        for path in unique_paths
    }
    structure_tensors = {
        str(path): _load_tensor(path, image_size, torch, "structure")
        for path in unique_paths
    }
    try:
        dists_model = DISTS(load_weights=True).to(actual_device).eval()
        style_pairs = _style_pairs(theme_paths, target_paths, generated_paths)
        dists_style_distances = _compute_dists_pair_components(
            dists_model,
            style_pairs,
            appearance_tensors,
            actual_device,
            torch,
        )
        identity_pairs = _identity_pairs(reference_raw_paths, theme_paths, target_paths, generated_paths)
        dists_identity_distances = _compute_dists_pair_components(
            dists_model,
            identity_pairs,
            structure_tensors,
            actual_device,
            torch,
        )

        lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(actual_device).eval()
        lpips_distances = _compute_lpips_pairs(
            lpips_model,
            identity_pairs,
            structure_tensors,
            actual_device,
            torch,
        )
    except Exception as exc:
        return _unavailable(f"Perceptual model execution failed: {type(exc).__name__}: {exc}")

    dists_texture = _style_component_score(
        theme_paths,
        target_paths,
        generated_paths,
        dists_style_distances,
        component="texture",
    )
    dists_structure = _identity_component_score(
        reference_raw_paths,
        theme_paths,
        target_paths,
        generated_paths,
        app_names,
        dists_identity_distances,
        component="structure",
    )
    lpips_content = _identity_component_score(
        reference_raw_paths,
        theme_paths,
        target_paths,
        generated_paths,
        app_names,
        lpips_distances,
        component="distance",
    )
    return {
        "available": True,
        "device": actual_device,
        "dists_texture": dists_texture,
        "dists_structure": dists_structure,
        "lpips_content": lpips_content,
        "models": {
            "dists": "DISTS-pytorch-0.1",
            "lpips": "LPIPS-vgg-0.1",
        },
        "reason": "DISTS and LPIPS computed from deterministic pretrained perceptual networks.",
    }


def _style_pairs(
    theme_paths: list[Path],
    target_paths: list[Path],
    generated_paths: list[Path],
) -> list[tuple[Path, Path]]:
    pairs = []
    pairs.extend(_upper_pairs(theme_paths))
    pairs.extend((item, ref) for item in target_paths for ref in theme_paths)
    pairs.extend((item, ref) for item in generated_paths for ref in theme_paths)
    return list(dict.fromkeys(_canonical_pair(left, right) for left, right in pairs))


def _identity_pairs(
    reference_raw_paths: list[Path],
    theme_paths: list[Path],
    target_paths: list[Path],
    generated_paths: list[Path],
) -> list[tuple[Path, Path]]:
    pairs = [*zip(reference_raw_paths, theme_paths), *zip(target_paths, generated_paths), *_upper_pairs(target_paths)]
    return list(dict.fromkeys(_canonical_pair(left, right) for left, right in pairs))


def _style_component_score(
    theme_paths: list[Path],
    target_paths: list[Path],
    generated_paths: list[Path],
    distances: dict[tuple[str, str], dict[str, float]],
    component: str,
) -> dict:
    rr = [_distance(distances, left, right, component) for left, right in _upper_pairs(theme_paths)]
    tr = [_distance(distances, target, ref, component) for target in target_paths for ref in theme_paths]
    gr = [_distance(distances, generated, ref, component) for generated in generated_paths for ref in theme_paths]
    d_rr = float(np.median(rr)) if rr else 0.0
    d_tr = float(np.mean(tr)) if tr else 0.0
    d_gr = float(np.mean(gr)) if gr else 0.0
    score = _relative_transfer(d_tr, d_gr, d_rr)
    return {
        "score": score,
        "reliable": len(theme_paths) >= 2 and d_tr > d_rr + 1e-8,
        "D_RR": d_rr,
        "D_TR": d_tr,
        "D_GR": d_gr,
    }


def _identity_component_score(
    reference_raw_paths: list[Path],
    theme_paths: list[Path],
    target_paths: list[Path],
    generated_paths: list[Path],
    app_names: list[str],
    distances: dict[tuple[str, str], dict[str, float]],
    component: str,
) -> dict:
    positive = [
        _distance(distances, raw, styled, component)
        for raw, styled in zip(reference_raw_paths, theme_paths)
    ]
    distractor = [
        _distance(distances, left, right, component)
        for left, right in _upper_pairs(target_paths)
    ]
    positive_baseline = float(np.median(positive)) if positive else 0.0
    distractor_baseline = float(np.median(distractor)) if distractor else positive_baseline + 0.25
    if distractor_baseline <= positive_baseline + 1e-6:
        distractor_baseline = positive_baseline + max(float(np.std(positive)) * 3.0, 0.10)

    per_app = []
    for app, target, generated in zip(app_names, target_paths, generated_paths):
        value = _distance(distances, target, generated, component)
        score = float(
            np.clip(
                (distractor_baseline - value) / max(distractor_baseline - positive_baseline, 1e-8),
                0.0,
                1.0,
            )
            * 100.0
        )
        per_app.append({"app": app, "score": score, "distance": value})
    scores = [item["score"] for item in per_app]
    return {
        "score": float(np.mean(scores)) if scores else 0.0,
        "p10_score": float(np.percentile(scores, 10)) if scores else 0.0,
        "positive_reference_distance": positive_baseline,
        "distractor_distance": distractor_baseline,
        "per_app": per_app,
        "reliable": bool(positive and distractor),
    }


def _compute_dists_pair_components(model, pairs, tensors, device, torch, batch_size: int = 8):
    output: dict[tuple[str, str], dict[str, float]] = {}
    alpha_sum = model.alpha.sum().detach()
    beta_sum = model.beta.sum().detach()
    alpha = torch.split(model.alpha / torch.clamp(alpha_sum, min=1e-8), model.chns, dim=1)
    beta = torch.split(model.beta / torch.clamp(beta_sum, min=1e-8), model.chns, dim=1)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        left = torch.cat([tensors[str(item[0])] for item in batch]).to(device)
        right = torch.cat([tensors[str(item[1])] for item in batch]).to(device)
        with torch.inference_mode():
            left_features = model.forward_once(left)
            right_features = model.forward_once(right)
            texture_similarity = torch.zeros((len(batch),), device=device)
            structure_similarity = torch.zeros((len(batch),), device=device)
            for index, (left_feature, right_feature) in enumerate(zip(left_features, right_features)):
                left_mean = left_feature.mean((2, 3), keepdim=True)
                right_mean = right_feature.mean((2, 3), keepdim=True)
                mean_similarity = (2 * left_mean * right_mean + 1e-6) / (
                    left_mean.square() + right_mean.square() + 1e-6
                )
                texture_similarity += (alpha[index] * mean_similarity).sum((1, 2, 3))

                left_var = (left_feature - left_mean).square().mean((2, 3), keepdim=True)
                right_var = (right_feature - right_mean).square().mean((2, 3), keepdim=True)
                covariance = (left_feature * right_feature).mean((2, 3), keepdim=True) - left_mean * right_mean
                covariance_similarity = (2 * covariance + 1e-6) / (left_var + right_var + 1e-6)
                structure_similarity += (beta[index] * covariance_similarity).sum((1, 2, 3))
        texture = (1.0 - texture_similarity).detach().cpu().numpy()
        structure = (1.0 - structure_similarity).detach().cpu().numpy()
        for pair, texture_value, structure_value in zip(batch, texture, structure):
            output[_pair_key(*pair)] = {
                "texture": float(np.clip(texture_value, 0.0, 2.0)),
                "structure": float(np.clip(structure_value, 0.0, 2.0)),
            }
    return output


def _compute_lpips_pairs(model, pairs, tensors, device, torch, batch_size: int = 12):
    output: dict[tuple[str, str], dict[str, float]] = {}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        left = torch.cat([tensors[str(item[0])] for item in batch]).to(device)
        right = torch.cat([tensors[str(item[1])] for item in batch]).to(device)
        with torch.inference_mode():
            values = model(left, right, normalize=True).reshape(-1).detach().cpu().numpy()
        for pair, value in zip(batch, values):
            output[_pair_key(*pair)] = {"distance": float(max(value, 0.0))}
    return output


def _load_tensor(path: Path, image_size: int, torch, view: str):
    rgb = load_image_view(path, view, image_size)
    arr = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _relative_transfer(d_tr: float, d_gr: float, d_rr: float) -> float:
    denominator = d_tr - d_rr
    if denominator <= 1e-8:
        return 0.0
    return float(np.clip((d_tr - d_gr) / denominator, 0.0, 1.0) * 100.0)


def _upper_pairs(paths: list[Path]) -> list[tuple[Path, Path]]:
    return [(paths[left], paths[right]) for left in range(len(paths)) for right in range(left + 1, len(paths))]


def _canonical_pair(left: Path, right: Path) -> tuple[Path, Path]:
    return (left, right) if str(left) <= str(right) else (right, left)


def _pair_key(left: Path, right: Path) -> tuple[str, str]:
    first, second = _canonical_pair(left, right)
    return str(first), str(second)


def _distance(distances, left: Path, right: Path, component: str) -> float:
    return float(distances[_pair_key(left, right)][component])


def _unavailable(reason: str) -> dict:
    empty = {"score": None, "reliable": False, "reason": reason, "per_app": []}
    return {
        "available": False,
        "device": None,
        "dists_texture": dict(empty),
        "dists_structure": dict(empty),
        "lpips_content": dict(empty),
        "models": {},
        "reason": reason,
    }
