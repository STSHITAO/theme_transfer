from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


STYLE_TEXT_FIELDS = [
    "style_eval_text",
    "theme_visual_language",
    "theme_board",
    "color_rule",
    "color_transform_rule",
    "background_style_rule",
    "background_transform_rule",
    "background_color_range",
    "background_texture_rule",
    "background_contrast_rule",
    "stroke_rule",
    "stroke_transform_rule",
    "composition_rule",
    "composition_transform_rule",
    "subject_scale_rule",
    "detail_complexity_rule",
    "theme_fidelity_constraints",
    "forbidden_style_drift",
    "shared_design_rules",
    "identity_handling_policy",
    "common_forbidden_failures",
    "material",
    "texture",
    "line_style",
    "palette",
]

OPENCLIP_TEXT_FIT_MARGIN = 0.08
OPENCLIP_RELIABLE_ANCHOR_GAP = 0.05

QWEN_STYLE_SECTION_HEADERS = [
    "Theme Style Analysis",
    "Reference Raw To Style Transform",
    "Color Transform Rule",
    "Background Transform Rule",
    "Stroke Transform Rule",
    "Composition Transform Rule",
    "Subject Scale Rule",
    "Detail Complexity Rule",
    "Theme Fidelity Constraints",
    "Theme Board",
    "Global Wan Constraints",
]


@dataclass
class FakeOpenClipBackend:
    image_scores: dict[str, float]

    def similarity(self, image_path: Path, text: str) -> float:
        return float(self.image_scores.get(image_path.name, self.image_scores.get(str(image_path), 0.0)))


class OpenClipBackend:
    def __init__(self, model_name: str, pretrained: str, device: str, root_dir: Path):
        _prepare_openclip_cache(root_dir)
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "TPQS_USE_OPENCLIP=true requires `open_clip_torch`. "
                "Install it in the SEG environment with `pip install open_clip_torch`."
            ) from exc

        self._torch = torch
        self._open_clip = open_clip
        self._device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            self._device = "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self._device,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    def similarity(self, image_path: Path, text: str) -> float:
        with Image.open(image_path) as image:
            image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        text_tensor = self._tokenizer([text]).to(self._device)
        with self._torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return float((image_features @ text_features.T).item())


def build_style_eval_text(theme_design_analysis: dict | None, qwen_instruction_text: str | None = None) -> str:
    if not theme_design_analysis:
        theme_design_analysis = {}
    direct = str(theme_design_analysis.get("style_eval_text", "")).strip()
    if direct:
        return direct

    fragments = []
    fragments.extend(_extract_qwen_style_sections(qwen_instruction_text or ""))
    for field in STYLE_TEXT_FIELDS:
        if field == "style_eval_text":
            continue
        value = theme_design_analysis.get(field)
        if isinstance(value, str) and value.strip():
            fragments.append(value.strip())
        elif isinstance(value, list):
            fragments.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, dict):
            fragments.extend(str(item).strip() for item in value.values() if str(item).strip())

    text = "; ".join(fragments)
    return _dedupe_text(text)[:1600]


def disabled_theme_style_text_fit() -> dict:
    return {
        "openclip_enabled": False,
        "score": None,
        "style_eval_text": "",
        "S_R": None,
        "S_T": None,
        "S_G": None,
        "reason": "TPQS_USE_OPENCLIP is false.",
    }


def compute_theme_style_text_fit(
    theme_ref_paths: list[Path],
    generated_paths: list[Path],
    target_paths: list[Path],
    theme_design_analysis: dict | None,
    backend,
    qwen_instruction_text: str | None = None,
) -> dict:
    style_eval_text = build_style_eval_text(theme_design_analysis, qwen_instruction_text)
    if not style_eval_text:
        return {
            "openclip_enabled": True,
            "score": None,
            "style_eval_text": "",
            "S_R": None,
            "S_T": None,
            "S_G": None,
            "reason": "theme_design_analysis has no usable style text fields.",
        }

    s_r = _mean_similarity(theme_ref_paths, style_eval_text, backend)
    s_t = _mean_similarity(target_paths, style_eval_text, backend)
    s_g = _mean_similarity(generated_paths, style_eval_text, backend)
    anchor_gap = s_r - s_t
    generated_lift = s_g - s_t
    denominator = max(anchor_gap, 0.0) + OPENCLIP_TEXT_FIT_MARGIN
    score = max(0.0, min(100.0, generated_lift / denominator * 100.0))
    return {
        "openclip_enabled": True,
        "score": score,
        "style_eval_text": style_eval_text,
        "S_R": s_r,
        "S_T": s_t,
        "S_G": s_g,
        "anchor_gap": anchor_gap,
        "generated_lift": generated_lift,
        "score_margin": OPENCLIP_TEXT_FIT_MARGIN,
        "text_anchor_reliable": anchor_gap >= OPENCLIP_RELIABLE_ANCHOR_GAP,
        "scoring_method": "relative_lift_with_margin",
        "reason": "",
    }


def _mean_similarity(paths: list[Path], text: str, backend) -> float:
    if not paths:
        return 0.0
    return float(np.mean([backend.similarity(path, text) for path in paths]))


def _extract_qwen_style_sections(text: str) -> list[str]:
    if not text.strip():
        return []
    sections: dict[str, list[str]] = {}
    current_header = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_header = line.strip("[]")
            sections.setdefault(current_header, [])
            continue
        if current_header in QWEN_STYLE_SECTION_HEADERS:
            sections.setdefault(current_header, []).append(line)
    fragments = []
    for header in QWEN_STYLE_SECTION_HEADERS:
        if sections.get(header):
            fragments.append(f"{header}: {' '.join(sections[header])}")
    return fragments


def _dedupe_text(text: str) -> str:
    seen = set()
    fragments = []
    for fragment in [item.strip() for item in text.split(";") if item.strip()]:
        key = fragment.lower()
        if key in seen:
            continue
        seen.add(key)
        fragments.append(fragment)
    return "; ".join(fragments)


def _prepare_openclip_cache(root_dir: Path) -> None:
    cache_dir = root_dir / "models" / "openclip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENCLIP_CACHE_DIR"] = str(cache_dir)
    os.environ["HF_HOME"] = str(root_dir / "models" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(root_dir / "models" / "huggingface" / "hub")
