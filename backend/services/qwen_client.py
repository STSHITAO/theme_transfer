import base64
import io
import json
import mimetypes
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image


REQUIRED_ANALYSIS_FIELDS = [
    "theme_style_analysis",
    "common_background_transform",
    "common_foreground_transform",
    "color_palette",
    "line_style",
    "texture_material",
    "lighting_shadow",
    "icon_composition_rules",
    "target_preservation",
    "generation_prompt",
    "negative_prompt",
    "qc_focus",
    "used_reference_examples",
]

REQUIRED_TARGET_IDENTITY_FIELDS = [
    "app",
    "identity_anchors",
    "must_preserve",
    "can_restyle",
    "must_not_replace_with",
    "color_preservation",
]

REQUIRED_TRANSFER_PLAN_FIELDS = [
    "app",
    "preserve",
    "must_preserve",
    "restyle",
    "decorate",
    "forbid",
    "generation_brief",
]


def analyze_theme(reference_examples, target_inputs, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_theme_analysis.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_analysis(reference_examples)

    prompt = (root / "prompts" / "qwen_theme_analysis.md").read_text(encoding="utf-8")
    content = [{"text": prompt}]
    for index, example in enumerate(reference_examples, start=1):
        content.append({"text": f"参考样例 {index}: {example['app_name']} background。只用于分析背景层变化。"})
        content.append({"image": _image_data_url(example["background_path"])})
        content.append({"text": f"参考样例 {index}: {example['app_name']} foreground。只用于分析主体层变化。"})
        content.append({"image": _image_data_url(example["foreground_path"])})
        content.append({"text": f"参考样例 {index}: {example['app_name']} style_ref。用于和 background/foreground 比较提取主题风格，严禁复制该参考 App 主体。"})
        content.append({"image": _image_data_url(example["style_ref_path"])})

    if isinstance(target_inputs, dict):
        content.append({"text": "目标 App target_background。目标身份只能来自目标图，不能来自任何参考 App。"})
        content.append({"image": _image_data_url(target_inputs["target_background"])})
        content.append({"text": "目标 App target_foreground。生成时必须保留这个目标主体身份和核心轮廓。"})
        content.append({"image": _image_data_url(target_inputs["target_foreground"])})
    else:
        content.append({"text": "目标 App 原始图。目标身份只能来自这张图。"})
        content.append({"image": _image_data_url(target_inputs)})

    text = _call_qwen(content)
    return _parse_analysis_json(text, reference_examples)


def analyze_theme_package(reference_examples, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_theme_analysis.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_analysis(reference_examples)

    prompt = (root / "prompts" / "qwen_theme_analysis.md").read_text(encoding="utf-8")
    content = [
        {
            "text": (
                f"{prompt}\n\n"
                "本次是批量整包主题分析。不要绑定任何单一目标 App，"
                "只总结所有参考样例共同的主题迁移规则，供整包所有目标 App 复用。"
            )
        }
    ]
    for index, example in enumerate(reference_examples, start=1):
        content.append({"text": f"参考样例 {index}: {example['app_name']} background。只用于分析背景层变化。"})
        content.append({"image": _image_data_url(example["background_path"])})
        content.append({"text": f"参考样例 {index}: {example['app_name']} foreground。只用于分析主体层变化。"})
        content.append({"image": _image_data_url(example["foreground_path"])})
        content.append({"text": f"参考样例 {index}: {example['app_name']} style_ref。用于比较提取整包主题风格，严禁复制该参考 App 主体。"})
        content.append({"image": _image_data_url(example["style_ref_path"])})

    text = _call_qwen(content)
    return _parse_analysis_json(text, reference_examples)


def analyze_target_identity(target_app, target_image, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_target_identity.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_target_identity(target_app)

    prompt = (root / "prompts" / "qwen_target_identity.md").read_text(encoding="utf-8")
    content = [
        {"text": f"{prompt}\n\n当前目标 App: {target_app}"},
        {"image": _image_data_url(target_image)},
    ]
    text = _call_qwen(content)
    return _parse_target_identity_json(text, target_app)


def build_transfer_plan(theme_rules, target_identity, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_transfer_plan.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_transfer_plan(theme_rules, target_identity)

    prompt = (root / "prompts" / "qwen_transfer_plan.md").read_text(encoding="utf-8")
    content = [
        {
            "text": (
                f"{prompt}\n\n"
                "【theme_rules】\n"
                f"{json.dumps(theme_rules, ensure_ascii=False, indent=2)}\n\n"
                "【target_identity】\n"
                f"{json.dumps(target_identity, ensure_ascii=False, indent=2)}"
            )
        }
    ]
    text = _call_qwen(content)
    return _parse_transfer_plan_json(text, target_identity)


def score_candidates(theme_style_refs, target_layout, candidate_paths, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_qc.md").read_text(encoding="utf-8")

    if _mock_mode():
        return {
            "candidates": [
                {
                    "file": path,
                    "style_score": 85,
                    "target_identity_score": 85,
                    "background_score": 85,
                    "composition_score": 85,
                    "artifact_score": 85,
                    "overall_score": 85,
                    "failure_reason": "",
                    "recommendation": "mock 通过",
                }
                for path in candidate_paths
            ],
            "best_candidate": candidate_paths[0] if candidate_paths else "",
            "warning": "MOCK_MODE=true，使用本地 mock 质检结果。",
        }

    prompt = (root / "prompts" / "qwen_qc.md").read_text(encoding="utf-8")
    content = [{"text": prompt}]
    for path in theme_style_refs:
        content.append({"text": f"主题参考图: {Path(path).name}"})
        content.append({"image": _image_data_url(path, max_size=(512, 512), quality=72)})
    content.append({"text": "目标原始合成图 target_layout"})
    content.append({"image": _image_data_url(target_layout, max_size=(512, 512), quality=72)})
    for path in candidate_paths:
        content.append({"text": f"候选图: {Path(path).name}"})
        content.append({"image": _image_data_url(path, max_size=(512, 512), quality=72)})

    text = _call_qwen(content)
    return _parse_qc_json(text, candidate_paths)


def score_package_consistency(theme_style_refs, contact_sheet, final_outputs, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_package_qc.md").read_text(encoding="utf-8")
    app_names = _final_output_app_names(final_outputs)

    if _mock_mode():
        return {
            "package_consistency_score": 8,
            "style_consistency_score": 8,
            "target_identity_score": 8,
            "problematic_apps": [],
            "accepted_apps": app_names,
            "retry_apps": [],
            "overall_comment": "MOCK_MODE=true，整包风格一致性 QC 使用本地 mock 报告。",
        }

    prompt = (root / "prompts" / "qwen_package_qc.md").read_text(encoding="utf-8")
    content = [{"text": prompt}]
    for path in theme_style_refs:
        content.append({"text": f"主题参考图: {Path(path).name}"})
        content.append({"image": _image_data_url(path, max_size=(512, 512), quality=72)})
    content.append({"text": "整包 contact_sheet，用于检查所有输出是否像同一套主题。"})
    content.append({"image": _image_data_url(contact_sheet, max_size=(1024, 1024), quality=78)})
    for app_name, path in _final_output_items(final_outputs):
        content.append({"text": f"最终输出图: {app_name}"})
        content.append({"image": _image_data_url(path, max_size=(512, 512), quality=72)})

    text = _call_qwen(content)
    return _parse_package_qc_json(text, app_names)


def _load_env(root):
    load_dotenv(root / ".env")


def _mock_mode():
    return os.getenv("MOCK_MODE", "false").lower() == "true"


def _mock_analysis(reference_examples):
    used = [example["app_name"] for example in reference_examples]
    return {
        "theme_style_analysis": "mock: 从多个参考 App 中归纳统一的柔和、干净、主题化图标绘制规律。",
        "common_background_transform": "mock: 保持背景结构，统一色彩和质感。",
        "common_foreground_transform": "mock: 保留主体轮廓，应用统一线条和材质。",
        "color_palette": ["mock red", "mock cream", "mock dark accent"],
        "line_style": "mock: 圆润清晰的边缘线。",
        "texture_material": "mock: 轻微纸感和柔和渐变。",
        "lighting_shadow": "mock: 统一左上光源和轻阴影。",
        "icon_composition_rules": "mock: 保持中心主体和原始背景层级。",
        "target_preservation": "mock: 保留小红书主体身份、核心符号、轮廓和整体构图。",
        "generation_prompt": "mock: 将目标图标重绘为 theme_001 同款主题包风格。",
        "negative_prompt": "文字、水印、额外 logo、额外物体、乱码、过度变形",
        "qc_focus": "mock: 检查主题一致性、目标身份、构图稳定性和伪影。",
        "used_reference_examples": used,
    }


def _mock_target_identity(target_app):
    return {
        "app": target_app,
        "identity_anchors": [f"{target_app} original silhouette", "primary brand color tendency"],
        "must_preserve": ["core logo geometry", "recognizable subject structure", "key brand color tendency"],
        "can_restyle": ["outline", "surface material", "shadow", "background finish", "small theme decoration"],
        "must_not_replace_with": ["generic plush ball", "unrelated animal", "unrecognizable cute blob"],
        "color_preservation": "Preserve or softly adapt the target app's key recognisable colors when they are part of identity.",
    }


def _mock_transfer_plan(theme_rules, target_identity):
    app = target_identity.get("app", "")
    preserve = target_identity.get("must_preserve", [])
    forbid = target_identity.get("must_not_replace_with", [])
    return {
        "app": app,
        "preserve": preserve,
        "must_preserve": preserve,
        "restyle": [
            "apply the shared theme outline, texture, lighting, and background rules",
            "soften edges without changing the target logo skeleton",
        ],
        "decorate": ["add small theme decoration only if it does not cover identity anchors"],
        "forbid": forbid,
        "generation_brief": (
            f"Transform {app} into the shared theme while preserving its core structure and recognisable colors. "
            "Do not replace the subject with a generic mascot."
        ),
    }


def _call_qwen(content):
    base_url = os.getenv("ALI_PLAN_BASE_URL")
    model = os.getenv("ALI_PLAN_MODEL")
    api_key = os.getenv("ALI_PLAN_API_KEY")
    if not base_url or not model or not api_key:
        raise RuntimeError("Missing ALI_PLAN_BASE_URL, ALI_PLAN_MODEL, or ALI_PLAN_API_KEY")

    import dashscope
    from dashscope import MultiModalConversation

    dashscope.base_http_api_url = base_url
    last_error = None
    for _ in range(2):
        try:
            response = MultiModalConversation.call(
                model=model,
                api_key=api_key,
                messages=[{"role": "user", "content": content}],
                enable_thinking=False,
                timeout=120,
            )
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
    else:
        raise last_error
    status_code = _response_value(response, "status_code")
    if status_code and status_code != 200:
        raise RuntimeError(f"Qwen API failed: {_safe_response(response)}")
    text = _extract_response_text(response)
    if not text:
        raise RuntimeError(f"Qwen returned empty response: {_safe_response(response)}")
    return text


def _parse_analysis_json(text, reference_examples):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_analysis(reference_examples),
            "warning": "Qwen returned invalid JSON; fallback analysis was used.",
        }
    for key in REQUIRED_ANALYSIS_FIELDS:
        parsed.setdefault(key, _mock_analysis(reference_examples)[key])
    return parsed


def _parse_qc_json(text, candidate_paths):
    parsed = _parse_json(text)
    if isinstance(parsed, dict):
        parsed.setdefault("candidates", [])
        parsed.setdefault("best_candidate", candidate_paths[0] if candidate_paths else "")
        parsed.setdefault("warning", "")
        return parsed
    return {
        "raw_response": text,
        "candidates": [],
        "best_candidate": candidate_paths[0] if candidate_paths else "",
        "warning": "Qwen returned invalid JSON; fallback first candidate was used.",
    }


def _parse_target_identity_json(text, target_app):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_target_identity(target_app),
            "warning": "Qwen returned invalid JSON; fallback target identity was used.",
        }
    parsed.setdefault("app", target_app)
    fallback = _mock_target_identity(target_app)
    for key in REQUIRED_TARGET_IDENTITY_FIELDS:
        parsed.setdefault(key, fallback[key])
    return parsed


def _parse_transfer_plan_json(text, target_identity):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_transfer_plan({}, target_identity),
            "warning": "Qwen returned invalid JSON; fallback transfer plan was used.",
        }
    fallback = _mock_transfer_plan({}, target_identity)
    for key in REQUIRED_TRANSFER_PLAN_FIELDS:
        parsed.setdefault(key, fallback[key])
    return parsed


def _parse_package_qc_json(text, app_names):
    parsed = _parse_json(text)
    if isinstance(parsed, dict):
        parsed.setdefault("package_consistency_score", 0)
        parsed.setdefault("style_consistency_score", 0)
        parsed.setdefault("target_identity_score", 0)
        parsed.setdefault("problematic_apps", [])
        parsed.setdefault("accepted_apps", [])
        parsed.setdefault("retry_apps", [])
        parsed.setdefault("overall_comment", "")
        return parsed
    return {
        "raw_response": text,
        "package_consistency_score": 0,
        "style_consistency_score": 0,
        "target_identity_score": 0,
        "problematic_apps": [],
        "accepted_apps": [],
        "retry_apps": app_names,
        "overall_comment": "Qwen returned invalid JSON; fallback package QC report was used.",
    }


def _final_output_items(final_outputs):
    if isinstance(final_outputs, dict):
        return [(app_name, path) for app_name, path in final_outputs.items()]
    return [(Path(path).stem, path) for path in final_outputs]


def _final_output_app_names(final_outputs):
    return [app_name for app_name, _ in _final_output_items(final_outputs)]


def _parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _extract_response_text(response):
    if isinstance(response, dict):
        content = (
            response.get("output", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", [])
        )
    else:
        output = getattr(response, "output", {})
        content = output.get("choices", [{}])[0].get("message", {}).get("content", [])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return "\n".join(texts)
    return ""


def _safe_response(response):
    return str(response).replace(os.getenv("ALI_PLAN_API_KEY", ""), "***")


def _response_value(response, key):
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def _image_data_url(path, max_size=(768, 768), quality=82):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"
