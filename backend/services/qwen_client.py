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
    "strategy_type",
    "identity_constraint_level",
    "preserve",
    "must_preserve",
    "recompose_allowed",
    "restyle",
    "decorate",
    "forbid",
    "generation_brief",
]

REQUIRED_THEME_DESIGN_FIELDS = [
    "theme_board",
    "reference_transformation_patterns",
    "shared_design_rules",
    "identity_handling_policy",
    "common_forbidden_failures",
]

REQUIRED_IDENTITY_STRATEGY_FIELDS = [
    "app",
    "strategy_type",
    "identity_constraint_level",
    "design_rationale",
    "must_preserve",
    "can_recompose",
    "forbid",
    "generation_direction",
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


def analyze_theme_design(reference_examples, theme_profile, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_theme_design_analysis.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_theme_design(reference_examples, theme_profile)

    prompt = (root / "prompts" / "qwen_theme_design_analysis.md").read_text(encoding="utf-8")
    content = [
        {
            "text": (
                f"{prompt}\n\n"
                "【theme_profile】\n"
                f"{json.dumps(theme_profile, ensure_ascii=False, indent=2)}"
            )
        }
    ]
    examples = theme_profile.get("examples", {}) if isinstance(theme_profile, dict) else {}
    for index, example in enumerate(reference_examples, start=1):
        app_name = example["app_name"]
        app_profile = examples.get(app_name, {})
        content.append(
            {
                "text": (
                    f"参考样例 {index}: {app_name}\n"
                    f"{json.dumps(app_profile, ensure_ascii=False, indent=2)}\n"
                    "接下来的三张图依次是 background、foreground、style_ref，用于分析主题化设计方式。"
                )
            }
        )
        content.append({"image": _image_data_url(example["background_path"])})
        content.append({"image": _image_data_url(example["foreground_path"])})
        content.append({"image": _image_data_url(example["style_ref_path"])})

    text = _call_qwen(content)
    return _parse_theme_design_json(text, reference_examples, theme_profile)


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


def build_identity_strategy(theme_design_analysis, theme_rules, target_profile, target_image, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_identity_strategy.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_identity_strategy(theme_design_analysis, target_profile)

    prompt = (root / "prompts" / "qwen_identity_strategy.md").read_text(encoding="utf-8")
    content = [
        {
            "text": (
                f"{prompt}\n\n"
                "【theme_design_analysis】\n"
                f"{json.dumps(theme_design_analysis, ensure_ascii=False, indent=2)}\n\n"
                "【theme_rules】\n"
                f"{json.dumps(theme_rules, ensure_ascii=False, indent=2)}\n\n"
                "【target_profile】\n"
                f"{json.dumps(target_profile, ensure_ascii=False, indent=2)}"
            )
        },
        {"image": _image_data_url(target_image)},
    ]
    text = _call_qwen(content)
    return _parse_identity_strategy_json(text, target_profile)


def build_transfer_plan(
    theme_rules,
    target_identity,
    root_dir=None,
    theme_design_analysis=None,
    target_profile=None,
    identity_strategy=None,
):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_transfer_plan.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_transfer_plan(theme_rules, target_identity, identity_strategy=identity_strategy)

    prompt = (root / "prompts" / "qwen_transfer_plan.md").read_text(encoding="utf-8")
    content = [
        {
            "text": (
                f"{prompt}\n\n"
                "【theme_rules】\n"
                f"{json.dumps(theme_rules, ensure_ascii=False, indent=2)}\n\n"
                "【target_identity】\n"
                f"{json.dumps(target_identity, ensure_ascii=False, indent=2)}\n\n"
                "【theme_design_analysis】\n"
                f"{json.dumps(theme_design_analysis or {}, ensure_ascii=False, indent=2)}\n\n"
                "【target_profile】\n"
                f"{json.dumps(target_profile or {}, ensure_ascii=False, indent=2)}\n\n"
                "【identity_strategy】\n"
                f"{json.dumps(identity_strategy or {}, ensure_ascii=False, indent=2)}"
            )
        }
    ]
    text = _call_qwen(content)
    return _parse_transfer_plan_json(
        text,
        target_identity,
        identity_strategy=identity_strategy,
        target_profile=target_profile,
    )


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
                    "style_match_score": 85,
                    "target_identity_score": 85,
                    "target_recognition_score": 85,
                    "semantic_fit_score": 85,
                    "identity_constraint_score": 85,
                    "over_recompose_risk": 10,
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


def _mock_theme_design(reference_examples, theme_profile):
    used = [example["app_name"] for example in reference_examples]
    return {
        "theme_board": {
            "palette": "mock shared palette from style_ref images",
            "line_style": "mock shared line language",
            "material": "mock shared surface material",
            "background": "mock shared icon background treatment",
            "composition": "mock shared centered icon composition",
        },
        "reference_transformation_patterns": [
            {
                "app": app_name,
                "source_semantics": (theme_profile.get("examples", {}).get(app_name, {}).get("core_function", "")),
                "observed_transformation": "mock: infer how source semantics are represented in the transferred style_ref.",
                "preserved_identity": "mock: retain at least one recognizable brand or function cue.",
                "redesigned_parts": "mock: allow structure to be simplified or recomposed when theme language requires it.",
            }
            for app_name in used
        ],
        "shared_design_rules": ["use one shared theme_board for every target app"],
        "identity_handling_policy": "mock: choose identity expression dynamically from target image and neutral app semantics.",
        "common_forbidden_failures": [
            "do not turn every app into a generic decoration",
            "do not redefine theme style per app",
        ],
    }


def _mock_identity_strategy(theme_design_analysis, target_profile):
    app = target_profile.get("app", "")
    return {
        "app": app,
        "strategy_type": "semantic_recompose" if app in {"xiaohongshu", "damai", "tieba"} else "logo_simplify",
        "identity_constraint_level": "balanced",
        "design_rationale": "mock: choose an app expression from the target image, neutral app profile, and shared theme_board.",
        "must_preserve": ["recognizable app name or symbol cue", "core function cue"],
        "can_recompose": ["layout", "surface treatment", "supporting symbols"],
        "forbid": ["generic decoration without app recognition", "copying reference app identity", "redefining the shared theme"],
        "generation_direction": (
            f"Create a themed icon for {target_profile.get('display_name', app)}. "
            "Follow the shared theme_board and preserve enough app recognition."
        ),
    }


def _mock_transfer_plan(theme_rules, target_identity, identity_strategy=None):
    app = target_identity.get("app", "")
    preserve = target_identity.get("must_preserve", [])
    forbid = target_identity.get("must_not_replace_with", [])
    strategy = identity_strategy or {}
    if strategy:
        preserve = strategy.get("must_preserve", preserve)
        forbid = strategy.get("forbid", forbid)
    return {
        "app": app,
        "strategy_type": strategy.get("strategy_type", "logo_simplify"),
        "identity_constraint_level": strategy.get("identity_constraint_level", "balanced"),
        "preserve": preserve,
        "must_preserve": preserve,
        "recompose_allowed": strategy.get("can_recompose", []),
        "restyle": [
            "apply the shared theme outline, texture, lighting, and background rules",
            "soften edges without changing the target logo skeleton",
        ],
        "decorate": ["add small theme decoration only if it does not cover identity anchors"],
        "forbid": forbid,
        "generation_brief": strategy.get(
            "generation_direction",
            (
                f"Transform {app} into the shared theme while preserving its core structure and recognisable colors. "
                "Do not replace the subject with a generic mascot."
            ),
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
        for index, item in enumerate(parsed["candidates"]):
            if index < len(candidate_paths):
                item.setdefault("file", candidate_paths[index])
            item.setdefault("style_match_score", item.get("style_score", 0))
            item.setdefault("target_recognition_score", item.get("target_identity_score", 0))
            item.setdefault("semantic_fit_score", 0)
            item.setdefault("identity_constraint_score", item.get("target_identity_score", 0))
            item.setdefault("over_recompose_risk", 0)
            item.setdefault("artifact_score", 0)
            item.setdefault("overall_score", 0)
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


def _parse_theme_design_json(text, reference_examples, theme_profile):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_theme_design(reference_examples, theme_profile),
            "warning": "Qwen returned invalid JSON; fallback theme design analysis was used.",
        }
    fallback = _mock_theme_design(reference_examples, theme_profile)
    for key in REQUIRED_THEME_DESIGN_FIELDS:
        parsed.setdefault(key, fallback[key])
    return parsed


def _parse_identity_strategy_json(text, target_profile):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_identity_strategy({}, target_profile),
            "warning": "Qwen returned invalid JSON; fallback identity strategy was used.",
        }
    fallback = _mock_identity_strategy({}, target_profile)
    for key in REQUIRED_IDENTITY_STRATEGY_FIELDS:
        parsed.setdefault(key, fallback[key])
    if parsed.get("identity_constraint_level") not in ["strict", "balanced", "flexible"]:
        parsed["identity_constraint_level"] = "balanced"
    _protect_brand_identity_cues(parsed, target_profile, preserve_keys=["must_preserve"], recompose_keys=["can_recompose"])
    return parsed


def _parse_transfer_plan_json(text, target_identity, identity_strategy=None, target_profile=None):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_transfer_plan({}, target_identity, identity_strategy=identity_strategy),
            "warning": "Qwen returned invalid JSON; fallback transfer plan was used.",
        }
    fallback = _mock_transfer_plan({}, target_identity, identity_strategy=identity_strategy)
    for key in REQUIRED_TRANSFER_PLAN_FIELDS:
        parsed.setdefault(key, fallback[key])
    _protect_brand_identity_cues(
        parsed,
        target_profile or {},
        preserve_keys=["preserve", "must_preserve"],
        recompose_keys=["recompose_allowed"],
    )
    return parsed


def _protect_brand_identity_cues(data, target_profile, preserve_keys, recompose_keys):
    cues = []
    if isinstance(target_profile, dict):
        value = target_profile.get("brand_identity_cues", [])
        if isinstance(value, list):
            cues = [str(item) for item in value if str(item).strip()]
    if not cues:
        return data

    for key in preserve_keys:
        existing = data.get(key, [])
        if not isinstance(existing, list):
            existing = []
        data[key] = _dedupe_list([*existing, *cues])

    for key in recompose_keys:
        data[key] = _remove_conflicting_items(data.get(key, []), cues)
    data["forbid"] = _remove_conflicting_items(data.get("forbid", []), cues)
    return data


def _remove_conflicting_items(items, cues):
    if not isinstance(items, list):
        return []
    cue_terms = _brand_cue_terms(cues)
    cleaned = []
    for item in items:
        item_text = str(item)
        lower = item_text.lower()
        if any(term in lower for term in cue_terms):
            continue
        cleaned.append(item)
    return cleaned


def _brand_cue_terms(cues):
    terms = set()
    for cue in cues:
        lower = str(cue).lower()
        if lower:
            terms.add(lower)
        if "bilibili" in lower:
            terms.add("bilibili")
        if "文字" in lower or "wordmark" in lower or "text" in lower:
            terms.update(["wordmark", "bilibili text", "bilibili 文字"])
        if "小电视" in lower:
            terms.add("小电视")
        if "粉色圆角方形底" in lower:
            terms.add("pink rounded square background")
            terms.add("粉色圆角方形底")
    return [term for term in terms if term]


def _dedupe_list(items):
    result = []
    seen = set()
    for item in items:
        key = str(item)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


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
