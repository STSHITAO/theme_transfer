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
    "color_application",
    "stroke_application",
    "composition_application",
    "identity_application",
    "fidelity_constraints",
    "negative_constraints",
    "structure_preservation_mode",
    "structure_identity_metric_applicable",
    "structure_policy_rationale",
]

REQUIRED_THEME_DESIGN_FIELDS = [
    "theme_board",
    "color_transform_rule",
    "background_transform_rule",
    "stroke_transform_rule",
    "composition_transform_rule",
    "subject_scale_rule",
    "detail_complexity_rule",
    "theme_fidelity_constraints",
    "forbidden_style_drift",
    "reference_transformation_patterns",
    "shared_design_rules",
    "identity_handling_policy",
    "structure_preservation_policy",
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
    "identity_anchor",
    "brand_cues_to_preserve",
    "semantic_cues_to_preserve",
    "style_fidelity_priority",
    "structure_preservation_mode",
    "structure_identity_metric_applicable",
    "structure_policy_rationale",
]

STRUCTURE_PRESERVATION_MODES = {
    "preserve_major_structure",
    "semantic_recompose",
}

THEME_DESIGN_BATCH_SIZE = 5

def analyze_theme(reference_examples, target_inputs, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_theme_analysis.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_analysis(reference_examples)

    prompt = (root / "prompts" / "qwen_theme_analysis.md").read_text(encoding="utf-8")
    content = [{"text": prompt}]
    for index, example in enumerate(reference_examples, start=1):
        _append_original_style_ref_example(content, index, example)

    content.append({"text": "目标 App 原始图标。目标身份只能来自这张图片。"})
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
                "这是整包级主题分析。不要把规则绑定到任何单个目标 App。"
                "只总结所有参考样例共有、可供整套主题复用的迁移规则。"
            )
        }
    ]
    for index, example in enumerate(reference_examples, start=1):
        _append_original_style_ref_example(content, index, example)

    text = _call_qwen(content)
    return _parse_analysis_json(text, reference_examples)


def analyze_theme_design(reference_examples, theme_profile, root_dir=None, batch_size=THEME_DESIGN_BATCH_SIZE):
    root = Path(root_dir) if root_dir else Path.cwd()
    _load_env(root)
    (root / "prompts" / "qwen_theme_design_analysis.md").read_text(encoding="utf-8")

    if _mock_mode():
        return _mock_theme_design(reference_examples, theme_profile)

    if batch_size < 1:
        raise ValueError("Theme design analysis batch_size must be at least 1.")
    batches = [
        reference_examples[index:index + batch_size]
        for index in range(0, len(reference_examples), batch_size)
    ]
    if len(batches) == 1:
        return _analyze_theme_design_batch(batches[0], theme_profile, root)

    partials = [
        _analyze_theme_design_batch(batch, _theme_profile_for_examples(theme_profile, batch), root)
        for batch in batches
    ]
    return _aggregate_theme_design_analyses(partials, reference_examples, theme_profile, root, batch_size)


def _analyze_theme_design_batch(reference_examples, theme_profile, root):

    prompt = (root / "prompts" / "qwen_theme_design_analysis.md").read_text(encoding="utf-8")
    content = [
        {
            "text": (
                f"{prompt}\n\n"
                "【主题资料 theme_profile】\n"
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
                    f"参考样例 {index}：{app_name}\n"
                    f"{json.dumps(app_profile, ensure_ascii=False, indent=2)}\n"
                    "接下来的两张图片依次是原始图和 style_ref，用于分析主题重设计方法。"
                )
            }
        )
        content.append({"image": _image_data_url(example["original_path"])})
        content.append({"image": _image_data_url(example["style_ref_path"])})

    text = _call_qwen(content)
    return _parse_theme_design_json(text, reference_examples, theme_profile)


def _aggregate_theme_design_analyses(partials, reference_examples, theme_profile, root, batch_size):
    prompt = (root / "prompts" / "qwen_theme_design_aggregate.md").read_text(encoding="utf-8")
    compact_partials = []
    all_patterns = []
    for index, partial in enumerate(partials, start=1):
        patterns = partial.get("reference_transformation_patterns", [])
        if isinstance(patterns, list):
            all_patterns.extend(_compact_transformation_pattern(item) for item in patterns if isinstance(item, dict))
        compact_partials.append(
            {
                "batch": index,
                "theme_board": partial.get("theme_board", {}),
                "color_transform_rule": partial.get("color_transform_rule", ""),
                "background_transform_rule": partial.get("background_transform_rule", ""),
                "stroke_transform_rule": partial.get("stroke_transform_rule", ""),
                "composition_transform_rule": partial.get("composition_transform_rule", ""),
                "subject_scale_rule": partial.get("subject_scale_rule", ""),
                "detail_complexity_rule": partial.get("detail_complexity_rule", ""),
                "theme_fidelity_constraints": partial.get("theme_fidelity_constraints", []),
                "forbidden_style_drift": partial.get("forbidden_style_drift", []),
                "shared_design_rules": partial.get("shared_design_rules", []),
                "identity_handling_policy": partial.get("identity_handling_policy", ""),
                "structure_preservation_policy": partial.get("structure_preservation_policy", {}),
                "common_forbidden_failures": partial.get("common_forbidden_failures", []),
                "reference_pattern_summaries": [
                    _compact_transformation_pattern(item)
                    for item in patterns
                    if isinstance(item, dict)
                ],
            }
        )

    content = [{"text": f"{prompt}\n\n【批次分析 batch_analyses】\n{json.dumps(compact_partials, ensure_ascii=False)}"}]
    text = _call_qwen(content)
    merged = _parse_theme_design_json(text, reference_examples, theme_profile)
    merged["reference_transformation_patterns"] = _dedupe_patterns(all_patterns)
    merged["analysis_coverage"] = {
        "reference_pair_count": len(reference_examples),
        "batch_count": len(partials),
        "batch_size": batch_size,
        "analyzed_apps": [example["app_name"] for example in reference_examples],
    }
    return merged


def _theme_profile_for_examples(theme_profile, reference_examples):
    profile = dict(theme_profile) if isinstance(theme_profile, dict) else {}
    examples = theme_profile.get("examples", {}) if isinstance(theme_profile, dict) else {}
    profile["examples"] = {
        example["app_name"]: examples.get(example["app_name"], {})
        for example in reference_examples
    }
    return profile


def _compact_transformation_pattern(pattern, text_limit=240):
    compact = {}
    for key in (
        "app",
        "source_semantics",
        "observed_transformation",
        "preserved_identity",
        "redesigned_parts",
        "preserve_major_structure",
        "structure_evidence",
    ):
        value = pattern.get(key)
        if isinstance(value, str) and len(value) > text_limit:
            value = value[:text_limit].rstrip() + "…"
        compact[key] = value
    compact["preserve_major_structure"] = bool(compact.get("preserve_major_structure", True))
    return compact


def _dedupe_patterns(patterns):
    result = []
    seen = set()
    for pattern in patterns:
        app = str(pattern.get("app", "")).strip()
        key = app.casefold()
        if not app or key in seen:
            continue
        seen.add(key)
        result.append(pattern)
    return result

def _append_original_style_ref_example(content, index, example):
    app_name = example["app_name"]
    content.append({
        "text": (
            f"参考样例 {index}：{app_name} 原始图。"
            "它是参考 App 的原始图标，与 style_ref 配对后用于推断可迁移的主题重设计规则。"
        )
    })
    content.append({"image": _image_data_url(example["original_path"])})
    content.append({
        "text": (
            f"参考样例 {index}：{app_name} style_ref。"
            "它是同一参考 App 的主题化结果。只学习可迁移的设计语言，不要学习该 App 的身份。"
        )
    })
    content.append({"image": _image_data_url(example["style_ref_path"])})

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


def score_candidates(theme_style_refs, target_layout, candidate_paths, root_dir=None, transfer_plan=None):
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
    content = [
        {
            "text": (
                f"{prompt}\n\n"
                "【transfer_plan】\n"
                f"{json.dumps(transfer_plan or {}, ensure_ascii=False, indent=2)}"
            )
        }
    ]
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
    sampled_outputs = _sample_final_output_items(final_outputs, max_items=12)
    content.append(
        {
            "text": (
                f"整包共 {len(app_names)} 个 App。contact_sheet 覆盖全部 App；"
                f"以下额外提供 {len(sampled_outputs)} 个按排序均匀抽取的单图用于细节检查。"
            )
        }
    )
    for app_name, path in sampled_outputs:
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
        "color_palette": ["mock 红色", "mock 奶油色", "mock 深色点缀"],
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
        "identity_anchors": [f"{target_app} 原始轮廓", "主要品牌色倾向"],
        "must_preserve": ["核心 logo 几何结构", "可识别的主体结构", "关键品牌色倾向"],
        "can_restyle": ["轮廓线", "表面材质", "阴影", "背景质感", "小型主题装饰"],
        "must_not_replace_with": ["通用毛绒球", "无关动物", "无法识别的可爱团块"],
        "color_preservation": "当关键识别色属于目标 App 身份时，保留该颜色或仅作柔和调整。",
    }


def _mock_theme_design(reference_examples, theme_profile):
    used = [example["app_name"] for example in reference_examples]
    return {
        "theme_board": {
            "palette": "mock：来自 style_ref 图片的共享色板",
            "line_style": "mock：共享线条语言",
            "material": "mock：共享表面材质",
            "background": "mock：共享图标背景处理",
            "composition": "mock：共享的居中图标构图",
        },
        "color_transform_rule": "mock: move each original icon toward the shared theme_001 palette and saturation range.",
        "background_transform_rule": "mock: convert original backgrounds into the shared theme_001 base shape and finish.",
        "stroke_transform_rule": "mock: match theme_001 stroke weight, rounded edges, and edge density.",
        "composition_transform_rule": "mock: match theme_001 subject centering, whitespace, and background ratio.",
        "subject_scale_rule": "mock: keep subject scale within the theme_001 reference range.",
        "detail_complexity_rule": "mock: reduce or add detail until complexity matches theme_001 references.",
        "theme_fidelity_constraints": [
            "输出必须像 theme_001 中缺失的成员",
            "不要另行创造一套内部自洽的新主题",
        ],
        "forbidden_style_drift": [
            "使用 theme_001 之外的新色板",
            "描边风格与 theme_001 不匹配",
            "构图忽略 theme_001 的主体比例和留白",
        ],
        "reference_transformation_patterns": [
            {
                "app": app_name,
                "source_semantics": (theme_profile.get("examples", {}).get(app_name, {}).get("core_function", "")),
                "observed_transformation": "mock: infer how source semantics are represented in the transferred style_ref.",
                "preserved_identity": "mock: retain at least one recognizable brand or function cue.",
                "redesigned_parts": "mock: allow structure to be simplified or recomposed when theme language requires it.",
                "preserve_major_structure": True,
                "structure_evidence": "mock: the main silhouette and spatial arrangement remain recognizable.",
            }
            for app_name in used
        ],
        "shared_design_rules": ["所有目标 App 共用同一个 theme_board"],
        "identity_handling_policy": "mock: choose identity expression dynamically from target image and neutral app semantics.",
        "structure_preservation_policy": {
            "decision_scope": "per_app_before_generation",
            "preserve_when": "主题样例保留主要轮廓和空间排列时",
            "recompose_when": "主题样例用语义符号或场景替换主要几何结构时",
        },
        "common_forbidden_failures": [
            "不要把所有 App 都变成通用装饰物",
            "不要针对每个 App 重新定义主题风格",
        ],
    }


def _mock_identity_strategy(theme_design_analysis, target_profile):
    app = target_profile.get("app", "")
    strategy_type = "semantic_recompose" if app in {"xiaohongshu", "damai", "tieba"} else "logo_simplify"
    structure_mode = (
        "semantic_recompose"
        if strategy_type in {"semantic_recompose", "symbolic_scene"}
        else "preserve_major_structure"
    )
    return {
        "app": app,
        "strategy_type": strategy_type,
        "identity_constraint_level": "balanced",
        "design_rationale": "mock: choose an app expression from the target image, neutral app profile, and shared theme_board.",
        "must_preserve": ["recognizable app name or symbol cue", "core function cue"],
        "can_recompose": ["layout", "surface treatment", "supporting symbols"],
        "forbid": ["generic decoration without app recognition", "copying reference app identity", "redefining the shared theme"],
        "generation_direction": (
            f"为 {target_profile.get('display_name', app)} 创建主题化图标。"
            "遵循共享 theme_board，并保留足够的 App 可识别性。"
        ),
        "identity_anchor": "mock：目标图的主要轮廓和 App 功能线索",
        "brand_cues_to_preserve": target_profile.get("brand_identity_cues", []),
        "semantic_cues_to_preserve": [target_profile.get("core_function", "")] if target_profile.get("core_function") else [],
        "style_fidelity_priority": "theme_fidelity_first",
        "structure_preservation_mode": structure_mode,
        "structure_identity_metric_applicable": structure_mode == "preserve_major_structure",
        "structure_policy_rationale": "mock: freeze the expected structure policy before image generation.",
    }


def _mock_transfer_plan(theme_rules, target_identity, identity_strategy=None):
    app = target_identity.get("app", "")
    preserve = target_identity.get("must_preserve", [])
    forbid = target_identity.get("must_not_replace_with", [])
    strategy = identity_strategy or {}
    if strategy:
        preserve = strategy.get("must_preserve", preserve)
        forbid = strategy.get("forbid", forbid)
    structure_mode = strategy.get("structure_preservation_mode")
    if structure_mode not in STRUCTURE_PRESERVATION_MODES:
        structure_mode = (
            "semantic_recompose"
            if strategy.get("strategy_type") in {"semantic_recompose", "symbolic_scene"}
            else "preserve_major_structure"
        )
    return {
        "app": app,
        "strategy_type": strategy.get("strategy_type", "logo_simplify"),
        "identity_constraint_level": strategy.get("identity_constraint_level", "balanced"),
        "preserve": preserve,
        "must_preserve": preserve,
        "recompose_allowed": strategy.get("can_recompose", []),
        "restyle": [
            "应用共享的主题轮廓、纹理、光照和背景规则",
            "在不改变目标 logo 骨架的前提下柔化边缘",
        ],
        "decorate": ["只有在不遮挡身份锚点时才增加小型主题装饰"],
        "forbid": forbid,
        "generation_brief": strategy.get(
            "generation_direction",
            (
                f"在保留 {app} 核心结构和可识别颜色的同时，将其转换为共享主题。"
                "不要用通用吉祥物替换主体。"
            ),
        ),
        "color_application": "将 theme_001 的颜色转换规则应用到目标图标，不要创造新色板。",
        "stroke_application": "应用 theme_001 的描边粗细、圆润边缘和边缘密度规则。",
        "composition_application": "匹配 theme_001 的主体比例、居中程度、留白和背景占比。",
        "identity_application": "保留身份锚点，并用共享的 theme_001 设计语言表现它们。",
        "fidelity_constraints": [
            "结果必须像 theme_001 中缺失的成员",
            "theme_001 的颜色、描边和构图规则优先于单个 App 的风格漂移",
        ],
        "negative_constraints": [
            "不要创造新的主题风格",
            "不要生成仅内部自洽、却与 theme_001 不一致的图标",
        ],
        "structure_preservation_mode": structure_mode,
        "structure_identity_metric_applicable": structure_mode == "preserve_major_structure",
        "structure_policy_rationale": strategy.get(
            "structure_policy_rationale",
            "mock: copied from the pre-generation identity strategy.",
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
            "warning": "Qwen 返回了无效 JSON，已使用后备主题分析。",
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
        "warning": "Qwen 返回了无效 JSON，已回退到第一个候选图。",
    }


def _parse_target_identity_json(text, target_app):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_target_identity(target_app),
            "warning": "Qwen 返回了无效 JSON，已使用后备目标身份分析。",
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
            "warning": "Qwen 返回了无效 JSON，已使用后备主题设计分析。",
        }
    fallback = _mock_theme_design(reference_examples, theme_profile)
    for key in REQUIRED_THEME_DESIGN_FIELDS:
        parsed.setdefault(key, fallback[key])
    patterns = parsed.get("reference_transformation_patterns")
    if not isinstance(patterns, list):
        patterns = fallback["reference_transformation_patterns"]
        parsed["reference_transformation_patterns"] = patterns
    present_apps = {
        str(item.get("app", "")).casefold()
        for item in patterns
        if isinstance(item, dict) and item.get("app")
    }
    patterns.extend(
        item
        for item in fallback["reference_transformation_patterns"]
        if str(item.get("app", "")).casefold() not in present_apps
    )
    for item in patterns:
        if isinstance(item, dict):
            item.setdefault("preserve_major_structure", True)
            if not isinstance(item.get("preserve_major_structure"), bool):
                item["preserve_major_structure"] = True
            item.setdefault("structure_evidence", "")
    return parsed


def _parse_identity_strategy_json(text, target_profile):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_identity_strategy({}, target_profile),
            "warning": "Qwen 返回了无效 JSON，已使用后备身份策略。",
        }
    fallback = _mock_identity_strategy({}, target_profile)
    for key in REQUIRED_IDENTITY_STRATEGY_FIELDS:
        parsed.setdefault(key, fallback[key])
    if parsed.get("identity_constraint_level") not in ["strict", "balanced", "flexible"]:
        parsed["identity_constraint_level"] = "balanced"
    _normalize_structure_policy(parsed)
    _protect_brand_identity_cues(parsed, target_profile, preserve_keys=["must_preserve"], recompose_keys=["can_recompose"])
    return parsed


def _parse_transfer_plan_json(text, target_identity, identity_strategy=None, target_profile=None):
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "raw_response": text,
            **_mock_transfer_plan({}, target_identity, identity_strategy=identity_strategy),
            "warning": "Qwen 返回了无效 JSON，已使用后备迁移计划。",
        }
    fallback = _mock_transfer_plan({}, target_identity, identity_strategy=identity_strategy)
    for key in REQUIRED_TRANSFER_PLAN_FIELDS:
        parsed.setdefault(key, fallback[key])
    _normalize_structure_policy(parsed, source=identity_strategy)
    _protect_brand_identity_cues(
        parsed,
        target_profile or {},
        preserve_keys=["preserve", "must_preserve"],
        recompose_keys=["recompose_allowed"],
    )
    return parsed


def _normalize_structure_policy(data, source=None):
    source = source if isinstance(source, dict) else {}
    source_mode = source.get("structure_preservation_mode")
    mode = source_mode if source_mode in STRUCTURE_PRESERVATION_MODES else data.get("structure_preservation_mode")
    if mode not in STRUCTURE_PRESERVATION_MODES:
        strategy_type = source.get("strategy_type", data.get("strategy_type"))
        mode = (
            "semantic_recompose"
            if strategy_type in {"semantic_recompose", "symbolic_scene"}
            else "preserve_major_structure"
        )
    data["structure_preservation_mode"] = mode
    data["structure_identity_metric_applicable"] = mode == "preserve_major_structure"
    data["structure_policy_rationale"] = source.get(
        "structure_policy_rationale",
        data.get("structure_policy_rationale", "根据主题样例与目标身份在生成前确定的策略。"),
    )
    return data


def _protect_brand_identity_cues(data, target_profile, preserve_keys, recompose_keys):
    cues = []
    if isinstance(target_profile, dict):
        value = target_profile.get("brand_identity_cues", [])
        if isinstance(value, list):
            cues = [str(item) for item in value if str(item).strip()]
    if not cues:
        return data

    preserve_cues, text_cues = _identity_preserve_cues(cues)
    for key in preserve_keys:
        existing = data.get(key, [])
        if not isinstance(existing, list):
            existing = []
        data[key] = _dedupe_list([*existing, *preserve_cues])

    for key in recompose_keys:
        data[key] = _remove_conflicting_items(data.get(key, []), cues)
    data["forbid"] = _remove_conflicting_items(data.get("forbid", []), cues)
    if text_cues:
        data["text_policy"] = _merge_text_identity_policy(data.get("text_policy"), text_cues)
    return data


def _identity_preserve_cues(cues):
    preserve_cues = []
    text_cues = []
    for cue in cues:
        cue_text = str(cue).strip()
        if not cue_text:
            continue
        if _is_text_identity_cue(cue_text):
            text_cues.append(cue_text)
            preserve_cues.append(
                f"将 {cue_text} 的身份布局和字形轮廓保留为不可读的图形，不要求生成可读文字。"
            )
        else:
            preserve_cues.append(cue_text)
    return preserve_cues, text_cues


def _is_text_identity_cue(cue):
    lower = str(cue).strip().lower()
    text_terms = [
        "wordmark",
        "logotype",
        "text mark",
        "text logo",
        "letterform",
        "typography",
        "typeface",
        "文字",
        "文本",
        "字标",
        "字形",
        "字母",
        "英文",
        "中文",
        "汉字",
        "字体",
    ]
    return any(term in lower for term in text_terms)


def _merge_text_identity_policy(existing_policy, text_cues):
    policy = existing_policy if isinstance(existing_policy, dict) else {}
    existing_cues = policy.get("source_cues", [])
    if not isinstance(existing_cues, list):
        existing_cues = []
    policy.update(
        {
            "mode": "preserve_identity_shape_without_readable_text",
            "source_cues": _dedupe_list([*existing_cues, *text_cues]),
            "positive_rule": "只通过整体布局、图标轮廓和字形轮廓保留品牌身份。",
            "negative_rule": "不要生成清晰可读的文字、类似 OCR 的字母或乱码字符。",
        }
    )
    return policy


def _remove_conflicting_items(items, cues):
    if not isinstance(items, list):
        return []
    cue_terms = _identity_cue_terms(cues)
    cleaned = []
    for item in items:
        item_text = str(item)
        lower = item_text.lower()
        if any(term in lower for term in cue_terms):
            continue
        cleaned.append(item)
    return cleaned


def _identity_cue_terms(cues):
    return [str(cue).strip().lower() for cue in cues if str(cue).strip()]


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
        "overall_comment": "Qwen 返回了无效 JSON，已使用后备整包 QC 报告。",
    }


def _final_output_items(final_outputs):
    if isinstance(final_outputs, dict):
        return [(app_name, path) for app_name, path in final_outputs.items()]
    return [(Path(path).stem, path) for path in final_outputs]


def _final_output_app_names(final_outputs):
    return [app_name for app_name, _ in _final_output_items(final_outputs)]


def _sample_final_output_items(final_outputs, max_items=12):
    items = _final_output_items(final_outputs)
    if len(items) <= max_items:
        return items
    indices = [round(index * (len(items) - 1) / (max_items - 1)) for index in range(max_items)]
    return [items[index] for index in indices]


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
