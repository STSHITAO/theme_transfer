import re
from pathlib import Path


def build_generation_prompt(analysis, theme_id, case_id, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    template = (root / "prompts" / "wan_generation.md").read_text(encoding="utf-8")
    output_path = root / "data" / "cases" / case_id / "generation_prompt.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = f"""{template}

【主题包】
theme_id: {theme_id}

【输入图片】
第一张图片是目标 App 原始图，标记为 TARGET_IMAGE，也是唯一的目标身份来源。
后续图片是同一主题包的 style_ref 风格参考图，标记为 STYLE_REFERENCE。

【主题一致性目标】
生成的图标应像 {theme_id} 中原本缺少的一员，而不是新创造的另一套主题。
严格遵循参考主题包的颜色、描边、背景、构图、材质和细节规则。

【Qwen 主题分析】
{analysis.get("theme_style_analysis", "")}

【生成方向】
{analysis.get("generation_prompt", "")}

【目标保留要求】
{analysis.get("target_preservation", "")}

【负面约束】
{analysis.get("negative_prompt", "")}
"""
    output_path.write_text(prompt, encoding="utf-8")
    return str(output_path)


def build_generation_base_prompt(analysis, theme_id, output_path, root_dir=None, theme_design_analysis=None):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    theme_design = theme_design_analysis or {}

    prompt = f"""【主题风格契约】
theme_id: {theme_id}

风格参考图只能作为视觉处理方式的依据。
它们可以提供颜色、材质、纹理、光照、阴影、背景处理、边缘渲染、主体比例和细节密度。
它们不得提供任何身份内容，包括 logo、文字、符号、主体轮廓、对象类别、品牌身份或内部布局。

【主题概述】
{analysis.get("theme_style_analysis", "")}

【共享的原图到主题图转换规则】
{analysis.get("common_original_to_style_transform", "")}

【颜色】
{theme_design.get("color_transform_rule", analysis.get("color_palette", ""))}

【背景】
{theme_design.get("background_transform_rule", analysis.get("common_background_transform", ""))}

【边缘与描边】
{theme_design.get("stroke_transform_rule", analysis.get("line_style", ""))}

【构图】
{theme_design.get("composition_transform_rule", analysis.get("icon_composition_rules", ""))}

【主体比例】
{theme_design.get("subject_scale_rule", "")}

【细节复杂度】
{theme_design.get("detail_complexity_rule", "")}

【必须满足的风格属性】
{_format_bullets(theme_design.get("theme_fidelity_constraints", []))}

【禁止的风格漂移】
{_format_bullets(theme_design.get("forbidden_style_drift", []))}

目标身份是硬约束。只能在目标身份范围内应用主题；主题一致性绝不能取代目标身份。
"""
    output.write_text(prompt, encoding="utf-8")
    return str(output)


def build_package_target_prompt(
    base_prompt,
    target_app,
    output_path,
    transfer_plan=None,
    target_profile=None,
    forbidden_reference_terms=None,
):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = transfer_plan or {}
    profile = target_profile or {}
    structure_mode = plan.get("structure_preservation_mode", "preserve_major_structure")
    if structure_mode == "semantic_recompose":
        structure_instruction = (
            "只有在保留目标 App 指定身份线索和功能线索的前提下，才允许进行语义重构。"
            "不得从风格参考图借用任何主体或几何结构。"
        )
    else:
        structure_instruction = (
            "这是材质与风格转换。必须保留目标的轮廓、拓扑、主要几何结构、负空间和内部空间关系；"
            "不得替换或重新解释目标主体。"
        )

    prompt = f"""{base_prompt}

【绝对身份锁定——最高优先级】

第一张输入图片标记为 TARGET_IMAGE，是唯一允许的主体身份来源。
其后每张图片均标记为 STYLE_REFERENCE，只能作为风格依据。

TARGET_IMAGE 是以下内容的唯一允许来源：
- logo 与文字内容
- 可识别符号与语义对象
- 主体轮廓与对象类别
- 内部几何结构、负空间与空间布局

STYLE_REFERENCE 图片只能提供视觉处理方式。它们不得提供任何 logo 内容、文字、可识别符号、主体轮廓、对象类别、内部布局或品牌身份。
如果结果中出现任何来自 STYLE_REFERENCE 的身份内容，该结果即为无效。

【当前目标】
target_app: {target_app}
display_name: {profile.get("display_name", target_app)}
category: {profile.get("category", "")}
core_function: {profile.get("core_function", profile.get("store_description", ""))}

【来自 target.json 的目标语义事实】
store_description: {profile.get("store_description", "")}
这些中立事实用于说明当前目标的功能。只能在下方已冻结的主题专属结构策略范围内推导视觉表达。它们不指定具体对象，也不允许引入 STYLE_REFERENCE 中的身份内容。

【结构策略】
mode: {structure_mode}
identity_constraint_level: {plan.get("identity_constraint_level", "strict")}
{structure_instruction}

【必须从 TARGET_IMAGE 保留的内容】
{_format_bullets(plan.get("must_preserve", []))}

【目标身份应用方式】
{plan.get("identity_application", "")}

【执行简述】
{plan.get("generation_brief", plan.get("generation_direction", "在主题契约范围内应用已冻结的目标身份。"))}

【允许的目标重构】
{_format_bullets(plan.get("recompose_allowed", []))}

【主题应用】
- 颜色：{plan.get("color_application", "")}
- 边缘与描边：{plan.get("stroke_application", "")}
- 构图：{plan.get("composition_application", "")}
- 必须满足的主题属性：{_format_inline(plan.get("fidelity_constraints", []))}

【无效输出条件】
- 从 STYLE_REFERENCE 复制任何文字、logo、可识别符号、主体、轮廓或内部布局。
- 出现 TARGET_IMAGE 中不存在的文字或符号。
- 结果与 STYLE_REFERENCE 主体的相似度高于与 TARGET_IMAGE 的相似度。
- 改变对象类别，或导致目标 App 无法识别。
- 违反以下目标专属约束：{_format_inline(plan.get("negative_constraints", []))}

【优先级】
1. 保留目标身份和要求保留的目标结构。
2. 在不改变身份的前提下应用主题材质与视觉处理。
3. 只有在优先级 1 和 2 仍被满足时，才可增加装饰细节。

如果身份与主题风格冲突，优先保留身份。主题一致性绝不能取代目标身份。

【最终内部检查】
生成前，先识别 TARGET_IMAGE 中的主体和必须保留的几何结构，将其与 STYLE_REFERENCE 图片的视觉处理方式明确分离，并确认输出不包含任何参考图身份。

【输出】
只返回一个完整的主题化目标图标。不要生成标题、对照布局、输入拼贴、水印、解释或额外 logo。
"""
    prompt = scrub_reference_identity_terms(prompt, forbidden_reference_terms or [])
    leaked_terms = find_reference_identity_terms(prompt, forbidden_reference_terms or [])
    if leaked_terms:
        raise ValueError("Generation prompt contains forbidden reference identity terms: " + ", ".join(leaked_terms))
    output.write_text(prompt, encoding="utf-8")
    return str(output)


def scrub_reference_identity_terms(text, terms):
    result = str(text)
    for term in _normalized_terms(terms):
        result = _term_pattern(term).sub("STYLE_REFERENCE", result)
    return result


def find_reference_identity_terms(text, terms):
    return [term for term in _normalized_terms(terms) if _term_pattern(term).search(str(text))]


def _normalized_terms(terms):
    unique = {str(term).strip() for term in terms if str(term).strip()}
    return sorted(unique, key=lambda item: (-len(item), item.casefold()))


def _term_pattern(term):
    escaped = re.escape(term)
    if term.isascii() and (term[0].isalnum() or term[0] == "_") and (term[-1].isalnum() or term[-1] == "_"):
        escaped = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    return re.compile(escaped, flags=re.IGNORECASE)


def _format_bullets(value):
    items = _as_items(value)
    return "\n".join(f"- {item}" for item in items) if items else "- 遵循已冻结的主题契约。"


def _format_inline(value):
    items = _as_items(value)
    return "；".join(items) if items else "除已说明的契约外无其他要求"


def _as_items(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
