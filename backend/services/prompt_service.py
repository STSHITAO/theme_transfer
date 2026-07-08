import json
from pathlib import Path


def build_generation_prompt(analysis, theme_id, case_id, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    template_path = root / "prompts" / "wan_generation.md"
    template = template_path.read_text(encoding="utf-8")
    output_path = root / "data" / "cases" / case_id / "generation_prompt.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = f"""{template}

【主题包】
theme_id: {theme_id}

【输入图片说明】
图 1 到图 N 是同一个主题包 {theme_id} 中的纯主题风格参考图 style_ref。
最后一张图是目标 App 的原始目标图。

【任务】
请学习参考图中共同的主体重绘规则、背景重绘规则、前景与背景关系、视觉语言和主题装饰规律。
请将这些统一主题规律迁移到最后一张目标图。
必须保留目标 App 的主体身份、核心符号、轮廓、背景结构和整体构图。
目标身份只能来自最后一张目标图。
严禁复制参考 App 的主体符号、logo、轮廓或身份。
严禁把 douyin/TikTok、wechat、alipay 等参考 App 主体迁移到目标图。
最终只输出一个完整图标，不要输出对照板、分栏布局、说明文字或输入图片拼贴。
不要文字、水印、额外 logo、额外物体、乱码。

【Qwen 主题分析】
{analysis.get("theme_style_analysis", "")}

【生成提示】
{analysis.get("generation_prompt", "")}

【目标保留要求】
{analysis.get("target_preservation", "")}

【负向提示】
{analysis.get("negative_prompt", "")}
"""
    output_path.write_text(prompt, encoding="utf-8")
    return str(output_path)


def build_generation_base_prompt(analysis, theme_id, output_path, root_dir=None, theme_design_analysis=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    template = (root / "prompts" / "wan_generation.md").read_text(encoding="utf-8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    prompt = f"""{template}

【批量主题包基础规则】
theme_id: {theme_id}

这是一份整包共用的基础 Wan prompt。所有 target App 必须复用同一份主题规则，不允许为单个 App 重新定义风格。

【统一主题分析】
{analysis.get("theme_style_analysis", "")}

【背景重绘规则】
{analysis.get("common_background_transform", "")}

【主体重绘规则】
{analysis.get("common_foreground_transform", "")}

【颜色规则】
{analysis.get("color_palette", "")}

【线条规则】
{analysis.get("line_style", "")}

【材质纹理规则】
{analysis.get("texture_material", "")}

【光照阴影规则】
{analysis.get("lighting_shadow", "")}

【构图规则】
{analysis.get("icon_composition_rules", "")}

【负向约束】
{analysis.get("negative_prompt", "")}

【V2 theme_board】
{json.dumps((theme_design_analysis or {}).get("theme_board", {}), ensure_ascii=False, indent=2)}

【V2 主题化设计原则】
所有 target App 必须共享同一个 theme_board。identity_strategy 只能决定目标 App 如何表达，不能重新定义主题风格。

整包一致性要求：
- 所有输出必须像同一套主题图标包。
- 所有输出必须共享同一套颜色、线条、材质、阴影、背景和构图语言。
- 每个目标 App 只能保留自己的主体身份，不能复制参考 App 的 logo、文字、轮廓或身份。
- 不要生成对照板、分栏图、说明文字、水印、额外 logo 或额外主体。
"""
    output.write_text(prompt, encoding="utf-8")
    return str(output)


def build_package_target_prompt(base_prompt, target_app, output_path, transfer_plan=None):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    transfer_plan_text = json.dumps(transfer_plan or {}, ensure_ascii=False, indent=2)
    prompt = f"""{base_prompt}

【当前目标 App】
target_app: {target_app}

【transfer_plan】
{transfer_plan_text}

【身份优先级】
请严格执行 transfer_plan。目标 App 的主体结构、关键符号、识别色和整体构图优先级高于风格化程度。
如果风格迁移会导致目标 App 无法识别，请降低风格化强度，而不是替换主体。
禁止把目标主体改成通用毛绒球、无关动物、无身份锚点的可爱团子。

【目标保留要求】
最后一张输入图是 {target_app} 的原始图标，是唯一的目标身份来源。
必须保留 {target_app} 的核心轮廓、主体结构、关键符号、背景结构和整体构图。
只允许把整包共用的主题规则迁移到 {target_app} 上，不允许重新定义本 App 的独立风格。

【输出要求】
只输出一个完整主题图标。不要输出文字说明、对照板、分栏布局、输入拼贴、水印或额外 logo。
"""
    output.write_text(prompt, encoding="utf-8")
    return str(output)
