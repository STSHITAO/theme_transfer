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
