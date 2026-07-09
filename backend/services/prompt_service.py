import json
from pathlib import Path


def build_generation_prompt(analysis, theme_id, case_id, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    template = (root / "prompts" / "wan_generation.md").read_text(encoding="utf-8")
    output_path = root / "data" / "cases" / case_id / "generation_prompt.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = f"""{template}

[Theme Package]
theme_id: {theme_id}

[Input Images]
Images 1..N are style_ref references from the same theme package.
The final image is the target App original icon and is the only source of target identity.

[Theme Fidelity Goal]
Generate an icon that looks like a missing member of {theme_id}, not a newly invented theme.
Strictly follow the reference package color, stroke, background, composition, material, and detail rules.

[Qwen Theme Analysis]
{analysis.get("theme_style_analysis", "")}

[Generation Direction]
{analysis.get("generation_prompt", "")}

[Target Preservation]
{analysis.get("target_preservation", "")}

[Negative Constraints]
{analysis.get("negative_prompt", "")}
"""
    output_path.write_text(prompt, encoding="utf-8")
    return str(output_path)


def build_generation_base_prompt(analysis, theme_id, output_path, root_dir=None, theme_design_analysis=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    template = (root / "prompts" / "wan_generation.md").read_text(encoding="utf-8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    theme_design = theme_design_analysis or {}

    prompt = f"""{template}

[Package Base Prompt]
theme_id: {theme_id}

This base prompt is shared by every target App in the package.
All outputs must reuse the same theme fidelity rules. Identity strategy may change how an App is expressed, but it must not redefine the global theme style.

[Theme Fidelity Goal]
The result must look like a missing App icon from {theme_id}.
Do not create a new theme style.
Do not generate only an internally consistent icon set that is inconsistent with {theme_id}.

[Theme Style Analysis]
{analysis.get("theme_style_analysis", "")}

[Reference Raw To Style Transform]
{analysis.get("common_original_to_style_transform", "")}

[Color Transform Rule]
{theme_design.get("color_transform_rule", analysis.get("color_palette", ""))}

[Background Transform Rule]
{theme_design.get("background_transform_rule", analysis.get("common_background_transform", ""))}

[Stroke Transform Rule]
{theme_design.get("stroke_transform_rule", analysis.get("line_style", ""))}

[Composition Transform Rule]
{theme_design.get("composition_transform_rule", analysis.get("icon_composition_rules", ""))}

[Subject Scale Rule]
{theme_design.get("subject_scale_rule", "")}

[Detail Complexity Rule]
{theme_design.get("detail_complexity_rule", "")}

[Theme Fidelity Constraints]
{json.dumps(theme_design.get("theme_fidelity_constraints", []), ensure_ascii=False, indent=2)}

[Forbidden Style Drift]
{json.dumps(theme_design.get("forbidden_style_drift", []), ensure_ascii=False, indent=2)}

[Theme Board]
{json.dumps(theme_design.get("theme_board", {}), ensure_ascii=False, indent=2)}

[Global Wan Constraints]
- Output must look like a missing member of {theme_id}.
- Do not invent a new color palette, stroke style, composition system, or background treatment.
- Color must match {theme_id}'s hue range, saturation, brightness, and background handling.
- Stroke must match {theme_id}'s line weight, roundedness, edge density, and edge complexity.
- Composition must match {theme_id}'s subject scale, centering, whitespace, and background ratio.
- Preserve the current target App identity cues, but theme fidelity has priority unless identity would become unrecognizable.
- Do not copy any reference App identity, logo, wordmark, or subject.
- Do not output text explanations, comparison boards, watermarks, extra logos, or pasted input images.
"""
    output.write_text(prompt, encoding="utf-8")
    return str(output)


def build_package_target_prompt(base_prompt, target_app, output_path, transfer_plan=None):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = transfer_plan or {}
    prompt = f"""{base_prompt}

[Current Target App]
target_app: {target_app}

[transfer_plan]
{json.dumps(plan, ensure_ascii=False, indent=2)}

[Executable Theme Fidelity Instructions]
Color application: {plan.get("color_application", "")}
Stroke application: {plan.get("stroke_application", "")}
Composition application: {plan.get("composition_application", "")}
Identity application: {plan.get("identity_application", "")}

Fidelity constraints:
{json.dumps(plan.get("fidelity_constraints", []), ensure_ascii=False, indent=2)}

Negative constraints:
{json.dumps(plan.get("negative_constraints", []), ensure_ascii=False, indent=2)}

[Priority]
Strictly execute transfer_plan.
The icon must preserve target App identity anchors and semantic cues, while matching {target_app}'s output to the shared theme fidelity rules.
Do not reduce the task to making a merely nice-looking icon. It must look like {target_app} as a missing member of the same theme package.

[Output]
Return exactly one complete themed icon. No captions, no comparison layout, no input collage, no watermark, no extra logo.
"""
    output.write_text(prompt, encoding="utf-8")
    return str(output)
