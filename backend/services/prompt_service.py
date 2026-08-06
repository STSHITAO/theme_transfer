import re
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    theme_design = theme_design_analysis or {}

    prompt = f"""[THEME STYLE CONTRACT]
theme_id: {theme_id}

The style-reference images are evidence for visual treatment only.
They may contribute color, material, texture, lighting, shadow, background treatment, edge rendering, subject scale, and detail density.
They must contribute zero identity content: no logo, text, symbol, subject silhouette, object category, brand identity, or internal layout.

[Theme summary]
{analysis.get("theme_style_analysis", "")}

[Shared original-to-style transform]
{analysis.get("common_original_to_style_transform", "")}

[Color]
{theme_design.get("color_transform_rule", analysis.get("color_palette", ""))}

[Background]
{theme_design.get("background_transform_rule", analysis.get("common_background_transform", ""))}

[Edges and strokes]
{theme_design.get("stroke_transform_rule", analysis.get("line_style", ""))}

[Composition]
{theme_design.get("composition_transform_rule", analysis.get("icon_composition_rules", ""))}

[Subject scale]
{theme_design.get("subject_scale_rule", "")}

[Detail complexity]
{theme_design.get("detail_complexity_rule", "")}

[Required style properties]
{_format_bullets(theme_design.get("theme_fidelity_constraints", []))}

[Forbidden style drift]
{_format_bullets(theme_design.get("forbidden_style_drift", []))}

Identity is a hard constraint. Apply this theme only within the target identity; theme fidelity must never replace target identity.
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
            "Semantic recomposition is allowed only when it preserves the target App's specified identity and functional cues. "
            "No subject or geometry may be borrowed from a style reference."
        )
    else:
        structure_instruction = (
            "This is a material-and-style transformation. Preserve the target silhouette, topology, major geometry, "
            "negative space, and internal spatial relationships; do not replace or reinterpret the target subject."
        )

    prompt = f"""{base_prompt}

[ABSOLUTE IDENTITY LOCK — HIGHEST PRIORITY]

The first input image, labelled TARGET_IMAGE, is the only allowed source of subject identity.
Every following image, labelled STYLE_REFERENCE, is style-only evidence.

TARGET_IMAGE is the only allowed source of:
- logo and text content
- recognizable symbol and semantic object
- subject silhouette and object category
- internal geometry, negative space, and spatial layout

STYLE_REFERENCE images may contribute only visual treatment. They must contribute zero logo content, text, recognizable symbol, subject silhouette, object category, internal layout, or brand identity.
If any identity content from a STYLE_REFERENCE appears in the result, the result is invalid.

[CURRENT TARGET]
target_app: {target_app}
display_name: {profile.get("display_name", target_app)}
category: {profile.get("category", "")}
core_function: {profile.get("core_function", profile.get("store_description", ""))}

[TARGET SEMANTIC FACTS — FROM TARGET.JSON]
store_description: {profile.get("store_description", "")}
These neutral facts explain what CURRENT TARGET does. Derive a visual interpretation only within the frozen theme-specific structure policy below. They do not prescribe an object and do not authorize identity content from STYLE_REFERENCE images.

[STRUCTURE POLICY]
mode: {structure_mode}
identity_constraint_level: {plan.get("identity_constraint_level", "strict")}
{structure_instruction}

[MUST PRESERVE FROM TARGET_IMAGE]
{_format_bullets(plan.get("must_preserve", []))}

[TARGET IDENTITY APPLICATION]
{plan.get("identity_application", "")}

[EXECUTION BRIEF]
{plan.get("generation_brief", plan.get("generation_direction", "Apply the frozen target identity within the theme contract."))}

[ALLOWED TARGET RECOMPOSITION]
{_format_bullets(plan.get("recompose_allowed", []))}

[THEME APPLICATION]
- Color: {plan.get("color_application", "")}
- Edges and strokes: {plan.get("stroke_application", "")}
- Composition: {plan.get("composition_application", "")}
- Required theme properties: {_format_inline(plan.get("fidelity_constraints", []))}

[INVALID OUTPUT CONDITIONS]
- Any text, logo, recognizable symbol, subject, silhouette, or internal layout copied from a STYLE_REFERENCE.
- Any text or symbol absent from TARGET_IMAGE.
- A result that resembles a STYLE_REFERENCE subject more than TARGET_IMAGE.
- A changed object category or an unrecognizable target App.
- Violating these target-specific constraints: {_format_inline(plan.get("negative_constraints", []))}

[PRIORITY]
1. Preserve target identity and the required target structure.
2. Apply theme material and visual treatment without changing identity.
3. Add decorative detail only when priorities 1 and 2 remain satisfied.

If identity and theme styling conflict, preserve identity. Theme fidelity must never replace target identity.

[FINAL INTERNAL CHECK]
Before generating, identify the subject and required geometry in TARGET_IMAGE, separate them from the visual treatment in STYLE_REFERENCE images, and confirm that the output contains zero reference identity.

[OUTPUT]
Return exactly one complete themed target icon. No captions, comparison layout, input collage, watermark, explanation, or extra logo.
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
    return "\n".join(f"- {item}" for item in items) if items else "- Follow the frozen theme contract."


def _format_inline(value):
    items = _as_items(value)
    return "; ".join(items) if items else "none beyond the stated contract"


def _as_items(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
