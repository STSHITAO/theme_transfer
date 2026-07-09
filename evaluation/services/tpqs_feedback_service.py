from __future__ import annotations

from pathlib import Path


DEFAULT_LOW_SCORE_THRESHOLD = 45.0


def build_tpqs_feedback_retry_prompt(report: dict, threshold: float = DEFAULT_LOW_SCORE_THRESHOLD) -> str:
    theme_id = report.get("theme_id") or report.get("theme") or "theme_001"
    lines = [
        f"# ITTE Generation Feedback Prompt",
        "",
        f"Theme: {theme_id}",
        "",
        "Goal: improve theme fidelity. The next generation should look like a missing App icon from the reference theme package, not a newly invented theme.",
        "",
        "Important: do not call Wan automatically from this prompt. This file is diagnostic generation feedback for human review or a later manual retry step; it is not an automatic retry mechanism.",
        "",
        "Global constraints:",
        f"- Match {theme_id}'s color, stroke, background, composition, subject scale, and detail complexity rules.",
        "- Preserve the target App identity anchors and semantic cues.",
        "- Do not create a new internally consistent icon style that drifts away from the reference theme package.",
    ]

    checks = [
        (
            "color_delta_score",
            "Color correction",
            f"Strengthen the {theme_id} color system. Do not introduce new high-saturation colors or colors outside the reference theme palette. Match the reference hue range, saturation, brightness, and background treatment.",
        ),
        (
            "edge_delta_score",
            "Stroke correction",
            f"Align stroke thickness, line density, edge roundedness, and edge complexity with {theme_id}. Avoid sharper, thinner, denser, or more complex outlines than the reference style.",
        ),
        (
            "composition_delta_score",
            "Composition correction",
            f"Align subject size, centering, whitespace ratio, and background occupancy with {theme_id}. Do not change the layout into a new composition language.",
        ),
    ]

    for key, title, instruction in checks:
        score = _numeric_score(report.get(key))
        if score is not None and score < threshold:
            lines.extend(["", f"{title} ({key}={score:.2f}):", f"- {instruction}"])

    complexity_score = _numeric_score(report.get("complexity_delta_score"))
    if complexity_score is not None and complexity_score < threshold:
        lines.extend(
            [
                "",
                f"Detail complexity correction (complexity_delta_score={complexity_score:.2f}):",
                f"- Match {theme_id}'s detail density and decorative complexity. Avoid both over-simplified generic shapes and over-detailed off-theme rendering.",
            ]
        )

    if report.get("identity_match_correct") is False:
        lines.extend(
            [
                "",
                "Identity correction (identity_match_correct=false):",
                "- Strengthen the current App identity_anchor, brand cues, and semantic cues while still applying the shared theme fidelity rules.",
            ]
        )

    if len(lines) == 9:
        lines.extend(
            [
                "",
                "No severe low-score delta trigger was found. Keep the current generation direction and use this prompt only as a theme fidelity reminder.",
            ]
        )

    return "\n".join(lines) + "\n"


def write_tpqs_feedback_retry_prompt(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    feedback = build_tpqs_feedback_retry_prompt(report)
    new_path = output_dir / "generation_feedback_prompt.md"
    new_path.write_text(feedback, encoding="utf-8")
    path = output_dir / "tpqs_feedback_retry_prompt.md"
    path.write_text(feedback, encoding="utf-8")
    return path


def _numeric_score(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
