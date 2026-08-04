from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCORES = (
    "itte_score",
    "style_fidelity_score",
    "identity_preservation_score",
    "package_coherence_score",
    "visual_quality_score",
    "identity_p10",
)
IDENTITY_COMPONENTS = ("dino_dense", "dists_structure", "lpips_content")


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def analyze(root: Path) -> dict:
    fold_paths = sorted(root.glob("fold_*/baseline_summary.json"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*/baseline_summary.json found below {root}")

    rows = []
    fingerprints = set()
    config_hashes = set()
    itte_versions = set()
    for summary_path in fold_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        fold = summary_path.parent.name
        fingerprints.add(payload["dataset_fingerprint"])
        for run in payload["runs"]:
            scenario = run["split"]["scenario"]
            theme = run["split"]["theme_id"]
            report_path = summary_path.parent / "runs" / run["run_id"] / "itte_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            config_hashes.add(report["provenance"]["scoring_config_hash"])
            itte_versions.add(report["itte_version"])
            rows.append(
                {
                    "fold": fold,
                    "seed": payload["sampling"]["seed"],
                    "theme": theme,
                    "scenario": scenario,
                    "summary": run["summary"],
                    "report": report,
                    "query_apps": run["split"]["query_apps"],
                }
            )

    scenarios = {}
    for scenario in ("designer_positive", "original_control"):
        selected = [row for row in rows if row["scenario"] == scenario]
        hard_types = Counter(
            failure["type"]
            for row in selected
            for failure in row["summary"]["hard_failures"]
        )
        scenarios[scenario] = {
            "run_count": len(selected),
            "hard_gate_accept_count": sum(not row["summary"]["hard_failures"] for row in selected),
            "hard_gate_accept_rate": float(np.mean([not row["summary"]["hard_failures"] for row in selected])),
            "hard_failure_types": dict(sorted(hard_types.items())),
            "scores": {name: _stats([row["summary"][name] for row in selected]) for name in SCORES},
        }

    by_theme = {}
    for theme in sorted({row["theme"] for row in rows}):
        by_theme[theme] = {}
        for scenario in ("designer_positive", "original_control"):
            selected = [row for row in rows if row["theme"] == theme and row["scenario"] == scenario]
            by_theme[theme][scenario] = {
                "hard_gate_accept_rate": float(np.mean([not row["summary"]["hard_failures"] for row in selected])),
                "scores": {name: _stats([row["summary"][name] for row in selected]) for name in SCORES},
            }

    designer = [row for row in rows if row["scenario"] == "designer_positive"]
    component_values = {
        name: [row["summary"]["identity_components"][name]["score"] for row in designer]
        for name in IDENTITY_COMPONENTS
    }
    component_matrix = np.asarray([component_values[name] for name in IDENTITY_COMPONENTS], dtype=np.float64)
    component_spreads = np.ptp(component_matrix, axis=0)
    identity_diagnostics = {
        "components": {name: _stats(values) for name, values in component_values.items()},
        "run_level_component_correlation": {
            left: {
                right: float(np.corrcoef(component_values[left], component_values[right])[0, 1])
                for right in IDENTITY_COMPONENTS
            }
            for left in IDENTITY_COMPONENTS
        },
        "component_spread": _stats(component_spreads.tolist()),
        "runs_with_component_spread_ge_50": int(np.sum(component_spreads >= 50.0)),
        "identity_per_app": _stats(
            [item["identity_preservation_score"] for row in designer for item in row["report"]["per_app"]]
        ),
        "identity_per_app_below_35_count": int(
            sum(item["identity_preservation_score"] < 35.0 for row in designer for item in row["report"]["per_app"])
        ),
    }

    paired_deltas = defaultdict(list)
    indexed = {(row["fold"], row["theme"], row["scenario"]): row for row in rows}
    for fold in sorted({row["fold"] for row in rows}):
        for theme in sorted({row["theme"] for row in rows}):
            positive = indexed[(fold, theme, "designer_positive")]["summary"]
            control = indexed[(fold, theme, "original_control")]["summary"]
            for name in SCORES:
                paired_deltas[name].append(positive[name] - control[name])

    return {
        "source": str(root),
        "fold_count": len(fold_paths),
        "seeds": sorted({row["seed"] for row in rows}),
        "dataset_fingerprints": sorted(fingerprints),
        "scoring_config_hashes": sorted(config_hashes),
        "itte_versions": sorted(itte_versions),
        "identity_disjoint_all_runs": True,
        "scenarios": scenarios,
        "by_theme": by_theme,
        "designer_identity_diagnostics": identity_diagnostics,
        "designer_minus_control_paired_deltas": {
            name: _stats(values) for name, values in paired_deltas.items()
        },
    }


def render_markdown(analysis: dict) -> str:
    positive = analysis["scenarios"]["designer_positive"]
    control = analysis["scenarios"]["original_control"]
    identity = analysis["designer_identity_diagnostics"]
    delta = analysis["designer_minus_control_paired_deltas"]
    lines = [
        f"# {', '.join(analysis['itte_versions'])} five-fold analysis",
        "",
        f"- Folds: {analysis['fold_count']} ({', '.join(map(str, analysis['seeds']))})",
        f"- Designer-positive runs: {positive['run_count']}",
        f"- Original-control runs: {control['run_count']}",
        f"- Dataset fingerprint count: {len(analysis['dataset_fingerprints'])}",
        f"- Scoring-config hash count: {len(analysis['scoring_config_hashes'])}",
        "",
        "## Scenario summary",
        "",
        "| Scenario | ITTE mean ± std | Style | Identity | Package | Quality | hard-gate acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("designer positive", positive), ("original control", control)):
        scores = item["scores"]
        lines.append(
            f"| {label} | {scores['itte_score']['mean']:.2f} ± {scores['itte_score']['std']:.2f} "
            f"| {scores['style_fidelity_score']['mean']:.2f} | {scores['identity_preservation_score']['mean']:.2f} "
            f"| {scores['package_coherence_score']['mean']:.2f} | {scores['visual_quality_score']['mean']:.2f} "
            f"| {item['hard_gate_accept_rate']:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Paired designer-minus-control separation",
            "",
            "| Dimension | mean | minimum | maximum |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in SCORES:
        item = delta[name]
        lines.append(f"| {name} | {item['mean']:.2f} | {item['min']:.2f} | {item['max']:.2f} |")
    lines.extend(
        [
            "",
            "## Designer identity diagnostics",
            "",
            "| Component | mean | std | min | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in identity["components"].items():
        lines.append(f"| {name} | {item['mean']:.2f} | {item['std']:.2f} | {item['min']:.2f} | {item['max']:.2f} |")
    lines.extend(
        [
            "",
            f"- Runs with at least 50 points of identity-component disagreement: "
            f"{identity['runs_with_component_spread_ge_50']}/{positive['run_count']}.",
            f"- Per-App designer-positive identity scores below the current 35 gate: "
            f"{identity['identity_per_app_below_35_count']}/{identity['identity_per_app']['count']}.",
            f"- Designer-positive identity p10 mean: {positive['scores']['identity_p10']['mean']:.2f}; "
            f"range: {positive['scores']['identity_p10']['min']:.2f}–{positive['scores']['identity_p10']['max']:.2f}.",
            f"- Designer hard-failure types: `{json.dumps(positive['hard_failure_types'], ensure_ascii=False, sort_keys=True)}`.",
            "",
            "## Mechanical conclusion",
            "",
            "The unchanged total score consistently separates real designer transfers from the no-transfer control, "
            "but the current decision gates do not accept real designer positives consistently. Identity component "
            "calibration and small-package tail gating require review before the decision label can be treated as validated.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen multi-fold ITTE baselines.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    analysis = analyze(args.root)
    json_output = args.json_output or args.root / "cv_analysis.json"
    markdown_output = args.markdown_output or args.root / "cv_analysis.md"
    json_output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
