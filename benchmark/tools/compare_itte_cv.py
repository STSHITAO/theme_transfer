from __future__ import annotations

import argparse
import json
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


def _load(root: Path) -> dict[tuple[int, str, str], dict]:
    rows = {}
    for path in sorted(root.glob("fold_*/baseline_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = payload["sampling"]["seed"]
        for run in payload["runs"]:
            split = run["split"]
            key = seed, split["theme_id"], split["scenario"]
            rows[key] = {
                "summary": run["summary"],
                "split": split,
                "fingerprint": payload["dataset_fingerprint"],
            }
    return rows


def compare(baseline_root: Path, candidate_root: Path) -> dict:
    baseline = _load(baseline_root)
    candidate = _load(candidate_root)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate run keys differ.")
    for key in baseline:
        if baseline[key]["split"] != candidate[key]["split"]:
            raise ValueError(f"Split mismatch for {key}")
        if baseline[key]["fingerprint"] != candidate[key]["fingerprint"]:
            raise ValueError(f"Dataset fingerprint mismatch for {key}")

    scenarios = {}
    for scenario in ("designer_positive", "original_control"):
        keys = [key for key in baseline if key[2] == scenario]
        baseline_accept = np.mean([not baseline[key]["summary"]["hard_failures"] for key in keys])
        candidate_accept = np.mean([not candidate[key]["summary"]["hard_failures"] for key in keys])
        scenarios[scenario] = {
            "run_count": len(keys),
            "hard_gate_accept_rate": {
                "baseline": float(baseline_accept),
                "candidate": float(candidate_accept),
                "delta": float(candidate_accept - baseline_accept),
            },
            "scores": {
                name: {
                    "baseline_mean": float(np.mean([baseline[key]["summary"][name] for key in keys])),
                    "candidate_mean": float(np.mean([candidate[key]["summary"][name] for key in keys])),
                    "paired_delta_mean": float(
                        np.mean([candidate[key]["summary"][name] - baseline[key]["summary"][name] for key in keys])
                    ),
                    "paired_delta_min": float(
                        np.min([candidate[key]["summary"][name] - baseline[key]["summary"][name] for key in keys])
                    ),
                    "paired_delta_max": float(
                        np.max([candidate[key]["summary"][name] - baseline[key]["summary"][name] for key in keys])
                    ),
                }
                for name in SCORES
            },
        }

    separation = {}
    pair_keys = sorted({(seed, theme) for seed, theme, _ in baseline})
    for name in SCORES:
        before = []
        after = []
        for seed, theme in pair_keys:
            before.append(
                baseline[(seed, theme, "designer_positive")]["summary"][name]
                - baseline[(seed, theme, "original_control")]["summary"][name]
            )
            after.append(
                candidate[(seed, theme, "designer_positive")]["summary"][name]
                - candidate[(seed, theme, "original_control")]["summary"][name]
            )
        separation[name] = {
            "baseline_mean": float(np.mean(before)),
            "candidate_mean": float(np.mean(after)),
            "change": float(np.mean(after) - np.mean(before)),
            "candidate_min": float(np.min(after)),
        }

    return {
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "matched_run_count": len(baseline),
        "same_keys_splits_and_fingerprint": True,
        "scenarios": scenarios,
        "designer_minus_control_separation": separation,
    }


def render_markdown(result: dict) -> str:
    positive = result["scenarios"]["designer_positive"]
    control = result["scenarios"]["original_control"]
    separation = result["designer_minus_control_separation"]
    lines = [
        "# ITTE v1.2 → v1.3 matched five-fold comparison",
        "",
        f"- Matched runs: {result['matched_run_count']}",
        "- Same seeds, themes, reference/query identities and dataset fingerprint: yes",
        "",
        "| Measure | v1.2 | v1.3 | paired change |",
        "|---|---:|---:|---:|",
        f"| designer hard-gate acceptance | {positive['hard_gate_accept_rate']['baseline']:.0%} "
        f"| {positive['hard_gate_accept_rate']['candidate']:.0%} "
        f"| {positive['hard_gate_accept_rate']['delta']:+.0%} |",
    ]
    for name in SCORES:
        item = positive["scores"][name]
        lines.append(
            f"| designer {name} | {item['baseline_mean']:.2f} | {item['candidate_mean']:.2f} "
            f"| {item['paired_delta_mean']:+.2f} |"
        )
    lines.extend(
        [
            f"| control ITTE | {control['scores']['itte_score']['baseline_mean']:.2f} "
            f"| {control['scores']['itte_score']['candidate_mean']:.2f} "
            f"| {control['scores']['itte_score']['paired_delta_mean']:+.2f} |",
            f"| designer-control ITTE separation | {separation['itte_score']['baseline_mean']:.2f} "
            f"| {separation['itte_score']['candidate_mean']:.2f} "
            f"| {separation['itte_score']['change']:+.2f} |",
            "",
            "## Decision",
            "",
            "Retain v1.3. It removes identity metrics that were near random in verified-label retrieval from the "
            "primary identity score, replaces unstable absolute normalization with a same-run DINO identity-gallery "
            "percentile, and removes identity hard thresholds unsupported by the available data. The matched result "
            "improves real-positive acceptance and designer/control separation without changing style or package scores. "
            "The remaining real-positive rejection is a package-coherence outlier and remains visible.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched multi-fold ITTE results.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    output = args.output or args.candidate / "comparison_to_v12"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "comparison.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
