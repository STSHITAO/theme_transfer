"""Evaluate the resumable full-generation packages with the current ITTE protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tpqs_workflow import run_tpqs
from scripts.run_full_generation import PACKAGE_PREFIX, THEME_IDS, package_id_for


EVAL_PREFIX = "eval_full_structure_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate full generated theme packages with ITTE.")
    parser.add_argument("--theme-id", action="append", choices=THEME_IDS, help="Limit evaluation to a theme; repeatable.")
    parser.add_argument("--package-prefix", default=PACKAGE_PREFIX)
    parser.add_argument("--eval-prefix", default=EVAL_PREFIX)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    selected = args.theme_id or list(THEME_IDS)
    summaries = []
    evaluated_apps_by_theme = {}
    for theme_id in selected:
        package_id = package_id_for(theme_id, args.package_prefix)
        eval_id = f"{args.eval_prefix}_{theme_id}"
        print(json.dumps({"event": "evaluation_start", "theme_id": theme_id, "eval_id": eval_id}), flush=True)
        report = run_tpqs(theme_id, package_id, eval_id, root_dir=root)["report"]
        evaluated_apps_by_theme[theme_id] = {item["app"] for item in report["per_app"]}
        coverage = report["evaluation_coverage"]
        summary = {
            "theme_id": theme_id,
            "package_id": package_id,
            "eval_id": eval_id,
            "itte_score": report["itte_score"],
            "style_fidelity_score": report["style_fidelity_score"],
            "identity_preservation_score": report["identity_preservation_score"],
            "identity_applicable_app_count": report["identity_preservation"]["applicable_app_count"],
            "identity_skipped_app_count": report["identity_preservation"]["skipped_app_count"],
            "package_coherence_score": report["package_coherence_score"],
            "visual_quality_score": report["visual_quality_score"],
            "evaluated_app_count": coverage["evaluated_app_count"],
            "skipped_app_count": coverage["skipped_app_count"],
            "skipped_apps": coverage["skipped_apps"],
            "coverage_ratio": coverage["ratio"],
            "decision": report["decision"],
        }
        summaries.append(summary)
        print(json.dumps({"event": "evaluation_complete", **summary}, ensure_ascii=False), flush=True)

    output = root / "data" / "evaluations" / f"{args.eval_prefix}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    common_apps = sorted(set.intersection(*evaluated_apps_by_theme.values())) if evaluated_apps_by_theme else []
    output.write_text(
        json.dumps(
            {
                "evaluations": summaries,
                "common_evaluated_app_count": len(common_apps),
                "common_evaluated_apps": common_apps,
                "comparison_note": "Per-theme ITTE uses every successful output; cross-theme comparison should use the common evaluated App intersection.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"event": "run_complete", "summary_path": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
