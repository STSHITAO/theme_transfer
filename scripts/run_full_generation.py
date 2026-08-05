"""Run resumable real package generation for all normalized dataset targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.package_workflow import run_package_workflow


THEME_IDS = ("theme_001", "theme_002", "theme_003", "theme_004")
PACKAGE_PREFIX = "package_full_structure_v1"


def package_id_for(theme_id: str, prefix: str = PACKAGE_PREFIX) -> str:
    return f"{prefix}_{theme_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate all target Apps for one or all normalized themes.")
    parser.add_argument("--theme-id", action="append", choices=THEME_IDS, help="Limit generation to a theme; repeatable.")
    parser.add_argument("--candidate-count", type=int, default=1, help="Wan candidates per App (default: 1).")
    parser.add_argument("--package-prefix", default=PACKAGE_PREFIX)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate cases even when complete case artifacts exist.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.candidate_count < 1:
        raise SystemExit("--candidate-count must be at least 1")
    load_dotenv(root / ".env")
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        raise SystemExit("MOCK_MODE=true; refusing to label mock outputs as full real generation.")
    missing = [
        key
        for key in (
            "ALI_PLAN_BASE_URL",
            "ALI_PLAN_MODEL",
            "ALI_PLAN_API_KEY",
            "ALI_IMAGE_BASE_URL",
            "ALI_IMAGE_MODEL",
            "ALI_IMAGE_API_KEY",
        )
        if not os.getenv(key)
    ]
    if missing:
        raise SystemExit("Missing required API configuration: " + ", ".join(missing))

    selected = args.theme_id or list(THEME_IDS)
    summaries = []
    for theme_id in selected:
        package_id = package_id_for(theme_id, args.package_prefix)
        print(json.dumps({"event": "package_start", "theme_id": theme_id, "package_id": package_id}), flush=True)
        result = run_package_workflow(
            theme_id,
            package_id,
            root_dir=root,
            candidate_count=args.candidate_count,
            resume=not args.no_resume,
            skip_rejected_cases=True,
        )
        resumed = sum(1 for case in result["cases"].values() if case.get("resumed"))
        failed_count = len(result["failed_cases"])
        summary = {
            "event": "package_complete",
            "status": "complete_with_skips" if failed_count else "complete",
            "theme_id": theme_id,
            "package_id": package_id,
            "target_count": len(result["target_apps"]),
            "resumed_case_count": resumed,
            "generated_case_count": len(result["cases"]) - resumed,
            "skipped_case_count": failed_count,
            "skipped_apps": sorted(result["failed_cases"]),
            "coverage_ratio": result["coverage"]["ratio"],
            "final_output_count": len(result["final_outputs"]),
            "metadata_path": result["metadata_path"],
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "run_complete", "packages": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
