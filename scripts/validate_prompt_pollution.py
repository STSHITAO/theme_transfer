"""Generate a focused prompt-v2 package and compare it with the frozen polluted baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.package_workflow import run_package_workflow


DEFAULT_APPS = (
    "tmall",
    "iqiyi",
    "iqiyi_lite",
    "qidian",
    "ximalaya",
    "karaoke",
    "toutiao",
    "zhihu",
    "tencent_maps",
    "crossfire_mobile",
)
BASELINE_PACKAGE = "package_full_structure_v1_theme_001"
VALIDATION_PACKAGE = "package_prompt_v2_pollution_theme_001"
REFERENCE_IDENTITY_PATTERN = re.compile(r"Alipay|Bilibili|Douban|支付宝|哔哩哔哩|豆瓣", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", action="append", dest="apps", help="Override focused App list; repeatable.")
    parser.add_argument("--candidate-count", type=int, default=2)
    parser.add_argument("--package-id", default=VALIDATION_PACKAGE)
    parser.add_argument("--report-only", action="store_true", help="Rebuild comparison report without API calls.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    return parser


def summarize_package(package_dir: Path, apps: list[str]) -> dict:
    candidate_count = 0
    low_recognition = 0
    low_identity_constraint = 0
    selected_low_recognition = []
    selected_low_identity_constraint = []
    reference_identity_leak_count = 0
    selected_reference_identity_leak = []
    cases_found = []
    for app_id in apps:
        report_path = package_dir / "cases" / app_id / "qc_report.json"
        if not report_path.exists():
            continue
        cases_found.append(app_id)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_name = Path(report.get("best_candidate", "")).name
        for item in report.get("candidates", []):
            candidate_count += 1
            recognition = item.get("target_recognition_score", item.get("target_identity_score", 0))
            constraint = item.get("identity_constraint_score", recognition)
            is_selected = Path(item.get("file", "")).name == selected_name
            leaked_reference_identity = bool(REFERENCE_IDENTITY_PATTERN.search(item.get("failure_reason", "")))
            if recognition < 60:
                low_recognition += 1
                if is_selected:
                    selected_low_recognition.append(app_id)
            if constraint < 60:
                low_identity_constraint += 1
                if is_selected:
                    selected_low_identity_constraint.append(app_id)
            if leaked_reference_identity:
                reference_identity_leak_count += 1
                if is_selected:
                    selected_reference_identity_leak.append(app_id)
    return {
        "package_id": package_dir.name,
        "requested_apps": apps,
        "cases_found": cases_found,
        "candidate_count": candidate_count,
        "candidate_target_recognition_below_60": low_recognition,
        "candidate_identity_constraint_below_60": low_identity_constraint,
        "selected_target_recognition_below_60": sorted(set(selected_low_recognition)),
        "selected_identity_constraint_below_60": sorted(set(selected_low_identity_constraint)),
        "candidate_reference_identity_leak": reference_identity_leak_count,
        "selected_reference_identity_leak": sorted(set(selected_reference_identity_leak)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    apps = sorted(set(args.apps or DEFAULT_APPS))
    if args.candidate_count < 1:
        raise SystemExit("--candidate-count must be at least 1")
    load_dotenv(root / ".env")
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        raise SystemExit("MOCK_MODE=true; refusing real prompt-pollution validation.")

    baseline = summarize_package(root / "data" / "packages" / BASELINE_PACKAGE, apps)
    print(json.dumps({"event": "baseline_summary", **baseline}, ensure_ascii=False), flush=True)
    validation_dir = root / "data" / "packages" / args.package_id
    if not args.report_only:
        result = run_package_workflow(
            "theme_001",
            args.package_id,
            root_dir=root,
            candidate_count=args.candidate_count,
            target_app_ids=apps,
            resume=True,
            skip_rejected_cases=True,
        )
        validation_dir = Path(result["package_dir"])
    revised = summarize_package(validation_dir, apps)
    report = {
        "protocol": "prompt-v2-only; same model, target images, fixed first-three style references, and candidate count",
        "baseline": baseline,
        "prompt_v2": revised,
    }
    report_path = validation_dir / "prompt_pollution_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "validation_complete", "report": str(report_path), **revised}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
