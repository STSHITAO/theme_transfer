from __future__ import annotations

import csv
import json
from pathlib import Path


def write_tpqs_outputs(eval_dir: Path, report: dict, per_app_rows: list[dict], pairwise: dict) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "tpqs_report.json"
    metrics_path = eval_dir / "metrics.csv"
    pairwise_path = eval_dir / "pairwise_distances.json"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pairwise_path.write_text(json.dumps(pairwise, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "app",
        "theme_membership_score",
        "identity_rank",
        "identity_top1_match",
        "nearest_target_app",
        "generated_to_theme_distance",
        "is_style_outlier",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_app_rows:
            writer.writerow(row)

    return {
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "pairwise_path": str(pairwise_path),
    }
