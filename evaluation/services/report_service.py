from __future__ import annotations

import csv
import json
from pathlib import Path


def write_tpqs_outputs(
    eval_dir: Path,
    report: dict,
    per_app_rows: list[dict],
    style_pairwise: dict,
    style_delta: dict,
    dino_pairwise: dict,
) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    itte_report_path = eval_dir / "itte_report.json"
    report_path = eval_dir / "tpqs_report.json"
    metrics_path = eval_dir / "metrics.csv"
    style_pairwise_path = eval_dir / "style_pairwise_distances.json"
    style_delta_path = eval_dir / "style_delta_distances.json"
    dino_pairwise_path = eval_dir / "dino_pairwise_distances.json"

    serialized_report = json.dumps(report, ensure_ascii=False, indent=2)
    itte_report_path.write_text(serialized_report, encoding="utf-8")
    report_path.write_text(serialized_report, encoding="utf-8")
    style_pairwise_path.write_text(json.dumps(style_pairwise, ensure_ascii=False, indent=2), encoding="utf-8")
    style_delta_path.write_text(json.dumps(style_delta, ensure_ascii=False, indent=2), encoding="utf-8")
    dino_pairwise_path.write_text(json.dumps(dino_pairwise, ensure_ascii=False, indent=2), encoding="utf-8")

    preferred = [
        "app",
        "itte_score",
        "style_fidelity_score",
        "identity_preservation_score",
        "package_membership_score",
        "visual_quality_score",
        "is_package_outlier",
        "quality_warnings",
        "identity_components",
    ]
    discovered = {key for row in per_app_rows for key in row}
    fieldnames = [name for name in preferred if name in discovered]
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_app_rows:
            writer.writerow(row)

    return {
        "itte_report_path": str(itte_report_path),
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "style_pairwise_path": str(style_pairwise_path),
        "style_delta_path": str(style_delta_path),
        "dino_pairwise_path": str(dino_pairwise_path),
    }
