import json
from datetime import datetime, timezone
from pathlib import Path


def save_json(data, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def save_theme_analysis(analysis, case_id, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    return save_json(analysis, root / "data" / "cases" / case_id / "theme_style_analysis.json")


def save_qc_report(qc_report, case_id, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    return save_json(qc_report, root / "data" / "cases" / case_id / "qc_report.json")


def save_metadata(case_id, metadata, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    payload = dict(metadata)
    payload["case_id"] = case_id
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    return save_json(payload, root / "data" / "cases" / case_id / "metadata.json")
