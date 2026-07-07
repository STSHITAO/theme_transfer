from pathlib import Path

from backend.services.qwen_client import score_package_consistency
from backend.services.storage_service import save_json


PACKAGE_QC_FIELDS = {
    "package_consistency_score": 0,
    "style_consistency_score": 0,
    "target_identity_score": 0,
    "problematic_apps": [],
    "accepted_apps": [],
    "retry_apps": [],
    "overall_comment": "",
}


def run_package_qc(theme_style_refs, contact_sheet, final_outputs, package_dir, root_dir=None):
    report = score_package_consistency(
        theme_style_refs,
        contact_sheet,
        final_outputs,
        root_dir=root_dir,
    )
    report = _normalize_package_qc_report(report, final_outputs)
    report_path = Path(package_dir) / "package_qc_report.json"
    return {
        "package_qc_report": report,
        "package_qc_report_path": save_json(report, report_path),
    }


def _normalize_package_qc_report(report, final_outputs):
    if not isinstance(report, dict):
        report = {
            "raw_response": str(report),
            "overall_comment": "Package QC returned an unreadable report; fallback report was used.",
        }
    normalized = dict(PACKAGE_QC_FIELDS)
    normalized.update(report)
    if not normalized["accepted_apps"] and not normalized["retry_apps"]:
        normalized["retry_apps"] = _final_output_app_names(final_outputs)
    return normalized


def _final_output_app_names(final_outputs):
    if isinstance(final_outputs, dict):
        return list(final_outputs.keys())
    return [Path(path).stem for path in final_outputs]
