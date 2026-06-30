import shutil
from pathlib import Path

from backend.services.storage_service import save_qc_report


def select_best_candidate(qc_report, candidate_paths, case_id, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    output_path = root / "data" / "outputs" / case_id / "best_output.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_path = _best_scored_candidate(qc_report, candidate_paths)
    if not best_path:
        if not candidate_paths:
            raise ValueError("No candidate images available for QC selection.")
        best_path = candidate_paths[0]
        warning = qc_report.get("warning", "")
        qc_report["warning"] = (warning + " " if warning else "") + "No valid overall_score found; selected first candidate."

    qc_report["best_candidate"] = best_path
    shutil.copyfile(best_path, output_path)
    qc_report_path = save_qc_report(qc_report, case_id, root_dir=root)
    return {
        "best_output_path": str(output_path),
        "qc_report_path": qc_report_path,
        "qc_report": qc_report,
    }


def _best_scored_candidate(qc_report, candidate_paths):
    candidate_by_name = {Path(path).name: path for path in candidate_paths}
    candidate_by_path = {path: path for path in candidate_paths}
    best = None
    best_score = None
    for item in qc_report.get("candidates", []):
        try:
            score = float(item.get("overall_score"))
        except (TypeError, ValueError):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = _resolve_candidate_path(item.get("file"), candidate_by_name, candidate_by_path)
    return best


def _resolve_candidate_path(file_value, candidate_by_name, candidate_by_path):
    if not file_value:
        return None
    if file_value in candidate_by_path:
        return file_value
    return candidate_by_name.get(Path(file_value).name)
