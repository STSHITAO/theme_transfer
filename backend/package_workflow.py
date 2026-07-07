import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from backend.services.image_service import compose_contact_sheet, prepare_target_layout
from backend.services.package_qc_service import run_package_qc
from backend.services.path_service import resolve_target_inputs, resolve_theme_examples
from backend.services.prompt_service import build_generation_base_prompt, build_package_target_prompt
from backend.services.qwen_client import analyze_target_identity, analyze_theme_package, build_transfer_plan, score_candidates
from backend.services.storage_service import save_json
from backend.services.wan_client import generate_candidates


def scan_target_apps(root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    targets_dir = root / "data" / "targets"
    if not targets_dir.exists():
        raise FileNotFoundError(f"Missing targets directory: {targets_dir}")

    apps = []
    for target_dir in sorted([item for item in targets_dir.iterdir() if item.is_dir()]):
        try:
            resolve_target_inputs(target_dir.name, root_dir=root)
        except FileNotFoundError:
            continue
        apps.append(target_dir.name)
    if not apps:
        raise ValueError(f"No valid target apps found in: {targets_dir}")
    return apps


def run_package_workflow(theme_id, package_id, root_dir=None, candidate_count=3):
    root = Path(root_dir) if root_dir else Path.cwd()
    load_dotenv(root / ".env")
    package_dir = root / "data" / "packages" / package_id
    package_dir.mkdir(parents=True, exist_ok=True)

    reference_examples = resolve_theme_examples(theme_id, root_dir=root)
    style_refs = [example["style_ref_path"] for example in reference_examples]
    target_apps = scan_target_apps(root_dir=root)
    save_json(target_apps, package_dir / "target_apps.json")

    theme_analysis = analyze_theme_package(reference_examples, root_dir=root)
    theme_analysis_path = save_json(theme_analysis, package_dir / "theme_style_analysis.json")
    theme_rules_path = save_json(theme_analysis, package_dir / "theme_rules.json")
    generation_base_prompt_path = build_generation_base_prompt(
        theme_analysis,
        theme_id,
        package_dir / "generation_base_prompt.txt",
        root_dir=root,
    )
    generation_base_prompt = Path(generation_base_prompt_path).read_text(encoding="utf-8")

    final_outputs = {}
    cases = {}
    for target_app in target_apps:
        case_result = _run_package_case(
            target_app,
            package_id,
            package_dir,
            generation_base_prompt,
            theme_analysis,
            style_refs,
            root,
            candidate_count,
        )
        cases[target_app] = case_result
        final_outputs[target_app] = case_result["final_output_path"]

    contact_sheet_path = compose_contact_sheet(
        [final_outputs[app_name] for app_name in target_apps],
        package_dir / "contact_sheet.png",
    )
    package_qc = run_package_qc(
        style_refs,
        contact_sheet_path,
        final_outputs,
        package_dir,
        root_dir=root,
    )
    metadata_path = _save_package_metadata(
        package_dir,
        package_id,
        theme_id,
        target_apps,
        reference_examples,
        theme_analysis_path,
        theme_rules_path,
        generation_base_prompt_path,
        final_outputs,
        contact_sheet_path,
        package_qc["package_qc_report_path"],
        cases,
    )

    return {
        "package_id": package_id,
        "theme_id": theme_id,
        "package_dir": str(package_dir),
        "target_apps": target_apps,
        "theme_style_analysis_path": theme_analysis_path,
        "theme_rules_path": theme_rules_path,
        "generation_base_prompt_path": generation_base_prompt_path,
        "final_outputs": final_outputs,
        "contact_sheet_path": contact_sheet_path,
        "package_qc_report_path": package_qc["package_qc_report_path"],
        "metadata_path": metadata_path,
        "cases": cases,
    }


def _run_package_case(target_app, package_id, package_dir, generation_base_prompt, theme_rules, style_refs, root, candidate_count):
    target_inputs = resolve_target_inputs(target_app, root_dir=root)
    target_image = target_inputs["target_image"]
    case_dir = package_dir / "cases" / target_app
    candidates_dir = case_dir / "candidates"
    case_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    target_layout = prepare_target_layout(
        target_image,
        f"{package_id}_{target_app}",
        root_dir=root,
        output_path=case_dir / "target_layout.png",
    )
    target_identity = analyze_target_identity(target_app, target_image, root_dir=root)
    target_identity_path = save_json(target_identity, case_dir / "target_identity.json")
    transfer_plan = build_transfer_plan(theme_rules, target_identity, root_dir=root)
    transfer_plan_path = save_json(transfer_plan, case_dir / "transfer_plan.json")
    generation_prompt_path = build_package_target_prompt(
        generation_base_prompt,
        target_app,
        case_dir / "generation_prompt.txt",
        transfer_plan=transfer_plan,
    )
    prompt_text = Path(generation_prompt_path).read_text(encoding="utf-8")
    generation = generate_candidates(
        prompt_text,
        style_refs,
        target_layout,
        f"{package_id}_{target_app}",
        root_dir=root,
        n=candidate_count,
        case_dir=case_dir,
        output_dir=candidates_dir,
    )
    qc_report = score_candidates(
        style_refs,
        target_layout,
        generation["candidate_paths"],
        root_dir=root,
    )
    best_output_path = _select_and_save_best_candidate(qc_report, generation["candidate_paths"], case_dir)
    final_output_path = package_dir / "final" / f"{target_app}.png"
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_output_path, final_output_path)

    return {
        "target_app": target_app,
        "target_image": target_image,
        "target_layout_path": target_layout,
        "target_identity_path": target_identity_path,
        "transfer_plan_path": transfer_plan_path,
        "generation_prompt_path": generation_prompt_path,
        "candidate_paths": generation["candidate_paths"],
        "wan_response_path": generation["wan_response_path"],
        "best_output_path": str(best_output_path),
        "qc_report_path": str(case_dir / "qc_report.json"),
        "final_output_path": str(final_output_path),
    }


def _select_and_save_best_candidate(qc_report, candidate_paths, case_dir):
    best_path = _best_scored_candidate(qc_report, candidate_paths)
    if not best_path:
        if not candidate_paths:
            raise ValueError("No candidate images available for package case QC selection.")
        best_path = candidate_paths[0]
        warning = qc_report.get("warning", "")
        qc_report["warning"] = (warning + " " if warning else "") + "No valid overall_score found; selected first candidate."

    qc_report["best_candidate"] = best_path
    output_path = Path(case_dir) / "best_output.png"
    shutil.copyfile(best_path, output_path)
    save_json(qc_report, Path(case_dir) / "qc_report.json")
    return output_path


def _best_scored_candidate(qc_report, candidate_paths, identity_threshold=75):
    candidate_by_name = {Path(path).name: path for path in candidate_paths}
    candidate_by_path = {path: path for path in candidate_paths}
    best = None
    best_score = None
    fallback = None
    fallback_score = None
    qc_report["needs_retry"] = False
    for item in qc_report.get("candidates", []):
        try:
            score = float(item.get("overall_score"))
        except (TypeError, ValueError):
            continue
        file_value = item.get("file")
        path = candidate_by_path.get(file_value) or candidate_by_name.get(Path(file_value or "").name)
        if fallback_score is None or score > fallback_score:
            fallback_score = score
            fallback = path
        try:
            identity_score = float(item.get("target_identity_score"))
        except (TypeError, ValueError):
            identity_score = 0
        if identity_score < identity_threshold:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = path
    if best:
        return best
    if fallback:
        qc_report["needs_retry"] = True
        warning = qc_report.get("warning", "")
        qc_report["warning"] = (
            (warning + " " if warning else "")
            + f"All candidates are below identity threshold {identity_threshold}; selected highest overall candidate and marked needs_retry=true."
        )
    return fallback


def _save_package_metadata(
    package_dir,
    package_id,
    theme_id,
    target_apps,
    reference_examples,
    theme_analysis_path,
    theme_rules_path,
    generation_base_prompt_path,
    final_outputs,
    contact_sheet_path,
    package_qc_report_path,
    cases,
):
    metadata = {
        "package_id": package_id,
        "theme_id": theme_id,
        "target_apps": target_apps,
        "used_reference_examples": [example["app_name"] for example in reference_examples],
        "theme_style_analysis": theme_analysis_path,
        "theme_rules": theme_rules_path,
        "generation_base_prompt": generation_base_prompt_path,
        "cases": cases,
        "final_outputs": final_outputs,
        "contact_sheet": contact_sheet_path,
        "package_qc_report": package_qc_report_path,
        "model_config": {
            "plan_model": os.getenv("ALI_PLAN_MODEL", ""),
            "image_model": os.getenv("ALI_IMAGE_MODEL", ""),
        },
        "mock_mode": os.getenv("MOCK_MODE", "false").lower() == "true",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return save_json(metadata, Path(package_dir) / "metadata.json")
