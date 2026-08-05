import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from backend.services.image_service import compose_contact_sheet, prepare_target_layout
from backend.services.package_qc_service import run_package_qc
from backend.services.path_service import resolve_target_inputs, resolve_theme_examples
from backend.services.profile_service import load_target_profile, load_theme_profile
from backend.services.prompt_service import build_generation_base_prompt, build_package_target_prompt
from backend.services.qwen_client import (
    analyze_target_identity,
    analyze_theme_design,
    analyze_theme_package,
    build_identity_strategy,
    build_transfer_plan,
    score_candidates,
)
from backend.services.storage_service import save_json
from backend.services.wan_client import WanApiError, generate_candidates


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


def run_package_workflow(
    theme_id,
    package_id,
    root_dir=None,
    candidate_count=3,
    target_app_ids=None,
    resume=False,
    skip_rejected_cases=False,
):
    root = Path(root_dir) if root_dir else Path.cwd()
    load_dotenv(root / ".env")
    package_dir = root / "data" / "packages" / package_id
    package_dir.mkdir(parents=True, exist_ok=True)

    reference_examples = resolve_theme_examples(theme_id, root_dir=root)
    theme_profile = load_theme_profile(theme_id, root_dir=root)
    style_refs = [example["style_ref_path"] for example in reference_examples]
    available_target_apps = scan_target_apps(root_dir=root)
    if target_app_ids is None:
        target_apps = available_target_apps
    else:
        requested = sorted(set(target_app_ids))
        unknown = sorted(set(requested) - set(available_target_apps))
        if unknown:
            raise ValueError("Unknown or invalid target apps: " + ", ".join(unknown))
        target_apps = requested
    if not target_apps:
        raise ValueError("No target apps selected for package generation.")
    save_json(target_apps, package_dir / "target_apps.json")

    theme_analysis = analyze_theme_package(reference_examples, root_dir=root)
    theme_analysis_path = save_json(theme_analysis, package_dir / "theme_style_analysis.json")
    theme_rules_path = save_json(theme_analysis, package_dir / "theme_rules.json")
    theme_design_analysis = analyze_theme_design(reference_examples, theme_profile, root_dir=root)
    theme_design_analysis_path = save_json(theme_design_analysis, package_dir / "theme_design_analysis.json")
    generation_base_prompt_path = build_generation_base_prompt(
        theme_analysis,
        theme_id,
        package_dir / "generation_base_prompt.txt",
        root_dir=root,
        theme_design_analysis=theme_design_analysis,
    )
    generation_base_prompt = Path(generation_base_prompt_path).read_text(encoding="utf-8")

    selected_outputs = {}
    cases = {}
    failed_cases = {}
    for case_index, target_app in enumerate(target_apps, start=1):
        case_result = (
            _load_completed_case(target_app, package_dir, root, candidate_count)
            if resume
            else None
        )
        if case_result is None:
            try:
                case_result = _run_package_case(
                    target_app,
                    package_id,
                    package_dir,
                    generation_base_prompt,
                    theme_analysis,
                    theme_design_analysis,
                    style_refs,
                    root,
                    candidate_count,
                )
            except WanApiError as exc:
                if not skip_rejected_cases or exc.code != "DataInspectionFailed":
                    raise
                failure = _record_case_failure(package_dir, target_app, exc)
                failed_cases[target_app] = failure
                print(
                    json.dumps(
                        {
                            "event": "case_skipped",
                            "package_id": package_id,
                            "theme_id": theme_id,
                            "app_id": target_app,
                            "case_index": case_index,
                            "case_total": len(target_apps),
                            "reason": exc.code,
                        }
                    ),
                    flush=True,
                )
                continue
        _clear_case_failure(package_dir, target_app)
        cases[target_app] = case_result
        selected_outputs[target_app] = case_result["best_output_path"]
        print(
            json.dumps(
                {
                    "event": "case_complete",
                    "package_id": package_id,
                    "theme_id": theme_id,
                    "app_id": target_app,
                    "case_index": case_index,
                    "case_total": len(target_apps),
                    "candidate_count": len(case_result["candidate_paths"]),
                    "resumed": bool(case_result.get("resumed")),
                }
            ),
            flush=True,
        )

    if not selected_outputs:
        raise RuntimeError(f"No successful outputs were generated for package: {package_id}")
    successful_apps = [app_name for app_name in target_apps if app_name in selected_outputs]
    failures_path = save_json(failed_cases, package_dir / "package_failures.json")
    contact_sheet_path = compose_contact_sheet(
        [selected_outputs[app_name] for app_name in successful_apps],
        package_dir / "contact_sheet.png",
    )
    package_qc = run_package_qc(
        style_refs,
        contact_sheet_path,
        selected_outputs,
        package_dir,
        root_dir=root,
    )
    final_outputs = _publish_final_outputs(package_dir, selected_outputs)
    for target_app, final_output_path in final_outputs.items():
        cases[target_app]["final_output_path"] = final_output_path
    metadata_path = _save_package_metadata(
        package_dir,
        package_id,
        theme_id,
        target_apps,
        reference_examples,
        theme_analysis_path,
        theme_rules_path,
        theme_design_analysis_path,
        generation_base_prompt_path,
        final_outputs,
        contact_sheet_path,
        package_qc["package_qc_report_path"],
        cases,
        failed_cases,
        failures_path,
    )

    return {
        "package_id": package_id,
        "theme_id": theme_id,
        "package_dir": str(package_dir),
        "target_apps": target_apps,
        "theme_style_analysis_path": theme_analysis_path,
        "theme_rules_path": theme_rules_path,
        "theme_design_analysis_path": theme_design_analysis_path,
        "generation_base_prompt_path": generation_base_prompt_path,
        "final_outputs": final_outputs,
        "contact_sheet_path": contact_sheet_path,
        "package_qc_report_path": package_qc["package_qc_report_path"],
        "metadata_path": metadata_path,
        "cases": cases,
        "failed_cases": failed_cases,
        "failures_path": failures_path,
        "coverage": {
            "requested_app_count": len(target_apps),
            "successful_app_count": len(successful_apps),
            "skipped_app_count": len(failed_cases),
            "skipped_apps": sorted(failed_cases),
            "ratio": len(successful_apps) / len(target_apps),
        },
    }


def _record_case_failure(package_dir, target_app, exc):
    failure = {
        "app_id": target_app,
        "error_type": type(exc).__name__,
        "status_code": exc.status_code,
        "code": exc.code,
        "message": str(exc),
        "wan_response_path": exc.response_path,
        "skipped_for_batch_continuation": True,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(failure, Path(package_dir) / "cases" / target_app / "case_failure.json")
    return failure


def _clear_case_failure(package_dir, target_app):
    path = Path(package_dir) / "cases" / target_app / "case_failure.json"
    if path.exists():
        path.unlink()


def _load_completed_case(target_app, package_dir, root, expected_candidate_count):
    case_dir = Path(package_dir) / "cases" / target_app
    required = {
        "target_layout_path": case_dir / "target_layout.png",
        "target_identity_path": case_dir / "target_identity.json",
        "identity_strategy_path": case_dir / "identity_strategy.json",
        "transfer_plan_path": case_dir / "transfer_plan.json",
        "generation_prompt_path": case_dir / "generation_prompt.txt",
        "wan_response_path": case_dir / "wan_response.json",
        "best_output_path": case_dir / "best_output.png",
        "qc_report_path": case_dir / "qc_report.json",
    }
    if not all(path.exists() and path.is_file() for path in required.values()):
        return None
    candidate_paths = sorted((case_dir / "candidates").glob("candidate_*.png"))
    if len(candidate_paths) < expected_candidate_count:
        return None
    try:
        transfer_plan = json.loads(required["transfer_plan_path"].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    mode = transfer_plan.get("structure_preservation_mode")
    applicable = transfer_plan.get("structure_identity_metric_applicable")
    if mode not in {"preserve_major_structure", "semantic_recompose"}:
        return None
    if not isinstance(applicable, bool) or applicable != (mode == "preserve_major_structure"):
        return None

    target_inputs = resolve_target_inputs(target_app, root_dir=root)
    return {
        "target_app": target_app,
        "target_image": target_inputs["target_image"],
        "target_layout_path": str(required["target_layout_path"]),
        "target_identity_path": str(required["target_identity_path"]),
        "target_profile": load_target_profile(target_app, root_dir=root),
        "identity_strategy_path": str(required["identity_strategy_path"]),
        "transfer_plan_path": str(required["transfer_plan_path"]),
        "structure_policy": {
            "structure_preservation_mode": mode,
            "structure_identity_metric_applicable": applicable,
            "structure_policy_rationale": transfer_plan.get("structure_policy_rationale", ""),
        },
        "generation_prompt_path": str(required["generation_prompt_path"]),
        "candidate_paths": [str(path) for path in candidate_paths],
        "wan_response_path": str(required["wan_response_path"]),
        "best_output_path": str(required["best_output_path"]),
        "qc_report_path": str(required["qc_report_path"]),
        "resumed": True,
    }


def _run_package_case(
    target_app,
    package_id,
    package_dir,
    generation_base_prompt,
    theme_rules,
    theme_design_analysis,
    style_refs,
    root,
    candidate_count,
):
    target_inputs = resolve_target_inputs(target_app, root_dir=root)
    target_profile = load_target_profile(target_app, root_dir=root)
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
    identity_strategy = build_identity_strategy(
        theme_design_analysis,
        theme_rules,
        target_profile,
        target_image,
        root_dir=root,
    )
    identity_strategy_path = save_json(identity_strategy, case_dir / "identity_strategy.json")
    transfer_plan = build_transfer_plan(
        theme_rules,
        target_identity,
        root_dir=root,
        theme_design_analysis=theme_design_analysis,
        target_profile=target_profile,
        identity_strategy=identity_strategy,
    )
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
        transfer_plan=transfer_plan,
    )
    best_output_path = _select_and_save_best_candidate(qc_report, generation["candidate_paths"], case_dir)

    return {
        "target_app": target_app,
        "target_image": target_image,
        "target_layout_path": target_layout,
        "target_identity_path": target_identity_path,
        "target_profile": target_profile,
        "identity_strategy_path": identity_strategy_path,
        "transfer_plan_path": transfer_plan_path,
        "structure_policy": {
            "structure_preservation_mode": transfer_plan["structure_preservation_mode"],
            "structure_identity_metric_applicable": transfer_plan["structure_identity_metric_applicable"],
            "structure_policy_rationale": transfer_plan["structure_policy_rationale"],
        },
        "generation_prompt_path": generation_prompt_path,
        "candidate_paths": generation["candidate_paths"],
        "wan_response_path": generation["wan_response_path"],
        "best_output_path": str(best_output_path),
        "qc_report_path": str(case_dir / "qc_report.json"),
    }


def _publish_final_outputs(package_dir, selected_outputs):
    package_dir = Path(package_dir)
    final_dir = package_dir / "final"
    if final_dir.exists() and not final_dir.is_dir():
        raise NotADirectoryError(f"Package final output path is not a directory: {final_dir}")

    staging_dir = Path(tempfile.mkdtemp(prefix=".final_staging_", dir=package_dir))
    backup_dir = Path(tempfile.mkdtemp(prefix=".final_backup_", dir=package_dir))
    backup_dir.rmdir()
    moved_existing_final = False

    try:
        for app_name, source_path in selected_outputs.items():
            shutil.copyfile(source_path, staging_dir / f"{app_name}.png")

        if final_dir.exists():
            final_dir.rename(backup_dir)
            moved_existing_final = True

        try:
            staging_dir.rename(final_dir)
        except Exception:
            if moved_existing_final and backup_dir.exists() and not final_dir.exists():
                backup_dir.rename(final_dir)
            raise
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    return {
        app_name: str(final_dir / f"{app_name}.png")
        for app_name in selected_outputs
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


def _best_scored_candidate(qc_report, candidate_paths, identity_threshold=75, artifact_threshold=60, over_recompose_risk_limit=70):
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
        identity_value = item.get("target_recognition_score", item.get("target_identity_score"))
        constraint_value = item.get("identity_constraint_score", identity_value)
        artifact_value = item.get("artifact_score", 100)
        risk_value = item.get("over_recompose_risk", 0)
        try:
            identity_score = float(identity_value)
            constraint_score = float(constraint_value)
            artifact_score = float(artifact_value)
            over_recompose_risk = float(risk_value)
        except (TypeError, ValueError):
            identity_score = 0
            constraint_score = 0
            artifact_score = 0
            over_recompose_risk = 100
        if identity_score < identity_threshold or constraint_score < identity_threshold:
            continue
        if artifact_score < artifact_threshold or over_recompose_risk > over_recompose_risk_limit:
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
    theme_design_analysis_path,
    generation_base_prompt_path,
    final_outputs,
    contact_sheet_path,
    package_qc_report_path,
    cases,
    failed_cases,
    failures_path,
):
    metadata = {
        "package_id": package_id,
        "theme_id": theme_id,
        "target_apps": target_apps,
        "used_reference_examples": [example["app_name"] for example in reference_examples],
        "theme_style_analysis": theme_analysis_path,
        "theme_rules": theme_rules_path,
        "theme_design_analysis": theme_design_analysis_path,
        "generation_base_prompt": generation_base_prompt_path,
        "cases": cases,
        "status": "complete_with_skips" if failed_cases else "complete",
        "failed_cases": failed_cases,
        "package_failures": failures_path,
        "evaluation_coverage": {
            "requested_app_count": len(target_apps),
            "evaluated_app_count": len(final_outputs),
            "skipped_app_count": len(failed_cases),
            "skipped_apps": sorted(failed_cases),
            "ratio": len(final_outputs) / len(target_apps),
        },
        "structure_evaluation_policy": {
            app: case.get("structure_policy", {})
            for app, case in sorted(cases.items())
        },
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
