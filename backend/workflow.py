import os
from pathlib import Path

from dotenv import load_dotenv

from backend.services.image_service import (
    compose_reference_layouts,
    prepare_target_layout,
)
from backend.services.path_service import resolve_case_paths
from backend.services.prompt_service import build_generation_prompt
from backend.services.qc_service import select_best_candidate
from backend.services.qwen_client import analyze_theme, score_candidates
from backend.services.storage_service import save_metadata, save_theme_analysis
from backend.services.wan_client import generate_candidates


def run_workflow(theme_id, target_app, case_id, root_dir=None, candidate_count=3):
    root = Path(root_dir) if root_dir else Path.cwd()
    load_dotenv(root / ".env")

    resolved = resolve_case_paths(theme_id, target_app, root_dir=root)
    reference_examples = compose_reference_layouts(
        resolved["reference_examples"],
        case_id,
        root_dir=root,
    )
    target_layout = prepare_target_layout(
        resolved["target_image"],
        case_id,
        root_dir=root,
    )
    target_inputs = resolved["target_image"]
    target_generation_image = resolved["target_image"]

    theme_analysis = analyze_theme(reference_examples, target_inputs, root_dir=root)
    theme_analysis_path = save_theme_analysis(theme_analysis, case_id, root_dir=root)
    prompt_path = build_generation_prompt(theme_analysis, theme_id, case_id, root_dir=root)

    style_refs = [example["style_ref_path"] for example in reference_examples]
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    generation = generate_candidates(
        prompt_text,
        style_refs,
        target_generation_image,
        case_id,
        root_dir=root,
        n=candidate_count,
    )

    qc_report = score_candidates(
        style_refs,
        target_generation_image,
        generation["candidate_paths"],
        root_dir=root,
    )
    selection = select_best_candidate(
        qc_report,
        generation["candidate_paths"],
        case_id,
        root_dir=root,
    )

    used_reference_examples = [example["app_name"] for example in reference_examples]
    metadata_path = save_metadata(
        case_id,
        {
            "theme_id": theme_id,
            "target_app": target_app,
            "used_reference_examples": used_reference_examples,
            "input_files": {
                "reference_examples": [_metadata_reference_example(example) for example in reference_examples],
                "target_image": resolved["target_image"],
            },
            "intermediate_files": {
                "reference_layouts": [example["reference_layout_path"] for example in reference_examples],
                "target_layout": target_layout,
                "theme_style_analysis": theme_analysis_path,
                "wan_response": generation["wan_response_path"],
            },
            "output_files": {
                "candidates": generation["candidate_paths"],
                "best_output": selection["best_output_path"],
                "qc_report": selection["qc_report_path"],
                "metadata": str(root / "data" / "cases" / case_id / "metadata.json"),
            },
            "model_config": {
                "plan_model": os.getenv("ALI_PLAN_MODEL", ""),
                "image_model": os.getenv("ALI_IMAGE_MODEL", ""),
            },
            "prompt_file": prompt_path,
            "mock_mode": os.getenv("MOCK_MODE", "false").lower() == "true",
        },
        root_dir=root,
    )

    return {
        "case_id": case_id,
        "theme_id": theme_id,
        "target_app": target_app,
        "used_reference_examples": used_reference_examples,
        "reference_layout_paths": [example["reference_layout_path"] for example in reference_examples],
        "target_layout_path": target_layout,
        "theme_style_analysis_path": theme_analysis_path,
        "generation_prompt_path": prompt_path,
        "wan_response_path": generation["wan_response_path"],
        "candidate_paths": generation["candidate_paths"],
        "best_output_path": selection["best_output_path"],
        "qc_report_path": selection["qc_report_path"],
        "metadata_path": metadata_path,
    }


def _metadata_reference_example(example):
    item = {
        "app_name": example["app_name"],
        "style_ref_path": example["style_ref_path"],
    }
    if "original_path" in example:
        item["original_path"] = example["original_path"]
        item["reference_raw_path"] = example.get("reference_raw_path", example["original_path"])
    return item
