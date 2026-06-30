from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.workflow import run_workflow


THEME_ID = "theme_001"
TARGET_APP = "bilibili"
CASE_ID = "case_001_theme_001_to_bilibili"


def main():
    result = run_workflow(THEME_ID, TARGET_APP, CASE_ID)
    print("used_reference_examples:")
    for app_name in result["used_reference_examples"]:
        print(f"- {app_name}")

    print("\nreference_layout paths:")
    for path in result["reference_layout_paths"]:
        print(f"- {path}")

    print(f"\ntarget_layout path: {result['target_layout_path']}")
    print(f"theme_style_analysis path: {result['theme_style_analysis_path']}")
    print(f"generation_prompt path: {result['generation_prompt_path']}")

    print("\ncandidate paths:")
    for path in result["candidate_paths"]:
        print(f"- {path}")

    print(f"\nbest_output path: {result['best_output_path']}")
    print(f"qc_report path: {result['qc_report_path']}")
    print(f"metadata path: {result['metadata_path']}")


if __name__ == "__main__":
    main()
