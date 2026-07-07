from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.package_workflow import run_package_workflow


THEME_ID = "theme_001"
PACKAGE_ID = "package_001_theme_001"


def main():
    result = run_package_workflow(THEME_ID, PACKAGE_ID)

    print(f"package_dir: {result['package_dir']}")

    print("\nfinal output paths:")
    for app_name, path in result["final_outputs"].items():
        print(f"- {app_name}: {path}")

    print(f"\ncontact_sheet path: {result['contact_sheet_path']}")
    print(f"package_qc_report path: {result['package_qc_report_path']}")
    print(f"metadata path: {result['metadata_path']}")


if __name__ == "__main__":
    main()
