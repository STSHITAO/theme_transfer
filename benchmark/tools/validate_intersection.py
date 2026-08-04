from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

from build_intersection import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    THEME_IDS,
    inspect_image,
    sha256_file,
    snapshots_match,
    source_integrity_snapshot,
    utc_now,
    write_json,
)


def read_manifest() -> list[dict[str, str]]:
    path = OUTPUT_ROOT / "intersection_manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def recompute_fingerprint() -> dict:
    core = OUTPUT_ROOT / "core_dataset"
    entries = []
    for path in sorted(item for item in core.rglob("*") if item.is_file()):
        relative = path.relative_to(core).as_posix()
        entries.append({"relative_path": relative, "sha256": sha256_file(path)})
    payload = "".join(f"{item['relative_path']}\t{item['sha256']}\n" for item in entries)
    return {
        "dataset_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def validate() -> dict:
    errors = []
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"Missing built dataset: {OUTPUT_ROOT}")
    manifest = read_manifest()
    valid_rows = [row for row in manifest if row.get("validation_status") == "VALID"]
    if len(valid_rows) != len(manifest):
        errors.append("intersection_manifest.csv must contain only VALID rows; excluded rows belong in excluded_apps.csv")

    core = OUTPUT_ROOT / "core_dataset"
    folders = {"original": core / "originals"}
    folders.update({theme_id: core / theme_id for theme_id in THEME_IDS})
    filename_sets = {}
    for name, folder in folders.items():
        if not folder.exists():
            errors.append(f"Missing core directory: {folder}")
            filename_sets[name] = set()
        else:
            filename_sets[name] = {path.name for path in folder.iterdir() if path.is_file()}
    expected_names = {f"{row['output_id']}.png" for row in valid_rows}
    for name, filenames in filename_sets.items():
        if filenames != expected_names:
            errors.append(f"Filename mismatch in {name}: expected {sorted(expected_names)}, got {sorted(filenames)}")
    if len({frozenset(items) for items in filename_sets.values()}) > 1:
        errors.append("The five core directories do not have identical filename sets")

    canonical_ids = [row.get("canonical_app_id", "") for row in valid_rows]
    output_ids = [row.get("output_id", "") for row in valid_rows]
    if len(canonical_ids) != len(set(canonical_ids)):
        errors.append("Duplicate canonical_app_id in manifest")
    if len(output_ids) != len(set(output_ids)):
        errors.append("Duplicate output_id in manifest")

    for row in valid_rows:
        output_id = row["output_id"]
        source_fields = ["original_path", *(f"{theme_id}_path" for theme_id in THEME_IDS)]
        if any(not row.get(field, "").strip() for field in source_fields):
            errors.append(f"{output_id}: empty source path in manifest")
        for field in source_fields:
            value = row.get(field, "")
            if value and not (PROJECT_ROOT / value).exists():
                errors.append(f"{output_id}: missing source path {value}")

        expected_hashes = {"original": row.get("original_sha256", "")}
        expected_hashes.update({theme_id: row.get(f"{theme_id}_sha256", "") for theme_id in THEME_IDS})
        for name, folder in folders.items():
            path = folder / f"{output_id}.png"
            info = inspect_image(path) if path.exists() else {"valid": False, "issues": ["MISSING_FILE"]}
            if not info["valid"]:
                errors.append(f"{output_id}/{name}: invalid image: {info['issues']}")
            elif sha256_file(path) != expected_hashes[name]:
                errors.append(f"{output_id}/{name}: SHA256 differs from manifest")

        pair_dir = OUTPUT_ROOT / "pairs" / output_id
        expected_pair_names = {"original.png", *(f"{theme_id}.png" for theme_id in THEME_IDS)}
        actual_pair_names = {path.name for path in pair_dir.iterdir() if path.is_file()} if pair_dir.exists() else set()
        if actual_pair_names != expected_pair_names:
            errors.append(f"{output_id}: pair directory must contain exactly five images")
        for name in ("original", *THEME_IDS):
            pair_path = pair_dir / f"{name}.png"
            core_path = folders[name] / f"{output_id}.png"
            if pair_path.exists() and core_path.exists() and sha256_file(pair_path) != sha256_file(core_path):
                errors.append(f"{output_id}/{name}: pairs image differs from core_dataset")

    summary = json.loads((OUTPUT_ROOT / "intersection_summary.json").read_text(encoding="utf-8"))
    if summary.get("valid_app_count") != len(valid_rows):
        errors.append("Manifest VALID count differs from intersection_summary valid_app_count")

    stored_fingerprint = json.loads((OUTPUT_ROOT / "core_dataset_fingerprint.json").read_text(encoding="utf-8"))
    current_fingerprint = recompute_fingerprint()
    if stored_fingerprint.get("dataset_fingerprint") != current_fingerprint["dataset_fingerprint"]:
        errors.append("Dataset fingerprint is not reproducible")
    if stored_fingerprint.get("file_count") != current_fingerprint["file_count"]:
        errors.append("Dataset fingerprint file_count differs")

    before = json.loads((OUTPUT_ROOT / "source_integrity_before.json").read_text(encoding="utf-8"))
    after = json.loads((OUTPUT_ROOT / "source_integrity_after.json").read_text(encoding="utf-8"))
    current = source_integrity_snapshot()
    if not snapshots_match(before, after):
        errors.append("Stored source before/after snapshots differ")
    if not snapshots_match(before, current):
        errors.append("Current source data differs from pre-build snapshot")
    if after.get("source_data_modified") is not False:
        errors.append("source_integrity_after.json does not confirm source_data_modified=false")

    report = {
        "validated_at": utc_now(),
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "valid_app_count": len(valid_rows),
        "core_file_count": current_fingerprint["file_count"],
        "dataset_fingerprint": current_fingerprint["dataset_fingerprint"],
        "source_data_modified": not snapshots_match(before, current),
    }
    write_json(OUTPUT_ROOT / "validation_report.json", report)
    return report


def main() -> int:
    try:
        report = validate()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
