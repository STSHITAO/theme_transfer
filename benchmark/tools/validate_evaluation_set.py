from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "evaluation_set_v1"
THEME_IDS = ("theme_001", "theme_002", "theme_003", "theme_004")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {
                "valid": image.width > 0 and image.height > 0,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "sha256": sha256_file(path),
                "issues": [],
            }
    except Exception as exc:
        return {"valid": False, "issues": [f"{type(exc).__name__}: {exc}"]}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def current_fingerprint() -> dict:
    asset_root = OUTPUT_ROOT / "assets"
    entries = []
    for path in sorted(item for item in asset_root.rglob("*") if item.is_file()):
        relative = path.relative_to(asset_root).as_posix()
        entries.append({"relative_path": relative, "sha256": sha256_file(path)})
    payload = "".join(f"{item['relative_path']}\t{item['sha256']}\n" for item in entries)
    return {
        "dataset_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def validate() -> dict:
    errors: list[str] = []
    required = (
        "identity_gallery.csv",
        "theme_assets_manifest.csv",
        "pair_manifest.csv",
        "excluded_pairs.csv",
        "package_pair_manifest.csv",
        "four_theme_core.csv",
        "app_catalog.csv",
        "theme_descriptions.json",
        "benchmark_protocol.json",
        "evaluation_set_summary.json",
        "evaluation_set_fingerprint.json",
        "GENERATION_AND_EVALUATION_LINEAGE.md",
        "README.md",
    )
    for name in required:
        if not (OUTPUT_ROOT / name).exists():
            errors.append(f"Missing required output: {name}")

    gallery = read_csv(OUTPUT_ROOT / "identity_gallery.csv")
    theme_assets = read_csv(OUTPUT_ROOT / "theme_assets_manifest.csv")
    pairs = read_csv(OUTPUT_ROOT / "pair_manifest.csv")
    package_pairs = read_csv(OUTPUT_ROOT / "package_pair_manifest.csv")
    four_theme_core = read_csv(OUTPUT_ROOT / "four_theme_core.csv")
    summary = read_json(OUTPUT_ROOT / "evaluation_set_summary.json")
    fingerprint = read_json(OUTPUT_ROOT / "evaluation_set_fingerprint.json")
    descriptions = read_json(OUTPUT_ROOT / "theme_descriptions.json")
    protocol = read_json(OUTPUT_ROOT / "benchmark_protocol.json")

    def duplicates(values):
        counts = Counter(values)
        return sorted(value for value, count in counts.items() if count > 1)

    if duplicates(row["canonical_app_id"] for row in gallery):
        errors.append("identity_gallery has duplicate canonical_app_id")
    if duplicates(row["app_slug"] for row in gallery):
        errors.append("identity_gallery has duplicate app_slug")
    if duplicates((row["theme_id"], row["app_slug"]) for row in theme_assets):
        errors.append("theme_assets_manifest has duplicate theme/app labels")
    if duplicates(row["pair_id"] for row in pairs):
        errors.append("pair_manifest has duplicate pair_id")
    if duplicates((row["theme_id"], row["canonical_app_id"]) for row in pairs):
        errors.append("pair_manifest has duplicate theme/app pairs")
    if duplicates(row["package_pair_id"] for row in package_pairs):
        errors.append("package_pair_manifest has duplicate package_pair_id")
    if duplicates(row["canonical_app_id"] for row in four_theme_core):
        errors.append("four_theme_core has duplicate canonical_app_id")

    gallery_by_id = {row["canonical_app_id"]: row for row in gallery}
    theme_by_key = {(row["theme_id"], row["canonical_app_id"]): row for row in theme_assets}
    validated_asset_paths: set[str] = set()

    for row in [*gallery, *theme_assets]:
        path = PROJECT_ROOT / row["asset_path"]
        if not path.exists():
            errors.append(f"Missing asset: {row['asset_path']}")
            continue
        info = inspect_image(path)
        if not info["valid"]:
            errors.append(f"Invalid asset {row['asset_path']}: {info['issues']}")
        elif info["sha256"] != row["sha256"]:
            errors.append(f"Asset SHA256 mismatch: {row['asset_path']}")
        validated_asset_paths.add(row["asset_path"])

    for row in pairs:
        if row.get("validation_status") != "VALID":
            errors.append(f"Non-VALID row in pair_manifest: {row['pair_id']}")
        gallery_row = gallery_by_id.get(row["canonical_app_id"])
        theme_row = theme_by_key.get((row["theme_id"], row["canonical_app_id"]))
        if not gallery_row:
            errors.append(f"Pair missing identity gallery entry: {row['pair_id']}")
        elif gallery_row["asset_path"] != row["original_asset_path"]:
            errors.append(f"Pair original path differs from gallery: {row['pair_id']}")
        if not theme_row:
            errors.append(f"Pair missing themed asset entry: {row['pair_id']}")
        elif theme_row["asset_path"] != row["themed_asset_path"]:
            errors.append(f"Pair themed path differs from theme manifest: {row['pair_id']}")
        for path_field, hash_field in (
            ("original_asset_path", "original_sha256"),
            ("themed_asset_path", "themed_sha256"),
        ):
            path = PROJECT_ROOT / row[path_field]
            if not path.exists() or sha256_file(path) != row[hash_field]:
                errors.append(f"Pair asset/hash mismatch: {row['pair_id']} {path_field}")

    for row in package_pairs:
        left = theme_by_key.get((row["theme_a"], row["canonical_app_id"]))
        right = theme_by_key.get((row["theme_b"], row["canonical_app_id"]))
        if not left or not right:
            errors.append(f"Package pair does not resolve to one app: {row['package_pair_id']}")
            continue
        if left["asset_path"] != row["theme_a_asset_path"] or right["asset_path"] != row["theme_b_asset_path"]:
            errors.append(f"Package pair path mismatch: {row['package_pair_id']}")
        if sha256_file(PROJECT_ROOT / row["theme_a_asset_path"]) != row["theme_a_sha256"]:
            errors.append(f"Package pair theme_a hash mismatch: {row['package_pair_id']}")
        if sha256_file(PROJECT_ROOT / row["theme_b_asset_path"]) != row["theme_b_sha256"]:
            errors.append(f"Package pair theme_b hash mismatch: {row['package_pair_id']}")

    for row in four_theme_core:
        canonical_id = row["canonical_app_id"]
        if canonical_id not in gallery_by_id:
            errors.append(f"Four-theme core app missing from identity gallery: {canonical_id}")
        for theme_id in THEME_IDS:
            theme_row = theme_by_key.get((theme_id, canonical_id))
            if not theme_row or theme_row["asset_path"] != row[f"{theme_id}_asset_path"]:
                errors.append(f"Four-theme core mapping mismatch: {canonical_id} {theme_id}")

    pair_counts = Counter(row["theme_id"] for row in pairs)
    asset_counts = Counter(row["theme_id"] for row in theme_assets)
    if len(gallery) != summary.get("identity_gallery_count"):
        errors.append("identity gallery count differs from summary")
    if len(theme_assets) != summary.get("theme_asset_count"):
        errors.append("theme asset count differs from summary")
    if len(pairs) != summary.get("pair_record_count"):
        errors.append("pair record count differs from summary")
    if len(package_pairs) != summary.get("package_pair_record_count"):
        errors.append("package pair count differs from summary")
    for theme_id in THEME_IDS:
        if pair_counts[theme_id] != summary["pair_counts_by_theme"][theme_id]:
            errors.append(f"pair count differs for {theme_id}")
        if asset_counts[theme_id] != summary["theme_asset_counts"][theme_id]:
            errors.append(f"theme asset count differs for {theme_id}")

    actual_assets = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (OUTPUT_ROOT / "assets").rglob("*")
        if path.is_file()
    }
    if actual_assets != validated_asset_paths:
        errors.append("Physical asset set differs from gallery/theme manifests")

    current = current_fingerprint()
    if current["dataset_fingerprint"] != fingerprint.get("dataset_fingerprint"):
        errors.append("Evaluation-set fingerprint is not reproducible")
    if current["file_count"] != fingerprint.get("file_count"):
        errors.append("Evaluation-set fingerprint file count differs")

    described_themes = set(descriptions.get("themes", {}))
    if described_themes != set(THEME_IDS):
        errors.append("theme_descriptions does not describe exactly four benchmark themes")
    if descriptions.get("scoring_role") != "metadata_only":
        errors.append("theme descriptions must remain metadata_only")
    if protocol.get("primary_scoring", {}).get("uses_text_descriptions") is not False:
        errors.append("primary benchmark scoring must remain image-only")

    report = {
        "validated_at": utc_now(),
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "identity_gallery_count": len(gallery),
        "theme_asset_count": len(theme_assets),
        "pair_record_count": len(pairs),
        "package_pair_record_count": len(package_pairs),
        "four_theme_core_count": len(four_theme_core),
        "asset_file_count": current["file_count"],
        "dataset_fingerprint": current["dataset_fingerprint"],
        "self_contained": True,
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
