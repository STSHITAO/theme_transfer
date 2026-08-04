from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
OUTPUT_ROOT = BENCHMARK_ROOT / "intersection_v1"
THEME_IDS = ("theme_001", "theme_002", "theme_003", "theme_004")
SOURCE_ROOTS = (
    BENCHMARK_ROOT / "original_icons",
    *(BENCHMARK_ROOT / f"{theme_id}_processed" for theme_id in THEME_IDS),
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
FORMAT_SUFFIXES = {
    "PNG": {".png"},
    "JPEG": {".jpg", ".jpeg"},
    "WEBP": {".webp"},
}
MANIFEST_FIELDS = (
    "output_id",
    "canonical_app_id",
    "app_name",
    "appstore_id",
    "bundle_id",
    "original_path",
    "theme_001_path",
    "theme_002_path",
    "theme_003_path",
    "theme_004_path",
    "original_sha256",
    "theme_001_sha256",
    "theme_002_sha256",
    "theme_003_sha256",
    "theme_004_sha256",
    "match_basis",
    "match_confidence",
    "validation_status",
    "notes",
)
EXCLUDED_FIELDS = (
    "canonical_app_id",
    "raw_app_name",
    "available_in_originals",
    "available_in_theme_001",
    "available_in_theme_002",
    "available_in_theme_003",
    "available_in_theme_004",
    "exclusion_reason",
    "candidate_paths",
    "notes",
)
DUPLICATE_FIELDS = (
    "theme_id",
    "app_name",
    "canonical_app_id",
    "candidate_paths",
    "candidate_sha256",
    "resolution",
    "selected_path",
    "selection_basis",
    "status",
    "notes",
)
ALIAS_FIELDS = (
    "raw_app_name",
    "normalized_app_name",
    "app_slug",
    "canonical_app_id",
    "appstore_id",
    "evidence",
    "status",
    "notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = "".join(char for char in value if unicodedata.category(char) != "Cf")
    return value.strip().lower()


def inspect_image(path: Path) -> dict:
    result = {
        "path": relative_to_project(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "width": 0,
        "height": 0,
        "mode": "",
        "format": "",
        "has_alpha": False,
        "alpha_nonzero_ratio": 1.0,
        "sha256": "",
        "valid": False,
        "issues": [],
    }
    if not path.exists() or not path.is_file():
        result["issues"].append("MISSING_FILE")
        return result
    if result["size_bytes"] <= 0:
        result["issues"].append("EMPTY_FILE")
        return result
    try:
        result["sha256"] = sha256_file(path)
        with Image.open(path) as image:
            image.load()
            result["width"], result["height"] = image.size
            result["mode"] = image.mode
            result["format"] = image.format or ""
            if image.width <= 0 or image.height <= 0:
                result["issues"].append("INVALID_DIMENSIONS")
            result["has_alpha"] = "A" in image.getbands()
            if result["has_alpha"]:
                alpha = image.getchannel("A")
                histogram = alpha.histogram()
                nonzero = sum(histogram[1:])
                result["alpha_nonzero_ratio"] = nonzero / max(image.width * image.height, 1)
                if nonzero == 0:
                    result["issues"].append("FULLY_TRANSPARENT")
            rgba = image.convert("RGBA")
            extrema = rgba.getextrema()
            if all(low == high for low, high in extrema):
                result["issues"].append("PURE_COLOR_IMAGE")
            expected_suffixes = FORMAT_SUFFIXES.get(result["format"].upper())
            if expected_suffixes and path.suffix.lower() not in expected_suffixes:
                result["issues"].append("FORMAT_EXTENSION_MISMATCH")
    except Exception as exc:
        result["issues"].append(f"BROKEN_IMAGE:{type(exc).__name__}:{exc}")
    fatal_prefixes = (
        "MISSING_FILE",
        "EMPTY_FILE",
        "INVALID_DIMENSIONS",
        "FULLY_TRANSPARENT",
        "PURE_COLOR_IMAGE",
        "BROKEN_IMAGE",
    )
    result["valid"] = not any(
        issue.startswith(fatal_prefixes) for issue in result["issues"]
    )
    return result


def source_integrity_snapshot() -> dict:
    files = []
    for source_root in SOURCE_ROOTS:
        if not source_root.exists():
            raise FileNotFoundError(f"Missing required source directory: {source_root}")
        for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": relative_to_project(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = "".join(f"{item['path']}\t{item['sha256']}\n" for item in files)
    return {
        "algorithm": "sha256",
        "created_at": utc_now(),
        "source_roots": [relative_to_project(path) for path in SOURCE_ROOTS],
        "file_count": len(files),
        "aggregate_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "files": files,
    }


def snapshots_match(before: dict, after: dict) -> bool:
    before_files = {(item["path"], item["size_bytes"], item["sha256"]) for item in before["files"]}
    after_files = {(item["path"], item["size_bytes"], item["sha256"]) for item in after["files"]}
    return before_files == after_files


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_appstore_manifest() -> tuple[dict[str, dict], dict]:
    path = BENCHMARK_ROOT / "appstore_originals_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["app_slug"]: item for item in payload.get("apps", [])}, payload


def load_theme_data() -> dict[str, dict]:
    themes = {}
    for theme_id in THEME_IDS:
        theme_dir = BENCHMARK_ROOT / f"{theme_id}_processed"
        mapping_path = theme_dir / f"{theme_id}_mapping_with_originals.csv"
        if not mapping_path.exists():
            mapping_path = theme_dir / f"{theme_id}_mapping.csv"
        rows = read_csv(mapping_path)
        by_slug: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            slug = normalize_name(row.get("app_slug", ""))
            row["_slug"] = slug
            row["_themed_path"] = theme_dir / row.get("themed_icon_path", "")
            by_slug[slug].append(row)
        themed_dir = theme_dir / "themed_icons"
        themed_files = sorted(
            path for path in themed_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        themes[theme_id] = {
            "dir": theme_dir,
            "mapping_path": mapping_path,
            "rows": rows,
            "by_slug": by_slug,
            "themed_files": themed_files,
            "file_by_slug": {normalize_name(path.stem): path for path in themed_files},
        }
    return themes


def appstore_id_from_rows(rows: list[dict]) -> set[str]:
    ids = set()
    for row in rows:
        value = (row.get("apple_id") or "").strip()
        if not value:
            page = row.get("appstore_page") or ""
            marker = "/id"
            if marker in page:
                value = page.rsplit(marker, 1)[1].split("?", 1)[0].split("/", 1)[0]
        if value:
            ids.add(value)
    return ids


def canonical_identity(slug: str, all_rows: list[dict], appstore: dict[str, dict]) -> dict:
    ids = appstore_id_from_rows(all_rows)
    manifest_item = appstore.get(slug, {})
    manifest_id = str(manifest_item.get("apple_id", "")).strip()
    if manifest_id:
        ids.add(manifest_id)
    names = [row.get("app_name", "").strip() for row in all_rows if row.get("app_name", "").strip()]
    app_name = names[0] if names else manifest_item.get("mapping_app_name", slug)
    if len(ids) == 1:
        appstore_id = next(iter(ids))
        return {
            "canonical_app_id": f"appstore:{appstore_id}",
            "appstore_id": appstore_id,
            "bundle_id": manifest_item.get("bundle_id", ""),
            "app_name": app_name,
            "match_basis": "APP_STORE_ID",
            "match_confidence": "high",
            "identity_ambiguous": False,
            "identity_notes": "Explicit App Store ID agrees across available mapping records.",
        }
    if len(ids) > 1:
        return {
            "canonical_app_id": f"ambiguous:{slug}",
            "appstore_id": "|".join(sorted(ids)),
            "bundle_id": "",
            "app_name": app_name,
            "match_basis": "CONFLICTING_APP_STORE_IDS",
            "match_confidence": "none",
            "identity_ambiguous": True,
            "identity_notes": f"Conflicting App Store IDs: {', '.join(sorted(ids))}",
        }
    return {
        "canonical_app_id": f"slug:{slug}",
        "appstore_id": "",
        "bundle_id": "",
        "app_name": app_name,
        "match_basis": "EXPLICIT_MAPPING_SLUG",
        "match_confidence": "medium",
        "identity_ambiguous": False,
        "identity_notes": "No fuzzy matching used; identity comes from the explicit mapping slug.",
    }


def crop_text_suspected(row: dict, image_info: dict) -> bool:
    notes = row.get("notes", "")
    if "不含下方文字" in notes or "不包含下方" in notes:
        return False
    if "包含下方文字" in notes or "包含价格" in notes:
        return True
    width = image_info.get("width", 0)
    height = image_info.get("height", 0)
    return bool(width and height and height > width * 1.15)


def gather_original_candidates(slug: str, themes: dict) -> list[Path]:
    candidates = []
    top = BENCHMARK_ROOT / "original_icons" / f"{slug}.png"
    if top.exists():
        candidates.append(top)
    for theme_id in THEME_IDS:
        for row in themes[theme_id]["by_slug"].get(slug, []):
            relative = (row.get("original_icon_path") or "").strip()
            if relative:
                candidate = themes[theme_id]["dir"] / relative
                if candidate.exists() and candidate not in candidates:
                    candidates.append(candidate)
    return candidates


def duplicate_review_rows(slugs: set[str], identities: dict, themes: dict, image_cache: dict) -> tuple[list[dict], set[str]]:
    output = []
    ambiguous_originals = set()
    for slug in sorted(slugs):
        candidates = gather_original_candidates(slug, themes)
        if len(candidates) <= 1:
            continue
        infos = [image_cache.setdefault(path, inspect_image(path)) for path in candidates]
        hashes = {item["sha256"] for item in infos if item["sha256"]}
        top = BENCHMARK_ROOT / "original_icons" / f"{slug}.png"
        selected = top if top.exists() and image_cache.setdefault(top, inspect_image(top))["valid"] else None
        identity = identities[slug]
        if len(hashes) == 1:
            status = "DUPLICATE_IDENTICAL"
            basis = "SHA256_IDENTICAL;TOP_LEVEL_ORIGINAL_PREFERRED" if selected else "SHA256_IDENTICAL"
            notes = "All available original candidates are byte-identical."
        elif selected and identity["appstore_id"]:
            status = "RESOLVED_VERSION_DIFFERENCE"
            basis = "TOP_LEVEL_APPSTORE_MANIFEST_AND_APP_STORE_ID"
            notes = "Original candidates differ; selected the top-level original backed by the App Store manifest."
        else:
            status = "AMBIGUOUS_ORIGINAL"
            basis = "NONE"
            notes = "Different original candidates exist without sufficient authoritative evidence."
            ambiguous_originals.add(slug)
        output.append(
            {
                "theme_id": "originals",
                "app_name": identity["app_name"],
                "canonical_app_id": identity["canonical_app_id"],
                "candidate_paths": "|".join(relative_to_project(path) for path in candidates),
                "candidate_sha256": "|".join(item["sha256"] for item in infos),
                "resolution": "|".join(f"{item['width']}x{item['height']}" for item in infos),
                "selected_path": relative_to_project(selected) if selected else "",
                "selection_basis": basis,
                "status": status,
                "notes": notes,
            }
        )

    for theme_id in THEME_IDS:
        for slug, rows in sorted(themes[theme_id]["by_slug"].items()):
            paths = sorted({row["_themed_path"] for row in rows})
            if len(rows) <= 1 and len(paths) <= 1:
                continue
            infos = [image_cache.setdefault(path, inspect_image(path)) for path in paths if path.exists()]
            hashes = {item["sha256"] for item in infos if item["sha256"]}
            status = "DUPLICATE_IDENTICAL" if len(hashes) == 1 else "DUPLICATE_CONFLICT"
            output.append(
                {
                    "theme_id": theme_id,
                    "app_name": identities[slug]["app_name"],
                    "canonical_app_id": identities[slug]["canonical_app_id"],
                    "candidate_paths": "|".join(relative_to_project(path) for path in paths),
                    "candidate_sha256": "|".join(item["sha256"] for item in infos),
                    "resolution": "|".join(f"{item['width']}x{item['height']}" for item in infos),
                    "selected_path": relative_to_project(paths[0]) if status == "DUPLICATE_IDENTICAL" and paths else "",
                    "selection_basis": "SHA256_IDENTICAL" if status == "DUPLICATE_IDENTICAL" else "NONE",
                    "status": status,
                    "notes": "Multiple mapping rows or icon paths exist for one explicit app label.",
                }
            )
    return output, ambiguous_originals


def build_source_audit(themes: dict, image_cache: dict) -> dict:
    sources = {}
    original_files = sorted(
        path for path in (BENCHMARK_ROOT / "original_icons").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    original_infos = [image_cache.setdefault(path, inspect_image(path)) for path in original_files]
    sources["original_icons"] = {
        "path": "benchmark/original_icons",
        "image_count": len(original_files),
        "recursive_image_count": len(original_files),
        "mapping_record_count": len(json.loads((BENCHMARK_ROOT / "appstore_originals_manifest.json").read_text(encoding="utf-8")).get("apps", [])),
        "unique_app_count": len({normalize_name(path.stem) for path in original_files}),
        "duplicate_app_count": len(original_files) - len({normalize_name(path.stem) for path in original_files}),
        "duplicate_app_groups": 0,
        "invalid_path_count": 0,
        "unrecognized_app_count": 0,
        "broken_image_count": sum(not item["valid"] for item in original_infos),
        "same_name_different_app_count": 0,
        "multi_icon_app_count": 0,
    }
    for theme_id in THEME_IDS:
        data = themes[theme_id]
        rows = data["rows"]
        themed_files = data["themed_files"]
        recursive_images = sorted(
            path for path in data["dir"].rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        recursive_infos = [image_cache.setdefault(path, inspect_image(path)) for path in recursive_images]
        duplicate_groups = [items for items in data["by_slug"].values() if len(items) > 1]
        invalid_paths = [row for row in rows if not row["_themed_path"].exists()]
        mapped_slugs = {row["_slug"] for row in rows if row["_slug"]}
        unrecognized = [path for path in themed_files if normalize_name(path.stem) not in mapped_slugs]
        name_groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            name_groups[normalize_name(row.get("app_name", ""))].add(row["_slug"])
        same_name_conflicts = [slugs for name, slugs in name_groups.items() if name and len(slugs) > 1]
        multi_icon = [
            slug for slug, items in data["by_slug"].items()
            if len({str(item["_themed_path"].resolve()) for item in items}) > 1
        ]
        hash_to_slugs: dict[str, set[str]] = defaultdict(set)
        for path in themed_files:
            info = image_cache.setdefault(path, inspect_image(path))
            if info["sha256"]:
                hash_to_slugs[info["sha256"]].add(normalize_name(path.stem))
        sources[theme_id] = {
            "path": relative_to_project(data["dir"]),
            "themed_icon_count": len(themed_files),
            "image_count": len(themed_files),
            "recursive_image_count": len(recursive_images),
            "mapping_record_count": len(rows),
            "unique_app_count": len(mapped_slugs),
            "duplicate_app_count": sum(len(group) - 1 for group in duplicate_groups),
            "duplicate_app_groups": len(duplicate_groups),
            "invalid_path_count": len(invalid_paths),
            "unrecognized_app_count": len(unrecognized),
            "broken_image_count": sum(not item["valid"] for item in recursive_infos),
            "same_name_different_app_count": len(same_name_conflicts),
            "multi_icon_app_count": len(multi_icon),
            "different_apps_same_sha256_count": sum(len(slugs) > 1 for slugs in hash_to_slugs.values()),
            "mapping_file": relative_to_project(data["mapping_path"]),
        }
    return {
        "created_at": utc_now(),
        "matching_policy": "App Store ID, bundle_id/package_name, canonical ID, then explicit mapping slug; no fuzzy matching.",
        "sources": sources,
        "image_validation_details": [image_cache[path] for path in sorted(image_cache, key=lambda item: str(item))],
    }


def copy_as_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".png":
        shutil.copy2(source, destination)
        return
    with Image.open(source) as image:
        image.load()
        image.save(destination, format="PNG")


def font_for_contact_sheet(size: int):
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def create_contact_sheet(apps: list[dict], destination: Path) -> None:
    tile = 144
    label_width = 220
    header_height = 52
    row_height = 170
    columns = ("original", *THEME_IDS)
    canvas = Image.new("RGB", (label_width + tile * len(columns), header_height + row_height * len(apps)), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font_for_contact_sheet(20)
    label_font = font_for_contact_sheet(17)
    small_font = font_for_contact_sheet(13)
    draw.text((12, 14), "APP", fill="black", font=header_font)
    for index, column in enumerate(columns):
        draw.text((label_width + index * tile + 10, 14), column, fill="black", font=header_font)
    for row_index, app in enumerate(apps):
        y = header_height + row_index * row_height
        draw.line((0, y, canvas.width, y), fill=(210, 210, 210), width=1)
        draw.text((12, y + 44), app["app_name"], fill="black", font=label_font)
        draw.text((12, y + 72), app["canonical_app_id"], fill=(80, 80, 80), font=small_font)
        paths = [app["original_source"], *(app["theme_sources"][theme_id] for theme_id in THEME_IDS)]
        for col_index, path in enumerate(paths):
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
                rgba.thumbnail((128, 128), Image.Resampling.LANCZOS)
                background = Image.new("RGBA", (128, 128), (245, 245, 245, 255))
                offset = ((128 - rgba.width) // 2, (128 - rgba.height) // 2)
                background.alpha_composite(rgba, offset)
                canvas.paste(background.convert("RGB"), (label_width + col_index * tile + 8, y + 20))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=92, optimize=True)


def fingerprint_core_dataset() -> dict:
    core = OUTPUT_ROOT / "core_dataset"
    entries = []
    for path in sorted(item for item in core.rglob("*") if item.is_file()):
        relative = path.relative_to(core).as_posix()
        entries.append({"relative_path": relative, "sha256": sha256_file(path)})
    payload = "".join(f"{item['relative_path']}\t{item['sha256']}\n" for item in entries)
    return {
        "algorithm": "sha256(relative_path + TAB + file_sha256 + LF)",
        "dataset_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "file_count": len(entries),
        "app_count": len(entries) // 5,
        "theme_count": 4,
        "created_at": utc_now(),
        "files": entries,
    }


def render_readme(summary: dict, fingerprint: dict, human_review: list[str]) -> str:
    return f"""# ITTE Benchmark strict five-way intersection v1

## Purpose

This dataset is the identity-aligned core used later to calibrate ITTE style fidelity, identity preservation, package coherence, visual quality, dimension weights, total weights, and decision thresholds. This build does not run ITTE or modify any scoring code.

## Inclusion rule

The set is the strict intersection of `original_icons`, `theme_001`, `theme_002`, `theme_003`, and `theme_004`. Every included app has exactly five readable and validated images. Matching priority is App Store ID, bundle/package ID, confirmed canonical ID, explicit mapping identity, then an evidence-backed alias. No fuzzy string matching is used.

Original selection prefers the top-level App Store-backed original, then a readable higher-resolution candidate with equally explicit identity evidence. Conflicting originals without decisive evidence are excluded. System apps are included only when a reliable original exists; generic substitute icons are forbidden. Byte-identical duplicates are deduplicated, while conflicting duplicates are excluded unless explicit evidence resolves them.

## Exclusion policy

An app is excluded when any of the five images is missing, identity is ambiguous, an original is ambiguous, a duplicate conflict is unresolved, a system original is unconfirmed, an image is broken/empty/transparent/uniform, or the themed crop is suspected to include a label/price. Every excluded app is listed in `excluded_apps.csv`.

## Result

- Final valid apps: {summary['valid_app_count']}
- Strict five-way intersection before validation gates: {summary['strict_five_way_intersection_count']}
- Four-theme intersection: {summary['four_theme_intersection_count']}
- Dataset fingerprint: `{fingerprint['dataset_fingerprint']}`

## Structure

`core_dataset/` contains five directories with identical filenames. `pairs/` groups exactly five images per app for manual identity review. `intersection_manifest.csv` preserves app identity and source paths. `contact_sheets/` is only for review and is not formal benchmark input.

## Build and validate

```powershell
python benchmark/tools/build_intersection.py
python benchmark/tools/validate_intersection.py
```

The builder refuses to overwrite `intersection_v1` unless `--force` is supplied.

## Known limitations

Crop-text screening is deterministic and does not use OCR. Contact sheets still require human visual confirmation. App Store originals represent a specific store version and may differ from historical Android artwork while retaining the explicitly mapped app identity.

Human review list: {', '.join(human_review) if human_review else 'none automatically flagged; review all contact-sheet rows before benchmark calibration'}.

## Later ITTE use

The aligned originals support identity benchmarks; four themed variants support style and package-level comparisons; all five images support visual-quality checks and later weight/threshold calibration. This build itself does not change or execute ITTE.
"""


def build(force: bool = False) -> dict:
    if OUTPUT_ROOT.exists():
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {OUTPUT_ROOT}. Use --force to rebuild."
            )
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    before = source_integrity_snapshot()
    write_json(OUTPUT_ROOT / "source_integrity_before.json", before)

    themes = load_theme_data()
    appstore, _ = load_appstore_manifest()
    original_files = {
        normalize_name(path.stem): path
        for path in (BENCHMARK_ROOT / "original_icons").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    all_slugs = set(original_files)
    for theme_id in THEME_IDS:
        all_slugs.update(themes[theme_id]["by_slug"])

    rows_by_slug: dict[str, list[dict]] = defaultdict(list)
    for theme_id in THEME_IDS:
        for row in themes[theme_id]["rows"]:
            rows_by_slug[row["_slug"]].append(row)
    identities = {
        slug: canonical_identity(slug, rows_by_slug.get(slug, []), appstore)
        for slug in all_slugs
    }

    image_cache: dict[Path, dict] = {}
    source_audit = build_source_audit(themes, image_cache)
    write_json(OUTPUT_ROOT / "source_audit.json", source_audit)
    duplicate_rows, ambiguous_originals = duplicate_review_rows(all_slugs, identities, themes, image_cache)
    write_csv(OUTPUT_ROOT / "duplicate_review.csv", DUPLICATE_FIELDS, duplicate_rows)

    alias_rows = []
    seen_aliases = set()
    for slug in sorted(all_slugs):
        identity = identities[slug]
        candidates = rows_by_slug.get(slug, [])
        if not candidates:
            candidates = [{"app_name": appstore.get(slug, {}).get("mapping_app_name", slug)}]
        for row in candidates:
            key = (row.get("app_name", ""), slug, identity["canonical_app_id"])
            if key in seen_aliases:
                continue
            seen_aliases.add(key)
            alias_rows.append(
                {
                    "raw_app_name": row.get("app_name", ""),
                    "normalized_app_name": normalize_name(row.get("app_name", "")),
                    "app_slug": slug,
                    "canonical_app_id": identity["canonical_app_id"],
                    "appstore_id": identity["appstore_id"],
                    "evidence": identity["match_basis"],
                    "status": "AMBIGUOUS" if identity["identity_ambiguous"] else "CONFIRMED",
                    "notes": identity["identity_notes"],
                }
            )
    write_csv(OUTPUT_ROOT / "app_aliases.csv", ALIAS_FIELDS, alias_rows)

    four_theme = {
        slug for slug in all_slugs
        if all(slug in themes[theme_id]["by_slug"] for theme_id in THEME_IDS)
    }
    strict_five = four_theme & set(original_files)
    valid_candidates = []
    excluded_rows = []
    counters = defaultdict(int)
    human_review = []

    for slug in sorted(all_slugs):
        identity = identities[slug]
        original_path = original_files.get(slug)
        theme_paths = {}
        theme_rows = {}
        reasons = []
        notes = []
        if identity["identity_ambiguous"]:
            reasons.append("AMBIGUOUS_IDENTITY")
            counters["ambiguous_identity"] += 1
        if slug in ambiguous_originals:
            reasons.append("AMBIGUOUS_ORIGINAL")
            counters["ambiguous_original"] += 1
        if not original_path:
            source_types = {row.get("source_type", "") for row in rows_by_slug.get(slug, [])}
            if "system_app" in source_types:
                reasons.append("SYSTEM_APP_UNCONFIRMED")
                counters["system_app_unconfirmed"] += 1
            else:
                reasons.append("MISSING_ORIGINAL")
        else:
            original_info = image_cache.setdefault(original_path, inspect_image(original_path))
            if not original_info["valid"]:
                reasons.append("BROKEN_IMAGE")
                counters["broken_image_apps"] += 1
                notes.extend(original_info["issues"])
        for theme_id in THEME_IDS:
            rows = themes[theme_id]["by_slug"].get(slug, [])
            if len(rows) != 1:
                reasons.append(f"MISSING_{theme_id.upper()}" if not rows else "DUPLICATE_CONFLICT")
                if len(rows) > 1:
                    counters["duplicate_conflict"] += 1
                continue
            row = rows[0]
            path = row["_themed_path"]
            theme_rows[theme_id] = row
            theme_paths[theme_id] = path
            info = image_cache.setdefault(path, inspect_image(path))
            if not info["valid"]:
                reasons.append("BROKEN_IMAGE")
                counters["broken_image_apps"] += 1
                notes.extend(info["issues"])
            if crop_text_suspected(row, info):
                reasons.append("TEXT_CROP_SUSPECTED")
                counters["text_crop_suspected"] += 1
                human_review.append(slug)
        reasons = list(dict.fromkeys(reasons))
        if not reasons and original_path and len(theme_paths) == len(THEME_IDS):
            valid_candidates.append(
                {
                    "slug": slug,
                    **identity,
                    "original_source": original_path,
                    "theme_sources": theme_paths,
                    "notes": identity["identity_notes"],
                }
            )
            continue
        candidate_paths = []
        if original_path:
            candidate_paths.append(relative_to_project(original_path))
        candidate_paths.extend(
            relative_to_project(path) for path in theme_paths.values() if path.exists()
        )
        excluded_rows.append(
            {
                "canonical_app_id": identity["canonical_app_id"],
                "raw_app_name": identity["app_name"],
                "available_in_originals": str(bool(original_path)).lower(),
                **{
                    f"available_in_{theme_id}": str(
                        len(themes[theme_id]["by_slug"].get(slug, [])) == 1
                        and themes[theme_id]["by_slug"][slug][0]["_themed_path"].exists()
                    ).lower()
                    for theme_id in THEME_IDS
                },
                "exclusion_reason": "|".join(reasons) if reasons else "FAILED_VALIDATION",
                "candidate_paths": "|".join(candidate_paths),
                "notes": " | ".join(dict.fromkeys(notes)) or identity["identity_notes"],
            }
        )

    valid_candidates.sort(key=lambda item: (item["canonical_app_id"], item["appstore_id"], normalize_name(item["app_name"])))
    manifest_rows = []
    for index, app in enumerate(valid_candidates, start=1):
        output_id = f"app_{index:03d}"
        core_paths = {"original": OUTPUT_ROOT / "core_dataset" / "originals" / f"{output_id}.png"}
        core_paths.update(
            {theme_id: OUTPUT_ROOT / "core_dataset" / theme_id / f"{output_id}.png" for theme_id in THEME_IDS}
        )
        copy_as_png(app["original_source"], core_paths["original"])
        for theme_id in THEME_IDS:
            copy_as_png(app["theme_sources"][theme_id], core_paths[theme_id])
        pair_dir = OUTPUT_ROOT / "pairs" / output_id
        copy_as_png(app["original_source"], pair_dir / "original.png")
        for theme_id in THEME_IDS:
            copy_as_png(app["theme_sources"][theme_id], pair_dir / f"{theme_id}.png")
        manifest_rows.append(
            {
                "output_id": output_id,
                "canonical_app_id": app["canonical_app_id"],
                "app_name": app["app_name"],
                "appstore_id": app["appstore_id"],
                "bundle_id": app["bundle_id"],
                "original_path": relative_to_project(app["original_source"]),
                **{f"{theme_id}_path": relative_to_project(app["theme_sources"][theme_id]) for theme_id in THEME_IDS},
                "original_sha256": sha256_file(core_paths["original"]),
                **{f"{theme_id}_sha256": sha256_file(core_paths[theme_id]) for theme_id in THEME_IDS},
                "match_basis": app["match_basis"],
                "match_confidence": app["match_confidence"],
                "validation_status": "VALID",
                "notes": app["notes"],
            }
        )
        app["output_id"] = output_id

    write_csv(OUTPUT_ROOT / "intersection_manifest.csv", MANIFEST_FIELDS, manifest_rows)
    write_csv(OUTPUT_ROOT / "excluded_apps.csv", EXCLUDED_FIELDS, excluded_rows)

    sheets = OUTPUT_ROOT / "contact_sheets"
    create_contact_sheet(valid_candidates, sheets / "all_apps_contact_sheet.jpg")
    for start in range(0, len(valid_candidates), 20):
        chunk = valid_candidates[start:start + 20]
        end = start + len(chunk)
        create_contact_sheet(chunk, sheets / f"apps_{start + 1:03d}_{end:03d}.jpg")

    source_counts = source_audit["sources"]
    summary = {
        "created_at": utc_now(),
        "source_original_count": source_counts["original_icons"]["image_count"],
        **{f"source_{theme_id}_count": source_counts[theme_id]["themed_icon_count"] for theme_id in THEME_IDS},
        "original_unique_apps": source_counts["original_icons"]["unique_app_count"],
        **{f"{theme_id}_unique_apps": source_counts[theme_id]["unique_app_count"] for theme_id in THEME_IDS},
        "four_theme_intersection_count": len(four_theme),
        "four_theme_but_missing_original_count": len(four_theme - set(original_files)),
        "strict_five_way_intersection_count": len(strict_five),
        "valid_app_count": len(valid_candidates),
        "excluded_app_count": len(excluded_rows),
        "ambiguous_identity_count": counters["ambiguous_identity"],
        "ambiguous_original_count": counters["ambiguous_original"],
        "duplicate_conflict_count": counters["duplicate_conflict"],
        "system_app_unconfirmed_count": counters["system_app_unconfirmed"],
        "broken_image_count": sum(
            1 for item in source_audit["image_validation_details"] if not item["valid"]
        ),
        "text_crop_suspected_count": counters["text_crop_suspected"],
        "human_review_apps": sorted(set(human_review)),
    }
    write_json(OUTPUT_ROOT / "intersection_summary.json", summary)
    fingerprint = fingerprint_core_dataset()
    write_json(OUTPUT_ROOT / "core_dataset_fingerprint.json", fingerprint)
    (OUTPUT_ROOT / "README.md").write_text(
        render_readme(summary, fingerprint, sorted(set(human_review))), encoding="utf-8"
    )

    after = source_integrity_snapshot()
    after["source_data_modified"] = not snapshots_match(before, after)
    write_json(OUTPUT_ROOT / "source_integrity_after.json", after)
    if after["source_data_modified"]:
        raise RuntimeError("Source data changed during the build; refusing to report success.")
    return {"summary": summary, "fingerprint": fingerprint}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the strict five-way ITTE Benchmark intersection.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild benchmark/intersection_v1.")
    args = parser.parse_args()
    try:
        result = build(force=args.force)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
