from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from build_intersection import (
    BENCHMARK_ROOT,
    PROJECT_ROOT,
    THEME_IDS,
    canonical_identity,
    copy_as_png,
    font_for_contact_sheet,
    inspect_image,
    load_appstore_manifest,
    load_theme_data,
    normalize_name,
    relative_to_project,
    sha256_file,
    snapshots_match,
    source_integrity_snapshot,
    utc_now,
    write_csv,
    write_json,
)


OUTPUT_ROOT = BENCHMARK_ROOT / "evaluation_set_v1"
PAIR_FIELDS = (
    "pair_id",
    "canonical_app_id",
    "app_slug",
    "app_name",
    "theme_id",
    "appstore_id",
    "bundle_id",
    "original_asset_path",
    "themed_asset_path",
    "source_original_path",
    "source_themed_path",
    "original_sha256",
    "themed_sha256",
    "match_basis",
    "match_confidence",
    "validation_status",
    "notes",
)
GALLERY_FIELDS = (
    "canonical_app_id",
    "app_slug",
    "app_name",
    "appstore_id",
    "bundle_id",
    "asset_path",
    "source_path",
    "sha256",
    "width",
    "height",
    "mode",
    "validation_status",
)
THEME_ASSET_FIELDS = (
    "canonical_app_id",
    "app_slug",
    "app_name",
    "theme_id",
    "source_type",
    "appstore_id",
    "asset_path",
    "source_path",
    "sha256",
    "width",
    "height",
    "mode",
    "has_reliable_original",
    "eligible_for_style",
    "eligible_for_quality",
    "validation_status",
    "notes",
)
EXCLUDED_PAIR_FIELDS = (
    "theme_id",
    "canonical_app_id",
    "app_slug",
    "app_name",
    "source_type",
    "available_original",
    "available_themed",
    "exclusion_reason",
    "source_original_path",
    "source_themed_path",
    "notes",
)
PACKAGE_PAIR_FIELDS = (
    "package_pair_id",
    "canonical_app_id",
    "app_slug",
    "app_name",
    "theme_a",
    "theme_b",
    "theme_a_asset_path",
    "theme_b_asset_path",
    "theme_a_sha256",
    "theme_b_sha256",
    "validation_status",
)
CATALOG_FIELDS = (
    "canonical_app_id",
    "app_slug",
    "app_name",
    "appstore_id",
    "bundle_id",
    "has_original",
    "available_themes",
    "pair_count",
    "identity_basis",
    "identity_confidence",
    "status",
    "notes",
)


def fingerprint_assets() -> dict:
    asset_root = OUTPUT_ROOT / "assets"
    entries = []
    for path in sorted(item for item in asset_root.rglob("*") if item.is_file()):
        relative = path.relative_to(asset_root).as_posix()
        entries.append({"relative_path": relative, "sha256": sha256_file(path)})
    payload = "".join(f"{item['relative_path']}\t{item['sha256']}\n" for item in entries)
    return {
        "algorithm": "sha256(relative_path + TAB + file_sha256 + LF)",
        "dataset_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "file_count": len(entries),
        "created_at": utc_now(),
        "files": entries,
    }


def create_pair_contact_sheet(rows: list[dict], destination: Path) -> None:
    tile = 180
    label_width = 260
    header_height = 54
    row_height = 206
    canvas = Image.new("RGB", (label_width + tile * 2, header_height + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font_for_contact_sheet(20)
    label_font = font_for_contact_sheet(18)
    small_font = font_for_contact_sheet(13)
    draw.text((12, 15), "APP", fill="black", font=header_font)
    draw.text((label_width + 18, 15), "original", fill="black", font=header_font)
    draw.text((label_width + tile + 18, 15), "themed", fill="black", font=header_font)
    for index, row in enumerate(rows):
        y = header_height + index * row_height
        draw.line((0, y, canvas.width, y), fill=(210, 210, 210), width=1)
        draw.text((12, y + 54), row["app_name"], fill="black", font=label_font)
        draw.text((12, y + 83), row["app_slug"], fill=(70, 70, 70), font=small_font)
        draw.text((12, y + 105), row["canonical_app_id"], fill=(90, 90, 90), font=small_font)
        for col, field in enumerate(("original_asset_path", "themed_asset_path")):
            path = PROJECT_ROOT / row[field]
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
                rgba.thumbnail((160, 160), Image.Resampling.LANCZOS)
                tile_image = Image.new("RGBA", (160, 160), (245, 245, 245, 255))
                tile_image.alpha_composite(rgba, ((160 - rgba.width) // 2, (160 - rgba.height) // 2))
                canvas.paste(tile_image.convert("RGB"), (label_width + col * tile + 10, y + 20))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=92, optimize=True)


def render_readme(summary: dict, fingerprint: dict) -> str:
    pair_counts = summary["pair_counts_by_theme"]
    return f"""# ITTE evaluation set v1

## Purpose

This is the larger, identity-aligned evaluation collection for diagnosing ITTE on real mobile themes. It complements, rather than replaces, `benchmark/intersection_v1`, whose five apps remain the highest-confidence cross-four-theme Gold Core.

## Assets and records

- Fixed identity gallery: {summary['identity_gallery_count']} reliable App Store originals.
- Real themed assets: {summary['theme_asset_count']} images.
- Strict original/theme positive pairs: {summary['pair_record_count']} records across {summary['pair_union_unique_app_count']} unique apps.
- Pair counts: theme_001={pair_counts['theme_001']}, theme_002={pair_counts['theme_002']}, theme_003={pair_counts['theme_003']}, theme_004={pair_counts['theme_004']}.
- Package same-app cross-theme swap records: {summary['package_pair_record_count']}.

All identities use App Store ID first and explicit mapping slugs second. No fuzzy matching is used. Source data is copied without crop, resize, enhancement, or content modification. Existing PNG files are copied byte-for-byte.

## Intended Benchmark use

### Style

Use `theme_assets_manifest.csv` as the four reference banks and `pair_manifest.csv` themed assets as labeled queries. The correct theme must be ranked against all four themes. Exclude the query app from its own reference bank, balance reference-bank sizes, and report both per-theme macro metrics and pooled micro metrics.

### Identity

Use all rows in `identity_gallery.csv` as a fixed 91-app gallery. Each row in `pair_manifest.csv` supplies one themed query and its correct original. Report Top-1, Top-5, MRR, positive similarity, strongest negative, and margin. Do not rebuild distractors from a temporary package.

### Package coherence

Use `package_pair_manifest.csv` to replace an app with the same app from another theme. This preserves identity while changing theme membership. Report results per theme pair and mixing ratio.

### Visual quality

All VALID rows in `theme_assets_manifest.csv` can be used as clean sources for deterministic controlled degradations. Store degradation type, severity, parameters, and random seed separately.

## Important statistical rule

theme_003 has substantially more pairs than theme_002. Always report per-theme macro results as well as pooled micro results. Split or cross-validate by `canonical_app_id`, never by image, so variants of one app cannot leak across calibration and evaluation.

## Build and validate

```powershell
python benchmark/tools/build_evaluation_set.py
python benchmark/tools/validate_evaluation_set.py
```

The builder refuses to overwrite an existing output unless `--force` is supplied.

## Fingerprint

`{fingerprint['dataset_fingerprint']}`

This build prepares data only. It does not execute ITTE, generate degradations, or tune metrics, weights, or thresholds.
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
    original_sources = {
        normalize_name(path.stem): path
        for path in (BENCHMARK_ROOT / "original_icons").iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    rows_by_slug: dict[str, list[dict]] = defaultdict(list)
    all_slugs = set(original_sources)
    for theme_id in THEME_IDS:
        for row in themes[theme_id]["rows"]:
            rows_by_slug[row["_slug"]].append(row)
            all_slugs.add(row["_slug"])
    identities = {
        slug: canonical_identity(slug, rows_by_slug.get(slug, []), appstore)
        for slug in all_slugs
    }

    gallery_rows = []
    original_assets = {}
    original_infos = {}
    for slug, source in sorted(original_sources.items()):
        identity = identities[slug]
        source_info = inspect_image(source)
        if identity["identity_ambiguous"] or not source_info["valid"]:
            continue
        asset = OUTPUT_ROOT / "assets" / "originals" / f"{slug}.png"
        copy_as_png(source, asset)
        asset_info = inspect_image(asset)
        original_assets[slug] = asset
        original_infos[slug] = asset_info
        gallery_rows.append(
            {
                "canonical_app_id": identity["canonical_app_id"],
                "app_slug": slug,
                "app_name": identity["app_name"],
                "appstore_id": identity["appstore_id"],
                "bundle_id": identity["bundle_id"],
                "asset_path": relative_to_project(asset),
                "source_path": relative_to_project(source),
                "sha256": asset_info["sha256"],
                "width": asset_info["width"],
                "height": asset_info["height"],
                "mode": asset_info["mode"],
                "validation_status": "VALID",
            }
        )
    gallery_rows.sort(key=lambda row: row["canonical_app_id"])
    write_csv(OUTPUT_ROOT / "identity_gallery.csv", GALLERY_FIELDS, gallery_rows)

    theme_asset_rows = []
    theme_assets: dict[str, dict[str, Path]] = {theme_id: {} for theme_id in THEME_IDS}
    theme_asset_infos: dict[str, dict[str, dict]] = {theme_id: {} for theme_id in THEME_IDS}
    invalid_theme_records = []
    for theme_id in THEME_IDS:
        for slug, mapped_rows in sorted(themes[theme_id]["by_slug"].items()):
            identity = identities[slug]
            if len(mapped_rows) != 1:
                invalid_theme_records.append((theme_id, slug, "DUPLICATE_MAPPING"))
                continue
            row = mapped_rows[0]
            source = row["_themed_path"]
            source_info = inspect_image(source)
            if identity["identity_ambiguous"] or not source_info["valid"]:
                invalid_theme_records.append(
                    (theme_id, slug, "AMBIGUOUS_IDENTITY" if identity["identity_ambiguous"] else "BROKEN_IMAGE")
                )
                continue
            asset = OUTPUT_ROOT / "assets" / "themes" / theme_id / f"{slug}.png"
            copy_as_png(source, asset)
            asset_info = inspect_image(asset)
            theme_assets[theme_id][slug] = asset
            theme_asset_infos[theme_id][slug] = asset_info
            theme_asset_rows.append(
                {
                    "canonical_app_id": identity["canonical_app_id"],
                    "app_slug": slug,
                    "app_name": row.get("app_name", identity["app_name"]),
                    "theme_id": theme_id,
                    "source_type": row.get("source_type", ""),
                    "appstore_id": identity["appstore_id"],
                    "asset_path": relative_to_project(asset),
                    "source_path": relative_to_project(source),
                    "sha256": asset_info["sha256"],
                    "width": asset_info["width"],
                    "height": asset_info["height"],
                    "mode": asset_info["mode"],
                    "has_reliable_original": str(slug in original_assets).lower(),
                    "eligible_for_style": "true",
                    "eligible_for_quality": "true",
                    "validation_status": "VALID",
                    "notes": row.get("notes", ""),
                }
            )
    theme_asset_rows.sort(key=lambda row: (row["theme_id"], row["canonical_app_id"]))
    write_csv(OUTPUT_ROOT / "theme_assets_manifest.csv", THEME_ASSET_FIELDS, theme_asset_rows)

    pair_rows = []
    excluded_rows = []
    for theme_id in THEME_IDS:
        for slug, mapped_rows in sorted(themes[theme_id]["by_slug"].items()):
            identity = identities[slug]
            row = mapped_rows[0] if len(mapped_rows) == 1 else {}
            themed_source = row.get("_themed_path")
            original_source = original_sources.get(slug)
            reasons = []
            if identity["identity_ambiguous"]:
                reasons.append("AMBIGUOUS_IDENTITY")
            if slug not in original_assets:
                if row.get("source_type") == "system_app":
                    reasons.append("SYSTEM_APP_WITHOUT_RELIABLE_ORIGINAL")
                else:
                    reasons.append("MISSING_RELIABLE_ORIGINAL")
            if slug not in theme_assets[theme_id]:
                reasons.append("INVALID_OR_MISSING_THEMED_IMAGE")
            if len(mapped_rows) != 1:
                reasons.append("DUPLICATE_MAPPING")
            if reasons:
                excluded_rows.append(
                    {
                        "theme_id": theme_id,
                        "canonical_app_id": identity["canonical_app_id"],
                        "app_slug": slug,
                        "app_name": row.get("app_name", identity["app_name"]),
                        "source_type": row.get("source_type", ""),
                        "available_original": str(slug in original_assets).lower(),
                        "available_themed": str(slug in theme_assets[theme_id]).lower(),
                        "exclusion_reason": "|".join(dict.fromkeys(reasons)),
                        "source_original_path": relative_to_project(original_source) if original_source else "",
                        "source_themed_path": relative_to_project(themed_source) if themed_source and themed_source.exists() else "",
                        "notes": row.get("notes", ""),
                    }
                )
                continue
            original_asset = original_assets[slug]
            themed_asset = theme_assets[theme_id][slug]
            pair_rows.append(
                {
                    "pair_id": f"pair_{theme_id}_{slug}",
                    "canonical_app_id": identity["canonical_app_id"],
                    "app_slug": slug,
                    "app_name": row.get("app_name", identity["app_name"]),
                    "theme_id": theme_id,
                    "appstore_id": identity["appstore_id"],
                    "bundle_id": identity["bundle_id"],
                    "original_asset_path": relative_to_project(original_asset),
                    "themed_asset_path": relative_to_project(themed_asset),
                    "source_original_path": relative_to_project(original_source),
                    "source_themed_path": relative_to_project(themed_source),
                    "original_sha256": original_infos[slug]["sha256"],
                    "themed_sha256": theme_asset_infos[theme_id][slug]["sha256"],
                    "match_basis": identity["match_basis"],
                    "match_confidence": identity["match_confidence"],
                    "validation_status": "VALID",
                    "notes": identity["identity_notes"],
                }
            )
    pair_rows.sort(key=lambda row: (row["theme_id"], row["canonical_app_id"]))
    write_csv(OUTPUT_ROOT / "pair_manifest.csv", PAIR_FIELDS, pair_rows)
    write_csv(OUTPUT_ROOT / "excluded_pairs.csv", EXCLUDED_PAIR_FIELDS, excluded_rows)

    package_pair_rows = []
    for left_index, theme_a in enumerate(THEME_IDS):
        for theme_b in THEME_IDS[left_index + 1:]:
            common = sorted(set(theme_assets[theme_a]) & set(theme_assets[theme_b]))
            for slug in common:
                identity = identities[slug]
                row = themes[theme_a]["by_slug"][slug][0]
                asset_a = theme_assets[theme_a][slug]
                asset_b = theme_assets[theme_b][slug]
                package_pair_rows.append(
                    {
                        "package_pair_id": f"package_pair_{theme_a}_{theme_b}_{slug}",
                        "canonical_app_id": identity["canonical_app_id"],
                        "app_slug": slug,
                        "app_name": row.get("app_name", identity["app_name"]),
                        "theme_a": theme_a,
                        "theme_b": theme_b,
                        "theme_a_asset_path": relative_to_project(asset_a),
                        "theme_b_asset_path": relative_to_project(asset_b),
                        "theme_a_sha256": theme_asset_infos[theme_a][slug]["sha256"],
                        "theme_b_sha256": theme_asset_infos[theme_b][slug]["sha256"],
                        "validation_status": "VALID",
                    }
                )
    write_csv(OUTPUT_ROOT / "package_pair_manifest.csv", PACKAGE_PAIR_FIELDS, package_pair_rows)

    pair_count_by_slug = defaultdict(int)
    for row in pair_rows:
        pair_count_by_slug[row["app_slug"]] += 1
    catalog_rows = []
    for slug in sorted(all_slugs, key=lambda item: identities[item]["canonical_app_id"]):
        identity = identities[slug]
        available_themes = [theme_id for theme_id in THEME_IDS if slug in theme_assets[theme_id]]
        catalog_rows.append(
            {
                "canonical_app_id": identity["canonical_app_id"],
                "app_slug": slug,
                "app_name": identity["app_name"],
                "appstore_id": identity["appstore_id"],
                "bundle_id": identity["bundle_id"],
                "has_original": str(slug in original_assets).lower(),
                "available_themes": "|".join(available_themes),
                "pair_count": pair_count_by_slug[slug],
                "identity_basis": identity["match_basis"],
                "identity_confidence": identity["match_confidence"],
                "status": "AMBIGUOUS" if identity["identity_ambiguous"] else "CONFIRMED",
                "notes": identity["identity_notes"],
            }
        )
    write_csv(OUTPUT_ROOT / "app_catalog.csv", CATALOG_FIELDS, catalog_rows)

    contact_root = OUTPUT_ROOT / "contact_sheets"
    for theme_id in THEME_IDS:
        rows = [row for row in pair_rows if row["theme_id"] == theme_id]
        for start in range(0, len(rows), 20):
            chunk = rows[start:start + 20]
            end = start + len(chunk)
            create_pair_contact_sheet(
                chunk,
                contact_root / f"{theme_id}_pairs_{start + 1:03d}_{end:03d}.jpg",
            )

    pair_counts = {
        theme_id: sum(row["theme_id"] == theme_id for row in pair_rows)
        for theme_id in THEME_IDS
    }
    package_counts = defaultdict(int)
    for row in package_pair_rows:
        package_counts[f"{row['theme_a']}__{row['theme_b']}"] += 1
    summary = {
        "created_at": utc_now(),
        "identity_gallery_count": len(gallery_rows),
        "theme_asset_count": len(theme_asset_rows),
        "theme_asset_counts": {
            theme_id: sum(row["theme_id"] == theme_id for row in theme_asset_rows)
            for theme_id in THEME_IDS
        },
        "pair_record_count": len(pair_rows),
        "pair_counts_by_theme": pair_counts,
        "pair_union_unique_app_count": len({row["canonical_app_id"] for row in pair_rows}),
        "excluded_pair_record_count": len(excluded_rows),
        "package_pair_record_count": len(package_pair_rows),
        "package_pair_counts": dict(sorted(package_counts.items())),
        "catalog_app_count": len(catalog_rows),
        "ambiguous_identity_count": sum(row["status"] == "AMBIGUOUS" for row in catalog_rows),
        "invalid_theme_record_count": len(invalid_theme_records),
        "gold_core_path": "benchmark/intersection_v1",
        "gold_core_valid_app_count": 5,
        "recommended_primary_metrics": {
            "style": ["top1", "mrr", "margin", "per_theme_macro", "micro"],
            "identity": ["top1", "top5", "mrr", "margin"],
            "package": ["score_drop", "mix_ratio_monotonicity", "outlier_f1"],
            "quality": ["clean_degraded_gap", "severity_monotonicity", "detection_rate"],
        },
    }
    write_json(OUTPUT_ROOT / "evaluation_set_summary.json", summary)
    fingerprint = fingerprint_assets()
    fingerprint.update(
        {
            "original_count": len(gallery_rows),
            "themed_count": len(theme_asset_rows),
            "pair_record_count": len(pair_rows),
            "unique_paired_app_count": summary["pair_union_unique_app_count"],
        }
    )
    write_json(OUTPUT_ROOT / "evaluation_set_fingerprint.json", fingerprint)
    (OUTPUT_ROOT / "README.md").write_text(render_readme(summary, fingerprint), encoding="utf-8")

    after = source_integrity_snapshot()
    after["source_data_modified"] = not snapshots_match(before, after)
    write_json(OUTPUT_ROOT / "source_integrity_after.json", after)
    if after["source_data_modified"]:
        raise RuntimeError("Source Benchmark data changed during the build.")
    return {"summary": summary, "fingerprint": fingerprint["dataset_fingerprint"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the expanded real-theme ITTE evaluation set.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild benchmark/evaluation_set_v1.")
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
