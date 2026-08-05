"""Prepare the normalized dataset for the existing generation entry points.

The source ``dataset`` tree is treated as read-only. Generated files are written
under ``data/styles`` and ``data/targets`` using the schemas consumed by the
current backend services.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
EXPECTED_FORMATS = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".webp": "webp",
}
REQUIRED_APP_FIELDS = ("app", "display_name", "category", "core_function")


class PreparationError(ValueError):
    """Raised when source data cannot be converted safely."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreparationError(f"Duplicate JSON key/app_id: {key!r}")
        result[key] = value
    return result


def _validate_identifier(value: str, label: str) -> None:
    if not value or value in {".", ".."}:
        raise PreparationError(f"Invalid {label}: {value!r}")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise PreparationError(f"Unsafe {label}: {value!r}")


def _load_apps(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except FileNotFoundError as exc:
        raise PreparationError(f"Missing apps metadata file: {path}") from exc
    except UnicodeDecodeError as exc:
        raise PreparationError(f"apps.json is not valid UTF-8: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PreparationError(
            f"Cannot parse apps.json at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("apps"), dict):
        raise PreparationError("apps.json must contain a top-level object field named 'apps'.")

    normalized: dict[str, dict[str, Any]] = {}
    seen_declared_ids: dict[str, str] = {}
    for app_id in sorted(raw["apps"], key=str.casefold):
        _validate_identifier(app_id, "app_id")
        metadata = raw["apps"][app_id]
        if not isinstance(metadata, dict):
            raise PreparationError(f"Metadata for app {app_id!r} must be a JSON object.")

        missing = [
            field
            for field in REQUIRED_APP_FIELDS
            if not isinstance(metadata.get(field), str) or not metadata[field].strip()
        ]
        if missing:
            raise PreparationError(
                f"Metadata for app {app_id!r} is missing non-empty fields: {', '.join(missing)}"
            )

        declared_id = metadata["app"]
        if declared_id != app_id:
            raise PreparationError(
                f"apps.json key {app_id!r} disagrees with its 'app' field {declared_id!r}."
            )
        folded_id = declared_id.casefold()
        if folded_id in seen_declared_ids:
            raise PreparationError(
                f"Duplicate app_id values: {seen_declared_ids[folded_id]!r} and {declared_id!r}."
            )
        seen_declared_ids[folded_id] = declared_id

        ordered: dict[str, Any] = {}
        for field in ("app", "display_name", "category", "store_description", "core_function"):
            if field in metadata:
                ordered[field] = metadata[field]
        for field in sorted(set(metadata) - set(ordered), key=str.casefold):
            ordered[field] = metadata[field]
        normalized[app_id] = ordered
    return normalized


def _detect_image_format(path: Path) -> str | None:
    with path.open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return None


def _collect_images(directory: Path, label: str, warnings: list[str]) -> dict[str, Path]:
    if not directory.exists() or not directory.is_dir():
        raise PreparationError(f"Missing {label} directory: {directory}")

    images: dict[str, Path] = {}
    seen_casefolded: dict[str, str] = {}
    for path in sorted(directory.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            warnings.append(f"Unsupported non-image file ignored in {label}: {path}")
            continue

        actual_format = _detect_image_format(path)
        expected_format = EXPECTED_FORMATS[suffix]
        if actual_format is None:
            raise PreparationError(f"Cannot identify image format from file contents: {path}")
        if actual_format != expected_format:
            raise PreparationError(
                f"Image extension/content mismatch for {path}: extension expects "
                f"{expected_format}, contents are {actual_format}."
            )

        app_id = path.stem
        _validate_identifier(app_id, f"image app_id in {label}")
        folded = app_id.casefold()
        if folded in seen_casefolded:
            raise PreparationError(
                f"Duplicate app_id/image stem in {label}: "
                f"{seen_casefolded[folded]!r} and {app_id!r}."
            )
        seen_casefolded[folded] = app_id
        images[app_id] = path
    return images


def _copy_if_changed(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and filecmp.cmp(source, destination, shallow=False):
        return False
    shutil.copy2(source, destination)
    return True


def _write_json_if_changed(payload: dict[str, Any], destination: Path) -> bool:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding="utf-8") == text:
        return False
    destination.write_text(text, encoding="utf-8", newline="\n")
    return True


def _safe_remove_directory(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    allowed_parent = (root / "data").resolve()
    if resolved.parent != allowed_parent and resolved.parent.parent != allowed_parent:
        raise PreparationError(f"Refusing to clean path outside data outputs: {resolved}")
    if not path.exists():
        return False
    if not path.is_dir():
        raise PreparationError(f"Cannot clean output because it is not a directory: {path}")
    shutil.rmtree(path)
    return True


def prepare_generation_data(
    root: Path,
    theme_id: str | None = None,
    clean: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    root = root.resolve()
    dataset_dir = root / "dataset"
    originals_dir = dataset_dir / "originals"
    themes_dir = dataset_dir / "themes"
    warnings: list[str] = []

    apps = _load_apps(dataset_dir / "apps.json")
    originals = _collect_images(originals_dir, "dataset/originals", warnings)
    if not originals:
        raise PreparationError(f"No supported original images found in: {originals_dir}")

    missing_metadata = sorted(set(originals) - set(apps), key=str.casefold)
    if missing_metadata:
        raise PreparationError(
            "Original images lack apps.json metadata: " + ", ".join(missing_metadata)
        )
    metadata_without_original = sorted(set(apps) - set(originals), key=str.casefold)
    for app_id in metadata_without_original:
        warnings.append(f"apps.json metadata has no original image: {app_id}")

    if not themes_dir.exists() or not themes_dir.is_dir():
        raise PreparationError(f"Missing themes directory: {themes_dir}")
    available_theme_dirs = {
        path.name: path
        for path in sorted(themes_dir.iterdir(), key=lambda item: (item.name.casefold(), item.name))
        if path.is_dir()
    }
    if theme_id is not None:
        _validate_identifier(theme_id, "theme_id")
        if theme_id not in available_theme_dirs:
            available = ", ".join(sorted(available_theme_dirs, key=str.casefold)) or "<none>"
            raise PreparationError(
                f"Unknown theme_id {theme_id!r}; available themes: {available}"
            )
        selected_theme_dirs = {theme_id: available_theme_dirs[theme_id]}
    else:
        selected_theme_dirs = available_theme_dirs
    if not selected_theme_dirs:
        raise PreparationError(f"No theme directories found in: {themes_dir}")

    theme_matches: dict[str, dict[str, Path]] = {}
    unmatched_theme_images: dict[str, list[str]] = {}
    for selected_id, directory in selected_theme_dirs.items():
        _validate_identifier(selected_id, "theme_id")
        theme_images = _collect_images(directory, f"dataset/themes/{selected_id}", warnings)
        matched = {
            app_id: theme_images[app_id]
            for app_id in sorted(set(theme_images) & set(originals), key=str.casefold)
        }
        unmatched = sorted(set(theme_images) - set(originals), key=str.casefold)
        for app_id in unmatched:
            warnings.append(
                f"Theme image has no matching original and was not converted: "
                f"{selected_id}/{theme_images[app_id].name}"
            )
        if not matched:
            warnings.append(f"Theme {selected_id} has no images matched to originals.")
        theme_matches[selected_id] = matched
        unmatched_theme_images[selected_id] = unmatched

    data_dir = root / "data"
    styles_dir = data_dir / "styles"
    targets_dir = data_dir / "targets"
    cleaned: list[str] = []
    if clean:
        if _safe_remove_directory(targets_dir, root):
            cleaned.append(str(targets_dir.relative_to(root)).replace("\\", "/"))
        styles_clean_target = styles_dir / theme_id if theme_id else styles_dir
        if _safe_remove_directory(styles_clean_target, root):
            cleaned.append(str(styles_clean_target.relative_to(root)).replace("\\", "/"))

    copied_files = 0
    unchanged_files = 0
    written_json = 0
    unchanged_json = 0

    for app_id in sorted(originals, key=str.casefold):
        original = originals[app_id]
        target_dir = targets_dir / app_id
        if _copy_if_changed(original, target_dir / original.name):
            copied_files += 1
        else:
            unchanged_files += 1
        if _write_json_if_changed(apps[app_id], target_dir / "target.json"):
            written_json += 1
        else:
            unchanged_json += 1

    per_theme: dict[str, dict[str, int]] = {}
    for selected_id in sorted(theme_matches, key=str.casefold):
        examples: dict[str, dict[str, Any]] = {}
        for app_id in sorted(theme_matches[selected_id], key=str.casefold):
            original = originals[app_id]
            style_ref = theme_matches[selected_id][app_id]
            output_dir = styles_dir / selected_id / app_id
            original_destination = output_dir / f"{app_id}{original.suffix.lower()}"
            style_destination = output_dir / f"{app_id}_style_ref{style_ref.suffix.lower()}"
            for source, destination in (
                (original, original_destination),
                (style_ref, style_destination),
            ):
                if _copy_if_changed(source, destination):
                    copied_files += 1
                else:
                    unchanged_files += 1
            examples[app_id] = apps[app_id]

        theme_payload = {
            "theme_id": selected_id,
            "description": (
                "应用图标主题风格参考包。每个参考样例由同一 App 的原始图标和设计师主题图组成，"
                "用于分析 original -> style_ref 的共同设计变化。"
            ),
            "input_schema": "reference_original_to_style_ref",
            "reference_pair_schema": {
                "original": "参考 App 的原始图标，是该 App 的身份和功能语义来源。",
                "style_ref": "同一 App 的设计师主题图，用于学习当前主题包的设计语言。",
            },
            "examples": examples,
        }
        if _write_json_if_changed(theme_payload, styles_dir / selected_id / "theme.json"):
            written_json += 1
        else:
            unchanged_json += 1
        per_theme[selected_id] = {
            "matched_examples": len(examples),
            "unmatched_theme_images": len(unmatched_theme_images[selected_id]),
        }

    summary: dict[str, Any] = {
        "root": str(root),
        "clean": clean,
        "cleaned": cleaned,
        "selected_themes": sorted(theme_matches, key=str.casefold),
        "apps_metadata": len(apps),
        "originals": len(originals),
        "targets_prepared": len(originals),
        "theme_examples_prepared": sum(len(items) for items in theme_matches.values()),
        "copied_files": copied_files,
        "unchanged_files": unchanged_files,
        "written_json": written_json,
        "unchanged_json": unchanged_json,
        "warnings": len(warnings),
        "themes": per_theme,
    }
    return summary, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert read-only dataset inputs into the existing generation directory schema."
    )
    parser.add_argument(
        "--theme-id",
        help="Prepare only one theme while still preparing every target app.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Remove generated targets first; also remove all styles, or only the selected "
            "theme when --theme-id is supplied. Other data directories are preserved."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, warnings = prepare_generation_data(
            root=args.root,
            theme_id=args.theme_id,
            clean=args.clean,
        )
    except (PreparationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for message in warnings:
        print(f"WARNING: {message}", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
