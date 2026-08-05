from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class GeneratedIcon:
    app: str
    path: Path
    structure_policy: dict[str, object] = field(
        default_factory=lambda: {
            "structure_preservation_mode": "preserve_major_structure",
            "structure_identity_metric_applicable": True,
            "structure_policy_rationale": "Legacy/default policy: evaluate structural identity.",
            "source": "legacy_default",
        }
    )


@dataclass(frozen=True)
class ThemeTransferExample:
    app: str
    original_path: Path
    style_ref_path: Path
    reference_raw_path: Path


@dataclass(frozen=True)
class ResolvedEvalInputs:
    theme_id: str
    package_id: str
    theme_examples: list[ThemeTransferExample]
    theme_refs: list[Path]
    generated_icons: list[GeneratedIcon]
    target_originals: dict[str, Path]
    missing_apps: list[str]
    skipped_apps: list[str]


def resolve_eval_inputs(theme_id: str, package_id: str, root_dir: Path | None = None) -> ResolvedEvalInputs:
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
    theme_dir = root / "data" / "styles" / theme_id
    final_dir = root / "data" / "packages" / package_id / "final"

    if not theme_dir.exists():
        raise FileNotFoundError(f"Missing theme directory: {theme_dir}")
    if not final_dir.exists():
        raise FileNotFoundError(f"Missing package final directory: {final_dir}")

    theme_examples = _find_theme_transfer_examples(theme_dir, root, theme_id)
    if not theme_examples:
        raise FileNotFoundError(f"Missing complete theme transfer examples under: {theme_dir}")
    theme_refs = [example.style_ref_path for example in theme_examples]

    generated_icons = [
        GeneratedIcon(
            app=path.stem,
            path=path,
            structure_policy=_load_structure_policy(root, package_id, path.stem),
        )
        for path in sorted(final_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not generated_icons:
        raise FileNotFoundError(f"Missing generated icons under: {final_dir}")

    expected_apps = _load_expected_apps(final_dir.parent)
    generated_app_names = {item.app for item in generated_icons}
    skipped_apps = sorted(set(expected_apps) - generated_app_names)

    target_originals: dict[str, Path] = {}
    missing_apps: list[str] = []
    for generated in generated_icons:
        target_path = _find_target_original(root, generated.app)
        if target_path is None:
            missing_apps.append(generated.app)
        else:
            target_originals[generated.app] = target_path

    if missing_apps:
        raise FileNotFoundError(
            "Missing target original for generated apps: " + ", ".join(sorted(missing_apps))
        )

    return ResolvedEvalInputs(
        theme_id=theme_id,
        package_id=package_id,
        theme_examples=theme_examples,
        theme_refs=theme_refs,
        generated_icons=generated_icons,
        target_originals=target_originals,
        missing_apps=missing_apps,
        skipped_apps=skipped_apps,
    )


def _load_expected_apps(package_dir: Path) -> list[str]:
    metadata_path = package_dir / "metadata.json"
    target_apps_path = package_dir / "target_apps.json"
    for path, key in ((metadata_path, "target_apps"), (target_apps_path, None)):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        apps = data.get(key) if key and isinstance(data, dict) else data
        if isinstance(apps, list) and all(isinstance(app, str) for app in apps):
            return sorted(set(apps))
    return []


def write_inputs_manifest(resolved: ResolvedEvalInputs, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "theme_id": resolved.theme_id,
        "package_id": resolved.package_id,
        "theme_transfer_examples": [
            _theme_example_manifest_item(item)
            for item in resolved.theme_examples
        ],
        "theme_refs": [str(path) for path in resolved.theme_refs],
        "generated_icons": [
            {
                "app": item.app,
                "path": str(item.path),
                "structure_policy": item.structure_policy,
            }
            for item in resolved.generated_icons
        ],
        "target_originals": {app: str(path) for app, path in sorted(resolved.target_originals.items())},
        "app_names": [item.app for item in resolved.generated_icons],
        "missing_apps": resolved.missing_apps,
        "skipped_apps": resolved.skipped_apps,
    }
    path = output_dir / "inputs_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _theme_example_manifest_item(item: ThemeTransferExample) -> dict[str, str]:
    return {
        "app": item.app,
        "original_path": str(item.original_path),
        "style_ref_path": str(item.style_ref_path),
        "reference_raw_path": str(item.reference_raw_path),
    }


def _find_theme_transfer_examples(theme_dir: Path, root: Path, theme_id: str) -> list[ThemeTransferExample]:
    examples = []
    for app_dir in sorted(path for path in theme_dir.iterdir() if path.is_dir()):
        app = app_dir.name
        style_ref_path = _find_named_image(app_dir, app, "style_ref")
        original_path = _find_original_image(app_dir, app)
        if not original_path or not style_ref_path:
            continue
        examples.append(
            ThemeTransferExample(
                app=app,
                original_path=original_path,
                style_ref_path=style_ref_path,
                reference_raw_path=original_path,
            )
        )
    return examples


def _find_named_image(app_dir: Path, app: str, role: str) -> Path | None:
    names = [f"{app}_{role}", role]
    for stem in names:
        for extension in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = app_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    matches = sorted(
        path
        for path in app_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and role in path.stem.lower()
    )
    return matches[0] if matches else None


def _find_original_image(app_dir: Path, app: str) -> Path | None:
    for stem in [app, "original", "raw", "input", "image"]:
        for extension in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = app_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate

    matches = sorted(
        path
        for path in app_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not _is_reference_output_name(path.stem.lower())
    )
    return matches[0] if len(matches) == 1 else None


def _is_reference_output_name(stem: str) -> bool:
    return any(
        token in stem
        for token in ["style_ref", "transferred_ref", "reference", "background", "foreground"]
    )


def _find_target_original(root: Path, app: str) -> Path | None:
    target_dir = root / "data" / "targets" / app
    if not target_dir.exists():
        return None

    exact_matches = [
        target_dir / f"{app}{extension}"
        for extension in [".png", ".jpg", ".jpeg", ".webp"]
        if (target_dir / f"{app}{extension}").exists()
    ]
    if exact_matches:
        return exact_matches[0]

    for stem in ["target", "original", "input", "image"]:
        for extension in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = target_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate

    images = sorted(
        path
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) == 1:
        return images[0]
    return None


def _load_structure_policy(root: Path, package_id: str, app: str) -> dict[str, object]:
    path = root / "data" / "packages" / package_id / "cases" / app / "transfer_plan.json"
    legacy = {
        "structure_preservation_mode": "preserve_major_structure",
        "structure_identity_metric_applicable": True,
        "structure_policy_rationale": "Legacy package without a frozen structure policy; structural identity remains enabled.",
        "source": "legacy_default",
    }
    if not path.exists():
        return legacy
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read structure policy for {app!r} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Transfer plan for {app!r} must be a JSON object: {path}")

    mode = payload.get("structure_preservation_mode")
    if mode is None:
        return legacy
    if mode not in {"preserve_major_structure", "semantic_recompose"}:
        raise ValueError(f"Invalid structure_preservation_mode for {app!r}: {mode!r}")
    expected_applicable = mode == "preserve_major_structure"
    declared_applicable = payload.get("structure_identity_metric_applicable")
    if not isinstance(declared_applicable, bool) or declared_applicable != expected_applicable:
        raise ValueError(
            f"Inconsistent structure identity policy for {app!r}: mode={mode!r}, "
            f"structure_identity_metric_applicable={declared_applicable!r}"
        )
    return {
        "structure_preservation_mode": mode,
        "structure_identity_metric_applicable": expected_applicable,
        "structure_policy_rationale": str(payload.get("structure_policy_rationale", "")),
        "source": str(path),
    }
