from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class GeneratedIcon:
    app: str
    path: Path


@dataclass(frozen=True)
class ResolvedEvalInputs:
    theme_id: str
    package_id: str
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

    theme_refs = sorted(theme_dir.glob("*/*_style_ref.*"))
    theme_refs = [path for path in theme_refs if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not theme_refs:
        raise FileNotFoundError(f"Missing theme style references under: {theme_dir}")

    generated_icons = [
        GeneratedIcon(app=path.stem, path=path)
        for path in sorted(final_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not generated_icons:
        raise FileNotFoundError(f"Missing generated icons under: {final_dir}")

    target_originals: dict[str, Path] = {}
    missing_apps: list[str] = []
    skipped_apps: list[str] = []
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
        theme_refs=theme_refs,
        generated_icons=generated_icons,
        target_originals=target_originals,
        missing_apps=missing_apps,
        skipped_apps=skipped_apps,
    )


def write_inputs_manifest(resolved: ResolvedEvalInputs, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "theme_id": resolved.theme_id,
        "package_id": resolved.package_id,
        "theme_refs": [str(path) for path in resolved.theme_refs],
        "generated_icons": [{"app": item.app, "path": str(item.path)} for item in resolved.generated_icons],
        "target_originals": {app: str(path) for app, path in sorted(resolved.target_originals.items())},
        "app_names": [item.app for item in resolved.generated_icons],
        "missing_apps": resolved.missing_apps,
        "skipped_apps": resolved.skipped_apps,
    }
    path = output_dir / "inputs_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
