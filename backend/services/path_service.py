from pathlib import Path


def resolve_case_paths(theme_id, target_app, root_dir=None, max_examples=5):
    root = Path(root_dir) if root_dir else Path.cwd()
    theme_dir = _find_theme_dir(root, theme_id)
    target_background = root / "data" / "targets" / target_app / "background.png"
    target_foreground = root / "data" / "targets" / target_app / "foreground.png"

    if not target_background.exists():
        raise FileNotFoundError(f"Missing target background: {target_background}")
    if not target_foreground.exists():
        raise FileNotFoundError(f"Missing target foreground: {target_foreground}")

    examples = []
    for app_dir in sorted([item for item in theme_dir.iterdir() if item.is_dir()]):
        example = _resolve_reference_example(app_dir)
        if example:
            examples.append(example)
        if len(examples) >= max_examples:
            break

    if not examples:
        raise ValueError(f"No valid reference examples found in theme: {theme_dir}")

    return {
        "theme_id": theme_id,
        "target_app": target_app,
        "reference_examples": examples,
        "target_background": str(target_background),
        "target_foreground": str(target_foreground),
    }


def _find_theme_dir(root, theme_id):
    theme_dir = root / "data" / "styles" / theme_id
    if theme_dir.exists() and theme_dir.is_dir():
        return theme_dir
    raise FileNotFoundError(f"Missing theme directory: {theme_dir}")


def _resolve_reference_example(app_dir):
    app_name = app_dir.name
    background = _first_existing(
        app_dir / f"{app_name}_background.png",
        app_dir / "background.png",
    )
    foreground = _first_existing(
        app_dir / f"{app_name}_foreground.png",
        app_dir / "foreground.png",
    )
    style_ref = _first_existing(
        app_dir / f"{app_name}_style_ref.jpg",
        app_dir / f"{app_name}_style_ref.png",
        app_dir / "style_ref.jpg",
        app_dir / "style_ref.png",
    )

    if not background or not foreground or not style_ref:
        return None

    return {
        "app_name": app_name,
        "background_path": str(background),
        "foreground_path": str(foreground),
        "style_ref_path": str(style_ref),
    }


def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return None
