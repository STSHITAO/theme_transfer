from pathlib import Path


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def resolve_case_paths(theme_id, target_app, root_dir=None, max_examples=5):
    root = Path(root_dir) if root_dir else Path.cwd()
    examples = resolve_theme_examples(theme_id, root_dir=root, max_examples=max_examples)
    target_inputs = _resolve_target_inputs(root, target_app)

    return {
        "theme_id": theme_id,
        "target_app": target_app,
        "reference_examples": examples,
        **target_inputs,
    }


def resolve_theme_examples(theme_id, root_dir=None, max_examples=5):
    root = Path(root_dir) if root_dir else Path.cwd()
    theme_dir = _find_theme_dir(root, theme_id)
    examples = []
    for app_dir in sorted([item for item in theme_dir.iterdir() if item.is_dir()]):
        example = _resolve_reference_example(app_dir)
        if example:
            examples.append(example)
        if len(examples) >= max_examples:
            break

    if not examples:
        raise ValueError(f"No valid reference examples found in theme: {theme_dir}")
    return examples


def resolve_target_inputs(target_app, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    return _resolve_target_inputs(root, target_app)


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
        app_dir / f"{app_name}_transferred_ref.jpg",
        app_dir / f"{app_name}_transferred_ref.png",
        app_dir / "style_ref.jpg",
        app_dir / "style_ref.png",
        app_dir / "transferred_ref.jpg",
        app_dir / "transferred_ref.png",
        app_dir / "reference.jpg",
        app_dir / "reference.png",
    )

    if not background or not foreground or not style_ref:
        return None

    return {
        "app_name": app_name,
        "background_path": str(background),
        "foreground_path": str(foreground),
        "style_ref_path": str(style_ref),
    }


def _resolve_target_inputs(root, target_app):
    target_dir = root / "data" / "targets" / target_app
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Missing target directory: {target_dir}")

    legacy_background = target_dir / "background.png"
    legacy_foreground = target_dir / "foreground.png"
    if legacy_background.exists() and legacy_foreground.exists():
        return {
            "target_background": str(legacy_background),
            "target_foreground": str(legacy_foreground),
            "target_image": str(legacy_foreground),
        }

    target_image = _find_single_target_image(target_dir, target_app)
    if not target_image:
        expected = ", ".join(
            [
                f"{target_app}.png/.jpg/.jpeg/.webp",
                "target.png/.jpg/.jpeg/.webp",
                "or exactly one image file in the target directory",
            ]
        )
        raise FileNotFoundError(f"Missing target image in {target_dir}; expected {expected}")

    return {"target_image": str(target_image)}


def _find_single_target_image(target_dir, target_app):
    preferred_names = [target_app, "target", "original", "input", "image"]
    for stem in preferred_names:
        target = _first_existing(*[target_dir / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES])
        if target:
            return target

    images = sorted([item for item in target_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES])
    if len(images) == 1:
        return images[0]
    return None


def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return None
