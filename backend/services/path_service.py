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
    style_ref = _find_style_ref(app_dir, app_name)
    original = _find_original_image(app_dir, app_name)
    if not original or not style_ref:
        return None

    return {
        "app_name": app_name,
        "original_path": str(original),
        "reference_raw_path": str(original),
        "style_ref_path": str(style_ref),
    }


def _find_style_ref(app_dir, app_name):
    return _first_existing(
        *[app_dir / f"{app_name}_style_ref{suffix}" for suffix in IMAGE_SUFFIXES],
        *[app_dir / f"{app_name}_transferred_ref{suffix}" for suffix in IMAGE_SUFFIXES],
        *[app_dir / f"style_ref{suffix}" for suffix in IMAGE_SUFFIXES],
        *[app_dir / f"transferred_ref{suffix}" for suffix in IMAGE_SUFFIXES],
        *[app_dir / f"reference{suffix}" for suffix in IMAGE_SUFFIXES],
    )


def _find_original_image(app_dir, app_name):
    preferred_names = [app_name, "original", "raw", "input", "image"]
    for stem in preferred_names:
        target = _first_existing(*[app_dir / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES])
        if target:
            return target

    images = sorted(
        item
        for item in app_dir.iterdir()
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and not _is_reference_output_name(item.stem.lower())
    )
    if len(images) == 1:
        return images[0]
    return None


def _is_reference_output_name(stem):
    excluded_tokens = ["style_ref", "transferred_ref", "reference", "background", "foreground"]
    return any(token in stem for token in excluded_tokens)


def _resolve_target_inputs(root, target_app):
    target_dir = root / "data" / "targets" / target_app
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Missing target directory: {target_dir}")

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
