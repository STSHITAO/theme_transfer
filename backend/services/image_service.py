from pathlib import Path

from PIL import Image


DEFAULT_SIZE = (1024, 1024)


def compose_layout(background_path, foreground_path, output_path, size=DEFAULT_SIZE):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    background = _open_rgba(background_path, size)
    foreground = _open_rgba(foreground_path, size)
    composed = Image.alpha_composite(background, foreground)
    composed.save(output, format="PNG")
    return str(output)


def compose_reference_layouts(reference_examples, case_id, root_dir=None, size=DEFAULT_SIZE):
    root = Path(root_dir) if root_dir else Path.cwd()
    output_dir = root / "data" / "cases" / case_id / "reference_layouts"
    layouts = []

    for example in reference_examples:
        layout_path = output_dir / f"{example['app_name']}_layout.png"
        compose_layout(
            example["background_path"],
            example["foreground_path"],
            layout_path,
            size=size,
        )
        updated = dict(example)
        updated["reference_layout_path"] = str(layout_path)
        layouts.append(updated)

    return layouts


def compose_target_layout(target_background, target_foreground, case_id, root_dir=None, size=DEFAULT_SIZE):
    root = Path(root_dir) if root_dir else Path.cwd()
    output_path = root / "data" / "cases" / case_id / "target_layout.png"
    return compose_layout(target_background, target_foreground, output_path, size=size)


def _open_rgba(path, size):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        if rgba.size != size:
            rgba = rgba.resize(size, Image.Resampling.LANCZOS)
        return rgba
