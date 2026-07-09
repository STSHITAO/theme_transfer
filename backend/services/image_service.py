from pathlib import Path
import math

from PIL import Image


DEFAULT_SIZE = (1024, 1024)


def compose_reference_layouts(reference_examples, case_id, root_dir=None, size=DEFAULT_SIZE):
    root = Path(root_dir) if root_dir else Path.cwd()
    output_dir = root / "data" / "cases" / case_id / "reference_layouts"
    layouts = []

    for example in reference_examples:
        layout_path = output_dir / f"{example['app_name']}_layout.png"
        _save_resized_image(example["original_path"], layout_path, size=size)
        updated = dict(example)
        updated["reference_layout_path"] = str(layout_path)
        layouts.append(updated)

    return layouts


def prepare_target_layout(target_image_path, case_id, root_dir=None, size=DEFAULT_SIZE, output_path=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    output_path = Path(output_path) if output_path else root / "data" / "cases" / case_id / "target_layout.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = _open_rgba(target_image_path, size)
    target.save(output_path, format="PNG")
    return str(output_path)


def compose_contact_sheet(image_paths, output_path, tile_size=(512, 512), columns=None):
    if not image_paths:
        raise ValueError("No images available for contact sheet.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(image_paths)
    column_count = columns or math.ceil(math.sqrt(count))
    row_count = math.ceil(count / column_count)
    sheet = Image.new(
        "RGBA",
        (column_count * tile_size[0], row_count * tile_size[1]),
        (255, 255, 255, 0),
    )

    for index, image_path in enumerate(image_paths):
        tile = _open_rgba(image_path, tile_size)
        x = (index % column_count) * tile_size[0]
        y = (index // column_count) * tile_size[1]
        sheet.alpha_composite(tile, (x, y))

    sheet.save(output, format="PNG")
    return str(output)


def _open_rgba(path, size):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        if rgba.size != size:
            rgba = rgba.resize(size, Image.Resampling.LANCZOS)
        return rgba


def _save_resized_image(image_path, output_path, size):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = _open_rgba(image_path, size)
    image.save(output, format="PNG")
    return str(output)
