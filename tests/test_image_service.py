import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.services.image_service import (
    compose_contact_sheet,
    compose_reference_layouts,
    prepare_target_layout,
)


def make_image(path: Path, size, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


class ImageServiceTests(unittest.TestCase):
    def test_prepare_reference_and_target_layouts_from_single_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_id = "case_001_theme_001_to_xiaohongshu"
            ref_original = root / "ref/wechat.png"
            target_image = root / "target/xiaohongshu.png"
            for path in [ref_original, target_image]:
                make_image(path, (8, 8), (0, 0, 255, 255))

            examples = [
                {
                    "app_name": "wechat",
                    "original_path": str(ref_original),
                    "reference_raw_path": str(ref_original),
                    "style_ref_path": str(root / "ref/style_ref.jpg"),
                }
            ]

            reference_layouts = compose_reference_layouts(examples, case_id, root_dir=root)
            target_layout = prepare_target_layout(str(target_image), case_id, root_dir=root)

            self.assertEqual(len(reference_layouts), 1)
            self.assertTrue(reference_layouts[0]["reference_layout_path"].endswith("wechat_layout.png"))
            self.assertTrue(Path(reference_layouts[0]["reference_layout_path"]).exists())
            self.assertTrue(target_layout.endswith("target_layout.png"))
            self.assertTrue(Path(target_layout).exists())

    def test_prepare_target_layout_resizes_single_target_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "target/bilibili.png"
            make_image(source, (20, 12), (255, 255, 0, 255))

            target_layout = prepare_target_layout(source, "case_001_theme_001_to_bilibili", root_dir=root)

            self.assertTrue(target_layout.endswith("target_layout.png"))
            with Image.open(target_layout) as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.mode, "RGBA")

    def test_compose_contact_sheet_uses_grid_without_text_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_paths = []
            for index in range(3):
                path = root / f"final/app_{index}.png"
                make_image(path, (20, 12), (index * 50, 255, 0, 255))
                image_paths.append(str(path))

            output_path = compose_contact_sheet(image_paths, root / "contact_sheet.png", tile_size=(64, 64))

            self.assertTrue(Path(output_path).exists())
            with Image.open(output_path) as sheet:
                self.assertEqual(sheet.size, (128, 128))
                self.assertEqual(sheet.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
