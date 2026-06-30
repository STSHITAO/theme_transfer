import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.services.image_service import (
    compose_layout,
    compose_reference_layouts,
    compose_target_layout,
)


def make_image(path: Path, size, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


class ImageServiceTests(unittest.TestCase):
    def test_compose_layout_resizes_to_1024_and_preserves_rgba(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            background = root / "background.png"
            foreground = root / "foreground.png"
            output = root / "layout.png"
            make_image(background, (32, 48), (255, 0, 0, 255))
            make_image(foreground, (16, 16), (0, 255, 0, 128))

            result = compose_layout(background, foreground, output)

            self.assertEqual(str(output), result)
            with Image.open(output) as image:
                self.assertEqual(image.size, (1024, 1024))
                self.assertEqual(image.mode, "RGBA")

    def test_compose_reference_and_target_layouts_return_expected_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_id = "case_001_theme_001_to_xiaohongshu"
            ref_background = root / "ref/background.png"
            ref_foreground = root / "ref/foreground.png"
            target_background = root / "target/background.png"
            target_foreground = root / "target/foreground.png"
            for path in [ref_background, ref_foreground, target_background, target_foreground]:
                make_image(path, (8, 8), (0, 0, 255, 255))

            examples = [
                {
                    "app_name": "wechat",
                    "background_path": str(ref_background),
                    "foreground_path": str(ref_foreground),
                    "style_ref_path": str(root / "ref/style_ref.jpg"),
                }
            ]

            reference_layouts = compose_reference_layouts(examples, case_id, root_dir=root)
            target_layout = compose_target_layout(
                str(target_background),
                str(target_foreground),
                case_id,
                root_dir=root,
            )

            self.assertEqual(len(reference_layouts), 1)
            self.assertTrue(reference_layouts[0]["reference_layout_path"].endswith("wechat_layout.png"))
            self.assertTrue(Path(reference_layouts[0]["reference_layout_path"]).exists())
            self.assertTrue(target_layout.endswith("target_layout.png"))
            self.assertTrue(Path(target_layout).exists())


if __name__ == "__main__":
    unittest.main()
