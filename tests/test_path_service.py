import tempfile
import unittest
from pathlib import Path

from backend.services.path_service import resolve_case_paths


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


class PathServiceTests(unittest.TestCase):
    def test_resolves_theme_001_with_prefixed_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch(root / "data/styles/theme_001/wechat/wechat_background.png")
            touch(root / "data/styles/theme_001/wechat/wechat_foreground.png")
            touch(root / "data/styles/theme_001/wechat/wechat_style_ref.jpg")
            touch(root / "data/targets/xiaohongshu/xiaohongshu.png")

            result = resolve_case_paths("theme_001", "xiaohongshu", root_dir=root)

            self.assertEqual(result["theme_id"], "theme_001")
            self.assertEqual(result["target_app"], "xiaohongshu")
            self.assertTrue(result["target_image"].endswith("xiaohongshu.png"))
            self.assertEqual(len(result["reference_examples"]), 1)
            example = result["reference_examples"][0]
            self.assertEqual(example["app_name"], "wechat")
            self.assertTrue(example["background_path"].endswith("wechat_background.png"))
            self.assertTrue(example["foreground_path"].endswith("wechat_foreground.png"))
            self.assertTrue(example["style_ref_path"].endswith("wechat_style_ref.jpg"))

    def test_supports_simplified_reference_names_and_limits_to_five(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(6):
                app_name = f"app{index}"
                touch(root / f"data/styles/theme_001/{app_name}/background.png")
                touch(root / f"data/styles/theme_001/{app_name}/foreground.png")
                touch(root / f"data/styles/theme_001/{app_name}/style_ref.jpg")
            touch(root / "data/targets/xiaohongshu/target.jpg")

            result = resolve_case_paths("theme_001", "xiaohongshu", root_dir=root)

            self.assertEqual(len(result["reference_examples"]), 5)
            self.assertEqual(
                [item["app_name"] for item in result["reference_examples"]],
                ["app0", "app1", "app2", "app3", "app4"],
            )

    def test_raises_clear_error_when_target_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch(root / "data/styles/theme_001/wechat/background.png")
            touch(root / "data/styles/theme_001/wechat/foreground.png")
            touch(root / "data/styles/theme_001/wechat/style_ref.jpg")

            with self.assertRaises(FileNotFoundError) as context:
                resolve_case_paths("theme_001", "xiaohongshu", root_dir=root)

            self.assertIn("Missing target directory", str(context.exception))

    def test_raises_clear_error_when_target_directory_has_multiple_unresolved_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch(root / "data/styles/theme_001/wechat/background.png")
            touch(root / "data/styles/theme_001/wechat/foreground.png")
            touch(root / "data/styles/theme_001/wechat/style_ref.jpg")
            touch(root / "data/targets/xiaohongshu/a.png")
            touch(root / "data/targets/xiaohongshu/b.jpg")

            with self.assertRaises(FileNotFoundError) as context:
                resolve_case_paths("theme_001", "xiaohongshu", root_dir=root)

            self.assertIn("Missing target image", str(context.exception))

    def test_supports_legacy_target_background_and_foreground(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch(root / "data/styles/theme_001/wechat/background.png")
            touch(root / "data/styles/theme_001/wechat/foreground.png")
            touch(root / "data/styles/theme_001/wechat/style_ref.jpg")
            touch(root / "data/targets/xiaohongshu/background.png")
            touch(root / "data/targets/xiaohongshu/foreground.png")

            result = resolve_case_paths("theme_001", "xiaohongshu", root_dir=root)

            self.assertTrue(result["target_background"].endswith("background.png"))
            self.assertTrue(result["target_foreground"].endswith("foreground.png"))
            self.assertTrue(result["target_image"].endswith("foreground.png"))

    def test_raises_clear_error_when_no_valid_reference_examples_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data/styles/theme_001/wechat").mkdir(parents=True)
            touch(root / "data/targets/xiaohongshu/xiaohongshu.png")

            with self.assertRaises(ValueError) as context:
                resolve_case_paths("theme_001", "xiaohongshu", root_dir=root)

            self.assertIn("No valid reference examples", str(context.exception))

    def test_does_not_map_theme001_to_theme_001(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            touch(root / "data/styles/theme_001/wechat/wechat_background.png")
            touch(root / "data/styles/theme_001/wechat/wechat_foreground.png")
            touch(root / "data/styles/theme_001/wechat/wechat_style_ref.jpg")
            touch(root / "data/targets/xiaohongshu/background.png")
            touch(root / "data/targets/xiaohongshu/foreground.png")

            with self.assertRaises(FileNotFoundError) as context:
                resolve_case_paths("theme001", "xiaohongshu", root_dir=root)

            self.assertIn("data", str(context.exception))
            self.assertIn("theme001", str(context.exception))


if __name__ == "__main__":
    unittest.main()
