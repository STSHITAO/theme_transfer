import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.workflow import run_workflow


def make_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (16, 16), (120, 30, 40, 255)).save(path)


def make_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (120, 30, 40)).save(path)


class WorkflowTests(unittest.TestCase):
    def test_full_mock_workflow_outputs_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for app_name in ["wechat", "alipay"]:
                make_png(root / f"data/styles/theme_001/{app_name}/{app_name}_background.png")
                make_png(root / f"data/styles/theme_001/{app_name}/{app_name}_foreground.png")
                make_jpg(root / f"data/styles/theme_001/{app_name}/{app_name}_style_ref.jpg")
            make_png(root / "data/targets/xiaohongshu/background.png")
            make_png(root / "data/targets/xiaohongshu/foreground.png")
            (root / "prompts").mkdir()
            (root / "prompts/qwen_theme_analysis.md").write_text("分析", encoding="utf-8")
            (root / "prompts/qwen_qc.md").write_text("质检", encoding="utf-8")
            (root / "prompts/wan_generation.md").write_text("生成", encoding="utf-8")

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = run_workflow(
                    "theme_001",
                    "xiaohongshu",
                    "case_001_theme_001_to_xiaohongshu",
                    root_dir=root,
                )

            self.assertEqual(result["used_reference_examples"], ["alipay", "wechat"])
            for key in [
                "target_layout_path",
                "theme_style_analysis_path",
                "generation_prompt_path",
                "best_output_path",
                "qc_report_path",
                "metadata_path",
            ]:
                self.assertTrue(Path(result[key]).exists(), key)
            self.assertEqual(len(result["candidate_paths"]), 4)
            self.assertEqual(len(result["reference_layout_paths"]), 2)
            self.assertNotIn("target_sheet_path", result)
            self.assertNotIn("reference_sheet_paths", result)


if __name__ == "__main__":
    unittest.main()
