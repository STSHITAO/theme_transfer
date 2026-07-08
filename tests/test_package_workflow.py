import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.package_workflow import _best_scored_candidate, run_package_workflow, scan_target_apps


def make_png(path: Path, color=(120, 30, 40, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (16, 16), color).save(path)


def make_jpg(path: Path, color=(120, 30, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def make_project_fixture(root: Path) -> None:
    for app_name in ["alipay", "douyin", "wechat"]:
        make_png(root / f"data/styles/theme_001/{app_name}/{app_name}_background.png")
        make_png(root / f"data/styles/theme_001/{app_name}/{app_name}_foreground.png")
        make_jpg(root / f"data/styles/theme_001/{app_name}/{app_name}_style_ref.jpg")
    (root / "data/styles/theme_001/theme.json").write_text(
        json.dumps(
            {
                "theme_id": "theme_001",
                "description": "application icon theme reference pack",
                "examples": {
                    "alipay": {
                        "app": "alipay",
                        "display_name": "Alipay",
                        "category": "payment",
                        "core_function": "payment and money transfer",
                    },
                    "douyin": {
                        "app": "douyin",
                        "display_name": "Douyin",
                        "category": "video community",
                        "core_function": "watch and publish short videos",
                    },
                    "wechat": {
                        "app": "wechat",
                        "display_name": "WeChat",
                        "category": "messaging",
                        "core_function": "chat and social communication",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    make_png(root / "data/targets/bilibili/bilibili.png", color=(20, 80, 200, 255))
    make_jpg(root / "data/targets/qq/qq.jpg", color=(20, 200, 80))
    make_png(root / "data/targets/xiaohongshu/xiaohongshu.png", color=(200, 40, 80, 255))
    for app_name, display_name, category, core_function in [
        ("bilibili", "Bilibili", "video community", "watch and publish videos"),
        ("qq", "QQ", "messaging", "chat and group communication"),
        ("xiaohongshu", "Xiaohongshu", "content community", "publish and browse image, video, and note content"),
    ]:
        (root / f"data/targets/{app_name}/target.json").write_text(
            json.dumps(
                {
                    "app": app_name,
                    "display_name": display_name,
                    "category": category,
                    "core_function": core_function,
                }
            ),
            encoding="utf-8",
        )

    prompts = root / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "qwen_theme_analysis.md").write_text("分析主题规则", encoding="utf-8")
    (prompts / "qwen_qc.md").write_text("单图质检", encoding="utf-8")
    (prompts / "wan_generation.md").write_text("生成模板", encoding="utf-8")
    (prompts / "qwen_package_qc.md").write_text("整包质检", encoding="utf-8")
    (prompts / "qwen_target_identity.md").write_text("目标身份分析", encoding="utf-8")
    (prompts / "qwen_transfer_plan.md").write_text("迁移计划", encoding="utf-8")
    (prompts / "qwen_theme_design_analysis.md").write_text("主题设计分析", encoding="utf-8")
    (prompts / "qwen_identity_strategy.md").write_text("身份表达策略", encoding="utf-8")


class PackageWorkflowTests(unittest.TestCase):
    def test_scan_target_apps_returns_dirs_with_valid_target_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_project_fixture(root)
            (root / "data/targets/broken").mkdir(parents=True)

            result = scan_target_apps(root_dir=root)

            self.assertEqual(result, ["bilibili", "qq", "xiaohongshu"])

    def test_full_mock_package_workflow_outputs_package_artifacts_without_real_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_project_fixture(root)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                with patch("backend.services.qwen_client._call_qwen", side_effect=AssertionError("real Qwen called")):
                    with patch("backend.services.wan_client._call_wan", side_effect=AssertionError("real Wan called")):
                        result = run_package_workflow(
                            "theme_001",
                            "package_001_theme_001",
                            root_dir=root,
                        )

            package_dir = Path(result["package_dir"])
            self.assertTrue((package_dir / "theme_rules.json").exists())
            self.assertTrue((package_dir / "theme_style_analysis.json").exists())
            self.assertTrue((package_dir / "theme_design_analysis.json").exists())
            self.assertTrue((package_dir / "generation_base_prompt.txt").exists())
            self.assertTrue((package_dir / "target_apps.json").exists())
            self.assertTrue((package_dir / "contact_sheet.png").exists())
            self.assertTrue((package_dir / "package_qc_report.json").exists())
            self.assertTrue((package_dir / "metadata.json").exists())

            target_apps = json.loads((package_dir / "target_apps.json").read_text(encoding="utf-8"))
            self.assertEqual(target_apps, ["bilibili", "qq", "xiaohongshu"])
            self.assertEqual(result["target_apps"], target_apps)
            self.assertIn("theme_design_analysis_path", result)

            theme_design = json.loads((package_dir / "theme_design_analysis.json").read_text(encoding="utf-8"))
            self.assertIn("theme_board", theme_design)
            self.assertIn("identity_handling_policy", theme_design)

            for app_name in target_apps:
                case_dir = package_dir / "cases" / app_name
                self.assertTrue((case_dir / "target_layout.png").exists(), app_name)
                self.assertTrue((case_dir / "target_identity.json").exists(), app_name)
                self.assertTrue((case_dir / "identity_strategy.json").exists(), app_name)
                self.assertTrue((case_dir / "transfer_plan.json").exists(), app_name)
                self.assertTrue((case_dir / "generation_prompt.txt").exists(), app_name)
                self.assertTrue((case_dir / "qc_report.json").exists(), app_name)
                self.assertTrue((case_dir / "best_output.png").exists(), app_name)
                self.assertEqual(len(list((case_dir / "candidates").glob("candidate_*.png"))), 3)
                self.assertTrue((package_dir / "final" / f"{app_name}.png").exists(), app_name)

                transfer_plan = json.loads((case_dir / "transfer_plan.json").read_text(encoding="utf-8"))
                identity_strategy = json.loads((case_dir / "identity_strategy.json").read_text(encoding="utf-8"))
                generation_prompt = (case_dir / "generation_prompt.txt").read_text(encoding="utf-8")
                self.assertEqual(transfer_plan["app"], app_name)
                self.assertEqual(identity_strategy["app"], app_name)
                self.assertIn(identity_strategy["identity_constraint_level"], ["strict", "balanced", "flexible"])
                self.assertIn("generation_direction", identity_strategy)
                self.assertIn("strategy_type", transfer_plan)
                self.assertIn("identity_constraint_level", transfer_plan)
                self.assertIn("must_preserve", transfer_plan)
                self.assertIn("forbid", transfer_plan)
                self.assertIn("transfer_plan", generation_prompt)

            with Image.open(package_dir / "contact_sheet.png") as sheet:
                self.assertEqual(sheet.mode, "RGBA")
                self.assertGreater(sheet.size[0], 0)
                self.assertGreater(sheet.size[1], 0)

            package_qc = json.loads((package_dir / "package_qc_report.json").read_text(encoding="utf-8"))
            self.assertEqual(package_qc["package_consistency_score"], 8)
            self.assertEqual(package_qc["accepted_apps"], target_apps)

            metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["mock_mode"])
            self.assertEqual(metadata["target_apps"], target_apps)
            self.assertEqual(len(metadata["final_outputs"]), len(target_apps))
            self.assertIn("theme_design_analysis", metadata)

    def test_best_selection_prefers_identity_safe_candidate_over_high_overall_low_identity(self):
        candidate_paths = [
            r"C:\tmp\candidate_01.png",
            r"C:\tmp\candidate_02.png",
        ]
        qc_report = {
            "candidates": [
                {"file": "candidate_01.png", "overall_score": 96, "target_identity_score": 40},
                {"file": "candidate_02.png", "overall_score": 82, "target_identity_score": 85},
            ]
        }

        result = _best_scored_candidate(qc_report, candidate_paths, identity_threshold=75)

        self.assertEqual(result, candidate_paths[1])

    def test_best_selection_marks_needs_retry_when_all_candidates_have_low_identity(self):
        candidate_paths = [
            r"C:\tmp\candidate_01.png",
            r"C:\tmp\candidate_02.png",
        ]
        qc_report = {
            "candidates": [
                {"file": "candidate_01.png", "overall_score": 96, "target_identity_score": 40},
                {"file": "candidate_02.png", "overall_score": 82, "target_identity_score": 55},
            ]
        }

        result = _best_scored_candidate(qc_report, candidate_paths, identity_threshold=75)

        self.assertEqual(result, candidate_paths[0])
        self.assertTrue(qc_report["needs_retry"])


if __name__ == "__main__":
    unittest.main()
