import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend import package_workflow
from backend.package_workflow import _best_scored_candidate, run_package_workflow, scan_target_apps
from backend.services.wan_client import WanApiError


def make_png(path: Path, color=(120, 30, 40, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (16, 16), color).save(path)


def make_jpg(path: Path, color=(120, 30, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def make_project_fixture(root: Path) -> None:
    for app_name in ["alipay", "douyin", "wechat"]:
        make_png(root / f"data/styles/theme_001/{app_name}/{app_name}.png")
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
    def test_case_reference_selection_matches_route_and_excludes_target(self):
        examples = [
            {"app_name": name, "style_ref_path": f"/{name}.png"}
            for name in ["a", "b", "c", "d", "target"]
        ]
        design = {
            "reference_transformation_patterns": [
                {"app": "a", "preserve_major_structure": True},
                {"app": "b", "preserve_major_structure": False},
                {"app": "c", "preserve_major_structure": False},
                {"app": "d", "preserve_major_structure": False},
                {"app": "target", "preserve_major_structure": False},
            ]
        }

        selected = package_workflow._select_case_reference_examples(
            examples,
            design,
            {"structure_preservation_mode": "semantic_recompose"},
            "target",
            limit=3,
        )

        self.assertEqual({item["app_name"] for item in selected}, {"b", "c", "d"})
        self.assertNotIn("target", [item["app_name"] for item in selected])

    def test_batch_records_data_inspection_failure_and_continues_other_apps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_project_fixture(root)
            original_run_case = package_workflow._run_package_case

            def run_case_or_reject(target_app, *args, **kwargs):
                if target_app == "qq":
                    raise WanApiError(
                        "Wan API rejected the image request: DataInspectionFailed",
                        status_code=400,
                        code="DataInspectionFailed",
                        response_path=str(root / "rejected_response.json"),
                    )
                return original_run_case(target_app, *args, **kwargs)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                with patch.object(package_workflow, "_run_package_case", side_effect=run_case_or_reject):
                    result = run_package_workflow(
                        "theme_001",
                        "package_001_theme_001",
                        root_dir=root,
                        skip_rejected_cases=True,
                    )

            self.assertEqual(set(result["final_outputs"]), {"bilibili", "xiaohongshu"})
            self.assertEqual(result["coverage"]["skipped_apps"], ["qq"])
            self.assertEqual(result["coverage"]["successful_app_count"], 2)
            self.assertTrue((Path(result["package_dir"]) / "cases/qq/case_failure.json").exists())
            metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "complete_with_skips")
            self.assertEqual(metadata["evaluation_coverage"]["skipped_apps"], ["qq"])

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
            self.assertIn("color_transform_rule", theme_design)
            self.assertIn("stroke_transform_rule", theme_design)
            self.assertIn("composition_transform_rule", theme_design)
            self.assertIn("theme_fidelity_constraints", theme_design)

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
                self.assertIn("identity_anchor", identity_strategy)
                self.assertIn("brand_cues_to_preserve", identity_strategy)
                self.assertIn("semantic_cues_to_preserve", identity_strategy)
                self.assertIn("style_fidelity_priority", identity_strategy)
                self.assertIn(
                    identity_strategy["structure_preservation_mode"],
                    ["preserve_major_structure", "semantic_recompose"],
                )
                self.assertEqual(
                    identity_strategy["structure_identity_metric_applicable"],
                    identity_strategy["structure_preservation_mode"] == "preserve_major_structure",
                )
                self.assertIn("strategy_type", transfer_plan)
                self.assertIn("identity_constraint_level", transfer_plan)
                self.assertIn("must_preserve", transfer_plan)
                self.assertIn("forbid", transfer_plan)
                self.assertIn("color_application", transfer_plan)
                self.assertIn("stroke_application", transfer_plan)
                self.assertIn("composition_application", transfer_plan)
                self.assertIn("identity_application", transfer_plan)
                self.assertIn("fidelity_constraints", transfer_plan)
                self.assertIn("negative_constraints", transfer_plan)
                self.assertEqual(
                    transfer_plan["structure_preservation_mode"],
                    identity_strategy["structure_preservation_mode"],
                )
                self.assertEqual(
                    transfer_plan["structure_identity_metric_applicable"],
                    identity_strategy["structure_identity_metric_applicable"],
                )
                self.assertIn("ABSOLUTE IDENTITY LOCK", generation_prompt)
                self.assertIn("STYLE_REFERENCE images may contribute only visual treatment", generation_prompt)
                self.assertIn("theme fidelity", generation_prompt.lower())

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
            self.assertEqual(
                set(metadata["structure_evaluation_policy"]),
                set(target_apps),
            )

    def test_rerun_replaces_final_directory_without_stale_apps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_project_fixture(root)
            final_dir = root / "data/packages/package_001_theme_001/final"
            make_png(final_dir / "removed_app.png")

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                run_package_workflow(
                    "theme_001",
                    "package_001_theme_001",
                    root_dir=root,
                )

            self.assertEqual(
                {path.name for path in final_dir.iterdir()},
                {"bilibili.png", "qq.png", "xiaohongshu.png"},
            )

    def test_resume_reuses_complete_cases_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_project_fixture(root)
            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                run_package_workflow(
                    "theme_001",
                    "package_resume",
                    root_dir=root,
                    candidate_count=1,
                )
                with patch(
                    "backend.package_workflow._run_package_case",
                    side_effect=AssertionError("completed case was regenerated"),
                ):
                    result = run_package_workflow(
                        "theme_001",
                        "package_resume",
                        root_dir=root,
                        candidate_count=1,
                        resume=True,
                    )

            self.assertTrue(all(case.get("resumed") for case in result["cases"].values()))
            self.assertEqual(len(result["final_outputs"]), 3)

    def test_failed_rerun_preserves_previous_final_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_project_fixture(root)
            final_dir = root / "data/packages/package_001_theme_001/final"
            previous_output = final_dir / "previous.png"
            make_png(previous_output, color=(1, 2, 3, 255))
            previous_bytes = previous_output.read_bytes()

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                with patch("backend.package_workflow.run_package_qc", side_effect=RuntimeError("QC failed")):
                    with self.assertRaisesRegex(RuntimeError, "QC failed"):
                        run_package_workflow(
                            "theme_001",
                            "package_001_theme_001",
                            root_dir=root,
                        )

            self.assertEqual([path.name for path in final_dir.iterdir()], ["previous.png"])
            self.assertEqual(previous_output.read_bytes(), previous_bytes)

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
