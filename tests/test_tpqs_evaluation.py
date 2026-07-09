import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from evaluation.services.embedding_service import (
    TpqsConfig,
    prepare_model_cache_dirs,
)
from evaluation.services.eval_path_service import resolve_eval_inputs
from evaluation.services.style_feature_service import extract_style_feature_groups, extract_style_features
from evaluation.services.tpqs_service import (
    _load_qwen_qc_scores,
    _package_consistency_score,
    _style_delta_transfer_score,
)
from evaluation.services.tpqs_feedback_service import build_tpqs_feedback_retry_prompt
from evaluation.tpqs_workflow import run_tpqs


def make_image(path: Path, color=(120, 40, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def make_eval_fixture(root: Path) -> None:
    for app_name, color in [
        ("alipay", (210, 60, 80)),
        ("douyin", (200, 70, 90)),
        ("wechat", (220, 80, 100)),
    ]:
        make_image(
            root / f"data/styles/theme_001/{app_name}/{app_name}.jpg",
            color=(35, 120, 210),
        )
        make_image(
            root / f"data/styles/theme_001/{app_name}/{app_name}_style_ref.jpg",
            color=color,
        )

    final_colors = {
        "bilibili": (205, 65, 85),
        "qq": (215, 75, 95),
        "wps": (225, 85, 105),
    }
    target_colors = {
        "bilibili": (30, 120, 220),
        "qq": (40, 140, 240),
        "wps": (70, 160, 210),
    }
    for app_name, color in final_colors.items():
        make_image(root / f"data/packages/package_001_theme_001/final/{app_name}.png", color=color)
    for app_name, color in target_colors.items():
        make_image(root / f"data/targets/{app_name}/{app_name}.png", color=color)
    package_dir = root / "data/packages/package_001_theme_001"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package_qc_report.json").write_text(
        json.dumps(
            {
                "package_consistency_score": 85,
                "style_consistency_score": 90,
                "target_identity_score": 88,
                "problematic_apps": ["qq"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (package_dir / "theme_design_analysis.json").write_text(
        json.dumps(
            {
                "theme_visual_language": "soft rounded sticker icon pack",
                "color_rule": "warm pastel palette",
                "stroke_rule": "thick rounded outline",
                "composition_rule": "centered subject with generous padding",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def make_low_delta_high_style_fixture(root: Path) -> None:
    for app_name in ["alipay", "douyin", "wechat"]:
        make_image(
            root / f"data/styles/theme_001/{app_name}/{app_name}.jpg",
            color=(205, 65, 85),
        )
        make_image(
            root / f"data/styles/theme_001/{app_name}/{app_name}_style_ref.jpg",
            color=(210, 60, 80),
        )

    for app_name in ["bilibili", "qq", "wps"]:
        make_image(root / f"data/packages/package_001_theme_001/final/{app_name}.png", color=(210, 60, 80))
        make_image(root / f"data/targets/{app_name}/{app_name}.png", color=(30, 120, 220))

    package_dir = root / "data/packages/package_001_theme_001"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package_qc_report.json").write_text(
        json.dumps(
            {
                "package_consistency_score": 92,
                "style_consistency_score": 94,
                "target_identity_score": 91,
                "problematic_apps": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (package_dir / "theme_design_analysis.json").write_text(
        json.dumps(
            {
                "style_eval_text_short": "soft pastel hand drawn app icon with cream rounded background and black sketch outline",
                "salient_style_attributes": [
                    {"name": "color", "importance": "high"},
                    {"name": "background", "importance": "high"},
                    {"name": "stroke", "importance": "medium"},
                    {"name": "texture_material", "importance": "medium"},
                    {"name": "composition", "importance": "low"},
                    {"name": "complexity", "importance": "low"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class TpqsEvaluationTests(unittest.TestCase):
    def test_resolve_eval_inputs_maps_generated_icons_to_matching_target_originals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)

            resolved = resolve_eval_inputs("theme_001", "package_001_theme_001", root_dir=root)

            self.assertEqual([item.app for item in resolved.generated_icons], ["bilibili", "qq", "wps"])
            self.assertEqual(len(resolved.theme_refs), 3)
            self.assertEqual([item.app for item in resolved.theme_examples], ["alipay", "douyin", "wechat"])
            self.assertTrue(resolved.theme_examples[0].original_path.name.endswith("alipay.jpg"))
            self.assertEqual(resolved.theme_examples[0].reference_raw_path, resolved.theme_examples[0].original_path)
            self.assertTrue(resolved.theme_examples[0].style_ref_path.name.endswith("_style_ref.jpg"))
            self.assertEqual(set(resolved.target_originals), {"bilibili", "qq", "wps"})
            self.assertEqual(resolved.missing_apps, [])
            self.assertEqual(resolved.skipped_apps, [])

    def test_stats_backend_generates_report_artifacts_without_api_or_model_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)

            with patch.dict(os.environ, {"TPQS_EMBEDDING_BACKEND": "stats"}, clear=False):
                stale_eval_dir = root / "data/evaluations/eval_001_package_001_theme_001"
                stale_eval_dir.mkdir(parents=True)
                (stale_eval_dir / "embedding_pca.png").write_bytes(b"old")
                (stale_eval_dir / "pairwise_distances.json").write_text("{}", encoding="utf-8")
                result = run_tpqs(
                    "theme_001",
                    "package_001_theme_001",
                    "eval_001_package_001_theme_001",
                    root_dir=root,
                )

            eval_dir = Path(result["eval_dir"])
            self.assertTrue((eval_dir / "tpqs_report.json").exists())
            self.assertTrue((eval_dir / "metrics.csv").exists())
            self.assertFalse((eval_dir / "embedding_pca.png").exists())
            self.assertFalse((eval_dir / "pairwise_distances.json").exists())
            self.assertTrue((eval_dir / "style_pairwise_distances.json").exists())
            self.assertTrue((eval_dir / "style_delta_distances.json").exists())
            self.assertTrue((eval_dir / "dino_pairwise_distances.json").exists())
            self.assertTrue((eval_dir / "inputs_manifest.json").exists())
            self.assertTrue((eval_dir / "tpqs_feedback_retry_prompt.md").exists())
            self.assertTrue((eval_dir / "generation_feedback_prompt.md").exists())

            report = json.loads((eval_dir / "tpqs_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((eval_dir / "inputs_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation_framework"], "ITTE")
            self.assertEqual(report["itte_version"], "v1.1")
            self.assertEqual(report["itte_score"], report["tpqs"])
            for key in [
                "style_transfer_effectiveness",
                "style_cue_profile_match",
                "style_attribute_transfer_scores",
                "package_coherence",
                "app_identity_coherence",
                "visual_quality",
                "strict_delta_diagnostics",
                "diagnostics",
                "warnings",
                "auxiliary_scores",
            ]:
                self.assertIn(key, report)
            self.assertEqual(
                report["style_transfer_score"],
                report["style_transfer_effectiveness"]["score"],
            )
            self.assertEqual(
                report["strict_delta_diagnostics"]["strict_delta_transfer_score"],
                report["style_delta_transfer_score"],
            )
            self.assertEqual(
                report["legacy_style_delta_transfer_score"],
                report["strict_delta_diagnostics"]["strict_delta_transfer_score"],
            )
            self.assertIn(
                report["decision"],
                [
                    "style_transfer_success",
                    "package_coherent_but_theme_transfer_weak",
                    "theme_transfer_effective_but_package_unity_weak",
                    "style_transfer_success_with_identity_risk",
                    "visual_quality_problem",
                    "needs_review",
                ],
            )
            self.assertFalse(report["is_official_tpqs"])
            self.assertEqual(report["embedding_backend"], "stats")
            self.assertEqual(report["model_source"], "modelscope")
            self.assertEqual(report["style_feature_backend"], "color_edge_composition")
            self.assertEqual(report["tpqs"], report["tpqs_primary_score"])
            self.assertIn("strict_delta_tpqs_score", report)
            self.assertIn("primary_scores", report)
            self.assertIn("diagnostic_scores", report)
            self.assertIn("risk_scores", report)
            self.assertIn("qwen_qc_scores", report)
            self.assertIn("style_cue_profile_match_score", report["style_transfer_effectiveness"])
            self.assertIn("theme_prompt_image_alignment_score", report["style_transfer_effectiveness"])
            self.assertIn("target_structure_retention_score", report["app_identity_coherence"])
            self.assertIn("generation_qwen_qc_prior", report["diagnostics"])
            self.assertIsNone(report["primary_scores"]["theme_style_text_fit_score"])
            self.assertFalse(report["style_transfer_effectiveness"]["theme_style_text_fit_reliable"])
            self.assertFalse(report["details"]["theme_style_text_fit"]["openclip_enabled"])
            self.assertEqual(report["qwen_qc_scores"]["package_consistency_score"], 85)
            self.assertEqual(report["qwen_qc_scores"]["problematic_apps"], ["qq"])
            self.assertEqual(len(manifest["theme_transfer_examples"]), 3)
            self.assertIn("original_path", manifest["theme_transfer_examples"][0])
            self.assertIn("reference_raw_path", manifest["theme_transfer_examples"][0])
            self.assertNotIn("background_path", manifest["theme_transfer_examples"][0])
            self.assertNotIn("foreground_path", manifest["theme_transfer_examples"][0])
            self.assertIn("style_delta_transfer", report["details"])
            self.assertIn(report["decision"], [
                "style_transfer_success",
                "package_coherent_but_theme_transfer_weak",
                "theme_transfer_effective_but_package_unity_weak",
                "style_transfer_success_with_identity_risk",
                "visual_quality_problem",
                "needs_review",
            ])
            self.assertIn("tpqs_summary", report)
            for section in [
                "style_transfer_diagnostics",
                "package_unity_diagnostics",
                "identity_diagnostics",
                "visual_quality_diagnostics",
            ]:
                self.assertIn(section, report["tpqs_summary"])
            self.assertIn("diagnosis_summary", report)
            for key in [
                "main_conclusion",
                "strong_points",
                "weak_points",
                "metric_warning",
            ]:
                self.assertIn(key, report["diagnosis_summary"])
            self.assertIn(
                "Strict Delta Diagnostics",
                " ".join(report["diagnosis_summary"]["metric_warning"]),
            )
            self.assertIn("color_delta_score", report)
            self.assertIn("edge_delta_score", report)
            self.assertIn("composition_delta_score", report)
            self.assertIn("complexity_delta_score", report)
            self.assertIn("visual_stats_transfer_score", report)
            self.assertIn("visual_artifact_quality_score", report)
            for key in [
                "tpqs",
                "tpqs_primary_score",
                "strict_delta_tpqs_score",
                "style_transfer_score",
                "style_delta_transfer_score",
                "color_delta_score",
                "edge_delta_score",
                "composition_delta_score",
                "complexity_delta_score",
                "package_internal_style_consistency_score",
                "reference_style_distribution_match_score",
                "theme_membership_score",
                "identity_separability_score",
                "visual_quality_score",
                "visual_stats_transfer_score",
                "visual_artifact_quality_score",
            ]:
                self.assertGreaterEqual(report[key], 0)
                self.assertLessEqual(report[key], 100)
            for key in [
                "theme_style_image_fit_score",
                "package_unity_score",
                "theme_membership_score",
                "visual_artifact_quality_score",
            ]:
                self.assertIn(key, report["primary_scores"])
                self.assertGreaterEqual(report["primary_scores"][key], 0)
                self.assertLessEqual(report["primary_scores"][key], 100)
            self.assertEqual(report["diagnostic_scores"]["style_delta_transfer_score"], report["style_delta_transfer_score"])
            self.assertEqual(
                report["risk_scores"]["dino_identity_top1_accuracy"],
                report["identity_top1_accuracy"],
            )

            metrics_text = (eval_dir / "metrics.csv").read_text(encoding="utf-8")
            metrics_header = metrics_text.splitlines()[0].split(",")
            for new_field in [
                "itte_score",
                "style_transfer_score",
                "theme_style_image_transfer_score",
                "style_cue_profile_match_score",
                "theme_prompt_image_alignment_score",
                "style_attribute_transfer_score",
                "color_transfer_score",
                "background_transfer_score",
                "stroke_transfer_score",
                "texture_material_transfer_score",
                "composition_transfer_score",
                "complexity_transfer_score",
                "package_coherence_score",
                "app_identity_coherence_score",
                "target_structure_retention_score",
                "over_recomposition_penalty",
                "qwen_identity_score",
                "visual_quality_score",
                "strict_delta_transfer_score",
            ]:
                self.assertIn(new_field, metrics_header)
            self.assertIn("style_delta_transfer_score", metrics_header)
            self.assertIn("d_to_reference_delta_centroid", metrics_header)
            self.assertIn("generated_internal_outlier", metrics_header)
            self.assertIn("theme_membership_score", metrics_header)
            self.assertIn("theme_style_image_fit_score", metrics_header)
            self.assertIn("theme_style_text_fit_score", metrics_header)
            self.assertIn("package_unity_score", metrics_header)
            self.assertIn("visual_artifact_quality_score", metrics_header)
            self.assertIn("dino_identity_structure_risk_score", metrics_header)
            self.assertIn("strict_delta_warning", metrics_header)
            self.assertIn("membership_warning", metrics_header)
            self.assertIn("qwen_problematic_app", metrics_header)
            retry_prompt = (eval_dir / "tpqs_feedback_retry_prompt.md").read_text(encoding="utf-8")
            self.assertIn("theme_001", retry_prompt)
            self.assertIn("theme fidelity", retry_prompt.lower())
            self.assertIn("not an automatic retry mechanism", retry_prompt)

            delta_distances = json.loads((eval_dir / "style_delta_distances.json").read_text(encoding="utf-8"))
            self.assertIn("reference_delta_pairwise", delta_distances)
            self.assertIn("generated_delta_to_reference_delta_centroid", delta_distances)
            self.assertIn("group_scores", delta_distances)
            self.assertEqual(
                set(delta_distances["group_scores"]),
                {"color", "edge", "composition", "complexity"},
            )

    def test_itte_style_transfer_score_is_not_strict_delta_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_low_delta_high_style_fixture(root)

            with patch.dict(os.environ, {"TPQS_EMBEDDING_BACKEND": "stats"}, clear=False):
                result = run_tpqs(
                    "theme_001",
                    "package_001_theme_001",
                    "eval_001_package_001_theme_001",
                    root_dir=root,
                )

            report = result["report"]
            self.assertEqual(report["evaluation_framework"], "ITTE")
            self.assertEqual(report["itte_version"], "v1.1")
            self.assertEqual(report["style_transfer_score"], report["style_transfer_effectiveness"]["score"])
            self.assertNotEqual(report["style_transfer_score"], report["style_delta_transfer_score"])
            self.assertIn("style_cue_profile_match_score", report["style_transfer_effectiveness"])
            self.assertIn("theme_prompt_image_alignment_score", report["style_transfer_effectiveness"])
            self.assertIn("target_structure_retention_score", report["app_identity_coherence"])
            self.assertIn("over_recomposition_penalty", report["app_identity_coherence"])
            self.assertIn("generation_qwen_qc_prior", report["diagnostics"])
            self.assertEqual(
                report["strict_delta_diagnostics"]["strict_delta_transfer_score"],
                report["style_delta_transfer_score"],
            )
            self.assertGreater(report["style_transfer_effectiveness"]["score"], 70.0)
            self.assertLess(
                report["strict_delta_diagnostics"]["strict_delta_transfer_score"],
                report["style_transfer_effectiveness"]["score"],
            )
            self.assertNotIn("strict_delta_diagnostic_weak", report["failed_reasons"])
            if report["strict_delta_diagnostics"]["strict_delta_transfer_score"] < 35.0:
                self.assertIn(
                    "strict_delta_diagnostic_weak",
                    report["warnings"]["strict_delta_warnings"],
                )

    def test_itte_decision_separates_package_coherence_from_style_transfer(self):
        from evaluation.services.tpqs_service import _itte_decision

        decision = _itte_decision(
            style_transfer_score=45.0,
            package_coherence_score=88.0,
            app_identity_score=90.0,
            visual_quality_score=95.0,
        )

        self.assertEqual(decision, "package_coherent_but_theme_transfer_weak")

    def test_itte_decision_allows_low_strict_delta_when_transfer_is_effective(self):
        from evaluation.services.tpqs_service import _itte_decision

        decision = _itte_decision(
            style_transfer_score=82.0,
            package_coherence_score=84.0,
            app_identity_score=88.0,
            visual_quality_score=92.0,
            strict_delta_score=0.0,
        )

        self.assertEqual(decision, "style_transfer_success")

    def test_itte_decision_flags_identity_risk_below_success_threshold(self):
        from evaluation.services.tpqs_service import _itte_decision

        decision = _itte_decision(
            style_transfer_score=82.0,
            package_coherence_score=84.0,
            app_identity_score=66.0,
            visual_quality_score=92.0,
            strict_delta_score=0.0,
        )

        self.assertEqual(decision, "style_transfer_success_with_identity_risk")

    def test_style_transfer_effectiveness_v11_uses_style_profile_and_prompt_alignment(self):
        from evaluation.services.tpqs_service import _style_transfer_effectiveness

        result = _style_transfer_effectiveness(
            theme_style_image_transfer_score=72.0,
            style_cue_profile_match_score=84.0,
            attribute_transfer_score=60.0,
            theme_style_text_fit={
                "score": 80.0,
                "theme_style_text_fit_reliable": True,
            },
        )

        self.assertEqual(result["itte_style_transfer_version"], "v1.1")
        self.assertEqual(result["theme_prompt_image_alignment_score"], 80.0)
        self.assertEqual(result["theme_style_text_fit_score"], 80.0)
        self.assertIn("style_cue_profile_match_score", result["included_components"])
        self.assertIn("theme_prompt_image_alignment_score", result["included_components"])
        self.assertGreater(result["score"], 74.0)
        self.assertLess(result["score"], 82.0)

    def test_identity_coherence_penalizes_over_recomposition_even_with_high_qwen_prior(self):
        from evaluation.services.tpqs_service import _app_identity_coherence_score

        identity = {
            "score": 25.0,
            "per_app": [
                {
                    "app": "kept",
                    "identity_match_correct": True,
                    "generated_to_own_target_similarity": 0.88,
                },
                {
                    "app": "redrawn",
                    "identity_match_correct": False,
                    "generated_to_own_target_similarity": 0.55,
                },
            ],
        }
        dino_gt = np.asarray(
            [
                [0.05, 0.18],
                [0.18, 0.19],
            ],
            dtype=np.float32,
        )

        result = _app_identity_coherence_score(
            identity=identity,
            qwen_qc={"target_identity_score": 95},
            dino_gt=dino_gt,
            app_names=["kept", "redrawn"],
            style_transfer_score=82.0,
        )

        self.assertEqual(result["identity_recognition_prior_score"], 95.0)
        self.assertLess(result["score"], 75.0)
        self.assertGreater(result["over_recomposition_penalty"], 0.0)
        self.assertIn("redrawn", result["target_structure_warning_apps"])

    def test_target_structure_retention_uses_relative_distance_scale_for_dinov3(self):
        from evaluation.services.tpqs_service import _target_structure_retention_score

        dino_scale_distances = np.asarray(
            [
                [0.34, 0.82, 0.86],
                [0.70, 0.65, 0.80],
                [0.66, 0.72, 0.47],
            ],
            dtype=np.float32,
        )

        result = _target_structure_retention_score(
            dino_scale_distances,
            ["strong", "weak", "moderate"],
        )

        self.assertGreater(result["score"], 45.0)
        self.assertGreater(result["per_app"][0]["target_structure_retention_score"], 90.0)
        self.assertGreater(result["per_app"][2]["target_structure_retention_score"], 60.0)
        self.assertIn("weak", result["warning_apps"])
        self.assertNotIn("strong", result["warning_apps"])

    def test_tpqs_feedback_retry_prompt_explains_low_delta_scores(self):
        report = {
            "theme_id": "theme_001",
            "color_delta_score": 12,
            "edge_delta_score": 18,
            "composition_delta_score": 22,
            "identity_match_correct": False,
        }

        prompt = build_tpqs_feedback_retry_prompt(report)

        self.assertIn("theme_001", prompt)
        self.assertIn("color", prompt.lower())
        self.assertIn("stroke", prompt.lower())
        self.assertIn("composition", prompt.lower())
        self.assertIn("identity", prompt.lower())
        self.assertIn("do not call Wan automatically", prompt)

    def test_missing_target_original_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)
            (root / "data/targets/qq/qq.png").unlink()

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_eval_inputs("theme_001", "package_001_theme_001", root_dir=root)

            self.assertIn("Missing target original for generated apps: qq", str(ctx.exception))

    def test_qwen_qc_problematic_app_objects_are_normalized_to_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir)
            (package_dir / "package_qc_report.json").write_text(
                json.dumps(
                    {
                        "package_consistency_score": 85,
                        "style_consistency_score": 90,
                        "target_identity_score": 95,
                        "problematic_apps": [
                            {"app": "tieba", "issue": "background mismatch"},
                            {"app": "qq", "issue": "too white"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            scores = _load_qwen_qc_scores(package_dir)

            self.assertEqual(scores["problematic_apps"], ["tieba", "qq"])
            self.assertEqual(scores["problematic_app_details"][0]["issue"], "background mismatch")

    def test_dinov3_cache_dirs_are_project_local(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            cache = prepare_model_cache_dirs(root)

            self.assertEqual(cache.hf_home, root / "models" / "huggingface")
            self.assertEqual(cache.hf_hub_cache, root / "models" / "huggingface" / "hub")
            self.assertEqual(cache.modelscope_cache, root / "models" / "modelscope")
            self.assertEqual(cache.torch_home, root / "models" / "torch")
            self.assertEqual(os.environ["HF_HOME"], str(cache.hf_home))
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(cache.hf_hub_cache))
            self.assertEqual(os.environ["MODELSCOPE_CACHE"], str(cache.modelscope_cache))
            self.assertEqual(os.environ["TORCH_HOME"], str(cache.torch_home))

    def test_default_config_uses_official_dinov3_backend(self):
        config = TpqsConfig.from_env({})

        self.assertEqual(config.embedding_backend, "dinov3")
        self.assertEqual(config.model_source, "modelscope")
        self.assertTrue(config.is_official_tpqs)
        self.assertEqual(config.model_id, "facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.pooling, "cls")
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.style_feature_backend, "color_edge_composition")
        self.assertFalse(config.use_openclip)
        self.assertEqual(config.openclip_model, "ViT-B-32")
        self.assertEqual(config.openclip_pretrained, "laion2b_s34b_b79k")

    def test_config_allows_huggingface_source_override(self):
        config = TpqsConfig.from_env({"TPQS_MODEL_SOURCE": "huggingface"})

        self.assertEqual(config.model_source, "huggingface")

    def test_readme_documents_itte_framework(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("ITTE = Icon Theme Transfer Evaluation", readme)
        self.assertIn("Package Coherence Score", readme)
        self.assertIn("Strict Delta Diagnostics", readme)
        self.assertIn("整包一致不等于风格迁移成功", readme)
        self.assertIn("vgg_gram_attribute", readme)

    def test_package_consistency_depends_only_on_generated_package_internal_distances(self):
        generated_pairwise = np.asarray([0.22, 0.25, 0.27, 0.24], dtype=np.float32)
        compact_reference_pairwise = np.asarray([0.05, 0.06, 0.07], dtype=np.float32)
        loose_reference_pairwise = np.asarray([0.85, 0.9, 0.95], dtype=np.float32)
        compact_target_pairwise = np.asarray([0.1, 0.11, 0.12], dtype=np.float32)
        loose_target_pairwise = np.asarray([0.7, 0.75, 0.8], dtype=np.float32)

        score_with_compact_refs = _package_consistency_score(
            generated_pairwise,
            compact_reference_pairwise,
            loose_target_pairwise,
        )
        score_with_loose_refs = _package_consistency_score(
            generated_pairwise,
            loose_reference_pairwise,
            compact_target_pairwise,
        )

        self.assertEqual(score_with_compact_refs["score"], score_with_loose_refs["score"])
        self.assertEqual(
            score_with_compact_refs["generated_pairwise_mean_distance"],
            score_with_loose_refs["generated_pairwise_mean_distance"],
        )
        self.assertIn("is_package_consistent", score_with_compact_refs)

    def test_style_delta_transfer_rewards_learning_reference_raw_to_styled_change(self):
        reference_raw = np.asarray([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32)
        reference_styled = np.asarray([[0.0, 1.0], [0.05, 0.95]], dtype=np.float32)
        targets = np.asarray([[0.8, 0.2], [0.75, 0.25]], dtype=np.float32)
        generated = np.asarray([[0.0, 1.2], [0.0, 1.15]], dtype=np.float32)

        score = _style_delta_transfer_score(reference_raw, reference_styled, targets, generated, ["a", "b"])

        self.assertGreater(score["score"], 70.0)
        self.assertLess(score["D_G_delta"], score["D_no_change_delta"])
        self.assertTrue(score["is_style_delta_transfer_effective"])
        self.assertEqual(
            set(score["group_scores"]),
            {"color", "edge", "composition", "complexity"},
        )
        self.assertEqual([row["app"] for row in score["per_app"]], ["a", "b"])

    def test_theme_style_text_fit_uses_style_text_not_app_function_words(self):
        from evaluation.services.style_text_clip_service import (
            FakeOpenClipBackend,
            build_style_eval_text_short,
            build_style_eval_text,
            compute_theme_style_text_fit,
        )

        theme_design = {
            "theme_visual_language": "soft sticker icon pack",
            "color_rule": "warm pastel colors",
            "color_transform_rule": "convert saturated brand colors to warm pastel colors",
            "background_style_rule": "cream rounded square background",
            "background_transform_rule": "use a cream rounded square background",
            "stroke_rule": "rounded thick outline",
            "stroke_transform_rule": "use hand drawn black outline",
            "composition_rule": "centered compact subject",
        }
        qwen_instruction_text = """
[Theme Style Analysis]
Healing hand-drawn cartoon app icon pack.
[Color Transform Rule]
Use low saturation pastel colors.
[Background Transform Rule]
Use a cream rounded square background.
[Global Wan Constraints]
Output must look like a missing member of theme_001.
"""
        backend = FakeOpenClipBackend(
            image_scores={
                "theme.png": 0.82,
                "target.png": 0.20,
                "generated.png": 0.75,
            }
        )

        result = compute_theme_style_text_fit(
            theme_ref_paths=[Path("theme.png")],
            generated_paths=[Path("generated.png")],
            target_paths=[Path("target.png")],
            theme_design_analysis=theme_design,
            qwen_instruction_text=qwen_instruction_text,
            backend=backend,
        )

        self.assertTrue(result["openclip_enabled"])
        self.assertIn("soft sticker icon pack", result["style_eval_text"])
        self.assertIn("cream rounded square background", result["style_eval_text"])
        self.assertIn("warm pastel", result["style_eval_text"])
        self.assertNotIn("ticket", result["style_eval_text"].lower())
        self.assertNotIn("social", result["style_eval_text"].lower())
        self.assertEqual(build_style_eval_text({"style_eval_text": "minimal warm icon"}), "minimal warm icon")
        self.assertGreater(result["score"], 70.0)
        self.assertLess(result["score"], 100.0)
        short_text = build_style_eval_text_short(
            {
                "theme_board": {
                    "palette": "warm pastel colors",
                    "line_style": "rounded black sketch outline",
                    "material": "matte paper texture",
                    "background": "cream rounded square background",
                    "composition": "centered simple subject",
                },
                "forbidden_style_drift": ["do not use glossy 3d effects"],
            },
            "Forbidden: do not use ticket or social app semantics.",
        )
        self.assertIn("warm pastel", short_text)
        self.assertIn("cream rounded square", short_text)
        self.assertNotIn("forbidden", short_text.lower())
        self.assertNotIn("do not", short_text.lower())
        self.assertNotIn("ticket", short_text.lower())

    def test_openclip_text_fit_uses_margin_to_avoid_tiny_denominator_100_score(self):
        from evaluation.services.style_text_clip_service import (
            FakeOpenClipBackend,
            compute_theme_style_text_fit,
        )

        backend = FakeOpenClipBackend(
            image_scores={
                "theme.png": 0.202,
                "target.png": 0.186,
                "generated.png": 0.213,
            }
        )

        result = compute_theme_style_text_fit(
            theme_ref_paths=[Path("theme.png")],
            generated_paths=[Path("generated.png")],
            target_paths=[Path("target.png")],
            theme_design_analysis={"style_eval_text": "healing hand drawn pastel icon pack"},
            backend=backend,
        )

        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 40.0)
        self.assertEqual(result["scoring_method"], "relative_lift_with_margin")
        self.assertFalse(result["text_anchor_reliable"])

    def test_unreliable_attribute_and_low_confidence_openclip_do_not_enter_style_transfer_score(self):
        from evaluation.services.tpqs_service import _style_transfer_effectiveness

        result = _style_transfer_effectiveness(
            theme_style_image_transfer_score=81.0,
            attribute_transfer_score=None,
            theme_style_text_fit={"score": 25.0, "theme_style_text_fit_reliable": False},
        )

        self.assertEqual(result["score"], 81.0)
        self.assertEqual(result["included_components"], ["theme_style_image_transfer_score"])
        self.assertFalse(result["theme_style_text_fit_reliable"])

    def test_style_features_are_normalized_and_cached_without_model_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            make_image(image_path)

            config = TpqsConfig.from_env({})
            features = extract_style_features([image_path], config, root_dir=root)
            grouped = extract_style_feature_groups([image_path], config, root_dir=root)
            second = extract_style_features([image_path], config, root_dir=root)

            vector = features[str(image_path)]
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
            self.assertEqual(
                set(grouped[str(image_path)]),
                {
                    "color",
                    "background",
                    "stroke",
                    "texture_material",
                    "composition",
                    "complexity",
                    "edge",
                },
            )
            for group_vector in grouped[str(image_path)].values():
                self.assertGreater(len(group_vector), 0)
            self.assertTrue(np.allclose(vector, second[str(image_path)]))
            cache_files = list((root / "data/evaluations/_cache/style_features").glob("*.npy"))
            self.assertEqual(len(cache_files), 1)


if __name__ == "__main__":
    unittest.main()
