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
from evaluation.services.style_feature_service import extract_style_features
from evaluation.services.tpqs_service import _package_consistency_score, _style_delta_transfer_score
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
            root / f"data/styles/theme_001/{app_name}/{app_name}_background.png",
            color=(35, 120, 210),
        )
        make_image(
            root / f"data/styles/theme_001/{app_name}/{app_name}_foreground.png",
            color=(70, 170, 240),
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


class TpqsEvaluationTests(unittest.TestCase):
    def test_resolve_eval_inputs_maps_generated_icons_to_matching_target_originals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)

            resolved = resolve_eval_inputs("theme_001", "package_001_theme_001", root_dir=root)

            self.assertEqual([item.app for item in resolved.generated_icons], ["bilibili", "qq", "wps"])
            self.assertEqual(len(resolved.theme_refs), 3)
            self.assertEqual([item.app for item in resolved.theme_examples], ["alipay", "douyin", "wechat"])
            self.assertTrue(resolved.theme_examples[0].background_path.name.endswith("_background.png"))
            self.assertTrue(resolved.theme_examples[0].foreground_path.name.endswith("_foreground.png"))
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

            report = json.loads((eval_dir / "tpqs_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((eval_dir / "inputs_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(report["is_official_tpqs"])
            self.assertEqual(report["embedding_backend"], "stats")
            self.assertEqual(report["model_source"], "modelscope")
            self.assertEqual(report["style_feature_backend"], "color_edge_composition")
            self.assertFalse(report["semantic_fit_enabled"])
            self.assertEqual(len(manifest["theme_transfer_examples"]), 3)
            self.assertIn("reference_raw_path", manifest["theme_transfer_examples"][0])
            self.assertIn("style_delta_transfer", report["details"])
            self.assertIn(report["decision"], [
                "style_transfer_failed",
                "identity_collapse_risk",
                "local_retry_recommended",
                "closed_loop_pass",
                "needs_review_or_retry",
            ])
            for key in [
                "tpqs",
                "style_transfer_score",
                "style_delta_transfer_score",
                "package_internal_style_consistency_score",
                "reference_style_distribution_match_score",
                "theme_membership_score",
                "identity_separability_score",
                "visual_quality_score",
            ]:
                self.assertGreaterEqual(report[key], 0)
                self.assertLessEqual(report[key], 100)

            metrics_text = (eval_dir / "metrics.csv").read_text(encoding="utf-8")
            metrics_header = metrics_text.splitlines()[0].split(",")
            self.assertIn("style_delta_transfer_score", metrics_header)
            self.assertIn("d_to_reference_delta_centroid", metrics_header)
            self.assertIn("theme_membership_score", metrics_header)

            delta_distances = json.loads((eval_dir / "style_delta_distances.json").read_text(encoding="utf-8"))
            self.assertIn("reference_delta_pairwise", delta_distances)
            self.assertIn("generated_delta_to_reference_delta_centroid", delta_distances)

    def test_missing_target_original_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)
            (root / "data/targets/qq/qq.png").unlink()

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_eval_inputs("theme_001", "package_001_theme_001", root_dir=root)

            self.assertIn("Missing target original for generated apps: qq", str(ctx.exception))

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

    def test_config_allows_huggingface_source_override(self):
        config = TpqsConfig.from_env({"TPQS_MODEL_SOURCE": "huggingface"})

        self.assertEqual(config.model_source, "huggingface")

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
        self.assertEqual([row["app"] for row in score["per_app"]], ["a", "b"])

    def test_style_features_are_normalized_and_cached_without_model_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            make_image(image_path)

            config = TpqsConfig.from_env({})
            features = extract_style_features([image_path], config, root_dir=root)
            second = extract_style_features([image_path], config, root_dir=root)

            vector = features[str(image_path)]
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
            self.assertTrue(np.allclose(vector, second[str(image_path)]))
            cache_files = list((root / "data/evaluations/_cache/style_features").glob("*.npy"))
            self.assertEqual(len(cache_files), 1)


if __name__ == "__main__":
    unittest.main()
