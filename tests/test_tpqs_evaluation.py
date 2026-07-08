import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from evaluation.services.embedding_service import (
    TpqsConfig,
    prepare_model_cache_dirs,
)
from evaluation.services.eval_path_service import resolve_eval_inputs
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
            self.assertEqual(set(resolved.target_originals), {"bilibili", "qq", "wps"})
            self.assertEqual(resolved.missing_apps, [])
            self.assertEqual(resolved.skipped_apps, [])

    def test_stats_backend_generates_report_artifacts_without_api_or_model_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)

            with patch.dict(os.environ, {"TPQS_EMBEDDING_BACKEND": "stats"}, clear=False):
                result = run_tpqs(
                    "theme_001",
                    "package_001_theme_001",
                    "eval_001_package_001_theme_001",
                    root_dir=root,
                )

            eval_dir = Path(result["eval_dir"])
            self.assertTrue((eval_dir / "tpqs_report.json").exists())
            self.assertTrue((eval_dir / "metrics.csv").exists())
            self.assertTrue((eval_dir / "embedding_pca.png").exists())
            self.assertTrue((eval_dir / "pairwise_distances.json").exists())
            self.assertTrue((eval_dir / "inputs_manifest.json").exists())

            report = json.loads((eval_dir / "tpqs_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["is_official_tpqs"])
            self.assertEqual(report["embedding_backend"], "stats")
            self.assertIn(report["decision"], [
                "transfer_failed",
                "identity_collapse_risk",
                "local_retry_recommended",
                "closed_loop_pass",
                "needs_review_or_retry",
            ])
            for key in [
                "tpqs_total_score",
                "theme_transfer_score",
                "package_consistency_score",
                "theme_membership_score",
                "identity_separability_score",
                "visual_statistics_score",
            ]:
                self.assertGreaterEqual(report[key], 0)
                self.assertLessEqual(report[key], 100)

            metrics_text = (eval_dir / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("app,theme_membership_score,identity_rank", metrics_text)

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
            self.assertEqual(cache.torch_home, root / "models" / "torch")
            self.assertEqual(os.environ["HF_HOME"], str(cache.hf_home))
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(cache.hf_hub_cache))
            self.assertEqual(os.environ["TORCH_HOME"], str(cache.torch_home))

    def test_default_config_uses_official_dinov3_backend(self):
        config = TpqsConfig.from_env({})

        self.assertEqual(config.embedding_backend, "dinov3")
        self.assertTrue(config.is_official_tpqs)
        self.assertEqual(config.model_id, "facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.pooling, "cls")
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
