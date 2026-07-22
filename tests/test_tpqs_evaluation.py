import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from evaluation.services.embedding_service import TpqsConfig, prepare_model_cache_dirs
from evaluation.services.eval_path_service import resolve_eval_inputs
from evaluation.services.style_feature_service import extract_style_feature_groups, extract_style_features
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
        make_image(root / f"data/styles/theme_001/{app_name}/{app_name}.jpg", (35, 120, 210))
        make_image(root / f"data/styles/theme_001/{app_name}/{app_name}_style_ref.jpg", color)

    for app_name, color in {
        "bilibili": (205, 65, 85),
        "qq": (215, 75, 95),
        "wps": (225, 85, 105),
    }.items():
        make_image(root / f"data/packages/package_001_theme_001/final/{app_name}.png", color)
    for app_name, color in {
        "bilibili": (30, 120, 220),
        "qq": (40, 140, 240),
        "wps": (70, 160, 210),
    }.items():
        make_image(root / f"data/targets/{app_name}/{app_name}.png", color)

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


class TpqsEvaluationTests(unittest.TestCase):
    def test_resolve_eval_inputs_maps_generated_icons_to_matching_target_originals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)

            resolved = resolve_eval_inputs("theme_001", "package_001_theme_001", root_dir=root)

            self.assertEqual([item.app for item in resolved.generated_icons], ["bilibili", "qq", "wps"])
            self.assertEqual([item.app for item in resolved.theme_examples], ["alipay", "douyin", "wechat"])
            self.assertEqual(set(resolved.target_originals), {"bilibili", "qq", "wps"})
            self.assertEqual(resolved.theme_examples[0].reference_raw_path, resolved.theme_examples[0].original_path)
            self.assertEqual(resolved.missing_apps, [])
            self.assertEqual(resolved.skipped_apps, [])

    def test_stats_backend_generates_v12_artifacts_without_model_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)
            eval_dir = root / "data/evaluations/eval_001"
            eval_dir.mkdir(parents=True)
            for name in [
                "embedding_pca.png",
                "pairwise_distances.json",
                "tpqs_feedback_retry_prompt.md",
                "generation_feedback_prompt.md",
            ]:
                (eval_dir / name).write_text("old", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "TPQS_EMBEDDING_BACKEND": "stats",
                    "ITTE_USE_PERCEPTUAL": "false",
                    "ITTE_USE_VGG_GRAM": "false",
                },
                clear=False,
            ):
                result = run_tpqs(
                    "theme_001",
                    "package_001_theme_001",
                    "eval_001",
                    root_dir=root,
                )

            for name in [
                "itte_report.json",
                "tpqs_report.json",
                "metrics.csv",
                "style_pairwise_distances.json",
                "style_delta_distances.json",
                "dino_pairwise_distances.json",
                "inputs_manifest.json",
            ]:
                self.assertTrue((eval_dir / name).exists(), name)
            for name in [
                "embedding_pca.png",
                "pairwise_distances.json",
                "tpqs_feedback_retry_prompt.md",
                "generation_feedback_prompt.md",
            ]:
                self.assertFalse((eval_dir / name).exists(), name)

            report = result["report"]
            self.assertEqual(report["evaluation_framework"], "ITTE")
            self.assertEqual(report["itte_version"], "v1.2-image-only")
            self.assertEqual(report["evaluation_scope"], "observable_image_transfer_only")
            self.assertEqual(report["itte_score"], report["tpqs"])
            self.assertFalse(report["diagnostics"]["generation_qwen_qc_used_in_score"])
            self.assertFalse(report["diagnostics"]["openclip_used_in_score"])
            self.assertEqual(report["diagnostics"]["text_policy"], "out_of_scope")
            self.assertEqual(
                json.loads((eval_dir / "itte_report.json").read_text(encoding="utf-8")),
                json.loads((eval_dir / "tpqs_report.json").read_text(encoding="utf-8")),
            )

    def test_itte_v12_main_score_is_image_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_eval_fixture(root)
            with patch.dict(
                os.environ,
                {
                    "TPQS_EMBEDDING_BACKEND": "stats",
                    "ITTE_USE_PERCEPTUAL": "false",
                    "ITTE_USE_VGG_GRAM": "false",
                },
                clear=False,
            ):
                report = run_tpqs(
                    "theme_001",
                    "package_001_theme_001",
                    "eval_001",
                    root_dir=root,
                )["report"]

            self.assertNotIn("theme_prompt_image_alignment_score", report)
            self.assertNotIn("qwen_identity_score", report)
            self.assertEqual(report["style_fidelity_score"], report["style_fidelity"]["score"])
            self.assertEqual(report["identity_preservation_score"], report["identity_preservation"]["score"])
            for key in [
                "itte_score",
                "style_fidelity_score",
                "identity_preservation_score",
                "package_coherence_score",
                "visual_quality_score",
            ]:
                self.assertGreaterEqual(report[key], 0)
                self.assertLessEqual(report[key], 100)

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

            self.assertEqual(cache.hf_home, root / "models/huggingface")
            self.assertEqual(cache.hf_hub_cache, root / "models/huggingface/hub")
            self.assertEqual(cache.modelscope_cache, root / "models/modelscope")
            self.assertEqual(cache.torch_home, root / "models/torch")
            self.assertEqual(os.environ["HF_HOME"], str(cache.hf_home))
            self.assertEqual(os.environ["TORCH_HOME"], str(cache.torch_home))

    def test_default_config_uses_official_v12_backends(self):
        config = TpqsConfig.from_env({})

        self.assertEqual(config.embedding_backend, "dinov3")
        self.assertEqual(config.model_source, "modelscope")
        self.assertTrue(config.is_official_tpqs)
        self.assertEqual(config.model_id, "facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.assertEqual(config.style_feature_backend, "color_edge_composition")
        self.assertTrue(config.use_perceptual)
        self.assertTrue(config.use_vgg_gram)

    def test_config_allows_huggingface_source_override(self):
        config = TpqsConfig.from_env({"TPQS_MODEL_SOURCE": "huggingface"})
        self.assertEqual(config.model_source, "huggingface")

    def test_readme_documents_itte_v12_framework(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("ITTE = Icon Theme Transfer Evaluation", readme)
        self.assertIn("Package Coherence", readme)
        self.assertIn("整包内部统一但远离参考主题，不能得到高分", readme)
        self.assertIn("VGG16 多层 Gram", readme)
        self.assertIn("不再生成 `generation_feedback_prompt.md`", readme)

    def test_style_features_are_normalized_and_cached_without_model_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.png"
            make_image(image_path)

            config = TpqsConfig.from_env({})
            features = extract_style_features([image_path], config, root_dir=root)
            grouped = extract_style_feature_groups([image_path], config, root_dir=root)
            second = extract_style_features([image_path], config, root_dir=root)

            self.assertAlmostEqual(float(np.linalg.norm(features[str(image_path)])), 1.0, places=5)
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
            self.assertTrue(np.allclose(features[str(image_path)], second[str(image_path)]))
            self.assertEqual(
                len(list((root / "data/evaluations/_cache/style_features").glob("*.npy"))),
                1,
            )


if __name__ == "__main__":
    unittest.main()
