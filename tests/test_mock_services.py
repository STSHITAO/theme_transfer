import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from PIL import Image

from backend.services.prompt_service import build_generation_prompt
from backend.services.qc_service import select_best_candidate
from backend.services.qwen_client import (
    _call_qwen,
    _image_data_url as qwen_image_data_url,
    analyze_theme_design,
    analyze_theme,
    analyze_theme_package,
    analyze_target_identity,
    build_identity_strategy,
    build_transfer_plan,
    score_candidates,
)
from backend.services.wan_client import _call_wan
from backend.services.storage_service import save_json, save_metadata
from backend.services.wan_client import generate_candidates


def make_image(path: Path, color=(20, 40, 80, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        Image.new("RGB", (16, 16), color[:3]).save(path)
    else:
        Image.new("RGBA", (16, 16), color).save(path)


class MockServiceTests(unittest.TestCase):
    def test_real_qwen_call_uses_plan_key_and_disables_thinking(self):
        response = {
            "status_code": 200,
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"text": "{\"ok\": true}"}],
                        }
                    }
                ]
            },
        }

        with patch.dict(
            os.environ,
            {
                "ALI_PLAN_BASE_URL": "https://example.test/api/v1",
                "ALI_PLAN_MODEL": "qwen3.7-plus",
                "ALI_PLAN_API_KEY": "plan-key",
            },
            clear=False,
        ):
            with patch("dashscope.MultiModalConversation.call", return_value=response) as call:
                text = _call_qwen([{"text": "hi"}])

        self.assertEqual(text, "{\"ok\": true}")
        kwargs = call.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "plan-key")
        self.assertEqual(kwargs["model"], "qwen3.7-plus")
        self.assertFalse(kwargs["enable_thinking"])
        self.assertEqual(kwargs["timeout"], 120)

    def test_real_qwen_call_retries_once_for_transient_network_error(self):
        response = {
            "status_code": 200,
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"text": "{\"ok\": true}"}],
                        }
                    }
                ]
            },
        }

        with patch.dict(
            os.environ,
            {
                "ALI_PLAN_BASE_URL": "https://example.test/api/v1",
                "ALI_PLAN_MODEL": "qwen3.7-plus",
                "ALI_PLAN_API_KEY": "plan-key",
            },
            clear=False,
        ):
            with patch(
                "dashscope.MultiModalConversation.call",
                side_effect=[requests.exceptions.SSLError("temporary"), response],
            ) as call:
                text = _call_qwen([{"text": "hi"}])

        self.assertEqual(text, "{\"ok\": true}")
        self.assertEqual(call.call_count, 2)

    def test_qwen_image_data_url_compresses_large_png_for_api_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "large.png"
            Image.new("RGBA", (1600, 1600), (200, 30, 60, 255)).save(source)

            data_url = qwen_image_data_url(source)

            self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
            self.assertLess(len(data_url), source.stat().st_size)

    def test_real_wan_call_uses_messages_with_plan_images(self):
        response = {
            "status_code": 200,
            "output": {"choices": []},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            style_ref = root / "style_ref.jpg"
            target = root / "target.png"
            make_image(style_ref)
            make_image(target)

            with patch.dict(
                os.environ,
                {
                    "ALI_IMAGE_BASE_URL": "https://example.test/api/v1",
                    "ALI_IMAGE_MODEL": "wan2.7-image-pro",
                    "ALI_IMAGE_API_KEY": "image-key",
                },
                clear=False,
            ):
                with patch("dashscope.aigc.image_generation.ImageGeneration.call", return_value=response) as call:
                    _call_wan("生成提示", [str(style_ref)], str(target), n=3, size="2K")

        kwargs = call.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "image-key")
        self.assertEqual(kwargs["model"], "wan2.7-image-pro")
        self.assertNotIn("prompt", kwargs)
        self.assertNotIn("images", kwargs)
        self.assertEqual(kwargs["n"], 3)
        self.assertFalse(kwargs["watermark"])
        message = kwargs["messages"][0]
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content[0]["text"], "生成提示")
        self.assertIn("image", message.content[1])

    def test_real_wan_call_retries_once_for_transient_network_error(self):
        response = {
            "status_code": 200,
            "output": {"choices": []},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            style_ref = root / "style_ref.jpg"
            target = root / "target.png"
            make_image(style_ref)
            make_image(target)

            with patch.dict(
                os.environ,
                {
                    "ALI_IMAGE_BASE_URL": "https://example.test/api/v1",
                    "ALI_IMAGE_MODEL": "wan2.7-image-pro",
                    "ALI_IMAGE_API_KEY": "image-key",
                },
                clear=False,
            ):
                with patch(
                    "dashscope.aigc.image_generation.ImageGeneration.call",
                    side_effect=[requests.exceptions.ProxyError("temporary"), response],
                ) as call:
                    result = _call_wan("生成提示", [str(style_ref)], str(target), n=3, size="2K")

        self.assertEqual(result, response)
        self.assertEqual(call.call_count, 2)

    def test_mock_qwen_analysis_returns_required_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = root / "prompts/qwen_theme_analysis.md"
            prompt.parent.mkdir(parents=True)
            prompt.write_text("分析主题", encoding="utf-8")
            layout = root / "layout.png"
            style_ref = root / "style_ref.jpg"
            target = root / "target.png"
            make_image(layout)
            make_image(style_ref)
            make_image(target)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = analyze_theme(
                    [
                        {
                            "app_name": "wechat",
                            "reference_layout_path": str(layout),
                            "style_ref_path": str(style_ref),
                        }
                    ],
                    str(target),
                    root_dir=root,
                )

            for key in [
                "theme_style_analysis",
                "common_background_transform",
                "common_foreground_transform",
                "color_palette",
                "line_style",
                "texture_material",
                "lighting_shadow",
                "icon_composition_rules",
                "target_preservation",
                "generation_prompt",
                "negative_prompt",
                "qc_focus",
                "used_reference_examples",
            ]:
                self.assertIn(key, result)
            self.assertEqual(result["used_reference_examples"], ["wechat"])

    def test_qwen_analysis_uses_separate_reference_layers_in_real_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_theme_analysis.md").write_text("分析", encoding="utf-8")
            paths = {
                "background_path": root / "background.png",
                "foreground_path": root / "foreground.png",
                "style_ref_path": root / "style_ref.jpg",
                "target_background": root / "target_background.png",
                "target_foreground": root / "target_foreground.png",
            }
            for path in paths.values():
                make_image(path)

            qwen_json = json.dumps(
                {
                    "theme_style_analysis": "style",
                    "common_background_transform": "bg",
                    "common_foreground_transform": "fg",
                    "color_palette": [],
                    "line_style": "line",
                    "texture_material": "texture",
                    "lighting_shadow": "light",
                    "icon_composition_rules": "rules",
                    "target_preservation": "preserve",
                    "generation_prompt": "prompt",
                    "negative_prompt": "negative",
                    "qc_focus": "qc",
                    "used_reference_examples": ["wechat"],
                }
            )

            with patch.dict(os.environ, {"MOCK_MODE": "false"}, clear=False):
                with patch("backend.services.qwen_client._call_qwen", return_value=qwen_json) as call:
                    analyze_theme(
                        [
                            {
                                "app_name": "wechat",
                                "background_path": str(paths["background_path"]),
                                "foreground_path": str(paths["foreground_path"]),
                                "style_ref_path": str(paths["style_ref_path"]),
                            }
                        ],
                        {
                            "target_background": str(paths["target_background"]),
                            "target_foreground": str(paths["target_foreground"]),
                        },
                        root_dir=root,
                    )

            content = call.call_args.args[0]
            text_blocks = [item["text"] for item in content if "text" in item]
            image_blocks = [item for item in content if "image" in item]
            self.assertEqual(len(image_blocks), 5)
            self.assertTrue(any("background" in text for text in text_blocks))
            self.assertTrue(any("foreground" in text for text in text_blocks))
            self.assertTrue(any("style_ref" in text for text in text_blocks))
            self.assertFalse(any("sheet" in text.lower() for text in text_blocks))

    def test_qwen_package_analysis_does_not_include_target_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_theme_analysis.md").write_text("分析", encoding="utf-8")
            paths = {
                "background_path": root / "background.png",
                "foreground_path": root / "foreground.png",
                "style_ref_path": root / "style_ref.jpg",
            }
            for path in paths.values():
                make_image(path)

            qwen_json = json.dumps(
                {
                    "theme_style_analysis": "style",
                    "common_background_transform": "bg",
                    "common_foreground_transform": "fg",
                    "color_palette": [],
                    "line_style": "line",
                    "texture_material": "texture",
                    "lighting_shadow": "light",
                    "icon_composition_rules": "rules",
                    "target_preservation": "preserve",
                    "generation_prompt": "prompt",
                    "negative_prompt": "negative",
                    "qc_focus": "qc",
                    "used_reference_examples": ["wechat"],
                }
            )

            with patch.dict(os.environ, {"MOCK_MODE": "false"}, clear=False):
                with patch("backend.services.qwen_client._call_qwen", return_value=qwen_json) as call:
                    analyze_theme_package(
                        [
                            {
                                "app_name": "wechat",
                                "background_path": str(paths["background_path"]),
                                "foreground_path": str(paths["foreground_path"]),
                                "style_ref_path": str(paths["style_ref_path"]),
                            }
                        ],
                        root_dir=root,
                    )

            content = call.call_args.args[0]
            text_blocks = [item["text"] for item in content if "text" in item]
            image_blocks = [item for item in content if "image" in item]
            self.assertEqual(len(image_blocks), 3)
            self.assertTrue(any("批量整包" in text for text in text_blocks))
            self.assertFalse(any("target_" in text for text in text_blocks))

    def test_mock_target_identity_analysis_returns_identity_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_target_identity.md").write_text("目标身份分析", encoding="utf-8")
            target = root / "wps.png"
            make_image(target)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = analyze_target_identity("wps", str(target), root_dir=root)

            self.assertEqual(result["app"], "wps")
            self.assertIn("identity_anchors", result)
            self.assertIn("must_preserve", result)
            self.assertIn("can_restyle", result)
            self.assertIn("must_not_replace_with", result)

    def test_mock_transfer_plan_uses_theme_rules_and_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_transfer_plan.md").write_text("迁移计划", encoding="utf-8")
            theme_rules = {"theme_style_analysis": "soft theme"}
            target_identity = {
                "app": "wps",
                "identity_anchors": ["W structure"],
                "must_preserve": ["W structure"],
                "can_restyle": ["texture"],
                "must_not_replace_with": ["generic plush ball"],
            }

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = build_transfer_plan(theme_rules, target_identity, root_dir=root)

            self.assertEqual(result["app"], "wps")
            self.assertIn("preserve", result)
            self.assertIn("must_preserve", result)
            self.assertIn("restyle", result)
            self.assertIn("forbid", result)

    def test_mock_theme_design_analysis_returns_theme_board(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_theme_design_analysis.md").write_text("主题设计分析", encoding="utf-8")
            paths = {
                "background_path": root / "background.png",
                "foreground_path": root / "foreground.png",
                "style_ref_path": root / "style_ref.jpg",
            }
            for path in paths.values():
                make_image(path)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = analyze_theme_design(
                    [
                        {
                            "app_name": "alipay",
                            "background_path": str(paths["background_path"]),
                            "foreground_path": str(paths["foreground_path"]),
                            "style_ref_path": str(paths["style_ref_path"]),
                        }
                    ],
                    {
                        "theme_id": "theme_001",
                        "examples": {
                            "alipay": {
                                "app": "alipay",
                                "display_name": "Alipay",
                                "category": "payment",
                                "core_function": "payment and transfer",
                            }
                        },
                    },
                    root_dir=root,
                )

            self.assertIn("theme_board", result)
            self.assertIn("reference_transformation_patterns", result)
            self.assertIn("identity_handling_policy", result)

    def test_mock_identity_strategy_uses_target_profile_and_theme_board(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_identity_strategy.md").write_text("身份表达策略", encoding="utf-8")
            target = root / "xiaohongshu.png"
            make_image(target)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = build_identity_strategy(
                    {"theme_board": {"line_style": "shared line"}},
                    {"theme_style_analysis": "theme rules"},
                    {
                        "app": "xiaohongshu",
                        "display_name": "Xiaohongshu",
                        "category": "content community",
                        "core_function": "publish and browse note content",
                    },
                    str(target),
                    root_dir=root,
                )

            self.assertEqual(result["app"], "xiaohongshu")
            self.assertIn(result["identity_constraint_level"], ["strict", "balanced", "flexible"])
            self.assertIn("strategy_type", result)
            self.assertIn("design_rationale", result)
            self.assertIn("generation_direction", result)

    def test_identity_strategy_preserves_brand_identity_cues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_identity_strategy.md").write_text("身份表达策略", encoding="utf-8")
            target = root / "bilibili.png"
            make_image(target)
            qwen_json = json.dumps(
                {
                    "app": "bilibili",
                    "strategy_type": "logo_simplify",
                    "identity_constraint_level": "strict",
                    "design_rationale": "model tried to over-recompose",
                    "must_preserve": ["tv outline"],
                    "can_recompose": ["bilibili wordmark", "remove internal bilibili text"],
                    "forbid": ["keep bilibili wordmark", "directly preserve original bilibili text"],
                    "generation_direction": "make a fluffy tv creature",
                }
            )
            target_profile = {
                "app": "bilibili",
                "display_name": "Bilibili",
                "category": "video community",
                "core_function": "watch and publish videos",
                "brand_identity_cues": [
                    "bilibili wordmark",
                    "pink rounded square background",
                    "small TV outline",
                ],
            }

            with patch.dict(os.environ, {"MOCK_MODE": "false"}, clear=False):
                with patch("backend.services.qwen_client._call_qwen", return_value=qwen_json):
                    result = build_identity_strategy(
                        {"theme_board": {"line_style": "shared line"}},
                        {"theme_style_analysis": "theme rules"},
                        target_profile,
                        str(target),
                        root_dir=root,
                    )

            self.assertIn("bilibili wordmark", result["must_preserve"])
            self.assertIn("pink rounded square background", result["must_preserve"])
            self.assertNotIn("bilibili wordmark", result["can_recompose"])
            self.assertFalse(any("bilibili" in item.lower() for item in result["can_recompose"]))
            self.assertFalse(any("bilibili wordmark" in item for item in result["forbid"]))
            self.assertFalse(any("bilibili" in item.lower() and "text" in item.lower() for item in result["forbid"]))

    def test_transfer_plan_preserves_brand_identity_cues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir(parents=True)
            (root / "prompts/qwen_transfer_plan.md").write_text("迁移计划", encoding="utf-8")
            target_identity = {"app": "bilibili", "must_preserve": ["tv outline"], "must_not_replace_with": []}
            target_profile = {
                "app": "bilibili",
                "brand_identity_cues": ["bilibili wordmark", "pink rounded square background"],
            }
            identity_strategy = {
                "app": "bilibili",
                "strategy_type": "logo_simplify",
                "identity_constraint_level": "strict",
                "must_preserve": ["tv outline"],
                "can_recompose": ["bilibili wordmark", "remove internal bilibili text"],
                "forbid": ["keep pink rounded square background", "directly preserve original bilibili text"],
                "generation_direction": "make a fluffy tv creature",
            }
            qwen_json = json.dumps(
                {
                    "app": "bilibili",
                    "strategy_type": "logo_simplify",
                    "identity_constraint_level": "strict",
                    "preserve": ["tv outline"],
                    "must_preserve": ["tv outline"],
                    "recompose_allowed": ["bilibili wordmark", "remove internal bilibili text"],
                    "restyle": ["shared style"],
                    "decorate": [],
                    "forbid": ["keep pink rounded square background", "directly preserve original bilibili text"],
                    "generation_brief": "make a fluffy tv creature",
                }
            )

            with patch.dict(os.environ, {"MOCK_MODE": "false"}, clear=False):
                with patch("backend.services.qwen_client._call_qwen", return_value=qwen_json):
                    result = build_transfer_plan(
                        {"theme_style_analysis": "theme rules"},
                        target_identity,
                        root_dir=root,
                        target_profile=target_profile,
                        identity_strategy=identity_strategy,
                    )

            self.assertIn("bilibili wordmark", result["must_preserve"])
            self.assertIn("pink rounded square background", result["must_preserve"])
            self.assertNotIn("bilibili wordmark", result["recompose_allowed"])
            self.assertFalse(any("bilibili" in item.lower() for item in result["recompose_allowed"]))
            self.assertFalse(any("pink rounded square background" in item for item in result["forbid"]))
            self.assertFalse(any("bilibili" in item.lower() and "text" in item.lower() for item in result["forbid"]))

    def test_prompt_service_saves_final_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "prompts/wan_generation.md"
            template.parent.mkdir(parents=True)
            template.write_text("Wan 模板", encoding="utf-8")
            analysis = {
                "theme_style_analysis": "统一柔和主题",
                "generation_prompt": "红色书本主体",
                "negative_prompt": "文字、水印",
                "target_preservation": "保留小红书身份",
            }

            prompt_path = build_generation_prompt(
                analysis,
                "theme_001",
                "case_001_theme_001_to_xiaohongshu",
                root_dir=root,
            )

            text = Path(prompt_path).read_text(encoding="utf-8")
            self.assertIn("Wan 模板", text)
            self.assertIn("theme_001", text)
            self.assertIn("统一柔和主题", text)
            self.assertIn("保留小红书身份", text)

    def test_mock_wan_generation_creates_three_candidates_and_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_layout = root / "target_layout.png"
            style_ref = root / "style_ref.jpg"
            stale_candidate = root / "data/outputs/case_001_theme_001_to_xiaohongshu/candidates/candidate_04.png"
            make_image(target_layout)
            make_image(style_ref)
            make_image(stale_candidate)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = generate_candidates(
                    "生成提示",
                    [str(style_ref)],
                    str(target_layout),
                    "case_001_theme_001_to_xiaohongshu",
                    root_dir=root,
                )

            self.assertEqual(len(result["candidate_paths"]), 3)
            for candidate in result["candidate_paths"]:
                self.assertTrue(Path(candidate).exists())
            self.assertFalse(stale_candidate.exists())
            self.assertTrue(Path(result["wan_response_path"]).exists())

    def test_qc_selects_highest_score_and_falls_back_to_first_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = []
            for index in range(2):
                path = root / f"candidate_{index + 1}.png"
                make_image(path)
                candidates.append(str(path))

            report = {
                "candidates": [
                    {"file": candidates[0], "overall_score": 70},
                    {"file": candidates[1], "overall_score": 92},
                ],
                "best_candidate": candidates[0],
                "warning": "",
            }
            result = select_best_candidate(
                report,
                candidates,
                "case_001_theme_001_to_xiaohongshu",
                root_dir=root,
            )
            self.assertEqual(Path(result["best_output_path"]).read_bytes(), Path(candidates[1]).read_bytes())
            self.assertEqual(result["qc_report"]["best_candidate"], candidates[1])

            fallback = select_best_candidate(
                {"candidates": [{"file": candidates[0], "overall_score": "bad"}], "warning": ""},
                candidates,
                "case_001_theme_001_to_xiaohongshu",
                root_dir=root,
            )
            self.assertIn("No valid", fallback["qc_report"]["warning"])

    def test_qc_maps_candidate_file_name_to_local_candidate_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "data/outputs/case_001_theme_001_to_xiaohongshu/candidates"
            candidates = []
            for index in range(2):
                path = output_dir / f"candidate_{index + 1:02d}.png"
                make_image(path, color=(index * 100, 20, 40, 255))
                candidates.append(str(path))

            result = select_best_candidate(
                {
                    "candidates": [
                        {"file": "candidate_01.png", "overall_score": 70},
                        {"file": "candidate_02.png", "overall_score": 95},
                    ],
                    "warning": "",
                },
                candidates,
                "case_001_theme_001_to_xiaohongshu",
                root_dir=root,
            )

            self.assertEqual(result["qc_report"]["best_candidate"], candidates[1])
            self.assertEqual(Path(result["best_output_path"]).read_bytes(), Path(candidates[1]).read_bytes())

    def test_storage_saves_json_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = save_json({"hello": "world"}, root / "data/cases/case/value.json")
            self.assertEqual(json.loads(Path(path).read_text(encoding="utf-8")), {"hello": "world"})

            metadata_path = save_metadata(
                "case_001_theme_001_to_xiaohongshu",
                {
                    "theme_id": "theme_001",
                    "target_app": "xiaohongshu",
                    "used_reference_examples": ["wechat"],
                    "input_files": {},
                    "intermediate_files": {},
                    "output_files": {},
                    "model_config": {},
                    "prompt_file": "prompt.txt",
                    "mock_mode": True,
                },
                root_dir=root,
            )
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["case_id"], "case_001_theme_001_to_xiaohongshu")
            self.assertIn("created_at", metadata)

    def test_mock_qwen_scores_choose_first_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt = root / "prompts/qwen_qc.md"
            prompt.parent.mkdir(parents=True)
            prompt.write_text("质检", encoding="utf-8")
            style_ref = root / "style_ref.jpg"
            target = root / "target.png"
            candidate = root / "candidate.png"
            for path in [style_ref, target, candidate]:
                make_image(path)

            with patch.dict(os.environ, {"MOCK_MODE": "true"}, clear=False):
                result = score_candidates([str(style_ref)], str(target), [str(candidate)], root_dir=root)

            self.assertEqual(result["best_candidate"], str(candidate))
            self.assertEqual(result["candidates"][0]["overall_score"], 85)
            self.assertIn("target_recognition_score", result["candidates"][0])
            self.assertIn("semantic_fit_score", result["candidates"][0])
            self.assertIn("identity_constraint_score", result["candidates"][0])
            self.assertIn("over_recompose_risk", result["candidates"][0])


if __name__ == "__main__":
    unittest.main()
