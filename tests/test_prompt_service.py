import tempfile
import unittest
from pathlib import Path

from backend.services.prompt_service import build_package_target_prompt


class PromptIsolationTests(unittest.TestCase):
    def test_active_prompt_templates_use_chinese_instructions(self):
        root = Path(__file__).resolve().parents[1]
        prompt_dir = root / "prompts"
        banned_english_instructions = [
            "You extract neutral",
            "Hard constraints",
            "Output JSON only",
            "The final image is the target",
            "Return exactly one",
        ]
        for path in sorted(prompt_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(prompt=path.name):
                self.assertRegex(text, r"[\u4e00-\u9fff]")
                for phrase in banned_english_instructions:
                    self.assertNotIn(phrase, text)

        for path in sorted(prompt_dir.glob("qwen_*.md")):
            with self.subTest(language_constraint=path.name):
                self.assertIn(
                    "所有自然语言字段值必须使用中文",
                    path.read_text(encoding="utf-8"),
                )

        wan_prompt = (prompt_dir / "wan_generation.md").read_text(encoding="utf-8")
        self.assertIn("第一张图标记为 `TARGET_IMAGE`", wan_prompt)
        self.assertIn("后续图片标记为 `STYLE_REFERENCE`", wan_prompt)

    def test_target_prompt_scrubs_reference_names_and_omits_reasoning(self):
        plan = {
            "identity_constraint_level": "strict",
            "structure_preservation_mode": "preserve_major_structure",
            "structure_policy_rationale": "Follow Alipay and Bilibili examples.",
            "must_preserve": ["Tmall cat silhouette", "天猫 glyph layout"],
            "identity_application": "Preserve Tmall identity, not Bilibili.",
            "color_application": "Use ice blue like Alipay.",
            "stroke_application": "Use rounded ice edges.",
            "composition_application": "Keep target layout.",
            "fidelity_constraints": ["translucent ice"],
            "negative_constraints": ["No Bilibili television frame"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prompt.txt"
            build_package_target_prompt(
                "Theme analysis mentions 支付宝 and 哔哩哔哩.",
                "tmall",
                output,
                transfer_plan=plan,
                forbidden_reference_terms=["alipay", "支付宝", "bilibili", "哔哩哔哩"],
            )
            prompt = output.read_text(encoding="utf-8")

        self.assertNotIn("structure_policy_rationale", prompt)
        self.assertNotIn("Alipay", prompt)
        self.assertNotIn("Bilibili", prompt)
        self.assertNotIn("支付宝", prompt)
        self.assertNotIn("哔哩哔哩", prompt)
        self.assertIn("tmall", prompt)
        self.assertIn("天猫", prompt)
        self.assertIn("绝对身份锁定——最高优先级", prompt)
        self.assertIn("主题一致性绝不能取代目标身份", prompt)

    def test_semantic_recompose_does_not_claim_strict_shape_preservation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prompt.txt"
            build_package_target_prompt(
                "Theme contract",
                "target_app",
                output,
                transfer_plan={"structure_preservation_mode": "semantic_recompose"},
            )
            prompt = output.read_text(encoding="utf-8")

        self.assertIn("才允许进行语义重构", prompt)
        self.assertNotIn("这是材质与风格转换", prompt)

    def test_target_profile_and_generation_brief_reach_final_prompt(self):
        profile = {
            "display_name": "穿越火线：枪战王者",
            "category": "游戏 / 射击竞技",
            "core_function": "第一人称射击和团队竞技",
            "store_description": "提供多人在线战术竞技体验",
        }
        plan = {
            "structure_preservation_mode": "semantic_recompose",
            "generation_brief": "Use the CF emblem as the only main subject.",
            "recompose_allowed": ["A restrained crosshair ring"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prompt.txt"
            build_package_target_prompt(
                "Theme contract",
                "crossfire_mobile",
                output,
                transfer_plan=plan,
                target_profile=profile,
            )
            prompt = output.read_text(encoding="utf-8")

        self.assertIn("穿越火线：枪战王者", prompt)
        self.assertIn("第一人称射击和团队竞技", prompt)
        self.assertIn("提供多人在线战术竞技体验", prompt)
        self.assertIn("Use the CF emblem as the only main subject.", prompt)
        self.assertIn("A restrained crosshair ring", prompt)
