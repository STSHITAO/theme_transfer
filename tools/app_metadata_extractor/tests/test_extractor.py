import json
import tempfile
import unittest
from pathlib import Path

from tools.app_metadata_extractor.extract import (
    ExtractionError,
    extract_directory,
    load_crawler_directory,
    parse_qwen_output,
)


def make_crawler_input(root: Path) -> Path:
    source = root / "应用描述"
    for name, description in {
        "甲应用": "提供地图查询、路线规划和实时导航服务。",
        "乙应用": "支持歌曲播放、歌单管理和播客收听。",
    }.items():
        app_dir = source / name
        app_dir.mkdir(parents=True)
        (app_dir / "应用描述.txt").write_text(description, encoding="utf-8")
        (app_dir / "主题图标.png").write_bytes(b"ignored")
    (source / "app_ids.json").write_text(
        json.dumps({"apps": {"甲应用": "alpha", "乙应用": "beta"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return source


class StandaloneExtractorTests(unittest.TestCase):
    def test_loads_crawler_directory_and_ignores_theme_icons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = make_crawler_input(Path(temp_dir))
            apps = load_crawler_directory(source)

        self.assertEqual(list(apps), ["alpha", "beta"])
        self.assertEqual(apps["alpha"]["display_name"], "甲应用")
        self.assertIn("实时导航", apps["alpha"]["store_description"])
        self.assertNotIn("主题图标", json.dumps(apps, ensure_ascii=False))

    def test_requires_explicit_stable_id_for_every_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = make_crawler_input(Path(temp_dir))
            (source / "app_ids.json").write_text(
                json.dumps({"apps": {"甲应用": "alpha"}}, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ExtractionError, "缺少稳定 App ID"):
                load_crawler_directory(source)

    def test_rejects_model_fields_outside_schema(self):
        source = [{"app": "alpha", "display_name": "甲应用", "store_description": "地图导航"}]
        response = json.dumps(
            {
                "apps": [
                    {
                        "app": "alpha",
                        "category": "地图 / 导航",
                        "core_function": "提供地图导航服务。",
                        "generation_policy": "绘制地图标记",
                    }
                ]
            },
            ensure_ascii=False,
        )
        with self.assertRaisesRegex(ExtractionError, "只能包含"):
            parse_qwen_output(response, source)

    def test_mock_writes_only_standalone_output_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = make_crawler_input(root)
            output = root / "standalone-output" / "apps.generated.json"
            prompt = root / "prompt.md"
            prompt.write_text("中文提取提示", encoding="utf-8")

            first = extract_directory(source, output, prompt_path=prompt, mock=True)
            second = extract_directory(source, output, prompt_path=prompt, mock=True)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(first["generated_app_count"], 2)
            self.assertEqual(second["generated_app_count"], 0)
            self.assertEqual(second["resumed_app_count"], 2)
            self.assertEqual(set(payload["apps"]), {"alpha", "beta"})
            self.assertFalse((root / "dataset").exists())
            self.assertFalse((root / "data").exists())

    def test_implementation_does_not_import_project_pipeline(self):
        implementation = Path(__file__).resolve().parents[1] / "extract.py"
        source = implementation.read_text(encoding="utf-8")
        self.assertNotIn("from backend", source)
        self.assertNotIn("from scripts", source)
        self.assertNotIn("dataset/apps.json", source)


if __name__ == "__main__":
    unittest.main()
