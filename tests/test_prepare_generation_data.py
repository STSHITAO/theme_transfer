import base64
import json
import tempfile
import unittest
from pathlib import Path

from backend.services.path_service import resolve_target_inputs, resolve_theme_examples
from scripts.prepare_generation_data import PreparationError, prepare_generation_data


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"test-jpeg"
WEBP_BYTES = b"RIFF\x04\x00\x00\x00WEBP"


def write_image(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def app_metadata(app_id: str) -> dict:
    return {
        "app": app_id,
        "display_name": app_id.title(),
        "category": "test category",
        "store_description": f"Store description for {app_id}",
        "core_function": f"Core function for {app_id}",
    }


def write_apps(root: Path, app_ids: list[str]) -> None:
    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    payload = {
        "originals_path": "dataset/originals",
        "apps": {app_id: app_metadata(app_id) for app_id in app_ids},
    }
    (dataset / "apps.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def file_snapshot(directory: Path) -> dict[str, bytes]:
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)).replace("\\", "/"): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class PrepareGenerationDataTests(unittest.TestCase):
    def test_converts_targets_and_only_matched_theme_examples_without_mutating_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_apps(root, ["alpha", "beta"])
            write_image(root / "dataset/originals/alpha.png", PNG_BYTES)
            write_image(root / "dataset/originals/beta.jpg", JPEG_BYTES)
            write_image(root / "dataset/themes/theme_001/alpha.webp", WEBP_BYTES)
            write_image(root / "dataset/themes/theme_001/orphan.png", PNG_BYTES)
            source_before = file_snapshot(root / "dataset")

            summary, warnings = prepare_generation_data(root)

            self.assertEqual(file_snapshot(root / "dataset"), source_before)
            self.assertEqual(summary["targets_prepared"], 2)
            self.assertEqual(summary["theme_examples_prepared"], 1)
            self.assertEqual(summary["themes"]["theme_001"]["matched_examples"], 1)
            self.assertTrue(any("theme_001/orphan.png" in item for item in warnings))

            self.assertEqual((root / "data/targets/alpha/alpha.png").read_bytes(), PNG_BYTES)
            self.assertEqual((root / "data/targets/beta/beta.jpg").read_bytes(), JPEG_BYTES)
            target_profile = json.loads(
                (root / "data/targets/alpha/target.json").read_text(encoding="utf-8")
            )
            self.assertEqual(target_profile, app_metadata("alpha"))

            self.assertEqual(
                (root / "data/styles/theme_001/alpha/alpha.png").read_bytes(), PNG_BYTES
            )
            self.assertEqual(
                (root / "data/styles/theme_001/alpha/alpha_style_ref.webp").read_bytes(),
                WEBP_BYTES,
            )
            self.assertFalse((root / "data/styles/theme_001/orphan").exists())
            theme_profile = json.loads(
                (root / "data/styles/theme_001/theme.json").read_text(encoding="utf-8")
            )
            self.assertEqual(list(theme_profile["examples"]), ["alpha"])
            self.assertEqual(theme_profile["examples"]["alpha"], app_metadata("alpha"))

            resolved_examples = resolve_theme_examples("theme_001", root_dir=root)
            self.assertEqual([item["app_name"] for item in resolved_examples], ["alpha"])
            self.assertTrue(resolved_examples[0]["original_path"].endswith("alpha.png"))
            self.assertTrue(resolved_examples[0]["style_ref_path"].endswith("alpha_style_ref.webp"))
            self.assertTrue(resolve_target_inputs("beta", root_dir=root)["target_image"].endswith("beta.jpg"))

            all_resolved_examples = resolve_theme_examples("theme_001", root_dir=root, max_examples=None)
            self.assertEqual([item["app_name"] for item in all_resolved_examples], ["alpha"])

            output_before = file_snapshot(root / "data")
            rerun_summary, rerun_warnings = prepare_generation_data(root)
            self.assertEqual(file_snapshot(root / "data"), output_before)
            self.assertEqual(rerun_summary["copied_files"], 0)
            self.assertEqual(rerun_summary["written_json"], 0)
            self.assertEqual(rerun_warnings, warnings)

    def test_theme_filter_prepares_all_targets_and_only_selected_theme(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_apps(root, ["alpha", "beta"])
            write_image(root / "dataset/originals/alpha.png", PNG_BYTES)
            write_image(root / "dataset/originals/beta.png", PNG_BYTES)
            write_image(root / "dataset/themes/theme_001/alpha.png", PNG_BYTES)
            write_image(root / "dataset/themes/theme_002/beta.png", PNG_BYTES)

            summary, _ = prepare_generation_data(root, theme_id="theme_002")

            self.assertEqual(summary["selected_themes"], ["theme_002"])
            self.assertTrue((root / "data/targets/alpha/alpha.png").exists())
            self.assertTrue((root / "data/targets/beta/beta.png").exists())
            self.assertFalse((root / "data/styles/theme_001").exists())
            self.assertTrue((root / "data/styles/theme_002/beta/beta_style_ref.png").exists())

    def test_clean_replaces_only_generation_input_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_apps(root, ["alpha"])
            write_image(root / "dataset/originals/alpha.png", PNG_BYTES)
            write_image(root / "dataset/themes/theme_001/alpha.png", PNG_BYTES)
            write_image(root / "data/styles/legacy/old.png", PNG_BYTES)
            write_image(root / "data/targets/legacy/old.png", PNG_BYTES)
            keep = root / "data/packages/keep/result.txt"
            keep.parent.mkdir(parents=True)
            keep.write_text("keep", encoding="utf-8")

            summary, _ = prepare_generation_data(root, clean=True)

            self.assertEqual(summary["cleaned"], ["data/targets", "data/styles"])
            self.assertFalse((root / "data/styles/legacy").exists())
            self.assertFalse((root / "data/targets/legacy").exists())
            self.assertEqual(keep.read_text(encoding="utf-8"), "keep")
            self.assertTrue((root / "data/styles/theme_001/alpha/alpha.png").exists())

    def test_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "dataset").mkdir()
            (root / "dataset/apps.json").write_text("{invalid", encoding="utf-8")

            with self.assertRaisesRegex(PreparationError, "Cannot parse apps.json"):
                prepare_generation_data(root)

    def test_rejects_duplicate_json_app_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "dataset").mkdir()
            metadata = json.dumps(app_metadata("alpha"), ensure_ascii=False)
            (root / "dataset/apps.json").write_text(
                '{"apps":{"alpha":' + metadata + ',"alpha":' + metadata + "}}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PreparationError, "Duplicate JSON key/app_id"):
                prepare_generation_data(root)

    def test_rejects_original_without_metadata_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_apps(root, ["alpha"])
            write_image(root / "dataset/originals/alpha.png", PNG_BYTES)
            write_image(root / "dataset/originals/beta.png", PNG_BYTES)
            write_image(root / "dataset/themes/theme_001/alpha.png", PNG_BYTES)

            with self.assertRaisesRegex(PreparationError, "lack apps.json metadata: beta"):
                prepare_generation_data(root)
            self.assertFalse((root / "data").exists())

    def test_rejects_duplicate_image_stems_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_apps(root, ["alpha"])
            write_image(root / "dataset/originals/alpha.png", PNG_BYTES)
            write_image(root / "dataset/originals/ALPHA.jpg", JPEG_BYTES)
            (root / "dataset/themes/theme_001").mkdir(parents=True)

            with self.assertRaisesRegex(PreparationError, "Duplicate app_id/image stem"):
                prepare_generation_data(root)

    def test_rejects_extension_content_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_apps(root, ["alpha"])
            write_image(root / "dataset/originals/alpha.jpg", PNG_BYTES)
            (root / "dataset/themes/theme_001").mkdir(parents=True)

            with self.assertRaisesRegex(PreparationError, "extension/content mismatch"):
                prepare_generation_data(root)


if __name__ == "__main__":
    unittest.main()
