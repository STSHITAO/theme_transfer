import tempfile
import unittest
from pathlib import Path

from scripts.build_experiment_manifest import build_manifest


class ExperimentManifestTests(unittest.TestCase):
    def test_manifest_hashes_experiment_artifacts_and_excludes_feature_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            included = root / "data/packages/package_a/final/app.png"
            included.parent.mkdir(parents=True)
            included.write_bytes(b"generated-image")
            report = root / "data/evaluations/eval_a/itte_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            cached = root / "data/evaluations/_cache/embeddings/app.npy"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"cache")

            manifest = build_manifest(root)

            self.assertEqual(
                [item["path"] for item in manifest["artifacts"]],
                ["data/packages/package_a/final/app.png", "data/evaluations/eval_a/itte_report.json"],
            )
            self.assertEqual(manifest["summary"]["data/evaluations"]["file_count"], 1)
            self.assertEqual(manifest["summary"]["data/packages"]["file_count"], 1)
