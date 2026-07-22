import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluation.services.dino_dense_service import DenseImageFeature, dense_correspondence
from evaluation.services.image_view_service import load_image_view
from evaluation.services.itte_v12_service import _decision, _dense_identity_score, _weighted_available
from evaluation.services.quality_service import compute_visual_quality


class ItteV12Tests(unittest.TestCase):
    def test_dense_correspondence_rewards_matching_patch_identity_and_layout(self):
        patches = np.eye(4, dtype=np.float32)
        coordinates = np.asarray(
            [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]],
            dtype=np.float32,
        )
        matching = DenseImageFeature(patches, coordinates, np.ones(4, dtype=bool))
        shifted = DenseImageFeature(patches, coordinates[::-1], np.ones(4, dtype=bool))

        exact = dense_correspondence(matching, matching)
        displaced = dense_correspondence(matching, shifted)

        self.assertAlmostEqual(exact["score"], 1.0, places=5)
        self.assertGreater(exact["score"], displaced["score"])
        self.assertGreater(exact["spatial_consistency"], displaced["spatial_consistency"])

    def test_missing_style_component_is_renormalized_instead_of_scored_as_zero(self):
        score, available = _weighted_available(
            {
                "available": {"score": 80.0, "reliable": True, "weight": 0.40},
                "missing": {"score": None, "reliable": False, "weight": 0.60},
            }
        )

        self.assertEqual(score, 80.0)
        self.assertEqual(available, 0.40)

    def test_dense_identity_keeps_normalized_zero_to_one_hundred_score(self):
        coordinates = np.asarray(
            [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]],
            dtype=np.float32,
        )
        first = DenseImageFeature(np.eye(4, dtype=np.float32), coordinates, np.ones(4, dtype=bool))
        second_patches = np.roll(np.eye(4, dtype=np.float32), 2, axis=1)
        second = DenseImageFeature(second_patches, coordinates, np.ones(4, dtype=bool))
        paths = [Path(name) for name in ["raw_a", "raw_b", "style_a", "style_b", "target_a", "target_b", "gen_a", "gen_b"]]
        dense = {
            str(paths[0]): first,
            str(paths[1]): second,
            str(paths[2]): first,
            str(paths[3]): second,
            str(paths[4]): first,
            str(paths[5]): second,
            str(paths[6]): first,
            str(paths[7]): second,
        }

        result = _dense_identity_score(
            paths[:2],
            paths[2:4],
            paths[4:6],
            paths[6:8],
            ["a", "b"],
            dense,
        )

        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["per_app"][0]["score"], 100.0)
        self.assertLessEqual(result["per_app"][0]["raw_correspondence"], 1.0)

    def test_structure_view_removes_launcher_label_region(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "launcher.png"
            image = Image.new("RGB", (120, 150), (120, 120, 120))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((10, 2, 110, 102), radius=18, fill=(240, 210, 30))
            for x in range(30, 90, 12):
                draw.rectangle((x, 125, x + 5, 145), fill="black")
            image.save(path)

            structure = load_image_view(path, "structure", 128)
            bottom = np.asarray(structure, dtype=np.uint8)[104:]

            self.assertEqual(structure.size, (128, 128))
            self.assertLess(float((bottom.mean(axis=2) < 30).mean()), 0.05)

    def test_low_confidence_cannot_return_style_transfer_success(self):
        decision = _decision(
            total=92.0,
            style=94.0,
            identity={"score": 90.0},
            package={"score": 93.0},
            quality={"score": 96.0},
            confidence="low",
            hard_failures=[],
        )

        self.assertEqual(decision, "insufficient_reference_evidence")

    def test_quality_detects_subject_touching_most_of_border(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.png"
            generated = root / "generated.png"
            ref_image = Image.new("RGB", (128, 128), "white")
            draw = ImageDraw.Draw(ref_image)
            draw.ellipse((30, 30, 98, 98), fill="red")
            ref_image.save(reference)
            generated_image = Image.new("RGB", (128, 128), "black")
            generated_image.save(generated)

            result = compute_visual_quality([reference] * 5, [generated], ["bad"])

            self.assertTrue(result["hard_failures"])
            self.assertLess(result["per_app"][0]["score"], 70.0)


if __name__ == "__main__":
    unittest.main()
