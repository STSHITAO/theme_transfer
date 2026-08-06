import unittest

from scripts.run_full_generation import evenly_spaced_targets


class FullGenerationScriptTests(unittest.TestCase):
    def test_evenly_spaced_targets_is_stable_and_covers_both_ends(self):
        targets = [f"app_{index:02d}" for index in range(10)]

        selected = evenly_spaced_targets(list(reversed(targets)), 4)

        self.assertEqual(selected, ["app_00", "app_03", "app_06", "app_09"])

    def test_evenly_spaced_targets_without_limit_returns_all_sorted(self):
        self.assertEqual(evenly_spaced_targets(["b", "a", "b"], None), ["a", "b"])
