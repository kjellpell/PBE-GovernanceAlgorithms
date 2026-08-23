import ast
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parents[1] / "Process_Change_Impact_Analysis.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {
    "compute_group_stats", "normal_cdf", "two_sided_p_value",
    "compute_diff_in_diff", "classify_effect",
}
CONSTANT_NAMES = {"Z_CRITICAL", "ALPHA", "MIN_OBS_PER_GROUP", "LAV_STYRKE_TERSKEL"}

EXTRACTED = []
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES:
        EXTRACTED.append(node)
    elif (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in CONSTANT_NAMES
    ):
        EXTRACTED.append(node)

NAMESPACE = {"np": np, "math": __import__("math")}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

compute_group_stats  = NAMESPACE["compute_group_stats"]
normal_cdf            = NAMESPACE["normal_cdf"]
two_sided_p_value      = NAMESPACE["two_sided_p_value"]
compute_diff_in_diff   = NAMESPACE["compute_diff_in_diff"]
classify_effect        = NAMESPACE["classify_effect"]
MIN_OBS_PER_GROUP       = NAMESPACE["MIN_OBS_PER_GROUP"]
LAV_STYRKE_TERSKEL      = NAMESPACE["LAV_STYRKE_TERSKEL"]


class GroupStatsTests(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(compute_group_stats([]), (0, None, None, None))

    def test_singleton_has_no_variance(self):
        n, mean, var, median = compute_group_stats([5.0])
        self.assertEqual(n, 1)
        self.assertEqual(mean, 5.0)
        self.assertIsNone(var)
        self.assertEqual(median, 5.0)

    def test_drops_none_and_nan(self):
        n, mean, _, _ = compute_group_stats([1.0, None, float("nan"), 3.0])
        self.assertEqual(n, 2)
        self.assertEqual(mean, 2.0)


class NormalCdfTests(unittest.TestCase):
    def test_cdf_at_zero_is_half(self):
        self.assertAlmostEqual(normal_cdf(0.0), 0.5)

    def test_p_value_at_zero_is_one(self):
        self.assertAlmostEqual(two_sided_p_value(0.0), 1.0)

    def test_p_value_at_196_is_about_005(self):
        self.assertAlmostEqual(two_sided_p_value(1.96), 0.05, places=2)


class DiffInDiffTests(unittest.TestCase):
    def test_fallback_before_after_flags_improvement_without_control(self):
        rng = np.random.default_rng(1)
        treatment_before = rng.normal(loc=100, scale=10, size=40).tolist()
        treatment_after  = rng.normal(loc=80, scale=10, size=40).tolist()

        result = compute_diff_in_diff(treatment_before, treatment_after)

        self.assertFalse(result["har_kontrollgruppe"])
        self.assertLess(result["estimate"], -10)
        self.assertLess(result["p_value"], 0.05)
        label = classify_effect(result["p_value"], result["estimate"], "Behandlingstid")
        self.assertEqual(label, "Forbedring")

    def test_diff_in_diff_nets_out_equal_confound(self):
        # Treatment looks like a big 20-unit improvement on its own, but the
        # control group improves by the SAME amount over the same window —
        # this is the whole reason DiD exists over naive before/after.
        rng = np.random.default_rng(2)
        treatment_before = rng.normal(loc=100, scale=10, size=40).tolist()
        treatment_after  = rng.normal(loc=80,  scale=10, size=40).tolist()
        control_before   = rng.normal(loc=50,  scale=10, size=40).tolist()
        control_after    = rng.normal(loc=30,  scale=10, size=40).tolist()

        naive_before_after = np.mean(treatment_after) - np.mean(treatment_before)
        self.assertLess(naive_before_after, -10)  # naive view: looks like a big win

        result = compute_diff_in_diff(treatment_before, treatment_after, control_before, control_after)

        self.assertTrue(result["har_kontrollgruppe"])
        self.assertAlmostEqual(result["estimate"], 0.0, delta=8)
        label = classify_effect(result["p_value"], result["estimate"], "Behandlingstid")
        self.assertIn(label, ("Ingen sikker effekt", "Ingen praktisk effekt"))

    def test_identical_distributions_are_not_significant(self):
        rng = np.random.default_rng(3)
        sample = lambda: rng.normal(loc=50, scale=10, size=50).tolist()
        result = compute_diff_in_diff(sample(), sample(), sample(), sample())

        self.assertGreater(result["p_value"], 0.05)
        label = classify_effect(result["p_value"], result["estimate"], "Behandlingstid")
        self.assertEqual(label, "Ingen sikker effekt")

    def test_control_configured_but_empty_stays_har_kontrollgruppe_true(self):
        result = compute_diff_in_diff([1, 2, 3], [4, 5, 6], [], [])

        self.assertTrue(result["har_kontrollgruppe"])
        self.assertIsNone(result["estimate"])
        self.assertFalse(result["tilstrekkelig_volum"])

    def test_no_control_configured_is_har_kontrollgruppe_false(self):
        result = compute_diff_in_diff([1, 2, 3], [4, 5, 6], None, None)

        self.assertFalse(result["har_kontrollgruppe"])
        self.assertIsNotNone(result["estimate"])

    def test_n_zero_or_one_does_not_raise(self):
        result_empty = compute_diff_in_diff([], [1, 2, 3])
        self.assertIsNone(result_empty["estimate"])

        result_singleton = compute_diff_in_diff([5], list(range(20)))
        self.assertIsNotNone(result_singleton["estimate"])
        self.assertIsNone(result_singleton["se"])  # can't estimate variance from n=1

    def test_low_power_flag_is_none_when_volume_insufficient(self):
        result = compute_diff_in_diff([1] * 3, [2] * 3)
        self.assertFalse(result["tilstrekkelig_volum"])
        self.assertIsNone(result["lav_styrke"])

    def test_low_power_flag_true_when_thin_but_above_floor(self):
        n = MIN_OBS_PER_GROUP + 1
        self.assertLess(n, LAV_STYRKE_TERSKEL)
        rng = np.random.default_rng(4)
        result = compute_diff_in_diff(
            rng.normal(100, 5, n).tolist(), rng.normal(90, 5, n).tolist()
        )
        self.assertTrue(result["tilstrekkelig_volum"])
        self.assertTrue(result["lav_styrke"])

    def test_low_power_flag_false_when_comfortably_above_threshold(self):
        n = LAV_STYRKE_TERSKEL + 10
        rng = np.random.default_rng(5)
        result = compute_diff_in_diff(
            rng.normal(100, 5, n).tolist(), rng.normal(90, 5, n).tolist()
        )
        self.assertTrue(result["tilstrekkelig_volum"])
        self.assertFalse(result["lav_styrke"])


class ClassifyEffectTests(unittest.TestCase):
    def test_not_significant(self):
        self.assertEqual(classify_effect(0.5, -10, "Behandlingstid"), "Ingen sikker effekt")

    def test_significant_but_below_practical_floor(self):
        self.assertEqual(
            classify_effect(0.01, -2, "Behandlingstid", min_effect=5),
            "Ingen praktisk effekt",
        )

    def test_significant_and_above_floor_behandlingstid_lower_is_better(self):
        self.assertEqual(
            classify_effect(0.01, -10, "Behandlingstid", min_effect=5),
            "Forbedring",
        )
        self.assertEqual(
            classify_effect(0.01, 10, "Behandlingstid", min_effect=5),
            "Forverring",
        )

    def test_significant_and_above_floor_fristprosent_higher_is_better(self):
        self.assertEqual(
            classify_effect(0.01, 0.03, "Fristprosent", min_effect=0.02),
            "Forbedring",
        )
        self.assertEqual(
            classify_effect(0.01, -0.03, "Fristprosent", min_effect=0.02),
            "Forverring",
        )

    def test_no_practical_floor_configured_still_classifies(self):
        self.assertEqual(
            classify_effect(0.01, -0.5, "Behandlingstid", min_effect=None),
            "Forbedring",
        )

    def test_none_inputs_are_unclassifiable(self):
        self.assertIsNone(classify_effect(None, -10, "Behandlingstid"))
        self.assertIsNone(classify_effect(0.01, None, "Behandlingstid"))

    def test_unknown_maaltall_raises(self):
        with self.assertRaises(ValueError):
            classify_effect(0.01, -10, "Produksjonsdifferanse")


if __name__ == "__main__":
    unittest.main()
