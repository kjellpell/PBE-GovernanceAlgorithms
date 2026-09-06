import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "CUSUM_Changepoint.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {"to_float_series", "run_cusum", "extract_changepoint_stats", "_periode_to_date", "_pelt_l2"}
CONSTANT_NAMES = {
    "CUSUM_K", "CUSUM_H",
    "CUSUM_BASELINE_MONTHLY", "CUSUM_BASELINE_WEEKLY", "CUSUM_MIN_POST_BASELINE_OBS",
}

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

NAMESPACE = {"np": np, "pd": pd}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

run_cusum                   = NAMESPACE["run_cusum"]
extract_changepoint_stats   = NAMESPACE["extract_changepoint_stats"]
_pelt_l2                    = NAMESPACE["_pelt_l2"]
CUSUM_BASELINE_MONTHLY      = NAMESPACE["CUSUM_BASELINE_MONTHLY"]
CUSUM_MIN_POST_BASELINE_OBS = NAMESPACE["CUSUM_MIN_POST_BASELINE_OBS"]


class ChangepointWindowingTests(unittest.TestCase):
    def test_after_window_is_bounded_to_next_breakpoint(self):
        # Three regimes: 0,0,0 | 5,5,5 | 10,10,10 with breakpoints at index 3 and 6.
        values = [0, 0, 0, 5, 5, 5, 10, 10, 10]
        periods = list(range(202401, 202401 + len(values)))
        series = pd.Series(values, index=periods, dtype="float64")

        results = extract_changepoint_stats(series, breakpoints=[3, 6], granularitet="Månedlig")

        self.assertEqual(len(results), 2)
        first, second = results
        # Old unbounded slice would give ~6.67 (average of 5,5,5,10,10,10) here.
        self.assertAlmostEqual(first["gjennomsnitt_etter"], 5.0)
        self.assertAlmostEqual(second["gjennomsnitt_foer"], 5.0)
        self.assertAlmostEqual(second["gjennomsnitt_etter"], 10.0)

    def test_single_breakpoint_still_runs_to_series_end(self):
        values = [0, 0, 0, 5, 5, 5]
        periods = list(range(202401, 202401 + len(values)))
        series = pd.Series(values, index=periods, dtype="float64")

        results = extract_changepoint_stats(series, breakpoints=[3], granularitet="Månedlig")

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["gjennomsnitt_foer"], 0.0)
        self.assertAlmostEqual(results[0]["gjennomsnitt_etter"], 5.0)


class CusumBaselineTests(unittest.TestCase):
    def test_standardises_against_baseline_window_only_not_whole_series(self):
        baseline_obs = CUSUM_BASELINE_MONTHLY
        rng = np.random.default_rng(42)
        baseline = rng.normal(loc=0.0, scale=1.0, size=baseline_obs)
        drift = np.full(10, 3.0)
        values = np.concatenate([baseline, drift])
        series = pd.Series(values, index=range(len(values)), dtype="float64")

        result = run_cusum(series, k=0.5, h=100.0, baseline_obs=baseline_obs)

        mu = np.mean(baseline)
        sigma = np.std(baseline)
        standardised = (values - mu) / sigma
        expected_pos = np.zeros(len(standardised))
        for i in range(1, len(standardised)):
            expected_pos[i] = max(0, expected_pos[i - 1] + standardised[i] - 0.5)

        np.testing.assert_allclose(result["cusum_pos"].to_numpy(), expected_pos)

    def test_returns_none_when_insufficient_post_baseline_observations(self):
        baseline_obs = CUSUM_BASELINE_MONTHLY
        # One short of the minimum required: baseline_obs + CUSUM_MIN_POST_BASELINE_OBS - 1
        total = baseline_obs + CUSUM_MIN_POST_BASELINE_OBS - 1
        values = list(range(total))
        series = pd.Series(values, index=range(len(values)), dtype="float64")

        result = run_cusum(series, k=0.5, h=5.0, baseline_obs=baseline_obs)

        self.assertIsNone(result)

    def test_returns_none_for_zero_variance_baseline(self):
        baseline_obs = CUSUM_BASELINE_MONTHLY
        values = [1.0] * baseline_obs + list(range(10))
        series = pd.Series(values, index=range(len(values)), dtype="float64")

        result = run_cusum(series, k=0.5, h=5.0, baseline_obs=baseline_obs)

        self.assertIsNone(result)


class PeltL2FallbackTests(unittest.TestCase):
    """_pelt_l2 is the ruptures-free PELT path run_changepoint uses when
    `ruptures` isn't importable — no pip install / Fabric Environment needed."""

    def test_detects_a_mean_shift(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(10, 1, 30), rng.normal(20, 1, 30)])
        penalty = 2 * np.log(len(values)) * np.std(values) ** 2

        breakpoints = _pelt_l2(values, min_size=6, penalty=penalty)

        self.assertTrue(any(25 <= bp <= 35 for bp in breakpoints))

    def test_flat_series_rarely_flags_a_changepoint(self):
        false_positives = 0
        for seed in range(100):
            values = np.random.default_rng(seed).normal(10, 1, 60)
            penalty = 2 * np.log(len(values)) * np.std(values) ** 2
            if _pelt_l2(values, min_size=6, penalty=penalty):
                false_positives += 1

        self.assertLess(false_positives, 10)

    def test_too_short_returns_no_breakpoints(self):
        values = np.arange(5.0)

        self.assertEqual(_pelt_l2(values, min_size=6, penalty=1.0), [])


if __name__ == "__main__":
    unittest.main()
