import ast
import unittest
from pathlib import Path

from datetime import date, timedelta

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "CUSUM_Changepoint.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {
    "to_float_series", "run_cusum", "extract_changepoint_stats", "_periode_to_date", "_pelt_l2",
    "production_balance", "iso_week_key",
}
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

NAMESPACE = {"np": np, "pd": pd, "timedelta": timedelta}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

run_cusum                   = NAMESPACE["run_cusum"]
extract_changepoint_stats   = NAMESPACE["extract_changepoint_stats"]
_pelt_l2                    = NAMESPACE["_pelt_l2"]
production_balance          = NAMESPACE["production_balance"]
iso_week_key                = NAMESPACE["iso_week_key"]
_periode_to_date            = NAMESPACE["_periode_to_date"]
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


class ProductionBalanceTests(unittest.TestCase):
    def test_started_and_ended_counted_in_their_own_periods(self):
        # 6 faser start in 202403 and end in 202405. The old single query put
        # both counts in 202405 and netted them to zero.
        started = pd.DataFrame([{"indikator": "A", "period": 202403, "antall_startet": 6}])
        ended = pd.DataFrame([{"indikator": "A", "period": 202405, "antall_avsluttet": 6}])

        balance = production_balance(started, ended)

        self.assertEqual(balance["period"].tolist(), [202403, 202405])
        self.assertEqual(balance["verdi"].tolist(), [6.0, -6.0])

    def test_period_with_both_sides(self):
        started = pd.DataFrame([{"indikator": "A", "period": 202401, "antall_startet": 10}])
        ended = pd.DataFrame([{"indikator": "A", "period": 202401, "antall_avsluttet": 8}])

        balance = production_balance(started, ended)

        self.assertEqual(balance["verdi"].tolist(), [2.0])


class IsoWeekKeyTests(unittest.TestCase):
    def test_new_year_dates_use_the_iso_year(self):
        # YEAR(d) * 100 + WEEKOFYEAR(d) gave 202401, 202252 and 202153 here.
        self.assertEqual(iso_week_key(date(2024, 12, 30)), 202501)
        self.assertEqual(iso_week_key(date(2022, 1, 1)), 202152)
        self.assertEqual(iso_week_key(date(2021, 1, 1)), 202053)

    def test_ordinary_dates_unchanged(self):
        self.assertEqual(iso_week_key(date(2024, 6, 12)), 202424)

    def test_matches_python_isocalendar_for_every_day_2015_to_2030(self):
        d = date(2015, 1, 1)
        while d < date(2031, 1, 1):
            year, week, _ = d.isocalendar()
            self.assertEqual(iso_week_key(d), year * 100 + week, d)
            d += timedelta(days=1)

    def test_every_key_converts_back_to_a_date_in_that_week(self):
        for d in (date(2024, 12, 30), date(2022, 1, 1), date(2021, 1, 1)):
            sunday = _periode_to_date(iso_week_key(d), "Ukentlig")
            self.assertEqual(sunday.weekday(), 6)
            self.assertTrue(0 <= (sunday - d).days <= 6)


class PeriodeToDateTests(unittest.TestCase):
    def test_week_53_of_a_53_week_year_is_valid(self):
        self.assertEqual(_periode_to_date(202053, "Ukentlig"), date(2021, 1, 3))

    def test_week_53_of_a_52_week_year_raises_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "ISO-uke 202153"):
            _periode_to_date(202153, "Ukentlig")

    def test_month_end(self):
        self.assertEqual(_periode_to_date(202402, "Månedlig"), date(2024, 2, 29))


class PeltL2FallbackTests(unittest.TestCase):
    """_pelt_l2 is the only PELT path run_changepoint uses — no external
    library, no pip install / Fabric Environment needed."""

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
