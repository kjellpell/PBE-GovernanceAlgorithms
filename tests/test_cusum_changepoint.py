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
    "production_balance", "iso_week_key", "standardise_seasonal",
}
CONSTANT_NAMES = {
    "CUSUM_K", "CUSUM_H",
    "CUSUM_BASELINE_YEARS", "CUSUM_MIN_BASELINE_YEARS", "CUSUM_MIN_POST_BASELINE_OBS",
    "CUSUM_RECOVERY_PERIODS",
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
standardise_seasonal        = NAMESPACE["standardise_seasonal"]
CUSUM_MIN_POST_BASELINE_OBS = NAMESPACE["CUSUM_MIN_POST_BASELINE_OBS"]
CUSUM_RECOVERY_PERIODS      = NAMESPACE["CUSUM_RECOVERY_PERIODS"]


def monthly_series(values, first_year=2015):
    """values in calendar order from January of first_year, indexed YYYYMM."""
    periods = [(first_year + i // 12) * 100 + (i % 12 + 1) for i in range(len(values))]
    return pd.Series(values, index=periods, dtype="float64")


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


class SeasonalBaselineTests(unittest.TestCase):
    def test_baseline_never_includes_the_observations_own_year(self):
        # Intent kept from the old anchored-baseline test: the period being
        # tested must not be part of its own baseline. 3 flat-ish years, then
        # a year shifted by +3: the shifted year is scored against the 3 prior
        # years only, so it stays clearly positive.
        rng = np.random.default_rng(42)
        prior = rng.normal(0.0, 1.0, 36)
        shifted = rng.normal(3.0, 1.0, 12)
        series = monthly_series(np.concatenate([prior, shifted]))

        z = standardise_seasonal(series, "Månedlig", years=3)

        base = pd.DataFrame({"pos": np.tile(np.arange(12), 3), "v": prior})
        mu = base.groupby("pos")["v"].mean().to_numpy()
        resid = prior - np.tile(mu, 3)
        sigma = np.sqrt(np.sum(resid ** 2) / (36 - 12))
        expected = (shifted - mu) / sigma
        np.testing.assert_allclose(z.iloc[36:].to_numpy(), expected)
        self.assertGreater(z.iloc[36:].mean(), 2.0)

    def test_same_month_comparison_removes_seasonality(self):
        # Every July is 20 points lower than the rest of the year. Against the
        # same month in earlier years, July is not an outlier.
        shape = np.array([80, 80, 80, 80, 80, 80, 60, 80, 80, 80, 80, 80], dtype=float)
        rng = np.random.default_rng(3)
        values = np.concatenate([shape + rng.normal(0, 1, 12) for _ in range(5)])

        z = standardise_seasonal(monthly_series(values), "Månedlig")

        july = [p for p in z.dropna().index if p % 100 == 7]
        self.assertTrue(all(abs(z[p]) < 4 for p in july))

    def test_no_score_without_two_prior_years(self):
        series = monthly_series(np.arange(36, dtype=float) % 5)

        z = standardise_seasonal(series, "Månedlig")

        self.assertTrue(z.iloc[:24].isna().all())
        self.assertTrue(z.iloc[24:].notna().all())

    def test_baseline_is_only_the_last_three_years(self):
        # 2015-2017 at level 100, 2018-2020 at level 0 (with noise). 2021 at
        # level 0 must be normal — a 2015 anchor would call it a huge drop.
        rng = np.random.default_rng(5)
        values = np.concatenate([
            100 + rng.normal(0, 1, 36), rng.normal(0, 1, 36), rng.normal(0, 1, 12),
        ])

        z = standardise_seasonal(monthly_series(values), "Månedlig")

        self.assertLess(z.iloc[72:].abs().max(), 4.0)

    def test_weekly_positions_are_iso_weeks(self):
        # 3 years of 52 weeks then one year: week 10 is compared with week 10.
        periods = [y * 100 + w for y in (2021, 2022, 2023, 2024) for w in range(1, 53)]
        values = [float(w) for y in (2021, 2022, 2023, 2024) for w in range(1, 53)]
        values = [v + 0.1 * (i % 3) for i, v in enumerate(values)]
        series = pd.Series(values, index=periods)

        z = standardise_seasonal(series, "Ukentlig")

        self.assertLess(z.loc[202410:202452].abs().max(), 4.0)

    def test_zero_variance_baseline_gives_no_score(self):
        series = monthly_series([1.0] * 36 + [5.0] * 12)

        z = standardise_seasonal(series, "Månedlig")

        self.assertTrue(z.isna().all())


class CusumRunTests(unittest.TestCase):
    def test_returns_none_when_too_few_standardised_points(self):
        # 2 years + 3 months -> only 3 standardised points.
        series = monthly_series(np.random.default_rng(1).normal(0, 1, 27))

        self.assertIsNone(run_cusum(series, "Månedlig"))

    def test_output_starts_where_the_baseline_exists(self):
        series = monthly_series(np.random.default_rng(1).normal(0, 1, 48))

        result = run_cusum(series, "Månedlig")

        self.assertEqual(result.index[0], 201701)
        self.assertEqual(len(result), 24)

    def test_cusum_recursion_on_standardised_values(self):
        series = monthly_series(np.random.default_rng(9).normal(0, 1, 60))

        result = run_cusum(series, "Månedlig", k=0.5, h=100.0)

        z = standardise_seasonal(series, "Månedlig").dropna().to_numpy()
        expected = np.zeros(len(z))
        prev = 0.0
        for i, v in enumerate(z):
            prev = max(0.0, prev + v - 0.5)
            expected[i] = prev
        np.testing.assert_allclose(result["cusum_pos"].to_numpy(), expected)


class CusumStateTests(unittest.TestCase):
    """tilstand is the persistent state a board card reads; signal_direction
    is only the firing-period event. A shift that lasts for years becomes
    part of the rolling 3-year baseline (the "new normal") — these tests look
    at the first 1-2 years after a shift, which is what an early warning is for."""

    @staticmethod
    def shifted(seed, shift, start, total=96):
        rng = np.random.default_rng(seed)
        values = rng.normal(0, 1, total)
        values[start:] += shift
        return monthly_series(values)

    @staticmethod
    def window_after_first_signal(result, direction, since, months):
        after = result.loc[result.index >= since]
        fired = after.index[after["signal_direction"] == direction]
        if not len(fired):
            return None
        return after.loc[after.index >= fired[0]].iloc[:months]

    def test_state_holds_through_a_sustained_decline(self):
        # Handover scenario: sustained -2 sigma shift from 2020-01. The event
        # column says Synkende only in firing months (the old board card read
        # Stabil every other month); the state stays Synkende.
        held = self.window_after_first_signal(
            run_cusum(self.shifted(seed=0, shift=-2.0, start=60), "Månedlig"),
            "Synkende", 202001, months=18,
        )

        self.assertLess((held["signal_direction"] == "Synkende").mean(), 0.5)
        self.assertTrue((held["tilstand"] == "Synkende").all())

    def test_state_mostly_holds_through_a_small_sustained_decline(self):
        shares = []
        for seed in range(40):
            held = self.window_after_first_signal(
                run_cusum(self.shifted(seed=seed, shift=-1.0, start=60), "Månedlig"),
                "Synkende", 202001, months=12,
            )
            if held is not None:
                shares.append((held["tilstand"] == "Synkende").mean())
        self.assertGreater(np.mean(shares), 0.8)

    def test_state_returns_to_stabil_after_recovery(self):
        rng = np.random.default_rng(4)
        values = rng.normal(0, 1, 96)
        values[48:54] -= 3.0      # six bad months in 2019, then back to normal
        result = run_cusum(monthly_series(values), "Månedlig")

        self.assertIn("Synkende", result.loc[201901:201906, "tilstand"].tolist())
        recovered = result.loc[201907:, "tilstand"].tolist()
        self.assertIn("Stabil", recovered)
        self.assertEqual(recovered[-1], "Stabil")

    def test_opposite_signal_flips_the_state(self):
        rng = np.random.default_rng(8)
        values = rng.normal(0, 1, 72)
        values[48:54] -= 3.0
        values[54:60] += 3.0
        result = run_cusum(monthly_series(values), "Månedlig")

        self.assertEqual(result.loc[201906, "tilstand"], "Synkende")
        self.assertEqual(result.loc[201912, "tilstand"], "Stigende")

    def test_flat_series_is_mostly_stabil(self):
        share_not_stabil = np.mean([
            (run_cusum(monthly_series(np.random.default_rng(s).normal(0, 1, 120)), "Månedlig")["tilstand"] != "Stabil").mean()
            for s in range(30)
        ])
        self.assertLess(share_not_stabil, 0.1)


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
