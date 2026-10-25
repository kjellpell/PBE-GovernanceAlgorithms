import ast
import unittest
from pathlib import Path

from datetime import date, timedelta

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "PELT_Changepoint.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {
    "to_float_series", "extract_changepoint_stats", "_periode_to_date", "_pelt_l2",
    "production_balance", "iso_week_key",
    "_periode_to_start_date", "_pelt_penalty", "decompose_segments",
}
CONSTANT_NAMES = {
    "MIN_SEGMENT_OBS", "PELT_MIN_SIZE", "RECENT_CHANGEPOINT_DAYS",
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

NAMESPACE = {"np": np, "pd": pd, "timedelta": timedelta, "date": date}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

extract_changepoint_stats   = NAMESPACE["extract_changepoint_stats"]
_pelt_l2                    = NAMESPACE["_pelt_l2"]
production_balance          = NAMESPACE["production_balance"]
iso_week_key                = NAMESPACE["iso_week_key"]
_periode_to_date            = NAMESPACE["_periode_to_date"]
_pelt_penalty               = NAMESPACE["_pelt_penalty"]
decompose_segments          = NAMESPACE["decompose_segments"]
PELT_MIN_SIZE               = NAMESPACE["PELT_MIN_SIZE"]
RECENT_CHANGEPOINT_DAYS     = NAMESPACE["RECENT_CHANGEPOINT_DAYS"]


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


class ChangepointWindowDatesTests(unittest.TestCase):
    def test_window_dates_bound_the_before_and_after_segments(self):
        values = [0, 0, 0, 5, 5, 5, 10, 10, 10]
        periods = [202401 + i for i in range(9)]
        series = pd.Series(values, index=periods, dtype="float64")

        first, second = extract_changepoint_stats(series, breakpoints=[3, 6], granularitet="Månedlig")

        self.assertEqual(first["segment_start"], date(2024, 1, 1))
        self.assertEqual(first["endringspunkt_start"], date(2024, 4, 1))
        self.assertEqual(first["segment_end"], date(2024, 6, 30))
        self.assertEqual(second["segment_start"], date(2024, 4, 1))
        self.assertEqual(second["segment_end"], date(2024, 9, 30))
        # analyse_dato stays the END of the breakpoint period (unchanged key)
        self.assertEqual(first["analyse_dato"], date(2024, 4, 30))

    def test_weekly_window_dates_are_monday_and_sunday(self):
        periods = [202450, 202451, 202452, 202501, 202502, 202503]
        series = pd.Series([0, 0, 0, 5, 5, 5], index=periods, dtype="float64")

        (cp,) = extract_changepoint_stats(series, breakpoints=[3], granularitet="Ukentlig")

        self.assertEqual(cp["segment_start"], date(2024, 12, 9))
        self.assertEqual(cp["endringspunkt_start"], date(2024, 12, 30))
        self.assertEqual(cp["segment_end"], date(2025, 1, 19))

    def test_most_recent_possible_monthly_changepoint_is_inside_the_window(self):
        # A breakpoint needs PELT_MIN_SIZE months after it. With the last
        # complete month ending yesterday, its analyse_dato is ~5-6 months back.
        oldest_possible_days = (PELT_MIN_SIZE + 1) * 31
        self.assertGreaterEqual(RECENT_CHANGEPOINT_DAYS["Månedlig"], oldest_possible_days)
        self.assertEqual(RECENT_CHANGEPOINT_DAYS["Ukentlig"], 90)


class DecomposeSegmentsTests(unittest.TestCase):
    @staticmethod
    def rows(spec):
        return pd.DataFrame(
            [{"segment": seg, "periode": per, "verdi": v, "antall": n} for seg, per, v, n in spec]
        )

    def test_rate_plus_mix_sums_to_aggregate_shift(self):
        # Team A: 80% -> 60% and grows; team B: 90% -> 90% and shrinks.
        segs = self.rows([
            ("A", "foer", 0.8, 50), ("A", "etter", 0.6, 80),
            ("B", "foer", 0.9, 50), ("B", "etter", 0.9, 20),
        ])

        out = decompose_segments(segs, additive=False)

        agg_foer = (0.8 * 50 + 0.9 * 50) / 100
        agg_etter = (0.6 * 80 + 0.9 * 20) / 100
        self.assertAlmostEqual(out["bidrag_til_endring"].sum(), agg_etter - agg_foer)
        np.testing.assert_allclose(out["bidrag_rate"] + out["bidrag_miks"], out["bidrag_til_endring"])
        # B's rate didn't change: all of its contribution is mix.
        self.assertAlmostEqual(out.loc["B", "bidrag_rate"], 0.0)
        self.assertAlmostEqual(out.loc["B", "bidrag_miks"], (0.2 - 0.5) * 0.9)
        self.assertAlmostEqual(out.loc["A", "bidrag_rate"], 0.8 * (0.6 - 0.8))

    def test_matches_old_single_contribution(self):
        segs = self.rows([
            ("A", "foer", 10.0, 30), ("A", "etter", 14.0, 40),
            ("B", "foer", 20.0, 70), ("B", "etter", 18.0, 60),
        ])

        out = decompose_segments(segs, additive=False)

        old = (40 / 100) * 14.0 - (30 / 100) * 10.0
        self.assertAlmostEqual(out.loc["A", "bidrag_til_endring"], old)

    def test_segment_missing_on_one_side_is_skipped_for_rates(self):
        segs = self.rows([("A", "foer", 0.8, 50), ("A", "etter", 0.6, 50), ("C", "etter", 0.5, 20)])

        out = decompose_segments(segs, additive=False)

        self.assertEqual(out.index.tolist(), ["A"])

    def test_additive_metric_is_per_period_with_no_mix(self):
        # Produksjonsdifferanse: totals over 6 periods before, 3 after.
        segs = self.rows([
            ("A", "foer", 12.0, 100), ("A", "etter", 30.0, 60),
            ("B", "etter", -3.0, 15),
        ])

        out = decompose_segments(segs, additive=True, perioder_foer=6, perioder_etter=3)

        self.assertAlmostEqual(out.loc["A", "verdi_foer"], 2.0)
        self.assertAlmostEqual(out.loc["A", "verdi_etter"], 10.0)
        self.assertAlmostEqual(out.loc["A", "bidrag_rate"], 8.0)
        self.assertAlmostEqual(out.loc["B", "verdi_foer"], 0.0)
        self.assertTrue((out["bidrag_miks"] == 0).all())
        self.assertFalse(out.loc["B", "tilstrekkelig_volum"])

    def test_volume_flag(self):
        segs = self.rows([
            ("A", "foer", 0.8, 9), ("A", "etter", 0.6, 50),
            ("B", "foer", 0.8, 10), ("B", "etter", 0.6, 10),
        ])

        out = decompose_segments(segs, additive=False)

        self.assertFalse(out.loc["A", "tilstrekkelig_volum"])
        self.assertTrue(out.loc["B", "tilstrekkelig_volum"])


class PeltPenaltyTests(unittest.TestCase):
    def test_two_shift_series_is_fully_detected(self):
        # A large old shift (0 -> 6) followed by a moderate recent one
        # (6 -> 7.5). The old np.std(values) penalty counts the big shift as
        # noise and never finds the recent one; the robust noise estimate does.
        hits_robust = hits_old = 0
        for seed in range(50):
            values = np.random.default_rng(seed).normal(0, 1, 120) + np.repeat([0.0, 6.0, 7.5], 40)
            old_penalty = 2 * np.log(len(values)) * np.std(values) ** 2
            for penalty, is_robust in ((_pelt_penalty(values), True), (old_penalty, False)):
                bps = _pelt_l2(values, min_size=PELT_MIN_SIZE, penalty=penalty)
                found = any(abs(b - 40) <= 4 for b in bps) and any(abs(b - 80) <= 5 for b in bps)
                if found and is_robust:
                    hits_robust += 1
                elif found:
                    hits_old += 1
        self.assertGreaterEqual(hits_robust, 45)
        self.assertLessEqual(hits_old, 5)

    def test_robust_sigma_ignores_level_shifts(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, 90)
        shifted = noise + np.repeat([0, 10, 0], 30)

        ratio = _pelt_penalty(shifted) / _pelt_penalty(noise)

        self.assertLess(ratio, 1.5)
        self.assertGreater(np.std(shifted) ** 2 / np.std(noise) ** 2, 10)

    def test_flat_series_rarely_flags_a_changepoint_with_robust_penalty(self):
        # 130 points ~ a monthly series since 2015. (The MAD estimate is
        # noisier than np.std on pure noise: ~10% false positives at 60
        # points, ~5% at 130.)
        false_positives = 0
        for seed in range(100):
            values = np.random.default_rng(seed).normal(10, 1, 130)
            if _pelt_l2(values, min_size=PELT_MIN_SIZE, penalty=_pelt_penalty(values)):
                false_positives += 1
        self.assertLess(false_positives, 10)

    def test_constant_series_has_no_penalty(self):
        self.assertIsNone(_pelt_penalty(np.full(30, 4.0)))


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

