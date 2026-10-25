import ast
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Seasonal_YTD_ratio_extrapolation.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION_NAMES = {
    "trimmed_stats",
    "compute_ytd",
    "monthly_rates",
    "monthly_counts",
    "monthly_rate_ratios",
    "monthly_volume_ratios",
    "project_month_volumes",
    "seasonal_ratios",
    "project_year_end",
    "project_month_rate",
    "seasonal_ratio_on",
    "build_forecast_rows",
    "validate_results",
    "recent_pace",
    "project_year_end_at_pace",
}
CONSTANT_NAMES = {"OUTPUT_COLUMNS", "TEMPO_MAANEDER"}

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

NAMESPACE = {"np": np, "pd": pd, "date": date, "timedelta": timedelta}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

compute_ytd = NAMESPACE["compute_ytd"]
seasonal_ratios = NAMESPACE["seasonal_ratios"]
project_year_end = NAMESPACE["project_year_end"]
monthly_rates = NAMESPACE["monthly_rates"]
monthly_counts = NAMESPACE["monthly_counts"]
monthly_rate_ratios = NAMESPACE["monthly_rate_ratios"]
monthly_volume_ratios = NAMESPACE["monthly_volume_ratios"]
project_month_volumes = NAMESPACE["project_month_volumes"]
project_month_rate = NAMESPACE["project_month_rate"]
seasonal_ratio_on = NAMESPACE["seasonal_ratio_on"]
build_forecast_rows = NAMESPACE["build_forecast_rows"]
validate_results = NAMESPACE["validate_results"]
recent_pace = NAMESPACE["recent_pace"]
project_year_end_at_pace = NAMESPACE["project_year_end_at_pace"]


def monthly_rows(years, missing_months=None, year_end_rate=0.8):
    missing_months = missing_months or set()
    rows = []
    for year in years:
        for month in range(1, 13):
            if (year, month) in missing_months:
                continue
            total = 10
            within = round(total * (year_end_rate if month == 12 else 0.5))
            rows.append(
                {
                    "indikator": "A",
                    "aar": year,
                    "mnd": month,
                    "innenfor": within,
                    "total": total,
                }
            )
    return pd.DataFrame(rows)


def seasonal_monthly_rows(years, shape):
    """One indicator, `shape` giving each month's rate, jittered per year."""
    rows = []
    for offset, year in enumerate(years):
        for month, rate in shape.items():
            total = 100
            rows.append(
                {
                    "indikator": "A",
                    "aar": year,
                    "mnd": month,
                    "innenfor": round(total * (rate + 0.01 * (offset % 3 - 1))),
                    "total": total,
                }
            )
    return pd.DataFrame(rows)


# A year that erodes: strong spring, weak autumn.
ERODING_YEAR = {
    1: 0.90, 2: 0.90, 3: 0.89, 4: 0.87, 5: 0.86, 6: 0.85,
    7: 0.80, 8: 0.82, 9: 0.78, 10: 0.76, 11: 0.74, 12: 0.72,
}


# Per-month rate ratios: how each month's own frist% sits against its year.
MONTH_RATIOS = {
    9:  {"mean_ratio": 1.02, "std_ratio": 0.05, "n_years": 4},
    10: {"mean_ratio": 0.98, "std_ratio": 0.06, "n_years": 4},
    11: {"mean_ratio": 0.94, "std_ratio": 0.07, "n_years": 4},
    12: {"mean_ratio": 0.90, "std_ratio": 0.08, "n_years": 4},
}

# Projected faser counts for the same months, deliberately uneven so tests
# can tell "same count in every month" apart from "actually modelled".
MONTH_VOLUMES = {9: 118.0, 10: 96.0, 11: 130.0, 12: 140.0}


class SeasonalForecastTests(unittest.TestCase):
    def test_compute_ytd_is_weighted_and_cumulative(self):
        data = pd.DataFrame(
            [
                {"indikator": "A", "aar": 2025, "mnd": 1, "innenfor": 1, "total": 2},
                {"indikator": "A", "aar": 2025, "mnd": 2, "innenfor": 8, "total": 8},
            ]
        )

        # Month 1: 1/2 = 0.5. Month 2 cumulates: (1+8)/(2+8) = 9/10 = 0.9 —
        # a volume-weighted cumulative ratio, not an average of monthly ratios.
        self.assertEqual(compute_ytd(data, "A", 2025), {1: 0.5, 2: 0.9})

    def test_incomplete_year_is_not_used_for_seasonal_ratios(self):
        data = monthly_rows([2022, 2023, 2024], missing_months={(2023, 6)})

        self.assertIsNone(seasonal_ratios(data, "A", 2025, min_years=3, trim_n=1))

    def test_three_year_sample_is_not_trimmed_to_one_observation(self):
        data = monthly_rows([2022, 2023, 2024])
        ratios = seasonal_ratios(data, "A", 2025, min_years=3, trim_n=1)

        self.assertEqual(ratios[6]["n_years"], 3)
        self.assertGreater(ratios[6]["std_ratio"], 0)

    def test_projection_is_bounded_and_interval_contains_estimate(self):
        ratios = {6: {"mean_ratio": 0.5, "std_ratio": 0.1, "n_years": 4}}

        estimate, lower, upper = project_year_end(0.9, 6, ratios)

        self.assertTrue(0 <= lower <= estimate <= upper <= 1)

    def test_seasonal_ratio_interpolates_within_the_month(self):
        ratios = {
            8: {"mean_ratio": 0.90, "std_ratio": 0.04, "n_years": 4},
            9: {"mean_ratio": 0.95, "std_ratio": 0.02, "n_years": 4},
        }

        # The last day of a month is that month's ratio unchanged...
        self.assertAlmostEqual(
            seasonal_ratio_on(ratios, date(2025, 9, 30))["mean_ratio"], 0.95
        )
        # ...and four days in, the year has barely moved past August, which is
        # what stops a part-finished month from dragging the projection down.
        self.assertLess(
            seasonal_ratio_on(ratios, date(2025, 9, 4))["mean_ratio"], 0.91
        )

    def test_monthly_rates_are_per_month_not_cumulative(self):
        data = pd.DataFrame(
            [
                {"indikator": "A", "aar": 2025, "mnd": 1, "innenfor": 1, "total": 2},
                {"indikator": "A", "aar": 2025, "mnd": 2, "innenfor": 8, "total": 8},
            ]
        )

        # February stands on its own at 8/8. compute_ytd would carry January in
        # and give 9/10 — that difference is the whole reason this exists: the
        # report's measure has no year-to-date filter over it.
        self.assertEqual(monthly_rates(data, "A", 2025), {1: 0.5, 2: 1.0})
        self.assertEqual(compute_ytd(data, "A", 2025)[2], 0.9)

    def test_monthly_rate_ratios_capture_the_within_year_shape(self):
        data = seasonal_monthly_rows([2021, 2022, 2023, 2024], ERODING_YEAR)

        ratios = monthly_rate_ratios(data, "A", 2025, min_years=3, trim_n=1)

        # A strong month sits above its year, a weak one below.
        self.assertGreater(ratios[1]["mean_ratio"], 1.0)
        self.assertLess(ratios[11]["mean_ratio"], 1.0)
        self.assertGreater(ratios[1]["mean_ratio"], ratios[11]["mean_ratio"])

    def test_month_band_carries_the_month_s_own_variation(self):
        steady   = {"mean_ratio": 1.0, "std_ratio": 0.00, "n_years": 4}
        volatile = {"mean_ratio": 1.0, "std_ratio": 0.06, "n_years": 4}

        _, steady_lo, steady_hi = project_month_rate(0.84, 0.82, 0.86, steady)
        _, wild_lo, wild_hi     = project_month_rate(0.84, 0.82, 0.86, volatile)

        # A month that has swung a lot historically gets a wider band than one
        # that hasn't, even off the same year-end estimate.
        self.assertGreater(wild_hi - wild_lo, steady_hi - steady_lo)
        # And the year-end uncertainty alone still gives it some width.
        self.assertGreater(steady_hi - steady_lo, 0)

    def test_monthly_counts_keeps_the_raw_numerator_and_denominator(self):
        data = pd.DataFrame(
            [
                {"indikator": "A", "aar": 2025, "mnd": 1, "innenfor": 1, "total": 2},
                {"indikator": "A", "aar": 2025, "mnd": 2, "innenfor": 8, "total": 8},
            ]
        )

        counts = monthly_counts(data, "A", 2025)

        # Not divided — this is what DIVIDE(SUM,SUM) needs, unlike
        # monthly_rates, which has already collapsed each month to a ratio.
        self.assertEqual(counts, {1: (1, 2), 2: (8, 8)})

    def test_monthly_volume_ratios_capture_each_month_s_share_of_the_year(self):
        data = seasonal_monthly_rows([2021, 2022, 2023, 2024], ERODING_YEAR)
        # Give December triple the volume of every other month, consistently
        # across all four years, so the share is unambiguous.
        data.loc[data["mnd"] == 12, "total"] *= 3
        data.loc[data["mnd"] == 12, "innenfor"] *= 3

        shares = monthly_volume_ratios(data, "A", 2025, min_years=3, trim_n=1)

        self.assertGreater(shares[12]["mean_ratio"], shares[1]["mean_ratio"])

    def test_project_month_volumes_scales_to_this_year_s_observed_level(self):
        volume_ratios = {
            m: {"mean_ratio": 1 / 12, "std_ratio": 0.0, "n_years": 4}
            for m in range(1, 13)
        }
        # Twice the historical monthly rate observed through August.
        current_year_totals = {m: 200 for m in range(1, 9)}

        volumes = project_month_volumes(current_year_totals, 8, volume_ratios)

        for m in range(9, 13):
            self.assertAlmostEqual(volumes[m], 200, delta=1e-6)

    def test_forecast_counts_reproduce_verdi_and_the_anchor_matches_exactly(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
            anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
        )

        # Every row's stored counts imply the same rate as verdi — this is
        # what a report reading DIVIDE(SUM(innenfor_prognose),
        # SUM(produserte_prognose)) needs to agree with a report reading
        # verdi directly, at the single-row grain.
        for row in rows:
            self.assertAlmostEqual(
                row["innenfor_prognose"] / row["produserte_prognose"],
                row["verdi"],
                delta=0.01,
            )

        # The anchor's counts are the real ones, untouched by the model.
        anchor = rows[0]
        self.assertEqual(anchor["innenfor_prognose"], 82)
        self.assertEqual(anchor["produserte_prognose"], 100)

        # One row per month, carrying that month's modelled total directly —
        # no splitting to reassemble.
        september = next(r for r in rows[1:] if r["analyse_dato"].month == 9)
        self.assertAlmostEqual(
            september["produserte_prognose"], MONTH_VOLUMES[9], delta=0.5
        )

    def test_a_rate_average_would_disagree_with_sum_divide_across_months(self):
        # The concrete failure this whole design avoids: two unequal months
        # averaged as rates give a different number than their counts summed
        # and divided once — which is how the report's own measure works.
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
        )
        by_month = {}
        for row in rows:
            by_month.setdefault(row["analyse_dato"].month, row)

        sep, oct_ = by_month[9], by_month[10]
        rate_average = (sep["verdi"] + oct_["verdi"]) / 2
        counts_divide = (
            (sep["innenfor_prognose"] + oct_["innenfor_prognose"])
            / (sep["produserte_prognose"] + oct_["produserte_prognose"])
        )

        self.assertNotAlmostEqual(rate_average, counts_divide, places=3)

    def test_forecast_is_a_period_rate_not_a_cumulative_path(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
            anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
        )
        by_month = {row["analyse_dato"].month: row["verdi"] for row in rows[1:]}

        # One row per month, and the months differ from each other — the
        # shape is the seasonal shape, not a drift towards the year-end
        # number: a weak November projects below a strong September.
        self.assertEqual(set(by_month), {9, 10, 11, 12})
        self.assertEqual(len(set(by_month.values())), 4)
        self.assertGreater(by_month[9], by_month[11])

    def test_forecast_forks_off_the_last_complete_month(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
            anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
        )
        dates = [row["analyse_dato"] for row in rows]

        # The anchor is August's observed rate, one row per remaining month
        # picks up from September, and the part-finished September is
        # projected rather than anchored on.
        self.assertEqual(rows[0]["type"], "Anker")
        self.assertEqual(rows[0]["analyse_dato"], date(2025, 8, 31))
        self.assertEqual(rows[0]["verdi"], 0.82)
        self.assertEqual(
            rows[0]["nedre_konfidensgrense"], rows[0]["oevre_konfidensgrense"]
        )
        self.assertEqual(dates[1], date(2025, 9, 30))
        self.assertEqual(dates[-1], date(2025, 12, 31))
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(len(dates), len(set(dates)))
        self.assertEqual(len(rows), 5)  # anchor + Sep, Oct, Nov, Dec
        self.assertEqual(set(row["type"] for row in rows[1:]), {"Prognose"})

    def test_year_end_interval_is_separate_from_the_month_bands(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
            anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
        )
        december = rows[-1]

        # The year-end interval rides on every row untouched; a month's band is
        # a different, wider thing and must not be read as the year's.
        self.assertEqual(set(row["nedre_aarsslutt"] for row in rows), {0.79})
        self.assertEqual(set(row["oevre_aarsslutt"] for row in rows), {0.89})
        self.assertNotEqual(
            december["nedre_konfidensgrense"], december["nedre_aarsslutt"]
        )

    def test_every_row_carries_the_year_end_estimate_for_the_cards(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
            anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
        )

        # The year-end number is cumulative YTD and is not the last point of
        # the line any more — the line is a month rate. It rides along on every
        # row so a card can read it without a date filter.
        self.assertEqual(set(row["prognose_aarsslutt"] for row in rows), {0.84})

    def test_forecast_rows_pass_validation(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
            anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
        )

        validate_results(rows)

    def test_forecast_works_without_an_anchor(self):
        # January: no complete month yet, so there is nothing to fork from.
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
        )

        self.assertEqual(len(rows), 4)  # Sep, Oct, Nov, Dec
        self.assertEqual(set(row["type"] for row in rows), {"Prognose"})

    def test_no_rows_when_no_month_can_be_projected(self):
        self.assertEqual(
            build_forecast_rows(
                "A", date(2025, 9, 1), {}, {}, 0.84, 0.79, 0.89,
                pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
                anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
            ),
            [],
        )

    def test_result_validation_rejects_schema_drift(self):
        row = {
            "indikator": "A",
            "analyse_dato": pd.Timestamp("2025-06-30").date(),
            "type": "Prognose",
            "verdi": 0.7,
            "nedre_konfidensgrense": 0.6,
            "oevre_konfidensgrense": 0.8,
            "innenfor_prognose": 70.0,
            "produserte_prognose": 100.0,
            "prognose_aarsslutt": 0.75,
            "nedre_aarsslutt": 0.72,
            "oevre_aarsslutt": 0.78,
            "innenfor_tempo": None,
            "prognose_aarsslutt_tempo": None,
            "kjoert_tidspunkt": pd.Timestamp("2025-06-30").to_pydatetime(),
            "kjoere_id": "batch",
        }

        validate_results([row])
        row["periode"] = 202506
        with self.assertRaises(ValueError):
            validate_results([row])


    def test_validation_rejects_counts_that_disagree_with_verdi(self):
        row = {
            "indikator": "A",
            "analyse_dato": pd.Timestamp("2025-06-30").date(),
            "type": "Prognose",
            "verdi": 0.7,
            "nedre_konfidensgrense": 0.6,
            "oevre_konfidensgrense": 0.8,
            "innenfor_prognose": 40.0,   # implies 0.4, not 0.7
            "produserte_prognose": 100.0,
            "prognose_aarsslutt": 0.75,
            "nedre_aarsslutt": 0.72,
            "oevre_aarsslutt": 0.78,
            "innenfor_tempo": None,
            "prognose_aarsslutt_tempo": None,
            "kjoert_tidspunkt": pd.Timestamp("2025-06-30").to_pydatetime(),
            "kjoere_id": "batch",
        }

        with self.assertRaises(ValueError):
            validate_results([row])


class PaceScenarioTests(unittest.TestCase):
    def frame(self):
        # 2024: Oct-Dec at 60 %; 2025: Jan-Aug at 90 %, then Jun-Aug drop to 70 %.
        rows = []
        for m in (10, 11, 12):
            rows.append({"indikator": "A", "aar": 2024, "mnd": m, "innenfor": 60, "total": 100})
        for m in range(1, 9):
            within = 70 if m >= 6 else 90
            rows.append({"indikator": "A", "aar": 2025, "mnd": m, "innenfor": within, "total": 100})
        return pd.DataFrame(rows)

    def test_pace_is_pooled_over_the_last_three_complete_months(self):
        self.assertAlmostEqual(recent_pace(self.frame(), "A", 2025, 8), 0.70)

    def test_pace_walks_back_across_the_year_boundary(self):
        # In February the last three complete months are Nov, Dec and Jan.
        self.assertAlmostEqual(recent_pace(self.frame(), "A", 2025, 1), (60 + 60 + 90) / 300)

    def test_no_pace_without_production(self):
        self.assertIsNone(recent_pace(self.frame(), "B", 2025, 8))

    def test_year_end_at_pace_combines_observed_months_with_pace_on_remaining_volume(self):
        counts = monthly_counts(self.frame(), "A", 2025)
        volumes = {9: 100.0, 10: 100.0, 11: 100.0, 12: 100.0}
        # Jan-Aug: 5 x 90 + 3 x 70 = 660 of 800; rest 400 at 70 % = 280.
        self.assertAlmostEqual(project_year_end_at_pace(counts, 8, volumes, 0.70), (660 + 280) / 1200, places=4)

    def test_year_end_at_pace_needs_pace_and_volume(self):
        counts = monthly_counts(self.frame(), "A", 2025)
        self.assertIsNone(project_year_end_at_pace(counts, 8, {}, 0.7))
        self.assertIsNone(project_year_end_at_pace(counts, 8, {9: 100.0}, None))

    def test_pace_line_forks_from_the_anchor_and_runs_at_the_pace(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
            anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100,
            pace=0.70, year_end_pace=0.80,
        )
        anchor = [r for r in rows if r["type"] == "Anker"][0]
        self.assertEqual(anchor["innenfor_tempo"], 82)          # real count: same fork point
        for r in rows:
            self.assertEqual(r["prognose_aarsslutt_tempo"], 0.80)
            if r["type"] == "Prognose":
                self.assertAlmostEqual(r["innenfor_tempo"] / r["produserte_prognose"], 0.70, places=3)
        validate_results(rows)

    def test_existing_prognose_columns_are_unchanged_by_the_pace(self):
        kwargs = dict(anchor_date=date(2025, 8, 31), anchor_innenfor=82, anchor_total=100)
        args = ("A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
                pd.Timestamp("2025-09-05").to_pydatetime(), "batch")
        without = build_forecast_rows(*args, **kwargs)
        with_pace = build_forecast_rows(*args, **kwargs, pace=0.70, year_end_pace=0.80)
        keep = ["verdi", "innenfor_prognose", "produserte_prognose", "prognose_aarsslutt"]
        self.assertEqual([{k: r[k] for k in keep} for r in without],
                         [{k: r[k] for k in keep} for r in with_pace])
        self.assertTrue(all(r["innenfor_tempo"] is None for r in without))

    def test_validation_rejects_pace_count_above_volume(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 1), MONTH_RATIOS, MONTH_VOLUMES, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch", pace=0.70, year_end_pace=0.80,
        )
        rows[0]["innenfor_tempo"] = rows[0]["produserte_prognose"] + 5
        with self.assertRaises(ValueError):
            validate_results(rows)


if __name__ == "__main__":
    unittest.main()
