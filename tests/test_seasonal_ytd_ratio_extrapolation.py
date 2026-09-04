import ast
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Seasonal_YTD_ratio_extrapolation.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION_NAMES = {
    "compute_ytd",
    "seasonal_ratios",
    "project_year_end",
    "seasonal_ratio_on",
    "build_forecast_rows",
    "validate_results",
}
CONSTANT_NAMES = {"OUTPUT_COLUMNS"}

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
seasonal_ratio_on = NAMESPACE["seasonal_ratio_on"]
build_forecast_rows = NAMESPACE["build_forecast_rows"]
validate_results = NAMESPACE["validate_results"]


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


SEASONAL_RATIOS = {
    8:  {"mean_ratio": 0.90, "std_ratio": 0.04, "n_years": 4},
    9:  {"mean_ratio": 0.95, "std_ratio": 0.03, "n_years": 4},
    10: {"mean_ratio": 0.97, "std_ratio": 0.02, "n_years": 4},
    11: {"mean_ratio": 0.99, "std_ratio": 0.01, "n_years": 4},
    12: {"mean_ratio": 1.00, "std_ratio": 0.00, "n_years": 4},
}


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

    def test_forecast_runs_daily_from_the_anchor_to_year_end(self):
        ratios = SEASONAL_RATIOS

        rows = build_forecast_rows(
            "A", date(2025, 9, 4), 0.80, ratios, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
        )
        dates = [row["analyse_dato"] for row in rows]

        # A continuous daily line, starting on the observed value and ending on
        # 31 December — not a handful of month-end points.
        self.assertEqual(rows[0]["type"], "Anker")
        self.assertEqual(rows[0]["analyse_dato"], date(2025, 9, 4))
        self.assertEqual(rows[0]["verdi"], 0.80)
        self.assertEqual(dates[-1], date(2025, 12, 31))
        self.assertEqual(len(dates), len(set(dates)))
        self.assertEqual(dates, sorted(dates))
        self.assertEqual((dates[-1] - dates[0]).days + 1, len(dates))
        self.assertEqual(set(row["type"] for row in rows[1:]), {"Prognose"})

    def test_forecast_band_widens_towards_the_year_end_interval(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 4), 0.80, SEASONAL_RATIOS, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
        )
        width = {
            row["analyse_dato"]:
                row["oevre_konfidensgrense"] - row["nedre_konfidensgrense"]
            for row in rows
        }

        # The anchor is observed, so it carries no uncertainty; the band then
        # opens up with the horizon and closes on the year-end interval on
        # 31 December, which is what the KPI cards read. Sampled monthly —
        # day to day the widths move by less than the stored rounding.
        self.assertEqual(width[date(2025, 9, 4)], 0)
        self.assertLess(width[date(2025, 9, 30)], width[date(2025, 10, 31)])
        self.assertLess(width[date(2025, 10, 31)], width[date(2025, 11, 30)])
        self.assertLess(width[date(2025, 11, 30)], width[date(2025, 12, 31)])
        self.assertEqual(rows[-1]["nedre_konfidensgrense"], 0.79)
        self.assertEqual(rows[-1]["oevre_konfidensgrense"], 0.89)

    def test_forecast_line_lands_on_the_year_end_card_value(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 4), 0.80, SEASONAL_RATIOS, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
        )

        december = rows[-1]
        self.assertEqual(december["verdi"], december["prognose_aarsslutt"])

    def test_forecast_path_may_decline(self):
        # frist% YTD is a ratio, not a running count: if the months ahead are
        # worse than the year so far, the projected line has to be allowed to
        # fall, otherwise it flattens into a line that says nothing.
        declining = {
            9:  {"mean_ratio": 1.10, "std_ratio": 0.02, "n_years": 4},
            10: {"mean_ratio": 1.06, "std_ratio": 0.02, "n_years": 4},
            11: {"mean_ratio": 1.03, "std_ratio": 0.02, "n_years": 4},
            12: {"mean_ratio": 1.00, "std_ratio": 0.00, "n_years": 4},
        }

        rows = build_forecast_rows(
            "A", date(2025, 9, 30), 0.88, declining, 0.80, 0.76, 0.84,
            pd.Timestamp("2025-10-01").to_pydatetime(), "batch",
        )
        forecast = [row["verdi"] for row in rows[1:]]

        self.assertEqual(forecast, sorted(forecast, reverse=True))
        self.assertLess(forecast[-1], rows[0]["verdi"])

    def test_forecast_rows_pass_validation(self):
        rows = build_forecast_rows(
            "A", date(2025, 9, 4), 0.80, SEASONAL_RATIOS, 0.84, 0.79, 0.89,
            pd.Timestamp("2025-09-05").to_pydatetime(), "batch",
        )

        validate_results(rows)

    def test_no_rows_when_there_is_nothing_left_to_forecast(self):
        # The year is over — an anchor on its own is not a forecast.
        self.assertEqual(
            build_forecast_rows(
                "A", date(2025, 12, 31), 0.84, SEASONAL_RATIOS, 0.84, 0.80, 0.88,
                pd.Timestamp("2026-01-01").to_pydatetime(), "batch",
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
            "prognose_aarsslutt": 0.75,
            "kjoert_tidspunkt": pd.Timestamp("2025-06-30").to_pydatetime(),
            "kjoere_id": "batch",
        }

        validate_results([row])
        row["periode"] = 202506
        with self.assertRaises(ValueError):
            validate_results([row])


if __name__ == "__main__":
    unittest.main()
