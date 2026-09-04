import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Seasonal_YTD_ratio_extrapolation.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION_NAMES = {
    "compute_ytd",
    "seasonal_ratios",
    "project_year_end",
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

NAMESPACE = {"np": np, "pd": pd}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

compute_ytd = NAMESPACE["compute_ytd"]
seasonal_ratios = NAMESPACE["seasonal_ratios"]
project_year_end = NAMESPACE["project_year_end"]
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

    def test_forecast_series_is_anchored_on_the_last_actual_month(self):
        ratios = {
            9:  {"mean_ratio": 0.95, "std_ratio": 0.02, "n_years": 4},
            10: {"mean_ratio": 0.97, "std_ratio": 0.02, "n_years": 4},
            11: {"mean_ratio": 0.99, "std_ratio": 0.02, "n_years": 4},
            12: {"mean_ratio": 1.00, "std_ratio": 0.00, "n_years": 4},
        }

        rows = build_forecast_rows(
            "A", 9, 0.80, ratios, 0.84, 0.79, 0.89,
            2025, pd.Timestamp("2025-10-01").to_pydatetime(), "batch",
        )

        # The first row is the observed YTD at the last closed month, so the
        # forecast line starts where the actual line ends instead of floating
        # unattached over the remaining months.
        anchor = rows[0]
        self.assertEqual(anchor["type"], "Anker")
        self.assertEqual(anchor["analyse_dato"], pd.Timestamp("2025-09-30").date())
        self.assertEqual(anchor["verdi"], 0.80)
        self.assertEqual([row["type"] for row in rows[1:]], ["Prognose"] * 3)
        self.assertEqual(
            [row["analyse_dato"].month for row in rows[1:]], [10, 11, 12]
        )

    def test_forecast_band_widens_towards_the_year_end_interval(self):
        ratios = {
            9:  {"mean_ratio": 0.95, "std_ratio": 0.02, "n_years": 4},
            10: {"mean_ratio": 0.97, "std_ratio": 0.02, "n_years": 4},
            12: {"mean_ratio": 1.00, "std_ratio": 0.00, "n_years": 4},
        }

        rows = build_forecast_rows(
            "A", 9, 0.80, ratios, 0.84, 0.79, 0.89,
            2025, pd.Timestamp("2025-10-01").to_pydatetime(), "batch",
        )
        widths = [
            row["oevre_konfidensgrense"] - row["nedre_konfidensgrense"]
            for row in rows
        ]

        # The anchor is observed, so it carries no uncertainty; the band then
        # opens up with the horizon and closes on the year-end interval in
        # December, which is what the KPI cards read.
        self.assertEqual(widths[0], 0)
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(rows[-1]["nedre_konfidensgrense"], 0.79)
        self.assertEqual(rows[-1]["oevre_konfidensgrense"], 0.89)

    def test_forecast_line_lands_on_the_year_end_card_value(self):
        ratios = {
            9:  {"mean_ratio": 0.95, "std_ratio": 0.02, "n_years": 4},
            12: {"mean_ratio": 1.00, "std_ratio": 0.00, "n_years": 4},
        }

        rows = build_forecast_rows(
            "A", 9, 0.80, ratios, 0.84, 0.79, 0.89,
            2025, pd.Timestamp("2025-10-01").to_pydatetime(), "batch",
        )

        december = rows[-1]
        self.assertEqual(december["verdi"], december["prognose_aarsslutt"])

    def test_forecast_rows_pass_validation(self):
        ratios = {
            9:  {"mean_ratio": 0.95, "std_ratio": 0.02, "n_years": 4},
            10: {"mean_ratio": 0.97, "std_ratio": 0.02, "n_years": 4},
            11: {"mean_ratio": 0.99, "std_ratio": 0.02, "n_years": 4},
            12: {"mean_ratio": 1.00, "std_ratio": 0.00, "n_years": 4},
        }

        rows = build_forecast_rows(
            "A", 9, 0.80, ratios, 0.84, 0.79, 0.89,
            2025, pd.Timestamp("2025-10-01").to_pydatetime(), "batch",
        )

        validate_results(rows)

    def test_no_rows_when_there_is_nothing_left_to_forecast(self):
        ratios = {12: {"mean_ratio": 1.0, "std_ratio": 0.0, "n_years": 4}}

        # December is closed — an anchor on its own is not a forecast.
        self.assertEqual(
            build_forecast_rows(
                "A", 12, 0.84, ratios, 0.84, 0.80, 0.88,
                2025, pd.Timestamp("2026-01-01").to_pydatetime(), "batch",
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
