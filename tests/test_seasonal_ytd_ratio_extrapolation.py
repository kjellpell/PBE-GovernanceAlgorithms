import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Seasonal_YTD_ratio_extrapolation.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION_NAMES = {"compute_ytd", "seasonal_ratios", "project_year_end", "validate_results"}
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
