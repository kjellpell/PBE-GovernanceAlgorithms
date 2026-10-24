import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Throughput_Pressure_Monitor.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {
    "_period_to_month_index", "_month_index_to_period", "fill_month_gaps", "combine_flow",
    "seasonal_flow_baseline", "flow_pressure", "pressure_label",
}
CONSTANT_NAMES = {
    "BASELINE_MONTHS", "MIN_BASELINE_OBS", "NETTO_FLYT_REL_TERSKEL",
    "SEASONAL_BASELINE_YEARS", "MIN_SEASONAL_BASELINE_YEARS",
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

combine_flow           = NAMESPACE["combine_flow"]
seasonal_flow_baseline = NAMESPACE["seasonal_flow_baseline"]
flow_pressure          = NAMESPACE["flow_pressure"]
pressure_label         = NAMESPACE["pressure_label"]


def team_months(mottatt, ferdigstilt, first_year=2020):
    n = len(mottatt)
    periods = [(first_year + i // 12) * 100 + (i % 12 + 1) for i in range(n)]
    g = pd.DataFrame({
        "period": periods,
        "mottatt_antall": mottatt,
        "ferdigstilt_antall": ferdigstilt,
    })
    g["netto_flyt"] = g["mottatt_antall"] - g["ferdigstilt_antall"]
    return g

PHASE_KEYS = ["indikator", "enhet", "fasetittel"]


class CombineFlowTests(unittest.TestCase):
    def test_phase_flow_counts_intake_and_output_in_their_own_months(self):
        keys = {"indikator": "A", "enhet": "E1", "fasetittel": "F1"}
        received = pd.DataFrame([{**keys, "period": 202403, "mottatt_antall": 5}])
        completed = pd.DataFrame([{
            **keys, "period": 202405, "ferdigstilt_antall": 5,
            "median_tidsbruk": 30.0, "p90_tidsbruk": 45.0,
        }])

        flow = combine_flow(received, completed, PHASE_KEYS)

        self.assertEqual(flow["period"].tolist(), [202403, 202404, 202405])
        self.assertEqual(flow["netto_flyt"].tolist(), [5, 0, -5])

    def test_year_boundary_gap_is_filled(self):
        keys = {"indikator": "A", "enhet": "E1"}
        received = pd.DataFrame([{**keys, "period": 202411, "mottatt_antall": 3}])
        completed = pd.DataFrame([{
            **keys, "period": 202502, "ferdigstilt_antall": 3,
            "median_tidsbruk": 1.0, "p90_tidsbruk": 2.0,
        }])

        flow = combine_flow(received, completed, ["indikator", "enhet"])

        self.assertEqual(flow["period"].tolist(), [202411, 202412, 202501, 202502])
        self.assertEqual(flow["netto_flyt"].tolist(), [3, 0, 0, -3])



class FlowPressureTests(unittest.TestCase):
    def test_small_net_inflow_on_a_large_team_is_not_pressure(self):
        # 200 finished a month; +1 is noise, +40 is pressure.
        g = team_months([200] * 12 + [201, 240], [200] * 14)

        out = flow_pressure(g)

        self.assertFalse(out["netto_flyt_press"].iloc[12])
        self.assertTrue(out["netto_flyt_press"].iloc[13])
        self.assertAlmostEqual(out["netto_flyt_rel"].iloc[13], 0.2)

    def test_normal_january_surge_is_not_pressure(self):
        # Every January takes in 30 more than it finishes, for years.
        mottatt = []
        for _ in range(4):
            mottatt += [130] + [100] * 11
        g = team_months(mottatt, [100] * 48)

        out = flow_pressure(g)

        januaries = out[out["period"] % 100 == 1]
        # 2020 and 2021 have no seasonal baseline yet; from 2022 the
        # January surge is the normal and no longer pressure.
        self.assertTrue(januaries["netto_flyt_press"].iloc[1])
        self.assertFalse(januaries["netto_flyt_press"].iloc[2])
        self.assertFalse(januaries["netto_flyt_press"].iloc[3])
        self.assertEqual(januaries["netto_flyt_saesong_baseline"].iloc[3], 30.0)

    def test_streak_counts_consecutive_pressure_months(self):
        g = team_months([100] * 12 + [130, 130, 100, 130], [100] * 16)

        out = flow_pressure(g)

        self.assertEqual(out["netto_flyt_streak"].iloc[12:].tolist(), [1, 2, 0, 1])

    def test_streak_does_not_run_across_a_month_without_activity(self):
        # A team with no rows in 202402 used to have that month skipped, so
        # Jan and Mar counted as two months in a row.
        keys = {"indikator": "A", "enhet": "E1"}
        received = pd.DataFrame(
            [{**keys, "period": p, "mottatt_antall": 10} for p in
             [202301 + i for i in range(12)]] +
            [{**keys, "period": 202401, "mottatt_antall": 20},
             {**keys, "period": 202403, "mottatt_antall": 20}]
        )
        completed = pd.DataFrame(
            [{**keys, "period": p, "ferdigstilt_antall": 10, "median_tidsbruk": 1.0, "p90_tidsbruk": 2.0}
             for p in [202301 + i for i in range(12)]] +
            [{**keys, "period": 202401, "ferdigstilt_antall": 10, "median_tidsbruk": 1.0, "p90_tidsbruk": 2.0},
             {**keys, "period": 202403, "ferdigstilt_antall": 10, "median_tidsbruk": 1.0, "p90_tidsbruk": 2.0}]
        )

        out = flow_pressure(combine_flow(received, completed, ["indikator", "enhet"]))

        self.assertEqual(out["period"].iloc[-3:].tolist(), [202401, 202402, 202403])
        self.assertEqual(out["netto_flyt_streak"].iloc[-3:].tolist(), [1, 0, 1])

    def test_no_output_baseline_means_no_pressure_flag(self):
        g = team_months([10, 12, 15], [0, 0, 0])

        out = flow_pressure(g)

        self.assertFalse(out["netto_flyt_press"].any())


class PressureLabelTests(unittest.TestCase):
    def test_norwegian_level_names(self):
        self.assertEqual(
            [pressure_label(x) for x in (0.0, 1.5, 3.0, 4.0)],
            ["Lav", "Moderat", "Høy", "Kritisk"],
        )


class SeasonalFlowBaselineTests(unittest.TestCase):
    def test_median_of_same_month_in_prior_three_years(self):
        periods = [202001, 202101, 202201, 202301, 202401]
        values = [5.0, 50.0, 7.0, 9.0, 0.0]

        baseline = seasonal_flow_baseline(periods, values)

        self.assertTrue(np.isnan(baseline[0]) and np.isnan(baseline[1]))
        self.assertEqual(baseline[2], 27.5)   # median(5, 50)
        self.assertEqual(baseline[3], 7.0)    # median(5, 50, 7)
        self.assertEqual(baseline[4], 9.0)    # median(50, 7, 9) — 2020 dropped


if __name__ == "__main__":
    unittest.main()
