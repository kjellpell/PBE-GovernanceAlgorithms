import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Throughput_Pressure_Monitor.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {
    "_period_to_month_index", "_month_index_to_period", "fill_month_gaps", "combine_flow",
}
CONSTANT_NAMES = set()

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

combine_flow = NAMESPACE["combine_flow"]

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


if __name__ == "__main__":
    unittest.main()
