import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Phase_Bottleneck_Detector.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {
    "_period_to_month_index", "_month_index_to_period", "fill_month_gaps",
    "combine_flow", "queue_at_month_end", "score_bottleneck", "severity_label",
}
CONSTANT_NAMES = {"PHASE_KEYS", "KO_VEKST_TERSKEL"}

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

combine_flow       = NAMESPACE["combine_flow"]
queue_at_month_end = NAMESPACE["queue_at_month_end"]
score_bottleneck   = NAMESPACE["score_bottleneck"]
severity_label     = NAMESPACE["severity_label"]


def phase_events(n, start_period, end_period=None):
    """n synthetic faser for one enhet x fase, started in start_period and
    (if end_period is given) ended in end_period. Returns the two aggregates
    the Spark queries produce: intake by start month, output by end month."""
    keys = {"indikator": "A", "enhet": "E1", "fasetittel": "F1"}
    received = pd.DataFrame([{**keys, "period": start_period, "inngang_antall": n}])
    if end_period is None:
        completed = pd.DataFrame(columns=list(keys) + ["period", "utgang_antall", "median_fasetid", "p90_fasetid"])
    else:
        completed = pd.DataFrame([{
            **keys, "period": end_period, "utgang_antall": n,
            "median_fasetid": 40.0, "p90_fasetid": 55.0,
        }])
    return received, completed


class CombineFlowTests(unittest.TestCase):
    def test_closed_faser_count_in_their_own_months(self):
        # n faser open in March, close in May. Old logic grouped both counts by
        # COALESCE(slutt, start) = May, giving 0/0/0. Real flow is +n/0/-n.
        received, completed = phase_events(7, 202403, 202405)

        flow = combine_flow(received, completed)

        self.assertEqual(flow["period"].tolist(), [202403, 202404, 202405])
        self.assertEqual(flow["netto_flyt"].tolist(), [7, 0, -7])
        self.assertEqual(flow["inngang_antall"].tolist(), [7, 0, 0])
        self.assertEqual(flow["utgang_antall"].tolist(), [0, 0, 7])

    def test_gap_month_is_a_zero_row_with_no_phase_time(self):
        received, completed = phase_events(3, 202411, 202501)

        flow = combine_flow(received, completed)

        self.assertEqual(flow["period"].tolist(), [202411, 202412, 202501])
        self.assertTrue(np.isnan(flow.loc[1, "p90_fasetid"]))
        self.assertAlmostEqual(flow.loc[2, "p90_fasetid"], 55.0)

    def test_still_open_faser_only_add_intake(self):
        received, completed = phase_events(4, 202406)

        flow = combine_flow(received, completed)

        self.assertEqual(flow["netto_flyt"].tolist(), [4])
        self.assertEqual(flow["utgang_antall"].tolist(), [0])

    def test_groups_are_filled_independently(self):
        r1, c1 = phase_events(2, 202401, 202403)
        r2, c2 = phase_events(5, 202402, 202402)
        r2["fasetittel"] = "F2"
        c2["fasetittel"] = "F2"

        flow = combine_flow(pd.concat([r1, r2]), pd.concat([c1, c2]))

        f1 = flow[flow["fasetittel"] == "F1"]
        f2 = flow[flow["fasetittel"] == "F2"]
        self.assertEqual(f1["netto_flyt"].tolist(), [2, 0, -2])
        self.assertEqual(f2["netto_flyt"].tolist(), [0])


class QueueTests(unittest.TestCase):
    def test_queue_is_opening_stock_plus_cumulative_net_flow(self):
        queue = queue_at_month_end(np.array([7, 0, -7]), opening_stock=10)

        self.assertEqual(queue.tolist(), [17, 17, 10])

    def test_queue_never_negative(self):
        queue = queue_at_month_end(np.array([-3, 1]), opening_stock=0)

        self.assertEqual(queue.tolist(), [0, 0])


class ScoringTests(unittest.TestCase):
    def test_balanced_month_without_deterioration_scores_zero(self):
        score, reasons = score_bottleneck(0, 0.0, 20, 20, False)

        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])
        self.assertEqual(severity_label(score), "Lav")

    def test_net_inflow_alone_is_only_moderate(self):
        score, reasons = score_bottleneck(2, np.nan, 22, 20, False)

        self.assertEqual(reasons, ["NETTO_FLYT_POS"])
        self.assertEqual(severity_label(score), "Moderat")

    def test_growing_queue_with_output_gap_is_high(self):
        score, reasons = score_bottleneck(10, 0.0, 30, 20, True)

        self.assertEqual(reasons, ["NETTO_FLYT_POS", "UTGANGSGAP", "KO_VEKST"])
        self.assertEqual(severity_label(score), "Høy")

    def test_everything_wrong_is_critical(self):
        score, _ = score_bottleneck(10, 0.4, 30, 20, True)

        self.assertEqual(severity_label(score), "Kritisk")


if __name__ == "__main__":
    unittest.main()
