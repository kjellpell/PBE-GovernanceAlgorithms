import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "Inflight_SLA_Risk_Monitor.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {"classify_risk"}
CONSTANT_NAMES = {"RISK_THRESHOLD_KRITISK", "RISK_THRESHOLD_RISIKO"}

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

NAMESPACE = {}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

classify_risk = NAMESPACE["classify_risk"]


class ClassifyRiskTests(unittest.TestCase):
    def test_negative_days_remaining_is_bruddet_even_without_andel_brukt(self):
        self.assertEqual(classify_risk(dager_igjen=-1, andel_brukt=None), "Bruddet")

    def test_negative_days_remaining_beats_a_low_andel_brukt(self):
        # dager_igjen < 0 always wins, regardless of andel_brukt.
        self.assertEqual(classify_risk(dager_igjen=-5, andel_brukt=0.1), "Bruddet")

    def test_kritisk_threshold_boundary(self):
        self.assertEqual(classify_risk(dager_igjen=1, andel_brukt=0.90), "Kritisk")
        self.assertEqual(classify_risk(dager_igjen=1, andel_brukt=0.8999), "Risiko")

    def test_risiko_threshold_boundary(self):
        self.assertEqual(classify_risk(dager_igjen=1, andel_brukt=0.75), "Risiko")
        self.assertEqual(classify_risk(dager_igjen=1, andel_brukt=0.7499), "Innenfor")

    def test_comfortable_margin_is_innenfor(self):
        self.assertEqual(classify_risk(dager_igjen=100, andel_brukt=0.1), "Innenfor")

    def test_none_dager_igjen_is_unclassifiable(self):
        self.assertIsNone(classify_risk(dager_igjen=None, andel_brukt=0.5))

    def test_none_andel_brukt_with_non_negative_dager_igjen_falls_through_to_innenfor(self):
        self.assertEqual(classify_risk(dager_igjen=10, andel_brukt=None), "Innenfor")

    def test_custom_thresholds_override_defaults(self):
        thresholds = {"kritisk": 0.5, "risiko": 0.2}
        self.assertEqual(classify_risk(dager_igjen=1, andel_brukt=0.6, thresholds=thresholds), "Kritisk")
        self.assertEqual(classify_risk(dager_igjen=1, andel_brukt=0.3, thresholds=thresholds), "Risiko")
        self.assertEqual(classify_risk(dager_igjen=1, andel_brukt=0.1, thresholds=thresholds), "Innenfor")


if __name__ == "__main__":
    unittest.main()
