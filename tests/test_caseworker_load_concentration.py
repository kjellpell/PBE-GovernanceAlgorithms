import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "Caseworker_Load_Concentration.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTIONS = {
    node.name: node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "gini_coefficient"
}
NAMESPACE = {}
exec(compile(ast.Module(body=list(FUNCTIONS.values()), type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)

gini_coefficient = NAMESPACE["gini_coefficient"]


class GiniCoefficientTests(unittest.TestCase):
    def test_empty_list_is_none(self):
        self.assertIsNone(gini_coefficient([]))

    def test_single_value_is_none(self):
        self.assertIsNone(gini_coefficient([5]))

    def test_all_zero_is_none(self):
        self.assertIsNone(gini_coefficient([0, 0, 0]))

    def test_perfectly_equal_caseloads_is_zero(self):
        self.assertAlmostEqual(gini_coefficient([5, 5, 5, 5]), 0.0)

    def test_single_dominant_caseworker_is_highly_concentrated(self):
        gini = gini_coefficient([0, 0, 10])
        self.assertGreater(gini, 0.0)
        self.assertLess(gini, 1.0)
        # Should be close to the theoretical max for n=3: (n-1)/n
        self.assertAlmostEqual(gini, 2 / 3, places=2)

    def test_negative_value_raises(self):
        with self.assertRaises(ValueError):
            gini_coefficient([-1, 5])

    def test_floats_and_ints_both_work(self):
        self.assertAlmostEqual(gini_coefficient([1, 2, 3, 4]), gini_coefficient([1.0, 2.0, 3.0, 4.0]))

    def test_none_entries_are_filtered_out_before_counting(self):
        # A single real value plus a None should behave like a single-value input (None).
        self.assertIsNone(gini_coefficient([5, None]))


if __name__ == "__main__":
    unittest.main()
