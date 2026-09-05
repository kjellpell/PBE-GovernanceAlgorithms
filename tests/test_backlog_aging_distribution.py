import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "Backlog_Aging_Distribution.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {"bucket_age"}
CONSTANT_NAMES = {"AGE_BUCKETS"}

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

bucket_age  = NAMESPACE["bucket_age"]
AGE_BUCKETS = NAMESPACE["AGE_BUCKETS"]


class BucketAgeTests(unittest.TestCase):
    def test_zero_days_is_first_bucket(self):
        self.assertEqual(bucket_age(0), "0-30")

    def test_seam_between_first_and_second_bucket(self):
        self.assertEqual(bucket_age(30), "0-30")
        self.assertEqual(bucket_age(31), "31-60")

    def test_seam_between_second_and_third_bucket(self):
        self.assertEqual(bucket_age(60), "31-60")
        self.assertEqual(bucket_age(61), "61-90")

    def test_seam_between_third_and_fourth_bucket(self):
        self.assertEqual(bucket_age(90), "61-90")
        self.assertEqual(bucket_age(91), "91-180")

    def test_seam_between_fourth_and_fifth_bucket(self):
        self.assertEqual(bucket_age(180), "91-180")
        self.assertEqual(bucket_age(181), "181-365")

    def test_seam_between_fifth_and_open_ended_bucket(self):
        self.assertEqual(bucket_age(365), "181-365")
        self.assertEqual(bucket_age(366), "365+")

    def test_open_ended_top_bucket_has_no_upper_limit(self):
        self.assertEqual(bucket_age(10_000), "365+")

    def test_negative_age_is_none(self):
        self.assertIsNone(bucket_age(-1))

    def test_none_age_is_none(self):
        self.assertIsNone(bucket_age(None))

    def test_custom_bucket_list_is_honoured(self):
        custom = [(0, 10, "fresh"), (11, None, "stale")]
        self.assertEqual(bucket_age(5, buckets=custom), "fresh")
        self.assertEqual(bucket_age(11, buckets=custom), "stale")
        self.assertEqual(bucket_age(1000, buckets=custom), "stale")


if __name__ == "__main__":
    unittest.main()
