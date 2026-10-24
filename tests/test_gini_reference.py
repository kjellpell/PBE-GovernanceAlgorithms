"""Reference check for the DAX measure Gini-koeffisient (nå) in
Caseworker_Load_Concentration_POWERBI_DAX.md. DAX can't run here, so this
reimplements the measure's arithmetic in numpy and checks it against the
textbook sorted-index Gini."""
import unittest

import numpy as np


def gini_sorted(x):
    """Textbook Gini: sort ascending, ranks 1..N."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    ranks = np.arange(1, n + 1)
    return 2 * np.sum(ranks * x) / (n * np.sum(x)) - (n + 1) / n


def gini_dax_average_rank(x):
    """Mirror of the fixed DAX: @Rang = (# strictly smaller) + (# equal + 1) / 2."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rank = np.array([np.sum(x < v) + (np.sum(x == v) + 1) / 2 for v in x])
    return 2 * np.sum(rank * x) / (n * np.sum(x)) - (n + 1) / n


def gini_dax_dense_rank(x):
    """Mirror of the old DAX: RANKX(..., ASC, DENSE)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    distinct = np.unique(x)
    rank = np.array([np.searchsorted(distinct, v) + 1 for v in x])
    return 2 * np.sum(rank * x) / (n * np.sum(x)) - (n + 1) / n


class GiniReferenceTests(unittest.TestCase):
    CASES = [
        ([10, 10, 10, 10], 0.000, -0.750),
        ([10, 10, 10, 40], 0.321, -0.464),
        ([5, 5, 20, 20, 20], 0.257, -0.457),
    ]

    def test_handover_cases(self):
        for loads, correct, old in self.CASES:
            with self.subTest(loads=loads):
                self.assertAlmostEqual(gini_dax_average_rank(loads), correct, places=3)
                self.assertAlmostEqual(gini_sorted(loads), correct, places=3)
                # documents the bug that was fixed
                self.assertAlmostEqual(gini_dax_dense_rank(loads), old, places=3)

    def test_average_rank_is_exact_for_random_arrays_with_ties(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            n = int(rng.integers(3, 15))
            x = rng.integers(1, 6, size=n)  # small range -> many ties
            self.assertAlmostEqual(gini_dax_average_rank(x), gini_sorted(x), places=12)

    def test_never_negative_for_positive_loads(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            x = rng.integers(1, 30, size=int(rng.integers(3, 12)))
            self.assertGreaterEqual(gini_dax_average_rank(x), -1e-12)

    def test_maximum_for_small_teams_is_n_minus_one_over_n(self):
        for n in (3, 5, 12):
            x = [0] * (n - 1) + [1]
            # one person carries everything (zero rows are filtered out in the
            # DAX table, so this is the theoretical ceiling, not a reachable value)
            self.assertAlmostEqual(gini_sorted(x), (n - 1) / n)


if __name__ == "__main__":
    unittest.main()
