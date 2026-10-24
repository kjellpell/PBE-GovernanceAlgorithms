import ast
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(__file__).parents[1] / "Pause_Reconstruction.py"
TREE = ast.parse(SOURCE.read_text())

FUNCTION_NAMES = {
    "to_day", "from_day", "week_sunday", "iso_week_key", "pair_pauses", "phase_segments",
    "overlap_days", "crossing_day", "clip_from", "count_covering", "build_phase_rows",
    "weekly_table", "agreement",
}
CONSTANT_NAMES = {
    "UENDELIG", "FARESONE_TERSKEL", "VALIDERING_TOLERANSE_DAGER", "STOCK_LABELS", "FLOW_LABELS",
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

# Constants first, so default arguments that reference them resolve.
EXTRACTED.sort(key=lambda n: 0 if isinstance(n, ast.Assign) else 1)
NAMESPACE = {"np": np, "pd": pd}
exec(compile(ast.Module(body=EXTRACTED, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
G = NAMESPACE
UENDELIG = G["UENDELIG"]


def d(s):
    return G["to_day"](date.fromisoformat(s))


def events(rows):
    return pd.DataFrame(rows, columns=["pk_faser", "dag", "hendelse"])


def faser(rows):
    cols = ["pk_faser", "indikator", "enhet", "fasetittel", "start", "slutt",
            "frist_dager", "tidsbruk", "bransjetid", "status"]
    return pd.DataFrame(rows, columns=cols).astype({"slutt": object})


class DayHelperTests(unittest.TestCase):
    def test_round_trip(self):
        for s in ("1970-01-01", "2024-12-30", "2026-10-24"):
            self.assertEqual(G["from_day"](d(s)), date.fromisoformat(s))

    def test_week_sunday_is_the_sunday_of_the_iso_week(self):
        self.assertEqual(G["from_day"](G["week_sunday"](d("2026-10-19"))), date(2026, 10, 25))  # Monday
        self.assertEqual(G["from_day"](G["week_sunday"](d("2026-10-25"))), date(2026, 10, 25))  # Sunday
        self.assertEqual(G["from_day"](G["week_sunday"](d("2024-12-30"))), date(2025, 1, 5))

    def test_iso_week_key_uses_iso_year(self):
        self.assertEqual(G["iso_week_key"](d("2024-12-30")), 202501)
        self.assertEqual(G["iso_week_key"](d("2021-01-01")), 202053)

    def test_missing_dates_are_none(self):
        self.assertIsNone(G["to_day"](None))
        self.assertIsNone(G["to_day"](pd.NaT))


class PairPausesTests(unittest.TestCase):
    def test_request_then_receipt_is_one_pause(self):
        iv, stats = G["pair_pauses"](events([("A", 10, "pause"), ("A", 25, "gjenoppta")]))
        self.assertEqual(iv[["start", "slutt"]].values.tolist(), [[10, 25]])
        self.assertEqual(stats, {"gjentatt_anmodning": 0, "mottatt_uten_anmodning": 0})

    def test_two_rounds(self):
        iv, _ = G["pair_pauses"](events([
            ("A", 10, "pause"), ("A", 20, "gjenoppta"), ("A", 30, "pause"), ("A", 35, "gjenoppta"),
        ]))
        self.assertEqual(iv[["start", "slutt"]].values.tolist(), [[10, 20], [30, 35]])

    def test_repeated_request_keeps_the_first(self):
        iv, stats = G["pair_pauses"](events([
            ("A", 10, "pause"), ("A", 15, "pause"), ("A", 25, "gjenoppta"),
        ]))
        self.assertEqual(iv[["start", "slutt"]].values.tolist(), [[10, 25]])
        self.assertEqual(stats["gjentatt_anmodning"], 1)

    def test_receipt_without_request_is_ignored(self):
        # "Komplett søknad" on day 1 of an application that never needed anything.
        iv, stats = G["pair_pauses"](events([("A", 1, "gjenoppta")]))
        self.assertEqual(len(iv), 0)
        self.assertEqual(stats["mottatt_uten_anmodning"], 1)

    def test_open_request_runs_to_infinity(self):
        iv, _ = G["pair_pauses"](events([("A", 10, "pause")]))
        self.assertEqual(iv[["start", "slutt"]].values.tolist(), [[10, UENDELIG]])

    def test_same_day_receipt_and_new_request_closes_then_reopens(self):
        iv, _ = G["pair_pauses"](events([
            ("A", 10, "pause"), ("A", 20, "pause"), ("A", 20, "gjenoppta"), ("A", 30, "gjenoppta"),
        ]))
        self.assertEqual(iv[["start", "slutt"]].values.tolist(), [[10, 20], [20, 30]])

    def test_faser_are_independent(self):
        iv, _ = G["pair_pauses"](events([
            ("A", 10, "pause"), ("B", 12, "gjenoppta"), ("B", 14, "pause"), ("A", 20, "gjenoppta"),
        ]))
        got = sorted(map(tuple, iv[["pk_faser", "start", "slutt"]].values.tolist()))
        self.assertEqual(got, [("A", 10, 20), ("B", 14, UENDELIG)])


class SegmentTests(unittest.TestCase):
    def test_no_pause(self):
        active, paused = G["phase_segments"](0, 50, [])
        self.assertEqual(active, [(0, 50)])
        self.assertEqual(paused, [])

    def test_pause_splits_the_phase(self):
        active, paused = G["phase_segments"](0, 50, [(10, 20)])
        self.assertEqual(active, [(0, 10), (20, 50)])
        self.assertEqual(paused, [(10, 20)])

    def test_pauses_are_clipped_to_the_phase(self):
        # A pause that started before this phase and one running past its end.
        active, paused = G["phase_segments"](10, 40, [(0, 15), (35, 60)])
        self.assertEqual(paused, [(10, 15), (35, 40)])
        self.assertEqual(active, [(15, 35)])

    def test_overlapping_pauses_are_merged(self):
        _, paused = G["phase_segments"](0, 100, [(10, 30), (20, 40)])
        self.assertEqual(paused, [(10, 40)])

    def test_open_phase_with_open_pause(self):
        active, paused = G["phase_segments"](0, None, [(10, UENDELIG)])
        self.assertEqual(active, [(0, 10)])
        self.assertEqual(paused, [(10, UENDELIG)])

    def test_overlap_days_stops_at_reference_day(self):
        self.assertEqual(G["overlap_days"]([(10, UENDELIG)], 25), 15)
        self.assertEqual(G["overlap_days"]([(10, 20), (30, 40)], 35), 15)


class CrossingTests(unittest.TestCase):
    def test_crossing_without_pause(self):
        self.assertEqual(G["crossing_day"]([(0, 100)], 75), 75)

    def test_pause_pushes_crossing_later(self):
        # 10 active days, 20-day pause, then active: 75 active days reached at 30 + 65.
        self.assertEqual(G["crossing_day"]([(0, 10), (30, 200)], 75), 95)

    def test_never_reached_when_closed_early(self):
        self.assertIsNone(G["crossing_day"]([(0, 50)], 75))

    def test_fractional_need_rounds_up(self):
        self.assertEqual(G["crossing_day"]([(0, 100)], 7.5), 8)


class BuildPhaseRowsTests(unittest.TestCase):
    def setUp(self):
        self.today = 200
        self.f = faser([
            # closed, one pause of 20 days, frist 60
            ("A", "Ind", "Team", "Fase", 0, 100, 60.0, 80.0, 20.0, "Avsluttet"),
            # open and waiting now
            ("B", "Ind", "Team", "Fase", 150, None, 60.0, 10.0, 40.0, "Venter"),
            # open, with us, past 75 % of frist
            ("C", "Ind", "Team", "Fase", 100, None, 84.0, 100.0, 0.0, "Under behandling"),
        ])
        self.pauses = {"A": [(30, 50)], "B": [(160, UENDELIG)]}

    def rows(self):
        rows, iv = G["build_phase_rows"](self.f, self.pauses, {"A"}, self.today)
        return {r["pk_faser"]: r for r in rows}, iv

    def test_clock_is_calendar_minus_pause(self):
        rows, _ = self.rows()
        self.assertEqual(rows["A"]["pause_dager"], 20)
        self.assertEqual(rows["A"]["klokke_dager"], 80)
        self.assertEqual(rows["B"]["pause_dager"], 40)   # still waiting: up to today
        self.assertEqual(rows["B"]["klokke_dager"], 10)

    def test_state(self):
        rows, _ = self.rows()
        self.assertEqual(rows["A"]["tilstand"], "Avsluttet")
        self.assertEqual(rows["B"]["tilstand"], "Venter på søker")
        self.assertEqual(rows["C"]["tilstand"], "Hos oss")

    def test_days_to_first_request_and_withdrawn_flag(self):
        rows, _ = self.rows()
        self.assertEqual(rows["A"]["dager_til_foerste_anmodning"], 30)
        self.assertIsNone(rows["C"]["dager_til_foerste_anmodning"])
        self.assertTrue(rows["A"]["trukket"])
        self.assertFalse(rows["C"]["trukket"])

    def test_risk_crossing_respects_pause(self):
        rows, _ = self.rows()
        # A: 75 % of 60 = 45 active days: 30 before the pause, 15 after -> day 65.
        self.assertEqual(rows["A"]["faresone_dag"], 65)
        # C: 75 % of 84 = 63 days from day 100 -> day 163; over frist at 85 active days.
        self.assertEqual(rows["C"]["faresone_dag"], 163)
        self.assertEqual(rows["C"]["over_frist_dag"], 185)

    def test_waiting_phase_is_never_at_risk_while_waiting(self):
        _, iv = self.rows()
        b_like = iv[(iv["etikett"] == "i_faresonen") & (iv["a"] >= 150)]
        # Only C contributes at-risk intervals after day 150.
        self.assertEqual(b_like[["a", "b"]].values.tolist(), [[163, UENDELIG]])


class WeeklyTableTests(unittest.TestCase):
    def test_counts_cover_week_ends(self):
        self.assertEqual(G["count_covering"]([0, 5], [10, UENDELIG], [4, 7, 10, 20]).tolist(),
                         [1, 2, 1, 1])

    def test_weekly_stocks_and_flows(self):
        intervals = pd.DataFrame([
            ("T", "I", "hos_oss", 0, 14),
            ("T", "I", "venter_paa_soeker", 14, 21),
            ("T", "I", "hos_oss", 21, UENDELIG),
        ], columns=["enhet", "indikator", "etikett", "a", "b"])
        flows = pd.DataFrame([
            ("T", "I", "mottatt", 0),
            ("T", "I", "satt_paa_vent", 14),
        ], columns=["enhet", "indikator", "etikett", "dag"])
        sundays = [G["week_sunday"](0) + 7 * k for k in range(4)]  # day 3, 10, 17, 24
        out = G["weekly_table"](intervals, flows, sundays).set_index("uke_dag")
        self.assertEqual(out.loc[sundays[0], "hos_oss"], 1)
        self.assertEqual(out.loc[sundays[0], "mottatt"], 1)
        self.assertEqual(out.loc[sundays[2], "hos_oss"], 0)
        self.assertEqual(out.loc[sundays[2], "venter_paa_soeker"], 1)
        self.assertEqual(out.loc[sundays[2], "satt_paa_vent"], 1)
        self.assertEqual(out.loc[sundays[3], "hos_oss"], 1)

    def test_all_columns_present_even_without_data_for_some(self):
        intervals = pd.DataFrame([("T", "I", "hos_oss", 0, 30)],
                                 columns=["enhet", "indikator", "etikett", "a", "b"])
        out = G["weekly_table"](intervals, pd.DataFrame(columns=["enhet", "indikator", "etikett", "dag"]),
                                [G["week_sunday"](0)])
        for col in G["STOCK_LABELS"] + G["FLOW_LABELS"]:
            self.assertIn(col, out.columns)


class AgreementTests(unittest.TestCase):
    def test_share_within_tolerance_and_median(self):
        share, median, n = G["agreement"]([10, 20, 30, 40], [10, 21, 35, None], tolerance=2)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(share, 2 / 3)
        self.assertEqual(median, -1.0)

    def test_nothing_to_compare(self):
        self.assertEqual(G["agreement"]([None], [None]), (None, None, 0))


class EndToEndOnSyntheticCasesTests(unittest.TestCase):
    def test_reconstructed_clock_matches_a_simulated_system_clock(self):
        # Simulate a case system that pauses the clock exactly as described:
        # random requests/receipts, tidsbruk = active days. Reconstruction must agree.
        rng = np.random.default_rng(0)
        rows, evs, expected = [], [], {}
        for i in range(300):
            start = int(rng.integers(0, 300))
            length = int(rng.integers(20, 200))
            end = start + length
            t, active, ev = start, 0, []
            while True:
                work = int(rng.integers(5, 60))
                if t + work >= end:
                    active += end - t
                    break
                active += work
                t += work
                wait = int(rng.integers(1, 40))
                ev += [(f"F{i}", t, "pause"), (f"F{i}", min(t + wait, end), "gjenoppta")]
                t = min(t + wait, end)
                if t >= end:
                    break
            rows.append((f"F{i}", "I", "T", "F", start, end, 90.0, float(active), None, "Avsluttet"))
            evs += ev
            expected[f"F{i}"] = active
        iv, _ = G["pair_pauses"](events(evs))
        by = {}
        for f, s, e in iv[["pk_faser", "start", "slutt"]].itertuples(index=False):
            by.setdefault(f, []).append((int(s), int(e)))
        out, _ = G["build_phase_rows"](faser(rows), by, set(), 10_000)
        got = {r["pk_faser"]: r["klokke_dager"] for r in out}
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
