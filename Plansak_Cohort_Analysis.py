# =============================================================================
# Plansak cohort analysis — tracks how planning cases (Plansak) progress
# from Oppstartsmøte through to Sendt til politisk behandling.
#
# A "case" (Saksnummer) spans multiple process steps (Prosess_id), each with
# its own Indikator. This script assembles the full picture per case and tracks
# cohorts defined by specific cut dates ("all cases started since date X").
#
# Output tables:
#   plansak_case_detail    — every Prosesser row for every Plansak case (QA)
#   plansak_case_metadata  — one row per Saksnummer with derived case fields
#   plansak_cohort_tracking — cohort × months_elapsed survival table
#
# Power BI data model:
#   plansak_case_detail  →[Saksnummer]→  plansak_case_metadata
#   plansak_cohort_tracking filtered by cohort_cut_date for each "since X" line
#
# Schedule: nightly after main data pipeline.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

spark    = SparkSession.builder.getOrCreate()
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY    = pd.Timestamp(date.today())

# -----------------------------------------------------------------------------
# CONFIG — adjust cut dates and horizon as needed
# Each entry creates one "since X" cohort line in plansak_cohort_tracking.
# Cases started on or after the cut date are included in that cohort.
# -----------------------------------------------------------------------------
COHORT_CUT_DATES = [
    "2024-01-01",
    "2026-01-01",
]
MAX_MONTHS      = 84   # tracking horizon — 7 years covers the longest cases
MIN_COHORT_SIZE = 5    # suppress cohorts smaller than this in tracking table
START_INDICATOR = "Oppstartsmøte"
END_INDICATOR   = "Sendt til politisk behandling"


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS plansak_case_detail (
    Saksnummer          STRING      NOT NULL,
    Prosess_id          STRING      NOT NULL,
    Indikator           STRING,
    Startdato           DATE,
    Sluttdato           DATE,
    Startoppgave        STRING,
    Sluttoppgave        STRING,
    Tidsbruk            DOUBLE,
    Bransjetid          DOUBLE,
    Totaltid            DOUBLE,
    process_is_open     BOOLEAN     NOT NULL,
    computed_at         TIMESTAMP   NOT NULL,
    batch_id            STRING      NOT NULL
)
USING DELTA
COMMENT 'One row per Prosess_id for all Plansak cases. Join to plansak_case_metadata on Saksnummer for case-level context.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS plansak_case_metadata (
    Saksnummer              STRING      NOT NULL,
    case_start_date         DATE,
    case_end_date           DATE,
    case_is_open            BOOLEAN     NOT NULL,
    case_status             STRING,
    last_activity_date      DATE,
    months_elapsed          INT,
    weeks_elapsed           INT,
    sum_tidsbruk            DOUBLE,
    sum_bransjetid          DOUBLE,
    sum_totaltid            DOUBLE,
    process_count           INT         NOT NULL,
    open_process_count      INT         NOT NULL,
    closed_process_count    INT         NOT NULL,
    computed_at             TIMESTAMP   NOT NULL,
    batch_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'One row per Saksnummer. Case lifecycle fields derived from Prosesser. Joins 1-to-many to plansak_case_detail.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS plansak_cohort_tracking (
    cohort_cut_date     STRING      NOT NULL,
    cohort_size         INT         NOT NULL,
    months_elapsed      INT         NOT NULL,
    weeks_elapsed       INT         NOT NULL,
    eligible_count      INT         NOT NULL,
    open_count          INT         NOT NULL,
    pct_open            DOUBLE      NOT NULL,
    avg_tidsbruk        DOUBLE,
    avg_bransjetid      DOUBLE,
    avg_totaltid        DOUBLE,
    sum_tidsbruk        DOUBLE,
    sum_bransjetid      DOUBLE,
    sum_totaltid        DOUBLE,
    computed_at         TIMESTAMP   NOT NULL,
    batch_id            STRING      NOT NULL
)
USING DELTA
COMMENT 'Cohort survival table. One row per cohort_cut_date × months_elapsed. Filter by cohort_cut_date in Power BI to show a single since-date line.'
""")

print("Output tables ready")


# =============================================================================
# CELL 2 — Load raw Plansak data
# =============================================================================
# Identify all Saksnummer that have at least one Indikator classified as
# Fagomrade = 'Plansak' in Felles.Indikatorer, then pull ALL Prosesser rows
# for those Saksnummer — not just the Plansak-classified ones, so we see the
# full process chain for every case.

raw = spark.sql("""
    SELECT
        p.Saksnummer,
        p.Prosess_id,
        p.Indikator,
        p.Startdato,
        p.Sluttdato,
        p.Startoppgave,
        p.Sluttoppgave,
        p.Tidsbruk,
        p.Bransjetid,
        p.Totaltid
    FROM Prosesser p
    WHERE p.Saksnummer IN (
        SELECT DISTINCT p2.Saksnummer
        FROM Prosesser p2
        JOIN Felles.Indikatorer fi ON p2.Indikator = fi.Indikator
        WHERE fi.Fagomrade = 'Plansak'
          AND p2.Saksnummer IS NOT NULL
    )
""").toPandas()

raw["Startdato"] = pd.to_datetime(raw["Startdato"])
raw["Sluttdato"] = pd.to_datetime(raw["Sluttdato"])

print(f"Rows loaded:    {len(raw):,}")
print(f"Unique cases:   {raw['Saksnummer'].nunique():,}")
print(f"Unique process: {raw['Prosess_id'].nunique():,}")
print(f"Indikators:     {raw['Indikator'].nunique()}")


# =============================================================================
# CELL 3 — Compute case-level metadata
# =============================================================================
# For each Saksnummer derive:
#   case_start_date  — Startdato of Oppstartsmøte row (fallback: earliest Startdato)
#   case_end_date    — Sluttdato of Sendt til politisk behandling (None if not reached)
#   case_status      — Indikator with the most recent Sluttdato; if none are
#                      finished, the Indikator with the most recent Startdato
#   last_activity_date — most recent date seen across all rows
#   months/weeks elapsed — from case_start_date to today or case_end_date
#   time aggregates  — sum of Tidsbruk, Bransjetid, Totaltid per case
#   process counts   — total, open, closed

case_meta_rows = []

for saksnummer, grp in raw.groupby("Saksnummer"):
    # case start
    start_rows = grp[grp["Indikator"] == START_INDICATOR]
    if not start_rows.empty and start_rows["Startdato"].notna().any():
        case_start = start_rows["Startdato"].dropna().min()
    else:
        case_start = grp["Startdato"].dropna().min() if grp["Startdato"].notna().any() else pd.NaT

    # case end
    end_rows = grp[grp["Indikator"] == END_INDICATOR]
    if not end_rows.empty and end_rows["Sluttdato"].notna().any():
        case_end = end_rows["Sluttdato"].dropna().max()
    else:
        case_end = pd.NaT

    case_is_open = pd.isna(case_end)

    # case status — last finished indicator, or last started if nothing finished
    finished = grp[grp["Sluttdato"].notna()]
    if not finished.empty:
        status_row = finished.loc[finished["Sluttdato"].idxmax()]
        case_status = status_row["Indikator"]
    else:
        started = grp[grp["Startdato"].notna()]
        if not started.empty:
            status_row = started.loc[started["Startdato"].idxmax()]
            case_status = status_row["Indikator"]
        else:
            case_status = None

    # last activity
    all_dates = pd.concat([
        grp["Startdato"].dropna(),
        grp["Sluttdato"].dropna(),
    ])
    last_activity = all_dates.max() if not all_dates.empty else pd.NaT

    # elapsed time
    if pd.notna(case_start):
        reference = case_end if pd.notna(case_end) else TODAY
        delta = relativedelta(reference.to_pydatetime(), case_start.to_pydatetime())
        months_elapsed = delta.years * 12 + delta.months
        weeks_elapsed  = int((reference - case_start).days / 7)
    else:
        months_elapsed = None
        weeks_elapsed  = None

    # time aggregates — NaN treated as 0 for sums
    sum_tidsbruk   = grp["Tidsbruk"].sum(min_count=1)
    sum_bransjetid = grp["Bransjetid"].sum(min_count=1)
    sum_totaltid   = grp["Totaltid"].sum(min_count=1)

    process_count        = len(grp)
    open_process_count   = int(grp["Sluttdato"].isna().sum())
    closed_process_count = process_count - open_process_count

    case_meta_rows.append({
        "Saksnummer":           saksnummer,
        "case_start_date":      case_start,
        "case_end_date":        case_end,
        "case_is_open":         case_is_open,
        "case_status":          case_status,
        "last_activity_date":   last_activity,
        "months_elapsed":       months_elapsed,
        "weeks_elapsed":        weeks_elapsed,
        "sum_tidsbruk":         sum_tidsbruk   if pd.notna(sum_tidsbruk)   else None,
        "sum_bransjetid":       sum_bransjetid if pd.notna(sum_bransjetid) else None,
        "sum_totaltid":         sum_totaltid   if pd.notna(sum_totaltid)   else None,
        "process_count":        process_count,
        "open_process_count":   open_process_count,
        "closed_process_count": closed_process_count,
    })

case_meta = pd.DataFrame(case_meta_rows)
case_meta["case_start_date"]    = pd.to_datetime(case_meta["case_start_date"])
case_meta["case_end_date"]      = pd.to_datetime(case_meta["case_end_date"])
case_meta["last_activity_date"] = pd.to_datetime(case_meta["last_activity_date"])

missing_start = case_meta["case_start_date"].isna().sum()
print(f"Case metadata built: {len(case_meta):,} cases")
print(f"Open cases:          {case_meta['case_is_open'].sum():,}")
print(f"Missing start date:  {missing_start:,}  (no {START_INDICATOR!r} row found — check data)")


# =============================================================================
# CELL 4 — Build and write plansak_case_detail
# =============================================================================

detail = raw.copy()
detail["process_is_open"] = detail["Sluttdato"].isna()
detail["computed_at"]     = datetime.now()
detail["batch_id"]        = BATCH_ID

spark.createDataFrame(detail).write.mode("overwrite").saveAsTable("plansak_case_detail")
print(f"plansak_case_detail written: {len(detail):,} rows")


# =============================================================================
# CELL 5 — Build and write plansak_case_metadata
# =============================================================================

meta_out = case_meta.copy()
meta_out["computed_at"] = datetime.now()
meta_out["batch_id"]    = BATCH_ID

spark.createDataFrame(meta_out).write.mode("overwrite").saveAsTable("plansak_case_metadata")
print(f"plansak_case_metadata written: {len(meta_out):,} rows")


# =============================================================================
# CELL 6 — Compute cohort tracking
# =============================================================================
# For each cut date, and for each month mark 0..MAX_MONTHS:
#   eligible  = cases whose case_start_date >= cut_date AND
#               case_start_date <= (today - month_mark months)
#               (only cases old enough to have reached this mark)
#   open      = eligible cases where case_end_date IS NULL OR
#               case_end_date > (case_start_date + month_mark months)
#   pct_open  = open / eligible
#
# Time metrics (tidsbruk etc.) are summed/averaged over the eligible pool
# at that point — reflecting accumulated work for cases at that stage.

# work only with cases that have a valid start date
trackable = case_meta[case_meta["case_start_date"].notna()].copy()

tracking_rows = []

for cut_date_str in COHORT_CUT_DATES:
    cut_date      = pd.Timestamp(cut_date_str)
    cohort_cases  = trackable[trackable["case_start_date"] >= cut_date].copy()
    cohort_size   = len(cohort_cases)

    if cohort_size < MIN_COHORT_SIZE:
        print(f"Skipping {cut_date_str}: only {cohort_size} cases (< {MIN_COHORT_SIZE})")
        continue

    print(f"Cohort {cut_date_str}: {cohort_size} cases")

    for month_mark in range(0, MAX_MONTHS + 1):
        # earliest point in time that a case must have started to have reached
        # month_mark months of age by today
        age_cutoff = TODAY - relativedelta(months=month_mark)

        eligible = cohort_cases[cohort_cases["case_start_date"] <= age_cutoff].copy()
        eligible_count = len(eligible)

        if eligible_count == 0:
            break  # no cases old enough — stop extending this cohort

        # reference date per case = case_start + month_mark months
        eligible["_ref"] = eligible["case_start_date"].apply(
            lambda d: d + relativedelta(months=month_mark)
        )

        open_mask = (
            eligible["case_end_date"].isna() |
            (eligible["case_end_date"] > eligible["_ref"])
        )
        open_count = int(open_mask.sum())
        pct_open   = round(open_count / eligible_count, 4)

        tracking_rows.append({
            "cohort_cut_date":  cut_date_str,
            "cohort_size":      cohort_size,
            "months_elapsed":   month_mark,
            "weeks_elapsed":    month_mark * 4,
            "eligible_count":   eligible_count,
            "open_count":       open_count,
            "pct_open":         pct_open,
            "avg_tidsbruk":     eligible["sum_tidsbruk"].mean()   if eligible["sum_tidsbruk"].notna().any()   else None,
            "avg_bransjetid":   eligible["sum_bransjetid"].mean() if eligible["sum_bransjetid"].notna().any() else None,
            "avg_totaltid":     eligible["sum_totaltid"].mean()   if eligible["sum_totaltid"].notna().any()   else None,
            "sum_tidsbruk":     eligible["sum_tidsbruk"].sum()    if eligible["sum_tidsbruk"].notna().any()   else None,
            "sum_bransjetid":   eligible["sum_bransjetid"].sum()  if eligible["sum_bransjetid"].notna().any() else None,
            "sum_totaltid":     eligible["sum_totaltid"].sum()    if eligible["sum_totaltid"].notna().any()   else None,
        })

tracking = pd.DataFrame(tracking_rows)

if tracking.empty:
    print("No cohort tracking rows produced — check COHORT_CUT_DATES and MIN_COHORT_SIZE.")
else:
    tracking["computed_at"] = datetime.now()
    tracking["batch_id"]    = BATCH_ID

    spark.createDataFrame(tracking).write.mode("overwrite").saveAsTable("plansak_cohort_tracking")
    print(f"\nplansak_cohort_tracking written: {len(tracking):,} rows")
    print(f"Cohorts: {tracking['cohort_cut_date'].unique().tolist()}")


# =============================================================================
# CELL 7 — Verification
# =============================================================================

print("\n=== ROW COUNTS ===")
spark.sql("SELECT COUNT(*) AS rows FROM plansak_case_detail").show()
spark.sql("SELECT COUNT(*) AS cases FROM plansak_case_metadata").show()
spark.sql("""
    SELECT cohort_cut_date, MAX(months_elapsed) AS max_months, COUNT(*) AS rows
    FROM plansak_cohort_tracking
    GROUP BY cohort_cut_date
    ORDER BY cohort_cut_date
""").show()

print("\n=== CASES MISSING START DATE (no Oppstartsmøte row) ===")
spark.sql("""
    SELECT Saksnummer, process_count, case_status, last_activity_date
    FROM plansak_case_metadata
    WHERE case_start_date IS NULL
    ORDER BY last_activity_date DESC
    LIMIT 20
""").show(truncate=False)

print("\n=== COHORT SURVIVAL AT MONTH 12 AND 24 ===")
spark.sql("""
    SELECT cohort_cut_date, months_elapsed, cohort_size, eligible_count,
           open_count, ROUND(pct_open * 100, 1) AS pct_open,
           ROUND(avg_totaltid, 1) AS avg_totaltid
    FROM plansak_cohort_tracking
    WHERE months_elapsed IN (12, 24)
    ORDER BY cohort_cut_date, months_elapsed
""").show(truncate=False)

print("\n=== CASE STATUS DISTRIBUTION ===")
spark.sql("""
    SELECT case_status, case_is_open, COUNT(*) AS cases,
           ROUND(AVG(months_elapsed), 1) AS avg_months_elapsed
    FROM plansak_case_metadata
    GROUP BY case_status, case_is_open
    ORDER BY cases DESC
    LIMIT 20
""").show(truncate=False)


# =============================================================================
# CELL 8 — Power BI notes
# =============================================================================
#
# DATA MODEL:
#   plansak_case_detail  → [Saksnummer] →  plansak_case_metadata
#   Relationship: many detail rows to one metadata row.
#
# RECOMMENDED VISUALS:
#
# 1. COHORT SURVIVAL LINE CHART (plansak_cohort_tracking):
#    X axis:  months_elapsed
#    Y axis:  pct_open (format as %)
#    Lines:   one per cohort_cut_date — add as legend/series
#    Filter:  select which cut dates to compare
#    Reading: a line dropping quickly = cases resolving faster.
#             A flatter line = cases stalling.
#
# 2. CASE STATUS TABLE (plansak_case_metadata):
#    Rows:    Saksnummer
#    Columns: case_start_date, months_elapsed, case_status, case_is_open,
#             sum_totaltid, sum_tidsbruk, sum_bransjetid
#    Filter:  case_is_open = TRUE for active cases
#    Use:     spot which active cases are oldest or have the most time logged
#
# 3. DETAIL DRILL-THROUGH (plansak_case_detail):
#    From a Saksnummer in the case table, drill through to see all
#    individual process rows — check for missing Sluttdato, unusual Tidsbruk etc.
#    Columns: Indikator, Startdato, Sluttdato, Tidsbruk, Bransjetid, Totaltid,
#             process_is_open
#
# 4. TIME AGGREGATES BY COHORT (plansak_cohort_tracking):
#    X axis:  months_elapsed
#    Y axis:  avg_totaltid or sum_totaltid
#    Lines:   one per cohort_cut_date
#    Use:     how much cumulative time are cases consuming at each stage
