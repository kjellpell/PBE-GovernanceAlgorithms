# =============================================================================
# Cohort analysis — tracks resolution rate of cases grouped by intake month.
# Answers: are cases received recently resolving at the same rate as
# historical cohorts, or are they accumulating?
#
# Output table: cohort_results
# Power BI visual: heatmap — cohort month on Y axis, weeks since intake
# on X axis, cell colour = fraction still open. Dark = slow resolution.
#
# Schedule: nightly after main data pipeline.
# Minimum cohort size: 10 cases. Suppress smaller cohorts.
# History: uses all available data — 15-20 years gives stable baseline.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()
BATCH_ID        = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY           = date.today()
MIN_COHORT_SIZE = 10
MAX_WEEKS       = 26  # track cohorts for up to 26 weeks after intake
START_YEAR      = 2015  # exclude data before this year — adjust if older data is reliable


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS cohort_results (
    indicator               STRING      NOT NULL,
    cohort_period           INT         NOT NULL,  -- YYYYMM intake month
    cohort_size             INT         NOT NULL,  -- cases received that month
    weeks_since_intake      INT         NOT NULL,  -- weeks since cohort intake
    open_count              INT         NOT NULL,  -- cases still open at this week
    pct_open                DOUBLE      NOT NULL,  -- fraction still open (0-1)
    pct_open_historical     DOUBLE,                -- historical avg at same week
    delta_historical        DOUBLE,                -- current - historical
    is_recent_cohort        BOOLEAN     NOT NULL,  -- last 6 months = TRUE
    computed_at             TIMESTAMP   NOT NULL,
    batch_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'Cohort resolution rates by intake month. One row per cohort × week combination.'
""")

print("cohort_results table ready")


# =============================================================================
# CELL 2 — Load case data
# =============================================================================
# One row per case with intake month and days to resolution (or NULL if open).
# Uses tidligste_startmilepael_dato for intake, Sluttdato for resolution.
# Excludes avtalt indicators — no fixed deadline to measure against.

cases = spark.sql(f"""
    SELECT
        pr.Indikator                                 AS indicator,
        DATE_FORMAT(pr.tidligste_startmilepael_dato, 'yyyyMM')  AS cohort_period,
        YEAR(pr.tidligste_startmilepael_dato) * 100
            + MONTH(pr.tidligste_startmilepael_dato)            AS cohort_period_int,
        pr.tidligste_startmilepael_dato                         AS intake_date,
        pr.siste_stoppmilepael_dato                              AS end_date,
        CASE
            WHEN pr.siste_stoppmilepael_dato IS NOT NULL
            THEN DATEDIFF(pr.siste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)
            ELSE NULL
        END                                                     AS dager_til_ferdig,
        CASE
            WHEN pr.seneste_stoppmilepael_dato IS NULL THEN 1 ELSE 0
        END                                                     AS is_open
    FROM Prosesser pr
    WHERE pr.tidligste_startmilepael_dato IS NOT NULL
      AND YEAR(pr.tidligste_startmilepael_dato) >= {START_YEAR}
""").toPandas()

cases["cohort_period_int"] = cases["cohort_period_int"].astype(int)
cases["intake_date"]        = pd.to_datetime(cases["intake_date"])
cases["end_date"]           = pd.to_datetime(cases["end_date"])

print(f"Cases loaded: {len(cases):,}")
print(f"Indicators:   {cases['indicator'].nunique()}")
print(f"Date range:   {cases['intake_date'].min().date()} → {cases['intake_date'].max().date()}")


# =============================================================================
# CELL 3 — Compute cohort resolution rates
# =============================================================================

results = []
today_ts = pd.Timestamp(TODAY)

for indicator, ind_cases in cases.groupby("indicator"):

    # All distinct cohort months for this indicator
    cohort_months = sorted(ind_cases["cohort_period_int"].unique())

    # Cutoff for "recent" cohorts — last 6 months
    current_yyyymm  = int(TODAY.strftime("%Y%m"))
        _rc_yr, _rc_mo  = TODAY.year, TODAY.month - 6
        if _rc_mo <= 0:
            _rc_mo += 12; _rc_yr -= 1
        recent_cutoff   = _rc_yr * 100 + _rc_mo
    for cohort_periode in cohort_months:
        cohort_cases = ind_cases[
            ind_cases["cohort_period_int"] == cohort_periode
        ].copy()

        cohort_size = len(cohort_cases)
        if cohort_size < MIN_COHORT_SIZE:
            continue

        # Cohort intake reference date — first day of intake month
        yr  = cohort_periode // 100
        mth = cohort_periode % 100
        cohort_start = pd.Timestamp(yr, mth, 1)
        is_recent     = cohort_periode >= recent_cutoff

        # For each week bucket up to MAX_WEEKS, count how many cases
        # from this cohort were still open at that point in time.
        for uke in range(1, MAX_WEEKS + 1):
            reference_date = cohort_start + pd.Timedelta(weeks=uke)

            # Can't measure future weeks for recent cohorts
            if reference_date > today_ts:
                break

            # Cases still open at reference_date:
            # either never finished, or finished after reference_date
            open_count = len(cohort_cases[
                cohort_cases["end_date"].isna() |
                (cohort_cases["end_date"] > reference_date)
            ])

            pct_open = open_count / cohort_size

            results.append({
                "indicator":            indicator,
                "cohort_period":       cohort_periode,
                "cohort_size":          cohort_size,
                "weeks_since_intake":   uke,
                "open_count":           open_count,
                "pct_open":             round(pct_open, 4),
                "is_recent_cohort":     is_recent,
            })

print(f"Cohort rows computed: {len(results):,}")


# =============================================================================
# CELL 4 — Compute historical baseline per indicator × week
# =============================================================================
# Historical average andel_åpen at each week across all non-recent cohorts.
# Used to compare recent cohorts against — are they resolving faster or slower?
# Uses trimmed mean (drop top and bottom 10%) to exclude exceptional years.

df = pd.DataFrame(results)

historical = (
    df[~df["is_recent_cohort"]]
    .groupby(["indicator", "weeks_since_intake"]) ["pct_open"]
    .apply(lambda x: float(np.mean(
        np.sort(x.values)[
            max(0, int(len(x) * 0.1)) : max(1, int(len(x) * 0.9))
        ]
    )) if len(x) >= 5 else float(x.mean()))
    .reset_index()
    .rename(columns={"pct_open": "pct_open_historical"})
)

df = df.merge(historical, on=["indicator", "weeks_since_intake"], how="left")
df["delta_historical"] = df["pct_open"] - df["pct_open_historical"]
df["delta_historical"] = df["delta_historical"].round(4)

print(f"Historical baseline computed for {historical['indicator'].nunique()} indicators")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

if df.empty:
    print("No cohort results to write.")
else:
    df["computed_at"] = datetime.now()
    df["batch_id"]    = BATCH_ID

    # Idempotent — full overwrite since cohort history is recomputed each run.
    # Open cases change as they resolve, so historical rows can change too.
    results_spark = spark.createDataFrame(df)
    results_spark.write.mode("overwrite").saveAsTable("cohort_results")

    print(f"\ncohort_results written: {len(df):,} rows")
    print(f"Indicators: {df['indicator'].nunique()}")
    print(f"Cohorts:    {df['cohort_period'].nunique()}")
    print(f"Recent cohorts flagged: {df[df['er_nylig_kohort']]['cohort_periode'].nunique()}")

    # Summary — recent cohorts vs historical baseline at week 12
    print("\n=== RECENT COHORTS VS BASELINE AT WEEK 12 ===")
    spark.sql("""
        SELECT
            indicator,
            cohort_period,
            cohort_size,
            ROUND(pct_open * 100, 1)             AS pct_open,
            ROUND(pct_open_historical * 100, 1)  AS pct_open_historical,
            ROUND(delta_historical * 100, 1)     AS delta_pct
        FROM cohort_results
        WHERE weeks_since_intake = 12
          AND is_recent_cohort   = TRUE
          AND pct_open_historical IS NOT NULL
        ORDER BY delta_historical DESC, indicator
    """).show(30, truncate=False)


# =============================================================================
# CELL 6 — Power BI visual notes
# =============================================================================
# HEATMAP (primary visual):
#   Rows:    cohort_periode (intake month) — recent at top
#   Columns: uker_siden_mottak (1 to 26)
#   Values:  andel_åpen — format as %
#   Colour:  gradient dark (high %) → light (low %)
#            Dark cell = many cases still open at that week = slow resolution
#   Filter:  er_nylig_kohort = TRUE for governance team view
#            Remove filter for full historical view
#
# LINE CHART (secondary visual — recent vs historical):
#   X axis:  uker_siden_mottak
#   Lines:   andel_åpen per recent cohort_periode (one line per month)
#            andel_åpen_historisk as a single reference line (trimmed mean)
#   Shading: area between recent lines and historical line
#            Above historical = resolving slower than normal
#            Below historical = resolving faster than normal
#
# KEY SIGNAL:
#   A recent cohort line consistently above the historical reference line
#   means cases from that intake month are sitting longer than normal.
#   This appears in the portfolio before it appears in frist%.
#   Governance team acts here — board sees it later in portfolio age chart.
