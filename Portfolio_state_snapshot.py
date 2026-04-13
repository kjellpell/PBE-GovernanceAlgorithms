# =============================================================================
# Portfolio state snapshot — open case classification per indicator.
# Computes workable, paused, mini-hearing (Begrenset høring), and
# tidligbehandling counts per indicator as of the last day of the
# previous calendar month.
#
# Requires Saksbehandling.Milepeler — the milestone event log.
# Window logic classifies each open case by its latest milestone state.
# This cannot be reproduced in DAX at case grain.
#
# Tidligbehandling is derived from Milepeler because the column is not
# available on the locked Prosesser partition.
#
# Output table: portefolje_snapshot (append-only, one row per indicator per month)
# Power BI:     filter snapshot_month = MAX(snapshot_month) for current view,
#               or any past month for historical board packs.
#
# Schedule: first working day of each month, after transaction lock closes.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, date, timedelta

spark = SparkSession.builder.getOrCreate()
BATCH_ID       = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY          = date.today()

# Snapshot month-end — last day of the previous calendar month.
# Override: SNAPSHOT_MONTH = date(2026, 2, 28)
SNAPSHOT_MONTH = TODAY.replace(day=1) - timedelta(days=1)

# Milestone names — adjust to match actual values in Milepeler.Milepel.
PAUSED_MILESTONES          = [
    "Behov for tilleggsdokumentasjon",
    "Anmodning om tilleggsdokumentasjon",
]
RESUME_MILESTONES          = [
    "Komplett søknad",
    "Mottatt tilleggsdokumentasjon",
]
WORKABLE_MILESTONES        = [
    "Søknad mottatt",
    "Komplett søknad",
    "Mottatt tilleggsdokumentasjon",
]
MINI_HEARING_MILESTONE     = "Begrenset høring"
TIDLIGBEHANDLING_MILESTONE = "Tidligbehandling"  # adjust if milestone name differs

OUTPUT_TABLE  = "portefolje_snapshot"
snapshot_str  = SNAPSHOT_MONTH.strftime("%Y-%m-%d")


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
    snapshot_month          DATE        NOT NULL,
    indikator               STRING      NOT NULL,
    open_total              INT         NOT NULL,  -- all open cases at month-end
    open_workable           INT         NOT NULL,  -- open + in active working state
    open_paused             INT         NOT NULL,  -- open + waiting on applicant
    open_mini_hearing       INT         NOT NULL,  -- open + latest milestone = Begrenset høring
    tidligbehandling_count  INT         NOT NULL,  -- cases that reached early processing this month
    computed_at             TIMESTAMP   NOT NULL,
    batch_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'Monthly portfolio state snapshot per indicator. open_workable/paused/mini_hearing require milestone window logic — computed here, not in DAX.'
""")

print(f"Output table {OUTPUT_TABLE} ready")
print(f"Snapshot month: {SNAPSHOT_MONTH}")


# =============================================================================
# CELL 2 — Load open cases at month-end
# =============================================================================
# A case is open at month-end if:
#   tidligste_startmilepael_dato <= SNAPSHOT_MONTH
#   AND (Sluttdato IS NULL OR Sluttdato > SNAPSHOT_MONTH)

open_cases = spark.sql(f"""
    SELECT
        Prosess_id,
        Indikator
    FROM Prosesser
    WHERE Indikator IS NOT NULL
      AND tidligste_startmilepael_dato IS NOT NULL
      AND tidligste_startmilepael_dato <= '{snapshot_str}'
      AND (
            siste_stoppmilepael_dato IS NULL
            OR siste_stoppmilepael_dato > '{snapshot_str}'
          )
""")

print(f"Open cases at {SNAPSHOT_MONTH}: {open_cases.count():,}")


# =============================================================================
# CELL 3 — Classify each open case by milestone state
# =============================================================================
# Load all milestone events up to and including the snapshot month.
# Restrict to open cases only to keep the join small.

milestones = spark.sql(f"""
    SELECT
        Prosess_id,
        Milepel,
        Sluttdato AS milestone_ts
    FROM Milepeler
    WHERE Sluttdato IS NOT NULL
      AND Sluttdato <= '{snapshot_str}'
""")

open_with_milestones = open_cases.join(milestones, "Prosess_id", "left")

# Latest milestone per case — determines current state
w_latest = Window.partitionBy("Prosess_id").orderBy(F.col("milestone_ts").desc())
latest_milestone = (
    open_with_milestones
    .withColumn("rn", F.row_number().over(w_latest))
    .filter(F.col("rn") == 1)
    .select("Prosess_id", F.col("Milepel").alias("latest_milepel"))
)

# Latest pause and resume timestamps — determines paused state
latest_pause = (
    open_with_milestones
    .filter(F.col("Milepel").isin(PAUSED_MILESTONES))
    .groupBy("Prosess_id")
    .agg(F.max("milestone_ts").alias("latest_pause_ts"))
)
latest_resume = (
    open_with_milestones
    .filter(F.col("Milepel").isin(RESUME_MILESTONES))
    .groupBy("Prosess_id")
    .agg(F.max("milestone_ts").alias("latest_resume_ts"))
)

# Classify each open case
case_state = (
    open_cases
    .join(latest_milestone, "Prosess_id", "left")
    .join(latest_pause,     "Prosess_id", "left")
    .join(latest_resume,    "Prosess_id", "left")
    .withColumn(
        "is_paused",
        F.col("latest_pause_ts").isNotNull() & (
            F.col("latest_resume_ts").isNull() |
            (F.col("latest_pause_ts") > F.col("latest_resume_ts"))
        )
    )
    .withColumn(
        "is_workable",
        (~F.col("is_paused")) &
        F.col("latest_milepel").isin(WORKABLE_MILESTONES)
    )
    .withColumn(
        "is_mini_hearing",
        F.col("latest_milepel") == F.lit(MINI_HEARING_MILESTONE)
    )
)

state_summary = case_state.groupBy("Indikator").agg(
    F.count("*")                                              .alias("open_total"),
    F.sum(F.when(F.col("is_workable"),     1).otherwise(0))  .alias("open_workable"),
    F.sum(F.when(F.col("is_paused"),       1).otherwise(0))  .alias("open_paused"),
    F.sum(F.when(F.col("is_mini_hearing"), 1).otherwise(0))  .alias("open_mini_hearing"),
)

print("Case state per indicator:")
state_summary.orderBy("Indikator").show(truncate=False)


# =============================================================================
# CELL 4 — Tidligbehandling from Milepeler
# =============================================================================
# Tidligbehandling_dato is not on the locked Prosesser partition.
# Derived from Milepeler: cases where the milestone event occurred
# within the snapshot month (not just any time in history).

year  = SNAPSHOT_MONTH.year
month = SNAPSHOT_MONTH.month

tidligbehandling = spark.sql(f"""
    SELECT
        pr.Indikator,
        COUNT(DISTINCT mi.Prosess_id) AS tidligbehandling_count
    FROM Milepeler mi
    INNER JOIN Prosesser pr ON mi.Prosess_id = pr.Prosess_id
    WHERE mi.Milepel  = '{TIDLIGBEHANDLING_MILESTONE}'
      AND YEAR(mi.Sluttdato)  = {year}
      AND MONTH(mi.Sluttdato) = {month}
      AND pr.Indikator IS NOT NULL
      AND pr.Indikator NOT LIKE '%avtalt%'
    GROUP BY pr.Indikator
""")

print(f"Tidligbehandling in {SNAPSHOT_MONTH.strftime('%Y-%m')}:")
tidligbehandling.orderBy("Indikator").show(truncate=False)


# =============================================================================
# CELL 5 — Combine and build snapshot rows
# =============================================================================

snapshot = (
    open_cases.select("Indikator").distinct()
    .join(state_summary,    "Indikator", "left")
    .join(tidligbehandling, "Indikator", "left")
    .fillna(0, subset=["open_total", "open_workable", "open_paused",
                       "open_mini_hearing", "tidligbehandling_count"])
    .withColumn("snapshot_month", F.lit(snapshot_str).cast("date"))
    .withColumn("computed_at",    F.lit(datetime.now()).cast("timestamp"))
    .withColumn("batch_id",       F.lit(BATCH_ID))
    .select(
        "snapshot_month",
        F.col("Indikator").alias("indikator"),
        F.col("open_total")             .cast("int"),
        F.col("open_workable")          .cast("int"),
        F.col("open_paused")            .cast("int"),
        F.col("open_mini_hearing")      .cast("int"),
        F.col("tidligbehandling_count") .cast("int"),
        "computed_at",
        "batch_id",
    )
)

print(f"Snapshot rows to write: {snapshot.count()}")
snapshot.orderBy("indikator").show(truncate=False)


# =============================================================================
# CELL 6 — Write snapshot (append-only, idempotent)
# =============================================================================
# Delete any existing rows for this snapshot month before appending.
# Safe to re-run — result is always one row per indicator per month.

spark.sql(f"""
    DELETE FROM {OUTPUT_TABLE}
    WHERE snapshot_month = '{snapshot_str}'
""")

snapshot.write.mode("append").saveAsTable(OUTPUT_TABLE)

written = spark.sql(f"""
    SELECT snapshot_month, COUNT(*) AS rader
    FROM {OUTPUT_TABLE}
    WHERE batch_id = '{BATCH_ID}'
    GROUP BY snapshot_month
""")
print("Written:")
written.show()
print(f"Total rows in {OUTPUT_TABLE}: "
      f"{spark.table(OUTPUT_TABLE).count():,}")


# =============================================================================
# CELL 7 — Power BI visual guidance and DAX measures
# =============================================================================
#
# OUTPUT TABLE → VISUALS
#
# portefolje_snapshot — one row per indicator per month, append-only.
# Connect directly to Power BI. Add snapshot_month to a date slicer.
# Filter to MAX(snapshot_month) for the current board report page.
#
# STACKED BAR — Portfolio composition over time (primary operational visual)
#   X axis:  snapshot_month
#   Stacks:  open_workable (green), open_paused (amber), open_mini_hearing (blue)
#   Shows whether the balance of the backlog is shifting over time.
#   A growing paused share is a delivery risk — cases the agency cannot
#   progress until applicant responds.
#
#   DAX — share measures:
#     Andel arbeidbar =
#         DIVIDE(SUM(portefolje_snapshot[open_workable]),
#                SUM(portefolje_snapshot[open_total]))
#
#     Andel pauset =
#         DIVIDE(SUM(portefolje_snapshot[open_paused]),
#                SUM(portefolje_snapshot[open_total]))
#
# CARD — Mini-hearing count per indicator
#   Single value card filtered to MAX(snapshot_month).
#   Apply conditional formatting: red if open_mini_hearing > threshold.
#   Store threshold in a small config table in the Lakehouse —
#   analyst edits it without touching this script.
#
# LINE CHART — Tidligbehandling trend
#   X axis:  snapshot_month
#   Y axis:  tidligbehandling_count
#   Shows whether early processing capacity is being deployed consistently.
#   Combine with cases received (from Prosesser directly via DAX) to show
#   early processing share:
#
#   DAX — share of incoming cases getting early processing:
#     Andel tidligbehandling =
#         DIVIDE(
#             SUM(portefolje_snapshot[tidligbehandling_count]),
#             CALCULATE(
#                 COUNTROWS(Prosesser),
#                 NOT ISBLANK(Prosesser[tidligste_startmilepael_dato]),
#                 MONTH(Prosesser[tidligste_startmilepael_dato])
#                     = MONTH(MAX(portefolje_snapshot[snapshot_month])),
#                 YEAR(Prosesser[tidligste_startmilepael_dato])
#                     = YEAR(MAX(portefolje_snapshot[snapshot_month]))
#             )
#         )
#
# NOTE: All RAG thresholds belong in Power BI conditional formatting or
# a config dimension table — not in this script. Thresholds change
# without touching code.
