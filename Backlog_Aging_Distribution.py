# =============================================================================
# Backlog aging TREND snapshot per indicator and team.
# Runs nightly after main data pipeline.
#
# Purpose:
#   "Is the existing backlog aging over time" is a trend question, and a live
#   DAX measure only ever knows what's open TODAY (TODAY() has no memory of
#   what was open last month). This script's only job is writing today's
#   age-bucket shape down each night so Power BI can chart the trend.
#
#   TODAY's shape doesn't need this table at all — that's computed live in
#   DAX directly against saksbehandling.faser (calculated columns, no
#   nightly run needed). See Backlog_Aging_Distribution_POWERBI_DAX.md for
#   both the live "today" measures and how to consume this trend table.
#
# Output table: sak_alder_fordeling
#   One row per indikator x enhet x aldersgruppe x snapshot_dato.
#
# Implementation note: pure Spark SQL, no pandas/numpy. Age bucketing is a
# SQL CASE expression and percentiles use percentile_approx — both native to
# Spark SQL, so there's no need to pull rows down locally at all. AGE_BUCKETS
# and bucket_age() below stay in Python only as the tested single source of
# truth that aldersgruppe_case_sql() generates the CASE expression from, so
# the two can never drift apart.
#
# Schedule: nightly after main data pipeline.
# =============================================================================

from pyspark.sql import SparkSession
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")

# Ordered (min_days, max_days, label) age buckets. max_days=None means
# unbounded (open-ended top bucket). Must stay contiguous — no gaps.
AGE_BUCKETS = [
    (0,   30,   "0-30"),
    (31,  60,   "31-60"),
    (61,  90,   "61-90"),
    (91,  180,  "91-180"),
    (181, None, "180+"),
]


def bucket_age(age_days, buckets=AGE_BUCKETS):
    """
    Map an age in days to its bucket label using an ordered list of
    (min_days, max_days, label) tuples; max_days=None means unbounded
    (open-ended top bucket). Returns None for negative or missing age_days,
    or if no bucket matches (should not happen given contiguous AGE_BUCKETS).

    Not used at runtime (aldersgruppe_case_sql() below builds the equivalent
    SQL CASE expression instead) — kept as the tested spec the two must
    agree on.
    """
    if age_days is None or age_days < 0:
        return None
    for lo, hi, label in buckets:
        if age_days < lo:
            continue
        if hi is None or age_days <= hi:
            return label
    return None


def aldersgruppe_case_sql(column, buckets=AGE_BUCKETS):
    """Build the SQL CASE expression equivalent to bucket_age() above."""
    lines = ["CASE"]
    for lo, hi, label in buckets:
        if hi is None:
            lines.append(f"        WHEN {column} >= {lo} THEN '{label}'")
        else:
            lines.append(f"        WHEN {column} BETWEEN {lo} AND {hi} THEN '{label}'")
    lines.append("    END")
    return "\n".join(lines)


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.sak_alder_fordeling (
    indikator            STRING      NOT NULL,
    enhet                STRING      NOT NULL,
    aldersgruppe         STRING      NOT NULL,
    snapshot_dato        DATE        NOT NULL,
    antall_saker         INT         NOT NULL,
    median_alder_dager   DOUBLE,
    p90_alder_dager      DOUBLE,
    kjoert_tidspunkt     TIMESTAMP   NOT NULL,
    kjoere_id            STRING      NOT NULL
)
USING DELTA
COMMENT 'Aldersfordeling for åpne saker per indikator/enhet/aldersgruppe. Append-modus, idempotent per snapshot_dato — filtrer på MAX(snapshot_dato) for dagens bilde, eller bruk hele tabellen for trend.'
""")

print("sak_alder_fordeling-tabellen er klar")


# =============================================================================
# CELL 2 — Snapshot today's open-case age distribution
# =============================================================================

snapshot_dato = date.today()

spark.sql(f"""
    DELETE FROM analyser.sak_alder_fordeling
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
""")

spark.sql(f"""
    INSERT INTO analyser.sak_alder_fordeling
    WITH open_cases AS (
        SELECT
            pr.indikator,
            COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
            DATEDIFF(CURRENT_DATE(), CAST(pr.startmilepaeldato AS DATE)) AS alder_dager
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.startmilepaeldato IS NOT NULL
          AND pr.sluttmilepaeldato IS NULL
          AND pr.indikator NOT LIKE '%avtalt%'
          AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
    ),
    bucketed AS (
        SELECT *, {aldersgruppe_case_sql("alder_dager")} AS aldersgruppe
        FROM open_cases
        WHERE alder_dager >= 0
    )
    SELECT
        indikator,
        enhet,
        aldersgruppe,
        DATE('{snapshot_dato.isoformat()}') AS snapshot_dato,
        COUNT(*) AS antall_saker,
        percentile_approx(alder_dager, 0.5, 1000) AS median_alder_dager,
        percentile_approx(alder_dager, 0.9, 1000) AS p90_alder_dager,
        current_timestamp() AS kjoert_tidspunkt,
        '{BATCH_ID}' AS kjoere_id
    FROM bucketed
    GROUP BY indikator, enhet, aldersgruppe
""")

print(f"sak_alder_fordeling oppdatert for {snapshot_dato}")


# =============================================================================
# CELL 3 — Verification
# =============================================================================

spark.sql(f"""
    SELECT indikator, enhet, aldersgruppe, antall_saker, median_alder_dager, p90_alder_dager
    FROM analyser.sak_alder_fordeling
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
    ORDER BY indikator, enhet,
        CASE aldersgruppe
            WHEN '0-30' THEN 1 WHEN '31-60' THEN 2 WHEN '61-90' THEN 3
            WHEN '91-180' THEN 4 ELSE 5
        END
""").show(100, truncate=False)

print("\n=== ELDSTE ALDERSGRUPPE PER INDIKATOR/ENHET, SISTE 2 SNAPSHOT ===")
spark.sql("""
    SELECT indikator, enhet, snapshot_dato,
           SUM(CASE WHEN aldersgruppe = '180+' THEN antall_saker ELSE 0 END) AS antall_180_pluss
    FROM analyser.sak_alder_fordeling
    GROUP BY indikator, enhet, snapshot_dato
    ORDER BY indikator, enhet, snapshot_dato DESC
""").show(50, truncate=False)


# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Backlog_Aging_Distribution_POWERBI_DAX.md
# =============================================================================
