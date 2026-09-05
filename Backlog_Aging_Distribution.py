# =============================================================================
# Backlog aging TREND snapshot per indicator and team.
# Runs nightly after main data pipeline.
#
# Two separate clocks, not one blended "age" — the whole reason this was
# rebuilt. The old version computed age as pure calendar time
# (TODAY() - startmilepaeldato), which blends client-caused delay into what
# looked like an internal-performance problem: a case open 90 days where 80
# of those days were us waiting on the client looked identical to one where
# all 90 days were on us.
#
#   Tidsbruk    — accumulated time on OUR side. This is the real internal
#                 performance signal ("is our processing keeping up").
#   Bransjetid  — accumulated time on the CLIENT's side, waiting for them to
#                 provide what's needed. Not our team's performance.
#
# Both are live, independently-accumulating totals already on the fact
# table (confirmed: they update continuously while a case is open) — so
# there's no need to determine which clock is "currently" running. A case
# can carry both a Tidsbruk figure and a Bransjetid figure at once; each is
# bucketed on its own, same age buckets, distinguished by the new `klokke`
# column ('Tidsbruk' / 'Bransjetid'). The completion-status column exists on
# the fact table but isn't used here — the two accumulators already say
# everything this page needs.
#
# Today's shape itself is live DAX against saksbehandling.faser, no table
# needed — see Backlog_Aging_Distribution_POWERBI_DAX.md (Del 1: live,
# Del 2: this table).
#
# Output table: sak_alder_fordeling
#   One row per indikator x enhet x klokke x aldersgruppe x snapshot_dato.
#
# Pure Spark SQL, no pandas/numpy — age bucketing is a SQL CASE expression,
# percentiles use percentile_approx. AGE_BUCKETS and bucket_age() stay in
# Python only as the tested spec aldersgruppe_case_sql() generates its SQL
# CASE expression from, so the two can't drift apart.
#
# Unit assumption: tidsbruk/bransjetid are assumed to already be day counts
# (same assumption every other script in this repo makes about tidsbruk,
# e.g. CUSUM_Changepoint.py's Behandlingstid). Verify against the Lakehouse
# schema before relying on this if that turns out to be wrong.
#
# Schedule: nightly after main data pipeline.
# =============================================================================

from pyspark.sql import SparkSession
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")

# Ordered (min_days, max_days, label) age buckets. max_days=None means
# unbounded (open-ended top bucket). Must stay contiguous — no gaps. Same
# buckets apply to both clocks.
AGE_BUCKETS = [
    (0,   30,   "0-30"),
    (31,  60,   "31-60"),
    (61,  90,   "61-90"),
    (91,  180,  "91-180"),
    (181, None, "180+"),
]

# Which fact-table column feeds which klokke label.
CLOCKS = {
    "Tidsbruk":   "tidsbruk",
    "Bransjetid": "bransjetid",
}


def bucket_age(age_days, buckets=AGE_BUCKETS):
    """
    Map an age in days to its bucket label using an ordered list of
    (min_days, max_days, label) tuples; max_days=None means unbounded
    (open-ended top bucket). Returns None for negative or missing age_days,
    or if no bucket matches (should not happen given contiguous AGE_BUCKETS).

    Not used at runtime (aldersgruppe_case_sql() below builds the equivalent
    SQL CASE expression instead) — kept as the tested spec the two must
    agree on. Applies identically regardless of which clock (Tidsbruk or
    Bransjetid) the age_days value came from.
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


def clock_snapshot_sql(klokke_label, column, snapshot_dato, batch_id):
    """
    Build the SELECT that buckets open cases by one clock column
    (tidsbruk or bransjetid). Both clocks share the exact same shape —
    generated once here instead of duplicated per clock.
    """
    return f"""
    SELECT
        indikator,
        enhet,
        '{klokke_label}' AS klokke,
        {aldersgruppe_case_sql(column)} AS aldersgruppe,
        DATE('{snapshot_dato.isoformat()}') AS snapshot_dato,
        COUNT(*) AS antall_saker,
        percentile_approx({column}, 0.5, 1000) AS median_alder_dager,
        percentile_approx({column}, 0.9, 1000) AS p90_alder_dager,
        current_timestamp() AS kjoert_tidspunkt,
        '{batch_id}' AS kjoere_id
    FROM open_cases
    WHERE {column} IS NOT NULL AND {column} >= 0
    GROUP BY indikator, enhet, {aldersgruppe_case_sql(column)}
    """


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.sak_alder_fordeling (
    indikator            STRING      NOT NULL,
    enhet                STRING      NOT NULL,
    klokke               STRING      NOT NULL,
    aldersgruppe         STRING      NOT NULL,
    snapshot_dato        DATE        NOT NULL,
    antall_saker         INT         NOT NULL,
    median_alder_dager   DOUBLE,
    p90_alder_dager      DOUBLE,
    kjoert_tidspunkt     TIMESTAMP   NOT NULL,
    kjoere_id            STRING      NOT NULL
)
USING DELTA
COMMENT 'Aldersfordeling for åpne saker per indikator/enhet/klokke/aldersgruppe. klokke=Tidsbruk (vårt ansvar) eller Bransjetid (venter på bransje) — samme sak kan ha en rad i hver, uavhengig av hverandre. Append-modus, idempotent per snapshot_dato — filtrer på MAX(snapshot_dato) for dagens bilde, eller bruk hele tabellen for trend.'
""")

print("sak_alder_fordeling-tabellen er klar")


# =============================================================================
# CELL 2 — Snapshot today's open-case age distribution, per clock
# =============================================================================

snapshot_dato = date.today()

spark.sql(f"""
    DELETE FROM analyser.sak_alder_fordeling
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
""")

open_cases_cte = """
    WITH open_cases AS (
        SELECT
            pr.indikator,
            COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
            CAST(pr.tidsbruk AS DOUBLE) AS tidsbruk,
            CAST(pr.bransjetid AS DOUBLE) AS bransjetid
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.startmilepaeldato IS NOT NULL
          AND pr.sluttmilepaeldato IS NULL
          AND pr.indikator NOT LIKE '%avtalt%'
          AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
    )
"""

clock_queries = [
    clock_snapshot_sql(label, column, snapshot_dato, BATCH_ID)
    for label, column in CLOCKS.items()
]

spark.sql(f"""
    INSERT INTO analyser.sak_alder_fordeling
    {open_cases_cte}
    {" UNION ALL ".join(clock_queries)}
""")

print(f"sak_alder_fordeling oppdatert for {snapshot_dato} (klokker: {', '.join(CLOCKS)})")


# =============================================================================
# CELL 3 — Verification
# =============================================================================

spark.sql(f"""
    SELECT indikator, enhet, klokke, aldersgruppe, antall_saker, median_alder_dager, p90_alder_dager
    FROM analyser.sak_alder_fordeling
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
    ORDER BY indikator, enhet, klokke,
        CASE aldersgruppe
            WHEN '0-30' THEN 1 WHEN '31-60' THEN 2 WHEN '61-90' THEN 3
            WHEN '91-180' THEN 4 ELSE 5
        END
""").show(200, truncate=False)

print("\n=== ELDSTE ALDERSGRUPPE PER INDIKATOR/ENHET/KLOKKE, SISTE 2 SNAPSHOT ===")
spark.sql("""
    SELECT indikator, enhet, klokke, snapshot_dato,
           SUM(CASE WHEN aldersgruppe = '180+' THEN antall_saker ELSE 0 END) AS antall_180_pluss
    FROM analyser.sak_alder_fordeling
    GROUP BY indikator, enhet, klokke, snapshot_dato
    ORDER BY indikator, enhet, klokke, snapshot_dato DESC
""").show(100, truncate=False)


# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Backlog_Aging_Distribution_POWERBI_DAX.md
# =============================================================================
