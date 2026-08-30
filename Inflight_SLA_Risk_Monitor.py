# =============================================================================
# In-flight SLA risk TREND snapshot per indicator and team.
# Runs nightly after main data pipeline.
#
# Fristprosent (CUSUM) only scores cases that have already closed. This
# scores cases that are STILL OPEN against their own frist_dager. Risikoklasse
# depends on TODAY() — this script's only job is writing that day's risk mix
# down so Power BI can chart the trend before a breach wave reaches the
# closed-case ratio. Today's per-case list is live DAX, no table needed —
# see Inflight_SLA_Risk_Monitor_POWERBI_DAX.md (Del 1: live, Del 2: this table).
#
# Assumption to verify against the Lakehouse schema:
#   it is unconfirmed whether frist_dager is reliably populated on rows
#   that have not yet closed (it is only proven populated on closed rows
#   elsewhere in this repo, via CUSUM's Fristprosent calculation). Rows
#   without a frist_dager are simply excluded here, not defaulted — if
#   frist_dager turns out to be sparse on open rows, this under-covers the
#   open-case population and that should be investigated before trusting
#   antall_totalt as a full open-case count.
#
# Output table: sak_frist_risiko_trend
#   One row per indikator x enhet x snapshot_dato.
#
# Implementation note: pure Spark SQL, no pandas/numpy. classify_risk() and
# the threshold constants below stay in Python only as the tested single
# source of truth that risikoklasse_case_sql() generates the equivalent SQL
# CASE expression from, so the two can never drift apart.
#
# Schedule: nightly after main data pipeline.
# =============================================================================

from pyspark.sql import SparkSession
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")

# andel_brukt (dager_forlopt / frist_dager) thresholds for risikoklasse.
RISK_THRESHOLD_KRITISK = 0.90
RISK_THRESHOLD_RISIKO  = 0.75

# Trend-table volume gate — same pattern as Throughput_Pressure_Monitor.py's
# MIN_TEAM_VOLUME.
MIN_TEAM_VOLUME = 10


def classify_risk(dager_igjen, andel_brukt, thresholds=None):
    """
    dager_igjen: frist_dager - dager_forlopt (negative = already breached)
    andel_brukt: dager_forlopt / frist_dager, or None if frist_dager <= 0
    thresholds: dict with 'kritisk' and 'risiko' andel_brukt cutoffs;
                defaults to RISK_THRESHOLD_KRITISK / RISK_THRESHOLD_RISIKO.

    Returns one of "Bruddet", "Kritisk", "Risiko", "Innenfor", or None if
    dager_igjen is None (insufficient data to classify).

    Not used at runtime (risikoklasse_case_sql() below builds the
    equivalent SQL CASE expression instead) — kept as the tested spec the
    two must agree on.
    """
    if thresholds is None:
        thresholds = {"kritisk": RISK_THRESHOLD_KRITISK, "risiko": RISK_THRESHOLD_RISIKO}
    if dager_igjen is None:
        return None
    if dager_igjen < 0:
        return "Bruddet"
    if andel_brukt is not None and andel_brukt >= thresholds["kritisk"]:
        return "Kritisk"
    if andel_brukt is not None and andel_brukt >= thresholds["risiko"]:
        return "Risiko"
    return "Innenfor"


def risikoklasse_case_sql(dager_igjen_col, andel_brukt_col, thresholds=None):
    """Build the SQL CASE expression equivalent to classify_risk() above."""
    if thresholds is None:
        thresholds = {"kritisk": RISK_THRESHOLD_KRITISK, "risiko": RISK_THRESHOLD_RISIKO}
    return f"""CASE
        WHEN {dager_igjen_col} < 0 THEN 'Bruddet'
        WHEN {andel_brukt_col} IS NOT NULL AND {andel_brukt_col} >= {thresholds['kritisk']} THEN 'Kritisk'
        WHEN {andel_brukt_col} IS NOT NULL AND {andel_brukt_col} >= {thresholds['risiko']} THEN 'Risiko'
        ELSE 'Innenfor'
    END"""


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.sak_frist_risiko_trend (
    indikator            STRING      NOT NULL,
    enhet                STRING      NOT NULL,
    snapshot_dato        DATE        NOT NULL,
    antall_totalt        INT         NOT NULL,
    antall_bruddet       INT         NOT NULL,
    antall_kritisk       INT         NOT NULL,
    antall_risiko        INT         NOT NULL,
    antall_innenfor      INT         NOT NULL,
    andel_bruddet        DOUBLE,
    andel_kritisk        DOUBLE,
    andel_risiko         DOUBLE,
    andel_innenfor       DOUBLE,
    tilstrekkelig_volum  BOOLEAN     NOT NULL,
    kjoert_tidspunkt     TIMESTAMP   NOT NULL,
    kjoere_id            STRING      NOT NULL
)
USING DELTA
COMMENT 'Daglig øyeblikksbilde av risikoklasse-fordeling for åpne saker per indikator/enhet. Append-modus, idempotent per snapshot_dato.'
""")

print("sak_frist_risiko_trend-tabellen er klar")


# =============================================================================
# CELL 2 — Snapshot today's risk-class mix
# =============================================================================

snapshot_dato = date.today()
RISIKOKLASSE_SQL = risikoklasse_case_sql("dager_igjen", "andel_brukt")

spark.sql(f"""
    DELETE FROM analyser.sak_frist_risiko_trend
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
""")

spark.sql(f"""
    INSERT INTO analyser.sak_frist_risiko_trend
    WITH open_cases AS (
        SELECT
            pr.indikator,
            COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
            DATEDIFF(CURRENT_DATE(), CAST(pr.startmilepaeldato AS DATE)) AS dager_forlopt,
            CAST(pr.frist_dager AS DOUBLE) AS frist_dager
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.startmilepaeldato IS NOT NULL
          AND pr.sluttmilepaeldato IS NULL
          AND pr.frist_dager IS NOT NULL
          AND pr.indikator NOT LIKE '%avtalt%'
          AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
    ),
    classified AS (
        SELECT
            *,
            ROUND(frist_dager - dager_forlopt) AS dager_igjen,
            CASE WHEN frist_dager > 0 THEN dager_forlopt / frist_dager END AS andel_brukt
        FROM open_cases
    ),
    risiko AS (
        SELECT *, {RISIKOKLASSE_SQL} AS risikoklasse
        FROM classified
    ),
    per_gruppe AS (
        SELECT
            indikator,
            enhet,
            COUNT(*) AS antall_totalt,
            COUNT(CASE WHEN risikoklasse = 'Bruddet'  THEN 1 END) AS antall_bruddet,
            COUNT(CASE WHEN risikoklasse = 'Kritisk'  THEN 1 END) AS antall_kritisk,
            COUNT(CASE WHEN risikoklasse = 'Risiko'   THEN 1 END) AS antall_risiko,
            COUNT(CASE WHEN risikoklasse = 'Innenfor' THEN 1 END) AS antall_innenfor
        FROM risiko
        GROUP BY indikator, enhet
    )
    SELECT
        indikator,
        enhet,
        DATE('{snapshot_dato.isoformat()}') AS snapshot_dato,
        antall_totalt,
        antall_bruddet,
        antall_kritisk,
        antall_risiko,
        antall_innenfor,
        ROUND(antall_bruddet  / antall_totalt, 4) AS andel_bruddet,
        ROUND(antall_kritisk  / antall_totalt, 4) AS andel_kritisk,
        ROUND(antall_risiko   / antall_totalt, 4) AS andel_risiko,
        ROUND(antall_innenfor / antall_totalt, 4) AS andel_innenfor,
        (antall_totalt >= {MIN_TEAM_VOLUME}) AS tilstrekkelig_volum,
        current_timestamp() AS kjoert_tidspunkt,
        '{BATCH_ID}' AS kjoere_id
    FROM per_gruppe
""")

print(f"sak_frist_risiko_trend oppdatert for {snapshot_dato}")


# =============================================================================
# CELL 3 — Verification
# =============================================================================

spark.sql(f"""
    SELECT indikator, enhet, antall_totalt, andel_bruddet, andel_kritisk, andel_risiko
    FROM analyser.sak_frist_risiko_trend
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
      AND tilstrekkelig_volum = TRUE
    ORDER BY andel_bruddet DESC, andel_kritisk DESC
""").show(50, truncate=False)


# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Inflight_SLA_Risk_Monitor_POWERBI_DAX.md
# =============================================================================
