# =============================================================================
# Caseworker load concentration per team.
# Runs nightly after main data pipeline.
#
# Purpose:
#   Team-level workload concentration / bus-factor / burnout early warning —
#   is active caseload piling up on a few caseworkers within a team, even
#   while the team's aggregate numbers (throughput pressure monitor,
#   Fristprosent) look fine? Concentration is measured with the Gini
#   coefficient of open-caseload counts per saksbehandler within each enhet.
#
# Per-saksbehandler open caseload counts and shares (aktiv_saksmengde,
# andel_av_enhetens_saksmengde) are NOT written by this script — Faser is
# already grouped on saksbehandler live in DAX in the semantic model, so a
# nightly copy of a plain COUNTROWS/DIVIDE would just be duplicated data.
# See Caseworker_Load_Concentration_POWERBI_DAX.md, Del 1.
#
# This script's only remaining job is the Gini coefficient — the one thing
# here that genuinely can't be a DAX measure (a rank-based Lorenz-curve
# computation), and, being a snapshot of TODAY's open caseload, is also a
# trend question a live measure can't answer on its own — same reasoning
# as Backlog_Aging_Distribution.py / Inflight_SLA_Risk_Monitor.py.
#
# Individual-level automated flagging is out of scope for this layer (see
# CUSUM_Changepoint.py's explicit exclusion of saksbehandler from its
# drilldown, for that exact reason) — saksbehandler_konsentrasjon only
# ever stores enhet-level aggregates, never a per-person breakdown.
#
# Schema assumption: the saksbehandler column name below is UNVERIFIED
# against the Lakehouse schema (it is only ever mentioned in comments
# elsewhere in this repo, never confirmed as an actual column name) —
# verify before relying on this script, same caveat as
# CUSUM_Changepoint.py's DRILLDOWN_DIMENSIONS.
#
# Output table:
#   saksbehandler_konsentrasjon   — Gini trend per enhet
# Power BI/DAX guidance:
#   see Caseworker_Load_Concentration_POWERBI_DAX.md
#
# Schedule: nightly after main data pipeline.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    TimestampType,
    DateType,
)
import pandas as pd
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")

SAKSBEHANDLER_COL  = "saksbehandler"  # verify this column name against the Lakehouse schema
MIN_SAKSBEHANDLERE = 3   # Gini on 1-2 people is meaningless — gate the enhet-level metric


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.saksbehandler_konsentrasjon (
    enhet                    STRING      NOT NULL,
    snapshot_dato            DATE        NOT NULL,
    antall_saksbehandlere    INT         NOT NULL,
    total_aktive_saker       INT         NOT NULL,
    gini_koeffisient         DOUBLE,
    tilstrekkelig_volum      BOOLEAN     NOT NULL,
    kjoert_tidspunkt         TIMESTAMP   NOT NULL,
    kjoere_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'Gini-koeffisient for arbeidsmengdekonsentrasjon per enhet. Ingen individdata — kun aggregert per enhet. Append-modus, idempotent per snapshot_dato.'
""")

print("saksbehandler_konsentrasjon-tabellen er klar")


# =============================================================================
# CELL 2 — Load open caseload per enhet/saksbehandler
# =============================================================================
# Rows with blank/null saksbehandler (unassigned cases) are excluded
# entirely rather than coalesced to 'Ukjent' — an 'Ukjent' pseudo-caseworker
# bucket would corrupt the per-person concentration metric this script
# exists to measure.

caseload = spark.sql(f"""
    SELECT
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        TRIM(pr.{SAKSBEHANDLER_COL}) AS saksbehandler,
        COUNT(*) AS aktiv_saksmengde
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.startmilepaeldato IS NOT NULL
      AND pr.sluttmilepaeldato IS NULL
      AND pr.{SAKSBEHANDLER_COL} IS NOT NULL
      AND TRIM(pr.{SAKSBEHANDLER_COL}) != ''
      AND pr.indikator NOT LIKE '%avtalt%'
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
    GROUP BY COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'), TRIM(pr.{SAKSBEHANDLER_COL})
""").toPandas()

caseload["aktiv_saksmengde"] = pd.to_numeric(caseload["aktiv_saksmengde"], errors="coerce").astype("int64")

print(f"Enhet x saksbehandler-rader: {len(caseload):,}")


# =============================================================================
# CELL 3 — Gini coefficient
# =============================================================================

def gini_coefficient(values):
    """
    Population Gini coefficient for a list of non-negative caseload counts.

    Returns None when the metric is not meaningful:
      - fewer than 2 values (nothing to compare concentration against)
      - all values sum to zero (no active caseload to concentrate)

    Raises ValueError if any value is negative — caseload counts should
    never be negative (they come from COUNT(*)); a negative input signals
    an upstream data bug and this function fails loudly rather than
    returning a silently-wrong coefficient.
    """
    values = [v for v in values if v is not None]
    n = len(values)
    if n < 2:
        return None
    if any(v < 0 for v in values):
        raise ValueError("gini_coefficient: caseload values must be non-negative")
    total = sum(values)
    if total == 0:
        return None
    sorted_values = sorted(values)
    weighted_sum = sum((i + 1) * x for i, x in enumerate(sorted_values))
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


# Per-saksbehandler counts and shares (aktiv_saksmengde,
# andel_av_enhetens_saksmengde) are NOT written anywhere — both are a plain
# COUNTROWS/DIVIDE grouped by saksbehandler, live DAX against
# saksbehandling.faser (Faser[saksbehandler] is already in the semantic
# model). See Caseworker_Load_Concentration_POWERBI_DAX.md. `caseload`
# above stays in this script only as the input the Gini computation below
# needs — that's the one thing here that genuinely can't be a DAX measure.

# ── Trend: Gini per enhet ────────────────────────────────────────────────────
snapshot_dato = date.today()

trend_rows = []
if not caseload.empty:
    for enhet, grp in caseload.groupby("enhet"):
        caseloads = grp["aktiv_saksmengde"].tolist()
        n_saksbehandlere = len(caseloads)
        total_saker      = int(sum(caseloads))
        tilstrekkelig    = n_saksbehandlere >= MIN_SAKSBEHANDLERE
        gini = gini_coefficient(caseloads) if tilstrekkelig else None

        trend_rows.append({
            "enhet":                 enhet,
            "snapshot_dato":         snapshot_dato,
            "antall_saksbehandlere": n_saksbehandlere,
            "total_aktive_saker":    total_saker,
            "gini_koeffisient":      round(gini, 4) if gini is not None else None,
            "tilstrekkelig_volum":   bool(tilstrekkelig),
            "kjoert_tidspunkt":      datetime.now(),
            "kjoere_id":             BATCH_ID,
        })

print(f"Konsentrasjon-rader beregnet: {len(trend_rows):,}")


# =============================================================================
# CELL 4 — Write to Lakehouse
# =============================================================================

TREND_SCHEMA = StructType([
    StructField("enhet",                 StringType(),    False),
    StructField("snapshot_dato",         DateType(),      False),
    StructField("antall_saksbehandlere", IntegerType(),   False),
    StructField("total_aktive_saker",    IntegerType(),   False),
    StructField("gini_koeffisient",      DoubleType(),    True),
    StructField("tilstrekkelig_volum",   BooleanType(),   False),
    StructField("kjoert_tidspunkt",      TimestampType(), False),
    StructField("kjoere_id",             StringType(),    False),
])


def to_records(rows, schema):
    casters = {
        StringType():  lambda v: None if v is None else str(v),
        IntegerType(): lambda v: None if v is None else int(v),
        DoubleType():  lambda v: None if v is None else float(v),
        BooleanType(): lambda v: None if v is None else bool(v),
    }
    out = []
    for row in rows:
        values = []
        for field in schema.fields:
            v = row.get(field.name)
            if v is not None and pd.isna(v):
                v = None
            cast = casters.get(field.dataType)
            values.append(cast(v) if cast else v)
        out.append(tuple(values))
    return out


if trend_rows:
    spark.sql(f"DELETE FROM analyser.saksbehandler_konsentrasjon WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')")
    trend_spark = spark.createDataFrame(to_records(trend_rows, TREND_SCHEMA), schema=TREND_SCHEMA)
    trend_spark.write.mode("append").saveAsTable("analyser.saksbehandler_konsentrasjon")
    print(f"saksbehandler_konsentrasjon oppdatert for {snapshot_dato}: {len(trend_rows):,} rader")


# =============================================================================
# CELL 5 — Verification
# =============================================================================

# Per-saksbehandler counts/shares aren't written to a table anymore (see
# note above CELL 3) — verify against the in-memory caseload frame instead.
if not caseload.empty:
    verify = caseload.copy()
    verify["andel"] = verify.groupby("enhet")["aktiv_saksmengde"].transform(
        lambda s: s / s.sum() if s.sum() else 0.0
    )
    summary = (
        verify.groupby("enhet")
        .agg(antall_saksbehandlere=("saksbehandler", "count"),
             total_saker=("aktiv_saksmengde", "sum"),
             storste_andel_pct=("andel", "max"))
    )
    summary["storste_andel_pct"] = (summary["storste_andel_pct"] * 100).round(1)
    print(summary.sort_values("storste_andel_pct", ascending=False).to_string())

spark.sql(f"""
    SELECT enhet, antall_saksbehandlere, total_aktive_saker, gini_koeffisient
    FROM analyser.saksbehandler_konsentrasjon
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
      AND tilstrekkelig_volum = TRUE
    ORDER BY gini_koeffisient DESC
""").show(50, truncate=False)


# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Caseworker_Load_Concentration_POWERBI_DAX.md
# =============================================================================
