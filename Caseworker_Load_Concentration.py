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
# DEPARTURE FROM REPO CONVENTION: unlike every other script here, this one
# DOES persist a per-saksbehandler (individual caseworker) detail table.
# This is a deliberate exception, made for internal capacity-planning /
# workload-balancing use by team managers — NOT for automated escalation
# or individual performance flagging (see CUSUM_Changepoint.py's explicit
# exclusion of saksbehandler from its drilldown, for that exact reason).
# Do not repurpose saksbehandler_arbeidsmengde for automated per-person
# alerting.
#
# Data retention: saksbehandler_arbeidsmengde is FULL OVERWRITE every night
# and deliberately keeps NO history at the individual grain, to limit
# personal-data retention. Only saksbehandler_konsentrasjon (enhet-level,
# no individual data) accumulates history.
#
# Schema assumption: the saksbehandler column name below is UNVERIFIED
# against the Lakehouse schema (it is only ever mentioned in comments
# elsewhere in this repo, never confirmed as an actual column name) —
# verify before relying on this script, same caveat as
# CUSUM_Changepoint.py's DRILLDOWN_DIMENSIONS.
#
# Output tables:
#   saksbehandler_arbeidsmengde   — current open caseload per saksbehandler
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
CREATE TABLE IF NOT EXISTS analyser.saksbehandler_arbeidsmengde (
    enhet                          STRING      NOT NULL,
    saksbehandler                  STRING      NOT NULL,
    aktiv_saksmengde               INT         NOT NULL,
    andel_av_enhetens_saksmengde   DOUBLE      NOT NULL,
    kjoert_tidspunkt               TIMESTAMP   NOT NULL,
    kjoere_id                      STRING      NOT NULL
)
USING DELTA
COMMENT 'Åpen saksmengde per saksbehandler/enhet. Kun for intern kapasitetsplanlegging hos ledere — ikke for automatisk individvarsling. Full overwrite hver natt, ingen historikk lagres på individnivå.'
""")

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

print("saksbehandler_arbeidsmengde og saksbehandler_konsentrasjon tabeller er klare")


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


# ── Detail: current open caseload per saksbehandler ─────────────────────────
detail_rows = []
if not caseload.empty:
    for enhet, grp in caseload.groupby("enhet"):
        enhet_total = int(grp["aktiv_saksmengde"].sum())
        for _, row in grp.iterrows():
            detail_rows.append({
                "enhet":                        enhet,
                "saksbehandler":                row["saksbehandler"],
                "aktiv_saksmengde":             int(row["aktiv_saksmengde"]),
                "andel_av_enhetens_saksmengde": round(row["aktiv_saksmengde"] / enhet_total, 4)
                                                if enhet_total else 0.0,
                "kjoert_tidspunkt":             datetime.now(),
                "kjoere_id":                    BATCH_ID,
            })

print(f"Arbeidsmengde-rader beregnet: {len(detail_rows):,}")


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

DETAIL_SCHEMA = StructType([
    StructField("enhet",                        StringType(),    False),
    StructField("saksbehandler",                StringType(),    False),
    StructField("aktiv_saksmengde",             IntegerType(),   False),
    StructField("andel_av_enhetens_saksmengde", DoubleType(),    False),
    StructField("kjoert_tidspunkt",              TimestampType(), False),
    StructField("kjoere_id",                     StringType(),    False),
])

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


if detail_rows:
    detail_spark = spark.createDataFrame(to_records(detail_rows, DETAIL_SCHEMA), schema=DETAIL_SCHEMA)
    detail_spark.write.mode("overwrite").saveAsTable("analyser.saksbehandler_arbeidsmengde")
    print(f"saksbehandler_arbeidsmengde skrevet: {len(detail_rows):,} rader")
else:
    print("Ingen data å skrive til saksbehandler_arbeidsmengde.")

if trend_rows:
    spark.sql(f"DELETE FROM analyser.saksbehandler_konsentrasjon WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')")
    trend_spark = spark.createDataFrame(to_records(trend_rows, TREND_SCHEMA), schema=TREND_SCHEMA)
    trend_spark.write.mode("append").saveAsTable("analyser.saksbehandler_konsentrasjon")
    print(f"saksbehandler_konsentrasjon oppdatert for {snapshot_dato}: {len(trend_rows):,} rader")


# =============================================================================
# CELL 5 — Verification
# =============================================================================

spark.sql("""
    SELECT enhet, COUNT(*) AS antall_saksbehandlere, SUM(aktiv_saksmengde) AS total_saker,
           ROUND(MAX(andel_av_enhetens_saksmengde) * 100, 1) AS storste_andel_pct
    FROM analyser.saksbehandler_arbeidsmengde
    GROUP BY enhet
    ORDER BY storste_andel_pct DESC
""").show(50, truncate=False)

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
