# =============================================================================
# In-flight SLA risk monitor per indicator and team.
# Runs nightly after main data pipeline.
#
# Purpose:
#   Fristprosent (EWMA/CUSUM) only scores cases that have already closed —
#   a lagging indicator. This script scores cases that are STILL OPEN
#   against their own frist_dager, so an SLA breach wave shows up here
#   before it ever reaches the closed-case ratio.
#
# Assumption to verify against the Lakehouse schema:
#   it is unconfirmed whether frist_dager is reliably populated on rows
#   that have not yet closed (it is only proven populated on closed rows
#   elsewhere in this repo, via EWMA/CUSUM's Fristprosent calculation).
#   Rows without a frist_dager are simply excluded here, not defaulted —
#   if frist_dager turns out to be sparse on open rows, this script will
#   under-cover the open-case population and that should be investigated
#   before trusting antall_totalt as a full open-case count.
#
# Output tables:
#   sak_frist_risiko        — one row per open case-phase (pk_faser)
#   sak_frist_risiko_trend  — aggregate risikoklasse counts per indikator/enhet/dag
# Power BI/DAX guidance:
#   see Inflight_SLA_Risk_Monitor_POWERBI_DAX.md
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
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY    = pd.Timestamp(date.today())

# andel_brukt (dager_forlopt / frist_dager) thresholds for risikoklasse.
RISK_THRESHOLD_KRITISK = 0.90
RISK_THRESHOLD_RISIKO  = 0.75

# Trend-table volume gate — same pattern as MIN_TEAM_VOLUME in
# Throughput_Pressure_Monitor.py.
MIN_TEAM_VOLUME = 10


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.sak_frist_risiko (
    pk_faser           STRING      NOT NULL,
    indikator          STRING      NOT NULL,
    enhet              STRING      NOT NULL,
    fasetittel         STRING      NOT NULL,
    startmilepaeldato  DATE        NOT NULL,
    frist_dager        DOUBLE      NOT NULL,
    dager_forlopt      INT         NOT NULL,
    dager_igjen        INT         NOT NULL,
    andel_brukt        DOUBLE,
    risikoklasse       STRING      NOT NULL,
    kjoert_tidspunkt   TIMESTAMP   NOT NULL,
    kjoere_id          STRING      NOT NULL
)
USING DELTA
COMMENT 'Åpne saker (pk_faser) vurdert mot egen frist_dager. Leading indicator — motstykke til den lukkede-saker-baserte Fristprosent i EWMA/CUSUM. Full overwrite hver natt.'
""")

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

print("sak_frist_risiko og sak_frist_risiko_trend tabeller er klare")


# =============================================================================
# CELL 2 — Load open case-phases
# =============================================================================

open_cases = spark.sql("""
    SELECT
        pr.pk_faser,
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
        CAST(pr.startmilepaeldato AS DATE) AS startmilepaeldato,
        CAST(pr.frist_dager AS DOUBLE) AS frist_dager
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.startmilepaeldato IS NOT NULL
      AND pr.sluttmilepaeldato IS NULL
      AND pr.frist_dager IS NOT NULL
      AND pr.indikator NOT LIKE '%avtalt%'
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
""").toPandas()

open_cases["startmilepaeldato"] = pd.to_datetime(open_cases["startmilepaeldato"])
open_cases["frist_dager"]       = pd.to_numeric(open_cases["frist_dager"], errors="coerce")

print(f"Åpne saker med kjent frist_dager: {len(open_cases):,}")


# =============================================================================
# CELL 3 — Classify risk
# =============================================================================

def classify_risk(dager_igjen, andel_brukt, thresholds=None):
    """
    dager_igjen: frist_dager - dager_forlopt (negative = already breached)
    andel_brukt: dager_forlopt / frist_dager, or None if frist_dager <= 0
    thresholds: dict with 'kritisk' and 'risiko' andel_brukt cutoffs;
                defaults to RISK_THRESHOLD_KRITISK / RISK_THRESHOLD_RISIKO.

    Returns one of "Bruddet", "Kritisk", "Risiko", "Innenfor", or None if
    dager_igjen is None (insufficient data to classify).
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


if open_cases.empty:
    detail_df = pd.DataFrame(columns=[
        "pk_faser", "indikator", "enhet", "fasetittel", "startmilepaeldato",
        "frist_dager", "dager_forlopt", "dager_igjen", "andel_brukt", "risikoklasse",
    ])
else:
    detail_df = open_cases.copy()
    detail_df["dager_forlopt"] = (TODAY - detail_df["startmilepaeldato"]).dt.days
    detail_df["andel_brukt"] = np.where(
        detail_df["frist_dager"] > 0,
        detail_df["dager_forlopt"] / detail_df["frist_dager"],
        np.nan,
    )
    detail_df["dager_igjen"] = (detail_df["frist_dager"] - detail_df["dager_forlopt"]).round().astype("int64")
    detail_df["risikoklasse"] = detail_df.apply(
        lambda r: classify_risk(
            int(r["dager_igjen"]),
            None if pd.isna(r["andel_brukt"]) else float(r["andel_brukt"]),
        ),
        axis=1,
    )
    # Spark's explicit-schema createDataFrame below expects native
    # datetime.date objects for DateType columns, not pandas Timestamps.
    detail_df["startmilepaeldato"] = detail_df["startmilepaeldato"].dt.date

print(f"Rader klassifisert: {len(detail_df):,}")
if not detail_df.empty:
    print(detail_df["risikoklasse"].value_counts().to_string())


# =============================================================================
# CELL 4 — Aggregate trend per indikator/enhet
# =============================================================================

snapshot_dato = date.today()

if detail_df.empty:
    trend_df = pd.DataFrame(columns=[
        "indikator", "enhet", "antall_totalt", "antall_bruddet", "antall_kritisk",
        "antall_risiko", "antall_innenfor",
    ])
else:
    trend_rows = []
    for (indikator, enhet), grp in detail_df.groupby(["indikator", "enhet"], dropna=False):
        total = len(grp)
        counts = grp["risikoklasse"].value_counts()
        trend_rows.append({
            "indikator":       indikator,
            "enhet":           enhet,
            "antall_totalt":   total,
            "antall_bruddet":  int(counts.get("Bruddet", 0)),
            "antall_kritisk":  int(counts.get("Kritisk", 0)),
            "antall_risiko":   int(counts.get("Risiko", 0)),
            "antall_innenfor": int(counts.get("Innenfor", 0)),
        })
    trend_df = pd.DataFrame(trend_rows)

trend_rows_out = []
for _, row in trend_df.iterrows():
    total = row["antall_totalt"]
    trend_rows_out.append({
        "indikator":           row["indikator"],
        "enhet":               row["enhet"],
        "snapshot_dato":       snapshot_dato,
        "antall_totalt":       int(total),
        "antall_bruddet":      int(row["antall_bruddet"]),
        "antall_kritisk":      int(row["antall_kritisk"]),
        "antall_risiko":       int(row["antall_risiko"]),
        "antall_innenfor":     int(row["antall_innenfor"]),
        "andel_bruddet":       round(row["antall_bruddet"] / total, 4) if total else None,
        "andel_kritisk":       round(row["antall_kritisk"] / total, 4) if total else None,
        "andel_risiko":        round(row["antall_risiko"] / total, 4) if total else None,
        "andel_innenfor":      round(row["antall_innenfor"] / total, 4) if total else None,
        "tilstrekkelig_volum": bool(total >= MIN_TEAM_VOLUME),
        "kjoert_tidspunkt":    datetime.now(),
        "kjoere_id":           BATCH_ID,
    })

print(f"Trend-rader beregnet: {len(trend_rows_out):,}")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

DETAIL_SCHEMA = StructType([
    StructField("pk_faser",          StringType(),    False),
    StructField("indikator",         StringType(),    False),
    StructField("enhet",             StringType(),    False),
    StructField("fasetittel",        StringType(),    False),
    StructField("startmilepaeldato", DateType(),      False),
    StructField("frist_dager",       DoubleType(),    False),
    StructField("dager_forlopt",     IntegerType(),   False),
    StructField("dager_igjen",       IntegerType(),   False),
    StructField("andel_brukt",       DoubleType(),    True),
    StructField("risikoklasse",      StringType(),    False),
    StructField("kjoert_tidspunkt",  TimestampType(), False),
    StructField("kjoere_id",         StringType(),    False),
])

TREND_SCHEMA = StructType([
    StructField("indikator",           StringType(),    False),
    StructField("enhet",               StringType(),    False),
    StructField("snapshot_dato",       DateType(),      False),
    StructField("antall_totalt",       IntegerType(),   False),
    StructField("antall_bruddet",      IntegerType(),   False),
    StructField("antall_kritisk",      IntegerType(),   False),
    StructField("antall_risiko",       IntegerType(),   False),
    StructField("antall_innenfor",     IntegerType(),   False),
    StructField("andel_bruddet",       DoubleType(),    True),
    StructField("andel_kritisk",       DoubleType(),    True),
    StructField("andel_risiko",        DoubleType(),    True),
    StructField("andel_innenfor",      DoubleType(),    True),
    StructField("tilstrekkelig_volum", BooleanType(),   False),
    StructField("kjoert_tidspunkt",    TimestampType(), False),
    StructField("kjoere_id",           StringType(),    False),
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


if detail_df.empty:
    print("Ingen åpne saker å skrive til sak_frist_risiko.")
else:
    detail_spark = spark.createDataFrame(
        to_records(
            [{**row, "kjoert_tidspunkt": datetime.now(), "kjoere_id": BATCH_ID}
             for row in detail_df.to_dict("records")],
            DETAIL_SCHEMA,
        ),
        schema=DETAIL_SCHEMA,
    )
    detail_spark.write.mode("overwrite").saveAsTable("analyser.sak_frist_risiko")
    print(f"sak_frist_risiko skrevet: {len(detail_df):,} rader")

if trend_rows_out:
    spark.sql(f"DELETE FROM analyser.sak_frist_risiko_trend WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')")
    trend_spark = spark.createDataFrame(
        to_records(trend_rows_out, TREND_SCHEMA), schema=TREND_SCHEMA
    )
    trend_spark.write.mode("append").saveAsTable("analyser.sak_frist_risiko_trend")
    print(f"sak_frist_risiko_trend oppdatert for {snapshot_dato}: {len(trend_rows_out):,} rader")


# =============================================================================
# CELL 6 — Verification
# =============================================================================

spark.sql("""
    SELECT indikator, enhet, risikoklasse, COUNT(*) AS antall
    FROM analyser.sak_frist_risiko
    GROUP BY indikator, enhet, risikoklasse
    ORDER BY indikator, enhet, risikoklasse
""").show(50, truncate=False)

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
