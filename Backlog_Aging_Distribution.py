# =============================================================================
# Backlog aging distribution per indicator and team.
# Runs nightly after main data pipeline.
#
# Purpose:
#   Throughput_Pressure_Monitor.py scores flow imbalance (received vs
#   completed), which is a different signal from whether the EXISTING
#   backlog is getting older. A team can have balanced net flow while its
#   oldest open cases keep aging — this script tracks that aging tail
#   directly, bucketed by how long each open case has been open.
#
# Output table: sak_alder_fordeling
#   One row per indikator x enhet x aldersgruppe x snapshot_dato.
#   snapshot_dato carries both roles at once: filter to MAX(snapshot_dato)
#   for "today's backlog shape", or chart the whole table for the trend.
# Power BI/DAX guidance:
#   see Backlog_Aging_Distribution_POWERBI_DAX.md
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
    TimestampType,
    DateType,
)
import pandas as pd
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY    = pd.Timestamp(date.today())

# Ordered (min_days, max_days, label) age buckets. max_days=None means
# unbounded (open-ended top bucket). Must stay contiguous — no gaps.
AGE_BUCKETS = [
    (0,   30,   "0-30"),
    (31,  60,   "31-60"),
    (61,  90,   "61-90"),
    (91,  180,  "91-180"),
    (181, None, "180+"),
]


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
# CELL 2 — Load open case-phases
# =============================================================================

open_cases = spark.sql("""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        CAST(pr.startmilepaeldato AS DATE) AS startmilepaeldato
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.startmilepaeldato IS NOT NULL
      AND pr.sluttmilepaeldato IS NULL
      AND pr.indikator NOT LIKE '%avtalt%'
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
""").toPandas()

open_cases["startmilepaeldato"] = pd.to_datetime(open_cases["startmilepaeldato"])

print(f"Åpne saker: {len(open_cases):,}")


# =============================================================================
# CELL 3 — Bucket by age
# =============================================================================

def bucket_age(age_days, buckets=AGE_BUCKETS):
    """
    Map an age in days to its bucket label using an ordered list of
    (min_days, max_days, label) tuples; max_days=None means unbounded
    (open-ended top bucket). Returns None for negative or missing age_days,
    or if no bucket matches (should not happen given contiguous AGE_BUCKETS).
    """
    if age_days is None or age_days < 0:
        return None
    for lo, hi, label in buckets:
        if age_days < lo:
            continue
        if hi is None or age_days <= hi:
            return label
    return None


if open_cases.empty:
    open_cases["alder_dager"]  = pd.Series(dtype="int64")
    open_cases["aldersgruppe"] = pd.Series(dtype="object")
else:
    open_cases["alder_dager"]  = (TODAY - open_cases["startmilepaeldato"]).dt.days
    open_cases["aldersgruppe"] = open_cases["alder_dager"].apply(bucket_age)

print(f"Rader bucketet: {len(open_cases):,}")
if not open_cases.empty:
    print(open_cases["aldersgruppe"].value_counts().to_string())


# =============================================================================
# CELL 4 — Aggregate per indikator/enhet/aldersgruppe
# =============================================================================
# Percentiles computed in pandas/numpy, not Spark SQL: aldersgruppe is a
# pandas-derived column (not something percentile_approx can group on
# without a second Spark round-trip), and the open-case backlog is small
# enough that pulling it down once and aggregating locally is simpler.

snapshot_dato = date.today()

agg_rows = []
if not open_cases.empty:
    for (indikator, enhet, aldersgruppe), grp in open_cases.groupby(
        ["indikator", "enhet", "aldersgruppe"], dropna=False
    ):
        ages = grp["alder_dager"].to_numpy()
        agg_rows.append({
            "indikator":          indikator,
            "enhet":              enhet,
            "aldersgruppe":       aldersgruppe,
            "snapshot_dato":      snapshot_dato,
            "antall_saker":       int(len(ages)),
            "median_alder_dager": round(float(np.median(ages)), 2),
            "p90_alder_dager":    round(float(np.percentile(ages, 90)), 2),
            "kjoert_tidspunkt":   datetime.now(),
            "kjoere_id":          BATCH_ID,
        })

print(f"Aggregerte rader: {len(agg_rows):,}")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

SCHEMA = StructType([
    StructField("indikator",          StringType(),    False),
    StructField("enhet",              StringType(),    False),
    StructField("aldersgruppe",       StringType(),    False),
    StructField("snapshot_dato",      DateType(),      False),
    StructField("antall_saker",       IntegerType(),   False),
    StructField("median_alder_dager", DoubleType(),    True),
    StructField("p90_alder_dager",    DoubleType(),    True),
    StructField("kjoert_tidspunkt",   TimestampType(), False),
    StructField("kjoere_id",          StringType(),    False),
])


def to_records(rows, schema):
    casters = {
        StringType():  lambda v: None if v is None else str(v),
        IntegerType(): lambda v: None if v is None else int(v),
        DoubleType():  lambda v: None if v is None else float(v),
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


if agg_rows:
    spark.sql(f"DELETE FROM analyser.sak_alder_fordeling WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')")
    agg_spark = spark.createDataFrame(to_records(agg_rows, SCHEMA), schema=SCHEMA)
    agg_spark.write.mode("append").saveAsTable("analyser.sak_alder_fordeling")
    print(f"sak_alder_fordeling oppdatert for {snapshot_dato}: {len(agg_rows):,} rader")
else:
    print("Ingen åpne saker å skrive til sak_alder_fordeling.")


# =============================================================================
# CELL 6 — Verification
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
