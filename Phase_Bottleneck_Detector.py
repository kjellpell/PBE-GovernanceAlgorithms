# =============================================================================
# Phase bottleneck detector per team, phase and indicator.
# Runs nightly after main data pipeline.
#
# Purpose:
#   Detect where queue pressure is forming inside process phases by combining
#   net flow and deterioration in p90 phase time versus recent baseline.
#
# Output table:
#   fase_flaskehals_enhet — enhet x fasetittel x indikator x month
# Power BI/DAX guidance:
#   see Phase_Bottleneck_Detector_POWERBI_DAX.md
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
from datetime import datetime

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
START_YEAR = 2015
BASELINE_MONTHS = 12
MIN_BASELINE_OBS = 6
MIN_SEGMENT_OBS = 10


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.fase_flaskehals_enhet (
    indikator              STRING      NOT NULL,
    enhet                  STRING      NOT NULL,
    fasetittel             STRING      NOT NULL,
    analyse_dato           DATE        NOT NULL,
    inngang_antall         INT         NOT NULL,
    utgang_antall          INT         NOT NULL,
    fase_ko_proxy          INT         NOT NULL,
    median_fasetid         DOUBLE,
    p90_fasetid            DOUBLE,
    baseline_median        DOUBLE,
    baseline_p90           DOUBLE,
    avvik_median_pct       DOUBLE,
    avvik_p90_pct          DOUBLE,
    tilstrekkelig_volum    BOOLEAN     NOT NULL,
    bottleneck_flagg       BOOLEAN     NOT NULL,
    alvorlighet            STRING      NOT NULL,
    arsak_kode             STRING,
    arsak_tekst            STRING,
    kjoert_tidspunkt       TIMESTAMP   NOT NULL,
    kjoere_id              STRING      NOT NULL
)
USING DELTA
COMMENT 'Flaskehalsdeteksjon per team/fase/indikator. Kombinerer fase-nettoflyt, køproxy og p90-avvik mot historisk baseline.'
""")

print("fase_flaskehals_enhet-tabellen er klar")


# =============================================================================
# CELL 2 — Load monthly phase aggregates
# =============================================================================

phase = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
        (YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) * 100
            + MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))) AS period,
        COUNT(CASE WHEN pr.startmilepaeldato IS NOT NULL THEN 1 END) AS inngang_antall,
        COUNT(CASE WHEN pr.sluttmilepaeldato IS NOT NULL THEN 1 END) AS utgang_antall,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.5, 1000) AS median_fasetid,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.9, 1000) AS p90_fasetid
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
      AND pr.indikator NOT LIKE '%avtalt%'
      AND YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) >= {START_YEAR}
    GROUP BY pr.indikator,
             COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase'),
             YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
             MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
""").toPandas()

if phase.empty:
    print("Ingen fase-data tilgjengelig.")

phase["period"] = pd.to_numeric(phase["period"], errors="coerce").astype("int64")
phase["inngang_antall"] = pd.to_numeric(phase["inngang_antall"], errors="coerce").fillna(0).astype("int32")
phase["utgang_antall"] = pd.to_numeric(phase["utgang_antall"], errors="coerce").fillna(0).astype("int32")
phase["median_fasetid"] = pd.to_numeric(phase["median_fasetid"], errors="coerce").astype("float64")
phase["p90_fasetid"] = pd.to_numeric(phase["p90_fasetid"], errors="coerce").astype("float64")

print(f"Fase-rader lastet: {len(phase):,}")


# =============================================================================
# CELL 3 — Compute bottlenecks
# =============================================================================


def _month_end_from_period(period: int):
    year = int(period) // 100
    month = int(period) % 100
    return (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()


rows = []
for (indikator, enhet, fasetittel), grp in phase.groupby(["indikator", "enhet", "fasetittel"], dropna=False):
    g = grp.sort_values("period").copy()
    g["netto_flyt"] = g["inngang_antall"] - g["utgang_antall"]
    g["fase_ko_proxy"] = g["netto_flyt"].cumsum().clip(lower=0)

    g["baseline_median"] = (
        g["median_fasetid"].shift(1)
        .rolling(BASELINE_MONTHS, min_periods=MIN_BASELINE_OBS)
        .median()
    )
    g["baseline_p90"] = (
        g["p90_fasetid"].shift(1)
        .rolling(BASELINE_MONTHS, min_periods=MIN_BASELINE_OBS)
        .median()
    )

    g["avvik_median_pct"] = np.where(
        g["baseline_median"].notna() & (g["baseline_median"] > 0),
        (g["median_fasetid"] - g["baseline_median"]) / g["baseline_median"],
        np.nan,
    )
    g["avvik_p90_pct"] = np.where(
        g["baseline_p90"].notna() & (g["baseline_p90"] > 0),
        (g["p90_fasetid"] - g["baseline_p90"]) / g["baseline_p90"],
        np.nan,
    )

    g["tilstrekkelig_volum"] = (
        (g["inngang_antall"] >= MIN_SEGMENT_OBS)
        | (g["utgang_antall"] >= MIN_SEGMENT_OBS)
    )

    for _, row in g.iterrows():
        score = 0.0
        reasons = []

        if row["netto_flyt"] > 0:
            score += 1.5
            reasons.append("NETTO_FLYT_POS")

        if pd.notna(row["avvik_p90_pct"]) and row["avvik_p90_pct"] > 0.15:
            score += 1.5
            reasons.append("P90_OPP")

        if pd.notna(row["avvik_p90_pct"]) and row["avvik_p90_pct"] > 0.30:
            score += 1.0
            reasons.append("P90_OPP_STERK")

        if row["inngang_antall"] > 0 and row["utgang_antall"] <= (row["inngang_antall"] * 0.8):
            score += 1.0
            reasons.append("UTGANGSGAP")

        if row["fase_ko_proxy"] > 0 and row["netto_flyt"] > 0:
            score += 1.0
            reasons.append("KO_VEKST")

        if score >= 4.0:
            severity = "Kritisk"
        elif score >= 3.0:
            severity = "Hoy"
        elif score >= 1.5:
            severity = "Moderat"
        else:
            severity = "Lav"

        bottleneck = bool(row["tilstrekkelig_volum"] and score >= 3.0)
        reason_code = "+".join(reasons) if reasons else "INGEN"

        reason_text = ""
        if reason_code == "INGEN":
            reason_text = "Ingen tydelig faseflaskehals i perioden."
        elif "KO_VEKST" in reason_code and "P90_OPP" in reason_code:
            reason_text = "Kø vokser samtidig som fasevariasjon i p90 forverres."
        elif "NETTO_FLYT_POS" in reason_code:
            reason_text = "Flere saker går inn i fasen enn ut i perioden."
        else:
            reason_text = "Fasesignal indikerer tregere flyt enn historisk baseline."

        rows.append({
            "indikator": indikator,
            "enhet": enhet,
            "fasetittel": fasetittel,
            "analyse_dato": _month_end_from_period(int(row["period"])),
            "inngang_antall": int(row["inngang_antall"]),
            "utgang_antall": int(row["utgang_antall"]),
            "fase_ko_proxy": int(row["fase_ko_proxy"]),
            "median_fasetid": float(row["median_fasetid"]) if pd.notna(row["median_fasetid"]) else None,
            "p90_fasetid": float(row["p90_fasetid"]) if pd.notna(row["p90_fasetid"]) else None,
            "baseline_median": float(row["baseline_median"]) if pd.notna(row["baseline_median"]) else None,
            "baseline_p90": float(row["baseline_p90"]) if pd.notna(row["baseline_p90"]) else None,
            "avvik_median_pct": float(row["avvik_median_pct"]) if pd.notna(row["avvik_median_pct"]) else None,
            "avvik_p90_pct": float(row["avvik_p90_pct"]) if pd.notna(row["avvik_p90_pct"]) else None,
            "tilstrekkelig_volum": bool(row["tilstrekkelig_volum"]),
            "bottleneck_flagg": bottleneck,
            "alvorlighet": severity,
            "arsak_kode": reason_code,
            "arsak_tekst": reason_text,
            "kjoert_tidspunkt": datetime.now(),
            "kjoere_id": BATCH_ID,
        })

out_df = pd.DataFrame(rows)
print(f"Flaskehalsrader beregnet: {len(out_df):,}")


# =============================================================================
# CELL 4 — Write to Lakehouse
# =============================================================================

SCHEMA = StructType([
    StructField("indikator", StringType(), False),
    StructField("enhet", StringType(), False),
    StructField("fasetittel", StringType(), False),
    StructField("analyse_dato", DateType(), False),
    StructField("inngang_antall", IntegerType(), False),
    StructField("utgang_antall", IntegerType(), False),
    StructField("fase_ko_proxy", IntegerType(), False),
    StructField("median_fasetid", DoubleType(), True),
    StructField("p90_fasetid", DoubleType(), True),
    StructField("baseline_median", DoubleType(), True),
    StructField("baseline_p90", DoubleType(), True),
    StructField("avvik_median_pct", DoubleType(), True),
    StructField("avvik_p90_pct", DoubleType(), True),
    StructField("tilstrekkelig_volum", BooleanType(), False),
    StructField("bottleneck_flagg", BooleanType(), False),
    StructField("alvorlighet", StringType(), False),
    StructField("arsak_kode", StringType(), True),
    StructField("arsak_tekst", StringType(), True),
    StructField("kjoert_tidspunkt", TimestampType(), False),
    StructField("kjoere_id", StringType(), False),
])


def to_records(rows_in, schema):
    casters = {
        StringType(): lambda v: None if v is None else str(v),
        IntegerType(): lambda v: None if v is None else int(v),
        DoubleType(): lambda v: None if v is None else float(v),
        BooleanType(): lambda v: None if v is None else bool(v),
        TimestampType(): lambda v: None if v is None else (
            v.to_pydatetime() if isinstance(v, pd.Timestamp) else v
        ),
        DateType(): lambda v: None if v is None else (
            v.to_pydatetime().date() if isinstance(v, pd.Timestamp) else v
        ),
    }
    out = []
    for row in rows_in:
        values = []
        for field in schema.fields:
            v = row.get(field.name)
            if v is not None and pd.isna(v):
                v = None
            cast = casters.get(field.dataType)
            values.append(cast(v) if cast else v)
        out.append(tuple(values))
    return out

if out_df.empty:
    print("Ingen flaskehalsresultater å skrive.")
else:
    out_spark = spark.createDataFrame(
        to_records(out_df.to_dict("records"), SCHEMA), schema=SCHEMA
    )
    out_spark.write.mode("overwrite").saveAsTable("analyser.fase_flaskehals_enhet")
    print(f"fase_flaskehals_enhet skrevet: {len(out_df):,} rader")

    spark.sql(f"""
        SELECT
            indikator,
            enhet,
            fasetittel,
            analyse_dato,
            inngang_antall,
            utgang_antall,
            ROUND(p90_fasetid, 1) AS p90_fasetid,
            ROUND(avvik_p90_pct * 100, 1) AS avvik_p90_pct,
            alvorlighet,
            arsak_kode
                FROM analyser.fase_flaskehals_enhet
        WHERE kjoere_id = '{BATCH_ID}'
          AND analyse_dato = (
              SELECT MAX(analyse_dato)
                            FROM analyser.fase_flaskehals_enhet
              WHERE kjoere_id = '{BATCH_ID}'
          )
          AND bottleneck_flagg = TRUE
        ORDER BY avvik_p90_pct DESC, inngang_antall DESC
    """).show(50, truncate=False)
