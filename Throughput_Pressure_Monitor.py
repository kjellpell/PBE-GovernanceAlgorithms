# =============================================================================
# Throughput pressure monitor per team and indicator.
# Runs nightly after main data pipeline.
#
# Purpose:
#   Detect where intake exceeds completions for multiple periods and where
#   processing time deteriorates versus recent baseline.
#
# Output tables:
#   gjennomstoremming_press_enhet  — pressignal per enhet x indikator x month
#   gjennomstroemming_press_fase  — stottetabell per fasetittel
# Power BI/DAX guidance:
#   see Throughput_Pressure_Monitor_POWERBI_DAX.md
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
from datetime import datetime

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
START_YEAR = 2015
BASELINE_MONTHS = 12
MIN_BASELINE_OBS = 6
MIN_TEAM_VOLUME = 10
POSITIVE_FLOW_STREAK = 3


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.gjennomstoremming_press_enhet (
    indikator                STRING      NOT NULL,
    enhet                    STRING      NOT NULL,
    analyse_dato             DATE        NOT NULL,
    mottatt_antall           INT         NOT NULL,
    ferdigstilt_antall       INT         NOT NULL,
    netto_flyt               INT         NOT NULL,
    median_tidsbruk          DOUBLE,
    p90_tidsbruk             DOUBLE,
    netto_flyt_streak        INT         NOT NULL,
    baseline_p90_tidsbruk    DOUBLE,
    tidsbruk_avvik_pct       DOUBLE,
    pressure_score           DOUBLE      NOT NULL,
    pressure_nivaa           STRING      NOT NULL,
    tilstrekkelig_volum      BOOLEAN     NOT NULL,
    kjoert_tidspunkt         TIMESTAMP   NOT NULL,
    kjoere_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'Teamnivå-gjennomstrømming per indikator/måned. Kombinerer netto flyt (mottatt-ferdigstilt) og tidsbruksforverring mot historisk baseline.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.gjennomstroemming_press_fase (
    indikator                STRING      NOT NULL,
    enhet                    STRING      NOT NULL,
    fasetittel               STRING      NOT NULL,
    analyse_dato             DATE        NOT NULL,
    mottatt_antall           INT         NOT NULL,
    ferdigstilt_antall       INT         NOT NULL,
    netto_flyt               INT         NOT NULL,
    median_tidsbruk          DOUBLE,
    p90_tidsbruk             DOUBLE,
    fase_press_flagg         BOOLEAN     NOT NULL,
    fase_bidrag_score        DOUBLE,
    kjoert_tidspunkt         TIMESTAMP   NOT NULL,
    kjoere_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'Fasenivå-støttetabell for team med throughput-press. Viser hvilke faser som bidrar mest til presset.'
""")

print("gjennomstoremming_press_enhet og gjennomstroemming_press_fase tabeller er klare")


# =============================================================================
# CELL 2 — Load monthly aggregates
# =============================================================================

received = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        (YEAR(pr.startmilepaeldato) * 100 + MONTH(pr.startmilepaeldato)) AS period,
        COUNT(1) AS mottatt_antall
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.startmilepaeldato IS NOT NULL
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
      AND pr.indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.startmilepaeldato) >= {START_YEAR}
    GROUP BY pr.indikator, COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             YEAR(pr.startmilepaeldato), MONTH(pr.startmilepaeldato)
""").toPandas()

completed = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        (YEAR(pr.sluttmilepaeldato) * 100 + MONTH(pr.sluttmilepaeldato)) AS period,
        COUNT(1) AS ferdigstilt_antall,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.5, 1000) AS median_tidsbruk,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.9, 1000) AS p90_tidsbruk
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.sluttmilepaeldato IS NOT NULL
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
      AND pr.indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
    GROUP BY pr.indikator, COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

phase = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
        (YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) * 100
            + MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))) AS period,
        COUNT(CASE WHEN pr.startmilepaeldato IS NOT NULL THEN 1 END) AS mottatt_antall,
        COUNT(CASE WHEN pr.sluttmilepaeldato IS NOT NULL THEN 1 END) AS ferdigstilt_antall,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.5, 1000) AS median_tidsbruk,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.9, 1000) AS p90_tidsbruk
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

for _df in (received, completed, phase):
    if _df.empty:
        continue
    _df["period"] = pd.to_numeric(_df["period"], errors="coerce").astype("int64")

print(f"Mottatt-rader:      {len(received):,}")
print(f"Ferdigstilt-rader:  {len(completed):,}")
print(f"Fase-rader:         {len(phase):,}")


# =============================================================================
# CELL 3 — Compute team-level pressure
# =============================================================================

if received.empty and completed.empty:
    team_df = pd.DataFrame(columns=[
        "indikator", "enhet", "period", "mottatt_antall", "ferdigstilt_antall",
        "netto_flyt", "median_tidsbruk", "p90_tidsbruk",
    ])
else:
    team_df = received.merge(
        completed,
        on=["indikator", "enhet", "period"],
        how="outer",
    )
    team_df["mottatt_antall"] = team_df["mottatt_antall"].fillna(0).astype("int32")
    team_df["ferdigstilt_antall"] = team_df["ferdigstilt_antall"].fillna(0).astype("int32")
    team_df["netto_flyt"] = team_df["mottatt_antall"] - team_df["ferdigstilt_antall"]

    for col in ("median_tidsbruk", "p90_tidsbruk"):
        team_df[col] = pd.to_numeric(team_df[col], errors="coerce").astype("float64")


def _month_end_from_period(period: int):
    year = int(period) // 100
    month = int(period) % 100
    return (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()


out_rows = []
for (indikator, enhet), grp in team_df.groupby(["indikator", "enhet"], dropna=False):
    g = grp.sort_values("period").copy()
    enhet_value = "Ukjent" if enhet is None or enhet == "" else str(enhet)

    streak = []
    current = 0
    for v in g["netto_flyt"].tolist():
        if v > 0:
            current += 1
        else:
            current = 0
        streak.append(current)
    g["netto_flyt_streak"] = streak

    g["baseline_p90_tidsbruk"] = (
        g["p90_tidsbruk"].shift(1)
        .rolling(BASELINE_MONTHS, min_periods=MIN_BASELINE_OBS)
        .median()
    )

    g["tidsbruk_avvik_pct"] = np.where(
        g["baseline_p90_tidsbruk"].notna() & (g["baseline_p90_tidsbruk"] > 0),
        (g["p90_tidsbruk"] - g["baseline_p90_tidsbruk"]) / g["baseline_p90_tidsbruk"],
        np.nan,
    )

    score = np.zeros(len(g), dtype="float64")
    score += (g["netto_flyt"] > 0).astype(float) * 1.5
    score += (g["netto_flyt_streak"] >= POSITIVE_FLOW_STREAK).astype(float) * 2.0
    score += (g["tidsbruk_avvik_pct"] > 0.10).astype(float) * 1.0
    score += (g["tidsbruk_avvik_pct"] > 0.25).astype(float) * 1.0
    score += ((g["ferdigstilt_antall"] > 0) & (g["mottatt_antall"] >= g["ferdigstilt_antall"] * 1.3)).astype(float) * 1.0
    g["pressure_score"] = score

    def _label(s):
        if s >= 4.0:
            return "Kritisk"
        if s >= 3.0:
            return "Hoy"
        if s >= 1.5:
            return "Moderat"
        return "Lav"

    g["pressure_nivaa"] = g["pressure_score"].apply(_label)
    g["tilstrekkelig_volum"] = (
        (g["mottatt_antall"] >= MIN_TEAM_VOLUME)
        | (g["ferdigstilt_antall"] >= MIN_TEAM_VOLUME)
    )

    for _, row in g.iterrows():
        out_rows.append({
            "indikator": indikator,
            "enhet": enhet_value,
            "analyse_dato": _month_end_from_period(int(row["period"])),
            "mottatt_antall": int(row["mottatt_antall"]),
            "ferdigstilt_antall": int(row["ferdigstilt_antall"]),
            "netto_flyt": int(row["netto_flyt"]),
            "median_tidsbruk": float(row["median_tidsbruk"]) if pd.notna(row["median_tidsbruk"]) else None,
            "p90_tidsbruk": float(row["p90_tidsbruk"]) if pd.notna(row["p90_tidsbruk"]) else None,
            "netto_flyt_streak": int(row["netto_flyt_streak"]),
            "baseline_p90_tidsbruk": float(row["baseline_p90_tidsbruk"]) if pd.notna(row["baseline_p90_tidsbruk"]) else None,
            "tidsbruk_avvik_pct": float(row["tidsbruk_avvik_pct"]) if pd.notna(row["tidsbruk_avvik_pct"]) else None,
            "pressure_score": float(row["pressure_score"]),
            "pressure_nivaa": row["pressure_nivaa"],
            "tilstrekkelig_volum": bool(row["tilstrekkelig_volum"]),
            "kjoert_tidspunkt": datetime.now(),
            "kjoere_id": BATCH_ID,
        })

team_out = pd.DataFrame(out_rows)
print(f"Team-rader beregnet: {len(team_out):,}")


# =============================================================================
# CELL 4 — Compute phase-level contribution
# =============================================================================

phase_rows = []
if not phase.empty and not team_out.empty:
    phase_df = phase.copy()
    phase_df["mottatt_antall"] = pd.to_numeric(phase_df["mottatt_antall"], errors="coerce").fillna(0).astype("int32")
    phase_df["ferdigstilt_antall"] = pd.to_numeric(phase_df["ferdigstilt_antall"], errors="coerce").fillna(0).astype("int32")
    phase_df["netto_flyt"] = phase_df["mottatt_antall"] - phase_df["ferdigstilt_antall"]
    phase_df["median_tidsbruk"] = pd.to_numeric(phase_df["median_tidsbruk"], errors="coerce").astype("float64")
    phase_df["p90_tidsbruk"] = pd.to_numeric(phase_df["p90_tidsbruk"], errors="coerce").astype("float64")

    team_join = team_out.copy()
    team_join["period"] = (
        pd.to_datetime(team_join["analyse_dato"]).dt.year * 100
        + pd.to_datetime(team_join["analyse_dato"]).dt.month
    ).astype("int64")

    phase_df = phase_df.merge(
        team_join[["indikator", "enhet", "period", "pressure_nivaa", "p90_tidsbruk", "tilstrekkelig_volum"]],
        on=["indikator", "enhet", "period"],
        how="left",
        suffixes=("", "_team"),
    )

    for _, row in phase_df.iterrows():
        team_p90 = row.get("p90_tidsbruk_team")
        phase_p90 = row.get("p90_tidsbruk")
        phase_score = 0.0
        if pd.notna(row["netto_flyt"]) and row["netto_flyt"] > 0:
            phase_score += 1.5
        if pd.notna(phase_p90) and pd.notna(team_p90) and team_p90 > 0 and phase_p90 >= (team_p90 * 0.9):
            phase_score += 1.0
        if row.get("pressure_nivaa") in ("Hoy", "Kritisk"):
            phase_score += 1.0

        phase_flag = bool(
            row.get("tilstrekkelig_volum", False)
            and phase_score >= 2.0
            and row["ferdigstilt_antall"] >= MIN_TEAM_VOLUME
        )

        phase_rows.append({
            "indikator": row["indikator"],
            "enhet": row["enhet"],
            "fasetittel": row["fasetittel"],
            "analyse_dato": _month_end_from_period(int(row["period"])),
            "mottatt_antall": int(row["mottatt_antall"]),
            "ferdigstilt_antall": int(row["ferdigstilt_antall"]),
            "netto_flyt": int(row["netto_flyt"]),
            "median_tidsbruk": float(row["median_tidsbruk"]) if pd.notna(row["median_tidsbruk"]) else None,
            "p90_tidsbruk": float(phase_p90) if pd.notna(phase_p90) else None,
            "fase_press_flagg": phase_flag,
            "fase_bidrag_score": round(float(phase_score), 3),
            "kjoert_tidspunkt": datetime.now(),
            "kjoere_id": BATCH_ID,
        })

phase_out = pd.DataFrame(phase_rows)
print(f"Fase-rader beregnet: {len(phase_out):,}")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

TEAM_SCHEMA = StructType([
    StructField("indikator", StringType(), False),
    StructField("enhet", StringType(), False),
    StructField("analyse_dato", DateType(), False),
    StructField("mottatt_antall", IntegerType(), False),
    StructField("ferdigstilt_antall", IntegerType(), False),
    StructField("netto_flyt", IntegerType(), False),
    StructField("median_tidsbruk", DoubleType(), True),
    StructField("p90_tidsbruk", DoubleType(), True),
    StructField("netto_flyt_streak", IntegerType(), False),
    StructField("baseline_p90_tidsbruk", DoubleType(), True),
    StructField("tidsbruk_avvik_pct", DoubleType(), True),
    StructField("pressure_score", DoubleType(), False),
    StructField("pressure_nivaa", StringType(), False),
    StructField("tilstrekkelig_volum", BooleanType(), False),
    StructField("kjoert_tidspunkt", TimestampType(), False),
    StructField("kjoere_id", StringType(), False),
])

PHASE_SCHEMA = StructType([
    StructField("indikator", StringType(), False),
    StructField("enhet", StringType(), False),
    StructField("fasetittel", StringType(), False),
    StructField("analyse_dato", DateType(), False),
    StructField("mottatt_antall", IntegerType(), False),
    StructField("ferdigstilt_antall", IntegerType(), False),
    StructField("netto_flyt", IntegerType(), False),
    StructField("median_tidsbruk", DoubleType(), True),
    StructField("p90_tidsbruk", DoubleType(), True),
    StructField("fase_press_flagg", BooleanType(), False),
    StructField("fase_bidrag_score", DoubleType(), True),
    StructField("kjoert_tidspunkt", TimestampType(), False),
    StructField("kjoere_id", StringType(), False),
])


def to_records(rows, schema):
    casters = {
        StringType(): lambda v: None if v is None else str(v),
        IntegerType(): lambda v: None if v is None else int(v),
        DoubleType(): lambda v: None if v is None else float(v),
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

if not team_out.empty:
    team_spark = spark.createDataFrame(
        to_records(team_out.to_dict("records"), TEAM_SCHEMA), schema=TEAM_SCHEMA
    )
    team_spark.write.mode("overwrite").saveAsTable("analyser.gjennomstoremming_press_enhet")
    print(f"gjennomstoremming_press_enhet skrevet: {len(team_out):,} rader")
else:
    print("Ingen team-resultater å skrive")

if not phase_out.empty:
    phase_spark = spark.createDataFrame(
        to_records(phase_out.to_dict("records"), PHASE_SCHEMA), schema=PHASE_SCHEMA
    )
    phase_spark.write.mode("overwrite").saveAsTable("analyser.gjennomstroemming_press_fase")
    print(f"gjennomstroemming_press_fase skrevet: {len(phase_out):,} rader")
else:
    print("Ingen fase-resultater å skrive")


# Latest top-risk snapshot
if not team_out.empty:
    spark.sql(f"""
        SELECT
            indikator,
            enhet,
            analyse_dato,
            mottatt_antall,
            ferdigstilt_antall,
            netto_flyt,
            ROUND(p90_tidsbruk, 1) AS p90_tidsbruk,
            ROUND(tidsbruk_avvik_pct * 100, 1) AS tidsbruk_avvik_pct,
            pressure_nivaa,
            ROUND(pressure_score, 2) AS pressure_score
                FROM analyser.gjennomstoremming_press_enhet
        WHERE kjoere_id = '{BATCH_ID}'
          AND analyse_dato = (
              SELECT MAX(analyse_dato)
                            FROM analyser.gjennomstoremming_press_enhet
              WHERE kjoere_id = '{BATCH_ID}'
          )
        ORDER BY pressure_score DESC, netto_flyt DESC
    """).show(50, truncate=False)
