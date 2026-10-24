# =============================================================================
# Throughput pressure monitor per team and indicator.
# Runs nightly after main data pipeline.
#
# Purpose:
#   Detect where intake exceeds completions for multiple periods and where
#   processing time deteriorates versus recent baseline.
#
# Output tables:
#   gjennomstroemming_press_enhet  — pressignal per enhet x indikator x month
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
# Net flow only counts as pressure when it is large relative to the team's
# normal output: (netto_flyt - same-month seasonal baseline) divided by the
# median ferdigstilt_antall over the previous BASELINE_MONTHS must exceed this.
NETTO_FLYT_REL_TERSKEL = 0.10
SEASONAL_BASELINE_YEARS = 3      # same calendar month in this many prior years
MIN_SEASONAL_BASELINE_YEARS = 2

# Only complete months are scored: the current calendar month is still
# filling up, so its counts would look like a drop in output every run. DAX "siste analysedato" measures pick
# up the last complete period automatically as a result.
MONTH_CUTOFF_SQL = "CAST(DATE_TRUNC('MONTH', CURRENT_DATE()) AS DATE)"


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.gjennomstroemming_press_enhet (
    indikator                STRING      NOT NULL,
    enhet                    STRING      NOT NULL,
    analyse_dato             DATE        NOT NULL,
    mottatt_antall           INT         NOT NULL,
    ferdigstilt_antall       INT         NOT NULL,
    netto_flyt               INT         NOT NULL,
    median_tidsbruk          DOUBLE,
    p90_tidsbruk             DOUBLE,
    netto_flyt_saesong_baseline DOUBLE,
    netto_flyt_rel           DOUBLE,
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
COMMENT 'Teamnivå-gjennomstrømming per indikator/måned (bare hele måneder, også måneder uten aktivitet). netto_flyt = mottatt (startmåned) - ferdigstilt (sluttmåned). netto_flyt_saesong_baseline er median netto flyt i samme kalendermåned de tre foregående årene. netto_flyt_rel = (netto_flyt - sesongbaseline) / median ferdigstilt siste 12 måneder; over 0,10 regnes som press. netto_flyt_streak er antall måneder på rad med slikt press. Kombineres med tidsbruksforverring mot historisk baseline.'
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
COMMENT 'Fasenivå-støttetabell for team med throughput-press. Viser hvilke faser som bidrar mest til presset. mottatt_antall telles i startmåneden, ferdigstilt_antall i sluttmåneden, netto_flyt = mottatt - ferdigstilt.'
""")

print("gjennomstroemming_press_enhet og gjennomstroemming_press_fase tabeller er klare")


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
      AND pr.startmilepaeldato < {MONTH_CUTOFF_SQL}
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
      AND pr.sluttmilepaeldato < {MONTH_CUTOFF_SQL}
    GROUP BY pr.indikator, COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

# Fase level uses the same split as the team level above: intake by start
# month, output (and tidsbruk percentiles) by end month. Grouping both counts
# by COALESCE(slutt, start) — an earlier version — made every closed fase
# net to zero in its end month.
phase_received = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
        (YEAR(pr.startmilepaeldato) * 100 + MONTH(pr.startmilepaeldato)) AS period,
        COUNT(1) AS mottatt_antall
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.startmilepaeldato IS NOT NULL
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
      AND pr.indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.startmilepaeldato) >= {START_YEAR}
      AND pr.startmilepaeldato < {MONTH_CUTOFF_SQL}
    GROUP BY pr.indikator,
             COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase'),
             YEAR(pr.startmilepaeldato), MONTH(pr.startmilepaeldato)
""").toPandas()

phase_completed = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
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
      AND pr.sluttmilepaeldato < {MONTH_CUTOFF_SQL}
    GROUP BY pr.indikator,
             COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase'),
             YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

for _df in (received, completed, phase_received, phase_completed):
    if _df.empty:
        continue
    _df["period"] = pd.to_numeric(_df["period"], errors="coerce").astype("int64")

print(f"Mottatt-rader:      {len(received):,}")
print(f"Ferdigstilt-rader:  {len(completed):,}")
print(f"Fase mottatt-rader: {len(phase_received):,}")
print(f"Fase ferdig-rader:  {len(phase_completed):,}")


def _period_to_month_index(period):
    period = int(period)
    return (period // 100) * 12 + (period % 100 - 1)


def _month_index_to_period(index):
    index = int(index)
    return (index // 12) * 100 + (index % 12 + 1)


def fill_month_gaps(df, keys, zero_cols):
    """Give every key group one row per calendar month between its first and
    last period. A month with no intake and no output is a real month with
    zero flow — dropping it would let a running count (the flow streak) run
    straight across it. zero_cols are filled with 0; other columns stay NaN."""
    if df.empty:
        return df
    parts = []
    for key_values, grp in df.groupby(keys, dropna=False, sort=False):
        idx = grp["period"].map(_period_to_month_index)
        full = pd.DataFrame({"period": [
            _month_index_to_period(i) for i in range(int(idx.min()), int(idx.max()) + 1)
        ]})
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        for key, value in zip(keys, key_values):
            full[key] = value
        parts.append(full.merge(grp, on=keys + ["period"], how="left"))
    out = pd.concat(parts, ignore_index=True)
    for col in zero_cols:
        out[col] = out[col].fillna(0)
    return out


def combine_flow(received_df, completed_df, keys):
    """Outer-merge intake (counted by start month) and output (counted by end
    month) on key + period, fill missing counts and gap months with 0, and
    compute netto_flyt = mottatt - ferdigstilt per month."""
    merged = received_df.merge(completed_df, on=keys + ["period"], how="outer")
    for col in ("mottatt_antall", "ferdigstilt_antall"):
        if col not in merged:
            merged[col] = 0
    merged = fill_month_gaps(merged, keys, ["mottatt_antall", "ferdigstilt_antall"])
    merged["mottatt_antall"] = pd.to_numeric(merged["mottatt_antall"], errors="coerce").fillna(0).astype("int64")
    merged["ferdigstilt_antall"] = pd.to_numeric(merged["ferdigstilt_antall"], errors="coerce").fillna(0).astype("int64")
    for col in ("median_tidsbruk", "p90_tidsbruk"):
        if col not in merged:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged["netto_flyt"] = merged["mottatt_antall"] - merged["ferdigstilt_antall"]
    return merged.sort_values(keys + ["period"]).reset_index(drop=True)


# =============================================================================
# CELL 3 — Compute team-level pressure
# =============================================================================

def seasonal_flow_baseline(periods, netto_flyt, years=SEASONAL_BASELINE_YEARS,
                           min_years=MIN_SEASONAL_BASELINE_YEARS):
    """Median netto_flyt for the same calendar month in the `years` prior
    years (NaN with fewer than `min_years` of them). January and the months
    after summer normally take in more than they finish; this is the "normal"
    a month's net flow is compared with."""
    lookup = dict(zip((int(p) for p in periods), netto_flyt))
    out = []
    for p in periods:
        year, month = int(p) // 100, int(p) % 100
        prior = [lookup[(year - k) * 100 + month] for k in range(1, years + 1)
                 if (year - k) * 100 + month in lookup]
        out.append(float(np.median(prior)) if len(prior) >= min_years else np.nan)
    return np.array(out, dtype="float64")


def flow_pressure(g):
    """For one enhet x indikator, sorted by period with no gap months:
    - netto_flyt_saesong_baseline: same-month median net flow, prior 3 years
    - netto_flyt_rel: (netto_flyt - seasonal baseline) / median
      ferdigstilt_antall over the previous BASELINE_MONTHS months. With no
      seasonal baseline yet, netto_flyt itself is used.
    - netto_flyt_press: netto_flyt_rel > NETTO_FLYT_REL_TERSKEL. +1 on a team
      finishing 200 a month is not pressure; +30 is.
    - netto_flyt_streak: consecutive months with netto_flyt_press
    Returns a copy of g with these columns added."""
    g = g.copy()
    g["netto_flyt_saesong_baseline"] = seasonal_flow_baseline(
        g["period"].tolist(), g["netto_flyt"].astype(float).tolist()
    )
    justert = g["netto_flyt"] - g["netto_flyt_saesong_baseline"].fillna(0.0)
    ferdig_baseline = (
        g["ferdigstilt_antall"].astype(float).shift(1)
        .rolling(BASELINE_MONTHS, min_periods=MIN_BASELINE_OBS)
        .median()
    )
    g["netto_flyt_rel"] = np.where(
        ferdig_baseline.notna() & (ferdig_baseline > 0),
        justert / ferdig_baseline,
        np.nan,
    )
    g["netto_flyt_press"] = g["netto_flyt_rel"] > NETTO_FLYT_REL_TERSKEL

    streak = []
    current = 0
    for v in g["netto_flyt_press"].tolist():
        current = current + 1 if v else 0
        streak.append(current)
    g["netto_flyt_streak"] = streak
    return g


def pressure_label(score):
    if score >= 4.0:
        return "Kritisk"
    if score >= 3.0:
        return "Høy"
    if score >= 1.5:
        return "Moderat"
    return "Lav"


TEAM_KEYS = ["indikator", "enhet"]

if received.empty and completed.empty:
    team_df = pd.DataFrame(columns=TEAM_KEYS + [
        "period", "mottatt_antall", "ferdigstilt_antall",
        "netto_flyt", "median_tidsbruk", "p90_tidsbruk",
    ])
else:
    # combine_flow also fills months with no activity, so the streak below
    # can't run across a month that had no row at all.
    team_df = combine_flow(received, completed, TEAM_KEYS)


def _month_end_from_period(period: int):
    year = int(period) // 100
    month = int(period) % 100
    return (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()


out_rows = []
for (indikator, enhet), grp in team_df.groupby(["indikator", "enhet"], dropna=False):
    g = flow_pressure(grp.sort_values("period"))
    enhet_value = "Ukjent" if enhet is None or enhet == "" else str(enhet)

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
    score += g["netto_flyt_press"].astype(float) * 1.5
    score += (g["netto_flyt_streak"] >= POSITIVE_FLOW_STREAK).astype(float) * 2.0
    score += (g["tidsbruk_avvik_pct"] > 0.10).astype(float) * 1.0
    score += (g["tidsbruk_avvik_pct"] > 0.25).astype(float) * 1.0
    score += ((g["ferdigstilt_antall"] > 0) & (g["mottatt_antall"] >= g["ferdigstilt_antall"] * 1.3)).astype(float) * 1.0
    g["pressure_score"] = score

    g["pressure_nivaa"] = g["pressure_score"].apply(pressure_label)
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
            "netto_flyt_saesong_baseline": float(row["netto_flyt_saesong_baseline"]) if pd.notna(row["netto_flyt_saesong_baseline"]) else None,
            "netto_flyt_rel": float(row["netto_flyt_rel"]) if pd.notna(row["netto_flyt_rel"]) else None,
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
if not (phase_received.empty and phase_completed.empty) and not team_out.empty:
    phase_df = combine_flow(phase_received, phase_completed, ["indikator", "enhet", "fasetittel"])

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
        if row.get("pressure_nivaa") in ("Høy", "Kritisk"):
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
    StructField("netto_flyt_saesong_baseline", DoubleType(), True),
    StructField("netto_flyt_rel", DoubleType(), True),
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
        # pandas may hand back pandas.Timestamp/np.datetime64-backed dates here
        # (e.g. after a DataFrame round-trip). Spark's type verifier only
        # accepts the exact datetime.datetime/datetime.date types, not
        # subclasses, so these must be converted to native python objects.
        TimestampType(): lambda v: None if v is None else pd.Timestamp(v).to_pydatetime(),
        DateType(): lambda v: None if v is None else pd.Timestamp(v).date(),
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
    # overwriteSchema: the table is fully rewritten every night, and new
    # columns (netto_flyt_rel, netto_flyt_saesong_baseline) must replace the
    # schema the CREATE TABLE IF NOT EXISTS above left from an earlier run.
    team_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("analyser.gjennomstroemming_press_enhet")
    print(f"gjennomstroemming_press_enhet skrevet: {len(team_out):,} rader")
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
                FROM analyser.gjennomstroemming_press_enhet
        WHERE kjoere_id = '{BATCH_ID}'
          AND analyse_dato = (
              SELECT MAX(analyse_dato)
                            FROM analyser.gjennomstroemming_press_enhet
              WHERE kjoere_id = '{BATCH_ID}'
          )
        ORDER BY pressure_score DESC, netto_flyt DESC
    """).show(50, truncate=False)
