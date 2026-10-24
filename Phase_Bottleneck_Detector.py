# =============================================================================
# Phase bottleneck detector per team, phase and indicator.
# Runs nightly after main data pipeline.
#
# Purpose:
#   Detect where queue pressure is forming inside process phases by combining
#   net flow and deterioration in p90 phase time versus recent baseline.
#
# Flow definition:
#   inngang (intake) is counted in the month of startmilepaeldato, utgang
#   (output) in the month of sluttmilepaeldato, and netto_flyt = inngang -
#   utgang per month. fase_ko_proxy is the real number of open faser at each
#   month end: faser already open when the series starts plus cumulative
#   netto_flyt since then.
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

# Only complete months are scored: the current calendar month is still
# filling up, so its counts would look like a drop in output every run. DAX "siste analysedato" measures pick
# up the last complete period automatically as a result.
MONTH_CUTOFF_SQL = "CAST(DATE_TRUNC('MONTH', CURRENT_DATE()) AS DATE)"
KO_VEKST_TERSKEL = 0.10  # queue must be this much above its median over the last BASELINE_MONTHS


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
COMMENT 'Flaskehalsdeteksjon per team/fase/indikator og måned. inngang_antall telles i startmåneden, utgang_antall i sluttmåneden, og netto_flyt = inngang - utgang. fase_ko_proxy er antall åpne faser ved månedsslutt (åpne ved seriestart + kumulativ netto flyt). Kombinerer netto flyt, køvekst og p90-avvik mot historisk baseline.'
""")

print("fase_flaskehals_enhet-tabellen er klar")


# =============================================================================
# CELL 2 — Load monthly intake, output and opening stock
# =============================================================================

# Intake and output are counted in their OWN month: a fase enters in the month
# of its startmilepaeldato and leaves in the month of its sluttmilepaeldato.
# (An earlier version grouped both counts by COALESCE(slutt, start), so a
# closed fase added +1 and -1 to the same month and netted to zero — the
# "net flow" was really just the still-open faser by start month.)
received = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
        (YEAR(pr.startmilepaeldato) * 100 + MONTH(pr.startmilepaeldato)) AS period,
        COUNT(1) AS inngang_antall
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

# Phase time percentiles come from completed faser only — an open fase has
# no final tidsbruk yet. startmilepaeldato IS NOT NULL keeps a fase from
# leaving a queue it never entered, so the queue below stays consistent.
completed = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
        (YEAR(pr.sluttmilepaeldato) * 100 + MONTH(pr.sluttmilepaeldato)) AS period,
        COUNT(1) AS utgang_antall,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.5, 1000) AS median_fasetid,
        percentile_approx(CAST(pr.tidsbruk AS DOUBLE), 0.9, 1000) AS p90_fasetid
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.sluttmilepaeldato IS NOT NULL
      AND pr.startmilepaeldato IS NOT NULL
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
      AND pr.indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
      AND pr.sluttmilepaeldato < {MONTH_CUTOFF_SQL}
    GROUP BY pr.indikator,
             COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase'),
             YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

# Faser already open when the series starts. Without this the queue would
# start at zero on {START_YEAR}-01-01 and ignore everything already waiting.
# With it, opening stock + cumulative (inngang - utgang) is the real number
# of open faser at each month end.
opening = spark.sql(f"""
    SELECT
        pr.indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent') AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase') AS fasetittel,
        COUNT(1) AS aapne_ved_start
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.startmilepaeldato < DATE'{START_YEAR}-01-01'
      AND (pr.sluttmilepaeldato IS NULL OR pr.sluttmilepaeldato >= DATE'{START_YEAR}-01-01')
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
      AND pr.indikator NOT LIKE '%avtalt%'
    GROUP BY pr.indikator,
             COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent'),
             COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase')
""").toPandas()

for _df in (received, completed):
    if not _df.empty:
        _df["period"] = pd.to_numeric(_df["period"], errors="coerce").astype("int64")
if not opening.empty:
    opening["aapne_ved_start"] = pd.to_numeric(opening["aapne_ved_start"], errors="coerce").fillna(0).astype("int64")

print(f"Inngang-rader lastet:   {len(received):,}")
print(f"Utgang-rader lastet:    {len(completed):,}")
print(f"Åpne ved start lastet:  {len(opening):,}")


# =============================================================================
# CELL 3 — Combine intake and output into one monthly flow table
# =============================================================================

PHASE_KEYS = ["indikator", "enhet", "fasetittel"]


def _period_to_month_index(period):
    period = int(period)
    return (period // 100) * 12 + (period % 100 - 1)


def _month_index_to_period(index):
    index = int(index)
    return (index // 12) * 100 + (index % 12 + 1)


def fill_month_gaps(df, keys, zero_cols):
    """Give every key group one row per calendar month between its first and
    last period. A month with no intake and no output is a real month with
    zero flow — dropping it would make a running count (queue, streak) skip
    over it. zero_cols are filled with 0; other columns stay NaN."""
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


def combine_flow(received_df, completed_df, keys=PHASE_KEYS):
    """Outer-merge intake (by start month) and output (by end month) on the
    key + period, fill missing counts with 0 and gap months with zero rows,
    and compute netto_flyt = inngang - utgang for each month."""
    merged = received_df.merge(completed_df, on=keys + ["period"], how="outer")
    for col in ("inngang_antall", "utgang_antall"):
        if col not in merged:
            merged[col] = 0
    merged = fill_month_gaps(merged, keys, ["inngang_antall", "utgang_antall"])
    merged["inngang_antall"] = pd.to_numeric(merged["inngang_antall"], errors="coerce").fillna(0).astype("int64")
    merged["utgang_antall"] = pd.to_numeric(merged["utgang_antall"], errors="coerce").fillna(0).astype("int64")
    for col in ("median_fasetid", "p90_fasetid"):
        if col not in merged:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged["netto_flyt"] = merged["inngang_antall"] - merged["utgang_antall"]
    return merged.sort_values(keys + ["period"]).reset_index(drop=True)


def queue_at_month_end(netto_flyt, opening_stock=0):
    """Open faser at each month end: opening stock plus cumulative net flow.
    Clipped at 0 — a negative queue only happens on bad milestone dates
    (slutt before start), and is not a real stock."""
    return (opening_stock + pd.Series(netto_flyt).cumsum()).clip(lower=0)


if received.empty and completed.empty:
    phase = pd.DataFrame(columns=PHASE_KEYS + [
        "period", "inngang_antall", "utgang_antall", "median_fasetid", "p90_fasetid", "netto_flyt",
    ])
    print("Ingen fase-data tilgjengelig.")
else:
    phase = combine_flow(received, completed)

print(f"Fase-rader etter sammenslåing: {len(phase):,}")


# =============================================================================
# CELL 4 — Compute bottlenecks
# =============================================================================


def _month_end_from_period(period: int):
    year = int(period) // 100
    month = int(period) % 100
    return (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()


def score_bottleneck(netto_flyt, avvik_p90_pct, inngang_antall, utgang_antall, ko_vekst):
    """Score one enhet x fase x month. Returns (score, reason codes).
    ko_vekst is True when the month-end queue is clearly above its own recent
    level (see KO_VEKST_TERSKEL) — with a real queue in fase_ko_proxy, "queue
    > 0" would be true almost every month and add nothing to NETTO_FLYT_POS."""
    score = 0.0
    reasons = []

    if netto_flyt > 0:
        score += 1.5
        reasons.append("NETTO_FLYT_POS")

    if pd.notna(avvik_p90_pct) and avvik_p90_pct > 0.15:
        score += 1.5
        reasons.append("P90_OPP")

    if pd.notna(avvik_p90_pct) and avvik_p90_pct > 0.30:
        score += 1.0
        reasons.append("P90_OPP_STERK")

    if inngang_antall > 0 and utgang_antall <= (inngang_antall * 0.8):
        score += 1.0
        reasons.append("UTGANGSGAP")

    if ko_vekst:
        score += 1.0
        reasons.append("KO_VEKST")

    return score, reasons


def severity_label(score):
    if score >= 4.0:
        return "Kritisk"
    if score >= 3.0:
        return "Høy"
    if score >= 1.5:
        return "Moderat"
    return "Lav"


if opening.empty:
    opening_lookup = {}
else:
    opening_lookup = {
        (r["indikator"], r["enhet"], r["fasetittel"]): int(r["aapne_ved_start"])
        for _, r in opening.iterrows()
    }

rows = []
for (indikator, enhet, fasetittel), grp in phase.groupby(PHASE_KEYS, dropna=False):
    g = grp.sort_values("period").copy()
    g["fase_ko_proxy"] = queue_at_month_end(
        g["netto_flyt"].to_numpy(), opening_lookup.get((indikator, enhet, fasetittel), 0)
    ).to_numpy()

    g["baseline_ko"] = (
        g["fase_ko_proxy"].shift(1)
        .rolling(BASELINE_MONTHS, min_periods=MIN_BASELINE_OBS)
        .median()
    )
    g["ko_vekst"] = (
        g["baseline_ko"].notna()
        & (g["fase_ko_proxy"] > g["baseline_ko"] * (1 + KO_VEKST_TERSKEL))
        & (g["netto_flyt"] > 0)
    )

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
        score, reasons = score_bottleneck(
            row["netto_flyt"], row["avvik_p90_pct"],
            row["inngang_antall"], row["utgang_antall"], bool(row["ko_vekst"]),
        )
        severity = severity_label(score)

        bottleneck = bool(row["tilstrekkelig_volum"] and score >= 3.0)
        reason_code = "+".join(reasons) if reasons else "INGEN"

        reason_text = ""
        if reason_code == "INGEN":
            reason_text = "Ingen tydelig faseflaskehals i perioden."
        elif "KO_VEKST" in reason_code and "P90_OPP" in reason_code:
            reason_text = "Kø vokser samtidig som fasevariasjon i p90 forverres."
        elif "KO_VEKST" in reason_code:
            reason_text = "Køen av åpne saker i fasen er tydelig høyere enn det siste året."
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
# CELL 5 — Write to Lakehouse
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
