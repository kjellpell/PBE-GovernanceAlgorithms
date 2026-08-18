
# =============================================================================
# Exponentially Weighted Moving Average smoothing per indicator.
# Produces smoothed trend lines for frist%, behandlingstid and
# produksjonsdifferanse for use in board and governance report charts.
#
# EWMA weights recent observations more heavily than a simple moving
# average, so trend changes show up faster. A declining EWMA line on
# a frist% chart signals deteriorating momentum before the raw monthly
# values make it obvious.
#
# Output table: ewma_analyse
# Power BI: plot EWMA line alongside raw monthly values on the same
# chart. The raw line shows actual performance, the EWMA line shows
# the underlying trend direction.
#
# Schedule: nightly after main data pipeline.
# No external libraries required — EWMA computed in pandas.
# =============================================================================

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()
BATCH_ID    = datetime.now().strftime("%Y%m%dT%H%M%S")
START_YEAR  = 2015  # exclude data before this year — adjust if older data is reliable

# EWMA smoothing parameter alpha — controls how quickly the average
# responds to new observations.
# alpha = 0.1 → slow, heavily smoothed, good for board (stable trend line)
# alpha = 0.3 → medium, good for governance team (picks up changes faster)
# alpha = 0.5 → fast, reactive, good for early warning
# Both slow and fast computed and written — Power BI slicer lets user choose.
ALPHA_SLOW  = 0.1
ALPHA_FAST  = 0.3


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS ewma_analyse (
    indikator           STRING      NOT NULL,
    maaltall            STRING      NOT NULL,  -- Fristprosent / Behandlingstid / Produksjonsdifferanse
    analyse_dato        DATE        NOT NULL,
    verdi               DOUBLE,                -- rå månedlig verdi
    ewma_sakte          DOUBLE,                -- alfa=0.1 utjevnet verdi
    ewma_rask           DOUBLE,                -- alfa=0.3 utjevnet verdi
    ewma_helning_sakte  DOUBLE,                -- endring fra måned til måned i ewma_sakte
    ewma_helning_rask   DOUBLE,                -- endring fra måned til måned i ewma_rask
    trendretning        STRING,                -- Stigende / Synkende / Stabil
    kjoert_tidspunkt    TIMESTAMP   NOT NULL,
    id                  STRING      NOT NULL
)
USING DELTA
COMMENT 'EWMA-utjevnede trendlinjer per indikator og måltall. Brukes i trenddiagrammer for styre- og virksomhetsoppfølging.'
""")

print("ewma_analyse-tabellen er klar")


# =============================================================================
# CELL 2 — Load monthly data
# =============================================================================

# Monthly frist% per indicator
monthly_frist = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(pr.sluttmilepaeldato) * 100 + MONTH(pr.sluttmilepaeldato)) AS period,
                CASE
                    WHEN COUNT(CASE WHEN pr.frist IS NOT NULL THEN 1 END) = 0 THEN NULL
                        ELSE COUNT(CASE WHEN pr.innenfor_frist = 1 THEN 1 END) * 1.0
                         / COUNT(CASE WHEN pr.frist IS NOT NULL THEN 1 END)
                END AS verdi
            FROM saksbehandling.faser pr
            INNER JOIN felles.indikatorer indikatorer
                ON indikatorer.pk_indikator = pr.indikator
            WHERE pr.sluttmilepaeldato IS NOT NULL
                AND pr.frist IS NOT NULL
                AND pr.indikator NOT LIKE '%avtalt%'
                AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
                AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
            GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
            ORDER BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

# Monthly average Tidsbruk per indicator
monthly_tid = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(pr.sluttmilepaeldato) * 100 + MONTH(pr.sluttmilepaeldato)) AS period,
                AVG(pr.tidsbruk) AS verdi
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikatorer indikatorer
            ON indikatorer.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.tidsbruk IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
        GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
        ORDER BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

# Monthly production balance per indicator (Mottatt - Produsert)
monthly_prod = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) * 100 +
                 MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))) AS period,
                COUNT(CASE WHEN pr.startmilepaeldato IS NOT NULL THEN 1 END)
                - COUNT(CASE WHEN pr.sluttmilepaeldato IS NOT NULL THEN 1 END) AS verdi
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikatorer indikatorer
            ON indikatorer.pk_indikator = pr.indikator
        WHERE pr.indikator NOT LIKE '%avtalt%'
            AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) >= {START_YEAR}
        GROUP BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                         MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
        ORDER BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                         MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
""").toPandas()

print(f"frist:    {monthly_frist['indikator'].nunique()} indikatorer, "
            f"{monthly_frist['period'].nunique()} måneder")
print(f"tidsbruk: {monthly_tid['indikator'].nunique()} indikatorer")
print(f"produksjonsdifferanse: {monthly_prod['indikator'].nunique()} indikatorer")


# =============================================================================
# CELL 3 — EWMA computation
# =============================================================================

def compute_ewma(series, alpha):
    """
    Compute EWMA on a pandas Series.
    pandas ewm() with adjust=False matches the standard recursive formula:
      EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}
    adjust=False is correct for a causal filter — each value depends only
    on current and past observations, not future ones.
    NaN values are skipped and interpolated across.
    """
    return series.ewm(alpha=alpha, adjust=False).mean()


def trend_label(slope, threshold=0.002):
    """
    Convert EWMA slope to a human-readable trend direction.
    threshold: minimum absolute change to count as a trend (default 0.2pp)
    For tidsbruk the threshold should be larger — adjust per metric if needed.
    """
    if pd.isna(slope):
        return "Stabil"
    if slope > threshold:
        return "Stigende"
    if slope < -threshold:
        return "Synkende"
    return "Stabil"


def process_metric(df, maaltall_navn, slope_threshold=0.002):
    """
    Compute EWMA for all indicators in a monthly dataframe.
    Returns list of result dicts ready for output table.
    """
    rows = []
    for indikator, group in df.groupby("indikator"):
        g = group.sort_values("period").copy()

        if len(g) < 3:
            continue

        g["ewma_slow"] = compute_ewma(g["verdi"], ALPHA_SLOW)
        g["ewma_fast"] = compute_ewma(g["verdi"], ALPHA_FAST)

        # Slope: month-on-month change in EWMA
        g["ewma_slope_slow"] = g["ewma_slow"].diff()
        g["ewma_slope_fast"] = g["ewma_fast"].diff()

        # Trend direction based on slow EWMA slope — stable signal for board
        g["trend_retning"] = g["ewma_slope_slow"].apply(
            lambda x: trend_label(x, slope_threshold)
        )

        for _, row in g.iterrows():
            rows.append({
                "indikator":       indikator,
                "maaltall":        maaltall_navn,
                "analyse_dato":    (pd.Timestamp(int(row["period"]) // 100,
                                                 int(row["period"]) % 100, 1)
                                    + pd.offsets.MonthEnd(0)).date(),
                "verdi":           round(float(row["verdi"]), 4)
                                   if pd.notna(row["verdi"]) else None,
                "ewma_sakte":      round(float(row["ewma_slow"]), 4)
                                   if pd.notna(row["ewma_slow"]) else None,
                "ewma_rask":       round(float(row["ewma_fast"]), 4)
                                   if pd.notna(row["ewma_fast"]) else None,
                "ewma_helning_sakte": round(float(row["ewma_slope_slow"]), 4)
                                   if pd.notna(row["ewma_slope_slow"]) else None,
                "ewma_helning_rask": round(float(row["ewma_slope_fast"]), 4)
                                   if pd.notna(row["ewma_slope_fast"]) else None,
                "trendretning":    row["trend_retning"],
                "kjoert_tidspunkt": datetime.now(),
                "id":              BATCH_ID,
            })
    return rows


# =============================================================================
# CELL 4 — Run for all three metrics
# =============================================================================
# Note on slope thresholds:
# frist_pct:  values are 0-1, threshold 0.002 = 0.2pp change per month
# tidsbruk:   values are days, threshold 0.5 = half day change per month
# prod_diff:  values are case counts, threshold 5 = 5 cases per month

frist_rows = process_metric(monthly_frist, "Fristprosent", slope_threshold=0.002)
tid_rows   = process_metric(monthly_tid,   "Behandlingstid", slope_threshold=0.5)
prod_rows  = process_metric(monthly_prod,  "Produksjonsdifferanse", slope_threshold=5.0)

all_rows = frist_rows + tid_rows + prod_rows

print(f"EWMA-rader beregnet: {len(all_rows):,}")
print(f"  Fristprosent: {len(frist_rows):,} rader")
print(f"  Behandlingstid: {len(tid_rows):,} rader")
print(f"  Produksjonsdifferanse: {len(prod_rows):,} rader")

# Trend summary for most recent period
df = pd.DataFrame(all_rows)
latest = df[df["analyse_dato"] == df["analyse_dato"].max()]
print(f"\n=== TRENDSAMMENDRAG — SISTE PERIODE {df['analyse_dato'].max()} ===")
print(latest.groupby(["maaltall", "trendretning"]) ["indikator"].count()
    .unstack(fill_value=0).to_string())


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

if not all_rows:
    print("Ingen EWMA-resultater å skrive.")
else:
    results_spark = spark.createDataFrame(df)

    # Full overwrite — EWMA recalculated from scratch each run since
    # it depends on the full history (each value depends on all prior values)
    results_spark.write.mode("overwrite").saveAsTable("ewma_analyse")

    print(f"ewma_analyse skrevet: {len(all_rows):,} rader")

    # Active trends for current period
    spark.sql(f"""
            SELECT indikator, maaltall,
                            ROUND(verdi,       3) AS verdi,
                            ROUND(ewma_sakte,  3) AS ewma_sakte,
                            ROUND(ewma_rask,   3) AS ewma_rask,
                            trendretning
            FROM ewma_analyse
            WHERE analyse_dato = (SELECT MAX(analyse_dato) FROM ewma_analyse)
                AND maaltall = 'Fristprosent'
                AND trendretning != 'Stabil'
            ORDER BY trendretning, indikator
    """).show(30, truncate=False)


# =============================================================================
# CELL 6 — Power BI visual guidance and DAX measures
# =============================================================================
#
# OUTPUT TABLE → VISUALS
#
# ewma_analyse inneholder rå månedlige verdier og utjevnede EWMA-linjer
# for alle tre måltall. Én rad per indikator per måned per måltall.
#
# LINE CHART — Raw + EWMA trend overlay (primary visual, board report)
#   X axis:  analyse_dato (Regnskapsperiode)
#   Lines:
#     Thin line, low opacity: verdi — faktisk månedlig verdi
#     Bold line:              ewma_sakte — utjevnet trend (alfa=0.1)
#     Optional dashed line:   ewma_rask — raskere signal (alfa=0.3)
#   Filter:  maaltall = 'Fristprosent' for styreoversikt
#            maaltall-utvalg for virksomhetsoppfølging
#            indikator slicer — one chart per Fagområde as small multiples
#   Ref line: Frist målverdi (constant from alert_config) — horizontal
#   Reading:  EWMA line bending downward toward the reference line = risk
#             building. EWMA line flat or rising = stable/improving.
#             The distance between raw line and EWMA line shows how much
#             monthly variance there is — wide gap = volatile indicator.
#
# LINE CHART — Behandlingstid trend (governance report)
#   Same pattern but maaltall = 'Behandlingstid'
#   No reference line needed — governance team reads direction
#   EWMA slope tells you if processing is getting faster or slower
#
# LINE CHART — Production balance trend (governance report)
#   maaltall = 'Produksjonsdifferanse'
#   Ref line: zero — EWMA above zero = intake outpacing production
#   EWMA crossing zero from below = backlog starting to build
#
# INDICATOR CARD — Current trend direction
#   Show trendretning for most recent analyse_dato
#   Conditional format: Synkende → red, Stigende → green, Stabil → neutral
#   Use ewma_sakte trend for board, ewma_rask for governance team
#
# DAX MEASURES — add to ewma_analyse table in semantic model

# EWMA Sakte fristprosent =
# CALCULATE(
#     MAX(ewma_analyse[ewma_sakte]),
#     ewma_analyse[maaltall] = "Fristprosent"
# )

# EWMA Rask fristprosent =
# CALCULATE(
#     MAX(ewma_analyse[ewma_rask]),
#     ewma_analyse[maaltall] = "Fristprosent"
# )

# EWMA Trend retning =
# CALCULATE(
#     MAX(ewma_analyse[trendretning]),
#     ewma_analyse[analyse_dato] = MAX(ewma_analyse[analyse_dato])
# )

# EWMA Trend verdi =
# -- Numeric version for conditional formatting
# -- 1 = Stigende (green), -1 = Synkende (red), 0 = Stabil (neutral)
# VAR Retning = [EWMA Trend retning]
# RETURN
#     SWITCH(Retning, "Stigende", 1, "Synkende", -1, 0)

# EWMA Behandlingstid =
# CALCULATE(
#     MAX(ewma_analyse[ewma_sakte]),
#     ewma_analyse[maaltall] = "Behandlingstid"
# )

# EWMA Produksjon differanse =
# CALCULATE(
#     MAX(ewma_analyse[ewma_sakte]),
#     ewma_analyse[maaltall] = "Produksjonsdifferanse"
# )
#
# NOTE ON ALPHA CHOICE FOR BOARD VS GOVERNANCE:
# Board report: always use ewma_sakte (alpha=0.1). Stable line, clear direction,
#               not distracted by single-month noise. Changes slowly and
#               deliberately — appropriate for monthly meeting cadence.
# Governance team: use ewma_rask (alpha=0.3) for early warning. Picks up
#                  trend changes 2-3 months sooner than ewma_sakte. Accept
#                  more false signals as the tradeoff for earlier detection.
