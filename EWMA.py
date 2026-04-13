
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
# Output table: ewma_results
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
from datetime import datetime

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
CREATE TABLE IF NOT EXISTS ewma_results (
    indicator       STRING      NOT NULL,
    metric          STRING      NOT NULL,  -- frist_pct / tidsbruk / prod_diff
    period          INT         NOT NULL,  -- YYYYMM
    value_raw       DOUBLE,                -- raw monthly value
    ewma_slow       DOUBLE,                -- alpha=0.1 smoothed value
    ewma_fast       DOUBLE,                -- alpha=0.3 smoothed value
    ewma_slope_slow DOUBLE,                -- month-on-month change in ewma_slow
    ewma_slope_fast DOUBLE,                -- month-on-month change in ewma_fast
    trend_direction STRING,                -- Rising / Falling / Stable
    computed_at     TIMESTAMP   NOT NULL,
    batch_id        STRING      NOT NULL
)
USING DELTA
COMMENT 'EWMA smoothed trend lines per indicator and metric. Used for board and governance report trend charts.'
""")

print("ewma_results table ready")


# =============================================================================
# CELL 2 — Load monthly data
# =============================================================================

# Monthly frist% per indicator
monthly_frist = spark.sql(f"""
        SELECT
                pr.Indikator,
                (YEAR(pr.seneste_stoppmilepael_dato) * 100 + MONTH(pr.seneste_stoppmilepael_dato)) AS period,
                CASE
                        WHEN COUNT(CASE WHEN pr.Frist IS NOT NULL THEN 1 END) = 0 THEN NULL
                        ELSE COUNT(CASE WHEN pr.innenfor_frist = 1 THEN 1 END) * 1.0
                                 / COUNT(CASE WHEN pr.Frist IS NOT NULL THEN 1 END)
                END AS verdi
        FROM Prosesser pr
        WHERE pr.seneste_stoppmilepael_dato IS NOT NULL
            AND pr.Frist IS NOT NULL
            AND pr.Indikator NOT LIKE '%avtalt%'
            AND YEAR(pr.seneste_stoppmilepael_dato) >= {START_YEAR}
        GROUP BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), MONTH(pr.seneste_stoppmilepael_dato)
        ORDER BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), MONTH(pr.seneste_stoppmilepael_dato)
""").toPandas()

# Monthly average Tidsbruk per indicator
monthly_tid = spark.sql(f"""
        SELECT
                pr.Indikator,
                (YEAR(pr.seneste_stoppmilepael_dato) * 100 + MONTH(pr.seneste_stoppmilepael_dato)) AS period,
                AVG(pr.Tidsbruk) AS verdi
        FROM Prosesser pr
        WHERE pr.seneste_stoppmilepael_dato IS NOT NULL
            AND pr.Tidsbruk IS NOT NULL
            AND pr.Indikator NOT LIKE '%avtalt%'
            AND YEAR(pr.seneste_stoppmilepael_dato) >= {START_YEAR}
        GROUP BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), MONTH(pr.seneste_stoppmilepael_dato)
        ORDER BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), MONTH(pr.seneste_stoppmilepael_dato)
""").toPandas()

# Monthly production balance per indicator (Mottatt - Produsert)
monthly_prod = spark.sql(f"""
        SELECT
                pr.Indikator,
                (YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)) * 100 +
                 MONTH(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato))) AS period,
                COUNT(CASE WHEN pr.tidligste_startmilepael_dato IS NOT NULL THEN 1 END)
                - COUNT(CASE WHEN pr.seneste_stoppmilepael_dato IS NOT NULL THEN 1 END) AS verdi
        FROM Prosesser pr
        WHERE pr.Indikator NOT LIKE '%avtalt%'
            AND YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)) >= {START_YEAR}
        GROUP BY pr.Indikator, YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)),
                         MONTH(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato))
        ORDER BY pr.Indikator, YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)),
                         MONTH(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato))
""").toPandas()

print(f"frist:    {monthly_frist['Indikator'].nunique()} indicators, "
            f"{monthly_frist['period'].nunique()} months")
print(f"tidsbruk: {monthly_tid['Indikator'].nunique()} indicators")
print(f"prod_diff:{monthly_prod['Indikator'].nunique()} indicators")


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


def process_metric(df, metrikk_navn, slope_threshold=0.002):
    """
    Compute EWMA for all indicators in a monthly dataframe.
    Returns list of result dicts ready for output table.
    """
    rows = []
    for indikator, group in df.groupby("Indikator"):
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
                "indicator":       indikator,
                "metric":          metrikk_navn,
                "period":          int(row["period"]),
                "value_raw":       round(float(row["verdi"]), 4)
                                   if pd.notna(row["verdi"]) else None,
                "ewma_slow":       round(float(row["ewma_slow"]), 4)
                                   if pd.notna(row["ewma_slow"]) else None,
                "ewma_fast":       round(float(row["ewma_fast"]), 4)
                                   if pd.notna(row["ewma_fast"]) else None,
                "ewma_slope_slow": round(float(row["ewma_slope_slow"]), 4)
                                   if pd.notna(row["ewma_slope_slow"]) else None,
                "ewma_slope_fast": round(float(row["ewma_slope_fast"]), 4)
                                   if pd.notna(row["ewma_slope_fast"]) else None,
                "trend_direction": row["trend_retning"],
            })
    return rows


# =============================================================================
# CELL 4 — Run for all three metrics
# =============================================================================
# Note on slope thresholds:
# frist_pct:  values are 0-1, threshold 0.002 = 0.2pp change per month
# tidsbruk:   values are days, threshold 0.5 = half day change per month
# prod_diff:  values are case counts, threshold 5 = 5 cases per month

frist_rows = process_metric(monthly_frist, "frist_pct",  slope_threshold=0.002)
tid_rows   = process_metric(monthly_tid,   "tidsbruk",   slope_threshold=0.5)
prod_rows  = process_metric(monthly_prod,  "prod_diff",  slope_threshold=5.0)

all_rows = frist_rows + tid_rows + prod_rows

print(f"EWMA rows computed: {len(all_rows):,}")
print(f"  frist_pct:  {len(frist_rows):,} rows")
print(f"  tidsbruk:   {len(tid_rows):,} rows")
print(f"  prod_diff:  {len(prod_rows):,} rows")

# Trend summary for most recent period
df = pd.DataFrame(all_rows)
latest = df[df["period"] == df["period"].max()]
print(f"\n=== TREND SUMMARY — LATEST PERIOD {df['period'].max()} ===")
print(latest.groupby(["metric", "trend_direction"]) ["indicator"].count()
    .unstack(fill_value=0).to_string())


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

if not all_rows:
    print("No EWMA results to write.")
else:
    results_spark = spark.createDataFrame(df)

    # Full overwrite — EWMA recalculated from scratch each run since
    # it depends on the full history (each value depends on all prior values)
    results_spark.write.mode("overwrite").saveAsTable("ewma_results")

    print(f"ewma_results written: {len(all_rows):,} rows")

    # Active trends for current period
    spark.sql(f"""
            SELECT indicator, metric,
                            ROUND(value_raw,   3) AS value,
                            ROUND(ewma_slow,   3) AS ewma_slow,
                            ROUND(ewma_fast,   3) AS ewma_fast,
                            trend_direction
            FROM ewma_results
            WHERE period = (SELECT MAX(period) FROM ewma_results)
                AND metric = 'frist_pct'
                AND trend_direction != 'Stabil'
            ORDER BY trend_direction, indicator
    """).show(30, truncate=False)


# =============================================================================
# CELL 6 — Power BI visual guidance and DAX measures
# =============================================================================
#
# OUTPUT TABLE → VISUALS
#
# ewma_results contains raw monthly values AND smoothed EWMA lines
# for all three metrics. One row per indicator per month per metric.
#
# LINE CHART — Raw + EWMA trend overlay (primary visual, board report)
#   X axis:  periode (Regnskapsperiode)
#   Lines:
#     Thin line, low opacity: verdi_raa — actual monthly value
#                             Shows real variance, confirms EWMA direction
#     Bold line:              ewma_slow — smoothed trend (alpha=0.1)
#                             This is the line the board reads
#     Optional dashed line:   ewma_fast — faster signal (alpha=0.3)
#                             Add for governance team view only
#   Filter:  metrikk = 'frist_pct' for board report
#            metrikk slicer for governance team
#            indikator slicer — one chart per Fagområde as small multiples
#   Ref line: Frist målverdi (constant from alert_config) — horizontal
#   Reading:  EWMA line bending downward toward the reference line = risk
#             building. EWMA line flat or rising = stable/improving.
#             The distance between raw line and EWMA line shows how much
#             monthly variance there is — wide gap = volatile indicator.
#
# LINE CHART — Behandlingstid trend (governance report)
#   Same pattern but metrikk = 'tidsbruk'
#   No reference line needed — governance team reads direction
#   EWMA slope tells you if processing is getting faster or slower
#
# LINE CHART — Production balance trend (governance report)
#   metrikk = 'prod_diff'
#   Ref line: zero — EWMA above zero = intake outpacing production
#   EWMA crossing zero from below = backlog starting to build
#
# INDICATOR CARD — Current trend direction
#   Show trend_retning for most recent periode
#   Conditional format: Synkende → red, Stigende → green, Stabil → neutral
#   Use ewma_slow trend for board, ewma_fast for governance team
#
# DAX MEASURES — add to ewma_results table in semantic model

# EWMA Slow frist pct =
# CALCULATE(
#     MAX(ewma_results[ewma_slow]),
#     ewma_results[metric] = "frist_pct"
# )

# EWMA Fast frist pct =
# CALCULATE(
#     MAX(ewma_results[ewma_fast]),
#     ewma_results[metric] = "frist_pct"
# )

# EWMA Trend retning =
# CALCULATE(
#     MAX(ewma_results[trend_direction]),
#     ewma_results[period] = MAX(ewma_results[period])
# )

# EWMA Trend verdi =
# -- Numeric version for conditional formatting
# -- 1 = Stigende (green), -1 = Synkende (red), 0 = Stabil (neutral)
# VAR Retning = [EWMA Trend retning]
# RETURN
#     SWITCH(Retning, "Stigende", 1, "Synkende", -1, 0)

# EWMA Behandlingstid =
# CALCULATE(
#     MAX(ewma_results[ewma_slow]),
#     ewma_results[metric] = "tidsbruk"
# )

# EWMA Produksjon differanse =
# CALCULATE(
#     MAX(ewma_results[ewma_slow]),
#     ewma_results[metric] = "prod_diff"
# )
#
# NOTE ON ALPHA CHOICE FOR BOARD VS GOVERNANCE:
# Board report: always use ewma_slow (alpha=0.1). Stable line, clear direction,
#               not distracted by single-month noise. Changes slowly and
#               deliberately — appropriate for monthly meeting cadence.
# Governance team: use ewma_fast (alpha=0.3) for early warning. Picks up
#                  trend changes 2-3 months sooner than ewma_slow. Accept
#                  more false signals as the tradeoff for earlier detection.
