# =============================================================================
# CUSUM drift detection and changepoint detection per indicator.
# Runs nightly after main data pipeline.
#
# CUSUM:
#   Detects small persistent shifts in frist%, behandlingstid,
#   and produksjonsdifferanse. More sensitive than threshold alerts
#   for gradual deterioration. Runs on both monthly and weekly series.
#
# Changepoint (PELT):
#   Identifies the exact period where a structural shift occurred.
#   Output marks the changepoint on the time series chart and shows
#   before/after means. Runs on monthly series only for stability,
#   weekly for early detection.
#
# Output tables:
#   cusum_results       — running CUSUM values and signal flags
#   changepoint_results — detected changepoints with before/after means
#
# Schedule: nightly, after main data pipeline.
# Requires: pip install ruptures --break-system-packages
# Minimum history: 24 monthly / 52 weekly observations per indicator.
# =============================================================================

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from datetime import datetime

spark = SparkSession.builder.getOrCreate()
BATCH_ID      = datetime.now().strftime("%Y%m%dT%H%M%S")
MIN_MONTHLY   = 24
MIN_WEEKLY    = 52
CUSUM_K       = 0.5   # allowance parameter — half sigma is standard
CUSUM_H       = 5.0   # decision threshold — 5 sigma cumulative
START_YEAR    = 2015  # exclude data before this year — adjust if older data is reliable


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS cusum_results (
    indicator       STRING      NOT NULL,
    metric          STRING      NOT NULL,
    granularity     STRING      NOT NULL,
    period          INT         NOT NULL,
    value           DOUBLE,
    cusum_pos       DOUBLE,
    cusum_neg       DOUBLE,
    signal          BOOLEAN     NOT NULL,
    signal_direction STRING,
    computed_at     TIMESTAMP   NOT NULL,
    batch_id        STRING      NOT NULL
)
USING DELTA
COMMENT 'CUSUM drift detection per indicator per metric. signal=True means statistically significant persistent drift detected. signal_direction: OPP=improving, NED=deteriorating.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS changepoint_results (
    indicator           STRING      NOT NULL,
    metric              STRING      NOT NULL,
    granularity         STRING      NOT NULL,
    changepoint_period  INT         NOT NULL,
    mean_before         DOUBLE,
    mean_after          DOUBLE,
    shift_magnitude     DOUBLE,
    shift_direction     STRING,
    n_obs_before        INT,
    n_obs_after         INT,
    computed_at         TIMESTAMP   NOT NULL,
    batch_id            STRING      NOT NULL
)
USING DELTA
COMMENT 'Detected structural changepoints per indicator per metric. shift_magnitude = mean_after - mean_before. shift_direction: OPP or NED.'
""")

print("Output tables ready")


# =============================================================================
# CELL 2 — Load data
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
      AND YEAR(pr.seneste_stoppmilepael_dato) >= {START_YEAR}
    GROUP BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), MONTH(pr.seneste_stoppmilepael_dato)
    ORDER BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), MONTH(pr.seneste_stoppmilepael_dato)
""").toPandas()

# Monthly production balance per indicator
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

# Weekly frist% per indicator
weekly_frist = spark.sql(f"""
    SELECT
        pr.Indikator,
        (YEAR(pr.seneste_stoppmilepael_dato) * 100
         + WEEKOFYEAR(pr.seneste_stoppmilepael_dato))               AS period,
        CASE
            WHEN COUNT(CASE WHEN pr.Frist IS NOT NULL
                            THEN 1 END) = 0 THEN NULL
            ELSE COUNT(CASE WHEN pr.innenfor_frist = 1
                            THEN 1 END) * 1.0
                 / COUNT(CASE WHEN pr.Frist IS NOT NULL
                              THEN 1 END)
        END                                                         AS verdi
    FROM Prosesser pr
    WHERE pr.seneste_stoppmilepael_dato IS NOT NULL
      AND pr.Frist IS NOT NULL
      AND pr.Indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.seneste_stoppmilepael_dato) >= {START_YEAR}
    GROUP BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), WEEKOFYEAR(pr.seneste_stoppmilepael_dato)
    ORDER BY pr.Indikator, YEAR(pr.seneste_stoppmilepael_dato), WEEKOFYEAR(pr.seneste_stoppmilepael_dato)
""").toPandas()

# Weekly production balance
weekly_prod = spark.sql(f"""
        SELECT
                pr.Indikator,
                (YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)) * 100 +
                 WEEKOFYEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato))) AS period,
                COUNT(CASE WHEN pr.tidligste_startmilepael_dato IS NOT NULL THEN 1 END)
                - COUNT(CASE WHEN pr.seneste_stoppmilepael_dato IS NOT NULL THEN 1 END) AS verdi
        FROM Prosesser pr
        WHERE pr.Indikator NOT LIKE '%avtalt%'
            AND YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)) >= {START_YEAR}
        GROUP BY pr.Indikator, YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)),
                         WEEKOFYEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato))
        ORDER BY pr.Indikator, YEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato)),
                         WEEKOFYEAR(COALESCE(pr.seneste_stoppmilepael_dato, pr.tidligste_startmilepael_dato))
""").toPandas()

print("Data loaded")
print(f"  Monthly frist: {monthly_frist['Indikator'].nunique()} indicators")
print(f"  Weekly frist:  {weekly_frist['Indikator'].nunique()} indicators")


# =============================================================================
# CELL 3 — CUSUM implementation
# =============================================================================

def run_cusum(series, k=CUSUM_K, h=CUSUM_H):
    """
    Two-sided CUSUM on a standardised series.
    k = allowance (typically 0.5 * expected shift in sigma units)
    h = decision threshold (typically 4-5)

    Returns DataFrame with cusum_pos, cusum_neg, signal, signal_retning.
    """
    if len(series) < 8:
        return None

    values = series.dropna().values
    if len(values) < 8:
        return None

    mu    = np.mean(values)
    sigma = np.std(values)
    if sigma == 0:
        return None

    standardised = (values - mu) / sigma

    cusum_pos = np.zeros(len(standardised))
    cusum_neg = np.zeros(len(standardised))

    for i in range(1, len(standardised)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + standardised[i] - k)
        cusum_neg[i] = max(0, cusum_neg[i-1] - standardised[i] - k)

    signal     = (cusum_pos > h) | (cusum_neg > h)
    retning    = np.where(cusum_pos > h, "OPP",
                 np.where(cusum_neg > h, "NED", None))

    return pd.DataFrame({
        "cusum_pos":       cusum_pos,
        "cusum_neg":       cusum_neg,
        "signal":          signal,
        "signal_direction": retning
    }, index=series.dropna().index)


# =============================================================================
# CELL 4 — Changepoint detection (PELT)
# =============================================================================

def run_changepoint(series, granularitet):
    """
    PELT changepoint detection using ruptures library.
    Returns list of changepoint indices, or empty list if none detected
    or insufficient data.
    """
    try:
        import ruptures as rpt
    except ImportError:
        print("ruptures not installed — skipping changepoint detection")
        return []

    values = series.dropna().values
    min_obs = MIN_MONTHLY if granularitet == "monthly" else MIN_WEEKLY

    if len(values) < min_obs:
        return []

    # PELT with RBF cost — detects mean and variance shifts
    algo = rpt.Pelt(model="rbf", min_size=6, jump=1).fit(values)

    try:
        # penalty scales with series length — prevents over-segmentation
        penalty = np.log(len(values)) * np.std(values) ** 2
        breakpoints = algo.predict(pen=penalty)
        # Last breakpoint is always len(values) — remove it
        return [bp for bp in breakpoints if bp < len(values)]
    except Exception:
        return []


def extract_changepoint_stats(series, breakpoints):
    """
    For each detected breakpoint, compute before/after mean and shift.
    Returns list of dicts.
    """
    values = series.dropna().values
    periods = series.dropna().index.tolist()
    results = []

    prev = 0
    for bp in breakpoints:
        before = values[prev:bp]
        after  = values[bp:]

        if len(before) < 3 or len(after) < 3:
            prev = bp
            continue

        mean_before = float(np.mean(before))
        mean_after  = float(np.mean(after))
        shift       = mean_after - mean_before

        results.append({
            "changepoint_period":  int(periods[bp]) if bp < len(periods) else None,
            "mean_before":         round(mean_before, 4),
            "mean_after":          round(mean_after, 4),
            "shift_magnitude":     round(shift, 4),
            "shift_direction":     "OPP" if shift > 0 else "NED",
            "n_obs_before":        len(before),
            "n_obs_after":         len(after),
        })
        prev = bp

    return results


# =============================================================================
# CELL 5 — Run on all series
# =============================================================================

# Define what to process
# (label, dataframe, granularitet, min_obs)
series_configs = [
    ("frist_pct",     monthly_frist, "monthly", MIN_MONTHLY),
    ("tidsbruk",      monthly_tid,   "monthly", MIN_MONTHLY),
    ("prod_diff",     monthly_prod,  "monthly", MIN_MONTHLY),
    ("frist_pct",     weekly_frist,  "weekly",  MIN_WEEKLY),
    ("prod_diff",     weekly_prod,   "weekly",  MIN_WEEKLY),
]

cusum_rows       = []
changepoint_rows = []

for metrikk, df, granularitet, min_obs in series_configs:

    for indikator in df["Indikator"].unique():
        ind_data = (
            df[df["Indikator"] == indikator]
            .sort_values("period")
            .set_index("period")["verdi"]
        )

        if len(ind_data.dropna()) < min_obs:
            continue

        # ── CUSUM ──────────────────────────────────────────────────
        cusum = run_cusum(ind_data)
        if cusum is not None:
            for idx, row in cusum.iterrows():
                        cusum_rows.append({
                            "indicator":       indikator,
                            "metric":          metrikk,
                            "granularity":     granularitet,
                            "period":          int(idx),
                            "value":           float(ind_data[idx]) if idx in ind_data else None,
                            "cusum_pos":       round(float(row["cusum_pos"]), 4),
                            "cusum_neg":       round(float(row["cusum_neg"]), 4),
                            "signal":          bool(row["signal"]),
                            "signal_direction": row["signal_direction"],
                            "computed_at":     datetime.now(),
                            "batch_id":        BATCH_ID,
                        })

        # ── Changepoint (monthly only for stability) ───────────────
        if granularitet == "monthly":
            breakpoints = run_changepoint(ind_data, granularitet)
            for cp in extract_changepoint_stats(ind_data, breakpoints):
                        changepoint_rows.append({
                            "indicator":    indikator,
                            "metric":       metrikk,
                            "granularity":  granularitet,
                            **cp,
                            "computed_at":  datetime.now(),
                            "batch_id":     BATCH_ID,
                        })

        # Weekly changepoints — separate pass
        if granularitet == "weekly":
            breakpoints = run_changepoint(ind_data, granularitet)
            for cp in extract_changepoint_stats(ind_data, breakpoints):
                changepoint_rows.append({
                    "indicator":    indikator,
                    "metric":       metrikk,
                    "granularity":  granularitet,
                    **cp,
                    "computed_at":  datetime.now(),
                    "batch_id":     BATCH_ID,
                })

print(f"CUSUM rows computed:       {len(cusum_rows)}")
print(f"Changepoint rows computed: {len(changepoint_rows)}")
print(f"Active CUSUM signals:      "
      f"{sum(1 for r in cusum_rows if r['signal'])}")


# =============================================================================
# CELL 6 — Write to Lakehouse
# =============================================================================

now = datetime.now()

if cusum_rows:
    cusum_df = pd.DataFrame(cusum_rows)
    cusum_spark = spark.createDataFrame(cusum_df)
    cusum_spark.write.mode("overwrite").saveAsTable("cusum_results")
    print(f"cusum_results written: {len(cusum_rows)} rows")

if changepoint_rows:
    cp_df = pd.DataFrame(changepoint_rows)
    cp_spark = spark.createDataFrame(cp_df)
    cp_spark.write.mode("overwrite").saveAsTable("changepoint_results")
    print(f"changepoint_results written: {len(changepoint_rows)} rows")

# Summary — active signals
if cusum_rows:
    spark.sql(f"""
                SELECT indicator, metric, granularity,
                             MAX(period) AS last_signal_period,
                             MAX(signal_direction) AS direction
                FROM cusum_results
                WHERE signal = TRUE
                    AND batch_id = '{BATCH_ID}'
                GROUP BY indicator, metric, granularity
                ORDER BY metric, indicator
    """).show(50, truncate=False)

if changepoint_rows:
    spark.sql(f"""
     SELECT indicator, metric, granularity,
         changepoint_period,
         ROUND(mean_before, 3) AS before,
         ROUND(mean_after,  3) AS after,
         ROUND(shift_magnitude, 3) AS change,
         shift_direction
     FROM changepoint_results
     WHERE batch_id = '{BATCH_ID}'
     ORDER BY ABS(shift_magnitude) DESC
    """).show(50, truncate=False)


# =============================================================================
# CELL 7 — Power BI visual guidance and DAX measures
# =============================================================================
#
# OUTPUT TABLES → VISUALS
#
# cusum_results:
#
#   LINE CHART — CUSUM values over time
#     X axis:  periode (Regnskapsperiode or Ukenummer)
#     Y axis:  cusum_pos (upper line), cusum_neg (lower line, negate for display)
#     Ref line: constant at CUSUM_H threshold (default 5.0) — horizontal line
#     Filter:  indikator slicer, metrikk slicer (frist_pct / tidsbruk / prod_diff)
#              granularitet slicer (monthly / weekly)
#     Colour:  cusum_pos in blue, cusum_neg in red
#     Signal:  conditional format background on data points where signal = TRUE
#              — amber fill so active signals stand out on the line
#     Reading: lines drifting toward the threshold = gradual deterioration
#              building. Line crossing threshold = structural shift confirmed.
#              Lines returning to zero = process stabilised.
#
#   TABLE — Active CUSUM signals
#     Columns: indikator | metrikk | granularitet | signal_retning | periode
#     Filter:  signal = TRUE, most recent periode per indicator
#     Sort:    metrikk, then indikator
#     Purpose: governance team morning check — which indicators have
#              active drift signals right now
#
# changepoint_results:
#
#   LINE CHART with changepoint markers — overlay on existing frist% or
#   behandlingstid time series charts
#     Add a vertical reference line at changepoint_periode
#     Show mean_before as a horizontal segment before the changepoint
#     Show mean_after as a horizontal segment after the changepoint
#     The visual gap between the two horizontal segments = shift_magnitude
#     In Power BI: use a calculated column or measure to draw segments,
#     or use the Analytics pane "average line" filtered to pre/post periods
#
#   TABLE — Detected changepoints
#     Columns: indikator | metrikk | changepoint_periode | mean_before
#              | mean_after | shift_magnitude | shift_retning | granularitet
#     Sort:    ABS(shift_magnitude) DESC — largest shifts first
#     Filter:  granularitet slicer so team can toggle monthly/weekly view
#
# DAX MEASURES — add to cusum_results table in semantic model

# Filters to most recent period per indicator for use in summary visuals.

# Har aktiv CUSUM signal =
# VAR SistePeriode =
#     CALCULATE(
#         MAX(cusum_results[period]),
#         ALLEXCEPT(cusum_results, cusum_results[indicator], cusum_results[metric])
#     )
# RETURN
#     CALCULATE(
#         MAX(cusum_results[signal]),
#         cusum_results[period] = LatestPeriod
#     ) = TRUE()

# Antall aktive signaler =
# CALCULATE(
#     DISTINCTCOUNT(cusum_results[indicator]),
#     cusum_results[signal] = TRUE(),
#     cusum_results[period] = MAX(cusum_results[period])
# )

# DAX MEASURES — add to changepoint_results table

# Siste endringspunkt periode =
# CALCULATE(
#     MAX(changepoint_results[changepoint_period]),
#     ALLEXCEPT(changepoint_results, changepoint_results[indicator],
#               changepoint_results[metric])
# )

# Endringspunkt størrelse =
# CALCULATE(
#     MAX(changepoint_results[shift_magnitude]),
#     changepoint_results[changepoint_period] = [Latest changepoint period]
# )
