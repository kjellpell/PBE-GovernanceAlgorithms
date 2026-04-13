# =============================================================================
# NB_07_SIGNALS.py
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
START_YEAR    = 2010  # exclude data before this year — adjust if older data is reliable


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS cusum_results (
    indikator       STRING      NOT NULL,
    metrikk         STRING      NOT NULL,
    granularitet    STRING      NOT NULL,
    periode         INT         NOT NULL,
    verdi           DOUBLE,
    cusum_pos       DOUBLE,
    cusum_neg       DOUBLE,
    signal          BOOLEAN     NOT NULL,
    signal_retning  STRING,
    computed_at     TIMESTAMP   NOT NULL,
    batch_id        STRING      NOT NULL
)
USING DELTA
COMMENT 'CUSUM drift detection per indicator per metric. signal=True means statistically significant persistent drift detected. signal_retning: OPP=improving, NED=deteriorating.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS changepoint_results (
    indikator           STRING      NOT NULL,
    metrikk             STRING      NOT NULL,
    granularitet        STRING      NOT NULL,
    changepoint_periode INT         NOT NULL,
    mean_before         DOUBLE,
    mean_after          DOUBLE,
    shift_magnitude     DOUBLE,
    shift_retning       STRING,
    n_obs_before        INT,
    n_obs_after         INT,
    computed_at         TIMESTAMP   NOT NULL,
    batch_id            STRING      NOT NULL
)
USING DELTA
COMMENT 'Detected structural changepoints per indicator per metric. shift_magnitude = mean_after - mean_before. shift_retning: OPP or NED.'
""")

print("Output tables ready")


# =============================================================================
# CELL 2 — Load data
# =============================================================================

# Monthly frist% per indicator
monthly_frist = spark.sql(f"""
    SELECT
        pr.Indikator,
        pe.Regnskapsperiode                                         AS periode,
        CASE
            WHEN COUNT(CASE WHEN pr.aggregert_frist IS NOT NULL
                            THEN 1 END) = 0 THEN NULL
            ELSE COUNT(CASE WHEN pr.innenfor_frist = 1
                            THEN 1 END) * 1.0
                 / COUNT(CASE WHEN pr.aggregert_frist IS NOT NULL
                              THEN 1 END)
        END                                                         AS verdi
    FROM Prosesser pr
    INNER JOIN Periode pe ON pr.Sluttdato = pe.Dato
    WHERE pr.Sluttdato      IS NOT NULL
      AND pr.aggregert_frist IS NOT NULL
      AND pr.Indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.Sluttdato) >= {START_YEAR}
    GROUP BY pr.Indikator, pe.Regnskapsperiode
    ORDER BY pr.Indikator, pe.Regnskapsperiode
""").toPandas()

# Monthly average Tidsbruk per indicator
monthly_tid = spark.sql(f"""
    SELECT
        pr.Indikator,
        pe.Regnskapsperiode AS periode,
        AVG(pr.Tidsbruk)    AS verdi
    FROM Prosesser pr
    INNER JOIN Periode pe ON pr.Sluttdato = pe.Dato
    WHERE pr.Sluttdato IS NOT NULL
      AND pr.Tidsbruk  IS NOT NULL
      AND pr.Indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.Sluttdato) >= {START_YEAR}
    GROUP BY pr.Indikator, pe.Regnskapsperiode
    ORDER BY pr.Indikator, pe.Regnskapsperiode
""").toPandas()

# Monthly production balance per indicator
monthly_prod = spark.sql(f"""
    SELECT
        pr.Indikator,
        pe.Regnskapsperiode                     AS periode,
        COUNT(CASE WHEN pr.tidligste_startmilepael_dato IS NOT NULL
                   THEN 1 END)
        - COUNT(CASE WHEN pr.Sluttdato IS NOT NULL
                     THEN 1 END)                AS verdi
    FROM Prosesser pr
    INNER JOIN Periode pe
        ON COALESCE(pr.Sluttdato, pr.tidligste_startmilepael_dato) = pe.Dato
    WHERE pr.Indikator NOT LIKE '%avtalt%'
      AND YEAR(COALESCE(pr.Sluttdato, pr.tidligste_startmilepael_dato)) >= {START_YEAR}
    GROUP BY pr.Indikator, pe.Regnskapsperiode
    ORDER BY pr.Indikator, pe.Regnskapsperiode
""").toPandas()

# Weekly frist% per indicator
weekly_frist = spark.sql(f"""
    SELECT
        pr.Indikator,
        pe.Ukenummer                                                AS periode,
        CASE
            WHEN COUNT(CASE WHEN pr.aggregert_frist IS NOT NULL
                            THEN 1 END) = 0 THEN NULL
            ELSE COUNT(CASE WHEN pr.innenfor_frist = 1
                            THEN 1 END) * 1.0
                 / COUNT(CASE WHEN pr.aggregert_frist IS NOT NULL
                              THEN 1 END)
        END                                                         AS verdi
    FROM Prosesser pr
    INNER JOIN Periode pe ON pr.Sluttdato = pe.Dato
    WHERE pr.Sluttdato      IS NOT NULL
      AND pr.aggregert_frist IS NOT NULL
      AND pr.Indikator NOT LIKE '%avtalt%'
      AND YEAR(pr.Sluttdato) >= {START_YEAR}
    GROUP BY pr.Indikator, pe.Ukenummer
    ORDER BY pr.Indikator, pe.Ukenummer
""").toPandas()

# Weekly production balance
weekly_prod = spark.sql(f"""
    SELECT
        pr.Indikator,
        pe.Ukenummer                            AS periode,
        COUNT(CASE WHEN pr.tidligste_startmilepael_dato IS NOT NULL
                   THEN 1 END)
        - COUNT(CASE WHEN pr.Sluttdato IS NOT NULL
                     THEN 1 END)                AS verdi
    FROM Prosesser pr
    INNER JOIN Periode pe
        ON COALESCE(pr.Sluttdato, pr.tidligste_startmilepael_dato) = pe.Dato
    WHERE pr.Indikator NOT LIKE '%avtalt%'
      AND YEAR(COALESCE(pr.Sluttdato, pr.tidligste_startmilepael_dato)) >= {START_YEAR}
    GROUP BY pr.Indikator, pe.Ukenummer
    ORDER BY pr.Indikator, pe.Ukenummer
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
        "cusum_pos":      cusum_pos,
        "cusum_neg":      cusum_neg,
        "signal":         signal,
        "signal_retning": retning
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


def extract_changepoint_stats(series, breakpoints, periode_index):
    """
    For each detected breakpoint, compute before/after mean and shift.
    Returns list of dicts.
    """
    values = series.dropna().values
    periods = serie_clean_index = series.dropna().index.tolist()
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
            "changepoint_periode": int(periods[bp]) if bp < len(periods) else None,
            "mean_before":         round(mean_before, 4),
            "mean_after":          round(mean_after, 4),
            "shift_magnitude":     round(shift, 4),
            "shift_retning":       "OPP" if shift > 0 else "NED",
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
            .sort_values("periode")
            .set_index("periode")["verdi"]
        )

        if len(ind_data.dropna()) < min_obs:
            continue

        # ── CUSUM ──────────────────────────────────────────────────
        cusum = run_cusum(ind_data)
        if cusum is not None:
            for idx, row in cusum.iterrows():
                cusum_rows.append({
                    "indikator":      indikator,
                    "metrikk":        metrikk,
                    "granularitet":   granularitet,
                    "periode":        int(idx),
                    "verdi":          float(ind_data[idx]) if idx in ind_data else None,
                    "cusum_pos":      round(float(row["cusum_pos"]), 4),
                    "cusum_neg":      round(float(row["cusum_neg"]), 4),
                    "signal":         bool(row["signal"]),
                    "signal_retning": row["signal_retning"],
                    "computed_at":    datetime.now(),
                    "batch_id":       BATCH_ID,
                })

        # ── Changepoint (monthly only for stability) ───────────────
        if granularitet == "monthly":
            breakpoints = run_changepoint(ind_data, granularitet)
            for cp in extract_changepoint_stats(ind_data, breakpoints, None):
                changepoint_rows.append({
                    "indikator":   indikator,
                    "metrikk":     metrikk,
                    "granularitet": granularitet,
                    **cp,
                    "computed_at": datetime.now(),
                    "batch_id":    BATCH_ID,
                })

        # Weekly changepoints — separate pass
        if granularitet == "weekly":
            breakpoints = run_changepoint(ind_data, granularitet)
            for cp in extract_changepoint_stats(ind_data, breakpoints, None):
                changepoint_rows.append({
                    "indikator":   indikator,
                    "metrikk":     metrikk,
                    "granularitet": granularitet,
                    **cp,
                    "computed_at": datetime.now(),
                    "batch_id":    BATCH_ID,
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
    spark.sql(f"DELETE FROM cusum_results WHERE batch_id != '{BATCH_ID}'")
    cusum_spark.write.mode("append").saveAsTable("cusum_results")
    print(f"cusum_results written: {len(cusum_rows)} rows")

if changepoint_rows:
    cp_df = pd.DataFrame(changepoint_rows)
    cp_spark = spark.createDataFrame(cp_df)
    spark.sql(f"DELETE FROM changepoint_results WHERE batch_id != '{BATCH_ID}'")
    cp_spark.write.mode("append").saveAsTable("changepoint_results")
    print(f"changepoint_results written: {len(changepoint_rows)} rows")

# Summary — active signals
if cusum_rows:
    spark.sql(f"""
        SELECT indikator, metrikk, granularitet,
               MAX(periode) AS siste_signal_periode,
               MAX(signal_retning) AS retning
        FROM cusum_results
        WHERE signal = TRUE
          AND batch_id = '{BATCH_ID}'
        GROUP BY indikator, metrikk, granularitet
        ORDER BY metrikk, indikator
    """).show(50, truncate=False)

if changepoint_rows:
    spark.sql(f"""
        SELECT indikator, metrikk, granularitet,
               changepoint_periode,
               ROUND(mean_before, 3) AS før,
               ROUND(mean_after,  3) AS etter,
               ROUND(shift_magnitude, 3) AS endring,
               shift_retning
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
#
# Filters to most recent period per indicator for use in summary visuals.

# Har aktiv CUSUM signal =
# VAR SistePeriode =
#     CALCULATE(
#         MAX(cusum_results[periode]),
#         ALLEXCEPT(cusum_results, cusum_results[indikator], cusum_results[metrikk])
#     )
# RETURN
#     CALCULATE(
#         MAX(cusum_results[signal]),
#         cusum_results[periode] = SistePeriode
#     ) = TRUE()

# Antall aktive signaler =
# CALCULATE(
#     DISTINCTCOUNT(cusum_results[indikator]),
#     cusum_results[signal] = TRUE(),
#     cusum_results[periode] = MAX(cusum_results[periode])
# )

# DAX MEASURES — add to changepoint_results table

# Siste endringspunkt periode =
# CALCULATE(
#     MAX(changepoint_results[changepoint_periode]),
#     ALLEXCEPT(changepoint_results, changepoint_results[indikator],
#               changepoint_results[metrikk])
# )

# Endringspunkt størrelse =
# CALCULATE(
#     MAX(changepoint_results[shift_magnitude]),
#     changepoint_results[changepoint_periode] = [Siste endringspunkt periode]
# )
