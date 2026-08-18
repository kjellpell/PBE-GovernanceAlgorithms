# =============================================================================
# Seasonal YTD ratio extrapolation for year-end frist% projection.
# Runs nightly after main data pipeline.
#
# Method:
#   1. Compute monthly YTD frist% per indicator for all historical years
#   2. At each calendar month, compute the ratio: YTD_at_month / year_end
#   3. Trim best and worst year per seasonal position (handles outliers)
#   4. Apply trimmed mean ratio to current YTD to project year-end
#   5. Confidence interval from trimmed variance across historical years
#
# Output table: frist_prognose
#   One row per indicator per month — actuals for past months,
#   forecast for remaining months of current year.
#
# Schedule: nightly, after main data pipeline.
# Minimum history: 3 years per indicator. Suppressed below that.
# =============================================================================

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID      = datetime.now().strftime("%Y%m%dT%H%M%S")
CURRENT_YEAR  = datetime.now().year
CURRENT_MONTH = datetime.now().month
MIN_YEARS     = 3    # minimum history for reliable seasonal pattern
TRIM_N        = 1    # drop N best and N worst years per seasonal position
START_YEAR    = 2015 # exclude data before this year — adjust if older data is reliable


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS prognoser.frist_ (
    indikator                   STRING      NOT NULL,
    analyse_dato                DATE        NOT NULL,
    type                        STRING      NOT NULL,
    verdi                       DOUBLE,
    nedre_konfidensgrense       DOUBLE,
    oevre_konfidensgrense       DOUBLE,
    prognose_aarsslutt          DOUBLE,
    kjoert_tidspunkt            TIMESTAMP   NOT NULL,
    kjoere_id                   STRING      NOT NULL
)
USING DELTA
COMMENT 'Årssluttprognose for fristprosent per indikator. type=Faktisk for måneder med data og type=Prognose for gjenstående måneder. Konfidensgrensene er 80 prosent og bygger på historisk variasjon i samme sesongposisjon.'
""")

print("prognoser.frist-tabellen er klar")


# =============================================================================
# CELL 2 — Load historical monthly frist% per indicator
# =============================================================================
# Full history — all years, all indicators.

monthly = spark.sql(f"""
    SELECT
        pr.indikator,
        YEAR(pr.sluttmilepaeldato)                 AS aar,
        MONTH(pr.sluttmilepaeldato)                AS mnd,
        COUNT(CASE WHEN pr.innenfor_frist = 1 THEN 1 END)           AS innenfor,
        COUNT(CASE WHEN pr.frist IS NOT NULL THEN 1 END)             AS total
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikatorer
        ON indikatorer.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.frist IS NOT NULL
            AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
        GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
    ORDER BY pr.indikator, aar, mnd
""").toPandas()

print(f"Månedlige data lastet: {len(monthly)} rader, "
      f"{monthly['indikator'].nunique()} indikatorer, "
      f"{monthly['aar'].min()}–{monthly['aar'].max()}")


# =============================================================================
# CELL 3 — Helper functions
# =============================================================================

def compute_ytd(df, indikator, year):
    """
    Compute cumulative YTD frist% for each month of a given year.
    Returns dict {month: ytd_pct} for months with data.
    """
    ind = df[(df["indikator"] == indikator) & (df["aar"] == year)].sort_values("mnd")
    result = {}
    cum_innenfor = 0
    cum_total    = 0
    for _, row in ind.iterrows():
        cum_innenfor += row["innenfor"]
        cum_total    += row["total"]
        if cum_total > 0:
            result[int(row["mnd"])] = cum_innenfor / cum_total
    return result


def seasonal_ratios(df, indikator, current_year, min_years=3, trim_n=1):
    """
    For each calendar month 1-12, compute the trimmed mean and std of
    the ratio: YTD_at_month / year_end across all complete historical years.

    Returns dict {month: {mean_ratio, std_ratio, n_years}} or None if
    insufficient history.
    """
    years = sorted(df[(df["indikator"] == indikator) &
                      (df["aar"] < current_year)]["aar"].unique())

    # Only use complete years — must have data in month 12
    complete_years = []
    for y in years:
        ytd = compute_ytd(df, indikator, y)
        if 12 in ytd:
            complete_years.append((y, ytd))

    if len(complete_years) < min_years:
        return None

    ratios = {m: [] for m in range(1, 13)}

    for year, ytd in complete_years:
        year_end = ytd.get(12)
        if year_end is None or year_end == 0:
            continue
        for m, ytd_val in ytd.items():
            ratios[m].append(ytd_val / year_end)

    result = {}
    for m in range(1, 13):
        vals = ratios[m]
        if len(vals) < min_years:
            continue
        # Trim best and worst N years
        vals_sorted = sorted(vals)
        if len(vals_sorted) > 2 * trim_n:
            trimmed = vals_sorted[trim_n:-trim_n]
        else:
            trimmed = vals_sorted
        result[m] = {
            "mean_ratio": float(np.mean(trimmed)),
            "std_ratio":  float(np.std(trimmed)) if len(trimmed) > 1 else 0.0,
            "n_years":    len(trimmed)
        }

    return result if result else None


def project_year_end(current_ytd, month, ratios, z=1.28):
    """
    Project year-end frist% from current YTD value.
    z=1.28 gives 80% confidence interval (appropriate for governance).

    Returns (year_end_estimate, ci_lower, ci_upper) or (None, None, None).
    """
    if month not in ratios:
        return None, None, None

    r = ratios[month]
    if r["mean_ratio"] == 0:
        return None, None, None

    estimate = min(1.0, max(0.0, current_ytd / r["mean_ratio"]))

    # Propagate uncertainty from ratio variance to year-end estimate
    if r["std_ratio"] > 0:
        # Delta method: var(X/r) ≈ X² * var(r) / r⁴
        std_estimate = current_ytd * r["std_ratio"] / (r["mean_ratio"] ** 2)
        ci_lower = max(0.0, estimate - z * std_estimate)
        ci_upper = min(1.0, estimate + z * std_estimate)
    else:
        ci_lower = estimate
        ci_upper = estimate

    return (
        round(float(estimate), 4),
        round(float(ci_lower), 4),
        round(float(ci_upper), 4)
    )


# =============================================================================
# CELL 4 — Compute projections per indicator
# =============================================================================

results = []
indicators = monthly["indikator"].unique()

for indikator in indicators:

    # Compute seasonal ratios from history
    ratios = seasonal_ratios(
        monthly, indikator, CURRENT_YEAR,
        min_years=MIN_YEARS, trim_n=TRIM_N
    )

    if ratios is None:
        print(f"Skipping {indikator} — insufficient history")
        continue

    # Current year actuals
    current_ytd = compute_ytd(monthly, indikator, CURRENT_YEAR)

    if not current_ytd:
        print(f"Skipping {indikator} — no current year data")
        continue

    # Latest month with data
    latest_month = max(current_ytd.keys())
    latest_ytd   = current_ytd[latest_month]

    # Year-end estimate from latest available YTD
    year_end_est, ci_lo, ci_hi = project_year_end(
        latest_ytd, latest_month, ratios
    )

    # Write actual rows — one per month with data this year
    for mnd, ytd_val in current_ytd.items():
        analyse_dato = (pd.Timestamp(CURRENT_YEAR, mnd, 1) + pd.offsets.MonthEnd(0)).date()
        results.append({
            "indikator":         indikator,
            "analyse_dato":      analyse_dato,
            "type":              "Faktisk",
            "verdi":             round(float(ytd_val), 4),
            "nedre_konfidensgrense": None,
            "oevre_konfidensgrense": None,
            "prognose_aarsslutt": year_end_est,
            "kjoert_tidspunkt":  datetime.now(),
            "kjoere_id":         BATCH_ID,
        })

    # Write forecast rows — remaining months of current year
    for mnd in range(latest_month + 1, 13):
        if mnd not in ratios:
            continue
        # Project forward: expected YTD at month mnd given current trajectory
        # Use ratio at forecast month relative to ratio at current month
        # to estimate what YTD will be at that future month
        ratio_current  = ratios[latest_month]["mean_ratio"]
        ratio_forecast = ratios[mnd]["mean_ratio"]
        if ratio_current == 0:
            continue
        forecast_ytd = latest_ytd * (ratio_forecast / ratio_current)
        _, f_ci_lo, f_ci_hi = project_year_end(forecast_ytd, mnd, ratios)

        analyse_dato = (pd.Timestamp(CURRENT_YEAR, mnd, 1) + pd.offsets.MonthEnd(0)).date()
        results.append({
            "indikator":         indikator,
            "analyse_dato":      analyse_dato,
            "type":              "Prognose",
            "verdi":             round(float(forecast_ytd), 4),
            "nedre_konfidensgrense": f_ci_lo,
            "oevre_konfidensgrense": f_ci_hi,
            "prognose_aarsslutt": year_end_est,
            "kjoert_tidspunkt":  datetime.now(),
            "kjoere_id":         BATCH_ID,
        })

print(f"\nProjection rows computed: {len(results)}")
print(f"Indicators projected: {len(set(r['indikator'] for r in results))}")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

if not results:
    print("Ingen prognoseresultater å skrive.")
else:
    df = pd.DataFrame(results)
    results_spark = spark.createDataFrame(df)

    # Idempotent — delete current year rows before inserting
    spark.sql(f"""
            DELETE FROM frist_prognose
            WHERE analyse_dato >= '{CURRENT_YEAR}-01-31'
                AND analyse_dato <= '{CURRENT_YEAR}-12-31'
    """)

    results_spark.write.mode("append").saveAsTable("frist_prognose")

    print(f"frist_prognose skrevet: {len(results)} rader")

    # Summary — year-end estimates for current indicators
    spark.sql(f"""
        SELECT
            indikator,
            MAX_BY(
                CASE WHEN type = 'Faktisk' THEN verdi END,
                CASE WHEN type = 'Faktisk' THEN analyse_dato END
            )                                                       AS verdi_hittil,
            MAX(prognose_aarsslutt)                      AS prognose_aarsslutt,
            MAX(CASE WHEN type = 'Prognose'
                     AND analyse_dato = '{CURRENT_YEAR}-12-31'
                     THEN nedre_konfidensgrense END)     AS nedre_konfidensgrense,
            MAX(CASE WHEN type = 'Prognose'
                     AND analyse_dato = '{CURRENT_YEAR}-12-31'
                     THEN oevre_konfidensgrense END)     AS oevre_konfidensgrense
        FROM frist_prognose
        WHERE kjoere_id = '{BATCH_ID}'
        GROUP BY indikator
        ORDER BY prognose_aarsslutt ASC
    """).show(30, truncate=False)


# =============================================================================
# CELL 6 — Power BI visual guidance and DAX measures
# =============================================================================
#
# OUTPUT TABLE → VISUALS
#
# frist_prognose inneholder både faktiske rader (type='Faktisk') og
# prognoserader (type='Prognose') for inneværende år, med
# prognose_aarsslutt på hver rad for enkel bruk i KPI-kort.
#
# LINE CHART — YTD actuals with forecast extension (primary visual)
#   X axis:  analyse_dato (Regnskapsperiode) — full current year Jan to Dec
#   Lines:
#     Solid line:  type = 'Faktisk'  — verdi (fristprosent hittil i år)
#     Dotted line: type = 'Prognose' — verdi (prognostisert verdi for gjenstående måneder)
#     Shaded band: nedre_konfidensgrense til oevre_konfidensgrense for prognosedelen
#                  — shows uncertainty range, narrows with more history
#     Flat ref line: year-end target from alert_config (Frist målverdi DAX measure)
#                  — horizontal line the projection must end above
#   Filter:  indikator slicer — one chart per Fagområde as small multiples
#   Reading: dotted line ending above the target line = on track.
#            Dotted line ending below = at risk. Shaded band crossing
#            the target line = uncertain outcome.
#   Note:    Remove report period filter on this visual so all months plot.
#            Use visual-level filter instead of report slicer.
#
# KPI CARD — Year-end estimate
#   Measure: Prognose årslutt (see DAX below)
#   Show alongside current YTD value for context
#   Conditional format: green if above Frist målverdi, red if below
#
# TABLE — Projection summary per indicator
#   Columns: indikator | verdi_hittil | prognose_aarsslutt | nedre_konfidensgrense | oevre_konfidensgrense | Mål
#   Sort:    prognose årslutt ASC — most at-risk indicators first
#   Conditional format on Prognose årslutt: RAG vs Frist målverdi
#
# DAX MEASURES — add to frist_prognose table in semantic model

# Prognose årslutt =
# CALCULATE(
#     MAX(frist_prognose[prognose_aarsslutt]),
#     frist_prognose[type] = "Prognose",
#     frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
# )

# Prognose CI lower =
# CALCULATE(
#     MAX(frist_prognose[nedre_konfidensgrense]),
#     frist_prognose[type] = "Prognose",
#     frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
# )

# Prognose CI upper =
# CALCULATE(
#     MAX(frist_prognose[oevre_konfidensgrense]),
#     frist_prognose[type] = "Prognose",
#     frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
# )

# Prognose RAG =
# VAR Prognose = [Prognose årslutt]
# VAR Mål =
#     CALCULATE(
#         MIN(alert_config[terskel_amber]),
#         alert_config[indikator] = MAX(frist_prognose[indikator]),
#         alert_config[aktiv] = TRUE()
#     )
# RETURN
#     IF(ISBLANK(Prognose) || ISBLANK(Mål), BLANK(),
#     IF(Prognose >= Mål, 3,
#     IF(Prognose >= Mål * 0.95, 2,
#     1)))
# -- 3=Green (on track), 2=Amber (marginal), 1=Red (at risk)
# -- 5% tolerance band below target before going red
