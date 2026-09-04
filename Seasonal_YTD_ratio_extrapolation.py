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
#   The forecast as a YTD trajectory, so it plots on the same axis as the live
#   `Fristprosent YTD` measure: one anchor row at the last closed month
#   (type='Anker', the observed YTD, so the forecast line starts where the
#   actual line ends) plus one row per remaining month (type='Prognose').
#   Actual YTD for the earlier months isn't written here: it's a plain live DAX
#   year-to-date measure against saksbehandling.faser (standard time
#   intelligence, no algorithm needed), so storing a copy of it here would just
#   be duplicated data. The confidence band on each row belongs to that row's
#   `verdi`; `prognose_aarsslutt` is the December endpoint repeated on every
#   row for the KPI cards. See Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md.
#
# Schedule: nightly, after main data pipeline.
# Minimum history: 3 years per indicator. Suppressed below that.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
import pandas as pd
import numpy as np
from datetime import datetime

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
CREATE TABLE IF NOT EXISTS prognoser.frist_prognose (
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
COMMENT 'Årssluttprognose for fristprosent per indikator som YTD-bane — ett ankerpunkt på siste lukkede måned (type Anker, faktisk YTD) og én rad per gjenstående måned (type Prognose). Faktisk YTD for tidligere måneder er en live DAX-måling mot saksbehandling.faser, ikke lagret her. Konfidensgrensene er 90 prosent, gjelder radens egen verdi og bygger på historisk variasjon i samme sesongposisjon. prognose_aarsslutt er desemberpunktet gjentatt på alle rader.'
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
        COUNT(CASE WHEN pr.frist_dager IS NOT NULL THEN 1 END)             AS total
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikatorer
        ON indikatorer.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.frist_dager IS NOT NULL
            AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
            AND (
                YEAR(pr.sluttmilepaeldato) < {CURRENT_YEAR}
                OR MONTH(pr.sluttmilepaeldato) <= {CURRENT_MONTH}
            )
        GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
    ORDER BY pr.indikator, aar, mnd
""").toPandas()

if monthly.empty:
    print("Ingen månedlige data lastet.")
else:
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

    # A seasonal ratio needs every month; otherwise a missing month can look
    # like seasonality.
    complete_years = []
    for y in years:
        ytd = compute_ytd(df, indikator, y)
        if set(ytd) == set(range(1, 13)):
            complete_years.append((y, ytd))

    if len(complete_years) < min_years:
        return None

    ratios = {m: [] for m in range(1, 13)}

    for year, ytd in complete_years:
        year_end = ytd.get(12)
        if year_end is None or not np.isfinite(year_end) or year_end <= 0:
            print(f"Skipping {indikator} {year} — invalid year-end YTD")
            continue
        for m, ytd_val in ytd.items():
            ratio = ytd_val / year_end
            if np.isfinite(ratio) and 0 <= ratio <= 1.5:
                ratios[m].append(ratio)

    result = {}
    for m in range(1, 13):
        vals = ratios[m]
        if len(vals) < min_years:
            continue
        # Avoid reducing a three-year sample to one observation and a false
        # zero-width uncertainty interval.
        vals_sorted = sorted(vals)
        if len(vals_sorted) >= 2 * trim_n + 2:
            trimmed = vals_sorted[trim_n:-trim_n]
        else:
            trimmed = vals_sorted
        result[m] = {
            "mean_ratio": float(np.mean(trimmed)),
            "std_ratio":  float(np.std(trimmed)) if len(trimmed) > 1 else 0.0,
            "n_years":    len(trimmed)
        }

    return result if result else None


def project_year_end(current_ytd, month, ratios, z=1.645):
    """
    Project year-end frist% from current YTD value.
    z=1.645 gives 90% confidence interval (appropriate for governance).

    Returns (year_end_estimate, ci_lower, ci_upper) or (None, None, None).
    """
    if month not in ratios:
        return None, None, None

    r = ratios[month]
    if not np.isfinite(current_ytd) or not 0 <= current_ytd <= 1:
        return None, None, None
    if not np.isfinite(r["mean_ratio"]) or r["mean_ratio"] <= 1e-9:
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
        round(float(min(ci_lower, estimate)), 4),
        round(float(max(ci_upper, estimate)), 4)
    )


def build_forecast_rows(indikator, latest_month, latest_ytd, ratios,
                        year_end_est, ci_lower, ci_upper,
                        current_year, kjoert_tidspunkt, kjoere_id):
    """
    Build the forecast series for one indicator, as YTD values.

    The series is a YTD trajectory so it can be drawn on the same axis as the
    live `Fristprosent YTD` measure — the only actual series it is comparable
    with. Two things make it usable in a line chart:

      * an anchor row at the last closed month carrying the observed YTD, so
        the forecast line starts where the actual line ends instead of
        floating unattached over the remaining months
      * a confidence band that belongs to the line it surrounds: the year-end
        interval scaled by the same seasonal ratio that shapes the trajectory,
        so it opens up with the forecast horizon and closes on
        [ci_lower, ci_upper] in December

    Scaling `year_end_est` by the seasonal ratio (rather than re-deriving the
    path from `latest_ytd`) keeps the trajectory and the year-end KPI card
    telling the same story: the December point of the line is the number on
    the card.
    """
    if year_end_est is None or latest_month not in ratios or latest_month >= 12:
        return []

    def month_end(mnd):
        return (pd.Timestamp(current_year, mnd, 1) + pd.offsets.MonthEnd(0)).date()

    def row(mnd, type_, verdi, lower, upper):
        return {
            "indikator":             indikator,
            "analyse_dato":          month_end(mnd),
            "type":                  type_,
            "verdi":                 round(float(verdi), 4),
            "nedre_konfidensgrense": None if lower is None else round(float(lower), 4),
            "oevre_konfidensgrense": None if upper is None else round(float(upper), 4),
            "prognose_aarsslutt":    year_end_est,
            "kjoert_tidspunkt":      kjoert_tidspunkt,
            "kjoere_id":             kjoere_id,
        }

    forecast_rows = []
    previous_ytd  = latest_ytd

    for mnd in range(latest_month + 1, 13):
        if mnd not in ratios:
            continue
        ratio = ratios[mnd]["mean_ratio"]
        if not np.isfinite(ratio) or ratio <= 1e-9:
            continue

        # YTD is cumulative, so the path may not fall back below the last
        # observed (or last forecast) value.
        forecast_ytd = min(1.0, max(previous_ytd, year_end_est * ratio))

        lower = None if ci_lower is None else max(0.0, ci_lower * ratio)
        upper = None if ci_upper is None else min(1.0, ci_upper * ratio)
        # The clamps above can push the band off the line; keep it bracketing.
        if lower is not None:
            lower = min(lower, forecast_ytd)
        if upper is not None:
            upper = max(upper, forecast_ytd)

        forecast_rows.append(row(mnd, "Prognose", forecast_ytd, lower, upper))
        previous_ytd = forecast_ytd

    if not forecast_rows:
        return []

    # The anchor is an observed value, not a projection — no band on it.
    anchor = row(latest_month, "Anker", latest_ytd, latest_ytd, latest_ytd)
    return [anchor] + forecast_rows


OUTPUT_COLUMNS = {
    "indikator",
    "analyse_dato",
    "type",
    "verdi",
    "nedre_konfidensgrense",
    "oevre_konfidensgrense",
    "prognose_aarsslutt",
    "kjoert_tidspunkt",
    "kjoere_id",
}


def validate_results(results):
    """Validate rows before Spark applies the Delta table schema."""
    for row_number, row in enumerate(results, start=1):
        if set(row) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Result row {row_number} has unexpected columns: "
                f"{sorted(set(row) ^ OUTPUT_COLUMNS)}"
            )
        for column in ("verdi", "prognose_aarsslutt"):
            value = row[column]
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{column} out of bounds in row {row_number}: {value}")
        lower = row["nedre_konfidensgrense"]
        upper = row["oevre_konfidensgrense"]
        # The band is checked against `verdi` — the value on the same row, and
        # the line it is drawn around — not against `prognose_aarsslutt`, which
        # is the December endpoint repeated on every row for the KPI cards.
        verdi = row["verdi"]
        if lower is not None and upper is not None:
            if not all(np.isfinite(value) and 0 <= value <= 1 for value in (lower, upper)):
                raise ValueError(f"Confidence interval out of bounds in row {row_number}")
            if lower > upper or (verdi is not None and not lower <= verdi <= upper):
                raise ValueError(f"Invalid confidence interval in row {row_number}")


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

    # Actual YTD rows for past months are NOT written here — that value is a
    # plain live DAX measure against saksbehandling.faser (standard
    # year-to-date time intelligence, no algorithm needed). This table only
    # stores what a live measure structurally can't produce: the
    # seasonal-ratio forecast and its confidence interval. The one exception
    # is the anchor row build_forecast_rows() puts at the last closed month —
    # a single observed point, duplicated on purpose so the forecast line has
    # somewhere to start. See Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md.
    indicator_rows = build_forecast_rows(
        indikator, latest_month, latest_ytd, ratios,
        year_end_est, ci_lo, ci_hi,
        CURRENT_YEAR, datetime.now(), BATCH_ID
    )

    if not indicator_rows:
        print(f"Skipping {indikator} — no forecast months from month {latest_month}")
        continue

    results.extend(indicator_rows)

print(f"\nProjection rows computed: {len(results)}")
print(f"Indicators projected: {len(set(r['indikator'] for r in results))}")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

if not results:
    print("Ingen prognoseresultater å skrive.")
else:
    validate_results(results)
    output_schema = StructType([
        StructField("indikator", StringType(), False),
        StructField("analyse_dato", DateType(), False),
        StructField("type", StringType(), False),
        StructField("verdi", DoubleType(), True),
        StructField("nedre_konfidensgrense", DoubleType(), True),
        StructField("oevre_konfidensgrense", DoubleType(), True),
        StructField("prognose_aarsslutt", DoubleType(), True),
        StructField("kjoert_tidspunkt", TimestampType(), False),
        StructField("kjoere_id", StringType(), False),
    ])
    output_columns = [field.name for field in output_schema.fields]
    output_rows = [
        tuple(row[column] for column in output_columns)
        for row in results
    ]
    results_spark = spark.createDataFrame(output_rows, schema=output_schema)

    # Idempotent — delete current year rows before inserting
    spark.sql(f"""
            DELETE FROM prognoser.frist_prognose
            WHERE analyse_dato >= '{CURRENT_YEAR}-01-31'
                AND analyse_dato <= '{CURRENT_YEAR}-12-31'
    """)

    results_spark.write.mode("append").saveAsTable("prognoser.frist_prognose")

    print(f"prognoser.frist_prognose skrevet: {len(results)} rader")

    # Summary — year-end estimates for current indicators. verdi_hittil
    # (actual YTD so far) isn't in this table anymore — it's a live DAX
    # measure — so this only shows what the table actually holds.
    spark.sql(f"""
        SELECT
            indikator,
            MAX(prognose_aarsslutt)                      AS prognose_aarsslutt,
            MAX(CASE WHEN analyse_dato = '{CURRENT_YEAR}-12-31'
                     THEN nedre_konfidensgrense END)     AS nedre_konfidensgrense,
            MAX(CASE WHEN analyse_dato = '{CURRENT_YEAR}-12-31'
                     THEN oevre_konfidensgrense END)     AS oevre_konfidensgrense
        FROM prognoser.frist_prognose
        WHERE kjoere_id = '{BATCH_ID}'
        GROUP BY indikator
        ORDER BY prognose_aarsslutt ASC
    """).show(30, truncate=False)

# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md
# =============================================================================
