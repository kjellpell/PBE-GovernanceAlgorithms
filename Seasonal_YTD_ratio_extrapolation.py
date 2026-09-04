# =============================================================================
# Seasonal YTD ratio extrapolation for year-end frist% projection.
# Runs nightly after main data pipeline.
#
# Method:
#   1. Compute monthly YTD frist% per indicator for all historical years
#   2. At each calendar month, compute the ratio: YTD_at_month / year_end
#   3. Trim best and worst year per seasonal position (handles outliers)
#   4. Apply trimmed mean ratio to current YTD to project year-end, taking the
#      ratio at the day the data actually reaches — a part-finished month is
#      not a finished one
#   5. Confidence interval from trimmed variance across historical years
#   6. Scale both back along the seasonal path to draw a daily line from today
#      to 31 December
#
# Output table: frist_prognose
#   The projection as a daily YTD series, because that is the grain the actual
#   `Fristprosent YTD` measure is plotted at: one anchor row at the last date
#   with data (type='Anker', the observed YTD, so the projection leaves the
#   actual line at the point the actual line reaches) and one row per day from
#   there to 31 December (type='Prognose'). Actual YTD for the earlier days
#   isn't written here: it's a plain live DAX year-to-date measure against
#   saksbehandling.faser (standard time intelligence, no algorithm needed), so
#   storing a copy of it would just be duplicated data. Each row's band belongs
#   to that row's `verdi`; `prognose_aarsslutt` is the 31 December endpoint,
#   repeated on every row for the KPI cards.
#   See Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md.
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
from datetime import date, datetime, timedelta

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID      = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY         = datetime.now().date()
CURRENT_YEAR  = TODAY.year
CURRENT_MONTH = TODAY.month
MIN_YEARS     = 3    # minimum history for reliable seasonal pattern
TRIM_N        = 1    # drop N best and N worst years per seasonal position
START_YEAR    = 2015 # exclude data before this year — adjust if older data is reliable


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.frist_prognose (
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
COMMENT 'Årssluttprognose for fristprosent per indikator som daglig YTD-bane — ett ankerpunkt på siste dato med data (type Anker, faktisk YTD) og én rad per dag derfra til 31. desember (type Prognose). Faktisk YTD fram til ankeret er en live DAX-måling mot saksbehandling.faser, ikke lagret her. Konfidensgrensene er 90 prosent, gjelder radens egen verdi og bygger på historisk variasjon i samme sesongposisjon. prognose_aarsslutt er 31. desember-punktet gjentatt på alle rader.'
""")

print("analyser.frist_prognose-tabellen er klar")


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


def seasonal_ratio_on(ratios, on_date):
    """
    Seasonal ratio at a specific *date*, not just at a month end.

    `seasonal_ratios` gives one ratio per completed month. The report axis is
    daily, and so is the run: a projection made on the 4th of September that
    divides by a whole September's ratio understates year-end, because it
    treats four days of data as a full month. Interpolate linearly between the
    previous month end and this one by how far through the month the date is —
    at the last day of the month this returns that month's ratio unchanged.

    Linear within a month is an approximation: case completions aren't spread
    evenly across a month. It is a much smaller error than pretending the
    month is already over.

    Returns a ratio dict shaped like `seasonal_ratios`' values, or None.
    """
    month = on_date.month
    if month not in ratios:
        return None

    fraction = on_date.day / pd.Timestamp(on_date).days_in_month

    if month == 1:
        # A year-to-date ratio starts the year at nothing.
        previous = {"mean_ratio": 0.0, "std_ratio": 0.0}
    elif month - 1 in ratios:
        previous = ratios[month - 1]
    else:
        return None

    current = ratios[month]
    mean = previous["mean_ratio"] + (current["mean_ratio"] - previous["mean_ratio"]) * fraction
    std  = previous["std_ratio"]  + (current["std_ratio"]  - previous["std_ratio"])  * fraction

    if not np.isfinite(mean) or mean <= 1e-9:
        return None

    return {"mean_ratio": float(mean), "std_ratio": float(std), "n_years": current["n_years"]}


def build_forecast_rows(indikator, anchor_date, anchor_ytd, ratios,
                        year_end_est, ci_lower, ci_upper,
                        kjoert_tidspunkt, kjoere_id):
    """
    Continue the YTD line from where the actuals end to 31 December, one row
    per day, with the confidence band around it.

    Daily, because that's the grain the actual `Fristprosent YTD` measure is
    plotted at. A forecast written only at month ends is three points on a
    daily axis, which is a flat segment floating over the last quarter, not a
    continuation of the line.

    The first row is the anchor: the observed YTD at `anchor_date`, with no
    band, so the projection leaves the actual line at exactly the point the
    actual line reaches. Every day after that is `year_end_est` scaled by the
    seasonal ratio for that date, and the band is the year-end interval scaled
    the same way — zero width at the anchor, opening up with the horizon, and
    closing on [ci_lower, ci_upper] on 31 December.

    Scaling `year_end_est` (rather than re-deriving the path from
    `anchor_ytd`) keeps the line and the year-end KPI card telling the same
    story: the last point of the line is the number on the card.
    """
    year_end_date = date(anchor_date.year, 12, 31)
    if year_end_est is None or anchor_date >= year_end_date:
        return []

    def make_row(analyse_dato, type_, verdi, lower, upper):
        return {
            "indikator":             indikator,
            "analyse_dato":          analyse_dato,
            "type":                  type_,
            "verdi":                 round(float(verdi), 4),
            "nedre_konfidensgrense": None if lower is None else round(float(lower), 4),
            "oevre_konfidensgrense": None if upper is None else round(float(upper), 4),
            "prognose_aarsslutt":    year_end_est,
            "kjoert_tidspunkt":      kjoert_tidspunkt,
            "kjoere_id":             kjoere_id,
        }

    # The anchor is observed, not projected — no band on it.
    rows = [make_row(anchor_date, "Anker", anchor_ytd, anchor_ytd, anchor_ytd)]

    for timestamp in pd.date_range(anchor_date + timedelta(days=1), year_end_date, freq="D"):
        on_date = timestamp.date()
        ratio = seasonal_ratio_on(ratios, on_date)
        if ratio is None:
            continue

        scale = ratio["mean_ratio"]
        # No monotonic floor here: this is a ratio, not a running count. YTD
        # frist% falls whenever the months ahead are worse than the year so
        # far, and forcing the path never to decline is what flattens it into
        # a line that says nothing.
        verdi = min(1.0, max(0.0, year_end_est * scale))
        lower = None if ci_lower is None else min(verdi, max(0.0, ci_lower * scale))
        upper = None if ci_upper is None else max(verdi, min(1.0, ci_upper * scale))

        rows.append(make_row(on_date, "Prognose", verdi, lower, upper))

    # An anchor on its own is not a forecast.
    return rows if len(rows) > 1 else []


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

    # Anchor where the actual line actually ends: today if the current month
    # already has data, otherwise the end of the last month that does (the
    # nightly load can lag). Everything after that date is projected.
    if latest_month == CURRENT_MONTH:
        anchor_date = TODAY
    else:
        anchor_date = (pd.Timestamp(CURRENT_YEAR, latest_month, 1)
                       + pd.offsets.MonthEnd(0)).date()

    # Year-end estimate from the YTD we have, against the seasonal ratio for
    # the point in the year that YTD actually reaches — a part-finished month
    # is not a finished one.
    ratio_now = seasonal_ratio_on(ratios, anchor_date)
    if ratio_now is None:
        print(f"Skipping {indikator} — no seasonal ratio at {anchor_date}")
        continue

    year_end_est, ci_lo, ci_hi = project_year_end(
        latest_ytd, anchor_date.month, {anchor_date.month: ratio_now}
    )

    # Actual YTD for the days up to the anchor is NOT written here — that is a
    # plain live DAX year-to-date measure against saksbehandling.faser
    # (standard time intelligence, no algorithm needed). This table stores what
    # a live measure structurally can't produce: the seasonal-ratio projection
    # and its confidence band. The anchor row is the one observed value it
    # duplicates, on purpose, so the projection has the actual line to leave
    # from. See Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md.
    indicator_rows = build_forecast_rows(
        indikator, anchor_date, latest_ytd, ratios,
        year_end_est, ci_lo, ci_hi,
        datetime.now(), BATCH_ID
    )

    if not indicator_rows:
        print(f"Skipping {indikator} — nothing left to project from {anchor_date}")
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
            DELETE FROM analyser.frist_prognose
            WHERE analyse_dato >= '{CURRENT_YEAR}-01-01'
                AND analyse_dato <= '{CURRENT_YEAR}-12-31'
    """)

    results_spark.write.mode("append").saveAsTable("analyser.frist_prognose")

    print(f"analyser.frist_prognose skrevet: {len(results)} rader")

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
        FROM analyser.frist_prognose
        WHERE kjoere_id = '{BATCH_ID}'
        GROUP BY indikator
        ORDER BY prognose_aarsslutt ASC
    """).show(30, truncate=False)

# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md
# =============================================================================
