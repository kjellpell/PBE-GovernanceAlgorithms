# =============================================================================
# Seasonal YTD ratio extrapolation for year-end frist% projection.
# Runs nightly after main data pipeline.
#
# Method:
#   1. Compute monthly YTD frist% per indicator for all historical years
#   2. At each calendar month, compute two ratios against the full-year rate:
#      one from cumulative YTD, one from that month alone
#   3. Trim best and worst year per seasonal position (handles outliers)
#   4. Apply the trimmed YTD ratio to current YTD to project year-end, taking
#      the ratio at the day the data actually reaches — a part-finished month
#      is not a finished one
#   5. Confidence interval from trimmed variance across historical years
#   6. Turn that year-end estimate back into a per-month rate for each
#      remaining month, which is the quantity the report plots
#
# Output table: frist_prognose
#   `verdi` is a PERIOD RATE — the projected frist% for the month, not a
#   cumulative year-to-date value. The report's `Faser innen frist %` is
#   DIVIDE([Faser innen frist], [Produserte faser]) in the period's filter
#   context with nothing cumulative over it, so only a period rate can
#   continue that line.
#
#   One anchor row at the end of the last complete month (type='Anker', that
#   month's observed rate, so the projection leaves the actual line where it
#   ends) and one row per day from the start of the next month to 31 December
#   (type='Prognose'), each carrying its month's projected rate. Daily so the
#   series lands on the axis at whatever grain the report rolls it to.
#
#   Actual rates before the anchor aren't written here: they're the report's
#   own live measure. `prognose_aarsslutt`, with `nedre_aarsslutt` and
#   `oevre_aarsslutt`, is the year-end YTD estimate and its interval, repeated
#   on every row for the KPI cards — that one IS cumulative, and is the
#   governance number. It is not the last point of the line.
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
    nedre_aarsslutt             DOUBLE,
    oevre_aarsslutt             DOUBLE,
    kjoert_tidspunkt            TIMESTAMP   NOT NULL,
    kjoere_id                   STRING      NOT NULL
)
USING DELTA
COMMENT 'Prognose for fristprosent per indikator. verdi er en PERIODERATE — prognosert frist% for måneden, ikke kumulativ hittil-i-år — fordi det er det rapportens Faser innen frist % viser. Ett ankerpunkt ved slutten av siste komplette måned (type Anker, faktisk månedsrate) og én rad per dag derfra til 31. desember (type Prognose), hver med sin måneds prognoserte rate. Faktiske rater før ankeret er en live DAX-måling mot saksbehandling.faser, ikke lagret her. Konfidensgrensene er 90 prosent og gjelder radens egen verdi. prognose_aarsslutt med nedre_aarsslutt og oevre_aarsslutt er årssluttprognosen for kumulativ YTD med 90 prosent intervall, gjentatt på alle rader.'
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


def trimmed_stats(values, min_years, trim_n):
    """
    Trimmed mean and std of one seasonal position's ratios across years.

    Drops the best and worst year, but not when that would reduce a three-year
    sample to a single observation and a false zero-width interval.
    """
    if len(values) < min_years:
        return None

    values_sorted = sorted(values)
    if len(values_sorted) >= 2 * trim_n + 2:
        trimmed = values_sorted[trim_n:-trim_n]
    else:
        trimmed = values_sorted

    return {
        "mean_ratio": float(np.mean(trimmed)),
        "std_ratio":  float(np.std(trimmed)) if len(trimmed) > 1 else 0.0,
        "n_years":    len(trimmed),
    }


def monthly_rates(df, indikator, year):
    """
    frist% for each month of a year on its own — not cumulative.

    This is the shape the report plots. `Faser innen frist %` is
    DIVIDE([Faser innen frist], [Produserte faser]) evaluated in whatever
    period the axis puts it in, with no year-to-date filter over it, so it is a
    period rate: each month independent of the ones before it. Despite the
    name, it is not a YTD measure — which is why a cumulative projection could
    never line up with it.

    Returns dict {month: rate} for months with produced faser.
    """
    ind = df[(df["indikator"] == indikator) & (df["aar"] == year)].sort_values("mnd")
    return {
        int(row["mnd"]): row["innenfor"] / row["total"]
        for _, row in ind.iterrows()
        if row["total"] > 0
    }


def monthly_rate_ratios(df, indikator, current_year, min_years=3, trim_n=1):
    """
    For each calendar month, the trimmed mean and std of
        that month's frist% / that year's full-year frist%
    across complete historical years.

    The per-month counterpart of `seasonal_ratios`. Both are needed, and they
    do different jobs: `seasonal_ratios` drives the year-end estimate, because
    cumulative YTD is the stable thing to extrapolate from; this one turns that
    estimate back into the per-month values the report actually draws.

    Its spread is also the honest uncertainty for a single month. A month's
    rate swings far more than the year does — that width is the point, not
    noise to be smoothed away.

    Returns dict {month: {mean_ratio, std_ratio, n_years}} or None.
    """
    years = sorted(df[(df["indikator"] == indikator) &
                      (df["aar"] < current_year)]["aar"].unique())

    complete_years = []
    for y in years:
        rates = monthly_rates(df, indikator, y)
        year_rate = compute_ytd(df, indikator, y).get(12)
        if set(rates) == set(range(1, 13)) and year_rate and np.isfinite(year_rate) and year_rate > 0:
            complete_years.append((rates, year_rate))

    if len(complete_years) < min_years:
        return None

    ratios = {m: [] for m in range(1, 13)}
    for rates, year_rate in complete_years:
        for m, rate in rates.items():
            ratio = rate / year_rate
            # A month's rate varies far more around the year than a YTD value
            # does, so this bound is wider than the one in seasonal_ratios —
            # it is there to drop broken months, not to tame real variation.
            if np.isfinite(ratio) and 0 <= ratio <= 3:
                ratios[m].append(ratio)

    result = {}
    for m in range(1, 13):
        stats = trimmed_stats(ratios[m], min_years, trim_n)
        if stats is not None:
            result[m] = stats

    return result if result else None


def project_month_rate(year_end_est, ci_lower, ci_upper, ratio, z=1.645):
    """
    Turn the year-end estimate into one month's projected frist%.

    Two independent sources of uncertainty, combined by the delta method: how
    uncertain the year-end level is, and how much this month has varied around
    its year historically. The second usually dominates, and should — a single
    month is a much smaller sample than a year.

    Returns (estimate, ci_lower, ci_upper) or (None, None, None).
    """
    if year_end_est is None or ratio is None:
        return None, None, None

    mean_ratio = ratio["mean_ratio"]
    std_ratio  = ratio["std_ratio"]
    if not np.isfinite(mean_ratio) or not np.isfinite(std_ratio) or mean_ratio < 0:
        return None, None, None

    estimate = min(1.0, max(0.0, year_end_est * mean_ratio))

    std_year_end = 0.0
    if ci_lower is not None and ci_upper is not None and ci_upper > ci_lower:
        std_year_end = (ci_upper - ci_lower) / (2 * z)
    std_estimate = np.sqrt((mean_ratio * std_year_end) ** 2
                           + (year_end_est * std_ratio) ** 2)

    lower = min(estimate, max(0.0, estimate - z * std_estimate))
    upper = max(estimate, min(1.0, estimate + z * std_estimate))

    return round(float(estimate), 4), round(float(lower), 4), round(float(upper), 4)


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
        stats = trimmed_stats(ratios[m], min_years, trim_n)
        if stats is not None:
            result[m] = stats

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


def build_forecast_rows(indikator, first_forecast_date, month_ratios,
                        year_end_est, ci_lower, ci_upper,
                        kjoert_tidspunkt, kjoere_id,
                        anchor_date=None, anchor_rate=None):
    """
    Continue the monthly frist% line from the last complete month to
    31 December, one row per day.

    `verdi` is a *period* rate — the projected frist% for the month the day
    falls in — because that is what the report draws. `Faser innen frist %` is
    DIVIDE([Faser innen frist], [Produserte faser]) in the period's filter
    context, with nothing cumulative over it, so a year-to-date projection is a
    different quantity and cannot continue that line no matter how it is drawn.

    One row per day, so the series lands on the axis at whatever grain it is
    rolled to. Every day in a month carries that month's projected rate, which
    makes the projection a step per month — which is what a monthly rate is.

    The anchor is the last *complete* month's observed rate, so the projection
    leaves the actual line at a point that won't move. The current month is
    projected rather than anchored on: part of a month is not a month, and its
    running rate is the least stable number on the chart.
    """
    year_end_date = date(first_forecast_date.year, 12, 31)
    if year_end_est is None or first_forecast_date > year_end_date:
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
            "nedre_aarsslutt":       ci_lower,
            "oevre_aarsslutt":       ci_upper,
            "kjoert_tidspunkt":      kjoert_tidspunkt,
            "kjoere_id":             kjoere_id,
        }

    rows = []
    if anchor_date is not None and anchor_rate is not None:
        # Observed, not projected — no band on it.
        rows.append(make_row(anchor_date, "Anker", anchor_rate, anchor_rate, anchor_rate))

    projected = {}
    for mnd in range(first_forecast_date.month, 13):
        if mnd not in month_ratios:
            continue
        estimate, lower, upper = project_month_rate(
            year_end_est, ci_lower, ci_upper, month_ratios[mnd]
        )
        if estimate is not None:
            projected[mnd] = (estimate, lower, upper)

    if not projected:
        return []

    for timestamp in pd.date_range(first_forecast_date, year_end_date, freq="D"):
        on_date = timestamp.date()
        if on_date.month not in projected:
            continue
        estimate, lower, upper = projected[on_date.month]
        rows.append(make_row(on_date, "Prognose", estimate, lower, upper))

    return rows


OUTPUT_COLUMNS = {
    "indikator",
    "analyse_dato",
    "type",
    "verdi",
    "nedre_konfidensgrense",
    "oevre_konfidensgrense",
    "prognose_aarsslutt",
    "nedre_aarsslutt",
    "oevre_aarsslutt",
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
        year_end = row["prognose_aarsslutt"]
        year_end_lower = row["nedre_aarsslutt"]
        year_end_upper = row["oevre_aarsslutt"]
        if year_end_lower is not None and year_end_upper is not None:
            if not all(np.isfinite(value) and 0 <= value <= 1
                       for value in (year_end_lower, year_end_upper)):
                raise ValueError(f"Year-end interval out of bounds in row {row_number}")
            if year_end_lower > year_end_upper or (
                year_end is not None and not year_end_lower <= year_end <= year_end_upper
            ):
                raise ValueError(f"Invalid year-end interval in row {row_number}")

        lower = row["nedre_konfidensgrense"]
        upper = row["oevre_konfidensgrense"]
        # This band belongs to `verdi` — the month rate on the same row, and
        # the line it is drawn around. The year-end estimate has its own
        # interval, checked just above; the two are different quantities.
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

    # Two seasonal models, two jobs. The YTD ratios drive the year-end
    # estimate — cumulative YTD is the stable thing to extrapolate from. The
    # per-month ratios turn that estimate back into the period rates the
    # report plots.
    ratios = seasonal_ratios(
        monthly, indikator, CURRENT_YEAR,
        min_years=MIN_YEARS, trim_n=TRIM_N
    )
    month_ratios = monthly_rate_ratios(
        monthly, indikator, CURRENT_YEAR,
        min_years=MIN_YEARS, trim_n=TRIM_N
    )

    if ratios is None or month_ratios is None:
        print(f"Skipping {indikator} — insufficient history")
        continue

    # Current year actuals
    current_ytd   = compute_ytd(monthly, indikator, CURRENT_YEAR)
    current_rates = monthly_rates(monthly, indikator, CURRENT_YEAR)

    if not current_ytd:
        print(f"Skipping {indikator} — no current year data")
        continue

    # Latest month with data
    latest_month = max(current_ytd.keys())
    latest_ytd   = current_ytd[latest_month]

    # The year-end estimate uses everything loaded, current part-month
    # included, against the seasonal ratio for the point in the year that data
    # actually reaches — a part-finished month is not a finished one.
    if latest_month == CURRENT_MONTH:
        data_date = TODAY
    else:
        data_date = (pd.Timestamp(CURRENT_YEAR, latest_month, 1)
                     + pd.offsets.MonthEnd(0)).date()

    ratio_now = seasonal_ratio_on(ratios, data_date)
    if ratio_now is None:
        print(f"Skipping {indikator} — no seasonal ratio at {data_date}")
        continue

    year_end_est, ci_lo, ci_hi = project_year_end(
        latest_ytd, data_date.month, {data_date.month: ratio_now}
    )

    # The line, though, forks off the last *complete* month: a part-month's
    # rate is the least stable number on the chart, and it moves every night.
    # The current month is projected instead. A month is complete on its last
    # day here — a fase closing 31 August is in August's number, not still
    # arriving in September — so this is the plain calendar month, with no
    # registration lag to back off for.
    last_complete_month = latest_month - 1 if latest_month == CURRENT_MONTH else latest_month
    if last_complete_month >= 12:
        print(f"Skipping {indikator} — the year is complete, nothing to project")
        continue

    anchor_date = None
    anchor_rate = current_rates.get(last_complete_month)
    if anchor_rate is not None:
        anchor_date = (pd.Timestamp(CURRENT_YEAR, last_complete_month, 1)
                       + pd.offsets.MonthEnd(0)).date()

    # Actual rates for the months up to the anchor are NOT written here — that
    # is the report's own `Faser innen frist %` measure against
    # saksbehandling.faser, no algorithm needed. This table stores what a live
    # measure structurally can't produce: the projection and its band. The
    # anchor row is the one observed value it duplicates, on purpose, so the
    # projection has the actual line to leave from.
    # See Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md.
    indicator_rows = build_forecast_rows(
        indikator, date(CURRENT_YEAR, last_complete_month + 1, 1), month_ratios,
        year_end_est, ci_lo, ci_hi,
        datetime.now(), BATCH_ID,
        anchor_date=anchor_date, anchor_rate=anchor_rate
    )

    if not indicator_rows:
        print(f"Skipping {indikator} — no month could be projected")
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
        StructField("nedre_aarsslutt", DoubleType(), True),
        StructField("oevre_aarsslutt", DoubleType(), True),
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

    # Summary — year-end estimates for current indicators. The per-row band is
    # a month rate's band now, so it isn't the year-end interval and isn't
    # summarised here; the year-end number is what this table adds.
    spark.sql(f"""
        SELECT
            indikator,
            MAX(prognose_aarsslutt)                             AS prognose_aarsslutt,
            MIN(CASE WHEN type = 'Prognose' THEN verdi END)     AS laveste_maanedsrate,
            MAX(CASE WHEN type = 'Prognose' THEN verdi END)     AS hoeyeste_maanedsrate
        FROM analyser.frist_prognose
        WHERE kjoere_id = '{BATCH_ID}'
        GROUP BY indikator
        ORDER BY prognose_aarsslutt ASC
    """).show(30, truncate=False)

# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md
# =============================================================================
