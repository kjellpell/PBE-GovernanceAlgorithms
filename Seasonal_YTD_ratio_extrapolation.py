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
#   `innenfor_prognose`/`produserte_prognose` carry that same rate as modelled
#   faser counts, so a report can read this table with the same
#   DIVIDE(SUM(...), SUM(...)) pattern it already uses on the fact table.
#   That is what makes the projection agree with itself at any rollup — an
#   average of `verdi` across rows only equals that arithmetic in the single-
#   month, single-indicator case; everywhere else it is different arithmetic
#   and gives a different number.
#
#   One anchor row at the end of the last complete month (type='Anker', the
#   real counts and rate for that month, so the projection leaves the actual
#   line where it ends, matching exactly) and one row per day from the start
#   of the next month to 31 December (type='Prognose'), each carrying a
#   fraction of its month's modelled counts. Daily so the series lands on the
#   axis at whatever grain the report rolls it to, and so summing a month's
#   rows back up reconstitutes that month's modelled total.
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
    innenfor_prognose           DOUBLE,
    produserte_prognose         DOUBLE,
    prognose_aarsslutt          DOUBLE,
    nedre_aarsslutt             DOUBLE,
    oevre_aarsslutt             DOUBLE,
    kjoert_tidspunkt            TIMESTAMP   NOT NULL,
    kjoere_id                   STRING      NOT NULL
)
USING DELTA
COMMENT 'Prognose for fristprosent per indikator. verdi er en PERIODERATE — prognosert frist% for måneden, ikke kumulativ hittil-i-år — fordi det er det rapportens Faser innen frist % viser. innenfor_prognose og produserte_prognose er modellerte faser-tellinger for samme rad; summer disse og divider (som rapportens egen DIVIDE(SUM,SUM)-mål) for et resultat som stemmer på tvers av indikatorer og perioder — et gjennomsnitt av verdi gjør det ikke. Ett ankerpunkt ved slutten av siste komplette måned (type Anker, faktiske tellinger og faktisk rate) og én rad per dag derfra til 31. desember (type Prognose), hver med en brøkdel av sin måneds modellerte tellinger. Faktiske rater før ankeret er en live DAX-måling mot saksbehandling.faser, ikke lagret her. Konfidensgrensene er 90 prosent og gjelder radens egen verdi. prognose_aarsslutt med nedre_aarsslutt og oevre_aarsslutt er årssluttprognosen for kumulativ YTD med 90 prosent intervall, gjentatt på alle rader.'
""")

# A table created by an earlier run predates columns added since, and
# CREATE TABLE IF NOT EXISTS won't add them — which is what makes the
# nightly append fail with a Delta schema mismatch. Evolve the schema here,
# explicitly and idempotently, rather than putting mergeSchema on the write:
# that would fix this one case and silently absorb every future drift too.
NEW_COLUMNS = (
    ("nedre_aarsslutt", "DOUBLE",
     "Nedre 90 prosent konfidensgrense for arssluttprognosen"),
    ("oevre_aarsslutt", "DOUBLE",
     "Ovre 90 prosent konfidensgrense for arssluttprognosen"),
    ("innenfor_prognose", "DOUBLE",
     "Modellert antall faser innenfor frist for raden, til bruk i DIVIDE(SUM,SUM)"),
    ("produserte_prognose", "DOUBLE",
     "Modellert antall produserte faser for raden, til bruk i DIVIDE(SUM,SUM)"),
)

existing_columns = {
    field.name.lower()
    for field in spark.table("analyser.frist_prognose").schema
}
for column, column_type, column_comment in NEW_COLUMNS:
    if column not in existing_columns:
        spark.sql(
            f"ALTER TABLE analyser.frist_prognose "
            f"ADD COLUMNS ({column} {column_type} COMMENT '{column_comment}')"
        )
        print(f"La til kolonne {column}")

print("analyser.frist_prognose-tabellen er klar")


# =============================================================================
# CELL 2 — Load historical monthly frist% per indicator
# =============================================================================
# Full history — all years, all indicators.

# "Produced" mirrors the report's own [Produserte faser] measure exactly —
# a fase counts once it has both a start and an end milestone date, nothing
# to do with frist_dager. (An earlier version of this query used
# `frist_dager IS NOT NULL` as the denominator, copied from a different
# measure, Fristprosent (måned), that happens to give the same headline
# numbers elsewhere but is not what [Produserte faser] actually filters on —
# it silently dropped faser without a tracked deadline from both sides of the
# ratio, which is why an indicator like Endringstillatelse could show a
# script-computed month rate a couple of points off the live measure's.)
# Every row that survives the WHERE clause is produced by that same
# definition, so `total` is a plain COUNT(*), not a conditional one.
monthly = spark.sql(f"""
    SELECT
        pr.indikator,
        YEAR(pr.sluttmilepaeldato)                 AS aar,
        MONTH(pr.sluttmilepaeldato)                AS mnd,
        COUNT(CASE WHEN pr.innenfor_frist = 1 THEN 1 END)           AS innenfor,
        COUNT(*)                                                    AS total
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikatorer
        ON indikatorer.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.startmilepaeldato IS NOT NULL
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


def monthly_counts(df, indikator, year):
    """
    Raw (innenfor, total) faser counts for each month of a year — the counts
    `monthly_rates` divides, kept intact.

    These are what let a stored row match the report's own measure exactly:
    `Faser innen frist %` is DIVIDE(SUM(innenfor), SUM(total)), so a row that
    carries the same two counts reproduces it by construction, at any
    aggregation the report rolls up to. A stored rate alone cannot — averaging
    a rate across months or indicators is not the same arithmetic as summing
    counts and dividing once, and the two diverge everywhere except a single
    month for a single indicator.

    Returns dict {month: (innenfor, total)} for months with produced faser.
    """
    ind = df[(df["indikator"] == indikator) & (df["aar"] == year)].sort_values("mnd")
    return {
        int(row["mnd"]): (row["innenfor"], row["total"])
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


def monthly_volume_ratios(df, indikator, current_year, min_years=3, trim_n=1):
    """
    For each calendar month, the trimmed mean share of a year's total
    production (`total` faser) that fell in that month, across complete
    historical years — the volume counterpart of `monthly_rate_ratios`.

    This is what lets the projection carry real counts instead of a bare
    rate: `project_month_volumes` extrapolates an annual case count from
    this year's volume so far the same way `project_year_end` extrapolates
    the frist% level, and this month-by-month share is what splits that
    total back across the remaining months.

    Returns dict {month: {mean_ratio, std_ratio, n_years}} or None.
    """
    years = sorted(df[(df["indikator"] == indikator) &
                      (df["aar"] < current_year)]["aar"].unique())

    complete_years = []
    for y in years:
        ind = df[(df["indikator"] == indikator) & (df["aar"] == y)]
        totals = dict(zip(ind["mnd"], ind["total"]))
        if set(totals) == set(range(1, 13)):
            complete_years.append(totals)

    if len(complete_years) < min_years:
        return None

    shares = {m: [] for m in range(1, 13)}
    for totals in complete_years:
        year_total = sum(totals.values())
        if year_total <= 0:
            continue
        for m, total in totals.items():
            shares[m].append(total / year_total)

    result = {}
    for m in range(1, 13):
        stats = trimmed_stats(shares[m], min_years, trim_n)
        if stats is not None:
            result[m] = stats

    return result if result else None


def project_month_volumes(current_year_totals, last_complete_month, volume_ratios):
    """
    Projected produced-faser count for each remaining month, from this
    year's observed monthly totals and each month's historical share of the
    year.

    Same pattern as `project_year_end`: sum the share observed so far,
    scale up to a full-year total, then split that total by each remaining
    month's own historical share.

    Returns {month: projected_total} for months after last_complete_month.
    Empty if the observed volume or share is degenerate.
    """
    observed_total = sum(
        current_year_totals.get(m, 0) for m in range(1, last_complete_month + 1)
    )
    observed_share = sum(
        volume_ratios[m]["mean_ratio"]
        for m in range(1, last_complete_month + 1)
        if m in volume_ratios
    )
    if observed_total <= 0 or observed_share <= 1e-9:
        return {}

    projected_annual_total = observed_total / observed_share
    return {
        m: max(0.0, projected_annual_total * volume_ratios[m]["mean_ratio"])
        for m in range(last_complete_month + 1, 13)
        if m in volume_ratios
    }


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


def build_forecast_rows(indikator, first_forecast_date, month_ratios, month_volumes,
                        year_end_est, ci_lower, ci_upper,
                        kjoert_tidspunkt, kjoere_id,
                        anchor_date=None, anchor_innenfor=None, anchor_total=None):
    """
    Continue the monthly frist% line from the last complete month to
    31 December, one row per day.

    Each row carries `innenfor_prognose`/`produserte_prognose` — modelled
    faser counts — alongside `verdi`, the rate they imply. The counts are
    what make the row aggregate the way `Faser innen frist %` does:
    DIVIDE(SUM(innenfor_prognose), SUM(produserte_prognose)) reproduces the
    report's own DIVIDE(SUM(innenfor), SUM(total)) arithmetic at any grain a
    report might roll up to — indicator, quarter, several months at once.
    `verdi` alone cannot: averaging a rate across rows is different
    arithmetic from summing counts and dividing once, and the two only agree
    in the single-month, single-indicator case.

    The anchor row's counts are the real ones for the last complete month —
    not modelled — so DIVIDE over the anchor row reproduces the live measure
    exactly, with no drift at all, for that one month. Every day inside a
    projected month carries an equal fraction of that month's modelled
    counts (count / days in month), so summing them back up over the month
    reconstitutes the month's total — the reason the rows are daily is so
    this holds regardless of what grain the report rolls up to.

    The anchor is the last *complete* month's observed rate, so the
    projection leaves the actual line at a point that won't move. The
    current month is projected rather than anchored on: part of a month is
    not a month, and its running rate is the least stable number on the
    chart.
    """
    year_end_date = date(first_forecast_date.year, 12, 31)
    if year_end_est is None or first_forecast_date > year_end_date:
        return []

    def make_row(analyse_dato, type_, verdi, lower, upper, innenfor, total):
        return {
            "indikator":             indikator,
            "analyse_dato":          analyse_dato,
            "type":                  type_,
            "verdi":                 round(float(verdi), 4),
            "nedre_konfidensgrense": None if lower is None else round(float(lower), 4),
            "oevre_konfidensgrense": None if upper is None else round(float(upper), 4),
            "innenfor_prognose":     round(float(innenfor), 4),
            "produserte_prognose":   round(float(total), 4),
            "prognose_aarsslutt":    year_end_est,
            "nedre_aarsslutt":       ci_lower,
            "oevre_aarsslutt":       ci_upper,
            "kjoert_tidspunkt":      kjoert_tidspunkt,
            "kjoere_id":             kjoere_id,
        }

    rows = []
    if anchor_date is not None and anchor_total:
        anchor_rate = anchor_innenfor / anchor_total
        # Observed, not projected — no band, and the real counts, not
        # modelled ones, so this row reproduces the live measure exactly.
        rows.append(make_row(
            anchor_date, "Anker", anchor_rate, anchor_rate, anchor_rate,
            anchor_innenfor, anchor_total
        ))

    projected = {}
    for mnd in range(first_forecast_date.month, 13):
        if mnd not in month_ratios or mnd not in month_volumes:
            continue
        estimate, lower, upper = project_month_rate(
            year_end_est, ci_lower, ci_upper, month_ratios[mnd]
        )
        volume = month_volumes[mnd]
        days_in_month = pd.Timestamp(year_end_date.year, mnd, 1).days_in_month
        if estimate is not None and volume > 0 and days_in_month > 0:
            projected[mnd] = (
                estimate, lower, upper,
                estimate * volume / days_in_month,   # innenfor, one day's share
                volume / days_in_month,               # total, one day's share
            )

    if not projected:
        return []

    for timestamp in pd.date_range(first_forecast_date, year_end_date, freq="D"):
        on_date = timestamp.date()
        if on_date.month not in projected:
            continue
        estimate, lower, upper, innenfor, total = projected[on_date.month]
        rows.append(make_row(on_date, "Prognose", estimate, lower, upper, innenfor, total))

    return rows


OUTPUT_COLUMNS = {
    "indikator",
    "analyse_dato",
    "type",
    "verdi",
    "nedre_konfidensgrense",
    "oevre_konfidensgrense",
    "innenfor_prognose",
    "produserte_prognose",
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

        # innenfor_prognose/produserte_prognose are what let a report
        # aggregate this row the way it aggregates the real data (SUM/SUM,
        # not an average of rates) — so they have to actually agree with
        # `verdi`, or a report reading the counts and one reading `verdi`
        # would show two different numbers for the same row.
        innenfor = row["innenfor_prognose"]
        produserte = row["produserte_prognose"]
        if innenfor is not None and produserte is not None:
            if innenfor < 0 or produserte < 0:
                raise ValueError(f"Negative faser count in row {row_number}")
            if innenfor > produserte:
                raise ValueError(
                    f"innenfor_prognose exceeds produserte_prognose in row {row_number}"
                )
            if produserte > 0 and verdi is not None:
                implied_rate = innenfor / produserte
                if abs(implied_rate - verdi) > 0.01:
                    raise ValueError(
                        f"verdi does not match innenfor/produserte in row {row_number}: "
                        f"{verdi} vs {implied_rate}"
                    )


# =============================================================================
# CELL 4 — Compute projections per indicator
# =============================================================================

results = []
indicators = monthly["indikator"].unique()

for indikator in indicators:

    # Three seasonal models, three jobs. The YTD ratios drive the year-end
    # estimate — cumulative YTD is the stable thing to extrapolate from. The
    # per-month rate ratios turn that estimate back into the period rates the
    # report plots. The per-month volume ratios turn this year's observed
    # caseload into a projected faser count per remaining month, which is
    # what lets the output carry counts instead of a bare rate.
    ratios = seasonal_ratios(
        monthly, indikator, CURRENT_YEAR,
        min_years=MIN_YEARS, trim_n=TRIM_N
    )
    month_ratios = monthly_rate_ratios(
        monthly, indikator, CURRENT_YEAR,
        min_years=MIN_YEARS, trim_n=TRIM_N
    )
    volume_ratios = monthly_volume_ratios(
        monthly, indikator, CURRENT_YEAR,
        min_years=MIN_YEARS, trim_n=TRIM_N
    )

    if ratios is None or month_ratios is None or volume_ratios is None:
        print(f"Skipping {indikator} — insufficient history")
        continue

    # Current year actuals
    current_ytd    = compute_ytd(monthly, indikator, CURRENT_YEAR)
    current_rates  = monthly_rates(monthly, indikator, CURRENT_YEAR)
    current_counts = monthly_counts(monthly, indikator, CURRENT_YEAR)

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
    anchor_innenfor = anchor_total = None
    if last_complete_month in current_counts:
        anchor_innenfor, anchor_total = current_counts[last_complete_month]
        anchor_date = (pd.Timestamp(CURRENT_YEAR, last_complete_month, 1)
                       + pd.offsets.MonthEnd(0)).date()

    # This year's observed monthly volume, used to extrapolate a projected
    # faser count for the remaining months — the same "share observed so
    # far, scaled up" pattern project_year_end uses for the rate.
    current_year_totals = {m: total for m, (_, total) in current_counts.items()}
    month_volumes = project_month_volumes(
        current_year_totals, last_complete_month, volume_ratios
    )
    if not month_volumes:
        print(f"Skipping {indikator} — no volume basis to project from")
        continue

    # Actual rates for the months up to the anchor are NOT written here — that
    # is the report's own `Faser innen frist %` measure against
    # saksbehandling.faser, no algorithm needed. This table stores what a live
    # measure structurally can't produce: the projection and its band. The
    # anchor row is the one observed value it duplicates, on purpose — with
    # its real counts, not modelled ones — so the projection has the actual
    # line to leave from.
    # See Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md.
    indicator_rows = build_forecast_rows(
        indikator, date(CURRENT_YEAR, last_complete_month + 1, 1),
        month_ratios, month_volumes,
        year_end_est, ci_lo, ci_hi,
        datetime.now(), BATCH_ID,
        anchor_date=anchor_date, anchor_innenfor=anchor_innenfor, anchor_total=anchor_total
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
        StructField("innenfor_prognose", DoubleType(), True),
        StructField("produserte_prognose", DoubleType(), True),
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

    # ALTER TABLE appends new columns at the end, so the table's column order
    # need not match output_schema's. Select in the table's order — an append
    # that lines columns up by position would otherwise write the wrong ones.
    results_spark = results_spark.select(
        *[field.name for field in spark.table("analyser.frist_prognose").schema]
    )

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
