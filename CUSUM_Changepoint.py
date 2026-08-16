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
#   cusum_analyse — løpende CUSUM-verdier og signalflagg
#   pelt_analyse  — oppdagede endringspunkter med gjennomsnitt før/etter
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
CREATE TABLE IF NOT EXISTS cusum_analyse (
    indikator        STRING      NOT NULL,
    maaltall         STRING      NOT NULL,
    granularitet     STRING      NOT NULL,
    periode          INT         NOT NULL,
    verdi            DOUBLE,
    cusum_positiv    DOUBLE,
    cusum_negativ    DOUBLE,
    signal           BOOLEAN     NOT NULL,
    signalretning    STRING,
    kjoert_tidspunkt TIMESTAMP   NOT NULL,
    kjoere_id        STRING      NOT NULL
)
USING DELTA
COMMENT 'CUSUM-driftdeteksjon per indikator og måltall. signal=true angir statistisk signifikant vedvarende drift. signalretning er Økning eller Nedgang.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS pelt_analyse (
    indikator                   STRING      NOT NULL,
    maaltall                    STRING      NOT NULL,
    granularitet                STRING      NOT NULL,
    pelt_periode                INT         NOT NULL,
    gjennomsnitt_foer           DOUBLE,
    gjennomsnitt_etter          DOUBLE,
    endringsstoerrelse          DOUBLE,
    endringsretning             STRING,
    antall_observasjoner_foer   INT,
    antall_observasjoner_etter  INT,
    kjoert_tidspunkt            TIMESTAMP   NOT NULL,
    kjoere_id                   STRING      NOT NULL
)
USING DELTA
COMMENT 'Oppdagede strukturelle endringspunkter per indikator og måltall. endringsstoerrelse er gjennomsnitt_etter minus gjennomsnitt_foer. endringsretning er Økning eller Nedgang.'
""")

print("Output tables ready")


# =============================================================================
# CELL 2 — Load data
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
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.frist IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
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
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.tidsbruk IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
        GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
        ORDER BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

# Monthly production balance per indicator
monthly_prod = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) * 100 +
                 MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))) AS period,
                COUNT(CASE WHEN pr.startmilepaeldato IS NOT NULL THEN 1 END)
                - COUNT(CASE WHEN pr.sluttmilepaeldato IS NOT NULL THEN 1 END) AS verdi
        FROM saksbehandling.faser pr
        WHERE pr.indikator NOT LIKE '%avtalt%'
            AND YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) >= {START_YEAR}
        GROUP BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                         MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
        ORDER BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                         MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
""").toPandas()

# Weekly frist% per indicator
weekly_frist = spark.sql(f"""
    SELECT
        pr.indikator,
        (YEAR(pr.sluttmilepaeldato) * 100
         + WEEKOFYEAR(pr.sluttmilepaeldato))       AS period,
        CASE
            WHEN COUNT(CASE WHEN pr.frist IS NOT NULL
                            THEN 1 END) = 0 THEN NULL
            ELSE COUNT(CASE WHEN pr.innenfor_frist = 1
                            THEN 1 END) * 1.0
                 / COUNT(CASE WHEN pr.frist IS NOT NULL
                              THEN 1 END)
        END                                                         AS verdi
    FROM saksbehandling.faser pr
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.frist IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
        GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), WEEKOFYEAR(pr.sluttmilepaeldato)
        ORDER BY pr.indikator, YEAR(pr.sluttmilepaeldato), WEEKOFYEAR(pr.sluttmilepaeldato)
""").toPandas()

# Weekly production balance
weekly_prod = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) * 100 +
                         WEEKOFYEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))) AS period,
                        COUNT(CASE WHEN pr.startmilepaeldato IS NOT NULL THEN 1 END)
                        - COUNT(CASE WHEN pr.sluttmilepaeldato IS NOT NULL THEN 1 END) AS verdi
                FROM saksbehandling.faser pr
                WHERE pr.indikator NOT LIKE '%avtalt%'
                    AND YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) >= {START_YEAR}
                GROUP BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                                 WEEKOFYEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
                ORDER BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                                 WEEKOFYEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
""").toPandas()

print("Data loaded")
print(f"  Månedlig frist: {monthly_frist['indikator'].nunique()} indikatorer")
print(f"  Ukentlig frist: {weekly_frist['indikator'].nunique()} indikatorer")


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
    retning    = np.where(cusum_pos > h, "Økning",
                 np.where(cusum_neg > h, "Nedgang", None))

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
    min_obs = MIN_MONTHLY if granularitet == "Månedlig" else MIN_WEEKLY

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
            "pelt_periode":                int(periods[bp]) if bp < len(periods) else None,
            "gjennomsnitt_foer":           round(mean_before, 4),
            "gjennomsnitt_etter":          round(mean_after, 4),
            "endringsstoerrelse":          round(shift, 4),
            "endringsretning":             "Økning" if shift > 0 else "Nedgang",
            "antall_observasjoner_foer":   len(before),
            "antall_observasjoner_etter":  len(after),
        })
        prev = bp

    return results


# =============================================================================
# CELL 5 — Run on all series
# =============================================================================

# Define what to process
# (label, dataframe, granularitet, min_obs)
series_configs = [
    ("Fristprosent",            monthly_frist, "Månedlig", MIN_MONTHLY),
    ("Behandlingstid",          monthly_tid,   "Månedlig", MIN_MONTHLY),
    ("Produksjonsdifferanse",   monthly_prod,  "Månedlig", MIN_MONTHLY),
    ("Fristprosent",            weekly_frist,  "Ukentlig", MIN_WEEKLY),
    ("Produksjonsdifferanse",   weekly_prod,   "Ukentlig", MIN_WEEKLY),
]

cusum_rows       = []
changepoint_rows = []

for metrikk, df, granularitet, min_obs in series_configs:

    for indikator in df["indikator"].unique():
        ind_data = (
            df[df["indikator"] == indikator]
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
                            "indikator":       indikator,
                            "maaltall":        metrikk,
                            "granularitet":    granularitet,
                            "periode":         int(idx),
                            "verdi":           float(ind_data[idx]) if idx in ind_data else None,
                            "cusum_positiv":   round(float(row["cusum_pos"]), 4),
                            "cusum_negativ":   round(float(row["cusum_neg"]), 4),
                            "signal":          bool(row["signal"]),
                            "signalretning":   row["signal_direction"],
                            "kjoert_tidspunkt": datetime.now(),
                            "kjoere_id":       BATCH_ID,
                        })

        # ── Changepoint (monthly only for stability) ───────────────
        if granularitet == "Månedlig":
            breakpoints = run_changepoint(ind_data, granularitet)
            for cp in extract_changepoint_stats(ind_data, breakpoints):
                        changepoint_rows.append({
                            "indikator":    indikator,
                            "maaltall":     metrikk,
                            "granularitet": granularitet,
                            **cp,
                            "kjoert_tidspunkt": datetime.now(),
                            "kjoere_id":    BATCH_ID,
                        })

        # Weekly changepoints — separate pass
        if granularitet == "Ukentlig":
            breakpoints = run_changepoint(ind_data, granularitet)
            for cp in extract_changepoint_stats(ind_data, breakpoints):
                changepoint_rows.append({
                    "indikator":    indikator,
                    "maaltall":     metrikk,
                    "granularitet": granularitet,
                    **cp,
                    "kjoert_tidspunkt": datetime.now(),
                    "kjoere_id":    BATCH_ID,
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
    cusum_spark.write.mode("overwrite").saveAsTable("cusum_analyse")
    print(f"cusum_analyse skrevet: {len(cusum_rows)} rader")

if changepoint_rows:
    cp_df = pd.DataFrame(changepoint_rows)
    cp_spark = spark.createDataFrame(cp_df)
    cp_spark.write.mode("overwrite").saveAsTable("pelt_analyse")
    print(f"pelt_analyse skrevet: {len(changepoint_rows)} rader")

# Summary — active signals
if cusum_rows:
    spark.sql(f"""
                SELECT indikator, maaltall, granularitet,
                             MAX(periode) AS siste_signalperiode,
                             MAX(signalretning) AS retning
                FROM cusum_analyse
                WHERE signal = TRUE
                    AND kjoere_id = '{BATCH_ID}'
                GROUP BY indikator, maaltall, granularitet
                ORDER BY maaltall, indikator
    """).show(50, truncate=False)

if changepoint_rows:
    spark.sql(f"""
     SELECT indikator, maaltall, granularitet,
         pelt_periode,
         ROUND(gjennomsnitt_foer, 3) AS foer,
         ROUND(gjennomsnitt_etter, 3) AS etter,
         ROUND(endringsstoerrelse, 3) AS endring,
         endringsretning
     FROM pelt_analyse
     WHERE kjoere_id = '{BATCH_ID}'
     ORDER BY ABS(endringsstoerrelse) DESC
    """).show(50, truncate=False)


# =============================================================================
# CELL 7 — Power BI visual guidance and DAX measures
# =============================================================================
#
# OUTPUT TABLES → VISUALS
#
# cusum_analyse:
#
#   LINE CHART — CUSUM values over time
#     X axis:  periode (Regnskapsperiode or Ukenummer)
#     Y axis:  cusum_positiv (upper line), cusum_negativ (lower line, negate for display)
#     Ref line: constant at CUSUM_H threshold (default 5.0) — horizontal line
#     Filter:  indikator slicer, maaltall slicer (Fristprosent / Behandlingstid / Produksjonsdifferanse)
#              granularitet slicer (Månedlig / Ukentlig)
#     Colour:  cusum_positiv in blue, cusum_negativ in red
#     Signal:  conditional format background on data points where signal = TRUE
#              — amber fill so active signals stand out on the line
#     Reading: lines drifting toward the threshold = gradual deterioration
#              building. Line crossing threshold = structural shift confirmed.
#              Lines returning to zero = process stabilised.
#
#   TABLE — Active CUSUM signals
#     Columns: indikator | maaltall | granularitet | signalretning | periode
#     Filter:  signal = TRUE, most recent periode per indicator
#     Sort:    metrikk, then indikator
#     Purpose: governance team morning check — which indicators have
#              active drift signals right now
#
# pelt_analyse:
#
#   LINE CHART with changepoint markers — overlay on existing frist% or
#   behandlingstid time series charts
#     Add a vertical reference line at pelt_periode
#     Show gjennomsnitt_foer as a horizontal segment before the changepoint
#     Show gjennomsnitt_etter as a horizontal segment after the changepoint
#     The visual gap between the two horizontal segments = endringsstoerrelse
#     In Power BI: use a calculated column or measure to draw segments,
#     or use the Analytics pane "average line" filtered to pre/post periods
#
#   TABLE — Detected changepoints
#     Columns: indikator | maaltall | pelt_periode | gjennomsnitt_foer
#              | gjennomsnitt_etter | endringsstoerrelse | endringsretning | granularitet
#     Sort:    ABS(endringsstoerrelse) DESC — largest shifts first
#     Filter:  granularitet slicer so team can toggle Månedlig/Ukentlig view
#
# DAX MEASURES — add to cusum_analyse table in semantic model

# Filters to most recent period per indicator for use in summary visuals.

# Har aktiv CUSUM signal =
# VAR SistePeriode =
#     CALCULATE(
#         MAX(cusum_analyse[periode]),
#         ALLEXCEPT(cusum_analyse, cusum_analyse[indikator], cusum_analyse[maaltall])
#     )
# RETURN
#     CALCULATE(
#         MAX(cusum_analyse[signal]),
#         cusum_analyse[periode] = SistePeriode
#     ) = TRUE()

# Antall aktive signaler =
# CALCULATE(
#     DISTINCTCOUNT(cusum_analyse[indikator]),
#     cusum_analyse[signal] = TRUE(),
#     cusum_analyse[periode] = MAX(cusum_analyse[periode])
# )

# DAX MEASURES — add to pelt_analyse table

# Siste endringspunkt periode =
# CALCULATE(
#     MAX(pelt_analyse[pelt_periode]),
#     ALLEXCEPT(pelt_analyse, pelt_analyse[indikator],
#               pelt_analyse[maaltall])
# )

# Endringspunkt størrelse =
# CALCULATE(
#     MAX(pelt_analyse[endringsstoerrelse]),
#     pelt_analyse[pelt_periode] = [Siste endringspunkt periode]
# )
