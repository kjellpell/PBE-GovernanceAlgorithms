
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
CREATE TABLE IF NOT EXISTS analyser.ewma_analyse (
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

# Pivoted companion table — one row per indikator/måned, columns per måltall.
# Joins cleanly onto faser via the shared indikator/dato dimensioner without fan-out.
spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.ewma_analyse_wide (
    indikator                  STRING      NOT NULL,
    analyse_dato                DATE        NOT NULL,
    verdi_frist                 DOUBLE,
    ewma_sakte_frist             DOUBLE,
    ewma_rask_frist              DOUBLE,
    ewma_helning_sakte_frist     DOUBLE,
    ewma_helning_rask_frist      DOUBLE,
    trendretning_frist           STRING,
    verdi_tid                   DOUBLE,
    ewma_sakte_tid               DOUBLE,
    ewma_rask_tid                DOUBLE,
    ewma_helning_sakte_tid       DOUBLE,
    ewma_helning_rask_tid        DOUBLE,
    trendretning_tid             STRING,
    verdi_prod                  DOUBLE,
    ewma_sakte_prod              DOUBLE,
    ewma_rask_prod               DOUBLE,
    ewma_helning_sakte_prod      DOUBLE,
    ewma_helning_rask_prod       DOUBLE,
    trendretning_prod            STRING,
    kjoert_tidspunkt             TIMESTAMP   NOT NULL,
    id                           STRING      NOT NULL
)
USING DELTA
COMMENT 'Pivotert ewma_analyse — én rad per indikator og måned, med egne kolonner per måltall. Kan joines direkte inn i faser via delte indikator- og datodimensjoner uten fan-out.'
""")

print("ewma_analyse- og ewma_analyse_wide-tabellene er klare")


# =============================================================================
# CELL 2 — Load monthly data
# =============================================================================

# Monthly frist% per indicator
monthly_frist = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(pr.sluttmilepaeldato) * 100 + MONTH(pr.sluttmilepaeldato)) AS period,
                CASE
                    WHEN COUNT(CASE WHEN pr.frist_dager IS NOT NULL THEN 1 END) = 0 THEN NULL
                        ELSE COUNT(CASE WHEN pr.innenfor_frist = 1 THEN 1 END) * 1.0
                         / COUNT(CASE WHEN pr.frist_dager IS NOT NULL THEN 1 END)
                END AS verdi
            FROM saksbehandling.faser pr
            INNER JOIN felles.indikator indikatorer
                ON indikatorer.pk_indikator = pr.indikator
            WHERE pr.sluttmilepaeldato IS NOT NULL
                AND pr.frist_dager IS NOT NULL
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
        INNER JOIN felles.indikator indikatorer
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
        INNER JOIN felles.indikator indikatorer
            ON indikatorer.pk_indikator = pr.indikator
        WHERE pr.indikator NOT LIKE '%avtalt%'
            AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) >= {START_YEAR}
        GROUP BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                         MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
        ORDER BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                         MONTH(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
""").toPandas()

# Spark DECIMAL columns (e.g. AVG(tidsbruk)) arrive as decimal.Decimal objects,
# which cannot be mixed with floats in the EWMA arithmetic.
for _df in (monthly_frist, monthly_tid, monthly_prod):
    _df["verdi"]  = pd.to_numeric(_df["verdi"], errors="coerce").astype("float64")
    _df["period"] = pd.to_numeric(_df["period"], errors="coerce").astype("int64")

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
    results_spark.write.mode("overwrite").saveAsTable("analyser.ewma_analyse")

    print(f"ewma_analyse skrevet: {len(all_rows):,} rader")

    # Active trends for current period
    spark.sql(f"""
            SELECT indikator, maaltall,
                            ROUND(verdi,       3) AS verdi,
                            ROUND(ewma_sakte,  3) AS ewma_sakte,
                            ROUND(ewma_rask,   3) AS ewma_rask,
                            trendretning
            FROM analyser.ewma_analyse
            WHERE analyse_dato = (SELECT MAX(analyse_dato) FROM analyser.ewma_analyse)
                AND maaltall = 'Fristprosent'
                AND trendretning != 'Stabil'
            ORDER BY trendretning, indikator
    """).show(30, truncate=False)

    # Pivot to one row per indikator/måned so it can join onto faser without fan-out
    MAALTALL_SUFFIX = {
        "Fristprosent":           "frist",
        "Behandlingstid":         "tid",
        "Produksjonsdifferanse":  "prod",
    }
    df_wide_src = df.copy()
    df_wide_src["maaltall"] = df_wide_src["maaltall"].map(MAALTALL_SUFFIX)

    df_wide = df_wide_src.pivot_table(
        index=["indikator", "analyse_dato"],
        columns="maaltall",
        values=["verdi", "ewma_sakte", "ewma_rask", "ewma_helning_sakte", "ewma_helning_rask", "trendretning"],
        aggfunc="first",
    )
    df_wide.columns = [f"{metric}_{suffix}" for metric, suffix in df_wide.columns]
    df_wide = df_wide.reset_index()
    df_wide["kjoert_tidspunkt"] = datetime.now()
    df_wide["id"] = BATCH_ID

    results_wide_spark = spark.createDataFrame(df_wide)
    results_wide_spark.write.mode("overwrite").saveAsTable("analyser.ewma_analyse_wide")

    print(f"ewma_analyse_wide skrevet: {len(df_wide):,} rader")


# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see EWMA_POWERBI_DAX.md
# =============================================================================
