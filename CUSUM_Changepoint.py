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
# Drill-down:
#   For the most recent changepoint per indikator/maaltall/granularitet,
#   breaks the shift down by enhet (team) and fasetittel (process step) so
#   a signal can be pinpointed to a team or a step, not just a product group.
#   Saksbehandler is deliberately excluded — too thin per-segment volume,
#   and individual-level automated flagging is out of scope for this layer.
#
# Output tables (addon signal only — no raw value; that's a live DAX
# measure against saksbehandling.faser, see CUSUM_Changepoint_POWERBI_DAX.md):
#   cusum_analyse     — løpende CUSUM-verdier og signalflagg
#   pelt_analyse      — oppdagede endringspunkter med gjennomsnitt før/etter
#   pelt_analyse_detaljer — nedbryting av siste endringspunkt per enhet/fasetittel
#
# Schedule: nightly, after main data pipeline.
# Requires: ruptures (PELT only — CUSUM runs without it). Do NOT install it with
# an inline `%pip install` cell — this tenant has inline library installation
# disabled, and that magic command fails the whole notebook run with
# MagicUsageError before any Python cell executes. Instead add "ruptures" as a
# public library in a Fabric Environment item and attach that environment to
# this notebook (or set it as the workspace default). See
# https://learn.microsoft.com/en-us/fabric/data-engineering/library-management#inline-installation
# Minimum history: 24 monthly / 52 weekly observations per indicator.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, TimestampType, DateType,
)
import pandas as pd
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()
BATCH_ID      = datetime.now().strftime("%Y%m%dT%H%M%S")
MIN_MONTHLY   = 24
MIN_WEEKLY    = 52
CUSUM_K       = 0.5   # allowance parameter — half sigma is standard
CUSUM_H       = 5.0   # decision threshold — 5 sigma cumulative
START_YEAR    = 2015  # exclude data before this year — adjust if older data is reliable

# Anchored baseline window — mu/sigma used to standardise each series are
# computed from only the FIRST this-many observations, not the whole series.
# Using the whole series lets a slow persistent drift get partially absorbed
# into "normal", which dampens CUSUM's sensitivity to exactly the kind of
# shift it exists to catch.
CUSUM_BASELINE_MONTHLY      = 12
CUSUM_BASELINE_WEEKLY       = 26
CUSUM_MIN_POST_BASELINE_OBS = 4   # need at least this many points after the baseline to test anything

DRILLDOWN_DIMENSIONS     = ["enhet", "fasetittel"]  # verify these column names against the Lakehouse schema
MIN_SEGMENT_OBS          = 10   # segments below this are marked utilstrekkelig_volum, not tested
RECENT_CHANGEPOINT_DAYS  = 90   # only drill into changepoints still recent enough to be actionable


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.cusum_analyse (
    indikator        STRING      NOT NULL,
    maaltall         STRING      NOT NULL,
    granularitet     STRING      NOT NULL,
    analyse_dato     DATE        NOT NULL,
    cusum_positiv    DOUBLE,
    cusum_negativ    DOUBLE,
    signal           BOOLEAN     NOT NULL,
    signalretning    STRING,
    kjoert_tidspunkt TIMESTAMP   NOT NULL,
    kjoere_id        STRING      NOT NULL
)
USING DELTA
COMMENT 'CUSUM-driftdeteksjon per indikator og måltall. signal=true angir statistisk signifikant vedvarende drift. signalretning er Økning eller Nedgang. Rå verdi (Fristprosent/Behandlingstid/Produksjonsdifferanse) er IKKE lagret her — den er en live DAX-mål mot saksbehandling.faser, join på indikator+maaltall+granularitet+analyse_dato.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.pelt_analyse (
    indikator                   STRING      NOT NULL,
    maaltall                    STRING      NOT NULL,
    granularitet                STRING      NOT NULL,
    analyse_dato                DATE        NOT NULL,
    endringspunkt_id            STRING      NOT NULL,
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
COMMENT 'Oppdagede strukturelle endringspunkter per indikator og måltall. endringspunkt_id (indikator|maaltall|granularitet|analyse_dato) er nøkkelen til pelt_analyse_detaljer. endringsstoerrelse er gjennomsnitt_etter minus gjennomsnitt_foer. endringsretning er Økning eller Nedgang.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.pelt_analyse_detaljer (
    indikator                   STRING      NOT NULL,
    maaltall                    STRING      NOT NULL,
    granularitet                STRING      NOT NULL,
    dimensjon                   STRING      NOT NULL,
    dimensjonsverdi             STRING      NOT NULL,
    analyse_dato                DATE        NOT NULL,
    endringspunkt_id            STRING      NOT NULL,
    gjennomsnitt_foer           DOUBLE,
    gjennomsnitt_etter          DOUBLE,
    endringsstoerrelse          DOUBLE,
    bidrag_til_endring          DOUBLE,
    antall_observasjoner_foer   INT,
    antall_observasjoner_etter  INT,
    tilstrekkelig_volum         BOOLEAN     NOT NULL,
    kjoert_tidspunkt            TIMESTAMP   NOT NULL,
    kjoere_id                   STRING      NOT NULL
)
USING DELTA
COMMENT 'Nedbryting av siste pelt_analyse-endringspunkt per indikator/maaltall, fordelt på enhet og fasetittel. endringspunkt_id er samme verdi som i pelt_analyse for raden dette bryter ned — bruk den som relasjonsnøkkel i den semantiske modellen (én-til-mange fra pelt_analyse). bidrag_til_endring er segmentets andel av det totale skiftet (summerer omtrent til pelt_analyse.endringsstoerrelse). tilstrekkelig_volum=false betyr for få observasjoner til å stole på tallene.'
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
            WHEN COUNT(CASE WHEN pr.frist_dager IS NOT NULL THEN 1 END) = 0 THEN NULL
            ELSE COUNT(CASE WHEN pr.innenfor_frist = 1 THEN 1 END) * 1.0
                 / COUNT(CASE WHEN pr.frist_dager IS NOT NULL THEN 1 END)
        END AS verdi
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.frist_dager IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
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
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.tidsbruk IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
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
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
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
            WHEN COUNT(CASE WHEN pr.frist_dager IS NOT NULL
                            THEN 1 END) = 0 THEN NULL
            ELSE COUNT(CASE WHEN pr.innenfor_frist = 1
                            THEN 1 END) * 1.0
                 / COUNT(CASE WHEN pr.frist_dager IS NOT NULL
                              THEN 1 END)
        END                                                         AS verdi
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.frist_dager IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
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
                INNER JOIN felles.indikator indikator
                    ON indikator.pk_indikator = pr.indikator
                WHERE pr.indikator NOT LIKE '%avtalt%'
                    AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
                    AND YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)) >= {START_YEAR}
                GROUP BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                                 WEEKOFYEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
                ORDER BY pr.indikator, YEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)),
                                 WEEKOFYEAR(COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato))
""").toPandas()

# Spark DECIMAL columns (e.g. AVG(tidsbruk)) arrive as decimal.Decimal objects,
# which cannot be mixed with floats in the CUSUM arithmetic.
for _df in (monthly_frist, monthly_tid, monthly_prod, weekly_frist, weekly_prod):
    _df["verdi"]  = pd.to_numeric(_df["verdi"], errors="coerce").astype("float64")
    _df["period"] = pd.to_numeric(_df["period"], errors="coerce").astype("int64")

print("Data loaded")
print(f"  Månedlig frist: {monthly_frist['indikator'].nunique()} indikator")
print(f"  Ukentlig frist: {weekly_frist['indikator'].nunique()} indikator")


def _periode_to_date(period_int, granularitet):
    """Convert YYYYMM (monthly) or YYYYWW (weekly) integer to an end-of-period date."""
    year  = int(period_int) // 100
    unit  = int(period_int) % 100
    if granularitet == "Månedlig":
        return (pd.Timestamp(year, unit, 1) + pd.offsets.MonthEnd(0)).date()
    else:
        # ISO week — return the Sunday (last day) of that week
        return pd.Timestamp.fromisocalendar(year, unit, 7).date()


def make_endringspunkt_id(indikator, maaltall, granularitet, analyse_dato):
    """Shared join key between pelt_analyse and pelt_analyse_detaljer —
    Power BI relationships need a single column, not the underlying
    composite (indikator, maaltall, granularitet, analyse_dato)."""
    return f"{indikator}|{maaltall}|{granularitet}|{analyse_dato.isoformat()}"



def to_float_series(series):
    """Spark DECIMAL columns arrive as decimal.Decimal, which cannot be mixed
    with floats in the arithmetic below."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def run_cusum(series, k=CUSUM_K, h=CUSUM_H, baseline_obs=CUSUM_BASELINE_MONTHLY):
    """
    Two-sided CUSUM on a standardised series.
    k = allowance (typically 0.5 * expected shift in sigma units)
    h = decision threshold (typically 4-5)
    baseline_obs = number of leading observations used to compute mu/sigma —
        an anchored reference period, not the whole series. Otherwise a slow
        persistent drift gets partially absorbed into "normal", dampening
        the very sensitivity CUSUM is meant to provide. The full series
        (including the baseline window itself) is standardised against this
        fixed mu/sigma before running the CUSUM recursion.

    Returns DataFrame with cusum_pos, cusum_neg, signal, signal_retning.
    """
    if len(series) < 8:
        return None

    series = to_float_series(series)
    values = series.dropna().values
    if len(values) < 8:
        return None

    if len(values) < baseline_obs + CUSUM_MIN_POST_BASELINE_OBS:
        return None

    baseline = values[:baseline_obs]
    mu    = np.mean(baseline)
    sigma = np.std(baseline)
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

    values = to_float_series(series).dropna().values
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


# Per-måltall value expression and date column, reused by the drilldown
# queries below — mirrors the aggregation logic in CELL 2.
METRIC_SQL = {
    "Fristprosent": {
        "value_expr": """CASE WHEN COUNT(CASE WHEN pr.frist_dager IS NOT NULL THEN 1 END) = 0 THEN NULL
                          ELSE COUNT(CASE WHEN pr.innenfor_frist = 1 THEN 1 END) * 1.0
                               / COUNT(CASE WHEN pr.frist_dager IS NOT NULL THEN 1 END) END""",
        "date_col":    "pr.sluttmilepaeldato",
        "extra_where": "pr.sluttmilepaeldato IS NOT NULL AND pr.frist_dager IS NOT NULL",
    },
    "Behandlingstid": {
        "value_expr":  "AVG(pr.tidsbruk)",
        "date_col":    "pr.sluttmilepaeldato",
        "extra_where": "pr.sluttmilepaeldato IS NOT NULL AND pr.tidsbruk IS NOT NULL",
    },
    "Produksjonsdifferanse": {
        "value_expr": """COUNT(CASE WHEN pr.startmilepaeldato IS NOT NULL THEN 1 END)
                          - COUNT(CASE WHEN pr.sluttmilepaeldato IS NOT NULL THEN 1 END)""",
        "date_col":    "COALESCE(pr.sluttmilepaeldato, pr.startmilepaeldato)",
        "extra_where": "1=1",
    },
}


def run_drilldown(indikator, maaltall, granularitet, breakpoint_dato, dimensjon):
    """
    Split the flagged indikator/maaltall shift by one dimension (enhet or
    fasetittel), using the same før/etter boundary PELT already found.
    Returns a list of row dicts, one per segment with enough data.
    """
    cfg = METRIC_SQL[maaltall]
    breakpoint_str = breakpoint_dato.isoformat()
    indikator_escaped = indikator.replace("'", "''")

    segments = spark.sql(f"""
        SELECT
            pr.{dimensjon} AS segment,
            CASE WHEN {cfg['date_col']} < DATE'{breakpoint_str}' THEN 'foer' ELSE 'etter' END AS periode,
            {cfg['value_expr']} AS verdi,
            COUNT(*) AS antall
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator ON indikator.pk_indikator = pr.indikator
        WHERE pr.indikator = '{indikator_escaped}'
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR({cfg['date_col']}) >= {START_YEAR}
            AND {cfg['extra_where']}
            AND pr.{dimensjon} IS NOT NULL
        GROUP BY pr.{dimensjon},
            CASE WHEN {cfg['date_col']} < DATE'{breakpoint_str}' THEN 'foer' ELSE 'etter' END
    """).toPandas()

    if segments.empty:
        return []

    segments["verdi"]  = pd.to_numeric(segments["verdi"], errors="coerce").astype("float64")
    segments["antall"] = segments["antall"].astype("int64")

    pivot = segments.pivot(index="segment", columns="periode", values=["verdi", "antall"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reindex(columns=["verdi_foer", "verdi_etter", "antall_foer", "antall_etter"])

    total_antall_foer  = pivot["antall_foer"].sum()
    total_antall_etter = pivot["antall_etter"].sum()

    rows = []
    for segment, r in pivot.iterrows():
        if pd.isna(r["verdi_foer"]) or pd.isna(r["verdi_etter"]):
            continue

        tilstrekkelig_volum = (r["antall_foer"] >= MIN_SEGMENT_OBS) and (r["antall_etter"] >= MIN_SEGMENT_OBS)

        # weighted contribution — exact decomposition of the aggregate shift for
        # mean/ratio måltall; for Produksjonsdifferanse (already additive) the
        # volume weights cancel out to the raw segment delta.
        bidrag = None
        if total_antall_foer and total_antall_etter:
            bidrag = (
                (r["antall_etter"] / total_antall_etter) * r["verdi_etter"]
                - (r["antall_foer"] / total_antall_foer) * r["verdi_foer"]
            )

        rows.append({
            "indikator":                  indikator,
            "maaltall":                   maaltall,
            "granularitet":               granularitet,
            "dimensjon":                  dimensjon,
            "dimensjonsverdi":            str(segment),
            "analyse_dato":               breakpoint_dato,
            "endringspunkt_id":           make_endringspunkt_id(
                indikator, maaltall, granularitet, breakpoint_dato
            ),
            "gjennomsnitt_foer":          round(float(r["verdi_foer"]), 4),
            "gjennomsnitt_etter":         round(float(r["verdi_etter"]), 4),
            "endringsstoerrelse":         round(float(r["verdi_etter"] - r["verdi_foer"]), 4),
            "bidrag_til_endring":         round(float(bidrag), 4) if bidrag is not None else None,
            "antall_observasjoner_foer":  int(r["antall_foer"]),
            "antall_observasjoner_etter": int(r["antall_etter"]),
            "tilstrekkelig_volum":        bool(tilstrekkelig_volum),
            "kjoert_tidspunkt":           datetime.now(),
            "kjoere_id":                  BATCH_ID,
        })

    return rows


def extract_changepoint_stats(series, breakpoints, granularitet):
    """
    For each detected breakpoint, compute before/after mean and shift.
    Returns list of dicts.
    """
    series = to_float_series(series)
    values = series.dropna().values
    periods = series.dropna().index.tolist()
    results = []

    prev = 0
    for i, bp in enumerate(breakpoints):
        next_bp = breakpoints[i + 1] if i + 1 < len(breakpoints) else len(values)
        before = values[prev:bp]
        after  = values[bp:next_bp]

        if len(before) < 3 or len(after) < 3:
            prev = bp
            continue

        mean_before = float(np.mean(before))
        mean_after  = float(np.mean(after))
        shift       = mean_after - mean_before

        results.append({
            "analyse_dato":                _periode_to_date(periods[bp], granularitet) if bp < len(periods) else None,
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

CUSUM_BASELINE_BY_GRANULARITET = {
    "Månedlig": CUSUM_BASELINE_MONTHLY,
    "Ukentlig": CUSUM_BASELINE_WEEKLY,
}

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
        baseline_obs = CUSUM_BASELINE_BY_GRANULARITET[granularitet]
        cusum = run_cusum(ind_data, baseline_obs=baseline_obs)
        if cusum is not None:
            for idx, row in cusum.iterrows():
                        cusum_rows.append({
                            "indikator":       indikator,
                            "maaltall":        metrikk,
                            "granularitet":    granularitet,
                            "analyse_dato":    _periode_to_date(idx, granularitet),
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
            for cp in extract_changepoint_stats(ind_data, breakpoints, granularitet):
                        changepoint_rows.append({
                            "indikator":    indikator,
                            "maaltall":     metrikk,
                            "granularitet": granularitet,
                            "endringspunkt_id": make_endringspunkt_id(
                                indikator, metrikk, granularitet, cp["analyse_dato"]
                            ),
                            **cp,
                            "kjoert_tidspunkt": datetime.now(),
                            "kjoere_id":    BATCH_ID,
                        })

        # Weekly changepoints — separate pass
        if granularitet == "Ukentlig":
            breakpoints = run_changepoint(ind_data, granularitet)
            for cp in extract_changepoint_stats(ind_data, breakpoints, granularitet):
                changepoint_rows.append({
                    "indikator":    indikator,
                    "maaltall":     metrikk,
                    "granularitet": granularitet,
                    "endringspunkt_id": make_endringspunkt_id(
                        indikator, metrikk, granularitet, cp["analyse_dato"]
                    ),
                    **cp,
                    "kjoert_tidspunkt": datetime.now(),
                    "kjoere_id":    BATCH_ID,
                })

print(f"CUSUM rows computed:       {len(cusum_rows)}")
print(f"Changepoint rows computed: {len(changepoint_rows)}")
print(f"Active CUSUM signals:      "
      f"{sum(1 for r in cusum_rows if r['signal'])}")


# =============================================================================
# CELL 5B — Drill down the most recent changepoint per series by enhet/fasetittel
# =============================================================================

drilldown_rows = []

if changepoint_rows:
    cp_df = pd.DataFrame(changepoint_rows)
    latest_cp = (
        cp_df.sort_values("analyse_dato")
        .groupby(["indikator", "maaltall", "granularitet"])
        .tail(1)
    )
    cutoff = pd.Timestamp.now().date() - pd.Timedelta(days=RECENT_CHANGEPOINT_DAYS)
    latest_cp = latest_cp[latest_cp["analyse_dato"] >= cutoff]

    for _, cp in latest_cp.iterrows():
        for dimensjon in DRILLDOWN_DIMENSIONS:
            drilldown_rows += run_drilldown(
                cp["indikator"], cp["maaltall"], cp["granularitet"],
                cp["analyse_dato"], dimensjon,
            )

print(f"Drilldown rows computed:   {len(drilldown_rows)}")


# =============================================================================
# CELL 6 — Write to Lakehouse
# =============================================================================

# Explicit schemas — pandas infers int64/object, which does not match the
# INT/DOUBLE columns in the Delta tables and breaks the overwrite merge.
CUSUM_SCHEMA = StructType([
    StructField("indikator",        StringType(),    False),
    StructField("maaltall",         StringType(),    False),
    StructField("granularitet",     StringType(),    False),
    StructField("analyse_dato",     DateType(),      False),
    StructField("cusum_positiv",    DoubleType(),    True),
    StructField("cusum_negativ",    DoubleType(),    True),
    StructField("signal",           BooleanType(),   False),
    StructField("signalretning",    StringType(),    True),
    StructField("kjoert_tidspunkt", TimestampType(), False),
    StructField("kjoere_id",        StringType(),    False),
])

PELT_SCHEMA = StructType([
    StructField("indikator",                  StringType(),    False),
    StructField("maaltall",                   StringType(),    False),
    StructField("granularitet",               StringType(),    False),
    StructField("analyse_dato",               DateType(),      False),
    StructField("endringspunkt_id",           StringType(),    False),
    StructField("gjennomsnitt_foer",          DoubleType(),    True),
    StructField("gjennomsnitt_etter",         DoubleType(),    True),
    StructField("endringsstoerrelse",         DoubleType(),    True),
    StructField("endringsretning",            StringType(),    True),
    StructField("antall_observasjoner_foer",  IntegerType(),   True),
    StructField("antall_observasjoner_etter", IntegerType(),   True),
    StructField("kjoert_tidspunkt",           TimestampType(), False),
    StructField("kjoere_id",                  StringType(),    False),
])

DRILLDOWN_SCHEMA = StructType([
    StructField("indikator",                  StringType(),    False),
    StructField("maaltall",                   StringType(),    False),
    StructField("granularitet",               StringType(),    False),
    StructField("dimensjon",                  StringType(),    False),
    StructField("dimensjonsverdi",            StringType(),    False),
    StructField("analyse_dato",               DateType(),      False),
    StructField("endringspunkt_id",           StringType(),    False),
    StructField("gjennomsnitt_foer",          DoubleType(),    True),
    StructField("gjennomsnitt_etter",         DoubleType(),    True),
    StructField("endringsstoerrelse",         DoubleType(),    True),
    StructField("bidrag_til_endring",         DoubleType(),    True),
    StructField("antall_observasjoner_foer",  IntegerType(),   True),
    StructField("antall_observasjoner_etter", IntegerType(),   True),
    StructField("tilstrekkelig_volum",        BooleanType(),   False),
    StructField("kjoert_tidspunkt",           TimestampType(), False),
    StructField("kjoere_id",                  StringType(),    False),
])


def to_records(rows, schema):
    """Coerce dict rows to native Python types matching the target schema."""
    casters = {
        StringType():  lambda v: None if v is None else str(v),
        IntegerType(): lambda v: None if v is None else int(v),
        DoubleType():  lambda v: None if v is None else float(v),
        BooleanType(): lambda v: None if v is None else bool(v),
    }
    out = []
    for row in rows:
        values = []
        for field in schema.fields:
            v = row.get(field.name)
            if v is not None and pd.isna(v):
                v = None
            cast = casters.get(field.dataType)
            values.append(cast(v) if cast else v)
        out.append(tuple(values))
    return out


if cusum_rows:
    cusum_spark = spark.createDataFrame(
        to_records(cusum_rows, CUSUM_SCHEMA), schema=CUSUM_SCHEMA
    )
    cusum_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("analyser.cusum_analyse")
    print(f"cusum_analyse skrevet: {len(cusum_rows)} rader")

if changepoint_rows:
    cp_spark = spark.createDataFrame(
        to_records(changepoint_rows, PELT_SCHEMA), schema=PELT_SCHEMA
    )
    cp_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("analyser.pelt_analyse")
    print(f"pelt_analyse skrevet: {len(changepoint_rows)} rader")

if drilldown_rows:
    drilldown_spark = spark.createDataFrame(
        to_records(drilldown_rows, DRILLDOWN_SCHEMA), schema=DRILLDOWN_SCHEMA
    )
    drilldown_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("analyser.pelt_analyse_detaljer")
    print(f"pelt_analyse_detaljer skrevet: {len(drilldown_rows)} rader")

# Summary — active signals
if cusum_rows:
    spark.sql(f"""
                SELECT indikator, maaltall, granularitet,
                             MAX(analyse_dato) AS siste_signaldato,
                             MAX(signalretning) AS retning
                FROM analyser.cusum_analyse
                WHERE signal = TRUE
                    AND kjoere_id = '{BATCH_ID}'
                GROUP BY indikator, maaltall, granularitet
                ORDER BY maaltall, indikator
    """).show(50, truncate=False)

if changepoint_rows:
    spark.sql(f"""
     SELECT indikator, maaltall, granularitet,
         analyse_dato,
         ROUND(gjennomsnitt_foer, 3) AS foer,
         ROUND(gjennomsnitt_etter, 3) AS etter,
         ROUND(endringsstoerrelse, 3) AS endring,
         endringsretning
     FROM analyser.pelt_analyse
     WHERE kjoere_id = '{BATCH_ID}'
     ORDER BY ABS(endringsstoerrelse) DESC
    """).show(50, truncate=False)

if drilldown_rows:
    spark.sql(f"""
     SELECT indikator, maaltall, dimensjon, dimensjonsverdi,
         ROUND(bidrag_til_endring, 3) AS bidrag,
         antall_observasjoner_foer, antall_observasjoner_etter,
         tilstrekkelig_volum
     FROM analyser.pelt_analyse_detaljer
     WHERE kjoere_id = '{BATCH_ID}'
     ORDER BY indikator, maaltall, dimensjon, ABS(bidrag_til_endring) DESC
    """).show(50, truncate=False)


# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see CUSUM_Changepoint_POWERBI_DAX.md
# =============================================================================

