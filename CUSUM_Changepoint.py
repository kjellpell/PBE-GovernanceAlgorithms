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
# Requires: nothing to install. PELT runs on a built-in pure-NumPy
# implementation with an L2 (mean-shift) cost — see _pelt_l2() below. No
# external library, so no `%pip install` cell: on tenants with inline
# library installation disabled, that magic command fails the whole
# notebook run with MagicUsageError before any Python cell executes.
# Minimum history: 24 monthly / 52 weekly observations per indicator; CUSUM
# output starts once a month/week has >= 2 prior years of the same position.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, TimestampType, DateType,
)
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

spark = SparkSession.builder.getOrCreate()
BATCH_ID      = datetime.now().strftime("%Y%m%dT%H%M%S")
MIN_MONTHLY   = 24
MIN_WEEKLY    = 52
CUSUM_K       = 0.5   # allowance parameter — half sigma is standard
CUSUM_H       = 5.0   # decision threshold — 5 sigma cumulative
START_YEAR    = 2015  # exclude data before this year — adjust if older data is reliable

# Only complete periods are scored: the current calendar month (and, for the
# weekly series, the current ISO week) is still filling up, so its counts and
# rates would look like a drop every run. DAX "siste analysedato" measures pick
# up the last complete period automatically as a result.
MONTH_CUTOFF_SQL = "CAST(DATE_TRUNC('MONTH', CURRENT_DATE()) AS DATE)"
WEEK_CUTOFF_SQL  = "CAST(DATE_TRUNC('WEEK', CURRENT_DATE()) AS DATE)"   # Monday of this ISO week

# Seasonal baseline — each observation is standardised against the SAME
# calendar month (monthly) or ISO week (weekly) in the preceding
# CUSUM_BASELINE_YEARS years, never including its own year. That removes
# seasonality (July is compared with earlier Julys, not with March), keeps the
# reference recent (not "different from 2015"), and still can't absorb a
# drift that builds up inside the current year. See standardise_seasonal().
CUSUM_BASELINE_YEARS        = 3
CUSUM_MIN_BASELINE_YEARS    = 2   # fewer prior years for a month/week -> no score for it
CUSUM_MIN_POST_BASELINE_OBS = 4   # need at least this many standardised points to run CUSUM

# tilstand (the persistent state a board card reads) returns to Stabil only
# after this many consecutive quiet periods — see run_cusum().
CUSUM_RECOVERY_PERIODS      = 3

DRILLDOWN_DIMENSIONS     = ["enhet", "fasetittel"]  # confirmed against the Lakehouse schema
MIN_SEGMENT_OBS          = 10   # segments below this are marked utilstrekkelig_volum, not tested
# Only drill into changepoints still recent enough to be actionable. Per
# granularity: PELT needs min_size=6 periods after a breakpoint, so the most
# recent possible monthly changepoint is already ~6 months old — a 90-day
# window could never admit one.
RECENT_CHANGEPOINT_DAYS  = {"Månedlig": 240, "Ukentlig": 90}
PELT_MIN_SIZE            = 6


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
    tilstand         STRING,
    kjoert_tidspunkt TIMESTAMP   NOT NULL,
    kjoere_id        STRING      NOT NULL
)
USING DELTA
COMMENT 'CUSUM-driftdeteksjon per indikator og måltall. Hver periode sammenlignes med samme måned/ISO-uke de tre foregående årene (sesongjustert, nylig baseline). signal=true og signalretning (Stigende/Synkende) markerer perioden akkumulatoren krysset terskelen (hendelse). tilstand (Stigende/Synkende/Stabil) er siste signalerte retning, holdt til motsatt signal eller tre rolige perioder på rad — det er denne et bord-nivå mål skal lese. Rå verdi (Fristprosent/Behandlingstid/Produksjonsdifferanse) er IKKE lagret her — den er en live DAX-mål mot saksbehandling.faser, join på indikator+maaltall+granularitet+analyse_dato (se CUSUM_Changepoint_POWERBI_DAX.md).'
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
    bidrag_rate                 DOUBLE,
    bidrag_miks                 DOUBLE,
    antall_observasjoner_foer   INT,
    antall_observasjoner_etter  INT,
    tilstrekkelig_volum         BOOLEAN     NOT NULL,
    kjoert_tidspunkt            TIMESTAMP   NOT NULL,
    kjoere_id                   STRING      NOT NULL
)
USING DELTA
COMMENT 'Nedbryting av siste pelt_analyse-endringspunkt per indikator/maaltall, fordelt på enhet og fasetittel. endringspunkt_id er samme verdi som i pelt_analyse for raden dette bryter ned — bruk den som relasjonsnøkkel i den semantiske modellen (én-til-mange fra pelt_analyse). Før/etter-vinduet er det samme som PELT brukte (forrige til neste endringspunkt), delt ved første dag i endringsperioden. bidrag_til_endring er segmentets andel av det totale skiftet = bidrag_rate (segmentet ble bedre/verre: vekt etter × endring) + bidrag_miks (segmentet ble større/mindre: vektendring × verdi før). For Produksjonsdifferanse er verdiene per periode og bidrag_miks er 0. tilstrekkelig_volum=false betyr for få observasjoner til å stole på tallene.'
""")

print("Output tables ready")

# =============================================================================
# CELL 1B — ISO week key
# =============================================================================

def iso_week_sql(date_col):
    """Spark SQL for a YYYYWW weekly key that uses the ISO year, not the
    calendar year. WEEKOFYEAR is the ISO week, so it must be paired with the
    ISO year — the year of that week's Thursday (Spark WEEKDAY: 0 = Monday).
    YEAR(d) * 100 + WEEKOFYEAR(d) put 2024-12-30 (ISO 2025-W01) in 202401 and
    2021-01-01 (ISO 2020-W53) in 202153, which does not exist and crashed
    _periode_to_date."""
    return (
        f"(YEAR(DATE_ADD({date_col}, 3 - WEEKDAY({date_col}))) * 100"
        f" + WEEKOFYEAR({date_col}))"
    )


def iso_week_key(d):
    """Python mirror of iso_week_sql, for tests: ISO year * 100 + ISO week."""
    thursday = d + timedelta(days=3 - d.weekday())
    return thursday.year * 100 + d.isocalendar()[1]



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
            AND pr.sluttmilepaeldato < {MONTH_CUTOFF_SQL}
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
            AND pr.sluttmilepaeldato < {MONTH_CUTOFF_SQL}
        GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
        ORDER BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

# Monthly production balance per indicator: faser started in the month (by
# startmilepaeldato) minus faser ended in the month (by sluttmilepaeldato).
# Two separate counts, merged in pandas — an earlier single query grouped
# both by COALESCE(slutt, start), so every closed fase netted to zero in its
# end month and the "balance" was really just still-open faser by start month.
monthly_started = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(pr.startmilepaeldato) * 100 + MONTH(pr.startmilepaeldato)) AS period,
                COUNT(1) AS antall_startet
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.startmilepaeldato IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.startmilepaeldato) >= {START_YEAR}
            AND pr.startmilepaeldato < {MONTH_CUTOFF_SQL}
        GROUP BY pr.indikator, YEAR(pr.startmilepaeldato), MONTH(pr.startmilepaeldato)
""").toPandas()

monthly_ended = spark.sql(f"""
        SELECT
                pr.indikator,
                (YEAR(pr.sluttmilepaeldato) * 100 + MONTH(pr.sluttmilepaeldato)) AS period,
                COUNT(1) AS antall_avsluttet
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
            AND pr.sluttmilepaeldato < {MONTH_CUTOFF_SQL}
        GROUP BY pr.indikator, YEAR(pr.sluttmilepaeldato), MONTH(pr.sluttmilepaeldato)
""").toPandas()

# Weekly frist% per indicator
weekly_frist = spark.sql(f"""
    SELECT
        pr.indikator,
        {iso_week_sql('pr.sluttmilepaeldato')} AS period,
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
            AND pr.sluttmilepaeldato < {WEEK_CUTOFF_SQL}
        GROUP BY pr.indikator, {iso_week_sql('pr.sluttmilepaeldato')}
        ORDER BY pr.indikator, period
""").toPandas()

# Weekly production balance — same split as monthly_prod
weekly_started = spark.sql(f"""
        SELECT
                pr.indikator,
                {iso_week_sql('pr.startmilepaeldato')} AS period,
                COUNT(1) AS antall_startet
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.startmilepaeldato IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.startmilepaeldato) >= {START_YEAR}
            AND pr.startmilepaeldato < {WEEK_CUTOFF_SQL}
        GROUP BY pr.indikator, {iso_week_sql('pr.startmilepaeldato')}
""").toPandas()

weekly_ended = spark.sql(f"""
        SELECT
                pr.indikator,
                {iso_week_sql('pr.sluttmilepaeldato')} AS period,
                COUNT(1) AS antall_avsluttet
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.sluttmilepaeldato IS NOT NULL
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.sluttmilepaeldato) >= {START_YEAR}
            AND pr.sluttmilepaeldato < {WEEK_CUTOFF_SQL}
        GROUP BY pr.indikator, {iso_week_sql('pr.sluttmilepaeldato')}
""").toPandas()


def production_balance(started, ended):
    """Produksjonsdifferanse per indikator x period: faser started in the
    period minus faser ended in the period. started/ended are counted by their
    own date (start vs. slutt), outer-merged so a period with only starts or
    only ends still gets a row, with the missing side as 0."""
    merged = started.merge(ended, on=["indikator", "period"], how="outer")
    for col in ("antall_startet", "antall_avsluttet"):
        if col not in merged:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    merged["verdi"] = merged["antall_startet"] - merged["antall_avsluttet"]
    return merged[["indikator", "period", "verdi"]].sort_values(["indikator", "period"]).reset_index(drop=True)


monthly_prod = production_balance(monthly_started, monthly_ended)
weekly_prod  = production_balance(weekly_started, weekly_ended)

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
        # ISO year + ISO week (see iso_week_sql) — return the Sunday (last
        # day) of that week
        try:
            return pd.Timestamp.fromisocalendar(year, unit, 7).date()
        except ValueError as exc:
            raise ValueError(
                f"Ugyldig ISO-uke {period_int}: {year} har ikke uke {unit}. "
                "Ukenøkkelen må bygges med ISO-år (iso_week_sql), ikke kalenderår."
            ) from exc


def _periode_to_start_date(period_int, granularitet):
    """First day of a YYYYMM month or the Monday of an ISO YYYYWW week."""
    year = int(period_int) // 100
    unit = int(period_int) % 100
    if granularitet == "Månedlig":
        return date(year, unit, 1)
    return pd.Timestamp.fromisocalendar(year, unit, 1).date()


def make_endringspunkt_id(indikator, maaltall, granularitet, analyse_dato):
    """Shared join key between pelt_analyse and pelt_analyse_detaljer —
    Power BI relationships need a single column, not the underlying
    composite (indikator, maaltall, granularitet, analyse_dato)."""
    return f"{indikator}|{maaltall}|{granularitet}|{analyse_dato.isoformat()}"



def to_float_series(series):
    """Spark DECIMAL columns arrive as decimal.Decimal, which cannot be mixed
    with floats in the arithmetic below."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def standardise_seasonal(series, granularitet, years=CUSUM_BASELINE_YEARS,
                         min_years=CUSUM_MIN_BASELINE_YEARS):
    """
    Standardise a YYYYMM (Månedlig) or ISO YYYYWW (Ukentlig) series against
    the same month/week in the preceding `years` years.

    For an observation in year Y at position p (month or ISO week):
      mu    = mean of position p in years Y-years .. Y-1 (needs >= min_years
              of them; otherwise the observation gets NaN = no score)
      sigma = pooled within-position standard deviation over those same
              years: every baseline value minus its own position's mean,
              squared, summed, divided by (values - positions). Pooling over
              all months/weeks gives a stable sigma; a per-position sigma
              from 2-3 values would not be.
    The observation's own year is never in its baseline. Returns a float
    Series on the same index, NaN where no baseline exists.
    """
    s = to_float_series(series).dropna()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    if s.empty:
        return out

    periods = np.asarray(s.index, dtype="int64")
    df = pd.DataFrame(
        {"aar": periods // 100, "pos": periods % 100, "verdi": s.to_numpy()},
        index=s.index,
    )

    for aar in np.unique(df["aar"]):
        base = df[(df["aar"] >= aar - years) & (df["aar"] < aar)]
        if base.empty:
            continue
        stats = base.groupby("pos")["verdi"].agg(["mean", "count"])
        stats = stats[stats["count"] >= min_years]
        if stats.empty:
            continue

        used  = base[base["pos"].isin(stats.index)]
        resid = used["verdi"] - used["pos"].map(stats["mean"])
        dof   = len(resid) - len(stats)
        if dof <= 0:
            continue
        sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
        if not np.isfinite(sigma) or sigma == 0:
            continue

        target = df[(df["aar"] == aar) & df["pos"].isin(stats.index)]
        out.loc[target.index] = (target["verdi"] - target["pos"].map(stats["mean"])) / sigma

    return out


def run_cusum(series, granularitet="Månedlig", k=CUSUM_K, h=CUSUM_H,
              years=CUSUM_BASELINE_YEARS, recovery_periods=CUSUM_RECOVERY_PERIODS):
    """
    Two-sided CUSUM on the seasonally standardised series (standardise_seasonal).
    k = allowance (typically 0.5 * expected shift in sigma units)
    h = decision threshold (typically 4-5)

    Event vs. state:
    - signal / signal_direction is the EVENT: True / Stigende / Synkende only
      in the period an accumulator crosses h. That accumulator is then reset
      to 0 (standard SPC practice — otherwise it saturates after the first
      sustained shift and can't register a recovery or a second shift).
    - tilstand is the STATE a board card should read: the last signalled
      direction, held until an opposite signal fires or the matching side has
      been quiet for `recovery_periods` consecutive periods. Quiet = the
      accumulator is below h/2 AND that period's value did not push it up
      (z - k <= 0 for Stigende, -z - k <= 0 for Synkende). Without the second
      condition, the reset itself (accumulator back at 0, refilling slowly
      under a small shift) looked like a recovery, and a sustained 1-sigma
      decline read Synkende only about half the time.

    Returns DataFrame with cusum_pos, cusum_neg, signal, signal_direction,
    tilstand — indexed by the periods that had a baseline — or None when too
    few periods could be standardised. Direction values are always
    Stigende/Synkende/Stabil, so DAX can read them with no translation.
    """
    z = standardise_seasonal(series, granularitet, years=years).dropna()
    if len(z) < CUSUM_MIN_POST_BASELINE_OBS:
        return None

    values    = z.to_numpy()
    n         = len(values)
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    signal    = np.zeros(n, dtype=bool)
    retning   = np.full(n, "Stabil", dtype=object)
    tilstand  = np.full(n, "Stabil", dtype=object)

    prev_pos = prev_neg = 0.0
    state    = "Stabil"
    quiet    = 0
    for i in range(n):
        pos = max(0.0, prev_pos + values[i] - k)
        neg = max(0.0, prev_neg - values[i] - k)

        if pos > h:
            signal[i], retning[i], pos = True, "Stigende", 0.0
        elif neg > h:
            signal[i], retning[i], neg = True, "Synkende", 0.0

        if signal[i]:
            state, quiet = retning[i], 0
        elif state != "Stabil":
            acc  = pos if state == "Stigende" else neg
            push = (values[i] - k) if state == "Stigende" else (-values[i] - k)
            quiet = quiet + 1 if (acc < h / 2 and push <= 0) else 0
            if quiet >= recovery_periods:
                state, quiet = "Stabil", 0

        cusum_pos[i], cusum_neg[i], tilstand[i] = pos, neg, state
        prev_pos, prev_neg = pos, neg

    return pd.DataFrame({
        "cusum_pos":        cusum_pos,
        "cusum_neg":        cusum_neg,
        "signal":           signal,
        "signal_direction": retning,
        "tilstand":         tilstand,
    }, index=z.index)


# =============================================================================
# CELL 4 — Changepoint detection (PELT)
# =============================================================================

def _pelt_l2(values, min_size, penalty):
    """
    Pure-NumPy PELT changepoint detection with an L2 (mean-shift) cost —
    exact-search PELT (Killick, Fearnhead & Eckley, 2012), no external
    library. Detects a shift in the series' average; a change in
    volatility with the average unchanged is out of scope, which matches
    what this table is for (see the "Requires" note at the top of this
    file). Returns changepoint indices, excluding the trailing
    len(values) endpoint.
    """
    n = len(values)
    if n < 2 * min_size:
        return []

    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    cumsum_sq = np.concatenate(([0.0], np.cumsum(values ** 2)))

    def segment_cost(s, t):
        length = t - s
        total = cumsum[t] - cumsum[s]
        total_sq = cumsum_sq[t] - cumsum_sq[s]
        return total_sq - total * total / length

    best_cost = {0: -penalty}
    last_change = {}
    candidates = [0]

    for t in range(min_size, n + 1):
        best_f, best_s = None, None
        for s in candidates:
            if t - s < min_size:
                continue
            f = best_cost[s] + segment_cost(s, t) + penalty
            if best_f is None or f < best_f:
                best_f, best_s = f, s
        if best_f is None:
            continue
        best_cost[t] = best_f
        last_change[t] = best_s
        candidates = [s for s in candidates if best_cost[s] + segment_cost(s, t) <= best_f]
        candidates.append(t)

    if n not in last_change:
        return []

    breakpoints = []
    t = n
    while t in last_change:
        s = last_change[t]
        if s > 0:
            breakpoints.append(s)
        t = s
    return sorted(breakpoints)


def _pelt_penalty(values):
    """BIC-style penalty for a mean-shift (L2) cost: 2 * sigma^2 * log(n),
    the standard calibration (Yao, 1988). sigma is the NOISE level, estimated
    robustly from first differences: 1.4826 * MAD(diff) / sqrt(2). The plain
    np.std(values) of the whole series also counts the shifts themselves as
    noise, which inflates the penalty and hides exactly the changes PELT is
    looking for. Returns None for a constant series (nothing to detect)."""
    values = np.asarray(values, dtype="float64")
    diffs  = np.diff(values)
    if len(diffs) == 0:
        return None
    sigma = 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2)
    if not np.isfinite(sigma) or sigma == 0:
        # MAD is 0 when most differences are identical (e.g. small integer
        # counts); fall back to the plain std of the differences.
        sigma = np.std(diffs) / np.sqrt(2)
    if not np.isfinite(sigma) or sigma == 0:
        return None
    return 2 * np.log(len(values)) * sigma ** 2


def run_changepoint(series, granularitet):
    """
    PELT changepoint detection (_pelt_l2). Returns list of changepoint
    indices, or empty list if none detected or insufficient data.
    """
    values = to_float_series(series).dropna().values
    min_obs = MIN_MONTHLY if granularitet == "Månedlig" else MIN_WEEKLY

    if len(values) < min_obs:
        return []

    penalty = _pelt_penalty(values)
    if penalty is None:
        return []
    return _pelt_l2(values, min_size=PELT_MIN_SIZE, penalty=penalty)


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
}
# Produksjonsdifferanse has no single date column — starts are placed by
# startmilepaeldato and ends by sluttmilepaeldato — so it gets its own query
# branch in run_drilldown (_production_drilldown_sql), not a METRIC_SQL entry.


def _production_drilldown_sql(indikator_escaped, dimensjon, window):
    """Starts (by start date) and ends (by end date) per segment on each side
    of the breakpoint, inside PELT's before/after window.
    verdi = started - ended (a total for the side), antall = started + ended."""
    def side(date_col, startet, avsluttet):
        return f"""
            SELECT
                pr.{dimensjon} AS segment,
                CASE WHEN {date_col} < DATE'{window["split"]}' THEN 'foer' ELSE 'etter' END AS periode,
                {startet} AS startet,
                {avsluttet} AS avsluttet
            FROM saksbehandling.faser pr
            INNER JOIN felles.indikator indikator ON indikator.pk_indikator = pr.indikator
            WHERE pr.indikator = '{indikator_escaped}'
                AND pr.indikator NOT LIKE '%avtalt%'
                AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
                AND {date_col} >= DATE'{window["start"]}'
                AND {date_col} <= DATE'{window["end"]}'
                AND pr.{dimensjon} IS NOT NULL"""

    return f"""
        SELECT
            segment,
            periode,
            SUM(startet) - SUM(avsluttet) AS verdi,
            SUM(startet) + SUM(avsluttet) AS antall
        FROM (
            {side("pr.startmilepaeldato", 1, 0)}
            UNION ALL
            {side("pr.sluttmilepaeldato", 0, 1)}
        ) hendelser
        GROUP BY segment, periode
    """


def _metric_drilldown(indikator_escaped, maaltall, dimensjon, window):
    """Fristprosent/Behandlingstid per segment on each side of the breakpoint,
    inside PELT's before/after window — both are placed by a single date
    column (METRIC_SQL)."""
    cfg = METRIC_SQL[maaltall]
    return spark.sql(f"""
        SELECT
            pr.{dimensjon} AS segment,
            CASE WHEN {cfg['date_col']} < DATE'{window["split"]}' THEN 'foer' ELSE 'etter' END AS periode,
            {cfg['value_expr']} AS verdi,
            COUNT(*) AS antall
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator ON indikator.pk_indikator = pr.indikator
        WHERE pr.indikator = '{indikator_escaped}'
            AND pr.indikator NOT LIKE '%avtalt%'
            AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND {cfg['date_col']} >= DATE'{window["start"]}'
            AND {cfg['date_col']} <= DATE'{window["end"]}'
            AND {cfg['extra_where']}
            AND pr.{dimensjon} IS NOT NULL
        GROUP BY pr.{dimensjon},
            CASE WHEN {cfg['date_col']} < DATE'{window["split"]}' THEN 'foer' ELSE 'etter' END
    """).toPandas()


def decompose_segments(segments, additive, perioder_foer=1, perioder_etter=1):
    """
    Per-segment before/after values and each segment's contribution to the
    aggregate shift, from rows (segment, periode in {foer, etter}, verdi,
    antall).

    Mean/ratio måltall (Fristprosent, Behandlingstid): the aggregate is the
    volume-weighted mean of the segments, w = segment antall / total antall.
    Its shift splits exactly into, per segment,
        bidrag_rate = w_etter * (v_etter - v_foer)   (the segment got better/worse)
        bidrag_miks = (w_etter - w_foer) * v_foer    (the segment got bigger/smaller)
    and bidrag_til_endring = bidrag_rate + bidrag_miks. Summed over segments
    that equals the aggregate's after-minus-before.

    Additive måltall (Produksjonsdifferanse): verdi is a total for the side, so
    it is divided by that side's number of periods to give a per-period value
    comparable with pelt_analyse; the aggregate is the plain sum of segments,
    so there are no weights: bidrag_rate is the segment's own change and
    bidrag_miks is 0. A segment missing on one side had 0 there.

    Returns a DataFrame indexed by segment.
    """
    cols = ["verdi_foer", "verdi_etter", "antall_foer", "antall_etter",
            "endring", "bidrag_rate", "bidrag_miks", "bidrag_til_endring",
            "tilstrekkelig_volum"]
    if segments.empty:
        return pd.DataFrame(columns=cols)

    seg = segments.copy()
    seg["verdi"]  = pd.to_numeric(seg["verdi"], errors="coerce").astype("float64")
    seg["antall"] = pd.to_numeric(seg["antall"], errors="coerce").fillna(0).astype("int64")

    pivot = seg.pivot(index="segment", columns="periode", values=["verdi", "antall"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reindex(columns=["verdi_foer", "verdi_etter", "antall_foer", "antall_etter"])
    pivot[["antall_foer", "antall_etter"]] = pivot[["antall_foer", "antall_etter"]].fillna(0)

    if additive:
        pivot["verdi_foer"]  = pivot["verdi_foer"].fillna(0) / max(perioder_foer, 1)
        pivot["verdi_etter"] = pivot["verdi_etter"].fillna(0) / max(perioder_etter, 1)
        pivot["bidrag_rate"] = pivot["verdi_etter"] - pivot["verdi_foer"]
        pivot["bidrag_miks"] = 0.0
    else:
        pivot = pivot.dropna(subset=["verdi_foer", "verdi_etter"])
        total_foer  = pivot["antall_foer"].sum()
        total_etter = pivot["antall_etter"].sum()
        if total_foer and total_etter:
            w_foer  = pivot["antall_foer"] / total_foer
            w_etter = pivot["antall_etter"] / total_etter
            pivot["bidrag_rate"] = w_etter * (pivot["verdi_etter"] - pivot["verdi_foer"])
            pivot["bidrag_miks"] = (w_etter - w_foer) * pivot["verdi_foer"]
        else:
            pivot["bidrag_rate"] = np.nan
            pivot["bidrag_miks"] = np.nan

    pivot["endring"] = pivot["verdi_etter"] - pivot["verdi_foer"]
    pivot["bidrag_til_endring"] = pivot["bidrag_rate"] + pivot["bidrag_miks"]
    pivot["tilstrekkelig_volum"] = (
        (pivot["antall_foer"] >= MIN_SEGMENT_OBS) & (pivot["antall_etter"] >= MIN_SEGMENT_OBS)
    )
    return pivot[cols]


def run_drilldown(indikator, maaltall, granularitet, cp, dimensjon):
    """
    Split the flagged indikator/maaltall shift by one dimension (enhet or
    fasetittel), on exactly PELT's before/after window: from the start of the
    before-segment (cp["segment_start"]) to the end of the after-segment
    (cp["segment_end"]), split at the first day of the breakpoint period
    (cp["endringspunkt_start"]). Returns a list of row dicts, one per segment.
    """
    indikator_escaped = indikator.replace("'", "''")
    window = {
        "start": cp["segment_start"].isoformat(),
        "split": cp["endringspunkt_start"].isoformat(),
        "end":   cp["segment_end"].isoformat(),
    }
    additive = maaltall == "Produksjonsdifferanse"

    if additive:
        segments = spark.sql(
            _production_drilldown_sql(indikator_escaped, dimensjon, window)
        ).toPandas()
    else:
        segments = _metric_drilldown(indikator_escaped, maaltall, dimensjon, window)

    parts = decompose_segments(
        segments, additive,
        perioder_foer=int(cp["antall_observasjoner_foer"]),
        perioder_etter=int(cp["antall_observasjoner_etter"]),
    )

    def _r(v):
        return round(float(v), 4) if pd.notna(v) else None

    rows = []
    for segment, r in parts.iterrows():
        rows.append({
            "indikator":                  indikator,
            "maaltall":                   maaltall,
            "granularitet":               granularitet,
            "dimensjon":                  dimensjon,
            "dimensjonsverdi":            str(segment),
            "analyse_dato":               cp["analyse_dato"],
            "endringspunkt_id":           make_endringspunkt_id(
                indikator, maaltall, granularitet, cp["analyse_dato"]
            ),
            "gjennomsnitt_foer":          _r(r["verdi_foer"]),
            "gjennomsnitt_etter":         _r(r["verdi_etter"]),
            "endringsstoerrelse":         _r(r["endring"]),
            "bidrag_til_endring":         _r(r["bidrag_til_endring"]),
            "bidrag_rate":                _r(r["bidrag_rate"]),
            "bidrag_miks":                _r(r["bidrag_miks"]),
            "antall_observasjoner_foer":  int(r["antall_foer"]),
            "antall_observasjoner_etter": int(r["antall_etter"]),
            "tilstrekkelig_volum":        bool(r["tilstrekkelig_volum"]),
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
            # PELT's before/after window, reused by the drill-down: first day of
            # the before-segment, first day of the breakpoint period (the split),
            # last day of the after-segment.
            "segment_start":               _periode_to_start_date(periods[prev], granularitet),
            "endringspunkt_start":         _periode_to_start_date(periods[bp], granularitet),
            "segment_end":                 _periode_to_date(periods[next_bp - 1], granularitet),
            "gjennomsnitt_foer":           round(mean_before, 4),
            "gjennomsnitt_etter":          round(mean_after, 4),
            "endringsstoerrelse":          round(shift, 4),
            "endringsretning":             "Stigende" if shift > 0 else "Synkende",
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
        cusum = run_cusum(ind_data, granularitet)
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
                    "tilstand":        row["tilstand"],
                    "kjoert_tidspunkt": datetime.now(),
                    "kjoere_id":       BATCH_ID,
                })

        # ── Changepoint (monthly and weekly — see PELT header note) ─
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
    today = pd.Timestamp.now().date()
    is_recent = [
        r["analyse_dato"] >= today - pd.Timedelta(days=RECENT_CHANGEPOINT_DAYS[r["granularitet"]])
        for _, r in latest_cp.iterrows()
    ]
    latest_cp = latest_cp[is_recent]

    for _, cp in latest_cp.iterrows():
        for dimensjon in DRILLDOWN_DIMENSIONS:
            drilldown_rows += run_drilldown(
                cp["indikator"], cp["maaltall"], cp["granularitet"], cp, dimensjon,
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
    StructField("tilstand",         StringType(),    True),
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
    StructField("bidrag_rate",                DoubleType(),    True),
    StructField("bidrag_miks",                DoubleType(),    True),
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

# Summary — series whose current state (latest row) is not Stabil
if cusum_rows:
    spark.sql(f"""
                SELECT indikator, maaltall, granularitet, analyse_dato, tilstand
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY indikator, maaltall, granularitet
                               ORDER BY analyse_dato DESC
                           ) AS rn
                    FROM analyser.cusum_analyse
                    WHERE kjoere_id = '{BATCH_ID}'
                ) siste
                WHERE rn = 1
                    AND tilstand <> 'Stabil'
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

