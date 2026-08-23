# =============================================================================
# Process change impact analysis (difference-in-differences).
# Runs nightly after main data pipeline — ideally after CUSUM_Changepoint.py
# so analyser.pelt_analyse is fresh for the cross-reference in CELL 4.
#
# Purpose:
#   Plansak_Cohort_Analysis.py tracks cases against deadlines, but has no
#   control group and a case can take up to 3 years to resolve — it cannot
#   tell you whether a specific process change (a new template, a changed
#   procedure for one phase) actually helped. A naive before/after
#   comparison of Tidsbruk or Fristprosent around the change date is
#   confounded by:
#     - seasonality — already modeled elsewhere, in
#       Seasonal_YTD_ratio_extrapolation.py, so it's a real, known effect
#       here, not a hypothetical one
#     - org-wide secular drift — already detected elsewhere, by
#       CUSUM_Changepoint.py
#   This script instead computes a difference-in-differences (DiD) estimate:
#   the before->after change in the affected (treatment) population, net of
#   the before->after change in an unaffected (control) population over the
#   SAME calendar window. Seasonal/secular effects that hit both groups
#   equally cancel out, leaving (approximately) just the effect of the
#   change itself.
#
# Scope: Fristprosent and Behandlingstid only. Produksjonsdifferanse
#   (mottatt - ferdigstilt, a monthly aggregate) is deliberately excluded —
#   it has no per-case realization, so it cannot be split into per-row
#   treatment/control samples the way innenfor_frist (0/1) and tidsbruk
#   (days) can. An effect on Produksjonsdifferanse should still surface
#   indirectly via a shift in one of these two metrics.
#
# The before-window ("control" period, x days back from virkningsdato) and
#   the after-window ("test" period, y days forward) are independently
#   configurable per change (vindu_dager_foer / vindu_dager_etter) — they
#   are NOT forced to the same length. This matters whenever the thing
#   being measured hasn't existed for as long on one side as the other:
#   e.g. a process/indikator that only came into being a few months before
#   virkningsdato has a naturally short "before" history no matter how long
#   you're willing to wait for the "after" side to accumulate.
#
# Volume reality: some phases (e.g. Plansak) see only ~30 cases/year. This
#   drives two deliberate choices below:
#     - DEFAULT_VINDU_DAGER_FOER/_ETTER = 365 (a full year on each side by
#       default), not a shorter window — this both gets close to a usable
#       per-group sample size at that volume AND cancels seasonality on its
#       own (a full year covers one whole seasonal cycle), reducing
#       reliance on exact calendar alignment with the control group.
#       Shorten either side per change when the actual history available or
#       the case volume justifies it.
#     - MIN_OBS_PER_GROUP is a pragmatic floor (10, matching this repo's
#       existing MIN_SEGMENT_OBS/MIN_TEAM_VOLUME convention), not a
#       stricter one — a higher wall sounds more rigorous but at this
#       volume would just mean the tool reports nothing. Thinness below a
#       comfortable margin is instead surfaced via lav_styrke, an explicit
#       "trust this p-value less" flag, rather than suppressed entirely.
#   Consequence: with ~30 cases/year and a 365-day after-window, a mature
#   reading (tilstrekkelig_moden = TRUE) is roughly a year after rollout.
#   Earlier snapshots are provisional trend signal, not a conclusion — an
#   honest property of low case volume, not a bug to engineer around.
#
# Control group is OPTIONAL per configured change. When configured, this
#   script computes full DiD. When not, it falls back to a plain before/
#   after comparison on the treatment group alone and sets
#   har_kontrollgruppe = FALSE, so consumers know seasonal/secular
#   confounding was NOT controlled for in that row. A change is never
#   skipped just because it has no obvious control population.
#
# No scipy dependency — p-values come from the standard normal CDF via
#   stdlib math.erf, consistent with every script here except
#   CUSUM_Changepoint.py's optional ruptures/PELT path.
#
# Output table: prosessendring_effekt
# Power BI/DAX guidance:
#   see Process_Change_Impact_Analysis_POWERBI_DAX.md
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    TimestampType,
    DateType,
)
import pandas as pd
import numpy as np
import math
from datetime import datetime, date, timedelta

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY    = date.today()

DEFAULT_VINDU_DAGER_FOER  = 365   # full year of baseline/"control" history by default — see header comment
DEFAULT_VINDU_DAGER_ETTER = 365   # full year of test period by default — see header comment
MIN_OBS_PER_GROUP      = 10    # pragmatic floor, not a statistical ideal — see header comment
LAV_STYRKE_TERSKEL     = 20    # below this, volume clears the floor but is still thin
PELT_MATCH_WINDOW_DAYS = 45    # how close a PELT changepoint must be to virkningsdato to corroborate
Z_CRITICAL             = 1.96  # 95% confidence
ALPHA                  = 0.05


# =============================================================================
# PROCESS_CHANGES — hand-edit whenever a real process change is rolled out.
#
# Each entry:
#   endring_navn       : unique id/name for this change.
#   maaltall           : "Fristprosent" or "Behandlingstid" only (see header).
#   virkningsdato      : ISO date the change went live (the changepoint).
#   vindu_dager_foer   : the baseline/"control" period — how many days
#                         BEFORE virkningsdato to pull outcome values from.
#                         None -> DEFAULT_VINDU_DAGER_FOER. Shorten this when
#                         the thing being measured hasn't existed for a full
#                         year before the change.
#   vindu_dager_etter  : the test period — how many days AFTER virkningsdato
#                         to pull outcome values from. None ->
#                         DEFAULT_VINDU_DAGER_ETTER. Independent of
#                         vindu_dager_foer — the two do not need to match.
#   treatment_scope    : dict with optional "indikator"/"fasetittel"/"enhet"
#                         lists. An omitted (or None) dimension means no
#                         filter on that dimension.
#   control_scope      : OPTIONAL, same shape as treatment_scope. Omit this
#                         key entirely (or leave it None) when there is no
#                         unaffected population to compare against — the
#                         script falls back to a plain before/after
#                         comparison on the treatment group and sets
#                         har_kontrollgruppe = FALSE. This is a valid,
#                         expected mode, not an error.
#   minst_effekt_dager : (Behandlingstid only) minimum |days| effect to call
#                         Forbedring/Forverring. None = no practical floor
#                         for THIS entry — any p<0.05 effect counts. A
#                         per-entry opt-out, not a global default.
#   minst_effekt_pp    : (Fristprosent only) same idea, in percentage points
#                         (e.g. 0.05 = 5 percentage points).
# =============================================================================

PROCESS_CHANGES = [
    {
        # ---- TEMPLATE ENTRY — edit or delete before relying on this in prod ----
        "endring_navn":       "MAL_Ny_sjekkliste_byggesak",
        "maaltall":           "Behandlingstid",
        "virkningsdato":      "2026-03-01",
        "vindu_dager_foer":   None,
        "vindu_dager_etter":  None,
        "treatment_scope": {
            "indikator":  ["Byggesak - Tiltak"],
            "fasetittel": ["Saksbehandling"],
            "enhet":      None,
        },
        "control_scope": {
            # A phase of the SAME indikator untouched by the change — ideally
            # something exposed to the same seasonal/secular forces.
            "indikator":  ["Byggesak - Tiltak"],
            "fasetittel": ["Klagebehandling"],
            "enhet":      None,
        },
        "minst_effekt_dager": 5,
        "minst_effekt_pp":    None,
    },
]


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.prosessendring_effekt (
    endring_navn                  STRING      NOT NULL,
    maaltall                      STRING      NOT NULL,
    virkningsdato                 DATE        NOT NULL,
    vindu_dager_foer              INT         NOT NULL,
    vindu_dager_etter             INT         NOT NULL,
    snapshot_dato                 DATE        NOT NULL,

    n_behandling_foer             INT,
    gjennomsnitt_behandling_foer  DOUBLE,
    median_behandling_foer        DOUBLE,
    n_behandling_etter            INT,
    gjennomsnitt_behandling_etter DOUBLE,
    median_behandling_etter       DOUBLE,

    har_kontrollgruppe            BOOLEAN     NOT NULL,
    n_kontroll_foer               INT,
    gjennomsnitt_kontroll_foer    DOUBLE,
    median_kontroll_foer          DOUBLE,
    n_kontroll_etter              INT,
    gjennomsnitt_kontroll_etter   DOUBLE,
    median_kontroll_etter         DOUBLE,

    effekt_estimat                DOUBLE,
    standardfeil                  DOUBLE,
    konfidens_nedre                DOUBLE,
    konfidens_ovre                 DOUBLE,
    p_verdi                       DOUBLE,
    minst_praktisk_effekt         DOUBLE,
    tilstrekkelig_volum           BOOLEAN     NOT NULL,
    lav_styrke                    BOOLEAN,
    tilstrekkelig_moden           BOOLEAN     NOT NULL,
    effekt_retning                STRING,
    pelt_stotte                   BOOLEAN,

    kjoert_tidspunkt              TIMESTAMP   NOT NULL,
    kjoere_id                     STRING      NOT NULL
)
USING DELTA
COMMENT 'Difference-in-differences (eller foer/etter-fallback uten kontrollgruppe) vurdering av konfigurerte prosessendringer i PROCESS_CHANGES. "behandling" = gruppen som ble paavirket av endringen (treatment). har_kontrollgruppe=false betyr sesong-/sekulaer drift IKKE er kontrollert for i denne raden. tilstrekkelig_moden=false betyr etter-vinduet ikke er fullt forloept ennaa. lav_styrke=true betyr volumet saa vidt klarer minimumskravet og p-verdien boer tolkes som veiledende, ikke presis. pelt_stotte er kun et stoettesignal fra den uavhengige CUSUM_Changepoint-analysen, ikke input til effekt_retning. Append-modus, idempotent per snapshot_dato — samme endring akkumulerer historikk slik at konfidensintervallet kan foelges mens det smalner inn.'
""")

print("prosessendring_effekt-tabellen er klar")


# =============================================================================
# CELL 2 — Pure helper functions
# =============================================================================

def compute_group_stats(values):
    """
    n, mean, sample variance (ddof=1), median for a list of numeric outcome
    values. None/NaN entries are dropped first.
    n=0 -> everything None. n=1 -> mean/median available, var=None (can't
    estimate spread from a single point).
    """
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(clean)
    if n == 0:
        return 0, None, None, None
    arr = np.asarray(clean, dtype="float64")
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    var = float(np.var(arr, ddof=1)) if n >= 2 else None
    return n, mean, var, median


def normal_cdf(z):
    """Standard normal CDF via stdlib math.erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_value(z):
    """Two-sided p-value for a z statistic under N(0,1)."""
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def compute_diff_in_diff(
    treatment_before,
    treatment_after,
    control_before=None,
    control_after=None,
    min_obs_per_group=MIN_OBS_PER_GROUP,
    low_power_threshold=LAV_STYRKE_TERSKEL,
):
    """
    Difference-in-differences estimate, falling back to a plain before/after
    comparison when no control group is configured.

    treatment_before/treatment_after: raw per-row outcome values (tidsbruk
        days, or innenfor_frist 0/1) for the affected population, before and
        after virkningsdato.
    control_before/control_after: same shape for an unaffected population
        over the SAME calendar window.
        Pass None for BOTH to mean "no control group configured" -> plain
        before/after fallback (har_kontrollgruppe=False in the result).
        Pass [] (empty list, not None) to mean "a control group WAS
        configured but this window returned zero rows" -> still
        har_kontrollgruppe=True (DiD was attempted), just with
        tilstrekkelig_volum=False. This None-vs-[] distinction lets a
        consumer tell "we didn't try to control for confounding" apart
        from "we tried, but the control sample is too thin to trust yet".

    Never raises. Thin/absent groups (n=0 or n=1) are an expected condition
    here (e.g. a change rolled out last week) — fields that can't be
    computed come back as None. Contrast with gini_coefficient in
    Caseworker_Load_Concentration.py, which correctly DOES raise on a
    negative input because that signals a genuine data bug, not a
    small-sample condition; this function's failure modes are different.
    """
    n_tb, mean_tb, var_tb, med_tb = compute_group_stats(treatment_before)
    n_ta, mean_ta, var_ta, med_ta = compute_group_stats(treatment_after)

    har_kontrollgruppe = control_before is not None or control_after is not None
    n_cb, mean_cb, var_cb, med_cb = compute_group_stats(control_before or [])
    n_ca, mean_ca, var_ca, med_ca = compute_group_stats(control_after or [])

    result = {
        "n_treatment_before": n_tb, "mean_treatment_before": mean_tb, "median_treatment_before": med_tb,
        "n_treatment_after": n_ta, "mean_treatment_after": mean_ta, "median_treatment_after": med_ta,
        "har_kontrollgruppe": har_kontrollgruppe,
        "n_control_before": n_cb if har_kontrollgruppe else None,
        "mean_control_before": mean_cb if har_kontrollgruppe else None,
        "median_control_before": med_cb if har_kontrollgruppe else None,
        "n_control_after": n_ca if har_kontrollgruppe else None,
        "mean_control_after": mean_ca if har_kontrollgruppe else None,
        "median_control_after": med_ca if har_kontrollgruppe else None,
        "estimate": None, "se": None, "z": None, "p_value": None,
        "ci_lower": None, "ci_upper": None,
        "tilstrekkelig_volum": False, "lav_styrke": None,
    }

    if mean_tb is None or mean_ta is None:
        return result  # no usable treatment data in one or both windows

    treatment_delta = mean_ta - mean_tb

    if har_kontrollgruppe:
        if mean_cb is None or mean_ca is None:
            return result  # control configured but unusable — can't net out its trend
        estimate = treatment_delta - (mean_ca - mean_cb)
        group_ns, group_vars = [n_tb, n_ta, n_cb, n_ca], [var_tb, var_ta, var_cb, var_ca]
    else:
        estimate = treatment_delta
        group_ns, group_vars = [n_tb, n_ta], [var_tb, var_ta]

    result["estimate"] = round(estimate, 4)
    result["tilstrekkelig_volum"] = all(n >= min_obs_per_group for n in group_ns)
    if result["tilstrekkelig_volum"]:
        result["lav_styrke"] = min(group_ns) < low_power_threshold

    # SE: independent-samples sum-of-variances (SE(mean_i)^2 = var_i/n_i,
    # summed) — a simple, legitimate approximation, not a pooled/regression
    # estimator. Mildly conservative (overstates SE) if groups share
    # unmodelled structure; an acceptable trade-off for a screening signal
    # with no new dependency.
    if any(v is None for v in group_vars):
        return result  # some group has n<2 — can't estimate variance; point estimate above still stands
    se_sq = sum(v / n for v, n in zip(group_vars, group_ns))
    if se_sq <= 0:
        return result

    se = math.sqrt(se_sq)
    z = estimate / se
    result["se"] = round(se, 4)
    result["z"] = round(z, 4)
    result["p_value"] = round(two_sided_p_value(z), 6)
    result["ci_lower"] = round(estimate - Z_CRITICAL * se, 4)
    result["ci_upper"] = round(estimate + Z_CRITICAL * se, 4)
    return result


def classify_effect(p_value, estimate, maaltall, min_effect=None, alpha=ALPHA):
    """
    maaltall drives the improvement direction:
      Fristprosent   — higher is better (positive estimate = improvement)
      Behandlingstid — lower is better (negative estimate = improvement)

    min_effect: configured practical-significance floor in the metric's own
    units (percentage points for Fristprosent, days for Behandlingstid).
    None means this entry opted OUT of the practical-significance gate —
    any statistically significant effect is labelled, regardless of size.

    Returns "Forbedring", "Forverring", "Ingen praktisk effekt" (significant
    but below min_effect), "Ingen sikker effekt" (p >= alpha), or None when
    there isn't enough data to say anything — mirrors classify_risk's
    None-for-unusable-input convention in Inflight_SLA_Risk_Monitor.py.

    Raises ValueError on an unrecognized maaltall — a config typo, unlike
    the small-sample cases above, is a genuine bug worth failing on loudly.
    """
    if p_value is None or estimate is None:
        return None
    if maaltall not in ("Fristprosent", "Behandlingstid"):
        raise ValueError(f"classify_effect: ukjent maaltall '{maaltall}'")

    if p_value >= alpha:
        return "Ingen sikker effekt"

    if min_effect is not None and abs(estimate) < min_effect:
        return "Ingen praktisk effekt"

    improved = estimate > 0 if maaltall == "Fristprosent" else estimate < 0
    return "Forbedring" if improved else "Forverring"


# =============================================================================
# CELL 3 — SQL helpers (Spark-dependent)
# =============================================================================

def _scope_clause(scope, alias="pr"):
    """Build a WHERE clause from an optional indikator/fasetittel/enhet
    scope filter dict. Missing/empty dimension = no filter on it."""
    if not scope:
        return "1=1"
    dim_exprs = {
        "indikator":  f"{alias}.indikator",
        "fasetittel": f"COALESCE(NULLIF(TRIM({alias}.fasetittel), ''), 'Ukjent fase')",
        "enhet":      f"COALESCE(NULLIF(TRIM({alias}.enhet), ''), 'Ukjent')",
    }
    clauses = []
    for dim, expr in dim_exprs.items():
        values = scope.get(dim)
        if not values:
            continue
        escaped = ",".join("'" + v.replace("'", "''") + "'" for v in values)
        clauses.append(f"{expr} IN ({escaped})")
    return " AND ".join(clauses) if clauses else "1=1"


def _fetch_outcome(maaltall, scope, window_start, window_end):
    """
    Per-row outcome values for one maaltall/scope over the half-open date
    window [window_start, window_end) on sluttmilepaeldato — deliberately
    half-open (not inclusive on both ends) to avoid double-counting the
    boundary date, matching CUSUM_Changepoint.py's existing 'foer'/'etter'
    split convention.
    """
    if maaltall == "Behandlingstid":
        value_expr  = "CAST(pr.tidsbruk AS DOUBLE)"
        extra_where = "pr.sluttmilepaeldato IS NOT NULL AND pr.tidsbruk IS NOT NULL"
    else:  # Fristprosent
        value_expr  = "CAST(pr.innenfor_frist AS DOUBLE)"
        extra_where = "pr.sluttmilepaeldato IS NOT NULL AND pr.frist_dager IS NOT NULL"

    scope_sql = _scope_clause(scope)
    df = spark.sql(f"""
        SELECT {value_expr} AS verdi
        FROM saksbehandling.faser pr
        INNER JOIN felles.indikator indikator
            ON indikator.pk_indikator = pr.indikator
        WHERE pr.indikator NOT LIKE '%avtalt%'
          AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
          AND {extra_where}
          AND {scope_sql}
          AND pr.sluttmilepaeldato >= DATE('{window_start.isoformat()}')
          AND pr.sluttmilepaeldato <  DATE('{window_end.isoformat()}')
    """).toPandas()
    return df["verdi"].dropna().tolist()


def _pelt_stotte(indikator_list, maaltall, virkningsdato):
    """
    True: a PELT changepoint exists for a matching indikator/maaltall
    within PELT_MATCH_WINDOW_DAYS of virkningsdato. False: table exists,
    no match. None: table doesn't exist yet, or treatment_scope has no
    indikator filter (nothing specific enough to match against).
    Purely corroborating context for a human reader — must NOT feed back
    into effekt_retning.
    """
    if not spark.catalog.tableExists("analyser.pelt_analyse"):
        return None
    if not indikator_list:
        return None
    ind_sql = ",".join("'" + i.replace("'", "''") + "'" for i in indikator_list)
    lo = (virkningsdato - timedelta(days=PELT_MATCH_WINDOW_DAYS)).isoformat()
    hi = (virkningsdato + timedelta(days=PELT_MATCH_WINDOW_DAYS)).isoformat()
    n = spark.sql(f"""
        SELECT COUNT(*) AS n FROM analyser.pelt_analyse
        WHERE granularitet = 'Månedlig'
          AND maaltall = '{maaltall}'
          AND indikator IN ({ind_sql})
          AND analyse_dato BETWEEN DATE('{lo}') AND DATE('{hi}')
    """).collect()[0]["n"]
    return bool(n > 0)


# =============================================================================
# CELL 4 — Evaluate each configured process change
# =============================================================================

snapshot_dato = TODAY
result_rows = []

for change in PROCESS_CHANGES:
    navn         = change["endring_navn"]
    maaltall     = change["maaltall"]
    virkningsdato = date.fromisoformat(change["virkningsdato"])
    vindu_dager_foer  = change.get("vindu_dager_foer") or DEFAULT_VINDU_DAGER_FOER
    vindu_dager_etter = change.get("vindu_dager_etter") or DEFAULT_VINDU_DAGER_ETTER
    treatment_scope = change.get("treatment_scope")
    control_scope   = change.get("control_scope")

    before_start = virkningsdato - timedelta(days=vindu_dager_foer)
    after_end    = min(TODAY, virkningsdato + timedelta(days=vindu_dager_etter))
    tilstrekkelig_moden = TODAY >= virkningsdato + timedelta(days=vindu_dager_etter)

    treatment_before = _fetch_outcome(maaltall, treatment_scope, before_start, virkningsdato)
    treatment_after  = _fetch_outcome(maaltall, treatment_scope, virkningsdato, after_end)

    if control_scope:
        control_before = _fetch_outcome(maaltall, control_scope, before_start, virkningsdato)
        control_after  = _fetch_outcome(maaltall, control_scope, virkningsdato, after_end)
    else:
        control_before = None
        control_after  = None

    did = compute_diff_in_diff(treatment_before, treatment_after, control_before, control_after)

    min_effect = change.get("minst_effekt_pp") if maaltall == "Fristprosent" else change.get("minst_effekt_dager")
    effekt_retning = classify_effect(did["p_value"], did["estimate"], maaltall, min_effect)

    pelt_stotte = _pelt_stotte(
        (treatment_scope or {}).get("indikator"), maaltall, virkningsdato
    )

    result_rows.append({
        "endring_navn":                  navn,
        "maaltall":                      maaltall,
        "virkningsdato":                 virkningsdato,
        "vindu_dager_foer":              int(vindu_dager_foer),
        "vindu_dager_etter":             int(vindu_dager_etter),
        "snapshot_dato":                 snapshot_dato,
        "n_behandling_foer":             did["n_treatment_before"],
        "gjennomsnitt_behandling_foer":  did["mean_treatment_before"],
        "median_behandling_foer":        did["median_treatment_before"],
        "n_behandling_etter":            did["n_treatment_after"],
        "gjennomsnitt_behandling_etter": did["mean_treatment_after"],
        "median_behandling_etter":       did["median_treatment_after"],
        "har_kontrollgruppe":            did["har_kontrollgruppe"],
        "n_kontroll_foer":               did["n_control_before"],
        "gjennomsnitt_kontroll_foer":    did["mean_control_before"],
        "median_kontroll_foer":          did["median_control_before"],
        "n_kontroll_etter":              did["n_control_after"],
        "gjennomsnitt_kontroll_etter":   did["mean_control_after"],
        "median_kontroll_etter":         did["median_control_after"],
        "effekt_estimat":                did["estimate"],
        "standardfeil":                  did["se"],
        "konfidens_nedre":               did["ci_lower"],
        "konfidens_ovre":                did["ci_upper"],
        "p_verdi":                       did["p_value"],
        "minst_praktisk_effekt":         float(min_effect) if min_effect is not None else None,
        "tilstrekkelig_volum":           did["tilstrekkelig_volum"],
        "lav_styrke":                    did["lav_styrke"],
        "tilstrekkelig_moden":           bool(tilstrekkelig_moden),
        "effekt_retning":                effekt_retning,
        "pelt_stotte":                   pelt_stotte,
        "kjoert_tidspunkt":              datetime.now(),
        "kjoere_id":                     BATCH_ID,
    })

print(f"Prosessendringer vurdert: {len(result_rows):,}")
for row in result_rows:
    print(f"  {row['endring_navn']}: effekt={row['effekt_retning']}, "
          f"estimat={row['effekt_estimat']}, p={row['p_verdi']}, "
          f"har_kontrollgruppe={row['har_kontrollgruppe']}, "
          f"tilstrekkelig_volum={row['tilstrekkelig_volum']}, "
          f"tilstrekkelig_moden={row['tilstrekkelig_moden']}")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

SCHEMA = StructType([
    StructField("endring_navn",                  StringType(),    False),
    StructField("maaltall",                      StringType(),    False),
    StructField("virkningsdato",                 DateType(),      False),
    StructField("vindu_dager_foer",              IntegerType(),   False),
    StructField("vindu_dager_etter",             IntegerType(),   False),
    StructField("snapshot_dato",                 DateType(),      False),
    StructField("n_behandling_foer",             IntegerType(),   True),
    StructField("gjennomsnitt_behandling_foer",  DoubleType(),    True),
    StructField("median_behandling_foer",        DoubleType(),    True),
    StructField("n_behandling_etter",            IntegerType(),   True),
    StructField("gjennomsnitt_behandling_etter", DoubleType(),    True),
    StructField("median_behandling_etter",       DoubleType(),    True),
    StructField("har_kontrollgruppe",            BooleanType(),   False),
    StructField("n_kontroll_foer",               IntegerType(),   True),
    StructField("gjennomsnitt_kontroll_foer",    DoubleType(),    True),
    StructField("median_kontroll_foer",          DoubleType(),    True),
    StructField("n_kontroll_etter",               IntegerType(),   True),
    StructField("gjennomsnitt_kontroll_etter",   DoubleType(),    True),
    StructField("median_kontroll_etter",         DoubleType(),    True),
    StructField("effekt_estimat",                DoubleType(),    True),
    StructField("standardfeil",                  DoubleType(),    True),
    StructField("konfidens_nedre",                DoubleType(),    True),
    StructField("konfidens_ovre",                 DoubleType(),    True),
    StructField("p_verdi",                       DoubleType(),    True),
    StructField("minst_praktisk_effekt",         DoubleType(),    True),
    StructField("tilstrekkelig_volum",           BooleanType(),   False),
    StructField("lav_styrke",                    BooleanType(),   True),
    StructField("tilstrekkelig_moden",           BooleanType(),   False),
    StructField("effekt_retning",                StringType(),    True),
    StructField("pelt_stotte",                   BooleanType(),   True),
    StructField("kjoert_tidspunkt",              TimestampType(), False),
    StructField("kjoere_id",                     StringType(),    False),
])


def to_records(rows, schema):
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


if result_rows:
    spark.sql(f"DELETE FROM analyser.prosessendring_effekt WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')")
    result_spark = spark.createDataFrame(to_records(result_rows, SCHEMA), schema=SCHEMA)
    result_spark.write.mode("append").saveAsTable("analyser.prosessendring_effekt")
    print(f"prosessendring_effekt oppdatert for {snapshot_dato}: {len(result_rows):,} rader")
else:
    print("Ingen prosessendringer konfigurert i PROCESS_CHANGES.")


# =============================================================================
# CELL 6 — Verification
# =============================================================================

spark.sql(f"""
    SELECT endring_navn, maaltall, effekt_retning,
           ROUND(effekt_estimat, 3) AS effekt_estimat,
           ROUND(p_verdi, 4) AS p_verdi,
           har_kontrollgruppe, tilstrekkelig_volum, lav_styrke, tilstrekkelig_moden,
           pelt_stotte
    FROM analyser.prosessendring_effekt
    WHERE snapshot_dato = DATE('{snapshot_dato.isoformat()}')
    ORDER BY endring_navn
""").show(50, truncate=False)

print("\n=== KONFIDENSINTERVALL-UTVIKLING PER ENDRING ===")
spark.sql("""
    SELECT endring_navn, snapshot_dato,
           ROUND(effekt_estimat, 3) AS effekt_estimat,
           ROUND(konfidens_nedre, 3) AS konfidens_nedre,
           ROUND(konfidens_ovre, 3) AS konfidens_ovre
    FROM analyser.prosessendring_effekt
    ORDER BY endring_navn, snapshot_dato
""").show(100, truncate=False)


# =============================================================================
# Power BI/DAX guidance moved to separate documentation:
#   see Process_Change_Impact_Analysis_POWERBI_DAX.md
# =============================================================================
