# PBE-GovernanceAlgorithms

One system for monitoring case-processing indicators (`Fristprosent`, `Behandlingstid`,
`Produksjonsdifferanse`) from several angles: trend, drift, forecast, flow, backlog,
in-flight risk, workload. All views read the same fact table
(`saksbehandling.faser`, `pk_indikator` from `felles.indikator`), all output values in
Norwegian, DAX measure names in Norwegian.

## One rule, applied everywhere

A script exists only for what a live DAX measure structurally cannot do:
- a genuine algorithm (changepoint detection, statistical forecasting, rank-based
  concentration), or
- a value that depends on `TODAY()` and needs its history preserved day by day, since a
  live measure only ever knows "now," never what "now" looked like last week.

Everything else — raw values, YTD ratios, per-period counts, per-person shares — is a live
DAX measure straight against `saksbehandling.faser`, joined by `indikator` + `analyse_dato`
(add `enhet` at team level, `saksbehandler` at individual level). A script's output table
is the "addon" a live measure can't produce; it is never a copy of something the fact table
already gives you for free.

DAX stays simple: a plain measure or a standard time-intelligence pattern, nothing a
report author has to squint at. If the only way to express something in DAX is an
iterative window-scan, a disconnected-date trick, or a string built up in a table
constructor — the kind of measure that's slow to evaluate and hard for the next person to
read — that's a script, not a fancy measure. Being *possible* in DAX isn't being *simple*
in DAX.

`Kostra.py` is the one exception — it's SSB API ingestion, not a governance algorithm, and
isn't part of this rule.

## Scripts — the addon signal DAX can't produce

| File | Output table(s) | What it computes |
|---|---|---|
| `CUSUM_Changepoint.py` | `cusum_analyse`, `pelt_analyse`, `pelt_analyse_detaljer` | CUSUM drift score + PELT changepoints — recursive/segmentation math |
| `Seasonal_YTD_ratio_extrapolation.py` | `frist_prognose` | Year-end forecast + confidence interval — a statistical model |
| `Throughput_Pressure_Monitor.py` | `gjennomstoremming_press_enhet`, `gjennomstroemming_press_fase` | Team-level flow imbalance + tidsbruk deviation vs. baseline — composite score, flow streak |
| `Phase_Bottleneck_Detector.py` | `fase_flaskehals_enhet` | Same, one level down at the phase grain |
| `Backlog_Aging_Distribution.py` | `sak_alder_fordeling` | Daily snapshot of open-case age buckets — `TODAY()`-dependent trend |
| `Inflight_SLA_Risk_Monitor.py` | `sak_frist_risiko_trend` | Daily snapshot of open-case risk mix — `TODAY()`-dependent trend |
| `Caseworker_Load_Concentration.py` | `saksbehandler_konsentrasjon` | Gini coefficient of workload concentration — rank-based math, `TODAY()`-dependent trend |
| `Kostra.py` | `kostra_*` (one table per SSB series) | External data sync — not a governance algorithm |

All scripts share `START_YEAR = 2015` at the top — adjust to match the earliest reliable
data in your Lakehouse. All (except `Kostra.py`) run nightly after the main data pipeline.

Throughput_Pressure_Monitor and Phase_Bottleneck_Detector were briefly native DAX; reverted
— the flow-streak/queue-proxy measures needed an iterative window-scan and a third date
role DAX has no clean primitive for, exactly the "possible but not simple" case above.

## Live DAX pages — no script, computed directly against Faser

| Page (see `*_POWERBI_DAX.md`) | Replaces the removed script |
|---|---|
| `Trendretning` | Rolling-average trend direction (was `EWMA.py`) |

## Closed-case trend, drift, and forecast

### CUSUM_Changepoint.py

Detects small persistent shifts (CUSUM) and structural breakpoints (PELT) per indicator,
on both monthly and weekly series.

- **Måltall:** `Fristprosent`, `Behandlingstid`, `Produksjonsdifferanse`
- **External dependency:** `pip install ruptures` (PELT only — CUSUM runs without it)
- **Key constants:** `CUSUM_K` (allowance), `CUSUM_H` (threshold), `CUSUM_BASELINE_MONTHLY`/`CUSUM_BASELINE_WEEKLY` (anchored baseline window for mu/sigma), `CUSUM_MIN_POST_BASELINE_OBS`
- `signal` is boolean; `signalretning`/`endringsretning` are `Økning`/`Nedgang`
- mu/sigma come from a fixed, anchored baseline window (the series' first N observations), not the whole history — a slow persistent drift would otherwise get partially absorbed into "normal" and dampen detection
- `cusum_analyse` stores only `cusum_positiv`/`cusum_negativ`/`signal` — the raw value is a live DAX measure (see `CUSUM_Changepoint_POWERBI_DAX.md`)

**`pelt_analyse_detaljer`** breaks the most recent changepoint down by `enhet`/`fasetittel`, reusing PELT's before/after window instead of re-running detection.
- Only drills into changepoints within `RECENT_CHANGEPOINT_DAYS` (90) days old
- Saksbehandler is excluded — too thin per-segment volume, and individual-level flagging is out of scope
- `bidrag_til_endring` is each segment's volume-weighted share of the aggregate shift
- `tilstrekkelig_volum = FALSE` marks segments below `MIN_SEGMENT_OBS` (10) — don't trust these

### Trendretning (live DAX — see `Trendretning_POWERBI_DAX.md`)

Board/governance trend direction (`Stigende`/`Synkende`/`Stabil`) from the slope of a
rolling average — 6-month window for board reporting, 3-month for operational.

### Seasonal_YTD_ratio_extrapolation.py

Projects the rest of the year's `frist%` from current YTD using trimmed seasonal ratios
from historical years, only for what a live measure can't do: the projection and its
confidence band. (Actuals are the report's own `Faser innen frist %` measure.)

- **Minimum history:** 3 complete years per indicator
- **Confidence interval:** 90% (z=1.645), derived from ratio variance (delta method)
- Three seasonal models: cumulative-YTD ratios drive the year-end estimate
  (`prognose_aarsslutt`, with its own interval); per-month rate ratios turn that estimate
  back into the month rates the report plots; per-month volume ratios turn this year's
  observed caseload into a projected faser count per month
- `frist_prognose[verdi]` is a **period rate**, matching `Faser innen frist %` /
  `Fristprosent (måned)` — not a year-to-date value. `innenfor_prognose`/
  `produserte_prognose` carry that same rate as modelled faser counts, so a report can
  read it with the same `DIVIDE(SUM(...), SUM(...))` pattern it already uses on the fact
  table — an average of `verdi` across rows is different arithmetic and only agrees with
  that in the single-month, single-indicator case. An `Anker` row holds the last complete
  month's real counts (not modelled) so the projection forks off the actual line exactly,
  then one row per remaining month carries that month's modelled counts (one row per
  month, not per day — the report's axis groups by month, so a finer grain bought nothing)
- Idempotent — deletes and rewrites current-year rows on each run

## Flow and queue health

### Throughput_Pressure_Monitor.py

Team-level (`enhet`) flow imbalance (received vs. completed) and processing-time
deviation vs. a rolling baseline, combined into a `pressure_nivaa`
(`Lav`/`Moderat`/`Hoy`/`Kritisk`).

- **Output:** `gjennomstoremming_press_enhet` (team × indikator × month), `gjennomstroemming_press_fase` (fase-level support table)
- **Key constants:** `BASELINE_MONTHS`, `MIN_BASELINE_OBS`, `MIN_TEAM_VOLUME`, `POSITIVE_FLOW_STREAK`
- `netto_flyt_streak` (consecutive positive-flow months) is exactly the kind of
  order-dependent running count DAX has no clean primitive for — a script, not a fancy
  measure

### Phase_Bottleneck_Detector.py

Same pattern one level down — phase-level (`fasetittel`) queue pressure and tidsbruk
deviation, `alvorlighet` (`Lav`/`Moderat`/`Hoy`/`Kritisk`) with an `arsak_kode`/`arsak_tekst`
explaining the flag.

- **Output:** `fase_flaskehals_enhet` (enhet × fasetittel × indikator × month)
- **Key constants:** `BASELINE_MONTHS`, `MIN_BASELINE_OBS`, `MIN_SEGMENT_OBS`

## In-flight (currently-open) state

Both of these score cases that are **still open**, before a problem shows up in the
closed-case ratio — and both split the same way: today's state is live DAX, only the
day-by-day trend needs a script, since `TODAY()`-dependent values have no memory of what
they looked like yesterday.

### Inflight_SLA_Risk_Monitor.py

`classify_risk()` (thresholds `RISK_THRESHOLD_KRITISK`=0.90, `RISK_THRESHOLD_RISIKO`=0.75)
scores open cases against their own `frist_dager`. Today's per-case list is live DAX (see
`Inflight_SLA_Risk_Monitor_POWERBI_DAX.md`, Del 1); the script writes only the daily
`sak_frist_risiko_trend` risk-mix snapshot (Del 2) — one `INSERT INTO ... SELECT`, pure
Spark SQL, `MIN_TEAM_VOLUME` (10) gating `tilstrekkelig_volum`. `classify_risk()` stays in
the script as the tested spec `risikoklasse_case_sql()` generates its SQL `CASE` from.

- Open assumption to verify: whether `frist_dager` is reliably populated on open rows (only proven populated on closed rows, via CUSUM) — rows missing it are excluded, not defaulted.

### Backlog_Aging_Distribution.py

`bucket_age()` (buckets `0-30`/`31-60`/`61-90`/`91-180`/`180+`) ages open cases by
`startmilepaeldato`. Today's shape is live DAX (see
`Backlog_Aging_Distribution_POWERBI_DAX.md`, Del 1); the script writes only the daily
`sak_alder_fordeling` age-bucket snapshot (Del 2) — one `INSERT INTO ... SELECT`, pure
Spark SQL (`percentile_approx` + a `CASE` expression). `bucket_age()` stays in the script
as the tested spec `aldersgruppe_case_sql()` generates its SQL `CASE` from.

## Workload

### Caseworker_Load_Concentration.py

Is active caseload concentrating on a few caseworkers within a team, even while the
team's aggregate numbers look fine? Today's per-person counts/shares are live DAX
(`Faser[saksbehandler]` is already in the model — see
`Caseworker_Load_Concentration_POWERBI_DAX.md`, Del 1); the script writes only the daily
Gini-coefficient snapshot (Del 2), since rank-based Lorenz-curve math genuinely can't be a
DAX measure.

- `saksbehandler_konsentrasjon` stores enhet-level aggregates only, never a per-person breakdown — individual-level flagging is out of scope for this layer, same reasoning as `CUSUM_Changepoint.py`'s drilldown exclusion
- **Key constants:** `MIN_SAKSBEHANDLERE` (3) — Gini on 1-2 people is meaningless, gates `tilstrekkelig_volum`
- `SAKSBEHANDLER_COL` is unverified against the Lakehouse schema — verify before relying on this script

## External ingestion (not a governance algorithm)

### Kostra.py

Syncs selected SSB KOSTRA key-figure tables into the Lakehouse (`kostra_*`, one Delta
table per series), append-only new rows via a pandas dedup against the existing table.
Independent of the main pipeline and the rule above — this is a data source, not an
analysis.
