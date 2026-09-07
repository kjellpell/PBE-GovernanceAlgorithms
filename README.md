# PBE-GovernanceAlgorithms

One system for monitoring case-processing indicators (`Fristprosent`, `Behandlingstid`,
`Produksjonsdifferanse`) from several angles: trend, drift, forecast, flow, workload. All views read the same fact table
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

## Two reports, not one

Not every output here is intuitive to a leader without someone explaining the method
first. Split into two Power BI reports along that line, rather than one report mixing
self-explanatory labels with numbers that need a briefing:

**Leder-rapport** — reads cold, no explanation needed:
- `Trendretning` — `Stigende`/`Synkende`/`Stabil` is just a word
- `Phase_Bottleneck_Detector` — `alvorlighet` plus `arsak_tekst`, a full plain-language sentence explaining the flag, not just a code

**Analytiker-rapport** — the label is fine, but the "why" is a statistical construct one
step removed from the raw quantity, and needs someone who can defend the method:
- `CUSUM_Changepoint` — why *this* month triggered, not last month, is an anchored-baseline judgment call
- `Seasonal_YTD_ratio_extrapolation` — the forecast range reads fine, but trusting the number means trusting the seasonal-ratio method behind it
- `Throughput_Pressure_Monitor` — `pressure_nivaa` reads fine; the five-factor weighted score behind it doesn't
- `Caseworker_Load_Concentration` — a Gini coefficient teaches a leader nothing on sight the way a percentage or a day count does; the color band is doing all the communicating, so keep the raw number out of any leader-facing view entirely

Note the asymmetry between the two flow/queue pages: `Phase_Bottleneck_Detector` earns
leader-report placement specifically because of `arsak_tekst` — its sibling
`Throughput_Pressure_Monitor` has no equivalent plain-language field, only the composite
score, so it stays in the analyst report despite being the same pattern one grain up.
That's a real difference in what the two tables expose, not an inconsistency to fix.

## Scripts — the addon signal DAX can't produce

| File | Output table(s) | What it computes |
|---|---|---|
| `CUSUM_Changepoint.py` | `cusum_analyse`, `pelt_analyse`, `pelt_analyse_detaljer` | CUSUM drift score + PELT changepoints — recursive/segmentation math |
| `Seasonal_YTD_ratio_extrapolation.py` | `frist_prognose` | Year-end forecast + confidence interval — a statistical model |
| `Throughput_Pressure_Monitor.py` | `gjennomstoremming_press_enhet`, `gjennomstroemming_press_fase` | Team-level flow imbalance + tidsbruk deviation vs. baseline — composite score, flow streak |
| `Phase_Bottleneck_Detector.py` | `fase_flaskehals_enhet` | Same, one level down at the phase grain |
| `Kostra.py` | `kostra_*` (one table per SSB series) | External data sync — not a governance algorithm |

`START_YEAR = 2015` is a top-of-file constant, not a shared config — it's duplicated as a
literal in four scripts: `CUSUM_Changepoint.py`, `Seasonal_YTD_ratio_extrapolation.py`,
`Throughput_Pressure_Monitor.py`, `Phase_Bottleneck_Detector.py`. If the earliest reliable
year in your Lakehouse changes, update it in all four — there's no single place that fixes
it for every script. (`Kostra.py` pulls whatever SSB has.)

Two things that used to be scripts here — daily-open-case age buckets and a persisted
Gini-concentration trend — were retired: a nightly pipeline plus an ever-growing table was
too much standing infrastructure for a future maintainer without a coding background to
keep running, for signals that don't need day-by-day history. See git history for
`Backlog_Aging_Distribution.py`/`sak_alder_fordeling` if that view is ever needed again;
`Caseworker_Load_Concentration` survived as a live DAX measure instead (see
`Caseworker_Load_Concentration_POWERBI_DAX.md`) since it's still worth having as a current
signal, just not a trend.

All (except `Kostra.py`) run nightly after the main data pipeline.

Throughput_Pressure_Monitor and Phase_Bottleneck_Detector were briefly native DAX; reverted
— the flow-streak/queue-proxy measures needed an iterative window-scan and a third date
role DAX has no clean primitive for, exactly the "possible but not simple" case above.

## Closed-case trend, drift, and forecast

### CUSUM_Changepoint.py

Detects small persistent shifts (CUSUM) and structural breakpoints (PELT) per indicator,
on both monthly and weekly series.

- **Måltall:** `Fristprosent`, `Behandlingstid`, `Produksjonsdifferanse`
- **No install step, no external dependency.** PELT runs on a built-in pure-NumPy
  implementation with an L2 (mean-shift) cost (`_pelt_l2` in `CUSUM_Changepoint.py`) —
  detects a shift in the average, not a volatility change with the average unchanged.
  Never add an inline `%pip install` cell for a changepoint library: it fails with
  `MagicUsageError` on tenants where inline installation is disabled, and isn't needed here
- **Key constants:** `CUSUM_K` (allowance), `CUSUM_H` (threshold), `CUSUM_BASELINE_MONTHLY`/`CUSUM_BASELINE_WEEKLY` (anchored baseline window for mu/sigma), `CUSUM_MIN_POST_BASELINE_OBS`
- `signal` is boolean; `signalretning`/`endringsretning` use `Stigende`/`Synkende`/`Stabil` —
  plain board vocabulary, so a board-level measure can read `cusum_analyse[signalretning]`
  directly with no translation SWITCH (`endringsretning` in `pelt_analyse` is never
  `Stabil` — a changepoint row only exists when a shift was actually detected)
- mu/sigma come from a fixed, anchored baseline window (the series' first N observations), not the whole history — a slow persistent drift would otherwise get partially absorbed into "normal" and dampen detection
- `cusum_analyse` stores only `cusum_positiv`/`cusum_negativ`/`signal` — the raw value is a
  live DAX measure against `Faser`, defined elsewhere in the model
- Board/governance trend direction (`Stigende`/`Synkende`/`Stabil`) is `Trendretning (CUSUM)`
  in `CUSUM_Changepoint_POWERBI_DAX.md` — the anchored-baseline CUSUM signal read directly,
  not a separately-computed DAX slope; every indicator has 20+ years of history, more than
  enough to build a CUSUM baseline from

**`pelt_analyse_detaljer`** breaks the most recent changepoint down by `enhet`/`fasetittel`, reusing PELT's before/after window instead of re-running detection.
- Only drills into changepoints within `RECENT_CHANGEPOINT_DAYS` (90) days old
- Saksbehandler is excluded — too thin per-segment volume, and individual-level flagging is out of scope
- `bidrag_til_endring` is each segment's volume-weighted share of the aggregate shift
- `tilstrekkelig_volum = FALSE` marks segments below `MIN_SEGMENT_OBS` (10) — don't trust these

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

## Workload

### Caseworker load concentration

Is active caseload concentrating on a few caseworkers within a team, even while the
team's aggregate numbers look fine? Entirely live DAX, no script — per-person
counts/shares (`Faser[saksansvarlig]` is already in the model) and the
Gini coefficient itself, via a `RANKX`/`SUMX` iterator over the current enhet x indikator
context, not a persisted trend table. See `Caseworker_Load_Concentration_POWERBI_DAX.md`
for the full measure set and the tradeoff (current signal only, no day-by-day history).

- Gini is scoped to enhet x indikator, never blended across an enhet's indicators (indicator
  effort/complexity isn't comparable and isn't in the data — blending could mask
  concentration on a heavier indicator behind a pile of lighter ones) and never broken down
  per person — individual-level flagging is out of scope for this layer, same reasoning as
  `CUSUM_Changepoint.py`'s drilldown exclusion.
- Gini on 1-2 people is meaningless — gated to enhet x indikator combinations with 3+ active
  caseworkers.

## External ingestion (not a governance algorithm)

### Kostra.py

Syncs selected SSB KOSTRA key-figure tables into the Lakehouse (`kostra_*`, one Delta
table per series), append-only new rows via a pandas dedup against the existing table.
Independent of the main pipeline and the rule above — this is a data source, not an
analysis.
