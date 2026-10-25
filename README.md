# PBE-GovernanceAlgorithms

One system for monitoring case-processing indicators (`Fristprosent`, `Behandlingstid`,
`Produksjonsdifferanse`) from several angles: trend, drift, forecast, flow. All views read the same fact table
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
- `At_Risk_Open_Cases` (Del 1) — open faser by the report's existing deadline colour code (red = past deadline, yellow = close to it, threshold per case type, green = the rest), with `Status` alongside: counts and shares, live DAX
- `At_Risk_Open_Cases` (Del 2) — intake, completions and open cases per month back to 2015, live DAX
- `At_Risk_Open_Cases` (Del 3) — P90 phase time per phase against its last 12 months, red dot when 25 % above; on the same timeline as net production, live DAX

**Analytiker-rapport** — the label is fine, but the "why" is a statistical construct one
step removed from the raw quantity, and needs someone who can defend the method:
- `PELT_Changepoint` — when an unexplained shift in the average happened and which team/phase it came from; look at it when needed
- `Seasonal_YTD_ratio_extrapolation` — the forecast range reads fine, but trusting the number means trusting the seasonal-ratio method behind it

## Scripts — the addon signal DAX can't produce

| File | Output table(s) | What it computes |
|---|---|---|
| `PELT_Changepoint.py` | `pelt_analyse`, `pelt_analyse_detaljer` | PELT changepoints + drill-down — segmentation math |
| `Seasonal_YTD_ratio_extrapolation.py` | `frist_prognose` | Year-end forecast + confidence interval — a statistical model |
| `Kostra.py` | `kostra_*` (one table per SSB series) | External data sync — not a governance algorithm |

`START_YEAR = 2015` is a top-of-file constant, not a shared config — it's duplicated as a
literal in two scripts: `PELT_Changepoint.py` and `Seasonal_YTD_ratio_extrapolation.py`.
If the earliest reliable year in your Lakehouse changes, update it in both — there's no single place that fixes
it for every script. (`Kostra.py` pulls whatever SSB has.)

Three things that used to be here were retired:
- daily-open-case age buckets and a persisted Gini-concentration trend (scripts) — a
  nightly pipeline plus an ever-growing table was too much standing infrastructure for a
  future maintainer without a coding background to keep running. See git history for
  `Backlog_Aging_Distribution.py`/`sak_alder_fordeling`.
- the live-DAX caseworker Gini page (`Caseworker_Load_Concentration_POWERBI_DAX.md`,
  retired October 2026) — hard for leaders to read, muddied by part-time and leave,
  individual-level monitoring that needs a separate employee-representative process, and
  not an early-warning signal. Team-level at-risk counts answer the operational question
  without naming individuals. See git history if it is ever needed again.
- CUSUM drift detection (`cusum_analyse`, `Trendretning (CUSUM)`, retired October 2026) —
  on a lagging closed-case measure it was slow (~9 months to see a sustained 1σ shift in
  monthly data) and, with a recent same-month baseline, raised frequent false alarms; a
  leader also couldn't see why the word on the card changed. Replaced by the leading
  indicators and the report's own monthly frist% tables. PELT, which shared the script, stays
  as `PELT_Changepoint.py`. See git history for `CUSUM_Changepoint.py`.
- Control charts for production and frist% (`Fristprosent_Kontrollkort_POWERBI_DAX.md`,
  removed October 2026) — leaders found them not actionable: a red dot saying a month was
  unusual doesn't point to a decision. The actionable versions are the gap to the annual
  production target (required per month vs. the last 3 months' average) and the year-end
  projection, both as columns per indicator in the report's year table. See git history.
- `Throughput_Pressure_Monitor.py` (`gjennomstroemming_press_enhet`/`_fase`, retired
  October 2026) — its team-level flow (intake vs. completions, open queue) is now plain
  live DAX back to 2015 (`At_Risk_Open_Cases_POWERBI_DAX.md`, Del 2), read directly instead
  of a five-factor score.
  See git history.
- `Phase_Bottleneck_Detector.py` (`fase_flaskehals_enhet`, retired October 2026) — a
  five-rule point score per team × phase × indicator × month, on cells often too small
  (10–30 cases) for a stable P90. Replaced by the report's existing net production and
  open-cases measures (is the queue growing?) plus live-DAX P90 phase time per phase
  against its last 12 months (`At_Risk_Open_Cases_POWERBI_DAX.md`, Del 3: where is work
  getting slow?). See git history.

All (except `Kostra.py`) run nightly after the main data pipeline.

`PELT_Changepoint` scores
only **complete** periods: the current calendar month (and the current ISO week for
PELT's weekly series) is excluded in SQL (`MONTH_CUTOFF_SQL`/`WEEK_CUTOFF_SQL`), so a
half-filled month never reads as a drop, and the DAX "siste analysedato" measures land on
the last complete period. `Seasonal_YTD_ratio_extrapolation` is the exception on purpose —
it handles the part-month itself and anchors on the last complete month.

## Closed-case trend, drift, and forecast

### PELT_Changepoint.py

Finds structural breakpoints (PELT) in the average of each indicator's series, monthly and
weekly, and breaks the latest one down by team and phase. Runs nightly; look at it when
needed (e.g. quarterly). It looks backwards: it finds breaks nobody logged (registration practice,
system migration, rule change) and needs ~6 periods after a break to see it, so it is not
an early warning. To evaluate a *known* organisational change, use an interrupted time
series at the known date, not PELT.

- **Måltall:** `Fristprosent`, `Behandlingstid`, `Produksjonsdifferanse`
- `Fristprosent` has one official definition across the repo: produced faser (start and
  slutt both set) as denominator, `innenfor_frist = 1` as numerator, `%avtalt%`
  indicators included — the same as `Seasonal_YTD_ratio_extrapolation.py` and the
  report's `[Produserte faser]`. The other two måltall still exclude `%avtalt%`
  (`Produksjonsdifferanse` = faser started in the period by start date minus faser ended
  in the period by end date)
- Weekly series are keyed `ISO year * 100 + ISO week` (`iso_week_sql`), so 30 Dec 2024
  is week 202501, not 202401. Mixing calendar year with ISO week mis-placed New Year
  days and would crash the nightly run on a date like 1 Jan 2021 (`202153` does not exist)
- **No install step, no external dependency.** PELT runs on a built-in pure-NumPy
  implementation with an L2 (mean-shift) cost (`_pelt_l2` in `PELT_Changepoint.py`) —
  detects a shift in the average, not a volatility change with the average unchanged.
  Never add an inline `%pip install` cell for a changepoint library: it fails with
  `MagicUsageError` on tenants where inline installation is disabled, and isn't needed here
- **Key constants:** `PELT_MIN_SIZE` (6), `MIN_SEGMENT_OBS` (10), `RECENT_CHANGEPOINT_DAYS`
- `endringsretning` in `pelt_analyse` is `Stigende`/`Synkende`, never `Stabil` — a
  changepoint row only exists when a shift was actually detected
- PELT's penalty uses a robust noise estimate, `1.4826 · MAD(diff(values)) / √2`, not
  `np.std(values)` — real shifts inflate the plain std, and a large old shift could hide
  a moderate recent one entirely

**`pelt_analyse_detaljer`** breaks the most recent changepoint down by `enhet`/`fasetittel`, reusing PELT's before/after window instead of re-running detection.
- The window is PELT's own segment: from the previous breakpoint (or series start) to the
  next one (or the last complete period), split at the first day of the breakpoint period
- Only drills into changepoints recent enough to act on: `RECENT_CHANGEPOINT_DAYS` is 240
  days for monthly, 90 for weekly (a monthly changepoint is always ≥ ~6 months old, since
  PELT needs 6 periods after it)
- Saksbehandler is excluded — too thin per-segment volume, and individual-level flagging is out of scope
- `bidrag_til_endring` is each segment's volume-weighted share of the aggregate shift, split
  into `bidrag_rate` (the segment got better/worse) + `bidrag_miks` (the segment got
  bigger/smaller). For `Produksjonsdifferanse` values are per period and the mix part is 0
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
- **Pace scenario («med dagens tempo»):** since frist% isn't seasonal, the model in practice
  says "the year ends near where year-to-date is now", and doesn't see whether the last few
  months were better or worse. So each row also carries `innenfor_tempo` (remaining months'
  projected volume × the frist% of the last `TEMPO_MAANEDER` = 3 complete months) and
  `prognose_aarsslutt_tempo`, drawn as a second dashed line from the same fork. Two
  endpoints close together: the projection holds; clearly apart: the recent pace is
  changing the picture. Added as columns, not rows, so existing measures are unaffected
- Idempotent and atomic — replaces current-year rows in one Delta transaction
  (`overwrite` + `replaceWhere`), so a failed run leaves last night's forecast in place
  instead of an empty year. No forecast history is kept (a history table was considered
  and not added — extra standing infrastructure)

## Leading indicator

### At-risk open faser (live DAX)

Every other signal here is computed on **closed** faser, so it lags: a fase that will
breach its deadline only shows up when it closes. `At_Risk_Open_Cases_POWERBI_DAX.md`
(Del 1) counts **open** faser by the colour code the case overview already uses for time
left to the deadline: red = past it, yellow = within a threshold that depends on the case
type (1–4 weeks before a 3- or 12-week deadline), green = the rest. It reuses that rule
rather than inventing a new "danger zone", so the numbers match what caseworkers and leaders
already see. The colour ignores `Status` by design; `Status` is a slicer/column next to it,
so the same measures answer "all open" and "only those with us". Entirely live DAX; today
only (the colour depends on today's date). Earlier versions of this page (a 75 % threshold
on `tidsbruk`, and the retired `Inflight_SLA_Risk_Monitor`) were dropped in favour of the
existing code.

### Phase time: P90 per phase (live DAX)

Del 3 of the same doc. Net production and open cases show *whether* the queue grows; P90
phase time (`tidsbruk`, our time only) shows *where* work gets slow, since the slow tail is
what causes deadline misses. Normal = P90 over the previous 12 months (P90 is not seasonal
in this data; long phase times smear a summer dip over many months). Red dot when 25 %
above normal; blank below 30 closed faser. Phases are combined across teams so there are
enough cases per month.

### Flow over time (live DAX)

Del 2 of the same doc: `Mottatt (periode)`, `Ferdigstilt (periode)`, `Netto flyt` and
`Åpne ved periodeslutt` per month back to 2015, from start and end dates alone (`TREATAS`
onto the right date column, so no inactive relationships are needed). The colour
counts are today only: the colour depends on today's date, and no table keeps yesterday's. A trend for those would need a small nightly snapshot. That was considered and
not added; reconstructing it from the completion milestones was tried and dropped
(October 2026), because the milestones only cover ~12 months and the system's own
`tidsbruk` already gives the clock for today.

## External ingestion (not a governance algorithm)

### Kostra.py

Syncs selected SSB KOSTRA key-figure tables into the Lakehouse (`kostra_*`, one Delta
table per series), append-only new rows via a pandas dedup against the existing table.
Independent of the main pipeline and the rule above — this is a data source, not an
analysis.
