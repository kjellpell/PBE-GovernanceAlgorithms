# PBE-GovernanceAlgorithms

DAX measures and docs for monitoring case processing (`Fristprosent`, production, portfolio,
phase time) in the Power BI reports, plus one data-ingestion script (`Kostra.py`). All views
read the same fact table (`saksbehandling.faser`, `pk_indikator` from `felles.indikator`).
Output values and DAX measure names are in Norwegian.

## One rule

**Everything is live DAX against the fact table, unless it truly can't be.** A script is
only justified for a genuine algorithm DAX can't express simply, or for a value that needs
its history preserved day by day. As of October 2026 no governance script meets that bar:
every view in the reports is plain DAX, and there are no nightly governance jobs to keep
running. That's deliberate — a future maintainer without a coding background must be able
to own this.

Every number on a leader page must answer a question a leader can act on. Statistically
interesting but not actionable views (drift scores, changepoints, control charts, ranges)
were tried and removed — see *Retired* below.

## What's here

| File | Report | Answers |
|---|---|---|
| `Status_i_aar_POWERBI_DAX.md` | «Status i år» table, one row per indicator | Where does frist% end at year-end at today's pace? How much must we produce per month to reach the production target, and what do we produce now? |
| `At_Risk_Open_Cases_POWERBI_DAX.md` (Del 1) | Product pages | Which open faser are red/yellow/green by time left to the deadline (the report's own colour code), with `Status` alongside? Today only |
| `At_Risk_Open_Cases_POWERBI_DAX.md` (Del 2) | Product pages | Is the portfolio growing? Intake, completions, net flow and open faser per month back to 2015 |
| `At_Risk_Open_Cases_POWERBI_DAX.md` (Del 3) | Product pages | Where is work getting slow? P90 phase time (`tidsbruk`, our time only) per phase against its last 12 months |
| `Kostra.py` | — | External data: syncs selected SSB KOSTRA tables into the Lakehouse |

### Notes that apply across the measures
- **Totals across indicators are not meaningful** (different deadlines, targets and
  volumes). Show figures per indicator, not as a single KPI card.
- **`tidsbruk` is our time only:** it pauses while the applicant completes documentation
  (that time is `bransjetid`). Use it for anything about the deadline; calendar lead time
  includes the applicant's waiting.
- **Seasonality:** only volume (intake/production) is seasonal. Frist% and phase time are
  not, so their "normal" is simply the recent past (e.g. the last 12 months).
- **Deadlines are 3 or 12 weeks.** Don't mix the two in one time measure; split by deadline
  type or express time relative to the deadline.
- **Small numbers:** a team × phase × indicator in one week is often 10–60 cases — too few
  for percentages or percentiles to mean much. Combine, or hide signals below ~30 cases.

## Kostra.py

Syncs selected SSB KOSTRA key-figure tables into the Lakehouse (`kostra_*`, one Delta table
per series), appending only new rows via a pandas dedup against the existing table. A data
source, not an analysis; independent of the rule above.

## Retired (October 2026 unless noted) — see git history

| Removed | Why | Replaced by |
|---|---|---|
| `Backlog_Aging_Distribution.py` / `sak_alder_fordeling`, persisted Gini trend (earlier) | Nightly pipeline + ever-growing table: too much infrastructure | — |
| Caseworker Gini page | Hard to read, individual-level monitoring, not an early warning | Team-level red/yellow counts |
| `CUSUM_Changepoint.py` (CUSUM) | Slow on a lagging measure (~9 months to see a 1σ shift), frequent false alarms, opaque to leaders | Red/yellow open faser; the report's monthly frist% tables |
| `Throughput_Pressure_Monitor.py` | Five-factor point score leaders couldn't check | Del 2 flow measures |
| `Phase_Bottleneck_Detector.py` | Point score on cells too small for a stable P90 | The report's net production + Del 3 P90 per phase |
| `Pause_Reconstruction.py` (never deployed) | Rebuilt pause history from milestones covering only ~12 months; `tidsbruk` already gives today's clock | `tidsbruk` directly |
| Control charts (`Fristprosent_Kontrollkort_POWERBI_DAX.md`) | Leaders found them not actionable | Production gap and year-end projection in «Status i år» |
| `PELT_Changepoint.py` (PELT changepoints + drill-down) | Retrospective (needs ~6 months after a shift), doesn't point to a decision, largest script to maintain. For evaluating a *known* organisational change, a one-off before/after analysis at the known date is better | — |
| `Seasonal_YTD_ratio_extrapolation.py` (`frist_prognose`) | Frist% isn't seasonal, so the seasonal model in practice reproduced year-to-date; its range and projection line weren't used for decisions | `Status_i_aar_POWERBI_DAX.md`: year-end at today's pace, plain DAX |
