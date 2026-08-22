# PBE-GovernanceAlgorithms

Statistical governance algorithms for indicator time series. Designed to run as nightly Fabric Notebooks (PySpark + pandas). All output tables are Delta Lake, all output values in Norwegian, DAX measure names in Norwegian.

## Scripts

| File | Output table(s) | Schedule |
|---|---|---|
| `CUSUM_Changepoint.py` | `cusum_analyse`, `pelt_analyse`, `pelt_analyse_detaljer` | Nightly after main data pipeline |
| `EWMA.py` | `ewma_analyse` | Nightly after main pipeline |
| `Seasonal_YTD_ratio_extrapolation.py` | `frist_prognose` | Nightly after main pipeline |
| `Throughput_Pressure_Monitor.py` | `gjennomstoremming_press_enhet`, `gjennomstroemming_press_fase` | Nightly after main pipeline |
| `Phase_Bottleneck_Detector.py` | `fase_flaskehals_enhet` | Nightly after main pipeline |
| `Stalled_Case_Early_Warning.py` | `sak_stillstand_varsler` | Nightly after main pipeline |

## CUSUM_Changepoint.py

Detects small persistent shifts (CUSUM) and structural breakpoints (PELT) per indicator.

- **Måltall:** `Fristprosent`, `Behandlingstid`, `Produksjonsdifferanse`
- **Granularitet:** `Månedlig` og `Ukentlig`
- **Minimum historikk:** 24 månedlige / 52 ukentlige observasjoner
- **External dependency:** `pip install ruptures` (PELT only — CUSUM runs without it)
- **Key constants:** `CUSUM_K` (allowance), `CUSUM_H` (threshold), `START_YEAR`
- `signalretning` og `endringsretning` har verdiene `Økning` og `Nedgang`.
- `signal` er en boolsk verdi.

### pelt_analyse_detaljer

Breaks down the most recent PELT changepoint per indikator/maaltall/granularitet by `enhet` (team) and `fasetittel` (process step), reusing PELT's before/after window instead of re-running changepoint detection.

- Only drills into changepoints within `RECENT_CHANGEPOINT_DAYS` (default 90) — old changepoints aren't re-drilled every night.
- Saksbehandler is intentionally excluded — too thin per-segment volume, and individual-level automated flagging is out of scope for this layer.
- `bidrag_til_endring` is each segment's share of the aggregate shift (volume-weighted, sums to `pelt_analyse.endringsstoerrelse`).
- `tilstrekkelig_volum = FALSE` marks segments below `MIN_SEGMENT_OBS` (default 10) — don't rank or trust these.
- **Key constants:** `DRILLDOWN_DIMENSIONS`, `MIN_SEGMENT_OBS`, `RECENT_CHANGEPOINT_DAYS`

## EWMA.py

Exponentially weighted moving average smoothing for trend lines in board and governance reports.

- **Måltall:** `Fristprosent`, `Behandlingstid`, `Produksjonsdifferanse`
- **To hastigheter:** `ewma_sakte` (α=0.1, styret), `ewma_rask` (α=0.3, virksomhetsoppfølging)
- `trendretning` har verdiene `Stigende`, `Synkende`, `Stabil`.
- Full overwrite each run (EWMA depends on full history)

## Cohort_analysis.py

Tracks resolution rate of cases grouped by intake month. Compares recent cohorts against a trimmed historical average.

- **Output:** one row per cohort × week (up to 26 weeks after intake)
- **Minimum cohort size:** 10 cases
- `is_recent_cohort = TRUE` for the last 6 full months
- `delta_historical`: positive = resolving slower than historical average

## Seasonal_YTD_ratio_extrapolation.py

Projects year-end `frist%` from current YTD using trimmed seasonal ratios from historical years.

- **Minimum history:** 3 complete years per indicator
- **Confidence interval:** 80% (z=1.28), derived from ratio variance (delta method)
- `type = 'actual'` for past months, `type = 'forecast'` for remaining months
- Idempotent — deletes and rewrites current-year rows on each run

## Configuration

All scripts share `START_YEAR = 2015` at the top. Adjust to match the earliest reliable data in your Lakehouse.
