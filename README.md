# PBE-GovernanceAlgorithms

Statistical governance algorithms for indicator time series. Designed to run as nightly Fabric Notebooks (PySpark + pandas). All output tables are Delta Lake, all output values in Norwegian, DAX measure names in Norwegian.

## Scripts

| File | Output table(s) | Schedule |
|---|---|---|
| `CUSUM_Changepoint.py` | `cusum_results`, `changepoint_results` | Nightly after main pipeline |
| `EWMA.py` | `ewma_results` | Nightly after main pipeline |
| `Cohort_analysis.py` | `cohort_results` | Nightly after main pipeline |
| `Portfolio_state_snapshot.py` | `portefolje_snapshot` | First working day of each month |
| `Seasonal_YTD_ratio_extrapolation.py` | `projection_results` | Nightly after main pipeline |

## CUSUM_Changepoint.py

Detects small persistent shifts (CUSUM) and structural breakpoints (PELT) per indicator.

- **Metrics:** `frist_pct`, `tidsbruk`, `prod_diff`
- **Granularity:** monthly and weekly
- **Minimum history:** 24 monthly / 52 weekly observations
- **External dependency:** `pip install ruptures` (PELT only — CUSUM runs without it)
- **Key constants:** `CUSUM_K` (allowance), `CUSUM_H` (threshold), `START_YEAR`
- `signal_direction` values: `OPP` (improving), `NED` (deteriorating)

## EWMA.py

Exponentially weighted moving average smoothing for trend lines in board and governance reports.

- **Metrics:** `frist_pct`, `tidsbruk`, `prod_diff`
- **Two speeds:** `ewma_slow` (α=0.1, board), `ewma_fast` (α=0.3, governance team)
- `trend_direction` values: `Stigende`, `Synkende`, `Stabil`
- Full overwrite each run (EWMA depends on full history)

## Cohort_analysis.py

Tracks resolution rate of cases grouped by intake month. Compares recent cohorts against a trimmed historical average.

- **Output:** one row per cohort × week (up to 26 weeks after intake)
- **Minimum cohort size:** 10 cases
- `is_recent_cohort = TRUE` for the last 6 full months
- `delta_historical`: positive = resolving slower than historical average

## Portfolio_state_snapshot.py

Classifies all open cases at month-end by milestone state: workable, paused, mini-hearing (Begrenset høring), and tidligbehandling. Requires `Milepeler` (milestone event log).

- **Snapshot date:** last day of previous calendar month (override: `SNAPSHOT_MONTH`)
- Append-only table — idempotent via delete-then-append per snapshot month
- `indikator NOT LIKE '%avtalt%'` for tidligbehandling count

## Seasonal_YTD_ratio_extrapolation.py

Projects year-end `frist%` from current YTD using trimmed seasonal ratios from historical years.

- **Minimum history:** 3 complete years per indicator
- **Confidence interval:** 80% (z=1.28), derived from ratio variance (delta method)
- `type = 'actual'` for past months, `type = 'forecast'` for remaining months
- Idempotent — deletes and rewrites current-year rows on each run

## Configuration

All scripts share `START_YEAR = 2015` at the top. Adjust to match the earliest reliable data in your Lakehouse. Milestone names in `Portfolio_state_snapshot.py` (`PAUSED_MILESTONES`, `RESUME_MILESTONES`, etc.) must match the actual values in `Milepeler.Milepel`.
