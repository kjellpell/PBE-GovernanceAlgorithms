# PBE-GovernanceAlgorithms

Statistical governance algorithms for indicator time series. Designed to run as nightly Fabric Notebooks (PySpark + pandas). All output tables are Delta Lake, all output values in Norwegian, DAX measure names in Norwegian.

## Scripts

| File | Output table(s) | Schedule |
|---|---|---|
| `CUSUM_Changepoint.py` | `cusum_analyse`, `pelt_analyse`, `pelt_analyse_detaljer` | Nightly after main data pipeline |
| `EWMA.py` | `ewma_analyse` | Nightly after main pipeline |
| `Seasonal_YTD_ratio_extrapolation.py` | `frist_prognose` | Nightly after main pipeline |
| `Throughput_Pressure_Monitor.py` | `gjennomstoremming_press_enhet`, `gjennomstroemming_press_fase` | Nightly after main data pipeline |
| `Phase_Bottleneck_Detector.py` | `fase_flaskehals_enhet` | Nightly after main data pipeline |
| `Plansak_Cohort_Analysis.py` | `plansak_2026_phase_detail`, `plansak_2026_dim_cases`, `plansak_2026_kpi_snapshot` | Nightly after main data pipeline |
| `Building_Application_Type.py` | `building_application_type` | Nightly |
| `Kostra.py` | `kostra_*` (one Delta table per SSB KOSTRA series, `kostra_` prefix) | Independent — SSB API sync, not part of the governance-algorithm pipeline |
| `Inflight_SLA_Risk_Monitor.py` | `sak_frist_risiko`, `sak_frist_risiko_trend` | Nightly after main data pipeline |
| `Backlog_Aging_Distribution.py` | `sak_alder_fordeling` | Nightly after main data pipeline |
| `Caseworker_Load_Concentration.py` | `saksbehandler_arbeidsmengde`, `saksbehandler_konsentrasjon` | Nightly after main data pipeline |
| `Process_Change_Impact_Analysis.py` | `prosessendring_effekt` | Nightly, ideally after `CUSUM_Changepoint.py` |

## CUSUM_Changepoint.py

Detects small persistent shifts (CUSUM) and structural breakpoints (PELT) per indicator.

- **Måltall:** `Fristprosent`, `Behandlingstid`, `Produksjonsdifferanse`
- **Granularitet:** `Månedlig` og `Ukentlig`
- **Minimum historikk:** 24 månedlige / 52 ukentlige observasjoner
- **External dependency:** `pip install ruptures` (PELT only — CUSUM runs without it)
- **Key constants:** `CUSUM_K` (allowance), `CUSUM_H` (threshold), `START_YEAR`, `CUSUM_BASELINE_MONTHLY`/`CUSUM_BASELINE_WEEKLY` (anchored baseline window for mu/sigma), `CUSUM_MIN_POST_BASELINE_OBS`
- `signalretning` og `endringsretning` har verdiene `Økning` og `Nedgang`.
- `signal` er en boolsk verdi.
- CUSUM's mu/sigma are computed from a fixed, anchored baseline window (the series' first `CUSUM_BASELINE_MONTHLY`/`CUSUM_BASELINE_WEEKLY` observations), not the whole history — otherwise a slow persistent drift would partially get absorbed into "normal" and dampen detection sensitivity.

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

## Seasonal_YTD_ratio_extrapolation.py

Projects year-end `frist%` from current YTD using trimmed seasonal ratios from historical years.

- **Minimum history:** 3 complete years per indicator
- **Confidence interval:** 80% (z=1.28), derived from ratio variance (delta method)
- `type = 'actual'` for past months, `type = 'forecast'` for remaining months
- Idempotent — deletes and rewrites current-year rows on each run

## Throughput_Pressure_Monitor.py

Detects where intake exceeds completions for multiple periods and where processing time deteriorates versus a recent baseline, at the team (`enhet`) level.

- **Output:** `gjennomstoremming_press_enhet` (team × indikator × month), `gjennomstroemming_press_fase` (fase-level support table showing which faser drive a team's pressure score)
- `pressure_nivaa` har verdiene `Lav`, `Moderat`, `Hoy`, `Kritisk`
- **Key constants:** `BASELINE_MONTHS`, `MIN_BASELINE_OBS`, `MIN_TEAM_VOLUME`, `POSITIVE_FLOW_STREAK`
- Full overwrite each run (monthly time series recomputed from full history)

## Phase_Bottleneck_Detector.py

Same pattern as `Throughput_Pressure_Monitor.py`, one level down: detects queue pressure forming inside individual process phases (fase-nettoflyt + p90 fasetid deviation vs. baseline).

- **Output:** `fase_flaskehals_enhet` (enhet × fasetittel × indikator × month)
- `alvorlighet` har verdiene `Lav`, `Moderat`, `Hoy`, `Kritisk`; `arsak_kode`/`arsak_tekst` explain the flag
- **Key constants:** `BASELINE_MONTHS`, `MIN_BASELINE_OBS`, `MIN_SEGMENT_OBS`

## Plansak_Cohort_Analysis.py

Tracks the 2026 planning-case cohort (Oppstartsmøte >= 2026-01-01) against a
rolling per-case deadline: every case must reach "Sendt til politisk
behandling" within 3 years of its own Oppstartsmøte date.

- **Four phases per case:** Utarbeidingsfasen Kostra, Offentlig ettersyn,
  Høringsperiode, Sendt til politisk behandling
- **Case_Deadline** = each case's own Oppstartsmøte date + 3 years (not a
  fixed portfolio-wide date)
- **Output:** `plansak_2026_phase_detail` (one row per Saksnummer × Phase),
  `plansak_2026_dim_cases` (one row per Saksnummer),
  `plansak_2026_kpi_snapshot` (aggregate KPIs, append-mode idempotent per day)
- `Overall_Status` in `dim_cases`: `On Track`, `At Risk`, `Off Track`, `Completed`

## Building_Application_Type.py

For each process recorded in Fakturalinjer, identifies the product code accounting for the largest total invoice amount — a raw product code downstream models join against `prisliste_varer` and `Prosesser`.

- **Output:** `building_application_type` (one row per `fk_faser`, full overwrite)
- Scoped to Byggesak and Eiendomssak fagomraade (not Plansak)
- No date filter — all Fakturalinjer rows in scope are included

## Inflight_SLA_Risk_Monitor.py

Leading indicator of SLA-breach risk. `Fristprosent` (EWMA/CUSUM) only scores cases that have already closed; this script scores cases that are **still open** against their own `frist_dager`, so a breach wave shows up here before it reaches the closed-case ratio.

- **Output:** `sak_frist_risiko` (one row per open case-phase `pk_faser`, full overwrite nightly — current-state detail), `sak_frist_risiko_trend` (indikator × enhet × snapshot_dato, append-mode idempotent per day — trend)
- `risikoklasse` har verdiene `Bruddet`, `Kritisk`, `Risiko`, `Innenfor`
- **Key constants:** `RISK_THRESHOLD_KRITISK` (0.90), `RISK_THRESHOLD_RISIKO` (0.75), `MIN_TEAM_VOLUME`
- Open assumption to verify against the Lakehouse schema: whether `frist_dager` is reliably populated on rows that haven't closed yet — rows missing it are excluded, not defaulted.

## Backlog_Aging_Distribution.py

Tracks whether the *existing* open-case backlog is aging, independent of `Throughput_Pressure_Monitor.py`'s net-flow score (which measures flow imbalance, not the age of work already in the queue).

- **Output:** `sak_alder_fordeling` (indikator × enhet × aldersgruppe × snapshot_dato, append-mode idempotent per day). `snapshot_dato` carries both roles — filter to `MAX(snapshot_dato)` for today's backlog shape, or chart the full table for the trend.
- `aldersgruppe` buckets (default `AGE_BUCKETS`): `0-30`, `31-60`, `61-90`, `91-180`, `180+`
- Percentiles (`median_alder_dager`, `p90_alder_dager`) computed in pandas/numpy, not Spark SQL — the grouping key is pandas-derived and backlog volume is small.

## Caseworker_Load_Concentration.py

Team-level workload concentration / bus-factor / burnout early warning: is active caseload piling up on a few caseworkers within a team, even while the team's aggregate numbers look fine? Concentration is measured with the Gini coefficient of open-caseload counts per saksbehandler within each enhet.

- **Deliberate exception to repo convention:** this is the only script here that persists per-individual (per-saksbehandler) data. It exists for internal manager capacity-planning/workload-balancing use — **not** for automated escalation or individual performance flagging (the same reason `CUSUM_Changepoint.py`'s drilldown explicitly excludes saksbehandler). Do not repurpose `saksbehandler_arbeidsmengde` for automated per-person alerting.
- **Output:** `saksbehandler_arbeidsmengde` (enhet × saksbehandler, **full overwrite nightly, no history retained at the individual grain**), `saksbehandler_konsentrasjon` (enhet × snapshot_dato, append-mode idempotent per day — Gini trend, **no individual data**, only accumulates history at the enhet level)
- **Key constants:** `MIN_SAKSBEHANDLERE` (3) — Gini on 1-2 people is meaningless, gates `tilstrekkelig_volum`
- `SAKSBEHANDLER_COL` is unverified against the Lakehouse schema — verify before relying on this script.

## Process_Change_Impact_Analysis.py

Did a specific process change actually work, net of seasonality and org-wide drift?
`Plansak_Cohort_Analysis.py` has no control group and takes years to resolve; a naive
before/after comparison is confounded by exactly the seasonality
`Seasonal_YTD_ratio_extrapolation.py` models and the secular drift `CUSUM_Changepoint.py`
detects. This script instead computes a **difference-in-differences (DiD)** estimate:
the before→after change in the affected population, net of the before→after change in an
unaffected control population over the same window.

- **Config:** hand-edit the `PROCESS_CHANGES` list whenever a real process change ships (ships with one template entry — replace or delete it)
- **Scope:** `Fristprosent` and `Behandlingstid` only — `Produksjonsdifferanse` has no per-case realization and can't be split into treatment/control rows
- **Control group is optional per entry** — DiD when configured; otherwise a plain before/after with `har_kontrollgruppe = FALSE`, never silently skipped
- **`effekt_retning`** har verdiene `Forbedring`, `Forverring`, `Ingen praktisk effekt` (statistically real but below the configured minimum size), `Ingen sikker effekt` (not statistically significant)
- **Volume reality:** some phases see only ~30 cases/year, so `DEFAULT_VINDU_DAGER = 365` (a full year each side — also cancels seasonality on its own) and `MIN_OBS_PER_GROUP = 10` is a pragmatic floor, not a statistical ideal; thinness above that floor is surfaced via `lav_styrke` rather than suppressed. A mature reading (`tilstrekkelig_moden = TRUE`) is typically ~12 months after rollout — earlier snapshots are trend signal, not a conclusion
- **`pelt_stotte`** cross-references `analyser.pelt_analyse` as corroborating context only — never feeds back into `effekt_retning`
- Output: `prosessendring_effekt`, append-mode idempotent per day, so the confidence interval visibly narrows across successive nightly runs on the same change

## Configuration

All scripts share `START_YEAR = 2015` at the top. Adjust to match the earliest reliable data in your Lakehouse.
