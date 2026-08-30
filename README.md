# PBE-GovernanceAlgorithms

Statistical governance algorithms for indicator time series. Designed to run as nightly Fabric Notebooks (PySpark + pandas). All output tables are Delta Lake, all output values in Norwegian, DAX measure names in Norwegian.

## Scripts

| File | Output table(s) | Schedule |
|---|---|---|
| `CUSUM_Changepoint.py` | `cusum_analyse`, `pelt_analyse`, `pelt_analyse_detaljer` | Nightly after main data pipeline |
| `Seasonal_YTD_ratio_extrapolation.py` | `frist_prognose` | Nightly after main pipeline |
| `Kostra.py` | `kostra_*` (one Delta table per SSB KOSTRA series, `kostra_` prefix) | Independent — SSB API sync, not part of the governance-algorithm pipeline |
| `Inflight_SLA_Risk_Monitor.py` | `sak_frist_risiko_trend` (trend snapshot only — today's list is live DAX) | Nightly after main data pipeline |
| `Backlog_Aging_Distribution.py` | `sak_alder_fordeling` (trend snapshot only — today's shape is live DAX) | Nightly after main data pipeline |
| `Caseworker_Load_Concentration.py` | `saksbehandler_konsentrasjon` (Gini trend only — today's per-person counts are live DAX) | Nightly after main data pipeline |

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
- `cusum_analyse` does **not** store the raw underlying value (Fristprosent/Behandlingstid/Produksjonsdifferanse) — that's a plain live DAX measure against `saksbehandling.faser` (see `CUSUM_Changepoint_POWERBI_DAX.md`), joined on indikator/maaltall/granularitet/analyse_dato. Only `cusum_positiv`/`cusum_negativ`/`signal` are stored — the recursive reset-at-each-step math genuinely can't be a DAX measure.

### pelt_analyse_detaljer

Breaks down the most recent PELT changepoint per indikator/maaltall/granularitet by `enhet` (team) and `fasetittel` (process step), reusing PELT's before/after window instead of re-running changepoint detection.

- Only drills into changepoints within `RECENT_CHANGEPOINT_DAYS` (default 90) — old changepoints aren't re-drilled every night.
- Saksbehandler is intentionally excluded — too thin per-segment volume, and individual-level automated flagging is out of scope for this layer.
- `bidrag_til_endring` is each segment's share of the aggregate shift (volume-weighted, sums to `pelt_analyse.endringsstoerrelse`).
- `tilstrekkelig_volum = FALSE` marks segments below `MIN_SEGMENT_OBS` (default 10) — don't rank or trust these.
- **Key constants:** `DRILLDOWN_DIMENSIONS`, `MIN_SEGMENT_OBS`, `RECENT_CHANGEPOINT_DAYS`

## Trendretning (native DAX, no nightly script)

Trend direction for board and governance report charts used to be computed by a nightly
`EWMA.py` script (exponentially weighted moving average, `ewma_analyse` table). It was
removed: the only output people actually used was the `Stigende`/`Synkende`/`Stabil` label,
and EWMA is harder to explain to a non-technical audience than a plain moving average for
no real benefit at monthly board-reporting resolution. The label is now computed directly
in Power BI with a rolling-average-slope DAX measure — see `Trendretning_POWERBI_DAX.md`.

- **Måltall:** `Fristprosent`, `Behandlingstid`, `Produksjonsdifferanse`
- **To hastigheter:** 6-month rolling average (styret), 3-month rolling average
  (virksomhetsoppfølging) — same board-vs-operational split as before, just as plain
  moving-average windows instead of EWMA alpha values
- `trendretning` still has the values `Stigende`, `Synkende`, `Stabil`, derived from the
  slope of the 6-month rolling average, same as the old script's slow-EWMA-based label

## Seasonal_YTD_ratio_extrapolation.py

Projects year-end `frist%` from current YTD using trimmed seasonal ratios from historical years. **Minimal script — forecast only.** Actual YTD is a plain live DAX year-to-date measure against `saksbehandling.faser` (standard time intelligence, no algorithm needed — see `Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md`, Del 1); the script's only remaining job is the part DAX genuinely can't do: the trimmed seasonal-ratio model and its confidence interval.

- **Minimum history:** 3 complete years per indicator
- **Confidence interval:** 80% (z=1.28), derived from ratio variance (delta method)
- `frist_prognose` now holds only forecast rows (`type` is always `'Prognose'`) for the remaining months of the current year
- Idempotent — deletes and rewrites current-year rows on each run

## Throughput pressure monitor (native DAX, no nightly script)

Used to be `Throughput_Pressure_Monitor.py` (`gjennomstoremming_press_enhet`,
`gjennomstroemming_press_fase`), detecting where intake exceeds completions for multiple
periods and where processing time deteriorates versus a recent baseline, at the team
(`enhet`) level. Removed: `mottatt`/`ferdigstilt` counts, baselines, and the composite
pressure score are all live-computable against the fact table — see
`Throughput_Pressure_Monitor_POWERBI_DAX.md` (same filename, now documents pure DAX
measures instead of a script's output tables). That doc also flags the two rough edges
worth knowing about before relying on this: the `netto_flyt_streak` measure needs an
iterative window-scan pattern DAX has no native primitive for, and the model needs a
second (inactive) date relationship on `startmilepaeldato` to separate "received" from
"completed" counts by month.

- **Måltall/output shape:** same as before — team (`enhet`) × indikator × month, plus a
  fase-level companion for drill-down
- `pressure_nivaa` still has the values `Lav`, `Moderat`, `Hoy`, `Kritisk`, from the same
  composite scoring rule (net flow, flow streak, tidsbruk deviation vs. baseline, intake
  ratio) — just evaluated as a measure instead of written to a table

## Phase bottleneck detector (native DAX, no nightly script)

Used to be `Phase_Bottleneck_Detector.py` (`fase_flaskehals_enhet`), one level down from
the throughput pressure monitor: detects queue pressure forming inside individual process
phases (fase-nettoflyt + p90 fasetid deviation vs. baseline), same composite-scoring
pattern. Removed for the same reason — see `Phase_Bottleneck_Detector_POWERBI_DAX.md`
(same filename, now pure DAX). That doc flags its own two rough edges: a third date role
is needed on the model (`COALESCE(sluttmilepaeldato, startmilepaeldato)`, since this table
groups each row by its own single representative date rather than two independent date
counts), and the reason-code string (`arsak_kode`) is more naturally Python than DAX —
worth dropping in favor of separate boolean flag measures if maintainability matters more
than exact parity.

- `alvorlighet` still has the values `Lav`, `Moderat`, `Hoy`, `Kritisk`; `arsak_kode`/`arsak_tekst` still explain the flag, computed live

## Inflight_SLA_Risk_Monitor.py

Leading indicator of SLA-breach risk. `Fristprosent` (CUSUM) only scores cases that have already closed; this scores cases that are **still open** against their own `frist_dager`, so a breach wave shows up here before it reaches the closed-case ratio. **Minimal script — trend snapshot only.** `risikoklasse` depends on `TODAY()`, so today's per-case risk list is computed live in DAX straight from `saksbehandling.faser` (no nightly wait, no output table — see `Inflight_SLA_Risk_Monitor_POWERBI_DAX.md`, Del 1). This script's only remaining job is writing the daily risk-mix down for the trend chart Del 1 structurally can't produce — one `INSERT INTO ... SELECT`, pure Spark SQL, no pandas. `classify_risk()` stays in the script as the tested spec that the SQL's `CASE` expression (`risikoklasse_case_sql()`) is generated from, so the two can't drift apart.

- **Output:** `sak_frist_risiko_trend` (indikator × enhet × snapshot_dato, append-mode idempotent per day)
- `risikoklasse` har verdiene `Bruddet`, `Kritisk`, `Risiko`, `Innenfor`
- **Key constants:** `RISK_THRESHOLD_KRITISK` (0.90), `RISK_THRESHOLD_RISIKO` (0.75), `MIN_TEAM_VOLUME`
- Open assumption to verify against the Lakehouse schema: whether `frist_dager` is reliably populated on rows that haven't closed yet — rows missing it are excluded, not defaulted.

## Backlog_Aging_Distribution.py

Tracks whether the *existing* open-case backlog is aging, independent of the throughput pressure monitor's net-flow score (which measures flow imbalance, not the age of work already in the queue). **Minimal script — trend snapshot only.** `Aldersgruppe` depends on `TODAY()`, so today's backlog shape is computed live in DAX straight from `saksbehandling.faser` (no nightly wait, no output table — see `Backlog_Aging_Distribution_POWERBI_DAX.md`, Del 1). This script's only remaining job is writing the daily age-bucket shape down for the trend chart Del 1 structurally can't produce — one `INSERT INTO ... SELECT`, pure Spark SQL (`percentile_approx` + a `CASE` expression), no pandas. `bucket_age()` stays in the script as the tested spec that the SQL's `CASE` expression (`aldersgruppe_case_sql()`) is generated from, so the two can't drift apart.

- **Output:** `sak_alder_fordeling` (indikator × enhet × aldersgruppe × snapshot_dato, append-mode idempotent per day)
- `aldersgruppe` buckets (default `AGE_BUCKETS`): `0-30`, `31-60`, `61-90`, `91-180`, `180+`

## Caseworker_Load_Concentration.py

Team-level workload concentration / bus-factor / burnout early warning: is active caseload piling up on a few caseworkers within a team, even while the team's aggregate numbers look fine? Concentration is measured with the Gini coefficient of open-caseload counts per saksbehandler within each enhet. **Minimal script — Gini trend only.** Per-person open caseload counts and shares are plain live DAX against `saksbehandling.faser` (`Faser[saksbehandler]` is already in the semantic model — see `Caseworker_Load_Concentration_POWERBI_DAX.md`, Del 1); this script's only remaining job is the Gini coefficient, which genuinely can't be a DAX measure (rank-based Lorenz-curve math), and which — being a snapshot of today's open caseload — is also a trend question a live measure can't answer on its own, same reasoning as `Backlog_Aging_Distribution.py`/`Inflight_SLA_Risk_Monitor.py`.

- Individual-level automated flagging is still out of scope for this layer (the same reason `CUSUM_Changepoint.py`'s drilldown explicitly excludes saksbehandler) — `saksbehandler_konsentrasjon` only ever stores enhet-level aggregates, never a per-person breakdown.
- **Output:** `saksbehandler_konsentrasjon` (enhet × snapshot_dato, append-mode idempotent per day — Gini trend, **no individual data**)
- **Key constants:** `MIN_SAKSBEHANDLERE` (3) — Gini on 1-2 people is meaningless, gates `tilstrekkelig_volum`
- `SAKSBEHANDLER_COL` is unverified against the Lakehouse schema — verify before relying on this script.

## Configuration

All scripts share `START_YEAR = 2015` at the top. Adjust to match the earliest reliable data in your Lakehouse.
