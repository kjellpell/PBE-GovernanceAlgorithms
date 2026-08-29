# PBE-GovernanceAlgorithms

Statistical governance algorithms for indicator time series. Designed to run as nightly Fabric Notebooks (PySpark + pandas). All output tables are Delta Lake, all output values in Norwegian, DAX measure names in Norwegian.

## Scripts

| File | Output table(s) | Schedule |
|---|---|---|
| `CUSUM_Changepoint.py` | `cusum_analyse`, `pelt_analyse`, `pelt_analyse_detaljer` | Nightly after main data pipeline |
| `Seasonal_YTD_ratio_extrapolation.py` | `frist_prognose` | Nightly after main pipeline |
| `Phase_Bottleneck_Detector.py` | `fase_flaskehals_enhet` | Nightly after main data pipeline |
| `Kostra.py` | `kostra_*` (one Delta table per SSB KOSTRA series, `kostra_` prefix) | Independent — SSB API sync, not part of the governance-algorithm pipeline |
| `Inflight_SLA_Risk_Monitor.py` | `sak_frist_risiko`, `sak_frist_risiko_trend` | Nightly after main data pipeline |
| `Caseworker_Load_Concentration.py` | `saksbehandler_arbeidsmengde`, `saksbehandler_konsentrasjon` | Nightly after main data pipeline |

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

Projects year-end `frist%` from current YTD using trimmed seasonal ratios from historical years.

- **Minimum history:** 3 complete years per indicator
- **Confidence interval:** 80% (z=1.28), derived from ratio variance (delta method)
- `type = 'actual'` for past months, `type = 'forecast'` for remaining months
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

## Phase_Bottleneck_Detector.py

Same pattern as the throughput pressure monitor above, one level down: detects queue pressure forming inside individual process phases (fase-nettoflyt + p90 fasetid deviation vs. baseline). Self-contained — does not read the throughput pressure monitor's output, so it was unaffected by that script's removal.

- **Output:** `fase_flaskehals_enhet` (enhet × fasetittel × indikator × month)
- `alvorlighet` har verdiene `Lav`, `Moderat`, `Hoy`, `Kritisk`; `arsak_kode`/`arsak_tekst` explain the flag
- **Key constants:** `BASELINE_MONTHS`, `MIN_BASELINE_OBS`, `MIN_SEGMENT_OBS`

## Inflight_SLA_Risk_Monitor.py

Leading indicator of SLA-breach risk. `Fristprosent` (CUSUM) only scores cases that have already closed; this script scores cases that are **still open** against their own `frist_dager`, so a breach wave shows up here before it reaches the closed-case ratio.

- **Output:** `sak_frist_risiko` (one row per open case-phase `pk_faser`, full overwrite nightly — current-state detail), `sak_frist_risiko_trend` (indikator × enhet × snapshot_dato, append-mode idempotent per day — trend)
- `risikoklasse` har verdiene `Bruddet`, `Kritisk`, `Risiko`, `Innenfor`
- **Key constants:** `RISK_THRESHOLD_KRITISK` (0.90), `RISK_THRESHOLD_RISIKO` (0.75), `MIN_TEAM_VOLUME`
- Open assumption to verify against the Lakehouse schema: whether `frist_dager` is reliably populated on rows that haven't closed yet — rows missing it are excluded, not defaulted.

## Backlog aging distribution (native DAX, no nightly script)

Used to be `Backlog_Aging_Distribution.py` (`sak_alder_fordeling`), tracking whether the
*existing* open-case backlog is aging, independent of the throughput pressure monitor's
net-flow score (which measures flow imbalance, not the age of work already in the queue).
Removed: age bucketing and the median/p90 age are plain row-level arithmetic — see
`Backlog_Aging_Distribution_POWERBI_DAX.md` (same filename, now documents pure DAX
measures instead of a script's output table).

- **Trade-off to know before relying on this:** the old script's `snapshot_dato`-stamped
  table let you trend the backlog's age shape over time. A live DAX measure only ever
  knows what's open *today* — it cannot reproduce that historical trend without something
  still persisting a daily snapshot somewhere. The DAX doc covers "today's backlog shape"
  only; it's explicit about what was dropped.
- `aldersgruppe` buckets (unchanged): `0-30`, `31-60`, `61-90`, `91-180`, `180+`

## Caseworker_Load_Concentration.py

Team-level workload concentration / bus-factor / burnout early warning: is active caseload piling up on a few caseworkers within a team, even while the team's aggregate numbers look fine? Concentration is measured with the Gini coefficient of open-caseload counts per saksbehandler within each enhet.

- **Deliberate exception to repo convention:** this is the only script here that persists per-individual (per-saksbehandler) data. It exists for internal manager capacity-planning/workload-balancing use — **not** for automated escalation or individual performance flagging (the same reason `CUSUM_Changepoint.py`'s drilldown explicitly excludes saksbehandler). Do not repurpose `saksbehandler_arbeidsmengde` for automated per-person alerting.
- **Output:** `saksbehandler_arbeidsmengde` (enhet × saksbehandler, **full overwrite nightly, no history retained at the individual grain**), `saksbehandler_konsentrasjon` (enhet × snapshot_dato, append-mode idempotent per day — Gini trend, **no individual data**, only accumulates history at the enhet level)
- **Key constants:** `MIN_SAKSBEHANDLERE` (3) — Gini on 1-2 people is meaningless, gates `tilstrekkelig_volum`
- `SAKSBEHANDLER_COL` is unverified against the Lakehouse schema — verify before relying on this script.

## Configuration

All scripts share `START_YEAR = 2015` at the top. Adjust to match the earliest reliable data in your Lakehouse.
