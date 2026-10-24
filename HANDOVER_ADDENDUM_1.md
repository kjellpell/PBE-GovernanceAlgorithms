# Handover addendum 1 (revised) — on-hold time, lean signal set, what's left

**Date:** 2026-10-24 · **Supersedes:** the first version of this addendum
**Start from branch:** `claude/lean-early-warning`. It sits on top of
`claude/eager-euler-pdzu9o`, which completed every task in the original `HANDOVER.md`
(P0-1 … P2-1, 85 tests).

Read §0, then work through §3 in order. The house rules from the original handover still
apply:
- Fabric notebooks; no local Spark; no `%pip`;
- Norwegian names;
- logic in top-level pure functions, tested through `ast` extraction;
- one commit per task;
- update the DAX docs and README with each change.

---

## 0. Why this addendum exists
Many cases sit **on hold while the applicant completes documentation**, or wait on external
parties. That time is outside the department's control and normally stops the deadline clock.
The original handover treated every open phase as "our work". This addendum:
- separates **with us** from **waiting**;
- plans a real deadline clock;
- trims the signal set to what earns its place as an early warning.

## 1. Facts confirmed by the human
1. `Faser[Status] = "Under behandling"` = the phase is **with us**. Other statuses on an open
   phase = waiting. (The full list of values is not yet known; ask if needed.)
2. There is **no status history or log table.**
3. Every case has milestones for **"asked for completion" (request)** and **"received
   completion" (receipt)**. Table and type names: **ask; don't guess.**
4. Already decided (in the first round): `frist_dager` is **calendar days**. Official
   Fristprosent = produced phases, with `%avtalt%` included. Severity value is `Høy`.
   No forecast-history table.

## 2. Already done on `claude/lean-early-warning` (commit `a4346a9`) — don't redo
- **Gini page retired.** `Caseworker_Load_Concentration_POWERBI_DAX.md` and
  `tests/test_gini_reference.py` were deleted; the README notes the retirement. Reasons:
  it is hard to read, it is individual-level monitoring, and it is not an early warning.
- **PELT on demand.** `CUSUM_Changepoint.py` has `KJOER_PELT = False`. CUSUM runs nightly;
  PELT and the drill-down run only when someone sets the flag to `True` and runs the
  notebook by hand. With the flag off, the `pelt_*` tables keep the last manual run. A test
  pins the default. The DAX doc tells readers to show `kjoert_tidspunkt` on PELT pages.
- **At-risk page scoped to cases with us.** `At_Risk_Open_Cases_POWERBI_DAX.md` now filters on
  `Status = "Under behandling"`. The measures were renamed: `Hos oss`,
  `Hos oss i faresonen`, `Hos oss over frist`, `Andel hos oss i faresonen`. New measures:
  - `Venter (ikke hos oss)` and `Andel som venter`;
  - `Ferdigstilt siste 28 dager`;
  - `Implisitt behandlingstid (dager)` (Little's law) vs. `Typisk frist (dager)`.

  Age is still **calendar** days (an interim limitation, documented); P2-1b fixes it.
- **`Throughput_Pressure_Monitor` marked for retirement** after P2-0 (README). It has not
  been removed yet, because it is the only team-level flow history until P2-0 exists.

---

## 3. Remaining tasks, in order

### P2-0 `Pause_Reconstruction.py` — deadline clock and weekly flow history
A genuine interval algorithm, so a script under the README rule. Nightly, overwrite.

1. **Load** request and receipt milestone dates per case.
2. **Pair them into pause intervals.** Sort events per case; a request opens a pause, the next
   receipt closes it. Count each data-quality case and print it in the run log:
   - repeated requests before a receipt → keep the first open request;
   - a receipt without an open request → ignore it;
   - a request still open at case closure → the pause ends at closure; flag
     `avsluttet_uten_komplettering` (probably withdrawn or dismissed) and exclude it from
     production counts;
   - a request still open today → the case is on hold now.
3. **Map pauses to phases.** Milestones are per case; `frist_dager` is per phase. Apply each
   pause to the phase(s) whose [start, end) overlaps it, clipped to the overlap.
4. **Per phase, compute:**
   - `pause_dager`;
   - `klokke_dager` = calendar days − pause days (up to closure, or up to today if open);
   - `antall_kompletteringsrunder`;
   - `dager_til_foerste_anmodning` (start → first request).
5. **Validate first.** Make the clock rule a constant, `KLOKKE_REGEL = "pause" | "restart"`:
   - `pause`: pause days are subtracted;
   - `restart`: age counts from the last receipt.

   For closed phases, recompute `klokke_dager <= frist_dager` and compare with
   `innenfor_frist`; print the agreement rate per indikator. Use the rule that reproduces
   it. **If neither rule passes ~95%, stop and ask.** The same check settles the boundary
   question Claude Code raised: whether a phase is on time when day = `frist_dager`.
6. **Cross-check.** For open phases, today's reconstructed state (with us / waiting) should
   match `Status = "Under behandling"`. Print the agreement rate.
7. **Output tables:**
   - `analyser.fase_klokke`: one row per phase, with the step-4 fields plus
     `tilstand` ∈ {`Hos oss`, `Venter på søker`, `Avsluttet`} and
     `avsluttet_uten_komplettering`.
   - `analyser.flyt_ukentlig`: one row per ISO week (use the existing `iso_week_key`
     approach) × enhet × indikator, with:
     - `hos_oss`: started, not closed, not inside a pause at week end;
     - `venter_paa_soeker`;
     - `i_faresonen`: with us and `klokke_dager >= 0.75 * frist_dager` at week end;
     - `mottatt`, `ferdigstilt` (excluding `avsluttet_uten_komplettering`),
       `satt_paa_vent` (requests sent), `komplettering_mottatt`.

     Complete weeks only.

**Tests** (synthetic cases):
- no pause;
- one pause;
- two rounds;
- a pause spanning two phases;
- a request still open today;
- a request still open at closure;
- a receipt without a request;
- both `KLOKKE_REGEL` values;
- the weekly state counts on a small hand-checked example.

Add a DAX doc `Pause_Reconstruction_POWERBI_DAX.md`:
- weekly trend of `hos_oss`, `venter_paa_soeker`, `i_faresonen`;
- implied processing time per week;
- median `dager_til_foerste_anmodning` per enhet (shows whether the clock is being stopped
  late).

### P2-1b Switch the at-risk page to the deadline clock
In `At_Risk_Open_Cases_POWERBI_DAX.md`, replace `DATEDIFF(Faser[startmilepaeldato], Idag, DAY)`
with `fase_klokke[klokke_dager]` (relationship on the phase key). Exclude
`avsluttet_uten_komplettering` from `Ferdigstilt siste 28 dager`. Remove the
interim-limitation paragraphs from the doc and the README.

### P2-2 Backtest the warning signals
Using `flyt_ukentlig` (and the existing monthly frist% query), replay 2017 → today. For each
candidate signal, record the weeks of warning before each historical dip in monthly frist%
and the number of false alarms. Candidates:
- (a) implied processing time vs. deadline;
- (b) `i_faresonen` share;
- (c) waiting share;
- (d) fixed CUSUM on frist% (`tilstand`), monthly and weekly;
- (e) a p-chart on monthly frist%: centre line = same-month mean over the prior 3 years,
  limits = ±3·√(p(1−p)/n).

Define a "dip" up front (e.g. monthly frist% ≥ 5 points below the same month's 3-year mean)
and state it in the report. Do the backtest as an on-demand script
`Backtest_Early_Warning.py`, with its pure functions tested. Write the results to
`BACKTEST_RESULTS.md`, not to a table. Recommend which signals go on the leader page.

### P2-3 Retire `Throughput_Pressure_Monitor` (after P2-0 is validated)
Delete the script, its DAX doc and its test file; update the README "retired" list. In the
final summary, list the Fabric/Power BI steps:
- remove the nightly notebook from the schedule;
- remove the visuals and relationships on `gjennomstroemming_press_enhet` and
  `gjennomstroemming_press_fase`;
- then `DROP TABLE` both.

### P2-4 Phase bottleneck: don't blame us for waiting time
Add a `fasetittel → ansvar` ∈ {`Intern`, `Søker`, `Ekstern`} mapping. **Ask the human for the
classification**; store it as a small table. In `Phase_Bottleneck_Detector.py`, never set
`bottleneck_flagg` for `Søker`/`Ekstern` phases; give them `arsak_tekst` "Venter på søker /
ekstern part — ikke intern flaskehals." Add a test.

### P2-5 Decision-gated (ask first; don't start without a yes)
- **CUSUM's fate.** If P2-2 shows CUSUM gives no earlier warning than (a)–(c) or the
  p-chart, replace it with the p-chart as live DAX and retire `CUSUM_Changepoint.py`'s
  nightly CUSUM part. PELT stays on demand either way.
- **Organisational-change evaluation.** A simple change log (date, enhet, description), kept
  as a CSV or SharePoint list that the human maintains, plus an on-demand
  `Intervention_Analysis.py`. That script runs an interrupted time series (segmented
  regression: level and slope change at the known date, seasonal terms) per logged change.
  Where some teams weren't changed, it adds a difference-in-differences against them. Pure
  NumPy least squares; no statsmodels. The log entries also go on trend charts as markers.
- **`Behandlingstid` in CUSUM.** If `tidsbruk` includes on-hold time (check against
  `fase_klokke`), the CUSUM `Behandlingstid` series mixes applicant behaviour with ours.
  Either switch it to clock days or drop it.

## 4. Questions to ask the human (in one go, when you reach P2-0)
1. Milestone table, and the exact type names for request and receipt.
2. How withdrawn/dismissed cases are coded.
3. The full list of `Faser[Status]` values.
4. The `fasetittel → ansvar` classification (needed for P2-4; ask early so it's ready).

## 5. Out of scope
- `Kostra.py`.
- `Seasonal_YTD_ratio_extrapolation.py` (keep as is; it is the year-end target number).
- Reintroducing any individual-level measure.
- New libraries.

## 6. Done means
- Tests are green.
- Docs and README match the code.
- `BACKTEST_RESULTS.md` exists with a recommendation.
- The final summary lists per task: what changed, the Fabric/Power BI steps for the human,
  the validation rates from P2-0 steps 5–6, and which decisions were assumed.
