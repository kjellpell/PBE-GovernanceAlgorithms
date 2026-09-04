# Seasonal_YTD_ratio_extrapolation.py — Power BI and DAX

## Formål
Continue the `Faser innen frist %` line from the last complete month to year end, with an
uncertainty band, and give the year-end figure as a KPI.

The page is split in two on purpose. **What has happened** is a live measure against the
fact table — no script can improve on it. **What is projected** comes from
`analyser.frist_prognose`, the one part that genuinely can't be a DAX measure: the
seasonal-ratio model (trimmed mean/std across historical years, delta-method interval) is
a statistical model, not a lookup or a rollup.

## Del 1 — Faktisk (live DAX)

The measures the report already uses:

```DAX
Faser innen frist =
VAR _innenfrist =
    CALCULATE(
        [Produserte faser],
        Faser[Innenfor frist]
    )
RETURN
    IF( ISBLANK(_innenfrist), 0, _innenfrist)
```

```DAX
Faser innen frist % =
    DIVIDE([Faser innen frist], [Produserte faser])
```

**`Faser innen frist %` is a period rate, not a year-to-date value**, and everything on
this page follows from that. There is no `DATESYTD` and nothing else cumulative over it,
so in a monthly axis each point is that month standing alone: produced faser within the
deadline over produced faser, that month. That is why the line swings — a weak month shows
up in full, not diluted by the months before it. A cumulative YTD line barely moves by
autumn and can't drop several points in one step.

This is the same quantity as `Fristprosent (måned)` in `Trendretning_POWERBI_DAX.md`,
written differently — they give the same headline numbers, but do not assume from that
alone that their filters are identical underneath. They aren't automatically: `[Produserte
faser]` is its own measure,

```DAX
Produserte faser =
    CALCULATE(
        COUNTROWS('Faser'),
        Faser[Sluttmilepældato] <> BLANK(),
        Faser[Startmilepældato] <> BLANK()
    )
```

— a fase counts once it has **both a start and an end milestone date**. Nothing about
`frist_dager`. An earlier version of the script's SQL used `frist_dager IS NOT NULL` as the
denominator instead, copied over from `Fristprosent (måned)`'s inline definition on the
assumption that matching headline numbers meant matching filters. It doesn't: a fase type
with no tracked deadline is still produced, and silently dropping it from the denominator
(and, since it can't be "innenfor" a deadline it doesn't have, implicitly from the
numerator's population too) shifts the rate — enough, for at least one indicator, to
visibly detach the forecast's anchor from the actual line. The script's `innenfor`/`total`
counts now match `[Faser innen frist]`/`[Produserte faser]` by construction:
`innenfor_frist = 1` over `sluttmilepaeldato IS NOT NULL AND startmilepaeldato IS NOT NULL`.

The forecast therefore has to be a **period rate too**. This is the one thing that has to
match: a projection of a different quantity cannot continue this line, however it is
drawn.

### Visual — faktisk linje
- X-akse: `Kalender[Dato]`, month grain (the rate for a single day is mostly sampling
  noise — a handful of faser)
- Hel linje: `[Faser innen frist %]`

### Den pågående måneden

The current month's point is not wrong, it is not comparable. On 3 September it is the
rate of the faser that closed on the 1st, 2nd and 3rd — a handful, against ~120 in every
other point on the line. Undifferentiated, three days of sampling noise reads as a
performance collapse.

Don't drop it; draw it as what it is. Two measures, overlapping by one month so the
provisional segment connects to the solid line:

```DAX
Faser innen frist % (lukket) =
IF( MAX(Kalender[Dato]) < EOMONTH(TODAY(), -1) + 1, [Faser innen frist %] )
```

```DAX
Faser innen frist % (pågående) =
IF( MAX(Kalender[Dato]) >= EOMONTH(TODAY(), -2) + 1, [Faser innen frist %] )
```

Plot both — `lukket` solid, `pågående` dotted or hollow-markered. A `Produserte faser`
column behind the line makes it self-evident without anyone needing to know the rule: a
stub bar under the thin point says "three days".

Cutting the line at the last complete month instead is defensible, and rules out any
misreading, but by the 28th it means looking at month-old data.

**A month here is complete on its last day.** There is no registration lag to allow for —
a fase closing on 31 August is in August's number, not still arriving in September — so
"last complete month" is the plain calendar month. That is what the nightly run anchors
on, and why it can anchor on the immediately preceding month rather than backing off
further.

## Del 2 — Prognose (persistert modell)

Datakilde: `analyser.frist_prognose` (written nightly by
`Seasonal_YTD_ratio_extrapolation.py`).

| `type` | rows | `verdi` | `innenfor_prognose` / `produserte_prognose` | bånd |
|---|---|---|---|---|
| `Anker` | one, at the end of the last complete month | that month's observed rate | the real counts for that month | zero width — it's an observation |
| `Prognose` | one per remaining month, dated at each month's end | that month's projected rate | that month's modelled counts | that month's historical spread |

One row per remaining month, not per day: the report's axis groups by month
(`Kalender[Dato]` at month grain), so a finer grain in the table bought nothing — every
day within a month would carry an identical value anyway.

`verdi` is a **month rate**, in the same units as `[Faser innen frist %]`, so the two go on
one axis and mean the same thing. But **do not build the primary measure on `verdi`
directly** — see the DAX section below. `Faser innen frist %` is
`DIVIDE(SUM(innenfor), SUM(total))`; averaging `verdi` across rows is different arithmetic
and only happens to agree with that in the single-month, single-indicator case. Everywhere
else — several indicators together, a rollup wider than one month — it gives a different
number, and that mismatch is exactly what `innenfor_prognose`/`produserte_prognose` exist
to prevent: sum those two and divide, the same pattern the report already uses, and the
row-level `verdi` becomes redundant except as a quick single-row sanity check.

`prognose_aarsslutt` is the year-end figure, repeated on every row so a card reads it
without a date filter. That one *is* cumulative (volume-weighted across the whole year),
because that is the governance number: it is not the last point of the line, and it should
not be plotted as one.

How the two relate: the model estimates year-end from cumulative YTD, because YTD is the
stable thing to extrapolate from, then converts that estimate back into per-month rates
using each month's historical position relative to its year. So the line carries the
seasonal shape — a historically weak November projects below a strong September — rather
than drifting towards a single number.

### Visualforslag

#### 1) Linjediagram: faktisk + prognose

- **Modell:** relate `Kalender[Dato]` (1) → `frist_prognose[analyse_dato]` (*), single
  direction — `analyse_dato` is month-end, so this is a plain date relationship at month
  grain.
- **X-akse:** `Kalender[Dato]` at month grain, jan–des.
- **Serie 1 (hel linje):** `[Faser innen frist %]` — actual, through the last complete
  month.
- **Serie 2 (stiplet linje):** `[Prognose rate]` — from that same month to December. The
  `Anker` row carries the last complete month's *real* counts, so this series reproduces
  the actual measure exactly at that one point, and the dashed line starts *on* the solid
  one — the fork marks exactly where projection begins.
- **Bånd:** `[Prognose nedre]` and `[Prognose oevre]`. Power BI's core line chart has no
  band primitive — either two thin dashed lines, or a band-capable visual underneath.
- **Mållinje:** constant line at `alert_config[terskel_amber]`.

Expect the band to be wide. A single month's rate has swung several points year to year in
this data, and that spread is the useful part of the picture — it is the difference
between "October will be 78%" and "October has been anywhere from 74% to 82%". A narrow
band on a monthly rate would mean the model is hiding variation it has actually seen.

#### 2) KPI-kort med intervall
- Stor verdi: `[Prognose årslutt]` — cumulative year-end frist%, not a month
- Undertekst: `[Prognose intervalltekst]` — "90 % KI: 79–89 %"
- Farge: `[Prognose RAG]` mot `alert_config`

#### 3) Oppsummeringstabell
- `indikator`, `[Prognose årslutt]`, `[Prognose CI lower]`, `[Prognose CI upper]`
- Sortering: `[Prognose årslutt]` asc

### DAX-forslag

```DAX
Prognose rate =
DIVIDE(
    SUM(frist_prognose[innenfor_prognose]),
    SUM(frist_prognose[produserte_prognose])
)
```

The same shape as `Faser innen frist %` — `DIVIDE(SUM(numerator), SUM(denominator))` —
because the table now carries a numerator and denominator, not just a pre-divided rate.
That is what makes this measure agree with the report's own arithmetic at every grain, not
only the single-month reading: two indicators together, several months rolled into one
quarter, whatever the page is sliced to. An `AVERAGE(frist_prognose[verdi])` measure
happens to return the same number in the single-month, single-indicator case — there is
only one row there to average — but is different arithmetic underneath, and diverges
everywhere else. That mismatch, not a filter or a grain problem, is why an
`AVERAGE`-based version of this measure can show a different number than
`[Faser innen frist % (lukket)]` even when both look at what should be the same data.

At the `Anker` row this measure reproduces `[Faser innen frist %]` exactly for that month,
because the anchor's counts are the real ones, not modelled — the two are the same
arithmetic over the same numbers.

Do not filter `frist_prognose[type]` on this visual. The `Anker` row is what joins the
dashed line to the solid one; excluding it puts the gap back.

```DAX
Prognose nedre = AVERAGE(frist_prognose[nedre_konfidensgrense])
```

```DAX
Prognose oevre = AVERAGE(frist_prognose[oevre_konfidensgrense])
```

The band *is* `AVERAGE`, unlike the rate measure above — and that is fine, not an
inconsistency. A confidence interval isn't a count; there's no sum-and-divide identity for
it to preserve. Averaging is already only a rough combination of independent bounds across
several rows (statistically, bounds don't sum), so this band is exact at single-month
grain and an approximation of the band's centre at any wider rollup — read it that way
rather than as a rigorously combined interval, at any grain wider than one month.

```DAX
Prognose årslutt =
CALCULATE(
    MAX(frist_prognose[prognose_aarsslutt]),
    REMOVEFILTERS(Kalender),
    REMOVEFILTERS(frist_prognose[analyse_dato])
)
```

```DAX
Prognose CI lower =
CALCULATE(
    MAX(frist_prognose[nedre_aarsslutt]),
    REMOVEFILTERS(Kalender),
    REMOVEFILTERS(frist_prognose[analyse_dato])
)
```

```DAX
Prognose CI upper =
CALCULATE(
    MAX(frist_prognose[oevre_aarsslutt]),
    REMOVEFILTERS(Kalender),
    REMOVEFILTERS(frist_prognose[analyse_dato])
)
```

These three read the year-end columns, which carry the same value on every row, so the
card shows the same number whatever month the page is sliced to — that is the point of a
year-end KPI. They remove the *date* filters only: `REMOVEFILTERS(frist_prognose)` would
strip `indikator` too and give every indicator the same number.

Note they do **not** read `nedre_konfidensgrense`/`oevre_konfidensgrense`. Those are a
month's band; the year-end figure is a cumulative, volume-weighted quantity with its own,
narrower interval. Mixing them is the mistake this page exists to prevent.

```DAX
Prognose intervalltekst =
VAR Nedre = [Prognose CI lower]
VAR Oevre = [Prognose CI upper]
RETURN
    IF(
        ISBLANK(Nedre) || ISBLANK(Oevre),
        BLANK(),
        "90 % KI: " & FORMAT(Nedre, "0 %") & "–" & FORMAT(Oevre, "0 %")
    )
```

```DAX
Prognose RAG =
VAR Prognose = [Prognose årslutt]
VAR Mål =
    CALCULATE(
        MIN(alert_config[terskel_amber]),
        alert_config[indikator] = MAX(frist_prognose[indikator]),
        alert_config[aktiv] = TRUE()
    )
RETURN
    IF(
        ISBLANK(Prognose) || ISBLANK(Mål),
        BLANK(),
        IF(Prognose >= Mål, 3, IF(Prognose >= Mål * 0.95, 2, 1))
    )
```

## Slicer-oppsett
- `indikator`
- `Kalender[Dato]` (Del 1 only — Del 2 covers the current year by construction, and the
  card measures above deliberately ignore the date filter)

## Tolkning
- `[Prognose rate]` on this page reads `innenfor_prognose`/`produserte_prognose`, never
  `verdi` directly, for any visual that might show more than one row at once (which is
  almost all of them — a line chart, a table, a card sliced to a quarter). `verdi` is
  there for a quick look at one row, not for a measure to aggregate.
- Prognosert månedsrate under mål = risiko i den måneden.
- `[Prognose årslutt]` under mål = risiko for året. The two can disagree: a weak December
  is normal for these indicators and doesn't by itself put the year below target.
- Et bredt bånd er ikke en feil. It is how much that month has actually varied year to
  year, and for a single month that is usually several points.
- The dashed line must *start* on the solid one. The `Anker` row is the last complete
  month's observed rate, computed from the same numerator and denominator as the report's
  measure — `innenfor_frist = 1` over `sluttmilepaeldato`/`startmilepaeldato` both present,
  the same population `[Produserte faser]` counts — so a visible step at the fork means the
  nightly run is stale, or the two sides' fagomraade filter has drifted, or the script's
  SQL has stopped matching a measure definition again. That check used to need a query;
  now it is just looking at the chart.
- The actual line's current-month point is provisional — a part-month rate that moves
  every day (see Del 1). The projection covers that month as a whole, so the two will
  differ, and the gap closes as the month fills in rather than indicating an error.
- The year-end estimate does use the part-month data — it is a cumulative quantity, where
  a few extra days barely move the level, and the model accounts for how far into the
  month the data reaches rather than treating a part-finished month as finished.

## Åpenbar begrensning
The nightly run deletes and rewrites the whole current year, so the table holds only the
newest forecast. Forecast drift over time — "in June we projected 88%, now we project
84%" — is not recoverable from it. That would be the genuinely interesting line chart
here, and it needs the run to keep its history first.
