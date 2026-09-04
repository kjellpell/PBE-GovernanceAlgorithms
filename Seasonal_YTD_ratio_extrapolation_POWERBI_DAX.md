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
written differently — worth knowing, because it means the script's `innenfor`/`total`
counts (`innenfor_frist = 1` over `frist_dager IS NOT NULL`) are the same numerator and
denominator the report uses. If those ever drift apart, the projection stops matching the
line it is drawn against.

The forecast therefore has to be a **period rate too**. This is the one thing that has to
match: a projection of a different quantity cannot continue this line, however it is
drawn.

### Visual — faktisk linje
- X-akse: `Kalender[Dato]`, month grain (the rate for a single day is mostly sampling
  noise — a handful of faser)
- Hel linje: `[Faser innen frist %]`

The current month is worth cutting off. Its rate is computed from however many faser have
closed so far, so early in the month it is a near-meaningless number that moves every day
and drops off the bottom of the chart — the projection covers that month properly. Either
filter the actual line to complete months, or read the last point as provisional:

```DAX
Faser innen frist % (komplette måneder) =
IF(
    MAX(Kalender[Dato]) < STARTOFMONTH(TODAY()),
    [Faser innen frist %]
)
```

## Del 2 — Prognose (persistert modell)

Datakilde: `analyser.frist_prognose` (written nightly by
`Seasonal_YTD_ratio_extrapolation.py`).

| `type` | rows | `verdi` | bånd |
|---|---|---|---|
| `Anker` | one, at the end of the last complete month | that month's observed rate | zero width — it's an observation |
| `Prognose` | every day from the next month to 31 Dec | that month's projected rate | that month's historical spread |

`verdi` is a **month rate**, in the same units as `[Faser innen frist %]`, so the two go on
one axis and mean the same thing. Every day inside a month carries that month's projected
rate — a monthly rate is a step, not a curve — and the rows are daily only so the series
lands on the axis whatever grain the report rolls it to.

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
  direction. Both are daily dates, so this is a plain date relationship.
- **X-akse:** `Kalender[Dato]` at month grain, jan–des.
- **Serie 1 (hel linje):** `[Faser innen frist %]` — actual, through the last complete
  month.
- **Serie 2 (stiplet linje):** `[Prognose rate]` — from that same month to December. The
  `Anker` row carries the last complete month's observed rate, so the dashed line starts
  *on* the solid one and the fork marks exactly where projection begins.
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
Prognose rate = AVERAGE(frist_prognose[verdi])
```

```DAX
Prognose nedre = AVERAGE(frist_prognose[nedre_konfidensgrense])
```

```DAX
Prognose oevre = AVERAGE(frist_prognose[oevre_konfidensgrense])
```

`AVERAGE`, never `SUM`. Every day of a month repeats that month's rate, so averaging over
a month returns the rate itself, and over a quarter returns a day-weighted average of its
three months — both sensible. Summing would add a rate to itself thirty times.

Do not filter `frist_prognose[type]` on this visual. The `Anker` row is what joins the
dashed line to the solid one; excluding it puts the gap back.

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
- Prognosert månedsrate under mål = risiko i den måneden.
- `[Prognose årslutt]` under mål = risiko for året. The two can disagree: a weak December
  is normal for these indicators and doesn't by itself put the year below target.
- Et bredt bånd er ikke en feil. It is how much that month has actually varied year to
  year, and for a single month that is usually several points.
- The dashed line must *start* on the solid one. The `Anker` row is the last complete
  month's observed rate, computed from the same numerator and denominator as the report's
  measure, so a visible step at the fork means the nightly run is stale or the two sides'
  filters (fagomraade, `frist_dager`) have drifted apart. That check used to need a query;
  now it is just looking at the chart.
- Ignore the actual line's current-month point, or filter it out. It is a part-month rate
  computed from however many faser have closed so far, and it moves every day; the
  projection covers that month as a whole.
- The year-end estimate does use the part-month data — it is a cumulative quantity, where
  a few extra days barely move the level, and the model accounts for how far into the
  month the data reaches rather than treating a part-finished month as finished.

## Åpenbar begrensning
The nightly run deletes and rewrites the whole current year, so the table holds only the
newest forecast. Forecast drift over time — "in June we projected 88%, now we project
84%" — is not recoverable from it. That would be the genuinely interesting line chart
here, and it needs the run to keep its history first.
