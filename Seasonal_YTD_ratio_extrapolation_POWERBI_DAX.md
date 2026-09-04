# Seasonal_YTD_ratio_extrapolation.py — Power BI and DAX

## Formål
Vise årssluttprognose for fristprosent med usikkerhetsintervall basert på sesongmønster.

This page is split in two, deliberately: **actual YTD so far** is standard DAX
time-intelligence against the fact table (no script needed — it's the "YTD in scripts
does not make sense" case), and **the year-end forecast with its confidence interval**
comes from `prognoser.frist_prognose`, the one thing that genuinely can't be a DAX measure
— the seasonal-ratio model (trimmed mean/std across historical years, delta-method CI) is
a statistical model, not a lookup or a rollup.

## Del 1 — Faktisk YTD (live DAX)

Reuses the `Fristprosent (måned)` base measure from `Trendretning_POWERBI_DAX.md`
(same `Faser`/`Kalender` assumptions) — YTD is just that same measure evaluated over a
wider, year-to-date filter context instead of one month, which is standard DAX time
intelligence, not an algorithm:

```DAX
Fristprosent YTD =
CALCULATE(
    [Fristprosent (måned)],
    DATESYTD(Kalender[Dato])
)
```

This gives the volume-weighted YTD ratio (cumulative innenfor / cumulative total across
the months so far this year) — the same definition the old script used, not an average of
monthly ratios.

### Visual — faktisk linje
- X-akse: `Kalender[År måned]` (jan–des inneværende år) — månedsgrain, ikke dagsgrain
- Hel linje: `[Fristprosent YTD]`

Note the axis grain. A YTD measure on a *daily* axis draws a sawtooth; on a monthly axis
it draws the cumulative curve everyone expects. Both parts of this page use the same
monthly axis, which is what lets them share one chart in Del 2.

## Del 2 — Årssluttprognose (persistert modell)

Datakilde: `prognoser.frist_prognose` (written nightly by
`Seasonal_YTD_ratio_extrapolation.py`).

The table holds the forecast **as a YTD trajectory**, one row per month:

| `type` | months | `verdi` | band |
|---|---|---|---|
| `Anker` | the last closed month | observed YTD | zero width — it's an observation |
| `Prognose` | remaining months | expected YTD at that month | widens with the horizon |

`prognose_aarsslutt` is the December endpoint, repeated on every row so the KPI cards can
read it without a date filter. `nedre_konfidensgrense`/`oevre_konfidensgrense` belong to
that row's own `verdi` — they are the 90% year-end interval scaled along the same seasonal
path, so they close on the year-end interval in December.

### Hva prognosen faktisk er — og hva den ikke tåler å bli tegnet som

The forecast is, at heart, **one number with an interval**: årsslutt ± CI. Everything else
is that number projected backwards along a seasonal path. That has two consequences for
the report layout:

1. The KPI card is the primary visual, not the line chart. A three-point line in the last
   quarter carries one number and spends a third of the chart doing it.
2. The line chart is worth building *only* if it shows the trajectory joined to the actual
   line and carries the uncertainty band. Without those, it adds nothing the card doesn't
   already say — and it invites the anti-pattern below.

#### Anti-mønster: prognose lagt oppå månedskurven

The one to avoid: `[Fristprosent (måned)]` (from `Trendretning_POWERBI_DAX.md`) as the
actual series, with the forecast as a short flat line hanging over okt–des. It looks
plausible and is wrong in three ways at once:

- **Two different quantities on one axis.** The monthly measure is each month on its own
  (it swings — 63% one month, 90% the next); the forecast is cumulative YTD. A viewer
  reads the flat forecast segment as "we expect 84% *in* October", which is not what the
  model says. It says: 84% *for the year as a whole*.
- **The gap.** The forecast series starts in the first unforecast month, so it floats
  disconnected from the actual line. Nothing tells the eye where the projection departs
  from observed data.
- **No band.** Drawn as a bare line, a 90% interval that may span several points reads as
  a point estimate.

If the monthly curve is the chart the board wants — a fair choice, it carries the real
variation — then keep it, and put the forecast on it as a **labelled constant reference
line** with `nedre`/`oevre` as a shaded region across the *whole* x-axis, titled
`Prognose årslutt 84 % (90 % KI: 79–89 %)`. A reference line makes no claim about
individual months; a series over okt–des does.

### Visualforslag

#### 1) KPI-kort med intervall (primærvisual)
- Stor verdi: `[Prognose årslutt]`
- Undertekst: `[Prognose intervalltekst]` — "90 % KI: 79–89 %"
- Farge: `[Prognose RAG]` mot `alert_config`
- Sammenlign mot målverdi (`alert_config[terskel_amber]`)

A bullet/KPI visual with the target as the reference marker and the CI as the range says
everything the model has to say, in the space of a tile.

#### 2) Linjediagram: faktisk YTD + prognosebane

Only worth the space if built as follows — otherwise use #1 alone.

- **Modell:** relate `Kalender[Dato]` (1) → `frist_prognose[analyse_dato]` (*), single
  direction. `analyse_dato` is month-end, so the two sides line up on a monthly axis.
- **X-akse:** `Kalender[År måned]` — the *same* axis as Del 1.
- **Serie 1 (hel linje):** `[Fristprosent YTD]` — actual, jan → last closed month.
- **Serie 2 (stiplet linje):** `[Prognose YTD]` — the trajectory. Because the table
  carries the `Anker` row at the last closed month, this series starts *on* the actual
  line: no gap, and the fork is visible at the exact month the projection takes over.
- **Bånd:** `[Prognose nedre]` and `[Prognose oevre]`. Power BI's core line chart has no
  band primitive — either add them as two thin dashed lines, or lay a
  `Line and stacked area` / band-capable visual underneath. The band is zero-width at the
  anchor and widest in December, which is the honest shape: uncertainty grows with the
  horizon.
- **Mållinje:** constant line at `alert_config[terskel_amber]`.

Both series are YTD, so the fork reads correctly: same quantity, one observed, one
projected. The December point of the dashed line is the number on the KPI card — if the
two ever disagree, the semantic model is filtering the two sides differently.

#### 3) Oppsummeringstabell
- `indikator`, `[Fristprosent YTD]` (Del 1), `[Prognose årslutt]`, `[Prognose nedre]`, `[Prognose oevre]`
- Sortering: `[Prognose årslutt]` asc

### DAX-forslag

```DAX
Prognose YTD = MAX(frist_prognose[verdi])
```

Do not filter `frist_prognose[type]` on this visual. The `Anker` row is what joins the
dashed line to the solid one; excluding it gives back the disconnected series from the
anti-pattern above.

```DAX
Prognose nedre = MAX(frist_prognose[nedre_konfidensgrense])
```

```DAX
Prognose oevre = MAX(frist_prognose[oevre_konfidensgrense])
```

The card measures below must ignore the page's month filter (they are year-end values, the
same no matter which month is in context) while keeping the `indikator` filter — so they
remove the *date* filters only, both the ones on `frist_prognose` itself and the ones
`Kalender` propagates through the relationship. `REMOVEFILTERS(frist_prognose)` would
strip `indikator` as well and give every indicator the same number.

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
VAR SisteDato =
    CALCULATE(
        MAX(frist_prognose[analyse_dato]),
        REMOVEFILTERS(Kalender),
        REMOVEFILTERS(frist_prognose[analyse_dato])
    )
RETURN
    CALCULATE(
        MAX(frist_prognose[nedre_konfidensgrense]),
        REMOVEFILTERS(Kalender),
        REMOVEFILTERS(frist_prognose[analyse_dato]),
        frist_prognose[analyse_dato] = SisteDato
    )
```

```DAX
Prognose CI upper =
VAR SisteDato =
    CALCULATE(
        MAX(frist_prognose[analyse_dato]),
        REMOVEFILTERS(Kalender),
        REMOVEFILTERS(frist_prognose[analyse_dato])
    )
RETURN
    CALCULATE(
        MAX(frist_prognose[oevre_konfidensgrense]),
        REMOVEFILTERS(Kalender),
        REMOVEFILTERS(frist_prognose[analyse_dato]),
        frist_prognose[analyse_dato] = SisteDato
    )
```

`SisteDato` is computed in its own unfiltered pass, so it is December — not whatever month
the page happens to be sliced to. December is where the band equals the year-end interval,
so these two are both the card's interval and the endpoints of the chart's band.

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
- `Kalender[Dato]` (Del 1 only — Del 2 is scoped to the current year already, and the card
  measures above deliberately ignore the month filter)

## Tolkning
- Prognose under mål = risiko.
- Konfidensbånd som krysser mållinjen = usikker måloppnåelse.
- Når flere måneder blir faktiske, skal båndene normalt bli smalere — the band is scaled
  from the year-end interval, and that interval narrows as more of the year is observed.
- The dashed line's December point equals `[Prognose årslutt]` on the card. They come from
  the same column, so a mismatch means the visual is filtering, not that the model
  disagrees with itself.
- Del 1's `[Fristprosent YTD]` at the latest closed month should match the `Anker` row's
  `verdi` within rounding — the anchor is that same observed YTD, written by the nightly
  run. This is the one check that catches a stale run or drifted filters (`%avtalt%`,
  fagomraade) between the two sides, and now it is visible on the chart itself: if the
  dashed line does not start exactly on the solid one, one of the two is wrong. There is
  still no single source of truth enforcing it automatically.

## Åpenbar begrensning
The nightly run deletes and rewrites the whole current year, so the table holds only the
newest forecast. Forecast drift over time — "in June we projected 88%, now we project
84%" — is not recoverable from it. That would be the genuinely interesting line chart
here, and it needs the run to keep its history first.
