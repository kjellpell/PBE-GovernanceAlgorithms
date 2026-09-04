# Seasonal_YTD_ratio_extrapolation.py — Power BI and DAX

## Formål
Vise årssluttprognose for fristprosent med usikkerhetsintervall basert på sesongmønster.

This page is split in two, deliberately: **actual YTD so far** is standard DAX
time-intelligence against the fact table (no script needed — it's the "YTD in scripts
does not make sense" case), and **the year-end forecast with its confidence interval**
comes from `analyser.frist_prognose`, the one thing that genuinely can't be a DAX measure
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
- X-akse: `Kalender[Dato]` (daglig)
- Hel linje: `[Fristprosent YTD]`

## Del 2 — Årssluttprognose (persistert modell)

Datakilde: `analyser.frist_prognose` (written nightly by
`Seasonal_YTD_ratio_extrapolation.py`).

The table continues the YTD line from where the actuals stop to 31 December,
**one row per day** — the same grain `[Fristprosent YTD]` is plotted at, so the two go on
one axis and meet:

| `type` | rows | `verdi` | bånd |
|---|---|---|---|
| `Anker` | one, at the last date with data | observed YTD that day | zero width — it's an observation |
| `Prognose` | every day from there to 31 Dec | projected YTD that day | opens up with the horizon |

`prognose_aarsslutt` is the 31 December endpoint, repeated on every row so the KPI cards
read it without a date filter. `nedre_konfidensgrense`/`oevre_konfidensgrense` belong to
that row's own `verdi` — the 90% year-end interval scaled along the same seasonal path, so
the band closes on the year-end interval on 31 December.

Two things follow from the daily grain, and both were what made the earlier version of
this page useless:

- **A month-end-only forecast is three points on a daily axis**, which a smoothed line
  draws as a flat segment hanging over the last quarter. Hence one row per day.
- **The forecast starts on the actual value**, not one period after it. The `Anker` row
  carries the observed YTD at the last date with data, so the dashed line leaves the solid
  one at exactly the point the solid one reaches — the fork is where the projection
  begins, and there is no gap to interpret.

### Visualforslag

#### 1) Linjediagram: faktisk YTD + prognose til årsslutt

- **Modell:** relate `Kalender[Dato]` (1) → `frist_prognose[analyse_dato]` (*), single
  direction. Both are daily, so this is a plain date relationship.
- **X-akse:** `Kalender[Dato]`, 1. jan – 31. des, same axis as Del 1.
- **Serie 1 (hel linje):** `[Fristprosent YTD]` — actual, up to the last date with data.
- **Serie 2 (stiplet linje):** `[Prognose YTD]` — from that same date to 31 December.
- **Bånd:** `[Prognose nedre]` and `[Prognose oevre]`. Power BI's core line chart has no
  band primitive — either add them as two thin dashed lines, or lay a band-capable visual
  underneath. Zero width at the anchor, widest at 31 December: uncertainty grows with the
  horizon.
- **Mållinje:** constant line at `alert_config[terskel_amber]`.

Both series are the same quantity — YTD frist% — one observed, one projected. The last
point of the dashed line is `[Prognose årslutt]` on the card; if they ever disagree, the
semantic model is filtering the two sides differently.

The projected line is *not* flat, and is not forced to rise. YTD frist% is a ratio, so it
falls whenever the months ahead are historically worse than the year so far — for most of
these indicators the seasonal pattern erodes towards December, and that slope is the part
of the picture worth looking at.

#### Én felle: månedsmålet på samme akse

`[Fristprosent (måned)]` (from `Trendretning_POWERBI_DAX.md`) is each month on its own and
swings; this forecast is cumulative YTD. On one axis a viewer reads the projected line as
"we expect 84% *in* October", which the model does not say — it says 84% *for the year*.
Keep the two apart, or, if the monthly curve is the chart you want, put the forecast on it
as a labelled constant reference line (`Prognose årslutt 84 %, 90 % KI 79–89 %`) rather
than as a series over the remaining months.

#### 2) KPI-kort med intervall
- Stor verdi: `[Prognose årslutt]`
- Undertekst: `[Prognose intervalltekst]` — "90 % KI: 79–89 %"
- Farge: `[Prognose RAG]` mot `alert_config`
- Sammenlign mot målverdi (`alert_config[terskel_amber]`)

#### 3) Oppsummeringstabell
- `indikator`, `[Fristprosent YTD]` (Del 1), `[Prognose årslutt]`, `[Prognose CI lower]`, `[Prognose CI upper]`
- Sortering: `[Prognose årslutt]` asc

### DAX-forslag

```DAX
Prognose YTD =
LASTNONBLANKVALUE(
    frist_prognose[analyse_dato],
    MAX(frist_prognose[verdi])
)
```

```DAX
Prognose nedre =
LASTNONBLANKVALUE(
    frist_prognose[analyse_dato],
    MAX(frist_prognose[nedre_konfidensgrense])
)
```

```DAX
Prognose oevre =
LASTNONBLANKVALUE(
    frist_prognose[analyse_dato],
    MAX(frist_prognose[oevre_konfidensgrense])
)
```

`LASTNONBLANKVALUE`, not a plain `MAX`, so the line still reads correctly if someone rolls
the axis up to months or quarters: a YTD value belongs to the *end* of the period, and the
projected path can slope downwards, where `MAX` would silently pick the period's first day
instead of its last. At the daily grain the two are the same.

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
- `Kalender[Dato]` (Del 1 only — Del 2 covers the current year by construction, and the
  card measures above deliberately ignore the date filter)

## Tolkning
- Prognose under mål = risiko.
- Konfidensbånd som krysser mållinjen = usikker måloppnåelse.
- Båndet blir smalere etter hvert som mer av året er faktisk — it is scaled from the
  year-end interval, and that interval narrows as the year fills in.
- The last point of the dashed line equals `[Prognose årslutt]` on the card. Same column,
  so a mismatch means the visual is filtering, not that the model disagrees with itself.
- The dashed line must *start* on the solid one. The `Anker` row is the same observed YTD
  the live measure computes, so a visible step at the fork means the nightly run is stale
  or the two sides' filters (`%avtalt%`, fagomraade) have drifted apart. That check used to
  need a query; now it is just looking at the chart. There is still nothing enforcing it
  automatically.
- Early in a month the anchor sits a few days into it. The projection accounts for that —
  it uses the seasonal ratio interpolated to the actual date rather than treating a
  part-finished month as finished, which otherwise biases the year-end estimate downwards
  by roughly the size of one month's seasonal step.

## Åpenbar begrensning
The nightly run deletes and rewrites the whole current year, so the table holds only the
newest forecast. Forecast drift over time — "in June we projected 88%, now we project
84%" — is not recoverable from it. That would be the genuinely interesting line chart
here, and it needs the run to keep its history first.
