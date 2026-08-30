# Seasonal_YTD_ratio_extrapolation.py — Power BI and DAX

**Målgruppe:** Analytiker-rapport — prognosen og intervallet leses greit, men å stole på
tallet krever å stole på sesongratio-metoden bak det.

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
- X-akse: `Kalender[Dato]` (jan–des inneværende år)
- Hel linje: `[Fristprosent YTD]`

## Del 2 — Årssluttprognose (persistert modell)

Datakilde: `prognoser.frist_prognose` (written nightly by
`Seasonal_YTD_ratio_extrapolation.py` — only forecast rows for the remaining months of the
current year; see that script's header for why actual YTD isn't in this table).

### Visualforslag

#### 1) Linjediagram: faktisk + prognose
- X-akse: `analyse_dato` — combine `[Fristprosent YTD]` (Del 1, solid line, past months)
  with `frist_prognose[verdi]` (Del 2, dashed line, remaining months) on the same chart.
  These are two different data sources on one visual — Power BI handles this fine as two
  series, they just don't come from the same table.
- Bånd: `nedre_konfidensgrense` til `oevre_konfidensgrense` for prognosedelen

#### 2) KPI-kort
- `Prognose årslutt`
- `Prognose CI lower`
- `Prognose CI upper`
- Sammenlign mot målverdi (`alert_config`)

#### 3) Oppsummeringstabell
- `indikator`, `[Fristprosent YTD]` (Del 1), `prognose_aarsslutt`, `nedre_konfidensgrense`, `oevre_konfidensgrense`
- Sortering: `prognose_aarsslutt` asc

### DAX-forslag

```DAX
Prognose årslutt =
CALCULATE(
    MAX(frist_prognose[prognose_aarsslutt]),
    frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
)
```

```DAX
Prognose CI lower =
CALCULATE(
    MAX(frist_prognose[nedre_konfidensgrense]),
    frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
)
```

```DAX
Prognose CI upper =
CALCULATE(
    MAX(frist_prognose[oevre_konfidensgrense]),
    frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
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
- `Kalender[Dato]` (Del 1 only — Del 2 is scoped to the current year already)

## Tolkning
- Prognose under mål = risiko.
- Konfidensbånd som krysser mållinjen = usikker måloppnåelse.
- Når flere måneder blir faktiske, skal båndene normalt bli smalere.
- Del 1's `[Fristprosent YTD]` at the latest closed month should match Del 2's
  `frist_prognose` forecast at that same month within rounding — if they diverge, the
  nightly run is stale or the two sides' filters (`%avtalt%`, fagomraade) have drifted
  apart. Keep both in sync by hand; there's no single source of truth to enforce it
  automatically.
