# Trendretning — native DAX (raw values; trend signal now comes from CUSUM)

**Målgruppe:** Leder-rapport — leser kaldt, ingen forklaring nødvendig.

## Formål

This file used to also compute trend direction (`Stigende`/`Synkende`/`Stabil`) natively,
from the month-on-month slope of a rolling average — a simpler-to-explain replacement for
`EWMA.py`. That's been superseded: every indicator here has 20+ years of history, more
than enough to build a proper CUSUM baseline, and `CUSUM_Changepoint.py` already runs
nightly for the analyst-facing drift signal — so the board's trend arrow now comes from
that anchored-baseline CUSUM test (`Trendretning (CUSUM)` in
`CUSUM_Changepoint_POWERBI_DAX.md`) instead of a hand-tuned slope threshold. A real
statistical test beats a threshold picked to "look right," and since the nightly pipeline
was already running for other reasons, reading its `signalretning` column directly cost
nothing extra — `signalretning`/`endringsretning` use the same `Stigende`/`Synkende`/
`Stabil` vocabulary as this file, so no translation layer was needed either.

What's left here is the plain per-period measures: the raw value line the board's CUSUM
arrow is drawn against, and the same live DAX measure
`CUSUM_Changepoint_POWERBI_DAX.md` points to instead of duplicating the raw value inside
`cusum_analyse`.

**No Delta table, no nightly run** for what remains below. These are measures against the
fact table (`saksbehandling.faser` in the Lakehouse — referred to below by whatever
name it's imported into the semantic model as).

## Antagelser (rename to match your model)
- `Faser` — the fact table (`saksbehandling.faser`), containing `indikator`,
  `sluttmilepaeldato`, `startmilepaeldato`, `frist_dager`, `innenfor_frist`, `tidsbruk`
- `Kalender` — a standard date table marked as the model's Date Table, with `Kalender[Dato]`
  related to `Faser[sluttmilepaeldato]` (use `startmilepaeldato` as well for
  `Produksjonsdifferanse`, per the original script's `COALESCE`-based period)
- All measures assume the same `START_YEAR` / fagomraade filtering as the old script is
  already applied elsewhere in the model (e.g. via a base filter measure or RLS) — add
  `indikatorer[fagomraade] IN {...}` back into the measures below if it isn't

## Grunnmål (per måned, i gjeldende filterkontekst)

```DAX
Fristprosent (måned) =
DIVIDE(
    CALCULATE(COUNTROWS(Faser), Faser[innenfor_frist] = TRUE()),
    CALCULATE(COUNTROWS(Faser), NOT ISBLANK(Faser[frist_dager]))
)
```

```DAX
Behandlingstid (måned) =
AVERAGE(Faser[tidsbruk])
```

```DAX
Produksjonsdifferanse (måned) =
CALCULATE(COUNTROWS(Faser), NOT ISBLANK(Faser[startmilepaeldato]))
    - CALCULATE(COUNTROWS(Faser), NOT ISBLANK(Faser[sluttmilepaeldato]))
```

## Visualforslag
- **Linjediagram:** `Fristprosent (måned)` (rå verdi) per `indikator` — the line CUSUM's
  signal is judged against; add `cusum_analyse[cusum_positiv]`/`cusum_negativ` from
  `CUSUM_Changepoint_POWERBI_DAX.md` if you want the accumulator visible too, on an
  analyst-facing page.
- **Trendkort:** use `Trendretning (CUSUM)` (see `CUSUM_Changepoint_POWERBI_DAX.md`), not a
  measure from this file — same Stigende=grønn / Synkende=rød / Stabil=nøytral color rule
  as before.

## Tolkning
- Trend direction lives in `CUSUM_Changepoint_POWERBI_DAX.md` now — this file only supplies
  the raw per-period numbers everything else (the board arrow, the CUSUM line, the raw
  line on a chart) is built from.
- `Fristprosent (måned)` is still a **period rate, not year-to-date** — see
  `Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md` for why that matters and how it
  relates to the forecast page.
