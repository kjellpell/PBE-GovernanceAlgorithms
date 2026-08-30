# Trendretning — native DAX (replaces EWMA.py)

## Formål
Trend direction (`Stigende` / `Synkende` / `Stabil`) for `Fristprosent`, `Behandlingstid`
and `Produksjonsdifferanse`, from the slope of a rolling average — simpler to read than an
exponentially weighted average, and native DAX.

**No Delta table, no nightly run.** These are measures against the fact table
(`saksbehandling.faser` in the Lakehouse — referred to below by whatever
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

## Glidende snitt (moving average measure pattern)

Standard DAX idiom: average the monthly measure over each of the last N months
individually (row-context transition via `SUMMARIZE` + `AVERAGEX`), rather than summing
raw rows over the window — this keeps ratio/average/count-diff measures all correct without
a separate formula per shape.

```DAX
Fristprosent glidende snitt rask (3 mnd) =
VAR SisteDato = MAX(Kalender[Dato])
VAR Maaneder = DATESINPERIOD(Kalender[Dato], SisteDato, -3, MONTH)
RETURN
    AVERAGEX(
        SUMMARIZE(Maaneder, Kalender[År], Kalender[Månedsnummer]),
        [Fristprosent (måned)]
    )
```

```DAX
Fristprosent glidende snitt sakte (6 mnd) =
VAR SisteDato = MAX(Kalender[Dato])
VAR Maaneder = DATESINPERIOD(Kalender[Dato], SisteDato, -6, MONTH)
RETURN
    AVERAGEX(
        SUMMARIZE(Maaneder, Kalender[År], Kalender[Månedsnummer]),
        [Fristprosent (måned)]
    )
```

Repeat both for `Behandlingstid (måned)` and `Produksjonsdifferanse (måned)`
(`Behandlingstid glidende snitt rask/sakte`, `Produksjonsdifferanse glidende snitt
rask/sakte`) — same two measures, swap the inner `[...]` reference.

## Helning og trendretning

Trend direction is based on the **slow (6-month)** rolling average's month-on-month slope —
same choice the old script made ("stable signal for board").

```DAX
Fristprosent helning sakte =
VAR SisteDato = MAX(Kalender[Dato])
VAR ForrigeMaaned = EOMONTH(SisteDato, -1)
VAR SnittNaa = [Fristprosent glidende snitt sakte (6 mnd)]
VAR SnittForrige =
    CALCULATE(
        [Fristprosent glidende snitt sakte (6 mnd)],
        FILTER(ALL(Kalender), Kalender[Dato] = ForrigeMaaned)
    )
RETURN
    SnittNaa - SnittForrige
```

```DAX
Trendretning Fristprosent =
VAR Helning = [Fristprosent helning sakte]
VAR Terskel = 0.002  -- 0.2 prosentpoeng/måned, samme terskel som EWMA.py brukte
RETURN
    SWITCH(
        TRUE(),
        ISBLANK(Helning), "Stabil",
        Helning > Terskel, "Stigende",
        Helning < -Terskel, "Synkende",
        "Stabil"
    )
```

Samme helningsmønster for de to andre måltallene (bytt ut `[Fristprosent glidende snitt
sakte (6 mnd)]`-referansen med den tilsvarende måltall-versjonen):

```DAX
Behandlingstid helning sakte =
VAR SisteDato = MAX(Kalender[Dato])
VAR ForrigeMaaned = EOMONTH(SisteDato, -1)
VAR SnittNaa = [Behandlingstid glidende snitt sakte (6 mnd)]
VAR SnittForrige =
    CALCULATE(
        [Behandlingstid glidende snitt sakte (6 mnd)],
        FILTER(ALL(Kalender), Kalender[Dato] = ForrigeMaaned)
    )
RETURN
    SnittNaa - SnittForrige
```

```DAX
Produksjonsdifferanse helning sakte =
VAR SisteDato = MAX(Kalender[Dato])
VAR ForrigeMaaned = EOMONTH(SisteDato, -1)
VAR SnittNaa = [Produksjonsdifferanse glidende snitt sakte (6 mnd)]
VAR SnittForrige =
    CALCULATE(
        [Produksjonsdifferanse glidende snitt sakte (6 mnd)],
        FILTER(ALL(Kalender), Kalender[Dato] = ForrigeMaaned)
    )
RETURN
    SnittNaa - SnittForrige
```

Med samme terskler som `EWMA.py` brukte:

```DAX
Trendretning Behandlingstid =
VAR Helning = [Behandlingstid helning sakte]
VAR Terskel = 0.5  -- en halv dag/måned
RETURN
    SWITCH(TRUE(), ISBLANK(Helning), "Stabil", Helning > Terskel, "Stigende",
        Helning < -Terskel, "Synkende", "Stabil")
```

```DAX
Trendretning Produksjonsdifferanse =
VAR Helning = [Produksjonsdifferanse helning sakte]
VAR Terskel = 5.0  -- 5 saker/måned
RETURN
    SWITCH(TRUE(), ISBLANK(Helning), "Stabil", Helning > Terskel, "Stigende",
        Helning < -Terskel, "Synkende", "Stabil")
```

## Visualforslag
- **Linjediagram:** `Fristprosent (måned)` (rå verdi) + `Fristprosent glidende snitt sakte`
  on the same chart, per `indikator` — replaces the old raw + EWMA chart 1:1
- **Trendkort:** `Trendretning Fristprosent` for the latest month, same
  Stigende=grønn / Synkende=rød / Stabil=nøytral color rule as before

## Tolkning
- Styrevisning: bruk `glidende snitt sakte` (6 mnd) — stabil retning, lite støy.
- Virksomhetsoppfølging: bruk `glidende snitt rask` (3 mnd) — reagerer raskere.
- En "6-måneders glidende snitt" er lettere å forklare i et møte enn en eksponentielt
  vektet utjevning — samme funksjon (vis underliggende trend, ikke rå støy), enklere språk.
