# Trendretning — native DAX (replaces EWMA.py)

**Målgruppe:** Leder-rapport — leser kaldt, ingen forklaring nødvendig.

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

## Siste dato med data (hjelpemål)

Every measure below anchors its window on "the last relevant date." A bare
`MAX(Kalender[Dato])` is not safe for that: it returns the last date **in the current
filter context**, and `Kalender` is a full calendar table that typically extends past the
fact data (through year-end, or further). Filter to a year with no month selected, or drop
the measure on a card with no date filter at all, and `MAX(Kalender[Dato])` returns a date
months into the future with zero `Faser` rows behind it — every measure downstream
(`DATESINPERIOD`, the glidende snitt, `helning`) silently evaluates to blank in that
window, and `Trendretning` reports "Stabil" forever, not because nothing is moving but
because there's nothing there to look at.

```DAX
Siste dato med data =
VAR SisteFaktaDato =
    CALCULATE(
        MAX(Faser[sluttmilepaeldato]),
        ALL(Kalender)
    )
RETURN
    CALCULATE(
        MAX(Kalender[Dato]),
        Kalender[Dato] <= SisteFaktaDato
    )
```

`SisteFaktaDato` ignores whatever filter is active and finds the true last date with a
closed fase. The outer `CALCULATE` then intersects that cap with the *existing* filter
context rather than replacing it — so a visual that already pins a narrower date (a line
chart axis, a single month) is unaffected, since its own max date is already at or before
the cap; only a wide-open context (a year filter, a filter-less card) gets reined in from
running off the end of the calendar. Every `SisteDato` below uses this measure instead of
`MAX(Kalender[Dato])` directly.

## Glidende snitt (moving average measure pattern)

Standard DAX idiom: average the monthly measure over each of the last N months
individually (row-context transition via `SUMMARIZE` + `AVERAGEX`), rather than summing
raw rows over the window — this keeps ratio/average/count-diff measures all correct without
a separate formula per shape.

```DAX
Fristprosent glidende snitt rask (3 mnd) =
VAR SisteDato = [Siste dato med data]
VAR Maaneder = DATESINPERIOD(Kalender[Dato], SisteDato, -3, MONTH)
RETURN
    AVERAGEX(
        SUMMARIZE(Maaneder, Kalender[År], Kalender[Månedsnummer]),
        [Fristprosent (måned)]
    )
```

```DAX
Fristprosent glidende snitt sakte (6 mnd) =
VAR SisteDato = [Siste dato med data]
VAR Maaneder = DATESINPERIOD(Kalender[Dato], SisteDato, -6, MONTH)
RETURN
    AVERAGEX(
        SUMMARIZE(Maaneder, Kalender[År], Kalender[Månedsnummer]),
        [Fristprosent (måned)]
    )
```

Repeat both for `Behandlingstid (måned)` and `Produksjonsdifferanse (måned)`
(`Behandlingstid glidende snitt rask/sakte`, `Produksjonsdifferanse glidende snitt
rask/sakte`) — same two measures, swap the inner `[...]` reference. Keep
`VAR SisteDato = [Siste dato med data]` in all of them; only the innermost `[...]`
reference changes per metric.

## Helning og trendretning

Trend direction is based on the **slow (6-month)** rolling average's month-on-month slope —
same choice the old script made ("stable signal for board").

```DAX
Fristprosent helning sakte =
VAR SisteDato = [Siste dato med data]
VAR ForrigeMaaned = EOMONTH(SisteDato, -1)
VAR SnittNaa = [Fristprosent glidende snitt sakte (6 mnd)]
VAR SnittForrige =
    CALCULATE(
        [Fristprosent glidende snitt sakte (6 mnd)],
        FILTER(ALL(Kalender), Kalender[Dato] = ForrigeMaaned)
    )
RETURN
    IF(
        ISBLANK(SnittNaa) || ISBLANK(SnittForrige),
        BLANK(),
        SnittNaa - SnittForrige
    )
```

`SnittNaa`/`SnittForrige` are guarded explicitly rather than trusting `Helning`'s own
blankness: DAX's `-` operator coerces a lone blank operand to 0, so without this guard a
genuinely missing side (e.g. no 6-month history yet at the start of the dataset) reads as
a large fake swing instead of "no signal."

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
VAR SisteDato = [Siste dato med data]
VAR ForrigeMaaned = EOMONTH(SisteDato, -1)
VAR SnittNaa = [Behandlingstid glidende snitt sakte (6 mnd)]
VAR SnittForrige =
    CALCULATE(
        [Behandlingstid glidende snitt sakte (6 mnd)],
        FILTER(ALL(Kalender), Kalender[Dato] = ForrigeMaaned)
    )
RETURN
    IF(
        ISBLANK(SnittNaa) || ISBLANK(SnittForrige),
        BLANK(),
        SnittNaa - SnittForrige
    )
```

```DAX
Produksjonsdifferanse helning sakte =
VAR SisteDato = [Siste dato med data]
VAR ForrigeMaaned = EOMONTH(SisteDato, -1)
VAR SnittNaa = [Produksjonsdifferanse glidende snitt sakte (6 mnd)]
VAR SnittForrige =
    CALCULATE(
        [Produksjonsdifferanse glidende snitt sakte (6 mnd)],
        FILTER(ALL(Kalender), Kalender[Dato] = ForrigeMaaned)
    )
RETURN
    IF(
        ISBLANK(SnittNaa) || ISBLANK(SnittForrige),
        BLANK(),
        SnittNaa - SnittForrige
    )
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
