# Throughput pressure monitor — native DAX (replaces Throughput_Pressure_Monitor.py)

## Formål
Denne siden skal svare på:
- Hvor er gjennomstrømmingspresset størst akkurat nå?
- Skyldes presset netto innflyt, økt tidsbruk, eller begge?
- Hvilken fase bidrar mest i pressede enheter?

**No Delta table, no nightly run.** Everything below is measures against the fact table
(`saksbehandling.faser` in the Lakehouse, referred to below as `Faser`) plus a standard
date table (`Kalender`). Unlike `Backlog_Aging_Distribution`, this one has no point-in-time
problem — `mottatt`/`ferdigstilt` counts by month are stable historical facts, not "what's
currently open," so trend-over-time charts work fine live, same as with the old table.

## Two honest caveats before you build this

1. **`Netto flyt streak` is the fragile part.** "How many consecutive prior months had
   positive net flow" is a run-length/sequence computation — DAX has no native primitive
   for it, so the measure below fakes it by scanning a bounded lookback window for the
   nearest non-positive month. It works, but it's exactly the kind of iterative,
   order-dependent logic that's more awkward and slower in DAX than in five lines of
   pandas — validate it against real data volumes before trusting it, and don't be
   surprised if it needs tuning.
2. **Two date roles need modeling.** `mottatt_antall` (received) is keyed by
   `startmilepaeldato`'s month and `ferdigstilt_antall` (completed) by
   `sluttmilepaeldato`'s month — independently, not the same row's single date. That means
   `Faser` needs **two relationships** to `Kalender[Dato]`: keep the existing one on
   `sluttmilepaeldato` active, add an inactive one on `startmilepaeldato`, and activate it
   with `USERELATIONSHIP` in the "mottatt" measure below. If your model doesn't already
   have both relationships, add the second one before using any of this.

## Antagelser (rename to match your model)
- `Faser` — `saksbehandling.faser`, with `indikator`, `enhet`, `fasetittel`,
  `startmilepaeldato`, `sluttmilepaeldato`, `tidsbruk`
- `Kalender` — the model's Date Table, `Kalender[Dato]` actively related to
  `Faser[sluttmilepaeldato]`, inactively related to `Faser[startmilepaeldato]`
- Same fagomraade/`%avtalt%` filtering assumption as elsewhere in this repo's DAX docs —
  add it back into the measures if it isn't already enforced upstream

## Grunnmål (team/enhet-nivå, per måned i gjeldende filterkontekst)

```DAX
Mottatt antall (måned) =
CALCULATE(
    COUNTROWS(Faser),
    USERELATIONSHIP(Faser[startmilepaeldato], Kalender[Dato]),
    NOT ISBLANK(Faser[startmilepaeldato])
)
```

```DAX
Ferdigstilt antall (måned) =
CALCULATE(COUNTROWS(Faser), NOT ISBLANK(Faser[sluttmilepaeldato]))
```

```DAX
Netto flyt (måned) = [Mottatt antall (måned)] - [Ferdigstilt antall (måned)]
```

```DAX
Median tidsbruk (måned) =
CALCULATE(MEDIANX(Faser, Faser[tidsbruk]), NOT ISBLANK(Faser[sluttmilepaeldato]))
```

```DAX
P90 tidsbruk (måned) =
CALCULATE(
    PERCENTILEX.INC(Faser, Faser[tidsbruk], 0.9),
    NOT ISBLANK(Faser[sluttmilepaeldato])
)
```

## Baseline og avvik (12-måneders rullerende median, ekskl. inneværende måned)

```DAX
Baseline P90 tidsbruk =
VAR SisteDato = MAX(Kalender[Dato])
VAR ForrigeMaanedSlutt = EOMONTH(SisteDato, -1)
VAR Vindu = DATESINPERIOD(Kalender[Dato], ForrigeMaanedSlutt, -12, MONTH)
VAR Maaneder = SUMMARIZE(Vindu, Kalender[År], Kalender[Månedsnummer])
VAR AntallMndMedData =
    COUNTROWS(FILTER(Maaneder, NOT ISBLANK([P90 tidsbruk (måned)])))
RETURN
    IF(AntallMndMedData >= 6, MEDIANX(Maaneder, [P90 tidsbruk (måned)]))
```

```DAX
Tidsbruk avvik pct =
VAR Naa = [P90 tidsbruk (måned)]
VAR Baseline = [Baseline P90 tidsbruk]
RETURN
    IF(NOT ISBLANK(Baseline) && Baseline > 0, DIVIDE(Naa - Baseline, Baseline))
```

## Netto flyt streak (se caveat over)

```DAX
Netto flyt streak =
VAR SisteDato = MAX(Kalender[Dato])
VAR MaksVindu = 24  -- pragmatisk grense, juster ved behov
VAR Historikk =
    ADDCOLUMNS(
        SUMMARIZE(
            DATESINPERIOD(Kalender[Dato], SisteDato, -MaksVindu, MONTH),
            Kalender[Dato]
        ),
        "@NettoFlyt", CALCULATE([Netto flyt (måned)]),
        "@Avstand", DATEDIFF(Kalender[Dato], SisteDato, MONTH)
    )
VAR FoersteIkkePositive =
    MINX(FILTER(Historikk, [@NettoFlyt] <= 0 || ISBLANK([@NettoFlyt])), [@Avstand])
RETURN
    IF(ISBLANK(FoersteIkkePositive), MaksVindu, FoersteIkkePositive)
```

## Pressure score og nivå

```DAX
Pressure score =
VAR NettoFlyt = [Netto flyt (måned)]
VAR Streak = [Netto flyt streak]
VAR Avvik = [Tidsbruk avvik pct]
VAR Mottatt = [Mottatt antall (måned)]
VAR Ferdigstilt = [Ferdigstilt antall (måned)]
RETURN
    IF(NettoFlyt > 0, 1.5, 0)
    + IF(Streak >= 3, 2.0, 0)
    + IF(NOT ISBLANK(Avvik) && Avvik > 0.10, 1.0, 0)
    + IF(NOT ISBLANK(Avvik) && Avvik > 0.25, 1.0, 0)
    + IF(Ferdigstilt > 0 && Mottatt >= Ferdigstilt * 1.3, 1.0, 0)
```

```DAX
Pressure nivaa =
VAR S = [Pressure score]
RETURN
    SWITCH(TRUE(), S >= 4.0, "Kritisk", S >= 3.0, "Hoy", S >= 1.5, "Moderat", "Lav")
```

```DAX
Tilstrekkelig volum (enhet) =
[Mottatt antall (måned)] >= 10 || [Ferdigstilt antall (måned)] >= 10
```

```DAX
Pressnivå fargekode =
SWITCH(
    [Pressure nivaa],
    "Kritisk", "#B00020",
    "Hoy", "#E65100",
    "Moderat", "#F9A825",
    "Lav", "#2E7D32",
    "#757575"
)
```

## Fase-nivå (fasetittel-grain, sammenlignet mot samme enhets team-tall)

Compare the fase-grain measures above against the same measures with `fasetittel` filters
removed (i.e. the team/enhet total) using `REMOVEFILTERS`:

```DAX
P90 tidsbruk (team) =
CALCULATE([P90 tidsbruk (måned)], REMOVEFILTERS(Faser[fasetittel]))
```

```DAX
Pressure nivaa (team) =
CALCULATE([Pressure nivaa], REMOVEFILTERS(Faser[fasetittel]))
```

```DAX
Tilstrekkelig volum (team) =
CALCULATE([Tilstrekkelig volum (enhet)], REMOVEFILTERS(Faser[fasetittel]))
```

```DAX
Fase bidrag score =
VAR FaseNettoFlyt = [Netto flyt (måned)]
VAR FaseP90 = [P90 tidsbruk (måned)]
VAR TeamP90 = [P90 tidsbruk (team)]
VAR TeamNivaa = [Pressure nivaa (team)]
RETURN
    IF(FaseNettoFlyt > 0, 1.5, 0)
    + IF(
        NOT ISBLANK(FaseP90) && NOT ISBLANK(TeamP90) && TeamP90 > 0
            && FaseP90 >= TeamP90 * 0.9,
        1.0, 0
      )
    + IF(TeamNivaa IN {"Hoy", "Kritisk"}, 1.0, 0)
```

```DAX
Fase press flagg =
[Tilstrekkelig volum (team)]
    && [Fase bidrag score] >= 2.0
    && [Ferdigstilt antall (måned)] >= 10
```

## Anbefalte visualer

### 1) KPI-kort
- Antall enheter med `[Pressure nivaa]` = `Kritisk` siste måned (`DISTINCTCOUNT` over
  `enhet` filtered to that condition)
- Sum `[Netto flyt (måned)]` siste måned
- Gjennomsnitt `[Tidsbruk avvik pct]` siste måned

### 2) Prioritert tabell
- `enhet`, `indikator`, `[Pressure nivaa]`, `[Pressure score]`, `[Netto flyt (måned)]`,
  `[Netto flyt streak]`, `[P90 tidsbruk (måned)]`, `[Tidsbruk avvik pct]`
- Sortering: `[Pressure score]` desc

### 3) Trendlinje per enhet
- X-akse: `Kalender[Dato]`
- Y-akse 1: `[Netto flyt (måned)]`, Y-akse 2: `[P90 tidsbruk (måned)]`
- Små multipler: `enhet`

### 4) Fasebidrag for valgt enhet
- Stolper: `fasetittel`, verdi: `[Fase bidrag score]`, farge: `[Fase press flagg]`

## Slicer-oppsett
- `indikator`, `enhet`, `Kalender[Dato]` (default: siste måned)

## Tolkning
- `[Netto flyt (måned)] > 0` over flere perioder + økende `[P90 tidsbruk (måned)]` =
  sannsynlig kapasitetsproblem.
- Høy `[Fase bidrag score]` identifiserer hvor tiltak bør settes inn først.
- Cross-reference with `Backlog_Aging_Distribution_POWERBI_DAX.md`'s aging share — flat
  net flow here with a growing `180+` bucket there means the backlog is stuck, not that
  intake is outpacing completion.
