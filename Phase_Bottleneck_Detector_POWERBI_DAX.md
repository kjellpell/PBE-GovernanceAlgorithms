# Phase bottleneck detector — native DAX (replaces Phase_Bottleneck_Detector.py)

## Formål
Denne siden skal svare på:
- Hvilke faser er reelle flaskehalser per enhet og indikator?
- Skyldes flaskehalsen inngang/utgang-ubalanse, p90-forverring, eller begge?
- Hvilke faser bør prioriteres først i neste forbedringsmøte?

**No Delta table, no nightly run.** Same composite-scoring pattern as
`Throughput_Pressure_Monitor_POWERBI_DAX.md` one level down (phase instead of team) — see
that doc's caveats section for background on the general approach. This one is actually
slightly easier: no streak logic, and the "queue proxy" turns out to be a plain running
total rather than anything sequential.

## Two things worth knowing before you build this

1. **A third date role.** Unlike the throughput monitor (which counts "received" and
   "completed" independently by their own dates), this table groups each row by its own
   *single* representative date — `COALESCE(sluttmilepaeldato, startmilepaeldato)` — and
   then checks which of the two underlying dates is populated. That needs its own
   calculated column and its own (inactive) relationship to `Kalender`, on top of whatever
   `Faser` already has from the throughput conversion.
2. **The reason-code string is the ugliest part.** The old script tags each row with a
   `+`-joined list of trigger codes (`NETTO_FLYT_POS`, `P90_OPP`, …) and a human-readable
   sentence built from which codes fired. DAX can build this (table constructor +
   `CONCATENATEX`, below), but it's genuinely more natural in five lines of Python than as
   a measure. If maintainability matters more than parity, consider dropping the
   concatenated reason string and exposing the underlying boolean flags as separate
   measures instead — a report can show "P90 opp: Ja/Nei" columns just as well as a
   packed code string.

## Antagelser (rename to match your model)
- `Faser` — `saksbehandling.faser`, with `indikator`, `enhet`, `fasetittel`,
  `startmilepaeldato`, `sluttmilepaeldato`, `tidsbruk`
- `Kalender` — the model's Date Table. In addition to whatever relationships the
  throughput conversion added, add:
  `Faser[AktivitetsDato] = COALESCE(Faser[sluttmilepaeldato], Faser[startmilepaeldato])`
  (calculated column) related to `Kalender[Dato]` — inactive, activated via
  `USERELATIONSHIP` in the measures below.
- Same fagomraade/`%avtalt%` filtering assumption as elsewhere in this repo's DAX docs

## Grunnmål (fase-nivå, per måned i gjeldende filterkontekst)

```DAX
Fase inngang antall (måned) =
CALCULATE(
    COUNTROWS(Faser),
    USERELATIONSHIP(Faser[AktivitetsDato], Kalender[Dato]),
    NOT ISBLANK(Faser[startmilepaeldato])
)
```

```DAX
Fase utgang antall (måned) =
CALCULATE(
    COUNTROWS(Faser),
    USERELATIONSHIP(Faser[AktivitetsDato], Kalender[Dato]),
    NOT ISBLANK(Faser[sluttmilepaeldato])
)
```

```DAX
Fase netto flyt (måned) = [Fase inngang antall (måned)] - [Fase utgang antall (måned)]
```

```DAX
Fase median fasetid (måned) =
CALCULATE(
    MEDIANX(Faser, Faser[tidsbruk]),
    USERELATIONSHIP(Faser[AktivitetsDato], Kalender[Dato])
)
```

```DAX
Fase p90 fasetid (måned) =
CALCULATE(
    PERCENTILEX.INC(Faser, Faser[tidsbruk], 0.9),
    USERELATIONSHIP(Faser[AktivitetsDato], Kalender[Dato])
)
```

## Kø-proxy (kumulativ nettoflyt, gulvet på 0)

The old script's `netto_flyt.cumsum().clip(lower=0)` looks sequential but isn't — it's a
plain running total, which is a standard DAX idiom (widen the date filter to "everything up
to and including the current month" and let the additive `COUNTROWS`-based measure
re-aggregate over it):

```DAX
Fase ko proxy =
VAR Kumulativ =
    CALCULATE(
        [Fase netto flyt (måned)],
        USERELATIONSHIP(Faser[AktivitetsDato], Kalender[Dato]),
        FILTER(ALL(Kalender), Kalender[Dato] <= MAX(Kalender[Dato]))
    )
RETURN
    MAX(Kumulativ, 0)
```

## Baseline og avvik (12-måneders rullerende median, ekskl. inneværende måned)

Same pattern as the throughput monitor's baseline, just against the fase-level measures:

```DAX
Fase baseline median =
VAR SisteDato = MAX(Kalender[Dato])
VAR ForrigeMaanedSlutt = EOMONTH(SisteDato, -1)
VAR Vindu = DATESINPERIOD(Kalender[Dato], ForrigeMaanedSlutt, -12, MONTH)
VAR Maaneder = SUMMARIZE(Vindu, Kalender[År], Kalender[Månedsnummer])
VAR AntallMndMedData = COUNTROWS(FILTER(Maaneder, NOT ISBLANK([Fase median fasetid (måned)])))
RETURN
    IF(AntallMndMedData >= 6, MEDIANX(Maaneder, [Fase median fasetid (måned)]))
```

```DAX
Fase baseline p90 =
VAR SisteDato = MAX(Kalender[Dato])
VAR ForrigeMaanedSlutt = EOMONTH(SisteDato, -1)
VAR Vindu = DATESINPERIOD(Kalender[Dato], ForrigeMaanedSlutt, -12, MONTH)
VAR Maaneder = SUMMARIZE(Vindu, Kalender[År], Kalender[Månedsnummer])
VAR AntallMndMedData = COUNTROWS(FILTER(Maaneder, NOT ISBLANK([Fase p90 fasetid (måned)])))
RETURN
    IF(AntallMndMedData >= 6, MEDIANX(Maaneder, [Fase p90 fasetid (måned)]))
```

```DAX
Fase avvik median pct =
VAR Naa = [Fase median fasetid (måned)]
VAR Baseline = [Fase baseline median]
RETURN
    IF(NOT ISBLANK(Baseline) && Baseline > 0, DIVIDE(Naa - Baseline, Baseline))
```

```DAX
Fase avvik p90 pct =
VAR Naa = [Fase p90 fasetid (måned)]
VAR Baseline = [Fase baseline p90]
RETURN
    IF(NOT ISBLANK(Baseline) && Baseline > 0, DIVIDE(Naa - Baseline, Baseline))
```

```DAX
Fase tilstrekkelig volum =
[Fase inngang antall (måned)] >= 10 || [Fase utgang antall (måned)] >= 10  -- MIN_SEGMENT_OBS
```

## Score, alvorlighet og årsak

```DAX
Fase score =
VAR NettoFlyt = [Fase netto flyt (måned)]
VAR AvvikP90 = [Fase avvik p90 pct]
VAR Inngang = [Fase inngang antall (måned)]
VAR Utgang = [Fase utgang antall (måned)]
VAR KoProxy = [Fase ko proxy]
RETURN
    IF(NettoFlyt > 0, 1.5, 0)
    + IF(NOT ISBLANK(AvvikP90) && AvvikP90 > 0.15, 1.5, 0)
    + IF(NOT ISBLANK(AvvikP90) && AvvikP90 > 0.30, 1.0, 0)
    + IF(Inngang > 0 && Utgang <= Inngang * 0.8, 1.0, 0)
    + IF(KoProxy > 0 && NettoFlyt > 0, 1.0, 0)
```

```DAX
Fase alvorlighet =
VAR S = [Fase score]
RETURN
    SWITCH(TRUE(), S >= 4.0, "Kritisk", S >= 3.0, "Hoy", S >= 1.5, "Moderat", "Lav")
```

```DAX
Fase bottleneck flagg =
[Fase tilstrekkelig volum] && [Fase score] >= 3.0
```

```DAX
Fase arsak kode =
VAR NettoFlyt = [Fase netto flyt (måned)]
VAR AvvikP90 = [Fase avvik p90 pct]
VAR Inngang = [Fase inngang antall (måned)]
VAR Utgang = [Fase utgang antall (måned)]
VAR KoProxy = [Fase ko proxy]
VAR Flagg =
    {
        IF(NettoFlyt > 0, "NETTO_FLYT_POS"),
        IF(NOT ISBLANK(AvvikP90) && AvvikP90 > 0.15, "P90_OPP"),
        IF(NOT ISBLANK(AvvikP90) && AvvikP90 > 0.30, "P90_OPP_STERK"),
        IF(Inngang > 0 && Utgang <= Inngang * 0.8, "UTGANGSGAP"),
        IF(KoProxy > 0 && NettoFlyt > 0, "KO_VEKST")
    }
VAR AktiveFlagg = FILTER(Flagg, NOT ISBLANK([Value]))
RETURN
    IF(COUNTROWS(AktiveFlagg) = 0, "INGEN", CONCATENATEX(AktiveFlagg, [Value], "+"))
```

```DAX
Fase arsak tekst =
VAR Kode = [Fase arsak kode]
RETURN
    SWITCH(
        TRUE(),
        Kode = "INGEN", "Ingen tydelig faseflaskehals i perioden.",
        CONTAINSSTRING(Kode, "KO_VEKST") && CONTAINSSTRING(Kode, "P90_OPP"),
            "Kø vokser samtidig som fasevariasjon i p90 forverres.",
        CONTAINSSTRING(Kode, "NETTO_FLYT_POS"),
            "Flere saker går inn i fasen enn ut i perioden.",
        "Fasesignal indikerer tregere flyt enn historisk baseline."
    )
```

## Anbefalte visualer

### 1) KPI-kort
- Antall rader med `[Fase bottleneck flagg] = TRUE` siste måned
- Antall enheter med minst én flaskehals siste måned
- Gjennomsnitt `[Fase avvik p90 pct]` for flaskehalser siste måned

### 2) Flaskehals-matrise (hovedvisual)
- Rader: `enhet`, kolonner: `fasetittel`, verdi: `[Fase alvorlighet]` eller
  `[Fase avvik p90 pct]`
- Filter: `[Fase bottleneck flagg] = TRUE` og siste måned

### 3) Prioritert tiltakstabell
- `enhet`, `indikator`, `fasetittel`, `[Fase alvorlighet]`, `[Fase arsak kode]`,
  `[Fase inngang antall (måned)]`, `[Fase utgang antall (måned)]`, `[Fase avvik p90 pct]`
- Sortering: alvorlighet (Kritisk først), deretter `[Fase avvik p90 pct]` desc

### 4) Trend per fase
- X-akse: `Kalender[Dato]`
- Linjer: `[Fase p90 fasetid (måned)]` og `[Fase baseline p90]`

## Slicer-oppsett
- `Kalender[Dato]` (default: siste måned)
- `indikator`, `enhet`, `fasetittel`

## Tolkning
- Høy `[Fase avvik p90 pct]` + `KO_VEKST` i `[Fase arsak kode]` indikerer sann
  prosessflaskehals.
- `[Fase tilstrekkelig volum] = FALSE` bør ikke brukes som styrende tiltakssignal.
