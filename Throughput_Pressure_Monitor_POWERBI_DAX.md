# Throughput_Pressure_Monitor.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Hvor er gjennomstrømmingspresset størst akkurat nå?
- Skyldes presset netto innflyt, økt tidsbruk, eller begge?
- Hvilken fase bidrar mest i pressede enheter?

Datakilder:
- `analyser.gjennomstoremming_press_enhet`
- `analyser.gjennomstroemming_press_fase`

## Anbefalte visualer

### 1) KPI-kort (øverst)
- Antall enheter med `pressure_nivaa` = `Kritisk`
- Antall enheter med `pressure_nivaa` i (`Hoy`, `Kritisk`)
- Sum `netto_flyt` siste måned
- Median `tidsbruk_avvik_pct` siste måned

### 2) Prioritert tabell (hovedliste)
Vis siste `analyse_dato`:
- `enhet`, `indikator`, `pressure_nivaa`, `pressure_score`, `netto_flyt`, `netto_flyt_streak`, `p90_tidsbruk`, `tidsbruk_avvik_pct`
- Sortering: `pressure_score` desc, `netto_flyt` desc

### 3) Trendlinje per enhet
- X-akse: `analyse_dato`
- Y-akse 1: `netto_flyt`
- Y-akse 2: `p90_tidsbruk` (sekundær akse)
- Små multipler: `enhet`
- Filter: `indikator`

### 4) Fasebidrag for valgt enhet
Fra `analyser.gjennomstroemming_press_fase`:
- Stolper: `fasetittel`
- Verdi: `fase_bidrag_score`
- Farge: `fase_press_flagg`
- Filter: siste `analyse_dato`

## Datamodell
- Relasjon mellom tabeller:
  - Nøkkel: `indikator`, `enhet`, `analyse_dato` (modell med sammensatt nøkkel i Power Query eller ved hjelp av sammenslått tekstnøkkel)
- Foreslått sammenslått nøkkel i begge tabeller:
  - `nokkel_enhet_dato = enhet & "|" & indikator & "|" & FORMAT(analyse_dato, "yyyy-MM-dd")`

## DAX-forslag

### Siste dato i kontekst
```DAX
Siste analysedato =
CALCULATE(
    MAX(gjennomstoremming_press_enhet[analyse_dato]),
    ALL(gjennomstoremming_press_enhet[analyse_dato])
)
```

### Er siste rad
```DAX
Er siste rad =
VAR SisteDato = [Siste analysedato]
RETURN IF(MAX(gjennomstoremming_press_enhet[analyse_dato]) = SisteDato, 1, 0)
```

### Antall kritiske enheter (siste måned)
```DAX
Antall kritiske enheter siste =
CALCULATE(
    DISTINCTCOUNT(gjennomstoremming_press_enhet[enhet]),
    FILTER(
        gjennomstoremming_press_enhet,
        gjennomstoremming_press_enhet[analyse_dato] = [Siste analysedato]
            && gjennomstoremming_press_enhet[pressure_nivaa] = "Kritisk"
    )
)
```

### Antall høyt press (Hoy + Kritisk)
```DAX
Antall høyt press siste =
CALCULATE(
    DISTINCTCOUNT(gjennomstoremming_press_enhet[enhet]),
    FILTER(
        gjennomstoremming_press_enhet,
        gjennomstoremming_press_enhet[analyse_dato] = [Siste analysedato]
            && gjennomstoremming_press_enhet[pressure_nivaa] IN {"Hoy", "Kritisk"}
    )
)
```

### Netto flyt siste måned
```DAX
Netto flyt siste =
CALCULATE(
    SUM(gjennomstoremming_press_enhet[netto_flyt]),
    FILTER(
        gjennomstoremming_press_enhet,
        gjennomstoremming_press_enhet[analyse_dato] = [Siste analysedato]
    )
)
```

### Tidsbruksavvik prosentpoeng (siste)
```DAX
Tidsbruksavvik pp siste =
CALCULATE(
    AVERAGE(gjennomstoremming_press_enhet[tidsbruk_avvik_pct]) * 100,
    FILTER(
        gjennomstoremming_press_enhet,
        gjennomstoremming_press_enhet[analyse_dato] = [Siste analysedato]
    )
)
```

### Fargekode for pressnivå
```DAX
Pressnivå fargekode =
SWITCH(
    MAX(gjennomstoremming_press_enhet[pressure_nivaa]),
    "Kritisk", "#B00020",
    "Hoy", "#E65100",
    "Moderat", "#F9A825",
    "Lav", "#2E7D32",
    "#757575"
)
```

## Slicer-oppsett
- `analyse_dato` (default: siste)
- `indikator`
- `enhet`
- `pressure_nivaa`

## Tolkning
- `netto_flyt > 0` over flere perioder + økende `p90_tidsbruk` = sannsynlig kapasitetsproblem.
- Høy `fase_bidrag_score` identifiserer hvor tiltak bør settes inn først.
