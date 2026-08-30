# Phase_Bottleneck_Detector.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Hvilke faser er reelle flaskehalser per enhet og indikator?
- Skyldes flaskehalsen inngang/utgang-ubalanse, p90-forverring, eller begge?
- Hvilke faser bør prioriteres først i neste forbedringsmøte?

Datakilde:
- `analyser.fase_flaskehals_enhet`

## Anbefalte visualer

### 1) KPI-kort
- Antall rader med `bottleneck_flagg = TRUE` siste måned
- Antall enheter med minst én flaskehals siste måned
- Gjennomsnitt `avvik_p90_pct` for flaskehalser siste måned

### 2) Flaskehals-matrise (hovedvisual)
- Rader: `enhet`
- Kolonner: `fasetittel`
- Verdi: `alvorlighet` eller `avvik_p90_pct`
- Filter: `bottleneck_flagg = TRUE` og siste `analyse_dato`

### 3) Prioritert tiltakstabell
- `enhet`, `indikator`, `fasetittel`, `alvorlighet`, `arsak_kode`, `inngang_antall`, `utgang_antall`, `avvik_p90_pct`
- Sortering: `alvorlighet` (Kritisk først), deretter `avvik_p90_pct` desc

### 4) Trend per fase
- X-akse: `analyse_dato`
- Linjer: `p90_fasetid` og `baseline_p90`
- Små multipler: `fasetittel` eller `enhet`

## DAX-forslag

### Siste analysedato
```DAX
Siste analysedato flaskehals =
CALCULATE(
    MAX(fase_flaskehals_enhet[analyse_dato]),
    ALL(fase_flaskehals_enhet[analyse_dato])
)
```

### Antall flaskehalser siste
```DAX
Antall flaskehalser siste =
CALCULATE(
    COUNTROWS(fase_flaskehals_enhet),
    FILTER(
        fase_flaskehals_enhet,
        fase_flaskehals_enhet[analyse_dato] = [Siste analysedato flaskehals]
            && fase_flaskehals_enhet[bottleneck_flagg] = TRUE()
    )
)
```

### Antall berørte enheter siste
```DAX
Antall berørte enheter siste =
CALCULATE(
    DISTINCTCOUNT(fase_flaskehals_enhet[enhet]),
    FILTER(
        fase_flaskehals_enhet,
        fase_flaskehals_enhet[analyse_dato] = [Siste analysedato flaskehals]
            && fase_flaskehals_enhet[bottleneck_flagg] = TRUE()
    )
)
```

### Gjennomsnitt p90-avvik siste
```DAX
Gjennomsnitt p90-avvik siste =
CALCULATE(
    AVERAGE(fase_flaskehals_enhet[avvik_p90_pct]) * 100,
    FILTER(
        fase_flaskehals_enhet,
        fase_flaskehals_enhet[analyse_dato] = [Siste analysedato flaskehals]
            && fase_flaskehals_enhet[bottleneck_flagg] = TRUE()
    )
)
```

### Alvorlighet sortering
```DAX
Alvorlighet rang =
SWITCH(
    MAX(fase_flaskehals_enhet[alvorlighet]),
    "Kritisk", 4,
    "Hoy", 3,
    "Moderat", 2,
    "Lav", 1,
    0
)
```

### Flaskehals fargekode
```DAX
Flaskehals fargekode =
SWITCH(
    MAX(fase_flaskehals_enhet[alvorlighet]),
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
- `fasetittel`
- `alvorlighet`

## Tolkning
- Høy `avvik_p90_pct` + `KO_VEKST` i `arsak_kode` indikerer sann prosessflaskehals.
- `tilstrekkelig_volum = FALSE` bør ikke brukes som styrende tiltakssignal.
