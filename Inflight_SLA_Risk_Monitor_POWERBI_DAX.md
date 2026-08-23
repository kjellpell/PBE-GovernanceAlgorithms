# Inflight_SLA_Risk_Monitor.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Hvor mange åpne saker er i ferd med å sprenge fristen, før de lukkes?
- Hvilke indikatorer/enheter har flest saker i `Kritisk` eller `Bruddet`?

Datakilder:
- `analyser.sak_frist_risiko` (dagens åpne saker, én rad per sak)
- `analyser.sak_frist_risiko_trend` (aggregert, historikk over tid)

## Anbefalte visualer

### 1) KPI-kort (fra sak_frist_risiko)
- Antall saker `risikoklasse = 'Bruddet'`
- Antall saker `risikoklasse = 'Kritisk'`
- Andel av åpne saker i `Bruddet` eller `Kritisk`

### 2) Risikotabell — daglig intervensjonsliste
- `indikator`, `enhet`, `fasetittel`, `dager_igjen`, `andel_brukt`, `risikoklasse`
- Filter: `risikoklasse IN ('Bruddet', 'Kritisk')`
- Sortering: `dager_igjen` stigende

### 3) Trend over tid (sak_frist_risiko_trend)
- X-akse: `snapshot_dato`
- Y-akse: stablet areal — `andel_bruddet`, `andel_kritisk`, `andel_risiko`, `andel_innenfor`
- Filter: `tilstrekkelig_volum = TRUE`
- Slicer: `indikator`, `enhet`

## DAX-forslag

### Andel i risiko siste snapshot
```DAX
Andel i risiko siste =
CALCULATE(
    DIVIDE(
        COUNTROWS(FILTER(sak_frist_risiko, sak_frist_risiko[risikoklasse] IN {"Bruddet", "Kritisk"})),
        COUNTROWS(sak_frist_risiko)
    )
)
```

### Risikoklasse rang (for sortering/fargekoding)
```DAX
Risikoklasse rang =
SWITCH(
    MAX(sak_frist_risiko[risikoklasse]),
    "Bruddet", 4,
    "Kritisk", 3,
    "Risiko", 2,
    "Innenfor", 1,
    0
)
```

## Slicer-oppsett
- `indikator`
- `enhet`
- `risikoklasse`

## Tolkning
- Dette er en *leading* indikator — sammenlign med `ewma_analyse`/`cusum_analyse` sin `Fristprosent`, som kun måler lukkede saker.
- `andel_brukt = NULL` betyr `frist_dager <= 0` for den saken — sjekk datakvalitet før den saken brukes i tiltak.
