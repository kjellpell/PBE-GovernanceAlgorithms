# Backlog_Aging_Distribution.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Blir den eksisterende saksbunken eldre over tid, uavhengig av nettoflyt?
- Hvilke indikatorer/enheter har flest saker i den eldste aldersgruppen?

Datakilde:
- `analyser.sak_alder_fordeling`

## Anbefalte visualer

### 1) Stablet stolpediagram — dagens aldersfordeling
- X-akse: `enhet` (eller `indikator`)
- Stabler: `aldersgruppe`
- Verdi: `antall_saker`
- Filter: `snapshot_dato = MAX(snapshot_dato)`

### 2) Trend over tid
- X-akse: `snapshot_dato`
- Y-akse: `antall_saker`, farge: `aldersgruppe`
- Slicer: `indikator`, `enhet`
- Shows: vokser den eldste bøtten (`180+`) over tid?

### 3) KPI-kort
- `p90_alder_dager` siste snapshot, per indikator
- Antall saker i `180+` siste snapshot

## DAX-forslag

### Siste snapshot
```DAX
Siste alder-snapshot =
CALCULATE(
    MAX(sak_alder_fordeling[snapshot_dato]),
    ALL(sak_alder_fordeling[snapshot_dato])
)
```

### Andel i eldste aldersgruppe
```DAX
Andel 180 pluss =
VAR Total = CALCULATE(SUM(sak_alder_fordeling[antall_saker]), sak_alder_fordeling[snapshot_dato] = [Siste alder-snapshot])
VAR Eldst = CALCULATE(
    SUM(sak_alder_fordeling[antall_saker]),
    sak_alder_fordeling[snapshot_dato] = [Siste alder-snapshot],
    sak_alder_fordeling[aldersgruppe] = "180+"
)
RETURN DIVIDE(Eldst, Total)
```

### Aldersgruppe sortering
```DAX
Aldersgruppe rang =
SWITCH(
    MAX(sak_alder_fordeling[aldersgruppe]),
    "0-30", 1,
    "31-60", 2,
    "61-90", 3,
    "91-180", 4,
    "180+", 5,
    0
)
```
(Set `Aldersgruppe rang` as the sort-by column for `aldersgruppe` so bucket order is chronological, not alphabetical.)

## Slicer-oppsett
- `indikator`
- `enhet`
- `snapshot_dato`

## Tolkning
- A growing `180+` share while the throughput pressure monitor's net-flow score (see `Throughput_Pressure_Monitor_POWERBI_DAX.md`) looks flat means old cases are stuck, not that intake is outpacing completion — a different intervention (case triage, not capacity).
