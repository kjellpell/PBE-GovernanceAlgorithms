# Backlog_Aging_Distribution.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Blir den eksisterende saksbunken eldre over tid, uavhengig av nettoflyt?
- Hvilke indikatorer/enheter har flest saker i den eldste aldersgruppen?

This page is split in two, deliberately: **today's shape** is computed live in DAX
straight from the fact table (no nightly wait), and **the trend over time** comes from
`sak_alder_fordeling`, the one thing a live measure structurally cannot reconstruct —
`Aldersgruppe` depends on `TODAY()`, so it can only ever tell you about today, never about
three months ago. `Backlog_Aging_Distribution.py`'s only remaining job is writing that
daily snapshot down.

## Del 1 — Dagens bilde (live DAX, ingen ventetid på nattlig kjøring)

### Antagelser (rename to match your model)
- `Faser` — the fact table (`saksbehandling.faser`), containing `startmilepaeldato`,
  `sluttmilepaeldato`, `indikator`, `enhet`
- The `indikator NOT LIKE '%avtalt%'` / fagomraade filtering assumption from elsewhere in
  this repo's DAX docs applies here too

### Beregnet kolonne — alder og aldersgruppe

```DAX
Faser[AlderDager] =
IF(
    ISBLANK(Faser[startmilepaeldato]) || NOT ISBLANK(Faser[sluttmilepaeldato]),
    BLANK(),
    DATEDIFF(Faser[startmilepaeldato], TODAY(), DAY)
)
```

```DAX
Faser[Aldersgruppe] =
VAR Alder = Faser[AlderDager]
RETURN
    SWITCH(
        TRUE(),
        ISBLANK(Alder), BLANK(),
        Alder <= 30,  "0-30",
        Alder <= 60,  "31-60",
        Alder <= 90,  "61-90",
        Alder <= 180, "91-180",
        "180+"
    )
```

```DAX
Faser[AldersgruppeRang] =
SWITCH(Faser[Aldersgruppe], "0-30", 1, "31-60", 2, "61-90", 3, "91-180", 4, "180+", 5, 0)
```
(Sort-by column for `Aldersgruppe` so bucket order is chronological, not alphabetical.)

### Mål (measures)

```DAX
Åpne saker (antall) =
CALCULATE(COUNTROWS(Faser), NOT ISBLANK(Faser[AlderDager]))
```

```DAX
Median alder dager =
CALCULATE(MEDIANX(Faser, Faser[AlderDager]), NOT ISBLANK(Faser[AlderDager]))
```

```DAX
P90 alder dager =
CALCULATE(
    PERCENTILEX.INC(Faser, Faser[AlderDager], 0.9),
    NOT ISBLANK(Faser[AlderDager])
)
```

```DAX
Andel 180 pluss (nå) =
VAR Total = [Åpne saker (antall)]
VAR Eldst = CALCULATE([Åpne saker (antall)], Faser[Aldersgruppe] = "180+")
RETURN
    DIVIDE(Eldst, Total)
```

### Visual — dagens aldersfordeling
- X-akse: `enhet` (eller `indikator`)
- Stabler: `Faser[Aldersgruppe]` (sortert på `AldersgruppeRang`)
- Verdi: `[Åpne saker (antall)]`

## Del 2 — Trend over tid (persistert snapshot)

Datakilde: `analyser.sak_alder_fordeling` (written nightly by `Backlog_Aging_Distribution.py`
— see that script's header for why this can't be live DAX).

### Visualforslag
- X-akse: `snapshot_dato`, Y-akse: `antall_saker`, farge: `aldersgruppe`
- Slicer: `indikator`, `enhet`
- Shows: vokser den eldste bøtten (`180+`) over tid?

### DAX-forslag

```DAX
Siste alder-snapshot =
CALCULATE(
    MAX(sak_alder_fordeling[snapshot_dato]),
    ALL(sak_alder_fordeling[snapshot_dato])
)
```

```DAX
Andel 180 pluss (siste snapshot) =
VAR Total = CALCULATE(SUM(sak_alder_fordeling[antall_saker]), sak_alder_fordeling[snapshot_dato] = [Siste alder-snapshot])
VAR Eldst = CALCULATE(
    SUM(sak_alder_fordeling[antall_saker]),
    sak_alder_fordeling[snapshot_dato] = [Siste alder-snapshot],
    sak_alder_fordeling[aldersgruppe] = "180+"
)
RETURN DIVIDE(Eldst, Total)
```

```DAX
Aldersgruppe rang (snapshot-tabell) =
SWITCH(
    MAX(sak_alder_fordeling[aldersgruppe]),
    "0-30", 1, "31-60", 2, "61-90", 3, "91-180", 4, "180+", 5,
    0
)
```
(Same sort-by fix as `Faser[AldersgruppeRang]` above, for this table's own `aldersgruppe`
column.)

## Slicer-oppsett
- `indikator`
- `enhet`
- `snapshot_dato` (Del 2 only — Del 1 is always "now")

## Tolkning
- A growing `180+` share while the throughput pressure monitor's net-flow score (see
  `Throughput_Pressure_Monitor_POWERBI_DAX.md`) looks flat means old cases are stuck, not
  that intake is outpacing completion — a different intervention (case triage, not
  capacity).
- Del 1 and Del 2 should roughly agree on "today" (Del 2's latest snapshot vs. Del 1 live)
  — if they diverge, the nightly run is stale or the live filters have drifted from the
  script's filters. Keep both filter sets in sync by hand; there's no single source of
  truth to enforce it automatically.
