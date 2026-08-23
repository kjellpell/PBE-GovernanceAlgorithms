# Caseworker_Load_Concentration.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Er aktiv saksmengde jevnt fordelt i et team, eller konsentrert på få personer?
- Hvilke enheter har høyest konsentrasjon (bus-factor / burnout-risiko)?

**Merk:** `saksbehandler_arbeidsmengde` inneholder individdata og er ment for
intern kapasitetsplanlegging hos ledere — ikke for automatisk individvarsling.
Vurder å begrense tilgang til denne siden i Power BI-rapportsikkerheten.

Datakilder:
- `analyser.saksbehandler_arbeidsmengde` (individnivå, ingen historikk)
- `analyser.saksbehandler_konsentrasjon` (enhetsnivå, historikk)

## Anbefalte visualer

### 1) Gini-trend per enhet
- X-akse: `snapshot_dato`
- Y-akse: `gini_koeffisient`
- Filter: `tilstrekkelig_volum = TRUE`
- Slicer: `enhet`

### 2) Arbeidsmengde-fordeling (individnivå, tilgangsbegrenset)
- Stolpediagram: `saksbehandler` (X) vs `aktiv_saksmengde` (Y), per `enhet`
- Referanselinje: gjennomsnittlig saksmengde for enheten

### 3) KPI-kort
- `gini_koeffisient` siste snapshot, per enhet
- `antall_saksbehandlere` og `total_aktive_saker` siste snapshot

## DAX-forslag

### Siste konsentrasjon-snapshot
```DAX
Siste konsentrasjon-snapshot =
CALCULATE(
    MAX(saksbehandler_konsentrasjon[snapshot_dato]),
    ALL(saksbehandler_konsentrasjon[snapshot_dato])
)
```

### Gini siste, kun tilstrekkelig volum
```DAX
Gini siste =
CALCULATE(
    AVERAGE(saksbehandler_konsentrasjon[gini_koeffisient]),
    saksbehandler_konsentrasjon[snapshot_dato] = [Siste konsentrasjon-snapshot],
    saksbehandler_konsentrasjon[tilstrekkelig_volum] = TRUE()
)
```

### Konsentrasjon-fargekode
```DAX
Konsentrasjon fargekode =
VAR G = [Gini siste]
RETURN
SWITCH(
    TRUE(),
    ISBLANK(G), "#757575",
    G >= 0.5, "#B00020",
    G >= 0.3, "#F9A825",
    "#2E7D32"
)
```

## Slicer-oppsett
- `enhet`
- `snapshot_dato`

## Tolkning
- `tilstrekkelig_volum = FALSE` means fewer than `MIN_SAKSBEHANDLERE` (3) active caseworkers in that enhet — the Gini value is NULL and should not be charted or acted on.
- A rising Gini trend at stable total caseload means the same work is concentrating on fewer people, not that the team is busier overall — a workload-balancing conversation, not a hiring one.
