# Caseworker_Load_Concentration.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Er aktiv saksmengde jevnt fordelt i et team, eller konsentrert på få personer?
- Hvilke enheter har høyest konsentrasjon (bus-factor / burnout-risiko)?

**Merk:** dette er individdata og er ment for intern kapasitetsplanlegging hos ledere —
ikke for automatisk individvarsling. Vurder å begrense tilgang til denne siden i Power
BI-rapportsikkerheten, uansett om tallene kommer fra `Faser` direkte (Del 1) eller fra
`saksbehandler_konsentrasjon` (Del 2) — tilgangsbegrensning må håndheves i rapportlaget,
ikke ved å gjemme en tabell.

This page is split in two: **today's per-caseworker counts and shares** are live DAX
straight from `Saker` (`Saker[saksansvarlig]` is already in the semantic model, so a
nightly copy of a plain `COUNTROWS`/`DIVIDE` would just be duplicated data), and **the
Gini concentration trend** comes from `saksbehandler_konsentrasjon` — the one thing here
that genuinely can't be a DAX measure (rank-based Lorenz-curve math) and, being a snapshot
of today's open caseload, is also a trend question a live measure can't answer on its own.

## Del 1 — Dagens arbeidsmengde (live DAX)

### Antagelser (rename to match your model)
- `Faser` — `saksbehandling.faser`, with `enhet`, `fk_saker`, `startmilepaeldato`,
  `sluttmilepaeldato`
- `Saker` — `saksbehandling.saker`, with `pk_saker`, `saksansvarlig`, related to `Faser`
  via `Saker[pk_saker]` = `Faser[fk_saker]`
- Same fagomraade/`%avtalt%` filtering assumption as the rest of this repo's DAX docs
- Rows with blank/null `saksansvarlig` (unassigned cases) should be excluded, not
  coalesced to "Ukjent" — an "Ukjent" pseudo-caseworker would corrupt the per-person
  concentration picture

### Mål (measures)

```DAX
Aktiv saksmengde (saksbehandler) =
CALCULATE(
    COUNTROWS(Faser),
    NOT ISBLANK(Faser[startmilepaeldato]),
    ISBLANK(Faser[sluttmilepaeldato]),
    NOT ISBLANK(Saker[saksansvarlig])
)
```

```DAX
Andel av enhetens saksmengde =
DIVIDE(
    [Aktiv saksmengde (saksbehandler)],
    CALCULATE([Aktiv saksmengde (saksbehandler)], REMOVEFILTERS(Saker[saksansvarlig]))
)
```

### Visual — arbeidsmengde-fordeling (individnivå, tilgangsbegrenset)
- Stolpediagram: `Saker[saksansvarlig]` (X) vs `[Aktiv saksmengde (saksbehandler)]` (Y),
  per `enhet`
- Referanselinje: gjennomsnittlig saksmengde for enheten

## Del 2 — Konsentrasjonstrend (persistert Gini)

Datakilde: `analyser.saksbehandler_konsentrasjon` (written nightly by
`Caseworker_Load_Concentration.py` — enhet-level only, no individual data, see that
script's header for why the Gini computation can't be live DAX).

### Visualforslag

#### 1) Gini-trend per enhet
- X-akse: `snapshot_dato`
- Y-akse: `gini_koeffisient`
- Filter: `tilstrekkelig_volum = TRUE`
- Slicer: `enhet`

#### 2) KPI-kort
- `gini_koeffisient` siste snapshot, per enhet
- `antall_saksbehandlere` og `total_aktive_saker` siste snapshot

### DAX-forslag

```DAX
Siste konsentrasjon-snapshot =
CALCULATE(
    MAX(saksbehandler_konsentrasjon[snapshot_dato]),
    ALL(saksbehandler_konsentrasjon[snapshot_dato])
)
```

```DAX
Gini siste, kun tilstrekkelig volum =
CALCULATE(
    AVERAGE(saksbehandler_konsentrasjon[gini_koeffisient]),
    saksbehandler_konsentrasjon[snapshot_dato] = [Siste konsentrasjon-snapshot],
    saksbehandler_konsentrasjon[tilstrekkelig_volum] = TRUE()
)
```

```DAX
Konsentrasjon fargekode =
VAR G = [Gini siste, kun tilstrekkelig volum]
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
- `snapshot_dato` (Del 2 only — Del 1 is always "now")

## Tolkning
- `tilstrekkelig_volum = FALSE` means fewer than `MIN_SAKSBEHANDLERE` (3) active caseworkers in that enhet — the Gini value is NULL and should not be charted or acted on.
- A rising Gini trend at stable total caseload means the same work is concentrating on fewer people, not that the team is busier overall — a workload-balancing conversation, not a hiring one.
- Del 1 and Del 2 should roughly agree on "today" (Del 2's latest snapshot's
  `total_aktive_saker`/`antall_saksbehandlere` vs. Del 1's live totals) — if they diverge,
  the nightly run is stale or the two sides' filters have drifted apart. Keep both filter
  sets in sync by hand; there's no single source of truth to enforce it automatically.
