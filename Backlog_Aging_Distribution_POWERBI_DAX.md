# Backlog_Aging_Distribution.py — Power BI and DAX

**Målgruppe:** Leder-rapport — leser kaldt, ingen forklaring nødvendig.

## Formål
Denne siden skal svare på:
- Blir den eksisterende saksbunken eldre over tid, uavhengig av nettoflyt?
- Hvilke indikatorer/enheter har flest saker i den eldste aldersgruppen?

**Two clocks, not one blended age.** The original version measured age as pure calendar
time (`TODAY() - startmilepaeldato`), which silently blends client-caused delay into what
looked like an internal-performance problem — a case open 90 days where 80 of those days
were us waiting on the client looked identical to one where all 90 days were on us. Fixed
by tracking the two accumulators separately:
- **Tidsbruk** — accumulated time on our side. The real internal-performance signal.
- **Bransjetid** — accumulated time waiting on the client. Not our team's performance.

Both are live, independently-accumulating totals already on the fact table, so there's no
need to determine which clock is "currently" running — a case can carry both a Tidsbruk
figure and a Bransjetid figure at once, bucketed independently. The completion-status
column on the fact table isn't used here; these two accumulators already say everything
this page needs.

This page is still split in two, deliberately: **today's shape** is computed live in DAX
straight from the fact table (no nightly wait), and **the trend over time** comes from
`sak_alder_fordeling`, the one thing a live measure structurally cannot reconstruct —
`Aldersgruppe` depends on the current value of a live-accumulating column, so it can only
ever tell you about today, never about three months ago. `Backlog_Aging_Distribution.py`'s
only remaining job is writing that daily snapshot down, once per clock.

## Del 1 — Dagens bilde (live DAX, ingen ventetid på nattlig kjøring)

### Antagelser (rename to match your model)
- `Faser` — the fact table (`saksbehandling.faser`), containing `startmilepaeldato`,
  `sluttmilepaeldato`, `indikator`, `enhet`, `tidsbruk`, `bransjetid`
- `tidsbruk`/`bransjetid` are assumed to already be day counts, live-updating while a case
  is open (same assumption every other script in this repo makes about `tidsbruk`) —
  verify against the Lakehouse schema if that turns out to be wrong
- The `indikator NOT LIKE '%avtalt%'` / fagomraade filtering assumption from elsewhere in
  this repo's DAX docs applies here too

### Beregnede kolonner — aldersgruppe per klokke

Same bucket boundaries, applied twice — once per clock. Neither depends on the other; a
row can have a value in both.

```DAX
Faser[Aldersgruppe_Tidsbruk] =
IF(
    ISBLANK(Faser[sluttmilepaeldato]) = FALSE() || ISBLANK(Faser[tidsbruk]),
    BLANK(),
    SWITCH(
        TRUE(),
        Faser[tidsbruk] < 0, BLANK(),
        Faser[tidsbruk] <= 30,  "0-30",
        Faser[tidsbruk] <= 60,  "31-60",
        Faser[tidsbruk] <= 90,  "61-90",
        Faser[tidsbruk] <= 180, "91-180",
        "180+"
    )
)
```

```DAX
Faser[Aldersgruppe_Bransjetid] =
IF(
    ISBLANK(Faser[sluttmilepaeldato]) = FALSE() || ISBLANK(Faser[bransjetid]),
    BLANK(),
    SWITCH(
        TRUE(),
        Faser[bransjetid] < 0, BLANK(),
        Faser[bransjetid] <= 30,  "0-30",
        Faser[bransjetid] <= 60,  "31-60",
        Faser[bransjetid] <= 90,  "61-90",
        Faser[bransjetid] <= 180, "91-180",
        "180+"
    )
)
```

```DAX
Faser[AldersgruppeRang] =
SWITCH(Faser[Aldersgruppe_Tidsbruk], "0-30", 1, "31-60", 2, "61-90", 3, "91-180", 4, "180+", 5, 0)
```
(Sort-by column — reuse the same rank logic for `Aldersgruppe_Bransjetid`'s visual, or
duplicate the column if both need independent sorting in the same view.)

### Mål (measures) — dupliser per klokke

```DAX
Åpne saker (Tidsbruk) =
CALCULATE(COUNTROWS(Faser), ISBLANK(Faser[sluttmilepaeldato]), NOT ISBLANK(Faser[tidsbruk]))
```

```DAX
Median Tidsbruk dager (åpne) =
CALCULATE(
    MEDIANX(Faser, Faser[tidsbruk]),
    ISBLANK(Faser[sluttmilepaeldato]), NOT ISBLANK(Faser[tidsbruk])
)
```

```DAX
P90 Tidsbruk dager (åpne) =
CALCULATE(
    PERCENTILEX.INC(Faser, Faser[tidsbruk], 0.9),
    ISBLANK(Faser[sluttmilepaeldato]), NOT ISBLANK(Faser[tidsbruk])
)
```

```DAX
Andel 180 pluss (Tidsbruk, nå) =
VAR Total = [Åpne saker (Tidsbruk)]
VAR Eldst = CALCULATE([Åpne saker (Tidsbruk)], Faser[Aldersgruppe_Tidsbruk] = "180+")
RETURN
    DIVIDE(Eldst, Total)
```

Repeat all four for `Bransjetid` (`Åpne saker (Bransjetid)`, `Median Bransjetid dager
(åpne)`, `P90 Bransjetid dager (åpne)`, `Andel 180 pluss (Bransjetid, nå)`) — identical
shape, swap the column.

### Visual — dagens bilde, asymmetrisk med vilje
Ikke to like store paneler — de to klokkene er ikke like handlingsrelevante for en leder,
og et speilvendt par gjør siden tyngre å lese enn den trenger å være. Bruk plain-language
titler i selve visualet, ikke feltnavnene `Tidsbruk`/`Bransjetid` — de er interne
kolonnenavn, ikke noe en leder skal måtte vite betydningen av.

- **Hovedpanel, tittel "Vårt ansvar":** X-akse `enhet` (eller `indikator`), stabler
  `Faser[Aldersgruppe_Tidsbruk]`, verdi `[Åpne saker (Tidsbruk)]`, med `[P90 Tidsbruk
  dager (åpne)]` som KPI-kort ved siden av. Dette er handlingspanelet — høy P90 eller
  voksende `180+`-andel her betyr vi må bemanne/prioritere disse sakene, og det er
  genuint vårt å fikse.
- **Referansetall, tittel "Venter på klient":** ett KPI-kort —
  `[Åpne saker (Bransjetid)]` filtrert til `Faser[Aldersgruppe_Bransjetid] = "180+"` —
  "X saker har ventet over 180 dager på klienten." Ikke en full søylefordeling; det er
  nok til at en leder vet det er verdt å spørre om, uten å måtte tolke en hel
  bøttefordeling for noe som ikke er internt vårt ansvar. Den fulle
  `Aldersgruppe_Bransjetid`-fordelingen (median/P90/alle bøtter) finnes fortsatt som mål
  ovenfor hvis analytiker-rapporten vil bruke den — den er bare ikke på leder-siden.

## Del 2 — Trend over tid (persistert snapshot)

Datakilde: `analyser.sak_alder_fordeling` (written nightly by `Backlog_Aging_Distribution.py`
— see that script's header for why this can't be live DAX). Now has a `klokke` column
(`'Tidsbruk'` / `'Bransjetid'`) — filter to the one you want, or use it as a legend/small-
multiple dimension to show both at once.

### Visualforslag
- X-akse: `snapshot_dato`, Y-akse: `antall_saker`, farge: `aldersgruppe`
- Default filter: `klokke = "Tidsbruk"` — leder-siden viser vår egen trend som standard;
  bytt til `"Bransjetid"` via slicer for å se klient-siden, samme asymmetri som Del 1
- Slicer: `indikator`, `enhet`, `klokke`
- Shows: vokser den eldste bøtten (`180+`) over tid, på vår klokke?

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
(Filter or slice by `sak_alder_fordeling[klokke]` to split this by Tidsbruk vs. Bransjetid.)

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
- `klokke` (Del 2 only, default `"Tidsbruk"` — Del 1 shows the Tidsbruk panel plus the
  Bransjetid reference card side by side instead of a slicer)
- `snapshot_dato` (Del 2 only — Del 1 is always "now")

## Tolkning
- A growing `180+` share on **Tidsbruk** while the throughput pressure monitor's net-flow
  score (see `Throughput_Pressure_Monitor_POWERBI_DAX.md`) looks flat means old cases are
  stuck on our side specifically — case triage, not capacity, and it's genuinely ours to
  fix.
- A growing `180+` share on **Bransjetid** is a client-responsiveness pattern, not an
  internal one — the action is chasing the client or the industry party, not reassigning
  caseworkers.
- Del 1 and Del 2 should roughly agree on "today" (Del 2's latest snapshot vs. Del 1 live,
  per klokke) — if they diverge, the nightly run is stale or the live filters have drifted
  from the script's filters. Keep both filter sets in sync by hand; there's no single
  source of truth to enforce it automatically.
