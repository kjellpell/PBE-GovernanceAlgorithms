# Caseworker load concentration — Power BI and DAX

**Målgruppe:** Analytiker-rapport — en Gini-koeffisient lærer en leder ingenting ved
første øyekast slik en prosentandel eller et antall dager gjør. Vis kun fargekode/nivå i
et eventuelt leder-vindu, aldri rå Gini-verdi.

## Formål
Denne siden skal svare på:
- Er aktiv saksmengde jevnt fordelt i et team, eller konsentrert på få personer?
- Hvilke enheter har høyest konsentrasjon (bus-factor / burnout-risiko)?

**Merk:** dette er individdata og er ment for intern kapasitetsplanlegging hos ledere —
ikke for automatisk individvarsling. Vurder å begrense tilgang til denne siden i Power
BI-rapportsikkerheten, uansett om tallene kommer fra `Faser[saksansvarlig]`-detaljene
(Del 1) eller fra Gini-målet (Del 2) — tilgangsbegrensning må håndheves i rapportlaget,
ikke ved å gjemme et mål.

This is entirely live DAX — no script, no nightly table. Rank-based Lorenz-curve math
(the Gini coefficient) is usually assumed to need a script, but it's a standard
`RANKX`/`SUMX` iterator pattern over the current filter context (enhet x indikator), the
same shape as any concentration/Pareto measure — not the disconnected-date or
window-scan trickery this repo's own rule (see `README.md`) flags as too iterative for
DAX. There is a real tradeoff: this gives you today's concentration only, not a trend
line of how Gini moved over time. That was judged not worth a standing pipeline
(script + a table growing one row per enhet x indikator every night, forever) to
maintain for a single management signal — see git history for the prior
snapshot-table version if a trend view is ever needed again.

## Del 1 — Dagens arbeidsmengde (per saksbehandler)

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

## Del 2 — Konsentrasjon nå (live Gini)

Same reasoning as `Caseworker_Load_Concentration.py`'s old gate: Gini on 1-2 people is
meaningless, and indicator effort isn't comparable, so this is computed per enhet x
indikator, never blended across indikatorer within an enhet (a blended Gini could hide
concentration on a heavier indikator behind a pile of lighter ones), and never
per-person (individual-level flagging stays out of scope, same as `CUSUM_Changepoint.py`'s
saksbehandler exclusion in its drilldown).

### Mål (measures)

```DAX
Saksmengde per saksbehandler (tabell) =
FILTER(
    ADDCOLUMNS(
        VALUES(Saker[saksansvarlig]),
        "@Saksmengde", [Aktiv saksmengde (saksbehandler)]
    ),
    NOT ISBLANK(Saker[saksansvarlig]) && [@Saksmengde] > 0
)
```

```DAX
Antall saksbehandlere (aktiv saksmengde) =
COUNTROWS([Saksmengde per saksbehandler (tabell)])
```

```DAX
Tilstrekkelig volum (konsentrasjon) =
[Antall saksbehandlere (aktiv saksmengde)] >= 3
```

```DAX
Gini-koeffisient (nå) =
VAR Saksmengder = [Saksmengde per saksbehandler (tabell)]
VAR N = COUNTROWS(Saksmengder)
VAR Total = SUMX(Saksmengder, [@Saksmengde])
VAR Rangert =
    ADDCOLUMNS(
        Saksmengder,
        "@Rang", RANKX(Saksmengder, [@Saksmengde], , ASC, DENSE)
    )
VAR VektetSum = SUMX(Rangert, [@Rang] * [@Saksmengde])
RETURN
    IF(
        [Tilstrekkelig volum (konsentrasjon)],
        DIVIDE(2 * VektetSum, N * Total) - DIVIDE(N + 1, N)
    )
```

```DAX
Konsentrasjon fargekode =
VAR G = [Gini-koeffisient (nå)]
RETURN
SWITCH(
    TRUE(),
    ISBLANK(G), "#757575",
    G >= 0.5, "#B00020",
    G >= 0.3, "#F9A825",
    "#2E7D32"
)
```

### Visualforslag
- KPI-kort med `[Konsentrasjon fargekode]`, per enhet x indikator — aldri rå
  `[Gini-koeffisient (nå)]` i et leder-vindu
- `[Antall saksbehandlere (aktiv saksmengde)]` og `Total_aktive_saker` (sum av
  `[@Saksmengde]`) som støttetall ved siden av fargekoden

## Slicer-oppsett
- `enhet`
- `indikator`

## Tolkning
- `[Tilstrekkelig volum (konsentrasjon)] = FALSE` betyr færre enn 3 aktive
  saksbehandlere for den enhet x indikator-kombinasjonen — Gini-målet returnerer BLANK og
  skal verken vises eller handles på.
- Gini beregnes per indikator, ikke blandet på tvers av en enhets indikatorer, fordi
  indikator-innsats ikke er sammenlignbar og ikke finnes i dataene — å blande ville la
  konsentrasjon på en tyngre indikator gjemme seg bak (eller bli skjult av) en haug
  lettere indikatorer.
- Dette er et øyeblikksbilde, ikke en trend — en stigende/synkende konsentrasjon over tid
  kan ikke leses av dette målet alene. Hvis en trendvisning blir nødvendig igjen, se
  git-historikken for `Caseworker_Load_Concentration.py` og
  `saksbehandler_konsentrasjon`-tabellen den skrev.
