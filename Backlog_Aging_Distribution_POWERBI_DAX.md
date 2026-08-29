# Backlog aging distribution — native DAX (replaces Backlog_Aging_Distribution.py)

## Formål
Denne siden skal svare på:
- Blir den eksisterende saksbunken eldre over tid, uavhengig av nettoflyt?
- Hvilke indikatorer/enheter har flest saker i den eldste aldersgruppen?

**No Delta table, no nightly run.** Age bucketing and the median/p90 age are plain
row-level arithmetic and percentiles over currently-open cases — no algorithm here needs
Python, so this is now a calculated column plus a handful of measures directly against the
fact table (`saksbehandling.faser` in the Lakehouse, referred to below as `Faser`).

## Viktig begrensning: ingen historisk trend uten et snapshot
The old script's `snapshot_dato`-stamped, append-mode table is what let you chart "did the
180+ bucket grow over the last six months" — it captured what the *open* backlog looked
like on each past night. A live DAX measure only ever knows what `Faser` looks like **right
now**; `TODAY()` has no memory of what was open three months ago, so this replacement can
answer "what does the backlog look like today" but **cannot** reproduce the old trend-over-
time chart. If that trend view still matters, something still needs to persist a daily
snapshot somewhere (even a trivial one) — that's not a DAX limitation to work around, it's
an inherent fact about point-in-time state vs. a live table. Everything below covers the
"today" view only.

## Antagelser (rename to match your model)
- `Faser` — the fact table (`saksbehandling.faser`), containing `startmilepaeldato`,
  `sluttmilepaeldato`, `indikator`, `enhet`
- The `indikator NOT LIKE '%avtalt%'` and `fagomraade IN ('Byggesak', 'Eiendomssak',
  'Plansak')` filters the old script applied are assumed already enforced elsewhere in the
  model (e.g. a base RLS/filter measure) — add them back into the measures below if not
- `enhet` blank/whitespace values should already be normalized to `"Ukjent"` in Power
  Query (`Faser[enhet]` cleanup step) — the old script did this with
  `COALESCE(NULLIF(TRIM(enhet), ''), 'Ukjent')`

## Beregnet kolonne — alder og aldersgruppe

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
SWITCH(
    Faser[Aldersgruppe],
    "0-30", 1, "31-60", 2, "61-90", 3, "91-180", 4, "180+", 5,
    0
)
```
(Set `AldersgruppeRang` as the sort-by column for `Aldersgruppe` so bucket order is
chronological, not alphabetical — same fix the old script's DAX doc needed.)

Calculated columns recompute at each model refresh, so on a nightly-refreshed model this
behaves the same as the old script's "recomputed every night" snapshot — it's the *history
across* refreshes that isn't kept, not "today's" number.

## Mål (measures)

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
Andel 180 pluss =
VAR Total = [Åpne saker (antall)]
VAR Eldst = CALCULATE([Åpne saker (antall)], Faser[Aldersgruppe] = "180+")
RETURN
    DIVIDE(Eldst, Total)
```

## Anbefalte visualer

### 1) Stablet stolpediagram — dagens aldersfordeling
- X-akse: `enhet` (eller `indikator`)
- Stabler: `Faser[Aldersgruppe]` (sortert på `AldersgruppeRang`)
- Verdi: `[Åpne saker (antall)]`

### 2) KPI-kort
- `[P90 alder dager]` og `[Median alder dager]` per indikator
- `[Andel 180 pluss]`

(The old "trend over time" visual is dropped here — see the limitation above.)

## Slicer-oppsett
- `indikator`
- `enhet`

## Tolkning
- A high `[Andel 180 pluss]` while the throughput/pressure measures (see
  `Throughput_Pressure_Monitor_POWERBI_DAX.md`) look flat means old cases are stuck, not
  that intake is outpacing completion — a different intervention (case triage, not
  capacity).
- Without a persisted snapshot, this page can only ever say "here is today's backlog
  shape" — not "here is how it's trending." Say so explicitly on the report page if that
  distinction matters to viewers.
