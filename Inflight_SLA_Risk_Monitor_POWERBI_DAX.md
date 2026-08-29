# In-flight SLA risk monitor — native DAX (replaces Inflight_SLA_Risk_Monitor.py)

## Formål
Denne siden skal svare på:
- Hvor mange åpne saker er i ferd med å sprenge fristen, før de lukkes?
- Hvilke indikatorer/enheter har flest saker i `Kritisk` eller `Bruddet`?

`Fristprosent` (CUSUM) only scores cases that have already closed — a lagging indicator.
This page scores cases that are **still open** against their own `frist_dager`, so a
breach wave shows up here before it ever reaches the closed-case ratio.

**No Delta table, no nightly run.** Of the whole repo, this was the easiest conversion:
per-case risk classification is pure row-level arithmetic and threshold comparisons — no
baseline, no rolling window, no sequence logic at all — against the fact table
(`saksbehandling.faser`, referred to below as `Faser`).

## Antagelser (rename to match your model)
- `Faser` — `saksbehandling.faser`, with `pk_faser`, `indikator`, `enhet`, `fasetittel`,
  `startmilepaeldato`, `sluttmilepaeldato`, `frist_dager`
- Same fagomraade/`%avtalt%` filtering assumption as the rest of this repo's DAX docs
- Open assumption carried over from the old script, still unverified: whether
  `frist_dager` is reliably populated on rows that haven't closed yet. Rows missing it are
  excluded below (`BLANK()`), not defaulted — if it turns out sparse on open rows, this
  under-covers the open-case population, same risk the old script flagged.

## Beregnede kolonner — klassifisering

These exactly reproduce `classify_risk()`'s boundary behavior, including its two
non-obvious rules: `DagerIgjen < 0` wins outright regardless of `AndelBrukt`, and a blank
`AndelBrukt` with a non-negative `DagerIgjen` falls through to `"Innenfor"` rather than
blank.

```DAX
Faser[DagerForlopt] =
IF(
    ISBLANK(Faser[startmilepaeldato])
        || NOT ISBLANK(Faser[sluttmilepaeldato])
        || ISBLANK(Faser[frist_dager]),
    BLANK(),
    DATEDIFF(Faser[startmilepaeldato], TODAY(), DAY)
)
```

```DAX
Faser[AndelBrukt] =
IF(
    ISBLANK(Faser[DagerForlopt]) || Faser[frist_dager] <= 0,
    BLANK(),
    DIVIDE(Faser[DagerForlopt], Faser[frist_dager])
)
```

```DAX
Faser[DagerIgjen] =
IF(ISBLANK(Faser[DagerForlopt]), BLANK(), ROUND(Faser[frist_dager] - Faser[DagerForlopt], 0))
```

```DAX
Faser[Risikoklasse] =
VAR Igjen = Faser[DagerIgjen]
VAR Andel = Faser[AndelBrukt]
RETURN
    SWITCH(
        TRUE(),
        ISBLANK(Igjen), BLANK(),
        Igjen < 0, "Bruddet",
        NOT ISBLANK(Andel) && Andel >= 0.90, "Kritisk",  -- RISK_THRESHOLD_KRITISK
        NOT ISBLANK(Andel) && Andel >= 0.75, "Risiko",   -- RISK_THRESHOLD_RISIKO
        "Innenfor"
    )
```

```DAX
Faser[RisikoklasseRang] =
SWITCH(Faser[Risikoklasse], "Bruddet", 4, "Kritisk", 3, "Risiko", 2, "Innenfor", 1, 0)
```
(Sort-by column for `Risikoklasse` — chronological severity order, not alphabetical.)

## Mål (measures)

```DAX
Åpne saker i risiko (antall) =
CALCULATE(COUNTROWS(Faser), NOT ISBLANK(Faser[Risikoklasse]))
```

```DAX
Andel i risiko siste =
VAR Total = [Åpne saker i risiko (antall)]
VAR Utsatt = CALCULATE([Åpne saker i risiko (antall)], Faser[Risikoklasse] IN {"Bruddet", "Kritisk"})
RETURN
    DIVIDE(Utsatt, Total)
```

```DAX
Tilstrekkelig volum (risiko) =
[Åpne saker i risiko (antall)] >= 10  -- MIN_TEAM_VOLUME
```

```DAX
Risikoklasse fargekode =
SWITCH(
    Faser[Risikoklasse],
    "Bruddet", "#B00020",
    "Kritisk", "#E65100",
    "Risiko", "#F9A825",
    "Innenfor", "#2E7D32",
    "#757575"
)
```

## Viktig begrensning: ingen historisk risikotrend uten et snapshot
The old script's `sak_frist_risiko_trend` table captured the daily risk-class mix, letting
you chart whether the risk distribution was worsening over time. That's the same
point-in-time problem `Backlog_Aging_Distribution` had: `Risikoklasse` depends on `TODAY()`,
so a live measure only ever knows today's mix, not what it was last month. This page can
answer "what does today's open-case risk look like" but not "is it trending worse" —
that still needs something persisting a daily snapshot if it matters.

## Anbefalte visualer

### 1) KPI-kort
- Antall saker `Risikoklasse = "Bruddet"`
- Antall saker `Risikoklasse = "Kritisk"`
- `[Andel i risiko siste]`

### 2) Risikotabell — daglig intervensjonsliste
- `indikator`, `enhet`, `fasetittel`, `DagerIgjen`, `AndelBrukt`, `Risikoklasse`
- Filter: `Risikoklasse IN ("Bruddet", "Kritisk")`
- Sortering: `DagerIgjen` stigende

(The old trend-over-time visual is dropped here — see the limitation above.)

## Slicer-oppsett
- `indikator`
- `enhet`
- `Risikoklasse`

## Tolkning
- Dette er en *leading* indikator — sammenlign med `cusum_analyse` sin `Fristprosent`, som
  kun måler lukkede saker.
- `AndelBrukt = BLANK()` betyr `frist_dager <= 0` for den saken — sjekk datakvalitet før
  den saken brukes i tiltak.
