# Inflight_SLA_Risk_Monitor.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Hvor mange åpne saker er i ferd med å sprenge fristen, før de lukkes?
- Hvilke indikatorer/enheter har flest saker i `Kritisk` eller `Bruddet`?

This page is split in two, deliberately: **today's per-case risk list** is computed live
in DAX straight from the fact table (no nightly wait), and **the risk-mix trend over time**
comes from `sak_frist_risiko_trend`, the one thing a live measure structurally cannot
reconstruct — `Risikoklasse` depends on `TODAY()`, so it can only ever tell you about
today, never about last week. `Inflight_SLA_Risk_Monitor.py`'s only remaining job is
writing that daily mix down.

## Del 1 — Dagens saksliste (live DAX, ingen ventetid på nattlig kjøring)

### Antagelser (rename to match your model)
- `Faser` — `saksbehandling.faser`, with `pk_faser`, `indikator`, `enhet`, `fasetittel`,
  `startmilepaeldato`, `sluttmilepaeldato`, `frist_dager`
- Same fagomraade/`%avtalt%` filtering assumption as the rest of this repo's DAX docs
- Open assumption carried over from the script, still unverified: whether `frist_dager`
  is reliably populated on rows that haven't closed yet

### Beregnede kolonner — klassifisering

These exactly reproduce `classify_risk()`'s boundary behavior (see the script — it's kept
as the tested spec both this DAX and the trend script's SQL are derived from), including
its two non-obvious rules: `DagerIgjen < 0` wins outright regardless of `AndelBrukt`, and
a blank `AndelBrukt` with a non-negative `DagerIgjen` falls through to `"Innenfor"` rather
than blank.

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

### Mål (measures)

```DAX
Åpne saker i risiko (antall) =
CALCULATE(COUNTROWS(Faser), NOT ISBLANK(Faser[Risikoklasse]))
```

```DAX
Andel i risiko (nå) =
VAR Total = [Åpne saker i risiko (antall)]
VAR Utsatt = CALCULATE([Åpne saker i risiko (antall)], Faser[Risikoklasse] IN {"Bruddet", "Kritisk"})
RETURN
    DIVIDE(Utsatt, Total)
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

### Visual — daglig intervensjonsliste
- `indikator`, `enhet`, `fasetittel`, `DagerIgjen`, `AndelBrukt`, `Risikoklasse`
- Filter: `Risikoklasse IN ("Bruddet", "Kritisk")`
- Sortering: `DagerIgjen` stigende

### Visual — forfaller neste 14 dager
A separate list from the one above, on purpose: this filters on calendar proximity
(`DagerIgjen`) rather than `% AndelBrukt`, so it catches a brand-new case with a short
`frist_dager` before it's used enough of its own time to trip `Kritisk`/`Bruddet` — a case
that started today with a 13-day deadline is `DagerIgjen = 13`, `AndelBrukt = 0%`
(`Innenfor`), invisible to the list above until much closer to the deadline.
- `indikator`, `enhet`, `fasetittel`, `DagerIgjen`, `Risikoklasse`
- Filter: `NOT ISBLANK(DagerIgjen) && DagerIgjen >= 0 && DagerIgjen <= 14`
- Sortering: `DagerIgjen` stigende

## Del 2 — Risikotrend over tid (persistert snapshot)

Datakilde: `analyser.sak_frist_risiko_trend` (written nightly by
`Inflight_SLA_Risk_Monitor.py` — see that script's header for why this can't be live DAX).

### Visualforslag
- X-akse: `snapshot_dato`
- Y-akse: stablet areal — `andel_bruddet`, `andel_kritisk`, `andel_risiko`, `andel_innenfor`
- Filter: `tilstrekkelig_volum = TRUE`
- Slicer: `indikator`, `enhet`

### DAX-forslag

```DAX
Andel i risiko (siste snapshot) =
CALCULATE(
    DIVIDE(
        SUM(sak_frist_risiko_trend[antall_bruddet]) + SUM(sak_frist_risiko_trend[antall_kritisk]),
        SUM(sak_frist_risiko_trend[antall_totalt])
    ),
    sak_frist_risiko_trend[snapshot_dato] = MAX(sak_frist_risiko_trend[snapshot_dato])
)
```

## Slicer-oppsett
- `indikator`
- `enhet`
- `Risikoklasse` (Del 1 only)
- `snapshot_dato` (Del 2 only — Del 1 is always "now")

## Tolkning
- Dette er en *leading* indikator — sammenlign med `cusum_analyse` sin `Fristprosent`, som
  kun måler lukkede saker.
- `AndelBrukt = BLANK()` betyr `frist_dager <= 0` for den saken — sjekk datakvalitet før
  den saken brukes i tiltak.
- Del 1 and Del 2 should roughly agree on "today" (Del 2's latest snapshot vs. Del 1 live)
  — if they diverge, the nightly run is stale or the live filters have drifted from the
  script's filters. Keep both filter sets in sync by hand; there's no single source of
  truth to enforce it automatically.
