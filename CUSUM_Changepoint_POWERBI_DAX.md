# CUSUM_Changepoint.py — Power BI and DAX

**Målgruppe:** Analytiker-rapport — hvorfor signalet utløste akkurat denne måneden, og
ikke forrige, krever å forstå den ankrede baseline-metoden.

## Formål
Skille mellom tidlig driftssignal (CUSUM) og bekreftet strukturelt skift (PELT), med drill-down for årsaksforklaring.

Datakilder:
- `analyser.cusum_analyse` — signal/score only; `signal` and `cusum_positiv`/`cusum_negativ`
  are genuinely not reproducible in DAX (the recursive `max(0, S_{t-1} + ...)` reset at
  each step is the same class of sequential logic that made the throughput monitor's
  streak measure fragile — not something to fake in a measure). The raw underlying value
  (Fristprosent/Behandlingstid/Produksjonsdifferanse) is deliberately **not** stored here
  — it's a live DAX measure against `Faser`, defined elsewhere in the model.
- `analyser.pelt_analyse`
- `analyser.pelt_analyse_detaljer`

## Visualforslag

### 1) CUSUM-linje (tidlig varsling)
- X-akse: `analyse_dato`
- Linjer: `cusum_positiv`, `cusum_negativ` (from `cusum_analyse`) — optionally add the raw
  value as a third line from the live DAX measure above, joined on `indikator` +
  `analyse_dato`
- Referanselinje: terskel (`CUSUM_H`)
- Filter: `indikator`, `maaltall`, `granularitet`

### 2) Aktiv signal-tabell
- Kolonner: `indikator`, `maaltall`, `granularitet`, `signalretning`, `analyse_dato`
- Filter: `signal = TRUE`

### 3) PELT-endringspunkter (bekreftede skift)
- Kolonner: `indikator`, `maaltall`, `analyse_dato`, `gjennomsnitt_foer`, `gjennomsnitt_etter`, `endringsstoerrelse`, `endringsretning`
- Sortering: `ABS(endringsstoerrelse)` desc

### 4) Drill-down etter endringspunkt
- Relasjon: `pelt_analyse.endringspunkt_id` -> `pelt_analyse_detaljer.endringspunkt_id`
- Vis bidrag per `dimensjonsverdi` med volumfelt.

## DAX-forslag

```DAX
Siste CUSUM-dato =
CALCULATE(
    MAX(cusum_analyse[analyse_dato]),
    ALLEXCEPT(
        cusum_analyse,
        cusum_analyse[indikator],
        cusum_analyse[maaltall],
        cusum_analyse[granularitet]
    )
)
```

```DAX
Har aktiv CUSUM signal =
CALCULATE(
    MAX(cusum_analyse[signal]),
    cusum_analyse[analyse_dato] = [Siste CUSUM-dato]
) = TRUE()
```

`signalretning` is written as `Stigende`/`Synkende`/`Stabil` — plain board vocabulary,
precisely so a board-level trend card can read it with no translation layer:

```DAX
Trendretning (CUSUM) =
CALCULATE(
    SELECTEDVALUE(cusum_analyse[signalretning]),
    cusum_analyse[analyse_dato] = [Siste CUSUM-dato]
)
```

This is the one to put on a board card instead of the CUSUM line chart: the real,
statistically-tested signal (anchored-baseline CUSUM, not a rolling self-referential
threshold), as one plain word — no chart, no `cusum_positiv`/`cusum_negativ` numbers.

```DAX
Antall aktive signaler =
COUNTROWS(
    FILTER(
        VALUES(cusum_analyse[indikator]),
        CALCULATE([Har aktiv CUSUM signal])
    )
)
```

```DAX
CUSUM bekreftet av PELT =
VAR GjeldendeIndikator = SELECTEDVALUE(cusum_analyse[indikator])
VAR GjeldendeMaaltall = SELECTEDVALUE(cusum_analyse[maaltall])
VAR GjeldendeGranularitet = SELECTEDVALUE(cusum_analyse[granularitet])
RETURN
    CALCULATE(
        COUNTROWS(pelt_analyse),
        pelt_analyse[indikator] = GjeldendeIndikator,
        pelt_analyse[maaltall] = GjeldendeMaaltall,
        pelt_analyse[granularitet] = GjeldendeGranularitet,
        REMOVEFILTERS(pelt_analyse)
    ) > 0
```

```DAX
Siste endringspunkt dato =
CALCULATE(
    MAX(pelt_analyse[analyse_dato]),
    ALLEXCEPT(
        pelt_analyse,
        pelt_analyse[indikator],
        pelt_analyse[maaltall],
        pelt_analyse[granularitet]
    )
)
```

```DAX
Endringspunkt størrelse =
CALCULATE(
    MAX(pelt_analyse[endringsstoerrelse]),
    pelt_analyse[analyse_dato] = [Siste endringspunkt dato]
)
```

## Tolkning
- CUSUM = tidlig driftssignal.
- PELT = bekreftet skift med tidspunkt og størrelse.
- `pelt_analyse_detaljer` svarer på hvor skiftet kommer fra (enhet/fasetittel).
