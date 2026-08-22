# EWMA.py — Power BI and DAX

## Formål
Bruk EWMA-tabellene til å vise underliggende trend i fristprosent, behandlingstid og produksjonsdifferanse.

Datakilder:
- `analyser.ewma_analyse`
- `analyser.ewma_analyse_wide`

## Visualforslag

### 1) Linjediagram: råverdi + EWMA-trend
- X-akse: `analyse_dato`
- Serie 1: `verdi` (rå månedlig verdi)
- Serie 2: `ewma_sakte` (stabil trend)
- Serie 3: `ewma_rask` (tidlig signal)
- Filter: `maaltall` og `indikator`

### 2) Trendkort siste periode
- Vis `trendretning` for siste `analyse_dato`
- Farger:
  - `Stigende` grønn
  - `Synkende` rød
  - `Stabil` nøytral

### 3) Enhetlig join mot faser
- Bruk `analyser.ewma_analyse_wide` når trendfelt skal joins direkte mot faser uten fan-out.

## DAX-forslag

```DAX
EWMA Sakte fristprosent =
CALCULATE(
    MAX(ewma_analyse[ewma_sakte]),
    ewma_analyse[maaltall] = "Fristprosent"
)
```

```DAX
EWMA Rask fristprosent =
CALCULATE(
    MAX(ewma_analyse[ewma_rask]),
    ewma_analyse[maaltall] = "Fristprosent"
)
```

```DAX
EWMA Trend retning =
CALCULATE(
    MAX(ewma_analyse[trendretning]),
    ewma_analyse[analyse_dato] = MAX(ewma_analyse[analyse_dato])
)
```

```DAX
EWMA Trend verdi =
VAR Retning = [EWMA Trend retning]
RETURN
    SWITCH(Retning, "Stigende", 1, "Synkende", -1, 0)
```

```DAX
EWMA Behandlingstid =
CALCULATE(
    MAX(ewma_analyse[ewma_sakte]),
    ewma_analyse[maaltall] = "Behandlingstid"
)
```

```DAX
EWMA Produksjon differanse =
CALCULATE(
    MAX(ewma_analyse[ewma_sakte]),
    ewma_analyse[maaltall] = "Produksjonsdifferanse"
)
```

## Tolkning
- Styrevisning: bruk `ewma_sakte` for robust retning.
- Virksomhetsoppfølging: bruk `ewma_rask` for tidligere varsling.
- Stor avstand mellom råverdi og EWMA signaliserer høy volatilitet.
