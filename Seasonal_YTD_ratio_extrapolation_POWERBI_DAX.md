# Seasonal_YTD_ratio_extrapolation.py — Power BI and DAX

## Formål
Vise årssluttprognose for fristprosent med usikkerhetsintervall basert på sesongmønster.

Datakilde:
- `prognoser.frist_prognose`

## Visualforslag

### 1) Linjediagram: faktisk + prognose
- X-akse: `analyse_dato` (jan-des inneværende år)
- Hel linje: `type = 'Faktisk'`, `verdi`
- Stiplet linje: `type = 'Prognose'`, `verdi`
- Bånd: `nedre_konfidensgrense` til `oevre_konfidensgrense` for prognosedelen

### 2) KPI-kort
- `Prognose årslutt`
- `Prognose CI lower`
- `Prognose CI upper`
- Sammenlign mot målverdi (`alert_config`)

### 3) Oppsummeringstabell
- `indikator`, `verdi_hittil`, `prognose_aarsslutt`, `nedre_konfidensgrense`, `oevre_konfidensgrense`
- Sortering: `prognose_aarsslutt` asc

## DAX-forslag

```DAX
Prognose årslutt =
CALCULATE(
    MAX(frist_prognose[prognose_aarsslutt]),
    frist_prognose[type] = "Prognose",
    frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
)
```

```DAX
Prognose CI lower =
CALCULATE(
    MAX(frist_prognose[nedre_konfidensgrense]),
    frist_prognose[type] = "Prognose",
    frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
)
```

```DAX
Prognose CI upper =
CALCULATE(
    MAX(frist_prognose[oevre_konfidensgrense]),
    frist_prognose[type] = "Prognose",
    frist_prognose[analyse_dato] = MAX(frist_prognose[analyse_dato])
)
```

```DAX
Prognose RAG =
VAR Prognose = [Prognose årslutt]
VAR Mål =
    CALCULATE(
        MIN(alert_config[terskel_amber]),
        alert_config[indikator] = MAX(frist_prognose[indikator]),
        alert_config[aktiv] = TRUE()
    )
RETURN
    IF(
        ISBLANK(Prognose) || ISBLANK(Mål),
        BLANK(),
        IF(Prognose >= Mål, 3, IF(Prognose >= Mål * 0.95, 2, 1))
    )
```

## Tolkning
- Prognose under mål = risiko.
- Konfidensbånd som krysser mållinjen = usikker måloppnåelse.
- Når flere måneder blir faktiske, skal båndene normalt bli smalere.
