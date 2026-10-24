# Phase_Bottleneck_Detector.py — Power BI and DAX

**Målgruppe:** Leder-rapport — `alvorlighet` pluss `arsak_tekst` er en fullstendig
forklarende setning, ikke bare en kode; leser kaldt.

## Formål
Denne siden skal svare på:
- Hvilke faser er reelle flaskehalser per enhet og indikator?
- Skyldes flaskehalsen inngang/utgang-ubalanse, p90-forverring, eller begge?
- Hvilke faser bør prioriteres først i neste forbedringsmøte?

Datakilde:
- `analyser.fase_flaskehals_enhet`

## Hva tallene betyr
- `inngang_antall` — faser som **startet** i måneden (etter `startmilepaeldato`).
- `utgang_antall` — faser som **ble avsluttet** i måneden (etter `sluttmilepaeldato`).
  `median_fasetid`/`p90_fasetid` er beregnet på disse avsluttede fasene.
- `netto_flyt` = `inngang_antall - utgang_antall` for måneden. Positiv = køen vokste.
- `fase_ko_proxy` — antall **åpne faser ved månedsslutt**: faser som allerede var åpne
  ved seriestart (`START_YEAR`) pluss kumulativ netto flyt siden da.
- `KO_VEKST` i `arsak_kode` betyr at køen ved månedsslutt er mer enn 10 %
  (`KO_VEKST_TERSKEL`) over sin egen median for de siste 12 månedene, og at netto flyt
  var positiv i måneden.
- Måneder uten aktivitet i en fase får en rad med 0 inn og 0 ut, slik at køen og
  baseline følger kalendermåneder.

> **Endret oktober 2026:** tidligere ble både inngang og utgang talt i måneden for
> `COALESCE(sluttmilepaeldato, startmilepaeldato)`. En avsluttet fase ga da +1 og −1 i
> samme måned og ble null, så «netto flyt» var egentlig bare antall fortsatt åpne faser
> etter startmåned — nær 0 i all historikk og positiv de siste 1–2 månedene. Det ga
> mange falske `bottleneck_flagg` i de siste månedene.

### Kontrollspørring etter utrulling (kjør i Fabric)
```sql
SELECT analyse_dato, COUNT_IF(bottleneck_flagg) AS antall_flaskehalser
FROM analyser.fase_flaskehals_enhet
GROUP BY 1
ORDER BY 1 DESC
```
Antall flaskehalser skal **ikke** lenger hope seg opp i de to siste månedene; det skal
ligge på omtrent samme nivå som tidligere måneder.

## Anbefalte visualer

### 1) KPI-kort
- Antall rader med `bottleneck_flagg = TRUE` siste måned
- Antall enheter med minst én flaskehals siste måned
- Gjennomsnitt `avvik_p90_pct` for flaskehalser siste måned

### 2) Flaskehals-matrise (hovedvisual)
- Rader: `enhet`
- Kolonner: `fasetittel`
- Verdi: `alvorlighet` eller `avvik_p90_pct`
- Filter: `bottleneck_flagg = TRUE` og siste `analyse_dato`

### 3) Prioritert tiltakstabell
- `enhet`, `indikator`, `fasetittel`, `alvorlighet`, `arsak_kode`, `inngang_antall`, `utgang_antall`, `avvik_p90_pct`
- Sortering: `alvorlighet` (Kritisk først), deretter `avvik_p90_pct` desc

### 4) Trend per fase
- X-akse: `analyse_dato`
- Linjer: `p90_fasetid` og `baseline_p90`
- Små multipler: `fasetittel` eller `enhet`

## DAX-forslag

### Siste analysedato
Tabellen inneholder bare **hele måneder**; inneværende måned skrives ikke før den er
ferdig, så `[Siste analysedato flaskehals]` er alltid slutten av forrige måned.
```DAX
Siste analysedato flaskehals =
CALCULATE(
    MAX(fase_flaskehals_enhet[analyse_dato]),
    ALL(fase_flaskehals_enhet[analyse_dato])
)
```

### Antall flaskehalser siste
```DAX
Antall flaskehalser siste =
CALCULATE(
    COUNTROWS(fase_flaskehals_enhet),
    FILTER(
        fase_flaskehals_enhet,
        fase_flaskehals_enhet[analyse_dato] = [Siste analysedato flaskehals]
            && fase_flaskehals_enhet[bottleneck_flagg] = TRUE()
    )
)
```

### Antall berørte enheter siste
```DAX
Antall berørte enheter siste =
CALCULATE(
    DISTINCTCOUNT(fase_flaskehals_enhet[enhet]),
    FILTER(
        fase_flaskehals_enhet,
        fase_flaskehals_enhet[analyse_dato] = [Siste analysedato flaskehals]
            && fase_flaskehals_enhet[bottleneck_flagg] = TRUE()
    )
)
```

### Gjennomsnitt p90-avvik siste
```DAX
Gjennomsnitt p90-avvik siste =
CALCULATE(
    AVERAGE(fase_flaskehals_enhet[avvik_p90_pct]) * 100,
    FILTER(
        fase_flaskehals_enhet,
        fase_flaskehals_enhet[analyse_dato] = [Siste analysedato flaskehals]
            && fase_flaskehals_enhet[bottleneck_flagg] = TRUE()
    )
)
```

### Alvorlighet sortering
```DAX
Alvorlighet rang =
SWITCH(
    MAX(fase_flaskehals_enhet[alvorlighet]),
    "Kritisk", 4,
    "Hoy", 3,
    "Moderat", 2,
    "Lav", 1,
    0
)
```

### Flaskehals fargekode
```DAX
Flaskehals fargekode =
SWITCH(
    MAX(fase_flaskehals_enhet[alvorlighet]),
    "Kritisk", "#B00020",
    "Hoy", "#E65100",
    "Moderat", "#F9A825",
    "Lav", "#2E7D32",
    "#757575"
)
```

## Slicer-oppsett
- `analyse_dato` (default: siste)
- `indikator`
- `enhet`
- `fasetittel`
- `alvorlighet`

## Tolkning
- Høy `avvik_p90_pct` + `KO_VEKST` i `arsak_kode` indikerer sann prosessflaskehals.
- `fase_ko_proxy` er et ekte antall åpne faser, men faser som har stått åpne svært
  lenge (f.eks. glemte registreringer fra før 2015) er med. Et høyt, flatt nivå kan
  være datakvalitet, ikke kø — se på `netto_flyt` og `KO_VEKST` for endringen.
- `tilstrekkelig_volum = FALSE` bør ikke brukes som styrende tiltakssignal.
