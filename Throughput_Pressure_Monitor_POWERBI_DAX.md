# Throughput_Pressure_Monitor.py — Power BI and DAX

**Målgruppe:** Analytiker-rapport — `pressure_nivaa` leser greit alene, men den
femfaktors-vektede poengsummen bak den gjør det ikke. (Søstersiden
`Phase_Bottleneck_Detector` er i leder-rapporten fordi den har `arsak_tekst` — denne
tabellen har ikke et tilsvarende forklarende tekstfelt.)

## Formål
Denne siden skal svare på:
- Hvor er gjennomstrømmingspresset størst akkurat nå?
- Skyldes presset netto innflyt, økt tidsbruk, eller begge?
- Hvilken fase bidrar mest i pressede enheter?

Datakilder:
- `analyser.gjennomstroemming_press_enhet`
- `analyser.gjennomstroemming_press_fase`

`mottatt_antall` telles i måneden for `startmilepaeldato`, `ferdigstilt_antall` i måneden
for `sluttmilepaeldato`, og `netto_flyt = mottatt_antall - ferdigstilt_antall` — både på
enhetsnivå og (fra oktober 2026) på fasenivå i `gjennomstroemming_press_fase`. Før det ble
fasetabellen gruppert på `COALESCE(slutt, start)`, slik at avsluttede faser nullet seg ut
og bare fortsatt åpne faser ga positiv netto flyt.

### Når regnes netto flyt som press? (endret oktober 2026)
- `netto_flyt_saesong_baseline` — median `netto_flyt` i **samme kalendermåned** de tre
  foregående årene (tom før det finnes to slike år). Januar og høsten tar normalt inn
  mer enn de ferdigstiller; det er ikke press.
- `netto_flyt_rel` = (`netto_flyt` − sesongbaseline) / median `ferdigstilt_antall` siste
  12 måneder. +1 i et team som ferdigstiller 200 i måneden er støy; +40 er 20 %.
- Netto flyt gir poeng i `pressure_score` bare når `netto_flyt_rel` > 0,10
  (`NETTO_FLYT_REL_TERSKEL`). `netto_flyt_streak` teller måneder **på rad** med slikt
  press — måneder uten aktivitet er egne rader (0 inn, 0 ut) og bryter rekken.
- Tidligere ga enhver `netto_flyt > 0` poeng, og rekken hoppet over måneder uten rader.

## Anbefalte visualer

### 1) KPI-kort (øverst)
- Antall enheter med `pressure_nivaa` = `Kritisk` (bare `tilstrekkelig_volum`, ikke `Ukjent`)
- Antall enheter med `pressure_nivaa` i (`Hoy`, `Kritisk`) (samme filter)
- Sum `netto_flyt` siste måned
- Median `tidsbruk_avvik_pct` siste måned

### 2) Prioritert tabell (hovedliste)
Vis siste `analyse_dato`:
- `enhet`, `indikator`, `pressure_nivaa`, `pressure_score`, `netto_flyt`, `netto_flyt_rel`, `netto_flyt_streak`, `p90_tidsbruk`, `tidsbruk_avvik_pct`
- Sortering: `pressure_score` desc, `netto_flyt` desc

### 3) Trendlinje per enhet
- X-akse: `analyse_dato`
- Y-akse 1: `netto_flyt`
- Y-akse 2: `p90_tidsbruk` (sekundær akse)
- Små multipler: `enhet`
- Filter: `indikator`

### 4) Fasebidrag for valgt enhet
Fra `analyser.gjennomstroemming_press_fase`:
- Stolper: `fasetittel`
- Verdi: `fase_bidrag_score`
- Farge: `fase_press_flagg`
- Filter: siste `analyse_dato`

## Datamodell
- Relasjon mellom tabeller:
  - Nøkkel: `indikator`, `enhet`, `analyse_dato` (modell med sammensatt nøkkel i Power Query eller ved hjelp av sammenslått tekstnøkkel)
- Foreslått sammenslått nøkkel i begge tabeller:
  - `nokkel_enhet_dato = enhet & "|" & indikator & "|" & FORMAT(analyse_dato, "yyyy-MM-dd")`

## DAX-forslag

### Siste dato i kontekst
Tabellen inneholder bare **hele måneder** — inneværende måned skrives ikke før den er
ferdig. `[Siste analysedato]` er derfor alltid slutten av forrige måned, ikke en halv
måned som ser ut som et fall i ferdigstilte saker.
```DAX
Siste analysedato =
CALCULATE(
    MAX(gjennomstroemming_press_enhet[analyse_dato]),
    ALL(gjennomstroemming_press_enhet[analyse_dato])
)
```

### Er siste rad
```DAX
Er siste rad =
VAR SisteDato = [Siste analysedato]
RETURN IF(MAX(gjennomstroemming_press_enhet[analyse_dato]) = SisteDato, 1, 0)
```

### Antall kritiske enheter (siste måned)
```DAX
Antall kritiske enheter siste =
CALCULATE(
    DISTINCTCOUNT(gjennomstroemming_press_enhet[enhet]),
    FILTER(
        gjennomstroemming_press_enhet,
        gjennomstroemming_press_enhet[analyse_dato] = [Siste analysedato]
            && gjennomstroemming_press_enhet[pressure_nivaa] = "Kritisk"
            && gjennomstroemming_press_enhet[tilstrekkelig_volum] = TRUE()
            && gjennomstroemming_press_enhet[enhet] <> "Ukjent"
    )
)
```

Små team (`tilstrekkelig_volum = FALSE`) og faser uten registrert enhet (`Ukjent`) holdes
utenfor KPI-tellingene — de kan få høy poengsum av noen få saker. De står fortsatt i
tabellen og kan vises i detaljlisten.

### Antall høyt press (Hoy + Kritisk)
```DAX
Antall høyt press siste =
CALCULATE(
    DISTINCTCOUNT(gjennomstroemming_press_enhet[enhet]),
    FILTER(
        gjennomstroemming_press_enhet,
        gjennomstroemming_press_enhet[analyse_dato] = [Siste analysedato]
            && gjennomstroemming_press_enhet[pressure_nivaa] IN {"Hoy", "Kritisk"}
            && gjennomstroemming_press_enhet[tilstrekkelig_volum] = TRUE()
            && gjennomstroemming_press_enhet[enhet] <> "Ukjent"
    )
)
```

### Netto flyt siste måned
```DAX
Netto flyt siste =
CALCULATE(
    SUM(gjennomstroemming_press_enhet[netto_flyt]),
    FILTER(
        gjennomstroemming_press_enhet,
        gjennomstroemming_press_enhet[analyse_dato] = [Siste analysedato]
    )
)
```

### Tidsbruksavvik prosentpoeng (siste)
```DAX
Tidsbruksavvik pp siste =
CALCULATE(
    AVERAGE(gjennomstroemming_press_enhet[tidsbruk_avvik_pct]) * 100,
    FILTER(
        gjennomstroemming_press_enhet,
        gjennomstroemming_press_enhet[analyse_dato] = [Siste analysedato]
    )
)
```

### Fargekode for pressnivå
```DAX
Pressnivå fargekode =
SWITCH(
    MAX(gjennomstroemming_press_enhet[pressure_nivaa]),
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
- `pressure_nivaa`

## Tolkning
- `netto_flyt_rel` > 0,10 over flere perioder (`netto_flyt_streak` ≥ 3) + økende `p90_tidsbruk` = sannsynlig kapasitetsproblem.
- Høy `fase_bidrag_score` identifiserer hvor tiltak bør settes inn først.
