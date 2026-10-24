# Pause_Reconstruction.py — Power BI and DAX

**Målgruppe:**
- **Leder-rapport:** trendlinjene «hos oss», «i faresonen» og «venter på søker», og
  implisitt behandlingstid mot frist. Antall og dager, ingen metode å forklare.
- **Analytiker-rapport:** dager til første anmodning, kontrollmålene.

## Formål
`At_Risk_Open_Cases_POWERBI_DAX.md` viser **nå**: hvor mange faser er hos oss, i faresonen,
på vent. Denne siden viser **utviklingen**, uke for uke tilbake til 2015, rekonstruert fra
milepælene for anmodning/mottak av tilleggsdokumentasjon. Spørsmålene:
- Vokser køen hos oss, eller tømmes den?
- Øker antallet i faresonen — før det viser seg i `Fristprosent`?
- Kommer veksten fra oss (hos oss) eller fra søkere (på vent)?
- Sendes anmodninger tidlig, eller blir saker liggende før fristen stoppes?

Datakilder (overskrives hver natt, bygges fra kilden — ingen snapshot):
- `analyser.flyt_ukentlig` — én rad per ISO-uke × enhet × indikator
- `analyser.fase_klokke` — én rad per fase

**Bare hele uker.** `uke_slutt` er søndagen i uken; inneværende uke er ikke med.

## Datamodell
- Bruk `flyt_ukentlig[uke_slutt]` direkte på x-aksen, **uten** relasjon til `Kalender`.
  Målene under filtrerer på den kolonnen selv (siste uke i utvalget, de fire siste ukene),
  og en datotabell-relasjon ville overstyre det.
- Relater `flyt_ukentlig[indikator]` og `fase_klokke[indikator]` til indikatortabellen
  (og `enhet` til enhetsdimensjonen hvis den finnes), slik at de samme slicerne virker som
  på resten av rapporten.

## Mål — ukentlig flyt (`flyt_ukentlig`)

**Beholdninger** (hos oss, venter, i faresonen, over frist) er *antall på et tidspunkt*.
De skal ikke summeres over uker. Ta siste uke i utvalget, slik at en måned på aksen viser
månedens siste uke:

```DAX
Siste uke (flyt) =
MAX(flyt_ukentlig[uke_slutt])
```

```DAX
Hos oss (uke) =
VAR U = [Siste uke (flyt)]
RETURN CALCULATE(SUM(flyt_ukentlig[hos_oss]), flyt_ukentlig[uke_slutt] = U)
```

```DAX
Venter på søker (uke) =
VAR U = [Siste uke (flyt)]
RETURN CALCULATE(SUM(flyt_ukentlig[venter_paa_soeker]), flyt_ukentlig[uke_slutt] = U)
```

```DAX
I faresonen (uke) =
VAR U = [Siste uke (flyt)]
RETURN CALCULATE(SUM(flyt_ukentlig[i_faresonen]), flyt_ukentlig[uke_slutt] = U)
```

```DAX
Over frist (uke) =
VAR U = [Siste uke (flyt)]
RETURN CALCULATE(SUM(flyt_ukentlig[over_frist]), flyt_ukentlig[uke_slutt] = U)
```

```DAX
Andel som venter (uke) =
DIVIDE([Venter på søker (uke)], [Venter på søker (uke)] + [Hos oss (uke)])
```

```DAX
Andel hos oss i faresonen (uke) =
DIVIDE([I faresonen (uke)], [Hos oss (uke)])
```

**Strømmer** (mottatt, ferdigstilt, trukket, satt på vent) er *antall i perioden* og
summeres vanlig:

```DAX
Mottatt (flyt) = SUM(flyt_ukentlig[mottatt])
Ferdigstilt (flyt) = SUM(flyt_ukentlig[ferdigstilt])
Trukket (flyt) = SUM(flyt_ukentlig[trukket])
Satt på vent (flyt) = SUM(flyt_ukentlig[satt_paa_vent])
```

`Ferdigstilt` teller ikke faser som ble trukket/utgått — de står i `Trukket`, så tempoet
ikke ser bedre ut enn det er.

### Implisitt behandlingstid (Littles lov)

Hvor mange dager ville det ta å tømme køen hos oss i tempoet fra de siste fire ukene?
Sammenlign med typisk frist: når linjen går over fristen, kan køen ikke tømmes i tide, og
`Fristprosent` faller ukene etter.

```DAX
Implisitt behandlingstid (dager, uke) =
VAR U = [Siste uke (flyt)]
VAR Ferdig4 =
    CALCULATE(
        SUM(flyt_ukentlig[ferdigstilt]),
        flyt_ukentlig[uke_slutt] > U - 28,
        flyt_ukentlig[uke_slutt] <= U
    )
RETURN DIVIDE([Hos oss (uke)], Ferdig4) * 28
```

```DAX
Typisk frist (dager) =
CALCULATE(MEDIAN(fase_klokke[frist_dager]), fase_klokke[tilstand] = "Hos oss")
```

## Mål — per fase (`fase_klokke`)

```DAX
Dager til første anmodning (median) =
CALCULATE(
    MEDIAN(fase_klokke[dager_til_foerste_anmodning]),
    fase_klokke[startdato] >= EDATE(TODAY(), -12)
)
```

```DAX
Andel faser med anmodning =
DIVIDE(
    CALCULATE(COUNTROWS(fase_klokke), fase_klokke[antall_pauser] > 0),
    COUNTROWS(fase_klokke)
)
```

```DAX
Pausedager per fase (snitt) =
AVERAGE(fase_klokke[pause_dager])
```

### Kontroll — stemmer rekonstruksjonen?

```DAX
Klokke stemmer med tidsbruk (andel) =
DIVIDE(
    COUNTROWS(FILTER(fase_klokke, ABS(fase_klokke[avvik_mot_tidsbruk]) <= 2)),
    COUNTROWS(FILTER(fase_klokke, NOT ISBLANK(fase_klokke[avvik_mot_tidsbruk])))
)
```

Vis denne per indikator på analytikersiden. Høy andel (f.eks. over 90 %) betyr at
milepælene og pauseregelen gjenspeiler saksbehandlingssystemets egen klokke, og at
historikken i `flyt_ukentlig` kan brukes. Lav andel for en indikator betyr at noe mangler
for den (andre milepælnavn, venting på ekstern part, annen klokkeregel). Da er historikken
for den indikatoren omtrentlig. Dagens tall på `At_Risk_Open_Cases` påvirkes ikke, for de
leser systemets `tidsbruk` direkte.

## Visualforslag

### Leder
- **Linje:** `[Hos oss (uke)]`, `[I faresonen (uke)]` og `[Venter på søker (uke)]` per
  `uke_slutt`, siste 52 uker
- **Linje:** `[Implisitt behandlingstid (dager, uke)]` med `[Typisk frist (dager)]` som
  konstant referanselinje
- **Stolper:** `[Mottatt (flyt)]` mot `[Ferdigstilt (flyt)]` per måned

### Analytiker
- **Stolper:** `[Dager til første anmodning (median)]` per `enhet`
- **Linje:** `[Andel som venter (uke)]` per `enhet`
- **Tabell:** `[Klokke stemmer med tidsbruk (andel)]` per `indikator`

## Tolkning
- **«Hos oss» vokser mens «venter» er flat:** vi henger etter. Det er et kapasitetsspørsmål.
- **«Venter» vokser mens «hos oss» er flat:** søknadene er svakere, eller fristen stoppes
  oftere. Se på andel med anmodning og dager til første anmodning.
- **«I faresonen» stiger før `Fristprosent` faller:** dette er varselet. Avstanden i uker
  mellom de to er ledetiden siden gir.
- **Dager til første anmodning øker for en enhet:** anmodningen sendes senere i
  behandlingen. Enten oppdages mangler sent, eller saken har ligget før den ble åpnet. Det
  er verdt en samtale, ikke en konklusjon.
- Faser som venter på **ekstern part** (høring, andre etater) rekonstrueres ikke fra disse
  milepælene. De telles som «hos oss» i historikken. `Status` viser dem riktig i dag.
