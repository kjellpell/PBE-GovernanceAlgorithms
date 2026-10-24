# Åpne faser i faresonen — Power BI and DAX

**Målgruppe:** Leder-rapport — «så mange åpne faser har brukt mer enn tre firedeler av
fristen sin» leser kaldt, uten metodeforklaring.

## Formål
Alle andre signaler i dette repoet (CUSUM, PELT, prognose, flyt, flaskehals) regnes på
**avsluttede** faser. De kommer derfor etter: en fase som kommer til å bryte fristen vises
først når den lukkes. Men den er synlig lenge før — som en åpen fase som eldes. Denne siden
svarer på:
- Hvor mange åpne faser er i ferd med å gå over fristen, akkurat nå?
- Hvor mange har allerede gått over, men er ikke lukket ennå?
- Hvilke enheter og faser har størst andel i faresonen?

Dette er et **ledende** signal. Det er ren live DAX mot `Faser`, uten skript: verdien
avhenger av `TODAY()`, men et øyeblikksbilde trenger ingen historikk (se regelen i
`README.md`). Konsekvens: siden viser bare «nå», ikke hvordan antallet har utviklet seg.
En liten nattlig snapshot-tabell (dato × enhet × indikator × antall) ville gitt en trend,
men er ikke lagt til — det ville delvis gjenopprette det pensjonerte
`Backlog_Aging_Distribution.py`, og ekstra stående infrastruktur er valgt bort.

## Antagelser (rename to match your model)
- `Faser` — `saksbehandling.faser`, med `startmilepaeldato`, `sluttmilepaeldato`,
  `frist_dager`, `enhet`, `fasetittel`, `indikator`
- `frist_dager` er **kalenderdager** (bekreftet). Derfor `DATEDIFF(..., DAY)`. Skulle det
  endres til virkedager, bytt `DATEDIFF(Faser[startmilepaeldato], Idag, DAY)` med
  `NETWORKDAYS(Faser[startmilepaeldato], Idag)` (eventuelt med en helligdagstabell som
  tredje argument).
- Samme omfang som den offisielle `Fristprosent`: fagområde `Byggesak`/`Eiendomssak`/
  `Plansak` (filtrert i modellen), og `%avtalt%`-indikatorer er **med** — en avtalt frist
  er fortsatt en frist.
- En fase regnes som brutt når antall dager siden start er **større enn** `frist_dager`
  (samme som at `tidsbruk <= frist_dager` gir `innenfor_frist = 1`). Kontroller mot
  hvordan `innenfor_frist` faktisk beregnes i datavarehuset; er det `<`, endre `>` til
  `>=` i `Åpne faser over frist`.
- Tellingen er per **fase** (fristen gjelder fasen), ikke per sak.

## Parameter

Lag en what-if-parameter i Power BI (Modellering → Ny parameter → Numerisk område):
navn `Faresone terskel`, fra 0,50 til 1,00, steg 0,05, standard 0,75. Det gir tabellen
`'Faresone terskel'` og et mål; sørg for at målet ser slik ut:

```DAX
Faresone terskel =
SELECTEDVALUE('Faresone terskel'[Faresone terskel], 0.75)
```

0,75 betyr «har brukt minst 75 % av fristen».

## Mål (measures)

```DAX
Åpne faser med frist =
CALCULATE(
    COUNTROWS(Faser),
    NOT ISBLANK(Faser[startmilepaeldato]),
    ISBLANK(Faser[sluttmilepaeldato]),
    NOT ISBLANK(Faser[frist_dager])
)
```

```DAX
Åpne faser i faresonen =
VAR Terskel = [Faresone terskel]
VAR Idag = TODAY()
RETURN
    CALCULATE(
        COUNTROWS(Faser),
        NOT ISBLANK(Faser[startmilepaeldato]),
        ISBLANK(Faser[sluttmilepaeldato]),
        NOT ISBLANK(Faser[frist_dager]),
        FILTER(
            Faser,
            DATEDIFF(Faser[startmilepaeldato], Idag, DAY)
                >= Faser[frist_dager] * Terskel
        )
    )
```

```DAX
Åpne faser over frist =
VAR Idag = TODAY()
RETURN
    CALCULATE(
        COUNTROWS(Faser),
        NOT ISBLANK(Faser[startmilepaeldato]),
        ISBLANK(Faser[sluttmilepaeldato]),
        NOT ISBLANK(Faser[frist_dager]),
        FILTER(
            Faser,
            DATEDIFF(Faser[startmilepaeldato], Idag, DAY) > Faser[frist_dager]
        )
    )
```

```DAX
Andel åpne faser i faresonen =
DIVIDE([Åpne faser i faresonen], [Åpne faser med frist])
```

```DAX
Åpne faser i faresonen, ikke over frist =
[Åpne faser i faresonen] - [Åpne faser over frist]
```

`Åpne faser i faresonen` **inkluderer** dem som allerede er over fristen. Den siste målet
over er de som fortsatt kan reddes — ofte det mest handlingsrettede tallet.

`Terskel` og `Idag` hentes som variabler før `FILTER`, slik at de beregnes én gang og ikke
per rad.

```DAX
Faresone fargekode =
VAR A = [Andel åpne faser i faresonen]
RETURN
SWITCH(
    TRUE(),
    ISBLANK(A), "#757575",
    A >= 0.30, "#B00020",
    A >= 0.15, "#F9A825",
    "#2E7D32"
)
```

Tersklene 15 %/30 % er et utgangspunkt — juster dem når dere har sett nivået noen uker.

## Visualforslag
- **KPI-kort:** `[Åpne faser i faresonen, ikke over frist]` og `[Åpne faser over frist]`,
  farget med `[Faresone fargekode]`
- **Stolpediagram:** `enhet` (X) vs `[Andel åpne faser i faresonen]` (Y), filter
  `indikator`
- **Matrise:** rader `enhet`, kolonner `fasetittel`, verdi `[Åpne faser i faresonen]`
- **Slicer:** `Faresone terskel` (standard 0,75)
- Ikke vis enkeltpersoner her — samme avgrensning som resten av leder-rapporten.

## Tolkning
- Et høyt `[Åpne faser i faresonen, ikke over frist]` betyr at fristbrudd er på vei, men
  ennå ikke skjedd — det kommer i `Fristprosent` først når fasene lukkes.
- `[Åpne faser over frist]` som vokser mens `Fristprosent` ser stabil ut, betyr at bruddene
  hoper seg opp uten å bli lukket — `Fristprosent` vil falle når de lukkes.
- Faser som har stått åpne svært lenge (f.eks. glemte registreringer) teller som over
  frist. Er tallet høyt og flatt, se etter datakvalitet før kapasitet.
