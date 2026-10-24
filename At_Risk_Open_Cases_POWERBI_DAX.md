# Faser hos oss i faresonen — Power BI and DAX

**Målgruppe:** Leder-rapport — «så mange faser hos oss har brukt mer enn tre firedeler av
fristen sin» leser kaldt, uten metodeforklaring.

**Hos oss vs. venter:** mange åpne saker står på vent mens søker kompletterer
dokumentasjon. Den tiden er utenfor avdelingens kontroll og stopper normalt fristen. Derfor
skilles **hos oss** (`Faser[Status] = "Under behandling"`) fra **venter** (åpen, annen
status). Bare faser hos oss kan være i faresonen; ventende faser vises som egen andel.

## Formål
Alle andre signaler i dette repoet (PELT, prognose, 12-måneders fristprosent, p-kort, flyt, flaskehals) regnes på
**avsluttede** faser. De kommer derfor etter: en fase som kommer til å bryte fristen vises
først når den lukkes. Men den er synlig lenge før — som en åpen fase som eldes. Denne siden
svarer på:
- Hvor mange faser hos oss er i ferd med å gå over fristen, akkurat nå?
- Hvor mange har allerede gått over, men er ikke lukket ennå?
- Hvilke enheter og faser har størst andel i faresonen?

Dette er et **ledende** signal. Det er ren live DAX mot `Faser`, uten skript: verdien
avhenger av `TODAY()`, men et øyeblikksbilde trenger ingen historikk (se regelen i
`README.md`). Siden viser «nå». **Trenden** — hvordan antallet hos oss, i faresonen og på
vent har utviklet seg uke for uke, bakover til 2015 — kommer fra `analyser.flyt_ukentlig`,
som `Pause_Reconstruction.py` bygger opp fra milepælene (se
`Pause_Reconstruction_POWERBI_DAX.md`). Ingen voksende snapshot-tabell.

## Antagelser (rename to match your model)
- `Faser` — `saksbehandling.faser`, med `startmilepaeldato`, `sluttmilepaeldato`,
  `frist_dager`, `tidsbruk`, `bransjetid`, `enhet`, `fasetittel`, `indikator`, `Status`
- `Faser[Status] = "Under behandling"` betyr at fasen er **hos oss** (bekreftet). Andre
  statuser på en åpen fase = venter (på søker eller ekstern part)
- **Fristklokken er `Faser[tidsbruk]`**: dager på *vår* side, en løpende akkumulator som
  oppdateres mens fasen er åpen og står stille mens den venter på søker (den tiden går i
  `Faser[bransjetid]`). Derfor brukes `tidsbruk`, ikke kalenderdager siden start — en fase
  som har ventet en måned på tegninger er ikke en måned nærmere fristen. (Tidligere
  versjoner av denne siden, og det pensjonerte `Inflight_SLA_Risk_Monitor`, regnet
  kalenderalder og overvarslet derfor.) `Pause_Reconstruction.py` kontrollerer hver natt at
  `tidsbruk` stemmer med pausene i milepælene — se utskriften der
- `frist_dager` og `tidsbruk` er **kalenderdager** (bekreftet)
- Samme omfang som den offisielle `Fristprosent`: fagområde `Byggesak`/`Eiendomssak`/
  `Plansak` (filtrert i modellen), og `%avtalt%`-indikatorer er **med** — en avtalt frist
  er fortsatt en frist.
- En fase regnes som brutt når `tidsbruk` er **større enn** `frist_dager` (samme som at
  `tidsbruk <= frist_dager` gir `innenfor_frist = 1`). Kontroller mot
  hvordan `innenfor_frist` faktisk beregnes i datavarehuset; er det `<`, endre `>` til
  `>=` i `Hos oss over frist`.
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

Alle mål under er per **fase** og bruker samme omfang som `Fristprosent`.

```DAX
Hos oss =
CALCULATE(
    COUNTROWS(Faser),
    Faser[Status] = "Under behandling",
    NOT ISBLANK(Faser[frist_dager])
)
```

```DAX
Venter (ikke hos oss) =
CALCULATE(
    COUNTROWS(Faser),
    NOT ISBLANK(Faser[startmilepaeldato]),
    ISBLANK(Faser[sluttmilepaeldato]),
    Faser[Status] <> "Under behandling"
)
```

```DAX
Andel som venter =
DIVIDE([Venter (ikke hos oss)], [Venter (ikke hos oss)] + [Hos oss])
```

```DAX
Hos oss i faresonen =
VAR Terskel = [Faresone terskel]
RETURN
    COUNTROWS(
        FILTER(
            Faser,
            Faser[Status] = "Under behandling"
                && NOT ISBLANK(Faser[tidsbruk])
                && NOT ISBLANK(Faser[frist_dager])
                && Faser[tidsbruk] >= Faser[frist_dager] * Terskel
        )
    )
```

```DAX
Hos oss over frist =
    COUNTROWS(
        FILTER(
            Faser,
            Faser[Status] = "Under behandling"
                && NOT ISBLANK(Faser[tidsbruk])
                && NOT ISBLANK(Faser[frist_dager])
                && Faser[tidsbruk] > Faser[frist_dager]
        )
    )
```

```DAX
Andel hos oss i faresonen =
DIVIDE([Hos oss i faresonen], [Hos oss])
```

```DAX
Hos oss i faresonen, ikke over frist =
[Hos oss i faresonen] - [Hos oss over frist]
```

`Hos oss i faresonen` **inkluderer** dem som allerede er over fristen. Det siste målet
over er de som fortsatt kan reddes — ofte det mest handlingsrettede tallet.

`Terskel` hentes som variabel før `FILTER`, slik at den beregnes én gang og ikke per rad.

### Implisitt behandlingstid (Littles lov)

Hvor mange dager tar det å tømme køen *hos oss* i dagens tempo? Sammenlign med fristen:
er implisitt behandlingstid lengre enn typisk `frist_dager`, vil `Fristprosent` falle
uken(e) etter — før det vises i de lukkede sakene.

```DAX
Ferdigstilt siste 28 dager =
VAR Idag = TODAY()
RETURN
    CALCULATE(
        COUNTROWS(Faser),
        Faser[sluttmilepaeldato] > Idag - 28,
        Faser[sluttmilepaeldato] <= Idag,
        REMOVEFILTERS('Kalender')   -- bytt til navnet på datotabellen
    )
```

```DAX
Implisitt behandlingstid (dager) =
DIVIDE([Hos oss], [Ferdigstilt siste 28 dager]) * 28
```

```DAX
Typisk frist (dager) =
CALCULATE(MEDIAN(Faser[frist_dager]), Faser[Status] = "Under behandling")
```

`Ferdigstilt` følger samme definisjon som rapportens `[Produserte faser]`: alle avsluttede
faser, også de som ble trukket — de har også tatt arbeid og forlatt køen.

```DAX
Faresone fargekode =
VAR A = [Andel hos oss i faresonen]
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
- **KPI-kort:** `[Hos oss i faresonen, ikke over frist]` og `[Hos oss over frist]`,
  farget med `[Faresone fargekode]`
- **KPI-kort:** `[Implisitt behandlingstid (dager)]` med `[Typisk frist (dager)]` som
  referanse
- **Stolpediagram:** `enhet` (X) vs `[Andel hos oss i faresonen]` (Y), filter
  `indikator`
- **Stolpediagram:** `enhet` (X) vs `[Andel som venter]` (Y)
- **Matrise:** rader `enhet`, kolonner `fasetittel`, verdi `[Hos oss i faresonen]`
- **Slicer:** `Faresone terskel` (standard 0,75)
- Ikke vis enkeltpersoner her — samme avgrensning som resten av leder-rapporten.

## Tolkning
- Et høyt `[Hos oss i faresonen, ikke over frist]` betyr at fristbrudd er på vei, men
  ennå ikke skjedd — det kommer i `Fristprosent` først når fasene lukkes.
- `[Hos oss over frist]` som vokser mens `Fristprosent` ser stabil ut, betyr at bruddene
  hoper seg opp uten å bli lukket — `Fristprosent` vil falle når de lukkes.
- `[Implisitt behandlingstid (dager)]` over `[Typisk frist (dager)]` betyr at køen hos oss
  ikke kan tømmes innen fristen i dagens tempo — det tidligste kvantitative varselet her.
- En stigende `[Andel som venter]` betyr enten svakere søknader (mer veiledning,
  sjekklister) eller at fristen stoppes oftere. Se på det per enhet over tid.
- Faser som har stått åpne svært lenge (f.eks. glemte registreringer) teller som over
  frist. Er tallet høyt og flatt, se etter datakvalitet før kapasitet.
