# Fristrisiko — Power BI and DAX

**Målgruppe:** Leder-rapport. Leser kaldt: rød, gul og grønn er fargene dere allerede
bruker i saksoversikten.

## Formål
`Fristprosent` regnes på **avsluttede** faser og kommer derfor etter: en fase som bryter
fristen vises først når den lukkes. Men den er synlig før det, som en åpen fase som nærmer
seg fristen. Denne siden teller de åpne fasene etter **fargekoden som allerede finnes** for
tid igjen til frist:

- **Rød** — over fristen
- **Gul** — nær fristen. Terskelen varierer med sakstypen (1–4 uker før frist, for frister
  på 3 og 12 uker)
- **Grønn** — resten

Siden lager ingen ny regel for hva som er «i faresonen». Den gjenbruker den dere har, så
tallene stemmer med saksoversikten. Fargekoden handler bare om tid igjen til frist og tar
ikke hensyn til `Status`; `Status` vises ved siden av, slik at det er lett å skille saker
som venter på søker fra saker som ligger hos oss.

Del 1 er **i dag** — fargen avhenger av dagens dato, og ingen tabell tar vare på gårsdagens
farger. Del 2 er flyten over tid tilbake til 2015.

## Antagelser (rename to match your model)
- `Faser[Fristfarge]` er fargekoden for tid igjen til frist, med verdiene `"Rød"`, `"Gul"`
  og `"Grønn"`. **Bytt til feltets egentlige navn og verdier.**
  Er fargen et **mål** (measure) og ikke en kolonne, bytt `Faser[Fristfarge] = "Rød"` med
  `[Fristfarge] = "Rød"` inne i `FILTER(Faser, ...)` i målene under — da beregnes målet per
  fase
- Bare **åpne** faser telles: start satt, slutt ikke satt
- Samme omfang som den offisielle `Fristprosent` (fagområde og indikatorer filtreres i
  modellen)

## Del 1 — Fristrisiko i dag

```DAX
Åpne faser =
CALCULATE(
    COUNTROWS(Faser),
    NOT ISBLANK(Faser[startmilepaeldato]),
    ISBLANK(Faser[sluttmilepaeldato])
)
```

```DAX
Røde (over frist) =
CALCULATE([Åpne faser], Faser[Fristfarge] = "Rød")
```

```DAX
Gule (nær frist) =
CALCULATE([Åpne faser], Faser[Fristfarge] = "Gul")
```

```DAX
Grønne =
CALCULATE([Åpne faser], Faser[Fristfarge] = "Grønn")
```

```DAX
Andel røde =
DIVIDE([Røde (over frist)], [Åpne faser])
```

```DAX
Andel gule =
DIVIDE([Gule (nær frist)], [Åpne faser])
```

`Status` brukes ikke i målene. Den legges på som slicer eller kolonne, så samme mål svarer
på begge spørsmål: alle åpne faser, eller bare dem som ligger hos oss
(`Status = "Under behandling"`).

### Visualforslag (Del 1)
- **KPI-kort:** `[Røde (over frist)]` og `[Gule (nær frist)]`, farget rødt og gult
- **Stablet stolpe per `enhet`:** `[Grønne]`, `[Gule (nær frist)]`, `[Røde (over frist)]` i
  de samme fargene — viser hvor fordelingen er verst
- **Matrise:** rader `enhet`, kolonner `Status`, verdier `[Røde (over frist)]` og
  `[Gule (nær frist)]` — skiller saker hos oss fra saker som venter
- **Tabell (tiltaksliste):** de gule og røde fasene med saksnummer, `indikator`,
  `fasetittel`, `Status` og fargen — sortert slik saksoversikten sorterer
- **Slicere:** `enhet`, `indikator`, `Status`

### Tolkning (Del 1)
- **Mange gule hos oss** betyr at fristbrudd er på vei, men ennå ikke skjedd. De vises i
  `Fristprosent` først når fasene lukkes. Dette er tallet å handle på.
- **Røde som vokser mens `Fristprosent` ser stabil ut** betyr at bruddene hoper seg opp uten
  å bli lukket. `Fristprosent` vil falle når de lukkes.
- **Gule og røde med en venter-status** ligger hos søker. Fristen er da normalt stoppet;
  det er ikke vår kø.

## Del 2 — Flyt over tid (tilbake til 2015)

Erstatter teamflyten i det pensjonerte `Throughput_Pressure_Monitor`. Tre tall per periode
på `Kalender[Dato]`-aksen (måned eller uke), per enhet og indikator via vanlige slicere:
hvor mange som kom inn, hvor mange som ble ferdige, og hvor mange som var åpne da perioden
sluttet.

Målene bruker `TREATAS` mot riktig datokolonne og fjerner filteret fra `Kalender` først.
Da virker de uavhengig av hvilken datokolonne `Kalender` er relatert til i modellen, og det
trengs ingen inaktive relasjoner. Forutsetning: datokolonnene i `Faser` er rene datoer
(ikke dato+klokkeslett); er de det, lag en beregnet datokolonne og bruk den.

```DAX
Mottatt (periode) =
CALCULATE(
    COUNTROWS(Faser),
    REMOVEFILTERS('Kalender'),
    TREATAS(VALUES('Kalender'[Dato]), Faser[startmilepaeldato])
)
```

```DAX
Ferdigstilt (periode) =
CALCULATE(
    COUNTROWS(Faser),
    REMOVEFILTERS('Kalender'),
    NOT ISBLANK(Faser[startmilepaeldato]),
    TREATAS(VALUES('Kalender'[Dato]), Faser[sluttmilepaeldato])
)
```

(Samme populasjon som rapportens `[Produserte faser]`: start og slutt er satt.)

```DAX
Netto flyt (periode) =
[Mottatt (periode)] - [Ferdigstilt (periode)]
```

```DAX
Åpne ved periodeslutt =
VAR Slutt = MIN(MAX('Kalender'[Dato]), TODAY())
RETURN
    CALCULATE(
        COUNTROWS(Faser),
        REMOVEFILTERS('Kalender'),
        Faser[startmilepaeldato] <= Slutt,
        OR(ISBLANK(Faser[sluttmilepaeldato]), Faser[sluttmilepaeldato] > Slutt)
    )
```

`Åpne ved periodeslutt` er en **beholdning** — antall på en dato — og er riktig på alle
nivåer: en måned på aksen viser antallet ved månedens slutt, et år ved årets slutt.

```DAX
Gjennomløpstid, implisitt (dager) =
VAR Slutt = MIN(MAX('Kalender'[Dato]), TODAY())
VAR Ferdig90 =
    CALCULATE(
        COUNTROWS(Faser),
        REMOVEFILTERS('Kalender'),
        NOT ISBLANK(Faser[startmilepaeldato]),
        Faser[sluttmilepaeldato] > Slutt - 90,
        Faser[sluttmilepaeldato] <= Slutt
    )
RETURN DIVIDE([Åpne ved periodeslutt], Ferdig90) * 90
```

Dette er Littles lov på **alle** åpne faser, også dem som venter på søker, så den måler
søkerens ventetid og vår tid sammen. Bruk den til å se langsiktige endringer i nivå, ikke
som fristvarsel — det er fargekoden i Del 1.

### Visualforslag (flyt)
- **Stolper:** `[Mottatt (periode)]` og `[Ferdigstilt (periode)]` per måned, siste 3–5 år
- **Linje:** `[Åpne ved periodeslutt]` per måned siden 2015
- **Små multipler** per `enhet`: `[Netto flyt (periode)]` — flere måneder på rad over null
  betyr at køen vokser
