# Fristprosent — p-kort og rullerende 12 måneder (Power BI and DAX)

**Målgruppe:** Leder-rapport.
- **p-kortet** — hver måneds `Fristprosent` mot et skyggelagt bånd for normal variasjon,
  med røde punkter der det er grunn til å reagere. Én linjes lesehjelp er nok.
- **Overskriftstallet** — `Fristprosent 12 mnd` med `Endring mot i fjor (pp)`.

Analytiker-rapporten kan bruke samme kort med grensene som tall og tolkningsnotatene nederst.

Erstatter CUSUM (`cusum_analyse`, `Trendretning (CUSUM)`), pensjonert i oktober 2026. CUSUM
på en lukket-sak-måling var tregt (ca. 9 måneder før et vedvarende skift på 1 σ ble sett på
månedsdata), og med en nylig baseline ga det hyppige falske signaler. Lederen kunne heller
ikke se hvorfor ordet på kortet endret seg. p-kortet viser det samme — «har nivået endret
seg?» — slik at leseren ser det selv. Alt er live DAX; ingen skript, ingen tabell.

**Hvorfor ledere skal ha det:** det beskytter mot de to vanligste feilene i
styringsrapportering: å reagere på en måned som ser dårlig ut, men er vanlig variasjon, og å
overse en langsom nedgang der ingen enkeltmåned ser alarmerende ut. Et enkelt tall eller ord
kan ikke vise noen av delene.

**Dette er ikke det tidlige varselet.** `Fristprosent` måles når fasene lukkes og kommer
derfor etter. Det ledende signalet er de røde og gule åpne fasene i
`At_Risk_Open_Cases_POWERBI_DAX.md`. Denne siden bekrefter at et skift har skjedd.

## Antagelser (rename to match your model)
- `[Faser innen frist]`, `[Produserte faser]` og `[Faser innen frist %]` finnes allerede
  (se `Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md`) — den offisielle definisjonen
- `Kalender[Dato]` er datotabellen, relatert til fasens sluttdato
- Bare **hele måneder**: inneværende måned vises ikke (den er ikke ferdig)

## Del 1 — Overskriftstallet: rullerende 12 måneder

Tolv måneder i én verdi er sesongnøytral: hver måned er med én gang, uansett når man ser.

```DAX
Siste hele måned =
EOMONTH(TODAY(), -1)
```

```DAX
Fristprosent 12 mnd =
CALCULATE(
    [Faser innen frist %],
    DATESINPERIOD(Kalender[Dato], [Siste hele måned], -12, MONTH)
)
```

```DAX
Fristprosent 12 mnd i fjor =
CALCULATE(
    [Faser innen frist %],
    DATESINPERIOD(Kalender[Dato], EDATE([Siste hele måned], -12), -12, MONTH)
)
```

```DAX
Endring mot i fjor (pp) =
VAR Naa = [Fristprosent 12 mnd]
VAR IFjor = [Fristprosent 12 mnd i fjor]
RETURN
    IF(NOT ISBLANK(Naa) && NOT ISBLANK(IFjor), (Naa - IFjor) * 100)
```

Vis som **KPI-kort**: `[Fristprosent 12 mnd]` med `[Endring mot i fjor (pp)]` som undertekst
(«+1,8 pp mot i fjor»). Ikke farg kortet etter endringen alene. Om endringen er reell, er det
p-kortet under som viser.

## Del 2 — p-kortet

Hver måned får et punkt og et bånd. **Midtlinjen** er normalen, og **båndet** er normal
variasjon rundt den. Kortet bruker **båndet dere allerede har**: snittet for måneden de tre
siste årene, ± 0,5 standardavvik. Det er kalibrert på deres data (1 standardavvik ga et for
bredt bånd, fordi standardavviket også fanger sesong og forskjeller mellom år), og det ser
likt ut som i rapportene ledere allerede kjenner.

Koble målene under til de eksisterende formlene. **Bytt navnene i hakeparentes** til målene
dere har:

```DAX
p-kort verdi =
IF(MAX(Kalender[Dato]) <= EOMONTH(TODAY(), -1), [Faser innen frist %])
```

```DAX
p-kort senterlinje =
[Snitt siste 3 år]                -- eksisterende mål: snittet for måneden de tre siste årene
```

```DAX
p-kort øvre grense =
[Øvre grense (0,5 SD)]            -- eksisterende mål
```

```DAX
p-kort nedre grense =
[Nedre grense (0,5 SD)]           -- eksisterende mål
```

**Sesong:** fristprosent svinger med årstiden (bekreftet). Midtlinjen bør derfor være
snittet for **samme kalendermåned** de tre siste årene (juli mot tidligere juli-måneder),
ikke snittet av alle måneder. Er den eksisterende midtlinjen snittet av alle måneder, vil de
svake sesongmånedene jevnlig bli gule eller røde uten at noe er galt. Bytt da
`p-kort senterlinje` til samme-måned-snittet under, og behold deres 0,5 SD-bånd rundt den:

```DAX
p-kort senterlinje (samme måned) =
VAR D = MAX(Kalender[Dato])
VAR M = MONTH(D)
VAR Y = YEAR(D)
RETURN
    CALCULATE(
        [Faser innen frist %],
        FILTER(
            ALL(Kalender),
            MONTH(Kalender[Dato]) = M
                && YEAR(Kalender[Dato]) >= Y - 3
                && YEAR(Kalender[Dato]) <= Y - 1
        )
    )
```

Når midtlinjen følger sesongen, blir avvikene rundt den mindre, så 0,5 SD kan da bli et
smalere bånd enn i dag. Kalibreringskontrollen under viser om det blir for smalt.

Båndet må gjelde **fristprosent per måned**, samme tall som punktene. Gjelder de eksisterende
formlene produksjonsvolum (antall saker), er det et annet kort — nyttig for kapasitet, men
ikke dette.

**Løpsregelen:** åtte måneder på rad under midtlinjen betyr et vedvarende skift nedover,
selv om ingen av dem er under båndet. Dette er det CUSUM skulle fange. Målet ser på denne
måneden og de sju før den:

```DAX
p-kort 8 under på rad =
VAR Slutt = EOMONTH(MAX(Kalender[Dato]), 0)
VAR Maaneder =
    ADDCOLUMNS(
        GENERATESERIES(0, 7, 1),
        "@Verdi",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN
                CALCULATE(
                    [p-kort verdi],
                    REMOVEFILTERS(Kalender),
                    FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til)
                ),
        "@Senter",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN
                CALCULATE(
                    [p-kort senterlinje],
                    REMOVEFILTERS(Kalender),
                    FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til)
                )
    )
RETURN
    COUNTROWS(
        FILTER(Maaneder, NOT ISBLANK([@Verdi]) && NOT ISBLANK([@Senter]) && [@Verdi] < [@Senter])
    ) = 8
```

**Kalibreringskontroll:** hvor ofte har måneden ligget under båndet de siste 36
månedene? Med et bånd som er justert på øyemål vet vi ikke det på forhånd, så kortet
måler det selv, per indikator:

```DAX
p-kort andel under båndet (36 mnd) =
VAR Slutt = EOMONTH(TODAY(), -1)
VAR Maaneder =
    ADDCOLUMNS(
        GENERATESERIES(0, 35, 1),
        "@Verdi",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN
                CALCULATE(
                    [p-kort verdi],
                    REMOVEFILTERS(Kalender),
                    FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til)
                ),
        "@Nedre",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN
                CALCULATE(
                    [p-kort nedre grense],
                    REMOVEFILTERS(Kalender),
                    FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til)
                )
    )
VAR Med = FILTER(Maaneder, NOT ISBLANK([@Verdi]) && NOT ISBLANK([@Nedre]))
RETURN
    DIVIDE(COUNTROWS(FILTER(Med, [@Verdi] < [@Nedre])), COUNTROWS(Med))
```

- **Opptil ca. 10 %** (rundt én måned i året under båndet): et punkt under båndet er et
  reelt varsel og vises **rødt**.
- **Mer enn 10 %:** båndet er for smalt for den indikatoren til at hvert punkt under det
  kan være et varsel. Da vises punktet **gult** («følg med»), og bare løpsregelen gir rødt.
  Det skjer automatisk i punktfargen under.

**Punktfargen** — det eneste leseren trenger å se etter:

```DAX
p-kort punktfarge =
VAR V = [p-kort verdi]
VAR Nedre = [p-kort nedre grense]
VAR BaandHolder = [p-kort andel under båndet (36 mnd)] <= 0.10
RETURN
    SWITCH(
        TRUE(),
        ISBLANK(V) || ISBLANK(Nedre), BLANK(),
        [p-kort 8 under på rad], "#B00020",              -- rød: vedvarende nedgang
        V < Nedre && BaandHolder, "#B00020",             -- rød: uvanlig dårlig måned
        V < Nedre, "#F9A825",                            -- gul: følg med
        V > [p-kort øvre grense], "#1565C0",             -- blå: uvanlig god
        "#9E9E9E"                                        -- grå: normal variasjon
    )
```

```DAX
p-kort signal =
IF([p-kort punktfarge] = "#B00020", [p-kort verdi])
```

### Slik bygges kortet for ledere
1. **Linjediagram**, x-akse `Kalender[Dato]` på månedsnivå (kontinuerlig), siste 36
   måneder. Én indikator om gangen: slicer på `indikator`, eller små multipler per indikator.
   Ikke bland indikatorer med ulikt nivå i samme kort.
2. **Linjer:** `[p-kort verdi]` (grå, med markører) og `[p-kort senterlinje]` (tynn,
   stiplet).
3. **Båndet:** under *Analyse → Feilfelt (error bars)* på `[p-kort senterlinje]`, sett øvre
   grense til `[p-kort øvre grense]` og nedre til `[p-kort nedre grense]`, og velg skyggelagt
   område i lys grå. Har ikke versjonen deres skyggelagt feilfelt, bruk to tynne, lyse
   linjer.
4. **Røde punkter:** sett markørfargen med betinget formatering (*fx*) på
   `[p-kort punktfarge]`. Går ikke det i deres versjon, legg til `[p-kort signal]` som egen
   serie: røde markører, ingen linje.
5. **Skjul tallene** for grensene. Ledere trenger båndet, ikke formelen.
6. **Lesehjelp** som tekst rett under kortet:
   > *Det grå båndet viser normal variasjon de tre siste årene. Reager på røde punkter
   > (uvanlig dårlig måned, eller åtte måneder på rad under midtlinjen); følg med på gule.*

## Tolkning
- **Grå punkter, også når de er lavere enn forrige måned:** normal variasjon. Ikke reager —
  det er dette kortet er til for å unngå.
- **Rødt punkt under båndet:** uvanlig dårlig måned. Se etter årsak.
- **Gult punkt under båndet:** svakere enn normalt, men for den indikatoren skjer det for
  ofte til at én måned alene er et varsel. Følg med; blir det åtte på rad under midtlinjen,
  blir det rødt.
- **Røde punkter fra løpsregelen:** nivået har sunket over tid. Det er en sak for
  styringsdialogen, ikke for én enkelt måned.
- **Blått punkt:** uvanlig god måned. Verdt å forstå hva som skjedde.
- **Alternativ for analytikere — binomiske grenser:** grensene kan også regnes fra antall
  faser i måneden, ± 3·√(p(1−p)/n) rundt midtlinjen. Da blir båndet smalere i måneder med
  mange saker og bredere i måneder med få. Det er standard p-kort, men undervurderer ofte
  variasjonen når saker kommer i klumper; det empiriske båndet over gjør ikke det.
- **Brudd i nivået uten forklaring:** kjør `PELT_Changepoint.py` for å tidfeste det og
  bryte det ned på enhet/fasetittel.
