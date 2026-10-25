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

Hver måned får et punkt og et bånd. **Midtlinjen** er normalen for den kalendermåneden:
samme måned de tre foregående årene (summerte tellinger, ikke snitt av andeler), så juli
måles mot tidligere juli-måneder. **Båndet** er hvor langt måneden kan avvike fra normalen
bare på grunn av tilfeldigheter, gitt hvor mange faser den hadde. En måned med 25 faser kan
svinge mye; en med 400 nesten ikke. Derfor er båndet bredt for små indikatorer og smalt for
store.

```DAX
p-kort verdi =
IF(MAX(Kalender[Dato]) <= EOMONTH(TODAY(), -1), [Faser innen frist %])
```

```DAX
p-kort senterlinje =
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

```DAX
p-kort sigma =
VAR P = [p-kort senterlinje]
VAR N = [Produserte faser]
RETURN
    IF(N > 0 && NOT ISBLANK(P), SQRT(P * (1 - P) / N))
```

```DAX
p-kort øvre grense =
MIN(1, [p-kort senterlinje] + 3 * [p-kort sigma])
```

```DAX
p-kort nedre grense =
MAX(0, [p-kort senterlinje] - 3 * [p-kort sigma])
```

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

**Punktfargen** — det eneste leseren trenger å se etter:

```DAX
p-kort punktfarge =
VAR V = [p-kort verdi]
RETURN
    SWITCH(
        TRUE(),
        ISBLANK(V) || ISBLANK([p-kort sigma]), BLANK(),
        V < [p-kort nedre grense] || [p-kort 8 under på rad], "#B00020",   -- rød: reager
        V > [p-kort øvre grense], "#1565C0",                               -- blå: uvanlig god
        "#9E9E9E"                                                          -- grå: normal variasjon
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
5. **Skjul tallene** for grensene og sigma. Ledere trenger båndet, ikke formelen.
6. **Lesehjelp** som tekst rett under kortet:
   > *Det grå båndet viser normal variasjon for antall saker den måneden. Reager bare på
   > røde punkter: en måned under båndet, eller åtte måneder på rad under midtlinjen.*

## Tolkning
- **Grå punkter, også når de er lavere enn forrige måned:** normal variasjon. Ikke reager —
  det er dette kortet er til for å unngå.
- **Rødt punkt under båndet:** uvanlig dårlig måned for det volumet. Se etter årsak.
- **Røde punkter fra løpsregelen:** nivået har sunket over tid. Det er en sak for
  styringsdialogen, ikke for én enkelt måned.
- **Blått punkt:** uvanlig god måned. Verdt å forstå hva som skjedde.
- **Mange røde punkter også i rolige perioder** (analytiker): variasjonen er større enn
  båndet antar (overdispersjon, f.eks. saker som kommer i klumper). Da er båndet for smalt;
  bruk Laneys p′-kort, der grensene skaleres med den faktiske variasjonen, eller les båndet
  som veiledende.
- **Brudd i nivået uten forklaring:** kjør `PELT_Changepoint.py` for å tidfeste det og
  bryte det ned på enhet/fasetittel.
