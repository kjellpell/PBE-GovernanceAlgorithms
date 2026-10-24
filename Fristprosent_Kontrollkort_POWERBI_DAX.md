# Fristprosent — rullerende 12 måneder og p-kort (Power BI and DAX)

**Målgruppe:**
- **Leder-rapport:** `Fristprosent 12 mnd`, `Endring mot i fjor (pp)` og
  `Trendretning (12 mnd)` leser kaldt — et prosenttall, en endring i prosentpoeng og ett ord.
- **Analytiker-rapport:** p-kortet (kontrollgrenser per måned).

Erstatter CUSUM (`cusum_analyse`, `Trendretning (CUSUM)`), pensjonert i oktober 2026. CUSUM
på en lukket-sak-måling var tregt (ca. 9 måneder før et vedvarende skift på 1 σ ble sett
på månedsdata), og med en nylig baseline ga det hyppige falske signaler. Lederen kunne
heller ikke se hvorfor ordet på kortet endret seg. Disse målene svarer på det samme
spørsmålet — «har nivået endret seg?» — med tall leseren kan kontrollere selv. Alt er live
DAX; ingen skript, ingen tabell.

**Dette er ikke det tidlige varselet.** `Fristprosent` måles når fasene lukkes og kommer
derfor etter. De ledende signalene står i `At_Risk_Open_Cases_POWERBI_DAX.md` (faser hos oss
i faresonen, implisitt behandlingstid, andel som venter). Denne siden bekrefter at et skift
har skjedd.

## Antagelser (rename to match your model)
- `[Faser innen frist]`, `[Produserte faser]` og `[Faser innen frist %]` finnes allerede
  (se `Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md`) — den offisielle definisjonen
- `Kalender[Dato]` er datotabellen, relatert til fasens sluttdato
- Bare **hele måneder**: alle mål under ankres på siste hele måned, `EOMONTH(TODAY(), -1)`

## Del 1 — Rullerende 12 måneder (leder-rapport)

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

```DAX
Produserte faser 12 mnd =
CALCULATE(
    [Produserte faser],
    DATESINPERIOD(Kalender[Dato], [Siste hele måned], -12, MONTH)
)
```

```DAX
Trendretning (12 mnd) =
VAR E = [Endring mot i fjor (pp)]
RETURN
    SWITCH(
        TRUE(),
        ISBLANK(E) || [Produserte faser 12 mnd] < 100, BLANK(),
        E >= 2, "Stigende",
        E <= -2, "Synkende",
        "Stabil"
    )
```

Grensen på ±2 prosentpoeng og minimumsvolumet på 100 produserte faser er et utgangspunkt.
Under ca. 100 faser per år er tilfeldig variasjon i en andel større enn 2 pp, og ordet ville
skifte uten grunn — da vises ingenting. Juster når dere har sett nivået.

Til en linje per måned (i stedet for et kort): bytt `[Siste hele måned]` med
`MAX(Kalender[Dato])` i en kopi av `Fristprosent 12 mnd`, og vis bare hele måneder.

### Visualforslag (leder)
- **KPI-kort:** `[Fristprosent 12 mnd]` med `[Endring mot i fjor (pp)]` som undertekst,
  og `[Trendretning (12 mnd)]` som ett ord
- **Linje:** rullerende 12 måneder per måned, siste 36 måneder, med endringsloggen som
  markører når den finnes

## Del 2 — p-kort per måned (analytiker-rapport)

Et p-kort viser hver måneds `Fristprosent` mot kontrollgrenser som tar hensyn til hvor mange
faser måneden hadde: en måned med 20 faser kan svinge langt mer enn en med 400 uten at noe
har endret seg. Senterlinjen er **samme kalendermåned de tre foregående årene** (summerte
tellinger, ikke snitt av andeler), så juli måles mot tidligere juli-måneder.

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

```DAX
p-kort utenfor =
VAR V = [Faser innen frist %]
RETURN
    SWITCH(
        TRUE(),
        ISBLANK(V) || ISBLANK([p-kort sigma]), BLANK(),
        V < [p-kort nedre grense], "Under",
        V > [p-kort øvre grense], "Over",
        "Innenfor"
    )
```

### Visualforslag (analytiker)
- **Linje** per måned (`Kalender[Dato]` på månedsnivå, bare hele måneder — bruk
  `[Faser innen frist % (lukket)]` for selve verdien): `Fristprosent`, senterlinje,
  øvre og nedre grense. Farg punktene med `[p-kort utenfor]`.
- **Filter:** `indikator`. Et p-kort per indikator; ikke bland indikatorer med ulikt nivå.

## Tolkning
- **Ett punkt under nedre grense:** uvanlig dårlig måned for det volumet. Se etter årsak.
- **Mange punkter på samme side av senterlinjen** (tommelfingerregel: 8 på rad) betyr et
  vedvarende skift, selv om ingen er utenfor grensene. Det er dette CUSUM skulle fange.
- **Mange punkter utenfor grensene også i rolige perioder** betyr at variasjonen er større
  enn binomisk (overdispersjon — f.eks. saker som kommer i klumper). Da er grensene for
  trange; bruk Laneys p′-kort (grensene skaleres med variasjonen i z-verdiene) — eller
  vurder grensene som veiledende.
- **Brudd i nivået som ikke har en forklaring i endringsloggen:** kjør `PELT_Changepoint.py`
  for å tidfeste det og bryte det ned på enhet/fasetittel.
