# Status i år — prognose for frist og produksjon (Power BI and DAX)

Kolonner i tabellen **«Status i år»**, én rad per indikator. Svarer på to spørsmål en leder
kan handle på:
- **Frist:** hvor ender vi ved årsslutt hvis resten av året går som de siste tre månedene?
- **Produksjon:** hvor mange må vi produsere per måned for å nå målet, og hva produserer vi nå?

Erstatter skriptet `Seasonal_YTD_ratio_extrapolation.py` (fjernet oktober 2026). Fristprosent
har ingen sesong, så skriptets sesongmodell ga i praksis «omtrent der vi er hittil i år».
Det som er nyttig for en leder er dagens tempo, og det er enkel aritmetikk.

## Antagelser (rename to match your model)
- `[Faser innen frist]` og `[Produserte faser]` (eller `[Prod]`) finnes allerede — samme
  definisjon som kolonnene *Innen frist %* og *Prod* i tabellen
- `[Mål]` er årsmålet for produksjon per indikator
- `Kalender[Dato]` er datotabellen, relatert til fasens sluttdato
- Bare hele måneder telles; inneværende måned regnes som «resten av året»

## Hjelpemål

```DAX
Siste hele måned =
EOMONTH(TODAY(), -1)
```

```DAX
Innen frist hittil i år =
CALCULATE(
    [Faser innen frist],
    REMOVEFILTERS(Kalender),
    DATESBETWEEN(Kalender[Dato], DATE(YEAR(TODAY()), 1, 1), [Siste hele måned])
)
```

```DAX
Produsert hittil i år =
CALCULATE(
    [Produserte faser],
    REMOVEFILTERS(Kalender),
    DATESBETWEEN(Kalender[Dato], DATE(YEAR(TODAY()), 1, 1), [Siste hele måned])
)
```

```DAX
Forventet produksjon resten av året =
-- samme måneder i fjor; volumet svinger med sesongen, så fjoråret er en god pekepinn
CALCULATE(
    [Produserte faser],
    REMOVEFILTERS(Kalender),
    DATESBETWEEN(
        Kalender[Dato],
        EDATE([Siste hele måned], -12) + 1,
        DATE(YEAR(TODAY()) - 1, 12, 31)
    )
)
```

```DAX
Fristprosent siste 3 mnd =
DIVIDE(
    CALCULATE([Faser innen frist], REMOVEFILTERS(Kalender),
        DATESINPERIOD(Kalender[Dato], [Siste hele måned], -3, MONTH)),
    CALCULATE([Produserte faser], REMOVEFILTERS(Kalender),
        DATESINPERIOD(Kalender[Dato], [Siste hele måned], -3, MONTH))
)
```

## Kolonnene i tabellen

```DAX
Prognose årsslutt (dagens tempo) =
VAR Innen = [Innen frist hittil i år]
VAR Prod  = [Produsert hittil i år]
VAR Rest  = [Forventet produksjon resten av året]
RETURN
    DIVIDE(Innen + [Fristprosent siste 3 mnd] * Rest, Prod + Rest)
```

Det vi har oppnådd hittil i år, pluss resten av året med fristprosenten fra de siste tre
månedene. Ligger den under *Innen frist %* (hittil i år), trekker det siste kvartalet året
ned. Sett formatet til prosent.

```DAX
Prognose produksjon årsslutt =
[Produsert hittil i år] + [Forventet produksjon resten av året]
```

Sammenlign med `[Mål]` i samme rad.

```DAX
Produksjon nødvendig per måned =
VAR MaanederIgjen = 12 - MONTH([Siste hele måned])    -- inkludert inneværende måned
RETURN DIVIDE([Mål] - [Produsert hittil i år], MaanederIgjen)
```

```DAX
Produksjon snitt siste 3 mnd =
CALCULATE([Produserte faser], REMOVEFILTERS(Kalender),
    DATESINPERIOD(Kalender[Dato], [Siste hele måned], -3, MONTH)) / 3
```

Ligger *nødvendig per måned* over *snitt siste 3 mnd*, når vi ikke målet i dagens tempo.
Forskjellen er hvor mange flere faser i måneden som trengs.

## Merk
- Målene ignorerer datoslicere med vilje: årsprognosen skal være den samme uansett hvilken
  måned som er valgt. Velger noen et tidligere år, viser de fortsatt inneværende års
  prognose — skjul kolonnene eller legg inn en sjekk på valgt år om det blir forvirrende.
- I januar finnes ingen hele måneder i år ennå; prognosen er da bare dagens tempo.
- *Nødvendig per måned* fordeler resten likt på månedene. Siden volumet svinger med
  sesongen, er `[Prognose produksjon årsslutt]` mot `[Mål]` det mer presise bildet;
  *nødvendig per måned* er det som er lettest å handle på.
