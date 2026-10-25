# Status i år — produksjon mot mål (Power BI and DAX)

Kolonner i tabellen **«Status i år»**, én rad per indikator. Svarer på spørsmålet en leder
kan handle på: **hvor mange må vi produsere per måned for å nå årsmålet, og hva produserer
vi nå?**

Ingen prognose for fristprosent ved årsslutt. Skriptet `Seasonal_YTD_ratio_extrapolation.py`
er fjernet (oktober 2026): fristprosent har ingen sesong, så det var i praksis en
framskriving av hittil i år, og en framskriving av siste tre måneder er ikke bedre. Lederne
ser allerede *Innen frist %* hittil i år, trendpilen (siste 3 mot forrige 3 måneder) og de
røde og gule åpne fasene — det siste er det nærmeste man kommer en prognose basert på
faktiske saker.

## Antagelser (rename to match your model)
- `[Produserte faser]` (eller `[Prod]`) finnes allerede — samme definisjon som kolonnen
  *Prod* i tabellen
- `[Mål]` er årsmålet for produksjon per indikator
- `Kalender[Dato]` er datotabellen, relatert til fasens sluttdato
- Bare hele måneder telles; inneværende måned regnes som «resten av året»

## Hjelpemål

```DAX
Siste hele måned =
EOMONTH(TODAY(), -1)
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

## Kolonnene i tabellen

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
- Målene ignorerer datoslicere med vilje: tallene skal være de samme uansett hvilken måned
  som er valgt. Velger noen et tidligere år, viser de fortsatt inneværende år — skjul
  kolonnene eller legg inn en sjekk på valgt år om det blir forvirrende.
- *Nødvendig per måned* fordeler resten likt på månedene. Siden volumet svinger med
  sesongen, er `[Prognose produksjon årsslutt]` mot `[Mål]` det mer presise bildet;
  *nødvendig per måned* er det som er lettest å handle på.
