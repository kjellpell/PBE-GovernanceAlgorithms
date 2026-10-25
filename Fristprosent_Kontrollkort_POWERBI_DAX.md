# Kontrollkort for produksjon og fristprosent (Power BI and DAX)

**Målgruppe:** Leder-rapport. Tre ting på siden:
1. **Overskriftstallet** — `Fristprosent 12 mnd` med `Endring mot i fjor (pp)`
2. **Kontrollkort for produksjon** — produserte faser per måned mot båndet dere allerede har
3. **Kontrollkort for fristprosent** — samme prinsipp, med et bånd bygget på samme måte

De to kortene svarer på hvert sitt spørsmål. **Produksjon** viser om tempoet faller
(kapasitet). **Fristprosent** viser om fristene glipper. Produksjonen faller som regel først,
så side om side viser de både årsak og virkning.

Erstatter CUSUM (`cusum_analyse`, `Trendretning (CUSUM)`), pensjonert i oktober 2026. CUSUM
var tregt på månedsdata og ga hyppige falske signaler, og lederen kunne ikke se hvorfor
ordet på kortet endret seg. Kontrollkortene viser det samme — «har nivået endret seg?» —
slik at leseren ser det selv. Alt er live DAX; ingen skript, ingen tabell.

**Hvorfor ledere skal ha dem:** de beskytter mot de to vanligste feilene i
styringsrapportering: å reagere på en måned som ser dårlig ut, men er vanlig variasjon, og å
overse en langsom nedgang der ingen enkeltmåned ser alarmerende ut.

**Dette er ikke det tidlige varselet.** Begge tallene måles når fasene lukkes. Det ledende
signalet er de røde og gule åpne fasene i `At_Risk_Open_Cases_POWERBI_DAX.md`.

## Antagelser (rename to match your model)
- `[Produserte faser]`, `[Faser innen frist]` og `[Faser innen frist %]` finnes allerede
  (se `Seasonal_YTD_ratio_extrapolation_POWERBI_DAX.md`) — de offisielle definisjonene
- Produksjonsbåndet finnes allerede som mål. Navnene under — `[Produksjon snitt 3 år]`,
  `[Produksjon nedre grense]`, `[Produksjon øvre grense]` — er plassholdere: **bytt til
  deres egne**
- `Kalender[Dato]` er datotabellen
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
(«+1,8 pp mot i fjor»).

## Prinsippet for kontrollkortene

Hver måned er et punkt mot en **midtlinje** (normalen) og et **bånd** (normal variasjon).
Hvordan normalen regnes, avhenger av om tallet svinger med sesongen:

| | Sesong? | Midtlinje | Bånd |
|---|---|---|---|
| **Produksjon** (volum) | Ja | Samme kalendermåned, snitt av de tre siste årene | ± 0,5 SD |
| **Fristprosent** | Nei | De siste 12 månedene | ± 2 SD |

Produksjonsbåndet er det dere allerede har. Standardavviket der fanger også
sesongsvingningene, så 1 SD ga et for bredt bånd og 0,5 SD er kalibrert på deres data.
Fristprosent har ingen sesong: standardavviket er bare vanlig variasjon fra måned til
måned. Der ville ± 0,5 SD lagt over halvparten av helt normale måneder utenfor båndet; ± 2 SD
gir omtrent én av tjue.

Punktene farges etter to regler:
- **Rødt:** måneden er under båndet, **eller** den er den åttende måneden på rad under
  midtlinjen (en langsom nedgang der ingen enkeltmåned ser alarmerende ut)
- **Blått:** måneden er over båndet — uvanlig god, verdt å forstå
- Resten er vanlig variasjon og får ingen ekstra markering

Fargene legges på som **egne serier** som bare har verdi i måneder der fargen gjelder. Da
vises de som prikker oppå linjen, og det virker i alle versjoner av Power BI.

## Del 2 — Kontrollkort for produksjon

Båndet finnes allerede. Disse målene kobler det til punktreglene:

```DAX
Produksjon verdi =
IF(MAX(Kalender[Dato]) <= EOMONTH(TODAY(), -1), [Produserte faser])
```

```DAX
Produksjon 8 under på rad =
VAR Slutt = EOMONTH(MAX(Kalender[Dato]), 0)
VAR Maaneder =
    ADDCOLUMNS(
        GENERATESERIES(0, 7, 1),
        "@Verdi",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN CALCULATE([Produksjon verdi], REMOVEFILTERS(Kalender),
                FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til)),
        "@Senter",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN CALCULATE([Produksjon snitt 3 år], REMOVEFILTERS(Kalender),
                FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til))
    )
RETURN
    COUNTROWS(FILTER(Maaneder,
        NOT ISBLANK([@Verdi]) && NOT ISBLANK([@Senter]) && [@Verdi] < [@Senter])) = 8
```

```DAX
Rødt punkt (produksjon) =
VAR V = [Produksjon verdi]
RETURN IF(NOT ISBLANK(V) && (V < [Produksjon nedre grense] || [Produksjon 8 under på rad]), V)
```

```DAX
Blått punkt (produksjon) =
VAR V = [Produksjon verdi]
RETURN IF(NOT ISBLANK(V) && V > [Produksjon øvre grense], V)
```

Er det eksisterende produksjonsbåndet bygget på snittet av **alle** måneder i stedet for
samme kalendermåned, vil de svake sesongmånedene jevnlig bli røde. Da bør midtlinjen byttes
til snittet for samme kalendermåned de tre siste årene.

## Del 3 — Kontrollkort for fristprosent

Samme prinsipp, men uten sesong: normalen er de **12 foregående månedene**, og båndet er
± 2 standardavvik av månedsverdiene i de samme 12 månedene.

```DAX
Frist verdi =
IF(MAX(Kalender[Dato]) <= EOMONTH(TODAY(), -1), [Faser innen frist %])
```

```DAX
Frist normal =
VAR Slutt = EOMONTH(MAX(Kalender[Dato]), -1)       -- slutten av forrige måned
VAR Start = EOMONTH(Slutt, -12) + 1                -- 12 måneder før det
RETURN
    CALCULATE(
        [Faser innen frist %],
        REMOVEFILTERS(Kalender),
        FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Start && Kalender[Dato] <= Slutt)
    )
```

(Summerte tellinger for de 12 månedene, ikke snitt av tolv andeler.)

```DAX
Frist SD 12 mnd =
VAR Slutt = EOMONTH(MAX(Kalender[Dato]), -1)
VAR Maaneder =
    ADDCOLUMNS(
        GENERATESERIES(0, 11, 1),
        "@V",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN CALCULATE([Faser innen frist %], REMOVEFILTERS(Kalender),
                FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til))
    )
RETURN STDEVX.S(Maaneder, [@V])
```

```DAX
Frist nedre grense = [Frist normal] - 2 * [Frist SD 12 mnd]
Frist øvre grense  = [Frist normal] + 2 * [Frist SD 12 mnd]
```

Normalen og båndet regnes fra månedene **før** den viste måneden, så en dårlig måned trekker
ikke sin egen normal ned.

```DAX
Frist 8 under på rad =
VAR Slutt = EOMONTH(MAX(Kalender[Dato]), 0)
VAR Maaneder =
    ADDCOLUMNS(
        GENERATESERIES(0, 7, 1),
        "@Verdi",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN CALCULATE([Frist verdi], REMOVEFILTERS(Kalender),
                FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til)),
        "@Senter",
            VAR Til = EOMONTH(Slutt, -[Value])
            VAR Fra = EOMONTH(Til, -1) + 1
            RETURN CALCULATE([Frist normal], REMOVEFILTERS(Kalender),
                FILTER(ALL(Kalender[Dato]), Kalender[Dato] >= Fra && Kalender[Dato] <= Til))
    )
RETURN
    COUNTROWS(FILTER(Maaneder,
        NOT ISBLANK([@Verdi]) && NOT ISBLANK([@Senter]) && [@Verdi] < [@Senter])) = 8
```

```DAX
Rødt punkt (frist) =
VAR V = [Frist verdi]
RETURN IF(NOT ISBLANK(V) && (V < [Frist nedre grense] || [Frist 8 under på rad]), V)
```

```DAX
Blått punkt (frist) =
VAR V = [Frist verdi]
RETURN IF(NOT ISBLANK(V) && V > [Frist øvre grense], V)
```

## Slik bygges kortene
1. **Linjediagram**, x-akse `Kalender[Dato]` på månedsnivå, siste 36 måneder. Én indikator
   om gangen: slicer på `indikator`, eller små multipler. Ikke bland indikatorer med ulikt
   nivå i samme kort.
2. **Linjer:** verdien (`Produksjon verdi` / `Frist verdi`, grå med markører), midtlinjen
   (tynn, stiplet) og båndet slik dere allerede tegner det for produksjon.
3. **Prikker:** legg til `Rødt punkt (…)` og `Blått punkt (…)` som to ekstra serier. Sett
   linjebredden til 0 under *Format → Linjer*, slå på markører, og farg dem rødt
   (`#B00020`) og blått (`#1565C0`), litt større enn de vanlige punktene.
4. **Lesehjelp** som tekst under kortene:
   > *Båndet viser normal variasjon. Reager på røde punkter: en måned under båndet, eller
   > åtte måneder på rad under midtlinjen. Blå punkter er uvanlig gode måneder.*

## Tolkning
- **Punkter uten farge, også når de er lavere enn forrige måned:** vanlig variasjon. Ikke
  reager — det er dette kortene er til for å unngå.
- **Rødt i produksjon, ikke ennå i fristprosent:** tempoet har falt; fristene vil som regel
  følge etter. Sjekk de gule og røde åpne fasene.
- **Rødt i fristprosent, produksjonen normal:** problemet er ikke tempoet, men hvilke saker
  som blir liggende. Se på P90 fasetid per fase (`At_Risk_Open_Cases_POWERBI_DAX.md`, Del 3).
- **Røde punkter fra åttemånedersregelen:** nivået har sunket over tid. Det er en sak for
  styringsdialogen, ikke for én enkelt måned.
- **Røde punkter svært ofte for én indikator** (mer enn ca. én gang i året i rolige
  perioder): båndet er for smalt for den indikatoren. Vurder 0,75 eller 1 standardavvik for
  den, eller les enkeltpunkter under båndet som «følg med» og reager bare på
  åttemånedersregelen.
- **Brudd i nivået uten forklaring:** kjør `PELT_Changepoint.py` for å tidfeste det og
  bryte det ned på enhet/fasetittel.
