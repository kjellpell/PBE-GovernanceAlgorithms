# PELT_Changepoint.py — Power BI and DAX

**Målgruppe:** Analytiker-rapport. Kjøres hver natt; se på siden ved behov.

## Formål
Finne strukturelle skift i gjennomsnittet av `Fristprosent`, `Behandlingstid` og
`Produksjonsdifferanse` som ingen har logget — endret registreringspraksis, systembytte,
regelendring — og bryte siste skift ned på enhet og fasetittel.

PELT ser **bakover**. Den trenger ca. seks perioder etter et brudd før den ser det, så den
er ikke et tidlig varsel. Den er heller ikke riktig verktøy for å evaluere en *kjent*
organisasjonsendring: da kjenner dere datoen, og avbrutt tidsserie (segmentert regresjon)
på den datoen gir et tydeligere svar. Se på siden f.eks. kvartalsvis, og sjekk hvert funnet
brudd mot endringsloggen: hvert brudd skal enten ha en forklaring der, eller utløse
spørsmålet «hva skjedde her?».

Datakilder (overskrives hver natt):
- `analyser.pelt_analyse`
- `analyser.pelt_analyse_detaljer`

CUSUM-driftdeteksjon (`cusum_analyse`, `Trendretning (CUSUM)`) lå tidligere i samme
skript og ble pensjonert i oktober 2026 — se `README.md`. Erstatning: rullerende
12-måneders `Fristprosent` og p-kort i `Fristprosent_Kontrollkort_POWERBI_DAX.md`.

`Fristprosent` følger den **offisielle definisjonen** (samme som
`Seasonal_YTD_ratio_extrapolation.py` og rapportens `[Produserte faser]`): nevner =
produserte faser (både start- og sluttmilepæl satt), teller = de med `innenfor_frist = 1`,
og `%avtalt%`-indikatorer er **med**. Fram til oktober 2026 brukte skriptet
`frist_dager IS NOT NULL` som nevner og holdt `%avtalt%` utenfor. `Behandlingstid` og
`Produksjonsdifferanse` holder fortsatt `%avtalt%` utenfor.

`Produksjonsdifferanse` = faser **startet** i perioden (etter `startmilepaeldato`) minus
faser **avsluttet** i perioden (etter `sluttmilepaeldato`). Til og med september 2026 ble
begge tellingene lagt i perioden for `COALESCE(slutt, start)`, så avsluttede faser nullet
seg ut og serien var bare «fortsatt åpne faser etter startperiode» — den ga
`Stigende`-signal av strukturelle grunner. En live DAX-verdi for samme måltall må bruke
samme definisjon: tell starter mot startdato og avslutninger mot sluttdato (to
`USERELATIONSHIP`-roller mot datotabellen), ikke én datokolonne.

Ukentlige rader har `analyse_dato` = søndagen i ISO-uken. Uke 1 kan derfor starte i
desember (30.12.2024 tilhører uke 1 i 2025). En datotabell som skal matche ukeradene må
bruke ISO-år og ISO-uke (`WEEKNUM(dato, 21)` og ISO-året fra torsdagen i uken), ikke
`YEAR(dato)`.

**Bare hele perioder:** inneværende måned (månedlig) og inneværende ISO-uke (ukentlig)
er ikke med.

## Visualforslag

### 1) PELT-endringspunkter (bekreftede skift)
- Kolonner: `indikator`, `maaltall`, `analyse_dato`, `gjennomsnitt_foer`, `gjennomsnitt_etter`, `endringsstoerrelse`, `endringsretning`
- Sortering: `ABS(endringsstoerrelse)` desc

### 2) Drill-down etter endringspunkt
- Relasjon: `pelt_analyse.endringspunkt_id` -> `pelt_analyse_detaljer.endringspunkt_id`
- Vis bidrag per `dimensjonsverdi` med volumfelt — gjerne `bidrag_rate` og `bidrag_miks`
  som to stablede stolper.
- Drill-down lages bare for siste endringspunkt per serie, og bare når det er nytt nok:
  240 dager for månedlige serier, 90 dager for ukentlige. (PELT trenger minst seks
  perioder etter et endringspunkt, så et månedlig endringspunkt er alltid minst ~6
  måneder gammelt; med 90 dager kom månedlige serier aldri med.)
- Før/etter er **samme vindu som PELT brukte**: fra forrige endringspunkt (eller
  seriestart) til neste endringspunkt (eller siste hele periode), delt ved første dag i
  endringsperioden. Tidligere ble all historikk siden 2015 delt i to.
- `bidrag_til_endring` = `bidrag_rate` + `bidrag_miks`:
  - `bidrag_rate` — segmentet ble bedre/verre (vekt etter × endring i segmentets verdi)
  - `bidrag_miks` — segmentet ble større/mindre (endring i vekt × verdi før)
  - Summert over segmentene gir de totalt skiftet i det volumvektede snittet.
  - For `Produksjonsdifferanse` er `gjennomsnitt_foer`/`etter` per periode (sum delt på
    antall perioder i vinduet), og `bidrag_miks` er 0 — måltallet er en sum, ikke et snitt.

## DAX-forslag

```DAX
Siste endringspunkt dato =
CALCULATE(
    MAX(pelt_analyse[analyse_dato]),
    ALLEXCEPT(
        pelt_analyse,
        pelt_analyse[indikator],
        pelt_analyse[maaltall],
        pelt_analyse[granularitet]
    )
)
```

```DAX
Endringspunkt størrelse =
CALCULATE(
    MAX(pelt_analyse[endringsstoerrelse]),
    pelt_analyse[analyse_dato] = [Siste endringspunkt dato]
)
```

## Tolkning
- PELT = skift i gjennomsnittet, med tidspunkt og størrelse — ikke årsak. Et brudd nær en
  kjent endring er en indikasjon, ikke et bevis.
- PELT = bekreftet skift med tidspunkt og størrelse. Straffeleddet bruker et robust
  støyestimat (MAD av førstedifferanser), så et stort gammelt skift ikke skjuler et
  moderat nytt. Ulempen er litt flere falske endringspunkter i helt flate serier (ca. 5 %
  på en månedsserie siden 2015).
- `pelt_analyse_detaljer` svarer på hvor skiftet kommer fra (enhet/fasetittel).
