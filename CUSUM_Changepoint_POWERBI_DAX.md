# CUSUM_Changepoint.py — Power BI and DAX

**Målgruppe:** Analytiker-rapport — hvorfor signalet utløste akkurat denne måneden, og
ikke forrige, krever å forstå baseline-metoden (samme måned de tre foregående årene).

## Formål
Skille mellom tidlig driftssignal (CUSUM) og bekreftet strukturelt skift (PELT), med drill-down for årsaksforklaring.

Datakilder:
- `analyser.cusum_analyse` — signal/score only; `signal` and `cusum_positiv`/`cusum_negativ`
  are genuinely not reproducible in DAX (the recursive `max(0, S_{t-1} + ...)` reset at
  each step is the same class of sequential logic that made the throughput monitor's
  streak measure fragile — not something to fake in a measure). The raw underlying value
  (Fristprosent/Behandlingstid/Produksjonsdifferanse) is deliberately **not** stored here
  — it's a live DAX measure against `Faser`, defined elsewhere in the model.
- `analyser.pelt_analyse`
- `analyser.pelt_analyse_detaljer`

`Fristprosent` følger den **offisielle definisjonen** (samme som
`Seasonal_YTD_ratio_extrapolation.py` og rapportens `[Produserte faser]`): nevner =
produserte faser (både start- og sluttmilepæl satt), teller = de med `innenfor_frist = 1`,
og `%avtalt%`-indikatorer er **med**. Fram til oktober 2026 brukte CUSUM
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

## Baseline og tilstand (endret oktober 2026)

**Baseline:** hver periode sammenlignes med **samme kalendermåned (eller ISO-uke) de tre
foregående årene** — aldri med sitt eget år. Juli måles mot tidligere juli-måneder, så
sommerfallet ser ikke ut som drift hvert år, og referansen er nylig, ikke 2015. Spredningen
(sigma) er samlet over alle måneder i de tre årene. En måned/uke uten minst to tidligere
år får ingen rad (derfor starter serien to år etter `START_YEAR`). Konsekvens: et skift
som varer i flere år blir gradvis «ny normal» — det er riktig for et tidlig varsel, men
betyr at et vedvarende skift tones ned i år 2–3.

**Hendelse vs. tilstand:**
- `signal` / `signalretning` er **hendelsen**: `TRUE` / `Stigende` / `Synkende` bare i
  perioden akkumulatoren krysser terskelen (deretter nullstilles den).
- `tilstand` er **tilstanden**: siste signalerte retning, holdt til et motsatt signal kommer
  eller siden har vært rolig i tre perioder på rad (`CUSUM_RECOVERY_PERIODS`). Rolig betyr
  at akkumulatoren er under halve terskelen **og** at perioden ikke trakk videre i samme
  retning.

Tidligere leste bordkortet `signalretning` på siste rad. Under en vedvarende nedgang viste
det da `Synkende` bare i måneden signalet fyrte og `Stabil` ellers — i en simulering
halvparten av månedene. Bruk `tilstand` på alle kort som skal svare på «hvordan går det nå».

**Bare hele perioder:** inneværende måned (månedlig) og inneværende ISO-uke (ukentlig)
er ikke med, så `[Siste CUSUM-dato]` er siste hele måned/uke.

## Visualforslag

### 1) CUSUM-linje (tidlig varsling)
- X-akse: `analyse_dato`
- Linjer: `cusum_positiv`, `cusum_negativ` (from `cusum_analyse`) — optionally add the raw
  value as a third line from the live DAX measure above, joined on `indikator` +
  `analyse_dato`
- Referanselinje: terskel (`CUSUM_H`)
- Filter: `indikator`, `maaltall`, `granularitet`

### 2) Aktiv signal-tabell
- Kolonner: `indikator`, `maaltall`, `granularitet`, `tilstand`, `analyse_dato`
- Filter: siste `analyse_dato` per serie og `tilstand <> "Stabil"`
- (Historikk over når signalene fyrte: filter `signal = TRUE`, kolonne `signalretning`.)

### 3) PELT-endringspunkter (bekreftede skift)
- Kolonner: `indikator`, `maaltall`, `analyse_dato`, `gjennomsnitt_foer`, `gjennomsnitt_etter`, `endringsstoerrelse`, `endringsretning`
- Sortering: `ABS(endringsstoerrelse)` desc

### 4) Drill-down etter endringspunkt
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
Siste CUSUM-dato =
CALCULATE(
    MAX(cusum_analyse[analyse_dato]),
    ALLEXCEPT(
        cusum_analyse,
        cusum_analyse[indikator],
        cusum_analyse[maaltall],
        cusum_analyse[granularitet]
    )
)
```

```DAX
Trendretning (CUSUM) =
CALCULATE(
    SELECTEDVALUE(cusum_analyse[tilstand]),
    cusum_analyse[analyse_dato] = [Siste CUSUM-dato]
)
```

`tilstand` is written as `Stigende`/`Synkende`/`Stabil` — plain board vocabulary, so a
board-level trend card reads it with no translation layer. This is the one to put on a
board card instead of the CUSUM line chart: the statistically-tested signal (CUSUM
against the same month in the three prior years), held as a state rather than flashing
for one period, as one plain word — no chart, no `cusum_positiv`/`cusum_negativ` numbers.

```DAX
Har aktiv CUSUM signal =
[Trendretning (CUSUM)] IN { "Stigende", "Synkende" }
```

```DAX
Signal fyrte siste periode =
CALCULATE(
    MAX(cusum_analyse[signal]),
    cusum_analyse[analyse_dato] = [Siste CUSUM-dato]
) = TRUE()
```

```DAX
Antall aktive signaler =
COUNTROWS(
    FILTER(
        VALUES(cusum_analyse[indikator]),
        CALCULATE([Har aktiv CUSUM signal])
    )
)
```

```DAX
CUSUM bekreftet av PELT =
VAR GjeldendeIndikator = SELECTEDVALUE(cusum_analyse[indikator])
VAR GjeldendeMaaltall = SELECTEDVALUE(cusum_analyse[maaltall])
VAR GjeldendeGranularitet = SELECTEDVALUE(cusum_analyse[granularitet])
RETURN
    CALCULATE(
        COUNTROWS(pelt_analyse),
        pelt_analyse[indikator] = GjeldendeIndikator,
        pelt_analyse[maaltall] = GjeldendeMaaltall,
        pelt_analyse[granularitet] = GjeldendeGranularitet,
        REMOVEFILTERS(pelt_analyse)
    ) > 0
```

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
- CUSUM = tidlig driftssignal. `tilstand` = gjeldende retning; `signal` = perioden den
  ble bekreftet.
- PELT = bekreftet skift med tidspunkt og størrelse. Straffeleddet bruker et robust
  støyestimat (MAD av førstedifferanser), så et stort gammelt skift ikke skjuler et
  moderat nytt. Ulempen er litt flere falske endringspunkter i helt flate serier (ca. 5 %
  på en månedsserie siden 2015).
- `pelt_analyse_detaljer` svarer på hvor skiftet kommer fra (enhet/fasetittel).
