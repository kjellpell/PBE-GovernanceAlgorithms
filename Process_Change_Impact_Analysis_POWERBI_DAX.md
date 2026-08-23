# Process_Change_Impact_Analysis.py — Power BI and DAX

## Formål
Denne siden skal svare på:
- Virket en konkret prosessendring — ikke bare "gikk tallene opp etterpå?"
- Er en observert endring reell, eller forklares den av sesong/andre trender?

Datakilde:
- `analyser.prosessendring_effekt`

**Viktig:** dette er en screeningsanalyse for en liten-N-populasjon (typisk
~30 saker/år per fase), ikke en presis kontrollert studie. Les `Tolkning`
nedenfor før du presenterer et tall herfra som et endelig svar.

## Anbefalte visualer

### 1) KPI-kort per endring
- `effekt_retning`, `effekt_estimat`, `konfidens_nedre`–`konfidens_ovre`
- Filter: `snapshot_dato = MAX(snapshot_dato)` per `endring_navn`

### 2) Konfidensintervall over tid (viktigste visual)
- X-akse: `snapshot_dato`
- Y-akse: `effekt_estimat` som linje, `konfidens_nedre`/`konfidens_ovre` som bånd
- Små multipler: `endring_navn`
- Shows: intervallet skal smalne inn etter hvert som `tilstrekkelig_moden` går fra false til true

### 3) Oversiktstabell — alle konfigurerte endringer
- `endring_navn`, `maaltall`, `virkningsdato`, `effekt_retning`, `p_verdi`,
  `har_kontrollgruppe`, `lav_styrke`, `tilstrekkelig_moden`, `pelt_stotte`
- Bruk `har_kontrollgruppe`/`lav_styrke`/`pelt_stotte` som forbeholdskolonner, ikke som styrende filter

## DAX-forslag

### Siste snapshot per endring
```DAX
Siste prosessendring-snapshot =
CALCULATE(
    MAX(prosessendring_effekt[snapshot_dato]),
    ALLEXCEPT(prosessendring_effekt, prosessendring_effekt[endring_navn])
)
```

### Konfidensintervall-bredde (for å visualisere innsnevring)
```DAX
KI-bredde =
prosessendring_effekt[konfidens_ovre] - prosessendring_effekt[konfidens_nedre]
```

### Effekt fargekode
```DAX
Effekt fargekode =
SWITCH(
    MAX(prosessendring_effekt[effekt_retning]),
    "Forbedring", "#2E7D32",
    "Forverring", "#B00020",
    "Ingen praktisk effekt", "#F9A825",
    "Ingen sikker effekt", "#757575",
    "#BDBDBD"
)
```

## Slicer-oppsett
- `endring_navn`
- `maaltall`
- `snapshot_dato`

## Tolkning
- `"Ingen praktisk effekt"` betyr endringen er statistisk sikker (p < 0,05) men mindre enn
  den konfigurerte minste praktiske effekten — statistisk ekte, men for liten til å bry seg om.
  `"Ingen sikker effekt"` betyr p ≥ 0,05 — ikke nok bevis for at effekten er reell i det hele tatt.
  Ikke slå disse sammen når du presenterer resultatet.
- `har_kontrollgruppe = false` betyr sesong-/sekulær drift IKKE er kontrollert for i den raden —
  behandle estimatet som en ukorrigert før/etter-observasjon, ikke som en kausal konklusjon.
- `lav_styrke = true` betyr volumet så vidt klarer minstekravet — p-verdien og estimatet bør
  tolkes som en indikasjon, ikke et presist tall.
- `tilstrekkelig_moden = false` betyr etter-vinduet ikke er fullt forløpt ennå. Med ~30
  saker/år og et helårsvindu tar det typisk ~12 måneder etter `virkningsdato` før et estimat
  er modent — en tidlig rad er en trend å følge med på, ikke et svar.
- `pelt_stotte` er et støttesignal fra den uavhengige `CUSUM_Changepoint.py`-analysen, ikke
  input til `effekt_retning` — bruk det til å styrke eller svekke tilliten til funnet, aldri
  som eneste grunnlag.
