# =============================================================================
# Cohort analysis — tracks resolution rate of cases grouped by intake month.
# Answers: are cases received recently resolving at the same rate as
# historical cohorts, or are they accumulating?
#
# Output table: kohortanalyse
# Power BI visual: heatmap — cohort month on Y axis, weeks since intake
# on X axis, cell colour = fraction still open. Dark = slow resolution.
#
# Schedule: nightly after main data pipeline.
# Minimum cohort size: 10 cases. Suppress smaller cohorts.
# History: uses all available data — 15-20 years gives stable baseline.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID        = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY           = date.today()
MIN_COHORT_SIZE = 10
MAX_WEEKS       = 26  # track cohorts for up to 26 weeks after intake
START_YEAR      = 2015  # exclude data before this year — adjust if older data is reliable


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS kohortanalyse (
    indikator                   STRING      NOT NULL,
    analyse_date                DATE        NOT NULL,  -- første dag i mottaksmåneden
    kohortstoerrelse            INT         NOT NULL,  -- saker mottatt den måneden
    uker_siden_mottak           INT         NOT NULL,  -- uker siden kohortmottak
    aapne_saker_antall          INT         NOT NULL,  -- saker fortsatt åpne denne uken
    andel_aapne                 DOUBLE      NOT NULL,  -- andel åpne saker (0-1)
    andel_aapne_historisk       DOUBLE,                -- historisk gjennomsnitt for samme uke
    avvik_historisk             DOUBLE,                -- nåværende minus historisk
    er_nylig_kohort             BOOLEAN     NOT NULL,  -- siste seks måneder = true
    kjoert_tidspunkt            TIMESTAMP   NOT NULL,
    kjoere_id                   STRING      NOT NULL
)
USING DELTA
COMMENT 'Kohorters ferdigstillingsrate per mottaksmåned. Én rad per kombinasjon av kohort og uke.'
""")

print("kohortanalyse-tabellen er klar")


# =============================================================================
# CELL 2 — Load case data
# =============================================================================
# One row per case with intake month and days to resolution (or NULL if open).
# Uses startmilepaeldato for intake, sluttmilepaeldato for resolution.

cases = spark.sql(f"""
    SELECT
        pr.indikator                          AS indikator,
        CAST(DATE_TRUNC('MONTH', pr.startmilepaeldato) AS DATE) AS analyse_date,
        pr.startmilepaeldato                  AS mottaksdato,
        pr.sluttmilepaeldato                  AS sluttdato,
        CASE
            WHEN pr.sluttmilepaeldato IS NOT NULL
            THEN DATEDIFF(pr.sluttmilepaeldato, pr.startmilepaeldato)
            ELSE NULL
        END                                    AS dager_til_ferdig,
        CASE
            WHEN pr.sluttmilepaeldato IS NULL THEN 1 ELSE 0
        END                                    AS er_aapen
    FROM saksbehandling.faser pr
    INNER JOIN felles.indikator indikatorer
        ON indikatorer.pk_indikator = pr.indikator
        WHERE pr.startmilepaeldato IS NOT NULL
            AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
            AND YEAR(pr.startmilepaeldato) >= {START_YEAR}
""").toPandas()

cases["analyse_date"] = pd.to_datetime(cases["analyse_date"]).dt.normalize()
cases["mottaksdato"]  = pd.to_datetime(cases["mottaksdato"])
cases["sluttdato"]          = pd.to_datetime(cases["sluttdato"])

print(f"Saker lastet: {len(cases):,}")
print(f"Indikatorer:  {cases['indikator'].nunique()}")
print(f"Datointervall: {cases['mottaksdato'].min().date()} → {cases['mottaksdato'].max().date()}")


# =============================================================================
# CELL 3 — Compute cohort resolution rates
# =============================================================================

results = []
today_ts = pd.Timestamp(TODAY)

for indikator, ind_cases in cases.groupby("indikator"):

    # All distinct cohort months for this indicator
    cohort_months = sorted(ind_cases["analyse_date"].unique())

    # Cutoff for "recent" cohorts — last 6 months
    _rc_yr, _rc_mo  = TODAY.year, TODAY.month - 6
    if _rc_mo <= 0:
        _rc_mo += 12; _rc_yr -= 1
    recent_cutoff = pd.Timestamp(_rc_yr, _rc_mo, 1)
    for cohort_start in cohort_months:
        kohort_saker = ind_cases[
            ind_cases["analyse_date"] == cohort_start
        ].copy()

        kohortstoerrelse = len(kohort_saker)
        if kohortstoerrelse < MIN_COHORT_SIZE:
            continue

        cohort_start = pd.Timestamp(cohort_start)
        is_recent     = cohort_start >= recent_cutoff

        # For each week bucket up to MAX_WEEKS, count how many cases
        # from this cohort were still open at that point in time.
        for uke in range(1, MAX_WEEKS + 1):
            reference_date = cohort_start + pd.Timedelta(weeks=uke)

            # Can't measure future weeks for recent cohorts
            if reference_date > today_ts:
                break

            # Cases still open at reference_date:
            # either never finished, or finished after reference_date
            aapne_saker_antall = len(kohort_saker[
                kohort_saker["sluttdato"].isna() |
                (kohort_saker["sluttdato"] > reference_date)
            ])

            andel_aapne = aapne_saker_antall / kohortstoerrelse

            results.append({
                "indikator":             indikator,
                "analyse_date":          cohort_start.date(),
                "kohortstoerrelse":      kohortstoerrelse,
                "uker_siden_mottak":     uke,
                "aapne_saker_antall":    aapne_saker_antall,
                "andel_aapne":           round(andel_aapne, 4),
                "er_nylig_kohort":       is_recent,
            })

print(f"Cohort rows computed: {len(results):,}")


# =============================================================================
# CELL 4 — Compute historical baseline per indicator × week
# =============================================================================
# Historical average andel_åpen at each week across all non-recent cohorts.
# Used to compare recent cohorts against — are they resolving faster or slower?
# Uses trimmed mean (drop top and bottom 10%) to exclude exceptional years.

df = pd.DataFrame(results)

historical = (
    df[~df["er_nylig_kohort"]]
    .groupby(["indikator", "uker_siden_mottak"]) ["andel_aapne"]
    .apply(lambda x: float(np.mean(
        np.sort(x.to_numpy())[
            max(0, int(len(x) * 0.1)) : max(1, int(len(x) * 0.9))
        ]
    )) if len(x) >= 5 else float(x.mean()))
    .reset_index()
    .rename(columns={"andel_aapne": "andel_aapne_historisk"})
)

df = df.merge(historical, on=["indikator", "uker_siden_mottak"], how="left")
df["avvik_historisk"] = df["andel_aapne"] - df["andel_aapne_historisk"]
df["avvik_historisk"] = df["avvik_historisk"].round(4)

print(f"Historisk grunnlinje beregnet for {historical['indikator'].nunique()} indikatorer")


# =============================================================================
# CELL 5 — Write to Lakehouse
# =============================================================================

if df.empty:
    print("Ingen kohortresultater å skrive.")
else:
    df["kjoert_tidspunkt"] = datetime.now()
    df["kjoere_id"]        = BATCH_ID

    # Idempotent — full overwrite since cohort history is recomputed each run.
    # Open cases change as they resolve, so historical rows can change too.
    results_spark = spark.createDataFrame(df)
    results_spark.write.mode("overwrite").saveAsTable("kohortanalyse")

    print(f"\nkohortanalyse skrevet: {len(df):,} rader")
    print(f"Indikatorer: {df['indikator'].nunique()}")
    print(f"Kohorter:    {df['analyse_date'].nunique()}")
    print(f"Nylige kohorter markert: {df[df['er_nylig_kohort']]['analyse_date'].nunique()}")

    # Summary — recent cohorts vs historical baseline at week 12
    print("\n=== RECENT COHORTS VS BASELINE AT WEEK 12 ===")
    spark.sql("""
        SELECT
                        indikator,
                        analyse_date,
                        kohortstoerrelse,
                        ROUND(andel_aapne * 100, 1)             AS andel_aapne,
                        ROUND(andel_aapne_historisk * 100, 1)   AS andel_aapne_historisk,
                        ROUND(avvik_historisk * 100, 1)         AS avvik_prosentpoeng
                FROM kohortanalyse
                WHERE uker_siden_mottak = 12
                    AND er_nylig_kohort = TRUE
                    AND andel_aapne_historisk IS NOT NULL
                ORDER BY avvik_historisk DESC, indikator
    """).show(30, truncate=False)


# =============================================================================
# CELL 6 — Power BI visual notes
# =============================================================================
# HEATMAP (primary visual):
#   Rows:    analyse_date (mottaksmåned) — nyere kohorter øverst
#   Columns: uker_siden_mottak (1 to 26)
#   Values:  andel_aapne — format as %
#   Colour:  gradient dark (high %) → light (low %)
#            Dark cell = many cases still open at that week = slow resolution
#   Filter:  er_nylig_kohort = TRUE for governance team view
#            Remove filter for full historical view
#
# LINE CHART (secondary visual — recent vs historical):
#   X axis:  uker_siden_mottak
#   Lines:   andel_aapne per nyere analyse_date (one line per month)
#            andel_aapne_historisk as a single reference line (trimmed mean)
#   Shading: area between recent lines and historical line
#            Above historical = resolving slower than normal
#            Below historical = resolving faster than normal
#
# KEY SIGNAL:
#   A recent cohort line consistently above the historical reference line
#   means cases from that intake month are sitting longer than normal.
#   This appears in the portfolio before it appears in frist%.
#   Governance team acts here — board sees it later in portfolio age chart.
