# =============================================================================
# Building application type — primary product code per Prosess_id.
#
# For each process recorded in Fakturalinjer, identifies the product code
# that accounts for the largest total invoice amount. This is the generalized,
# unfiltered version of the soeknadsgruppe concept used in BuildingForecast —
# expressed as a raw product code so downstream semantic models can join
# prisliste_varer for names and categories, and Prosesser for case metadata.
#
# No date filter is applied — all rows in Fakturalinjer are included.
# No indicator filter — covers all case types, not just building permits.
#
# Output table : building_application_type  (full overwrite, nightly)
# Sources      : Fakturalinjer
# Schedule     : nightly
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

spark    = SparkSession.builder.getOrCreate()
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")

OUTPUT_TABLE = "building_application_type"


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
    pk_faser           STRING    NOT NULL,
    primary_product_code STRING    NOT NULL,
    kjoert_tidspunkt          TIMESTAMP NOT NULL,
    kjoere_id            STRING    NOT NULL
)
USING DELTA
COMMENT 'Primary product code per Prosess_id, derived from the largest total invoice amount in Fakturalinjer. Join primary_product_code -> prisliste_varer.varenummer for description and category. Join prosess_id -> Prosesser for case metadata.'
""")

print(f"Output table {OUTPUT_TABLE} ready")


# =============================================================================
# CELL 2 — Sum invoice amounts by Prosess_id and Produktnr
# =============================================================================
# Produktnr in Fakturalinjer is Long — cast to STRING for the output column.
# Group before windowing to keep the working set small.

totals = spark.sql("""
    SELECT
        CAST(fk_faser AS STRING) AS fk_faser,
        CAST(p  roduktnr  AS STRING) AS produktnr,
        SUM(linjebeloep)           AS total_belop
        FROM Fakturalinjer fakturalinjer
        INNER JOIN saksbehandling.faser faser
                ON faser.pk_faser = fakturalinjer.fk_faser
        INNER JOIN felles.indikatorer indikatorer
                ON indikatorer.pk_indikator = faser.indikator
    WHERE fk_faser  IS NOT NULL
      AND produktnr   IS NOT NULL
      AND linjebeloep IS NOT NULL
            AND indikatorer.fagomraade IN ('Byggesak', 'Eiendomssak')
    GROUP BY fk_faser, produktnr
""")

print(f"Unique (Prosess_id, Produktnr) pairs: {totals.count():,}")


# =============================================================================
# CELL 3 — Rank and select dominant product code per Prosess_id
# =============================================================================
# Tie-break by Produktnr DESC for determinism — mirrors BuildingForecast logic.

w_rank = Window.partitionBy("fk_faser").orderBy(
    F.col("total_belop").desc(),
    F.col("produktnr").desc(),
)

result = (
    totals
    .withColumn("rn", F.row_number().over(w_rank))
    .filter(F.col("rn") == 1)
    .withColumn("kjoert_tidspunkt", F.lit(datetime.now()).cast("timestamp"))
    .withColumn("kjoere_id",       F.lit(BATCH_ID))
    .select(
        F.col("fk_faser"),
        F.col("produktnr").alias("primary_product_code"),
        "kjoert_tidspunkt",
        "kjoere_id",
    )
)

print(f"Rows to write (one per Prosess_id): {result.count():,}")
result.orderBy("fk_faser").show(10, truncate=False)


# =============================================================================
# CELL 4 — Write (full overwrite)
# =============================================================================

result.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

written = spark.sql(f"""
    SELECT
        COUNT(*)                             AS total_rows,
        COUNT(DISTINCT fk_faser)           AS unique_prosess_ids,
        COUNT(DISTINCT primary_product_code) AS unique_product_codes
    FROM {OUTPUT_TABLE}
        WHERE kjoere_id = '{BATCH_ID}'
""")
print("Written:")
written.show()


# =============================================================================
# CELL 5 — Power BI visual guidance
# =============================================================================
#
# OUTPUT TABLE → building_application_type
# One row per Prosess_id. Full overwrite on each run.
#
# RECOMMENDED JOINS IN SEMANTIC MODEL (Power BI / Direct Lake)
#
#   building_application_type.primary_product_code
#       → prisliste_varer.varenummer
#   Gives: product description, application size category (small/medium/large),
#          fee type, and any other dimensions on prisliste_varer.
#
#   building_application_type.fk_faser
#       → Prosesser.pk_faser
#   Gives: Indikator, Saksnummer, Sluttdato, Tidsbruk, Frist, etc.
#
# STACKED BAR — Case volume by application type over time
#   X axis:  case close year/month (from Prosesser.Sluttdato via join)
#   Stacks:  product description or size category (from prisliste_varer)
#   Filter:  Indikator (e.g. Byggesak 12 uker, Byggesak 3 uker)
#
# BAR CHART — Application size distribution
#   Categories: small / medium / large (from prisliste_varer category column)
#   Values:     COUNTROWS per category
#   Slicer:     year, Indikator
#
#   DAX — share of large applications:
#     Andel store søknader =
#         DIVIDE(
#             CALCULATE(COUNTROWS(building_application_type),
#                       prisliste_varer[category] = "Large"),
#             COUNTROWS(building_application_type)
#         )
#
#   DAX — application count per product:
#     Antall søknader =
#         COUNTROWS(building_application_type)
#
# CROSS-ANALYSIS WITH CASE DATA
#   Join to Prosesser to compare processing time (Tidsbruk) or deadline
#   compliance (innenfor_frist) by application type. Example:
#
#   DAX — average processing days by application type:
#     Gj.sn. tidsbruk =
#         AVERAGEX(
#             RELATEDTABLE(Prosesser),
#             Prosesser[Tidsbruk]
#         )
#
# NOTE: Apply category filters through prisliste_varer in the semantic model,
# not directly on primary_product_code — keeps filter logic in one place.
