# =============================================================================
# Building application type — primary product code per Prosess_id.
#
# For each process recorded in Fakturalinjer, identifies the product code
# that accounts for the largest total invoice amount. This is the generalized,
# unfiltered version of the application_group concept used in BuildingForecast —
# expressed as a raw product code so downstream semantic models can join
# Pricelist_items for names and categories, and Prosesser for case metadata.
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
    prosess_id           STRING    NOT NULL,
    primary_product_code STRING    NOT NULL,
    computed_at          TIMESTAMP NOT NULL,
    batch_id             STRING    NOT NULL
)
USING DELTA
COMMENT 'Primary product code per Prosess_id, derived from the largest total invoice amount in Fakturalinjer. Join primary_product_code -> Pricelist_items.varenr for description and category. Join prosess_id -> Prosesser for case metadata.'
""")

print(f"Output table {OUTPUT_TABLE} ready")


# =============================================================================
# CELL 2 — Sum invoice amounts by Prosess_id and Produktnr
# =============================================================================
# Produktnr in Fakturalinjer is Long — cast to STRING for the output column.
# Group before windowing to keep the working set small.

totals = spark.sql("""
    SELECT
        CAST(Prosess_id AS STRING) AS prosess_id,
        CAST(Produktnr  AS STRING) AS produktnr,
        SUM(Linje_belop)           AS total_belop
    FROM Fakturalinjer
    WHERE Prosess_id  IS NOT NULL
      AND Produktnr   IS NOT NULL
      AND Linje_belop IS NOT NULL
    GROUP BY Prosess_id, Produktnr
""")

print(f"Unique (Prosess_id, Produktnr) pairs: {totals.count():,}")


# =============================================================================
# CELL 3 — Rank and select dominant product code per Prosess_id
# =============================================================================
# Tie-break by Produktnr DESC for determinism — mirrors BuildingForecast logic.

w_rank = Window.partitionBy("prosess_id").orderBy(
    F.col("total_belop").desc(),
    F.col("produktnr").desc(),
)

result = (
    totals
    .withColumn("rn", F.row_number().over(w_rank))
    .filter(F.col("rn") == 1)
    .withColumn("computed_at", F.lit(datetime.now()).cast("timestamp"))
    .withColumn("batch_id",    F.lit(BATCH_ID))
    .select(
        "prosess_id",
        F.col("produktnr").alias("primary_product_code"),
        "computed_at",
        "batch_id",
    )
)

print(f"Rows to write (one per Prosess_id): {result.count():,}")
result.orderBy("prosess_id").show(10, truncate=False)


# =============================================================================
# CELL 4 — Write (full overwrite)
# =============================================================================

result.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

written = spark.sql(f"""
    SELECT
        COUNT(*)                             AS total_rows,
        COUNT(DISTINCT prosess_id)           AS unique_prosess_ids,
        COUNT(DISTINCT primary_product_code) AS unique_product_codes
    FROM {OUTPUT_TABLE}
    WHERE batch_id = '{BATCH_ID}'
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
#       → Pricelist_items.varenr
#   Gives: product description, application size category (small/medium/large),
#          fee type, and any other dimensions on Pricelist_items.
#
#   building_application_type.prosess_id
#       → Prosesser.Prosess_id
#   Gives: Indikator, Saksnummer, Sluttdato, Tidsbruk, Frist, etc.
#
# STACKED BAR — Case volume by application type over time
#   X axis:  case close year/month (from Prosesser.Sluttdato via join)
#   Stacks:  product description or size category (from Pricelist_items)
#   Filter:  Indikator (e.g. Byggesak 12 uker, Byggesak 3 uker)
#
# BAR CHART — Application size distribution
#   Categories: small / medium / large (from Pricelist_items category column)
#   Values:     COUNTROWS per category
#   Slicer:     year, Indikator
#
#   DAX — share of large applications:
#     Andel store søknader =
#         DIVIDE(
#             CALCULATE(COUNTROWS(building_application_type),
#                       Pricelist_items[category] = "Large"),
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
# NOTE: Apply category filters through Pricelist_items in the semantic model,
# not directly on primary_product_code — keeps filter logic in one place.
