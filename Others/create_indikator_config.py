#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import argparse

_INDICATORS = [
    ("Byggesak 3 uker", 3, 0.70, 0.90),
    ("Byggesak 12 uker", 12, 0.70, 0.90),
    ("Delesak 3 uker", 3, 0.70, 0.90),
    ("Delesak 12 uker", 12, 0.70, 0.90),
    ("Seksjonering", 12, 0.70, 0.90),
    ("Oppmåling", 18, 0.70, 0.90),
    ("Offentlig ettersyn privat plan", 12, 0.70, 0.90),
    ("Til politisk behandling", 18, 0.70, 0.90),
    ("Sykefravær", 6, 0.70, 0.90),
    ("Turnover", 6, 0.70, 0.90),
    ("AML", 0, 0.70, 0.90),
]


def main(database: str | None, table: str):
    spark = SparkSession.builder.getOrCreate()

    schema = StructType([
        StructField("indikator", StringType(), False),
        StructField("target", IntegerType(), False),
        StructField("target_red", DoubleType(), False),
        StructField("target_amber", DoubleType(), False),
    ])

    rows = [(i[0], int(i[1]), float(i[2]), float(i[3])) for i in _INDICATORS]
    df = spark.createDataFrame(rows, schema=schema)

    if database:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
        full_table = f"{database}.{table}"
    else:
        full_table = table

    # Write as a Delta table in the Lakehouse. Overwrites existing table.
    df.write.format("delta").mode("overwrite").saveAsTable(full_table)

    print(f"Wrote {df.count()} rows to table {full_table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create IndikatorConfig table in Lakehouse")
    parser.add_argument("--database", help="Target database/catalog name (optional)")
    parser.add_argument("--table", default="IndikatorConfig", help="Target table name")
    args = parser.parse_args()
    main(args.database, args.table)
