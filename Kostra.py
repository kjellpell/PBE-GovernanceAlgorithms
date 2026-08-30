# =============================================================================
# SSB KOSTRA → Microsoft Fabric Lakehouse Sync
#
# Fetches selected KOSTRA key-figure tables from SSB PxWebApi v2
# and appends only new rows to each Delta table in the Lakehouse.
#
# Run in a Fabric Notebook with Spark runtime.
# Requires: requests (pre-installed), delta (Fabric default runtime).
#
# Rate limit: 30 requests / 10 seconds per IP  →  throttle built in.
# Cell size limit: 150 000 cells per request   →  chunked fetching if needed.
# =============================================================================

import re
import time
import unicodedata
import requests
import pandas as pd
from datetime import datetime
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]

# Target Lakehouse schema (leave empty string "" for default schema)
LAKEHOUSE_SCHEMA = "kostra"

# Table name prefix  e.g.  kostra_14304_plan_og_byggesaksbehandling
TABLE_PREFIX     = "kostra_"

# SSB base URL
BASE_URL         = "https://data.ssb.no/api/pxwebapi/v2"

# Fetch all time periods available ("*") or only the latest ("top(1)")
TIME_FILTER      = "*"

# Parallel workers for HTTP fetching — stay well within 30 req/10 s.
# Spark writes always run sequentially; SparkSession is not thread-safe for writes.
MAX_WORKERS      = 4

# Seconds to sleep between HTTP requests to respect rate limit
THROTTLE_SLEEP   = 0.5

# Selected KOSTRA key-figure tables. The direct data endpoint is used;
# table discovery and fallback handling are intentionally not needed.
KOSTRA_TABLES = [
    {"id": "14304", "label": "KOSTRA-nøkkeltall for plan- og byggesaksbehandling"},
    {"id": "14317", "label": "KOSTRA-nøkkeltall for fysisk planlegging"},
    {"id": "14324", "label": "KOSTRA-nøkkeltall for bolig"}
]


# =============================================================================
# CELL 1 — Create schema and define helper functions
# =============================================================================

if LAKEHOUSE_SCHEMA:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {LAKEHOUSE_SCHEMA}")
    print(f"Schema '{LAKEHOUSE_SCHEMA}' ready")


def _slugify(text: str) -> str:
    """Convert a Norwegian label to a lowercase snake_case slug."""
    replacements = {"æ": "ae", "ø": "o", "å": "a", "Æ": "ae", "Ø": "o", "Å": "a"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"^[\d:]+\s*", "", text)
    text = re.sub(r"kostra[- ]n.kkeltall\s+for\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:60]


def make_table_name(table_id: str, label: str) -> str:
    slug = _slugify(label)
    return f"{TABLE_PREFIX}{table_id}_{slug}"


def _jsonstat2_to_rows(js2: dict) -> list[dict]:
    """Flatten a JSON-stat2 dataset into a list of dicts."""
    if "_rows" in js2:
        return js2["_rows"]

    ids      = js2["id"]
    sizes    = js2["size"]
    dims     = js2["dimension"]
    values   = js2["value"]
    statuses = js2.get("status", {})

    dim_cats = []
    for dim_id in ids:
        cat_index = dims[dim_id]["category"]["index"]
        ordered   = sorted(cat_index.keys(), key=lambda c: cat_index[c])
        dim_cats.append(ordered)

    rows = []
    for flat_idx, combo in enumerate(product(*[range(s) for s in sizes])):
        row = {}
        for dim_pos, cat_idx in enumerate(combo):
            dim_id    = ids[dim_pos]
            cat_code  = dim_cats[dim_pos][cat_idx]
            cat_label = dims[dim_id]["category"].get("label", {}).get(cat_code, cat_code)
            row[dim_id]             = cat_code
            row[f"{dim_id}_label"] = cat_label

        raw_val        = values[flat_idx]
        row["verdi"]   = float(raw_val) if raw_val is not None else None
        row["_status"] = str(statuses[str(flat_idx)]) if str(flat_idx) in statuses else None
        rows.append(row)

    return rows


def _fetch_chunked(table_id: str, time_filter: str) -> dict:
    """Fetch in chunks when a table exceeds the 150 000-cell limit."""
    meta_url = f"{BASE_URL}/tables/{table_id}/metadata?lang=no&outputFormat=json-stat2"
    meta     = requests.get(meta_url, timeout=30).json()

    dims      = meta.get("id", [])
    dim_values = {}
    for d in dims:
        codes          = list(meta["dimension"][d]["category"]["index"].keys())
        dim_values[d]  = codes

    time_dim   = next(
        (d for d in dims if "Tid" in d or d == meta.get("role", {}).get("time", [None])[0]),
        dims[-1],
    )
    other_dims = [d for d in dims if d != time_dim]
    split_dim  = max(other_dims, key=lambda d: len(dim_values[d])) if other_dims else time_dim

    chunk_size = max(1, 150_000 // (
        len(dim_values[time_dim]) *
        max((len(dim_values[d]) for d in dims if d not in (split_dim, time_dim)), default=1)
    ))

    all_rows = []
    for i in range(0, len(dim_values[split_dim]), chunk_size):
        chunk_codes = dim_values[split_dim][i: i + chunk_size]
        post_url    = f"{BASE_URL}/tables/{table_id}/data"
        body = {
            "query": [
                {"code": split_dim, "selection": {"filter": "item", "values": chunk_codes}},
                {"code": time_dim,  "selection": {"filter": "all",  "values": [time_filter]}},
            ],
            "response": {"format": "json-stat2"},
        }
        r = requests.post(post_url, params={"lang": "no", "outputFormat": "json-stat2"},
                          json=body, timeout=60)
        r.raise_for_status()
        all_rows.extend(_jsonstat2_to_rows(r.json()))
        time.sleep(THROTTLE_SLEEP)

    return {"_rows": all_rows, "label": meta.get("label", table_id)}


def fetch_jsonstat2(table_id: str, time_filter: str = "*") -> dict:
    url    = f"{BASE_URL}/tables/{table_id}/data"
    params = {"lang": "no", "outputFormat": "json-stat2", "valueCodes[*]": "*"}
    resp   = requests.get(url, params=params, timeout=60)

    if resp.status_code == 400:
        return _fetch_chunked(table_id, time_filter)

    resp.raise_for_status()
    return resp.json()


print("Helper functions defined")


# =============================================================================
# CELL 2 — Select table IDs
# =============================================================================

tables = KOSTRA_TABLES
print(f"\nValgt {len(tables)} KOSTRA-tabeller for behandling")


# =============================================================================
# CELL 3 — Fetch all data via parallel HTTP requests
# =============================================================================
# HTTP fetching is parallelised across MAX_WORKERS threads.
# Spark writes are NOT done here — SparkSession is not thread-safe for writes.

def _fetch_one(entry: dict) -> tuple[str, str, list[dict], str]:
    table_id  = entry["id"]
    raw_label = entry["label"]
    js2       = fetch_jsonstat2(table_id, TIME_FILTER)
    label     = js2.get("label", raw_label)
    rows      = _jsonstat2_to_rows(js2)
    time.sleep(THROTTLE_SLEEP)
    return table_id, raw_label, rows, label


print(f"Fetching {len(tables)} tables with {MAX_WORKERS} parallel workers...\n")

fetched: dict[str, dict | None] = {}

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(_fetch_one, t): t["id"] for t in tables}
    for fut in as_completed(futures):
        tid = futures[fut]
        try:
            table_id, raw_label, rows, label = fut.result()
            fetched[table_id] = {"raw_label": raw_label, "rows": rows, "label": label}
            print(f"  Fetched {table_id}: {len(rows):,} rows")
        except Exception as exc:
            print(f"  FAILED  {tid}: {exc}")
            fetched[tid] = None

print(f"\nFetch complete: {sum(1 for v in fetched.values() if v is not None)} succeeded, "
      f"{sum(1 for v in fetched.values() if v is None)} failed")


# =============================================================================
# CELL 4 — Write to Delta tables (sequential — one table at a time)
# =============================================================================

def migrate_metadata_columns(full_name: str) -> None:
    """Rename legacy metadata columns without replacing existing records."""
    if not spark.catalog.tableExists(full_name):
        return

    columns = {field.name for field in spark.table(full_name).schema.fields}
    renames = {
        "value": "verdi",
        "_source_table_id": "kilde_tabell_id",
        "_source_label": "kilde_etikett",
        "_loaded_at": "lastet_tidspunkt",
    }
    for old_name, new_name in renames.items():
        if old_name in columns and new_name not in columns:
            spark.sql(
                f"ALTER TABLE {full_name} RENAME COLUMN {old_name} TO {new_name}"
            )
            columns.remove(old_name)
            columns.add(new_name)


succeeded = []
failed    = []

for entry in tables:
    table_id  = entry["id"]
    raw_label = entry["label"]
    tbl_name  = make_table_name(table_id, raw_label)
    data      = fetched.get(table_id)

    if data is None:
        failed.append((table_id, "fetch failed"))
        continue

    rows  = data["rows"]
    label = data["label"]

    if not rows:
        print(f"  No rows for {tbl_name}, skipping.")
        failed.append((table_id, "no rows"))
        continue

    try:
        # KOSTRA key-figure tables are small (a handful of national indicator
        # tables, at most a few years of history) — small enough to dedupe in
        # pandas rather than a Spark DataFrame join, matching every other
        # script in this repo.
        new_df = pd.DataFrame(rows)
        new_df["kilde_tabell_id"] = table_id
        new_df["kilde_etikett"]   = label

        full_name = f"{LAKEHOUSE_SCHEMA}.{tbl_name}" if LAKEHOUSE_SCHEMA else tbl_name
        migrate_metadata_columns(full_name)

        if spark.catalog.tableExists(full_name):
            existing_df = spark.table(full_name).drop("lastet_tidspunkt").toPandas()
            if set(existing_df.columns) != set(new_df.columns):
                raise ValueError(
                    f"Schema mismatch for {full_name}; existing columns must match "
                    "the current SSB response before new rows can be appended."
                )

            comparison_columns = list(existing_df.columns)
            merged = new_df.merge(
                existing_df[comparison_columns],
                on=comparison_columns,
                how="left",
                indicator=True,
            )
            new_rows_df = merged.loc[merged["_merge"] == "left_only", comparison_columns]
        else:
            new_rows_df = new_df

        new_row_count = len(new_rows_df)
        if new_row_count:
            new_rows_df = new_rows_df.copy()
            new_rows_df["lastet_tidspunkt"] = datetime.now()

            fields = [
                StructField(
                    col,
                    DoubleType() if col == "verdi"
                    else TimestampType() if col == "lastet_tidspunkt"
                    else StringType(),
                    True,
                )
                for col in new_rows_df.columns
            ]
            schema = StructType(fields)
            # NaN (e.g. an all-blank status column) must become None, not a
            # float, before it hits a StringType/TimestampType field.
            spark_rows = [
                tuple(None if pd.isna(v) else v for v in row)
                for row in new_rows_df[[f.name for f in fields]].itertuples(index=False, name=None)
            ]

            spark.createDataFrame(spark_rows, schema=schema) \
                .write.format("delta").mode("append").saveAsTable(full_name)
        print(f"  La til {new_row_count:>7,} nye rader  →  {full_name}")
        succeeded.append(table_id)
    except Exception as exc:
        print(f"  FAILED  {table_id}: {exc}")
        failed.append((table_id, str(exc)))


# =============================================================================
# CELL 5 — Summary
# =============================================================================

print("=" * 70)
print(f"Finished: {len(succeeded)} succeeded, {len(failed)} failed")
if failed:
    print("\nFailed tables:")
    for tid, msg in failed:
        print(f"  {tid}: {msg}")
print("=" * 70)
