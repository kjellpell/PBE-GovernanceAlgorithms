# =============================================================================
# Planning case timeline — chronological milestone events per Prosess_id.
#
# Combines system milestones from Milepeler with invoiced meeting events from
# Fakturalinjer (Avklaringsmøte and Kart- og bestemmelsesmøte). These two
# meeting types are not recorded in Milepeler — they are invoiced events
# identified by product code.
#
# The result is a flat event log enabling time-between-events analysis:
# time from Oppstartsmøte to first Avklaringsmøte, between consecutive
# Avklaringsmøter, from last Avklaringsmøte to Planforslag, and so on.
#
# Milestones from Milepeler (Milepel column):
#   Bestilling av oppstartsmøte
#   Innkalling til oppstartsmøte          (OM completed — normal)
#   P02c Område- og prosessavklaring til fagkyndig  (OM completed — normal)
#   P02 Område/prosessavklaring           (OM completed — normal)
#   Planinitiativet er stoppet            (case stopped)
#   P02a Behandling av planinitiativ/-forslag stoppes/avvises  (case stopped)
#   00e Sak trukket/utgått/uaktuell       (case stopped)
#   Planforslag mottatt
#   Offentlig ettersyn
#
# Meeting events from Fakturalinjer (Produktnr is Long — numeric codes only):
#   Avklaringsmøte:            62432 (Simple), 62433 (Medium), 62434 (Complex)
#   Kart- og bestemmelsesmøte: 62441, 62442 (Simple), 62443 (Medium), 62444 (Complex)
#
# Multiple Avklaringsmøte and KB rows per Prosess_id are expected.
#
# Output table : planning_case_timeline  (full overwrite, nightly)
# Sources      : Milepeler, Fakturalinjer
# Schedule     : nightly
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

spark    = SparkSession.builder.getOrCreate()
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")

MILEPELER_MILESTONES = [
    "Bestilling av oppstartsmøte",
    # OM completed normally (SLUTTOPPGAVE_OM_OK in PlanningConfig)
    "Innkalling til oppstartsmøte",
    "P02c Område- og prosessavklaring til fagkyndig",
    "P02 Område/prosessavklaring",
    # Case stopped (SLUTTOPPGAVE_STOPPED in PlanningConfig)
    "Planinitiativet er stoppet",
    "P02a Behandling av planinitiativ/-forslag stoppes/avvises",
    "00e Sak trukket/utgått/uaktuell",
    # Later planning stages
    "Planforslag mottatt",
    "Offentlig ettersyn",
]

# Avklaringsmøte product codes — Produktnr is Long, compare as numeric
AM_PRODUKTNR = [62432, 62433, 62434]

# Kart- og bestemmelsesmøte product codes
KB_PRODUKTNR = [62441, 62442, 62443, 62444]

# Sort order for deterministic lag when two events share the same end_date
MILESTONE_SORT = {
    "Bestilling av oppstartsmøte":                          1,
    "Innkalling til oppstartsmøte":                         2,
    "P02c Område- og prosessavklaring til fagkyndig":       2,
    "P02 Område/prosessavklaring":                          2,
    "Avklaringsmøte":                                       3,
    "Kart- og bestemmelsesmøte":                            4,
    "Planforslag mottatt":                                  5,
    "Offentlig ettersyn":                                   6,
    "Planinitiativet er stoppet":                           7,
    "P02a Behandling av planinitiativ/-forslag stoppes/avvises": 7,
    "00e Sak trukket/utgått/uaktuell":                      7,
}

OUTPUT_TABLE = "planning_case_timeline"

_milepeler_in = ", ".join(f"'{m}'" for m in MILEPELER_MILESTONES)
_am_in        = ", ".join(str(c) for c in AM_PRODUKTNR)
_kb_in        = ", ".join(str(c) for c in KB_PRODUKTNR)


# =============================================================================
# CELL 1 — Create output table
# =============================================================================

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
    prosess_id      STRING    NOT NULL,
    milestone       STRING    NOT NULL,
    end_date        DATE,
    prev_milestone  STRING,
    days_since_prev INT,
    source          STRING    NOT NULL,
    computed_at     TIMESTAMP NOT NULL,
    batch_id        STRING    NOT NULL
)
USING DELTA
COMMENT 'Chronological milestone timeline for planning cases. Combines Milepeler system milestones with Avklaringsmøte and Kart- og bestemmelsesmøte invoiced events from Fakturalinjer. prev_milestone and days_since_prev are pre-computed per case for time-between-events analysis in Power BI. Join prosess_id -> Prosesser for case metadata.'
""")

print(f"Output table {OUTPUT_TABLE} ready")


# =============================================================================
# CELL 2 — Load system milestones from Milepeler
# =============================================================================

milepeler_events = spark.sql(f"""
    SELECT
        CAST(Prosess_id AS STRING) AS prosess_id,
        Milepel                    AS milestone,
        CAST(Sluttdato  AS DATE)   AS end_date,
        'Milepeler'                AS source
    FROM Milepeler
    WHERE Milepel    IN ({_milepeler_in})
      AND Sluttdato  IS NOT NULL
      AND Prosess_id IS NOT NULL
""")

print(f"Milepeler events: {milepeler_events.count():,}")


# =============================================================================
# CELL 3 — Load Avklaringsmøte events from Fakturalinjer
# =============================================================================
# Multiple invoices per Prosess_id are intentional — one per meeting held.

am_events = spark.sql(f"""
    SELECT
        CAST(Prosess_id AS STRING)    AS prosess_id,
        'Avklaringsmøte'              AS milestone,
        CAST(Fakturert_dato AS DATE)  AS end_date,
        'Fakturalinjer'               AS source
    FROM Fakturalinjer
    WHERE Produktnr      IN ({_am_in})
      AND Prosess_id     IS NOT NULL
      AND Fakturert_dato IS NOT NULL
""")

print(f"Avklaringsmøte events: {am_events.count():,}")


# =============================================================================
# CELL 4 — Load Kart- og bestemmelsesmøte events from Fakturalinjer
# =============================================================================

kb_events = spark.sql(f"""
    SELECT
        CAST(Prosess_id AS STRING)    AS prosess_id,
        'Kart- og bestemmelsesmøte'   AS milestone,
        CAST(Fakturert_dato AS DATE)  AS end_date,
        'Fakturalinjer'               AS source
    FROM Fakturalinjer
    WHERE Produktnr      IN ({_kb_in})
      AND Prosess_id     IS NOT NULL
      AND Fakturert_dato IS NOT NULL
""")

print(f"Kart- og bestemmelsesmøte events: {kb_events.count():,}")


# =============================================================================
# CELL 5 — Union, compute prev_milestone and days_since_prev, add metadata
# =============================================================================
# Sort by (end_date, milestone_sort) so the lag is deterministic when two
# events share the same date. milestone_sort is dropped before writing.

sort_map = F.create_map(*[item for k, v in MILESTONE_SORT.items()
                          for item in (F.lit(k), F.lit(v))])

w_lag = Window.partitionBy("prosess_id").orderBy("end_date", "milestone_sort")

all_events = (
    milepeler_events
    .union(am_events)
    .union(kb_events)
    .withColumn("milestone_sort",  F.coalesce(sort_map[F.col("milestone")], F.lit(99)))
    .withColumn("prev_milestone",  F.lag("milestone").over(w_lag))
    .withColumn("prev_end_date",   F.lag("end_date").over(w_lag))
    .withColumn("days_since_prev", F.datediff(F.col("end_date"), F.col("prev_end_date")).cast("int"))
    .drop("prev_end_date", "milestone_sort")
    .withColumn("computed_at", F.lit(datetime.now()).cast("timestamp"))
    .withColumn("batch_id",    F.lit(BATCH_ID))
    .orderBy("prosess_id", "end_date")
)

print(f"Total timeline events: {all_events.count():,}")
all_events.show(20, truncate=False)


# =============================================================================
# CELL 6 — Write (full overwrite)
# =============================================================================

all_events.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

written = spark.sql(f"""
    SELECT
        milestone,
        COUNT(*) AS event_count
    FROM {OUTPUT_TABLE}
    WHERE batch_id = '{BATCH_ID}'
    GROUP BY milestone
    ORDER BY event_count DESC
""")
print("Events by milestone:")
written.show(truncate=False)

print(f"Total rows in {OUTPUT_TABLE}: "
      f"{spark.table(OUTPUT_TABLE).count():,}")


# =============================================================================
# CELL 7 — Power BI visual guidance
# =============================================================================
#
# OUTPUT TABLE → planning_case_timeline
# One row per milestone event per Prosess_id. Full overwrite on each run.
# Multiple Avklaringsmøte and KB rows per case are expected and meaningful.
#
# RECOMMENDED JOIN IN SEMANTIC MODEL
#   planning_case_timeline.prosess_id → Prosesser.Prosess_id
#   Gives: Indikator, Saksnummer, Startdato, Sluttdato.
#
# HOW TO USE days_since_prev IN POWER BI
#
# All time-between-events analysis works by filtering milestone and
# prev_milestone in a visual or slicer, then averaging days_since_prev.
# No complex DAX needed — just plain column aggregation.
#
# Key filter combinations and what they measure:
#
#   OM → first Avklaringsmøte
#     milestone       = "Avklaringsmøte"
#     prev_milestone  IN ("Innkalling til oppstartsmøte",
#                         "P02c Område- og prosessavklaring til fagkyndig",
#                         "P02 Område/prosessavklaring")
#
#   Between consecutive Avklaringsmøter (AM1→AM2, AM2→AM3, ...)
#     milestone       = "Avklaringsmøte"
#     prev_milestone  = "Avklaringsmøte"
#
#   Last Avklaringsmøte → Planforslag mottatt
#     milestone       = "Planforslag mottatt"
#     prev_milestone  = "Avklaringsmøte"
#
#   Last Avklaringsmøte → Kart- og bestemmelsesmøte
#     milestone       = "Kart- og bestemmelsesmøte"
#     prev_milestone  = "Avklaringsmøte"
#
#   Kart- og bestemmelsesmøte → Planforslag mottatt
#     milestone       = "Planforslag mottatt"
#     prev_milestone  = "Kart- og bestemmelsesmøte"
#
#   Planforslag mottatt → Offentlig ettersyn
#     milestone       = "Offentlig ettersyn"
#     prev_milestone  = "Planforslag mottatt"
#
# In Power BI: add both columns as filters/slicers on a visual, set the
# value field to days_since_prev with Average aggregation. Slice by year
# (from end_date), Indikator (via Prosesser join), or case complexity
# (from AM product code via Fakturalinjer join:
#  62432 = Simple, 62433 = Medium, 62434 = Complex).
#
# COUNT OF MEETINGS PER CASE
#   Filter milestone = "Avklaringsmøte" (or "Kart- og bestemmelsesmøte"),
#   then Count rows — gives meeting count per case or average per indicator.
#
# TIMELINE / GANTT VISUALIZATION
#   X axis: end_date
#   Y axis: prosess_id or Saksnummer (via Prosesser join)
#   Color:  milestone
#   Filter to a single Indikator or year to keep the visual readable.
#
# NOTE: Milestone names in MILEPELER_MILESTONES are system strings from the
# case management system. If they change, update the list at the top of this
# script. AM_PRODUKTNR and KB_PRODUKTNR are from PlanningConfig in
# PBE-IncomeForecast — sync if the fee schedule changes.
