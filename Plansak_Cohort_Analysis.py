# =============================================================================
# Plansak_Cohort_Analysis.py
#
# Tracks the 2026 planning-case cohort (Oppstartsmøte >= 01.01.2026) against
# a rolling per-case goal: every case must reach "Sendt til politisk behandling"
# within 3 years of its own Oppstartsmøte date (Case_Deadline = OM + 3 years).
# Cases entering the cohort later in time get correspondingly later deadlines.
#
# Four phases are measured per case:
#   1. Utarbeidingsfasen Kostra  — OM to Planforslag mottatt (Milepeler)
#   2. Offentlig ettersyn        — one of 3 Prosesser variants
#   3. Høringsperiode            — Prosesser
#   4. Sendt til politisk behandling — Prosesser
#
# Output tables (Delta, Power BI-ready):
#   plansak_2026_phase_detail  — one row per (Saksnummer × Phase)
#   plansak_2026_dim_cases     — one row per Saksnummer
#   plansak_2026_kpi_snapshot  — aggregate KPIs, append-mode for trend tracking
#
# Power BI data model:
#   plansak_2026_dim_cases →[Saksnummer]→ plansak_2026_phase_detail
#   plansak_2026_kpi_snapshot: standalone
#
# Schedule: nightly after main data pipeline.
# =============================================================================

from pyspark.sql import SparkSession
import pandas as pd
from datetime import datetime, date

spark    = SparkSession.builder.getOrCreate()
BATCH_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
TODAY    = pd.Timestamp(date.today())


# =============================================================================
# CONFIG
# =============================================================================
# expected_phase_days drives two things:
#   1. Projected completion date per case
#   2. Utarbeidingsfasen budget = 3-year window minus the sum of phases 2–4
#
# Phases 2–4 use realistic durations (not just legal minimums) because cases
# regularly exceed the legal deadlines. Update these if the portfolio changes.
# =============================================================================

CONFIG = {
    "cohort_start":          "2026-01-01",
    # Each case's deadline = its own OM date + rolling_deadline_years
    "rolling_deadline_years": 3,

    # Cases projected to miss their deadline by fewer than this many days → "At Risk"
    "at_risk_buffer_days": 180,

    # KS milestone reached but Planforslag mottatt missing for this long → alert
    "ks_to_pf_max_days": 60,

    # Calendar-time stalled-phase thresholds (forward-looking intervention).
    # Fires when a phase is "In Progress" longer than this — independent of Tidsbruk.
    "stalled_phase_days": {
        "Offentlig ettersyn":             180,
        "Høringsperiode":                 120,
        "Sendt til politisk behandling":  365,
    },

    # Fallback Frist (statutory) when Prosesser.Frist is NULL — same units (days)
    "frist_default_days": {
        "Offentlig ettersyn":              84,   # 12 weeks
        "Sendt til politisk behandling":  126,   # 18 weeks
    },

    # Høringsperiode statutory minimum (6 weeks)
    "høringsperiode_min_days": 42,

    # Expected realistic duration per phase — used for projected completion
    # and to derive the Utarbeidingsfasen budget.
    # Phases 2–4 realistic (exceeds legal minimums where practice shows overrun):
    "expected_phase_days": {
        "Offentlig ettersyn":             84,   # 12 weeks (legal = 12 wk, met in practice)
        "Høringsperiode":                 63,   # ~9 weeks (above statutory minimum)
        "Sendt til politisk behandling": 245,   # 35 weeks (18-wk legal deadline rarely met)
    },

    # Historical average days OM→OE by complexity (from PlanningConfig calibration).
    # Used as context alongside the budget-derived threshold — not as the alert trigger.
    "utarbeidingsfasen_historical_avg": {
        "Simple":  620,
        "Medium":  732,
        "Complex": 1068,
        "Unknown": 732,
    },

    # Expected AM count per case by complexity (AM_PROB_ANY × AM_AVG_GIVEN_ANY
    # from PlansakCalibrationParams — update if recalibrated)
    "am_expected_avg": {
        "Simple":  1.2,
        "Medium":  2.1,
        "Complex": 3.5,
        "Unknown": 2.1,
    },

    # Offentlig ettersyn Prosesser variants (mutually exclusive per case)
    "offentlig_ettersyn_variants": [
        "Offentlig ettersyn privat plan",
        "Offentlig ettersyn privat plan med avtalt frist",
        "Offentlig ettersyn offentlig plan",
    ],

    # Fakturalinjer Produktnr for complexity (Long type, compare as integer)
    "om_complexity_codes": {
        62422: "Simple",
        62423: "Medium",
        62424: "Complex",
    },
    "am_codes": [62432, 62433, 62434],
    "kb_codes": [62441, 62442, 62443, 62444],

    "table_milepeler":     "Saksbehandling.milepeler",
    "table_prosesser":     "Saksbehandling.prosesser",
    "table_fakturalinjer": "Saksbehandling.fakturalinjer",

    "out_phase_detail": "plansak_2026_phase_detail",
    "out_dim_cases":    "plansak_2026_dim_cases",
    "out_kpi_snapshot": "plansak_2026_kpi_snapshot",
}

# Derived: budget days for Utarbeidingsfasen = 3-year window − expected phases 2–4
# 3 years ≈ 1096 days (includes one leap year on average)
_TOTAL_DAYS    = CONFIG["rolling_deadline_years"] * 365 + 1
_PHASES_2_4    = sum(CONFIG["expected_phase_days"].values())
UTARB_BUDGET   = _TOTAL_DAYS - _PHASES_2_4

MILEPELER     = CONFIG["table_milepeler"]
PROSESSER     = CONFIG["table_prosesser"]
FAKTURALINJER = CONFIG["table_fakturalinjer"]
OUT_PHASE     = CONFIG["out_phase_detail"]
OUT_CASES     = CONFIG["out_dim_cases"]
OUT_KPI       = CONFIG["out_kpi_snapshot"]

PHASE_ORDER = [
    "Utarbeidingsfasen Kostra",
    "Offentlig ettersyn",
    "Høringsperiode",
    "Sendt til politisk behandling",
]

print(f"3-year window:            {_TOTAL_DAYS} days")
print(f"Expected phases 2–4:      {_PHASES_2_4} days")
print(f"Utarbeidingsfasen budget: {UTARB_BUDGET} days (~{UTARB_BUDGET // 7} weeks)")


# =============================================================================
# CELL 1 — Create output tables
# =============================================================================

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUT_PHASE} (
    Saksnummer              STRING      NOT NULL,
    Phase_Name              STRING      NOT NULL,
    Phase_Variant           STRING,
    Phase_Startdato         DATE,
    Phase_Sluttdato         DATE,
    Frist                   DOUBLE,
    Tidsbruk                DOUBLE,
    Bransjetid              DOUBLE,
    Calendar_Days           INT,
    Phase_Status            STRING      NOT NULL,
    Alert_Flag              BOOLEAN     NOT NULL,
    Alert_Reason            STRING,
    AM_Count                INT,
    AM_Expected_Avg         DOUBLE,
    Last_AM_Date            DATE,
    Days_Since_Last_AM      INT,
    KB_Count                INT,
    Last_KB_Date            DATE,
    KS_Date                 DATE,
    Utarb_Budget_Days       INT,
    Utarb_Historical_Avg    INT,
    Structural_Risk         BOOLEAN,
    computed_at             TIMESTAMP   NOT NULL,
    batch_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'One row per Saksnummer x Phase for the 2026 Plansak cohort. Alert_Flag and Alert_Reason explain off-track cases. Join to plansak_2026_dim_cases on Saksnummer.'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUT_CASES} (
    Saksnummer                  STRING      NOT NULL,
    Case_Start_Date             DATE,
    Case_End_Date               DATE,
    Case_Deadline               DATE,
    Case_Complexity             STRING,
    Current_Phase               STRING,
    Total_Elapsed_Days          INT,
    Projected_Completion_Date   DATE,
    Days_To_Deadline            INT,
    Days_Since_Last_AM_Utarb    INT,
    Overall_Status              STRING      NOT NULL,
    Alert_Count                 INT         NOT NULL,
    computed_at                 TIMESTAMP   NOT NULL,
    batch_id                    STRING      NOT NULL
)
USING DELTA
COMMENT 'One row per Saksnummer in the 2026 cohort. Case_Deadline = OM date + 3 years (rolling per case). Days_To_Deadline negative means projected to miss the individual case deadline. Days_Since_Last_AM_Utarb rolled up from phase 1 for slicing.'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUT_KPI} (
    Snapshot_Date           DATE        NOT NULL,
    Total_Cases             INT         NOT NULL,
    Completed               INT         NOT NULL,
    On_Track                INT         NOT NULL,
    At_Risk                 INT         NOT NULL,
    Off_Track               INT         NOT NULL,
    Phase1_In_Progress      INT         NOT NULL,
    Phase1_Alert            INT         NOT NULL,
    Phase2_In_Progress      INT         NOT NULL,
    Phase2_Alert            INT         NOT NULL,
    Phase3_In_Progress      INT         NOT NULL,
    Phase3_Alert            INT         NOT NULL,
    Phase4_In_Progress      INT         NOT NULL,
    Phase4_Alert            INT         NOT NULL,
    computed_at             TIMESTAMP   NOT NULL,
    batch_id                STRING      NOT NULL
)
USING DELTA
COMMENT 'Aggregate KPI snapshot, append-mode. One row per nightly run. Use Snapshot_Date for trend charts in Power BI.'
""")

print("Output tables ready")


# =============================================================================
# CELL 2 — Cohort definition from Milepeler
# =============================================================================

cohort_start = CONFIG["cohort_start"]

cohort_spark = spark.sql(f"""
    SELECT DISTINCT Saksnummer
    FROM {MILEPELER}
    WHERE Milepel    = 'Oppstartsmøte'
      AND Sluttdato >= '{cohort_start}'
      AND Saksnummer IS NOT NULL
      AND Sluttdato  IS NOT NULL
""")
cohort_saksnummer = [row.Saksnummer for row in cohort_spark.collect()]
print(f"Cohort cases: {len(cohort_saksnummer):,}  (Oppstartsmøte >= {cohort_start})")

if not cohort_saksnummer:
    raise ValueError(f"No cases found with Oppstartsmøte >= {cohort_start}. Check Milepeler table.")

cohort_spark.createOrReplaceTempView("_cohort_cases")

# Load OM and Planforslag mottatt dates for cohort cases
milepeler_cohort = spark.sql(f"""
    SELECT m.Saksnummer,
           m.Milepel,
           CAST(m.Sluttdato AS DATE) AS Sluttdato
    FROM {MILEPELER} m
    JOIN _cohort_cases c ON m.Saksnummer = c.Saksnummer
    WHERE m.Milepel IN ('Oppstartsmøte', 'Planforslag mottatt')
      AND m.Sluttdato IS NOT NULL
""").toPandas()

milepeler_cohort["Sluttdato"] = pd.to_datetime(milepeler_cohort["Sluttdato"])

om_date = (
    milepeler_cohort[milepeler_cohort["Milepel"] == "Oppstartsmøte"]
    .set_index("Saksnummer")["Sluttdato"]
    .to_dict()
)
pf_date = (
    milepeler_cohort[milepeler_cohort["Milepel"] == "Planforslag mottatt"]
    .set_index("Saksnummer")["Sluttdato"]
    .to_dict()
)

# KS milestone: documentation completeness check — marks when the clock starts
# ticking on formal processing. Type_kode = 'KS' in Milepeler.
ks_cohort = spark.sql(f"""
    SELECT m.Saksnummer,
           MAX(CAST(m.Sluttdato AS DATE)) AS KS_Date
    FROM {MILEPELER} m
    JOIN _cohort_cases c ON m.Saksnummer = c.Saksnummer
    WHERE m.Type_kode = 'KS'
      AND m.Sluttdato IS NOT NULL
    GROUP BY m.Saksnummer
""").toPandas()

ks_cohort["KS_Date"] = pd.to_datetime(ks_cohort["KS_Date"])
ks_date = ks_cohort.set_index("Saksnummer")["KS_Date"].to_dict()

print(f"OM dates found:                {len(om_date):,}")
print(f"Planforslag mottatt dates:     {len(pf_date):,}")
print(f"KS dates found:                {len(ks_date):,}")


# =============================================================================
# CELL 3 — Complexity and Avklaringsmøter from Fakturalinjer
# =============================================================================
# Join Fakturalinjer → Prosesser to get Saksnummer.
# OM product codes → case complexity. AM product codes → meeting activity.
# Produktnr is Long type in Fakturalinjer — compare as integer.

_om_codes_sql = ",".join(str(c) for c in CONFIG["om_complexity_codes"])
_am_codes_sql = ",".join(str(c) for c in CONFIG["am_codes"])
_kb_codes_sql = ",".join(str(c) for c in CONFIG["kb_codes"])

invoices = spark.sql(f"""
    SELECT
        p.Saksnummer,
        CAST(f.Produktnr      AS BIGINT) AS Produktnr,
        CAST(f.Fakturert_dato AS DATE)   AS Fakturert_dato
    FROM {FAKTURALINJER} f
    JOIN {PROSESSER}     p  ON CAST(f.Prosess_id AS STRING) = CAST(p.Prosess_id AS STRING)
    JOIN _cohort_cases   c  ON p.Saksnummer = c.Saksnummer
    WHERE f.Produktnr IN ({_om_codes_sql},{_am_codes_sql},{_kb_codes_sql})
      AND f.Fakturert_dato IS NOT NULL
""").toPandas()

invoices["Fakturert_dato"] = pd.to_datetime(invoices["Fakturert_dato"])

om_invoices = invoices[invoices["Produktnr"].isin(CONFIG["om_complexity_codes"])]
am_invoices = invoices[invoices["Produktnr"].isin(CONFIG["am_codes"])]
kb_invoices = invoices[invoices["Produktnr"].isin(CONFIG["kb_codes"])]

complexity_map = {
    sak: CONFIG["om_complexity_codes"].get(int(grp["Produktnr"].iloc[0]), "Unknown")
    for sak, grp in om_invoices.groupby("Saksnummer")
}
am_count_map     = am_invoices.groupby("Saksnummer").size().to_dict()
am_last_date_map = am_invoices.groupby("Saksnummer")["Fakturert_dato"].max().to_dict()
kb_count_map     = kb_invoices.groupby("Saksnummer").size().to_dict()
kb_last_date_map = kb_invoices.groupby("Saksnummer")["Fakturert_dato"].max().to_dict()

print(f"Invoice rows loaded:         {len(invoices):,}")
print(f"Cases with complexity code:  {len(complexity_map):,}")
print(f"Cases with ≥1 AM:            {len(am_count_map):,}")
print(f"Cases with ≥1 KB:            {len(kb_count_map):,}")
if complexity_map:
    print(f"Complexity distribution:     {pd.Series(complexity_map).value_counts().to_dict()}")


# =============================================================================
# CELL 4 — Phases 2–4 from Prosesser
# =============================================================================

_all_indicators = CONFIG["offentlig_ettersyn_variants"] + [
    "Høringsperiode",
    "Sendt til politisk behandling",
]
_ind_sql = ",".join(f"'{i}'" for i in _all_indicators)

prosesser_raw = spark.sql(f"""
    SELECT
        p.Saksnummer,
        p.Indikator,
        CAST(p.Startdato  AS DATE)   AS Startdato,
        CAST(p.Sluttdato  AS DATE)   AS Sluttdato,
        CAST(p.Tidsbruk   AS DOUBLE) AS Tidsbruk,
        CAST(p.Bransjetid AS DOUBLE) AS Bransjetid,
        CAST(p.Frist      AS DOUBLE) AS Frist
    FROM {PROSESSER}   p
    JOIN _cohort_cases c ON p.Saksnummer = c.Saksnummer
    WHERE p.Indikator IN ({_ind_sql})
""").toPandas()

prosesser_raw["Startdato"] = pd.to_datetime(prosesser_raw["Startdato"])
prosesser_raw["Sluttdato"] = pd.to_datetime(prosesser_raw["Sluttdato"])

oe_set = set(CONFIG["offentlig_ettersyn_variants"])
prosesser_raw["Phase_Name"] = prosesser_raw["Indikator"].apply(
    lambda i: "Offentlig ettersyn" if i in oe_set else i
)
prosesser_raw["Phase_Variant"] = prosesser_raw.apply(
    lambda r: r["Indikator"] if r["Phase_Name"] == "Offentlig ettersyn" else None,
    axis=1,
)

# Per (Saksnummer, canonical Phase_Name): keep row with latest Startdato
prosesser_dedup = (
    prosesser_raw
    .sort_values("Startdato", ascending=False, na_position="last")
    .drop_duplicates(subset=["Saksnummer", "Phase_Name"])
    .set_index(["Saksnummer", "Phase_Name"])
)

print(f"Prosesser rows loaded: {len(prosesser_raw):,}")
print(prosesser_raw["Indikator"].value_counts().to_string())


# =============================================================================
# CELL 5 — Build phase_detail and dim_cases
# =============================================================================

HOER_MIN    = CONFIG["høringsperiode_min_days"]
AT_RISK_BUF = CONFIG["at_risk_buffer_days"]
DEADLINE_YRS = CONFIG["rolling_deadline_years"]

phase_rows = []
dim_rows   = []

for saksnummer in sorted(cohort_saksnummer):
    complexity  = complexity_map.get(saksnummer, "Unknown")
    am_count    = am_count_map.get(saksnummer, 0)
    am_last     = am_last_date_map.get(saksnummer, pd.NaT)
    kb_count    = kb_count_map.get(saksnummer, 0)
    kb_last     = kb_last_date_map.get(saksnummer, pd.NaT)
    ks_val      = ks_date.get(saksnummer, pd.NaT)
    am_exp_avg  = CONFIG["am_expected_avg"].get(complexity, CONFIG["am_expected_avg"]["Unknown"])
    hist_avg    = CONFIG["utarbeidingsfasen_historical_avg"].get(
                      complexity, CONFIG["utarbeidingsfasen_historical_avg"]["Unknown"])
    structural_risk = hist_avg > UTARB_BUDGET

    # ── Phase 1: Utarbeidingsfasen Kostra ──────────────────────────────────────
    p1_start = om_date.get(saksnummer, pd.NaT)
    p1_end   = pf_date.get(saksnummer, pd.NaT)

    if pd.notna(p1_start) and pd.notna(p1_end):
        p1_cal = (p1_end - p1_start).days
        p1_status = "Completed"
    elif pd.notna(p1_start):
        p1_cal = (TODAY - p1_start).days
        p1_status = "In Progress"
    else:
        p1_cal = None
        p1_status = "Not Started"

    # Days since last AM — informational only (AM is customer-driven, not controlled by us)
    # For completed phase 1, freeze at phase end so the value doesn't grow forever.
    if pd.notna(am_last):
        if p1_status == "Completed" and pd.notna(p1_end):
            days_since_last_am = (p1_end - am_last).days
        else:
            days_since_last_am = (TODAY - am_last).days
    else:
        days_since_last_am = None

    p1_alerts = []
    if structural_risk:
        p1_alerts.append(
            f"Strukturell risiko: historisk snitt for {complexity}-saker "
            f"({hist_avg} d) overskrider budsjett ({UTARB_BUDGET} d)"
        )
    if p1_cal is not None and p1_cal > UTARB_BUDGET:
        p1_alerts.append(
            f"Utarbeidingsfasen overskrider budsjett ({p1_cal} av {UTARB_BUDGET} dager)"
        )
    # KS reached but Planforslag not yet submitted → stuck on customer side
    if pd.notna(ks_val) and pd.isna(p1_end):
        days_since_ks = (TODAY - ks_val).days
        if days_since_ks > CONFIG["ks_to_pf_max_days"]:
            p1_alerts.append(
                f"Dokumentasjon klar (KS) men Planforslag ikke mottatt på {days_since_ks} dager"
            )

    phase_rows.append({
        "Saksnummer":           saksnummer,
        "Phase_Name":           "Utarbeidingsfasen Kostra",
        "Phase_Variant":        None,
        "Phase_Startdato":      p1_start if pd.notna(p1_start) else None,
        "Phase_Sluttdato":      p1_end   if pd.notna(p1_end)   else None,
        "Frist":                None,
        "Tidsbruk":             None,
        "Bransjetid":           None,
        "Calendar_Days":        p1_cal,
        "Phase_Status":         p1_status,
        "Alert_Flag":           len(p1_alerts) > 0,
        "Alert_Reason":         "; ".join(p1_alerts) if p1_alerts else None,
        "AM_Count":             am_count,
        "AM_Expected_Avg":      am_exp_avg,
        "Last_AM_Date":         am_last if pd.notna(am_last) else None,
        "Days_Since_Last_AM":   days_since_last_am,
        "KB_Count":             kb_count,
        "Last_KB_Date":         kb_last if pd.notna(kb_last) else None,
        "KS_Date":              ks_val  if pd.notna(ks_val)  else None,
        "Utarb_Budget_Days":    UTARB_BUDGET,
        "Utarb_Historical_Avg": hist_avg,
        "Structural_Risk":      structural_risk,
    })

    # ── Phases 2–4: from Prosesser ─────────────────────────────────────────────
    for phase_name in ("Offentlig ettersyn", "Høringsperiode", "Sendt til politisk behandling"):
        key = (saksnummer, phase_name)
        if key not in prosesser_dedup.index:
            phase_rows.append({
                "Saksnummer": saksnummer,
                "Phase_Name": phase_name,
                "Phase_Variant": None,
                "Phase_Startdato": None, "Phase_Sluttdato": None,
                "Frist": None, "Tidsbruk": None, "Bransjetid": None,
                "Calendar_Days": None, "Phase_Status": "Not Started",
                "Alert_Flag": False, "Alert_Reason": None,
                "AM_Count": None, "AM_Expected_Avg": None,
                "Last_AM_Date": None, "Days_Since_Last_AM": None,
                "KB_Count": None, "Last_KB_Date": None, "KS_Date": None,
                "Utarb_Budget_Days": None, "Utarb_Historical_Avg": None,
                "Structural_Risk": None,
            })
            continue

        r         = prosesser_dedup.loc[key]
        p_start   = r["Startdato"]
        p_end     = r["Sluttdato"]
        tidsbruk  = r["Tidsbruk"]
        bransjetid= r["Bransjetid"]
        frist     = r["Frist"]

        if pd.notna(p_start) and pd.notna(p_end):
            cal_days = (p_end - p_start).days
            status   = "Completed"
        elif pd.notna(p_start):
            cal_days = (TODAY - p_start).days
            status   = "In Progress"
        else:
            cal_days = None
            status   = "Not Started"

        alerts = []
        if phase_name in ("Offentlig ettersyn", "Sendt til politisk behandling"):
            # Frist null fallback to statutory default so alerts still fire
            effective_frist = frist if pd.notna(frist) else CONFIG["frist_default_days"].get(phase_name)
            if pd.notna(tidsbruk) and effective_frist is not None and tidsbruk > effective_frist:
                overshoot = int(tidsbruk - effective_frist)
                via = "" if pd.notna(frist) else " (statutory default)"
                alerts.append(f"{phase_name} overskrider Frist med {overshoot} dager{via}")
        elif phase_name == "Høringsperiode":
            if status == "Completed" and cal_days is not None and cal_days < HOER_MIN:
                alerts.append(
                    f"Høringsperiode under lovpålagt minimum ({cal_days} av {HOER_MIN} dager)"
                )

        # Forward-looking: stalled phase open longer than calendar threshold
        stalled_max = CONFIG["stalled_phase_days"].get(phase_name)
        if status == "In Progress" and cal_days is not None and stalled_max is not None \
                and cal_days > stalled_max:
            alerts.append(f"{phase_name} åpen i {cal_days} dager uten avslutning (terskel {stalled_max})")

        variant = r.get("Phase_Variant")

        phase_rows.append({
            "Saksnummer":           saksnummer,
            "Phase_Name":           phase_name,
            "Phase_Variant":        variant if pd.notna(variant) else None,
            "Phase_Startdato":      p_start    if pd.notna(p_start)    else None,
            "Phase_Sluttdato":      p_end      if pd.notna(p_end)      else None,
            "Frist":                frist      if pd.notna(frist)      else None,
            "Tidsbruk":             tidsbruk   if pd.notna(tidsbruk)   else None,
            "Bransjetid":           bransjetid if pd.notna(bransjetid) else None,
            "Calendar_Days":        cal_days,
            "Phase_Status":         status,
            "Alert_Flag":           len(alerts) > 0,
            "Alert_Reason":         "; ".join(alerts) if alerts else None,
            "AM_Count":             None,
            "AM_Expected_Avg":      None,
            "Last_AM_Date":         None,
            "Days_Since_Last_AM":   None,
            "KB_Count":             None,
            "Last_KB_Date":         None,
            "KS_Date":              None,
            "Utarb_Budget_Days":    None,
            "Utarb_Historical_Avg": None,
            "Structural_Risk":      None,
        })

# ── Assemble phase_detail ──────────────────────────────────────────────────────
phase_detail = pd.DataFrame(phase_rows)
phase_detail["computed_at"] = datetime.now()
phase_detail["batch_id"]    = BATCH_ID

for col in ("Phase_Startdato", "Phase_Sluttdato", "Last_AM_Date", "Last_KB_Date", "KS_Date"):
    phase_detail[col] = pd.to_datetime(phase_detail[col])

# Index for fast lookups in dim_cases loop
_pd_idx = phase_detail.set_index(["Saksnummer", "Phase_Name"])

print(f"\nPhase detail rows: {len(phase_detail):,}")
print(phase_detail.groupby("Phase_Name")["Phase_Status"].value_counts().to_string())

# ── Build dim_cases ────────────────────────────────────────────────────────────
# Precompute per-case rollups so the inner loop is O(1) per case.
alert_counts_by_case = phase_detail.groupby("Saksnummer")["Alert_Flag"].sum().to_dict()
utarb_rows = phase_detail[phase_detail["Phase_Name"] == "Utarbeidingsfasen Kostra"]
days_since_am_by_case = (
    utarb_rows.set_index("Saksnummer")["Days_Since_Last_AM"].to_dict()
)

for saksnummer in sorted(cohort_saksnummer):
    case_start = om_date.get(saksnummer, pd.NaT)
    complexity = complexity_map.get(saksnummer, "Unknown")

    # Case end = Sluttdato of "Sendt til politisk behandling"
    p4_key = (saksnummer, "Sendt til politisk behandling")
    if p4_key in _pd_idx.index:
        case_end = _pd_idx.loc[p4_key, "Phase_Sluttdato"]
        case_end = case_end if pd.notna(case_end) else pd.NaT
    else:
        case_end = pd.NaT

    # Current phase: first In Progress or Not Started; otherwise last (Completed).
    # phase_detail always has a row per (Saksnummer × Phase), so no missing-key branch needed.
    current_phase = PHASE_ORDER[0]
    for ph in PHASE_ORDER:
        st = _pd_idx.loc[(saksnummer, ph), "Phase_Status"]
        current_phase = ph
        if st in ("In Progress", "Not Started"):
            break

    # Total elapsed days
    if pd.notna(case_start) and pd.notna(case_end):
        total_elapsed = (case_end - case_start).days
    elif pd.notna(case_start):
        total_elapsed = (TODAY - case_start).days
    else:
        total_elapsed = None

    # Projected completion: sum actual days for completed phases +
    # elapsed-so-far (or expected, whichever is larger) for in-progress +
    # expected for future phases
    projected = None
    if pd.notna(case_start):
        projected_days = 0
        for ph in PHASE_ORDER:
            expected = (
                UTARB_BUDGET if ph == "Utarbeidingsfasen Kostra"
                else CONFIG["expected_phase_days"].get(ph, 90)
            )
            key = (saksnummer, ph)
            if key not in _pd_idx.index:
                projected_days += expected
                continue
            ph_start = _pd_idx.loc[key, "Phase_Startdato"]
            ph_end   = _pd_idx.loc[key, "Phase_Sluttdato"]
            st       = _pd_idx.loc[key, "Phase_Status"]
            if st == "Completed" and pd.notna(ph_start) and pd.notna(ph_end):
                projected_days += (ph_end - ph_start).days
            elif st == "In Progress" and pd.notna(ph_start):
                elapsed_ph = (TODAY - ph_start).days
                projected_days += max(elapsed_ph, expected)
            else:
                projected_days += expected
        projected = case_start + pd.Timedelta(days=projected_days)

    # Per-case rolling deadline: OM date + 3 years
    case_deadline = (
        case_start + pd.DateOffset(years=DEADLINE_YRS)
        if pd.notna(case_start) else pd.NaT
    )

    # Days to deadline: prefer actual case_end for completed cases (accurate),
    # fall back to projected for open cases (estimate).
    if pd.notna(case_end) and pd.notna(case_deadline):
        days_to_deadline = (case_deadline - case_end).days
    elif projected is not None and pd.notna(case_deadline):
        days_to_deadline = (case_deadline - projected).days
    else:
        days_to_deadline = None

    # Alert count — O(1) lookup from precomputed dict
    alert_count = int(alert_counts_by_case.get(saksnummer, 0))

    # Overall status
    at_risk_threshold = (
        case_deadline - pd.Timedelta(days=AT_RISK_BUF)
        if pd.notna(case_deadline) else pd.NaT
    )
    if pd.notna(case_end):
        overall_status = "Completed"
        current_phase = "Completed"   # override so Power BI doesn't show "Sendt til politisk"
    elif alert_count > 0:
        overall_status = "Off Track"
    elif projected is not None and pd.notna(at_risk_threshold) and projected > at_risk_threshold:
        overall_status = "At Risk"
    else:
        overall_status = "On Track"

    # Days since last AM, rolled up from phase 1 for slicing
    days_since_am_utarb = days_since_am_by_case.get(saksnummer)
    if days_since_am_utarb is not None and pd.isna(days_since_am_utarb):
        days_since_am_utarb = None

    dim_rows.append({
        "Saksnummer":               saksnummer,
        "Case_Start_Date":          case_start    if pd.notna(case_start)    else None,
        "Case_End_Date":            case_end      if pd.notna(case_end)      else None,
        "Case_Deadline":            case_deadline if pd.notna(case_deadline) else None,
        "Case_Complexity":          complexity,
        "Current_Phase":            current_phase,
        "Total_Elapsed_Days":       total_elapsed,
        "Projected_Completion_Date": projected if projected is not None else None,
        "Days_To_Deadline":         days_to_deadline,
        "Days_Since_Last_AM_Utarb": days_since_am_utarb,
        "Overall_Status":           overall_status,
        "Alert_Count":              alert_count,
    })

dim_cases = pd.DataFrame(dim_rows)
dim_cases["computed_at"] = datetime.now()
dim_cases["batch_id"]    = BATCH_ID

for col in ("Case_Start_Date", "Case_End_Date", "Case_Deadline", "Projected_Completion_Date"):
    dim_cases[col] = pd.to_datetime(dim_cases[col])

print(f"\ndim_cases rows: {len(dim_cases):,}")
print(dim_cases["Overall_Status"].value_counts().to_string())
print(dim_cases["Case_Complexity"].value_counts().to_string())


# =============================================================================
# CELL 6 — Build kpi_snapshot
# =============================================================================

def _phase_counts(name):
    ph = phase_detail[phase_detail["Phase_Name"] == name]
    return int((ph["Phase_Status"] == "In Progress").sum()), int(ph["Alert_Flag"].sum())

p1_ip, p1_al = _phase_counts("Utarbeidingsfasen Kostra")
p2_ip, p2_al = _phase_counts("Offentlig ettersyn")
p3_ip, p3_al = _phase_counts("Høringsperiode")
p4_ip, p4_al = _phase_counts("Sendt til politisk behandling")

kpi_row = {
    "Snapshot_Date":      date.today(),
    "Total_Cases":        len(dim_cases),
    "Completed":          int((dim_cases["Overall_Status"] == "Completed").sum()),
    "On_Track":           int((dim_cases["Overall_Status"] == "On Track").sum()),
    "At_Risk":            int((dim_cases["Overall_Status"] == "At Risk").sum()),
    "Off_Track":          int((dim_cases["Overall_Status"] == "Off Track").sum()),
    "Phase1_In_Progress": p1_ip, "Phase1_Alert": p1_al,
    "Phase2_In_Progress": p2_ip, "Phase2_Alert": p2_al,
    "Phase3_In_Progress": p3_ip, "Phase3_Alert": p3_al,
    "Phase4_In_Progress": p4_ip, "Phase4_Alert": p4_al,
    "computed_at":        datetime.now(),
    "batch_id":           BATCH_ID,
}
kpi_snapshot = pd.DataFrame([kpi_row])

print(f"\nKPI snapshot — {kpi_row['Snapshot_Date']}")
for key in ("Total_Cases", "Completed", "On_Track", "At_Risk", "Off_Track"):
    print(f"  {key}: {kpi_row[key]}")


# =============================================================================
# CELL 7 — Write output tables
# =============================================================================

spark.createDataFrame(phase_detail).write.mode("overwrite").saveAsTable(OUT_PHASE)
print(f"{OUT_PHASE} written: {len(phase_detail):,} rows")

spark.createDataFrame(dim_cases).write.mode("overwrite").saveAsTable(OUT_CASES)
print(f"{OUT_CASES} written: {len(dim_cases):,} rows")

# Idempotent: drop any existing row for today's Snapshot_Date before appending,
# so same-day re-runs don't create duplicates.
spark.sql(f"DELETE FROM {OUT_KPI} WHERE Snapshot_Date = DATE('{kpi_row['Snapshot_Date']}')")
spark.createDataFrame(kpi_snapshot).write.mode("append").saveAsTable(OUT_KPI)
print(f"{OUT_KPI} upserted: snapshot {kpi_row['Snapshot_Date']}")


# =============================================================================
# CELL 8 — Verification
# =============================================================================

print("\n=== ROW COUNTS ===")
spark.sql(f"SELECT COUNT(*) AS rows FROM {OUT_PHASE}").show()
spark.sql(f"SELECT COUNT(DISTINCT Saksnummer) AS cases FROM {OUT_CASES}").show()
spark.sql(f"SELECT COUNT(*) AS kpi_rows FROM {OUT_KPI}").show()

print("\n=== PHASE STATUS DISTRIBUTION ===")
spark.sql(f"""
    SELECT Phase_Name, Phase_Status, COUNT(*) AS cases
    FROM {OUT_PHASE}
    GROUP BY Phase_Name, Phase_Status
    ORDER BY Phase_Name, Phase_Status
""").show(truncate=False)

print("\n=== OVERALL STATUS BY COMPLEXITY ===")
spark.sql(f"""
    SELECT Overall_Status, Case_Complexity, COUNT(*) AS cases
    FROM {OUT_CASES}
    GROUP BY Overall_Status, Case_Complexity
    ORDER BY Overall_Status, Case_Complexity
""").show(truncate=False)

print("\n=== ACTIVE ALERTS (first 50) ===")
spark.sql(f"""
    SELECT Saksnummer, Phase_Name, Alert_Reason
    FROM {OUT_PHASE}
    WHERE Alert_Flag = TRUE
    ORDER BY Saksnummer, Phase_Name
    LIMIT 50
""").show(truncate=False)

print("\n=== OFF-TRACK AND AT-RISK CASES ===")
spark.sql(f"""
    SELECT Saksnummer, Case_Complexity, Current_Phase,
           Total_Elapsed_Days, Days_To_Deadline,
           DATE_FORMAT(Projected_Completion_Date, 'yyyy-MM-dd') AS Projected,
           Alert_Count, Overall_Status
    FROM {OUT_CASES}
    WHERE Overall_Status IN ('Off Track', 'At Risk')
    ORDER BY Days_To_Deadline ASC
""").show(truncate=False)

print("\n=== DUPLICATE PHASE CHECK (expect 0) ===")
spark.sql(f"""
    SELECT Saksnummer, Phase_Name, COUNT(*) AS cnt
    FROM {OUT_PHASE}
    GROUP BY Saksnummer, Phase_Name
    HAVING cnt > 1
""").show()

print("\n=== CASES MISSING OM DATE (expect 0) ===")
spark.sql(f"""
    SELECT Saksnummer, Case_Complexity, Overall_Status
    FROM {OUT_CASES}
    WHERE Case_Start_Date IS NULL
""").show(truncate=False)


# =============================================================================
# CELL 9 — Power BI notes
# =============================================================================
#
# DATA MODEL:
#   plansak_2026_dim_cases  →[Saksnummer]→  plansak_2026_phase_detail
#   plansak_2026_kpi_snapshot: standalone (no join needed)
#
# RECOMMENDED VISUALS:
#
# 1. STATUS KPI CARDS (dim_cases):
#    Cards: Total / Completed / On Track / At Risk / Off Track
#    Slicer: Case_Complexity, Current_Phase
#
# 2. ALERT INTERVENTION TABLE (phase_detail WHERE Alert_Flag = TRUE):
#    Columns: Saksnummer, Phase_Name, Phase_Status, Alert_Reason, Calendar_Days
#    Sort: Days_To_Deadline (from dim_cases, via Saksnummer join)
#    Use: daily intervention list for case officers
#
# 3. DAYS-TO-DEADLINE HISTOGRAM (dim_cases):
#    X axis: Days_To_Deadline (bucketed: <0, 0–180, 180–365, 365+)
#    Y axis: count of cases
#    Reference line at Days_To_Deadline = 0 (deadline reached)
#    Color: Overall_Status  |  Slicer: Case_Complexity
#    Shows: how much buffer the portfolio has against rolling deadlines.
#    Alternative: scatter of Projected_Completion_Date vs Case_Deadline.
#
# 4. CASE DRILL-THROUGH (dim_cases → phase_detail):
#    Click any Saksnummer → drill to 4 phase rows.
#    Shows: Phase_Status, Calendar_Days, Tidsbruk vs Frist,
#           AM_Count vs AM_Expected_Avg, Last_AM_Date, KS_Date, Structural_Risk
#
# 5. UTARBEIDINGSFASEN SCATTER (phase_detail, Phase_Name = 'Utarbeidingsfasen Kostra'):
#    X axis: Calendar_Days  |  Y axis: AM_Count
#    Color: Structural_Risk  |  Reference line: Utarb_Budget_Days
#    Shows: cases using too many days with too few AMs → stuck cases
#
# 6. STALLED-CASE TABLE (dim_cases):
#    Filter: Overall_Status IN ('Off Track', 'At Risk')
#    Columns: Saksnummer, Current_Phase, Case_Complexity,
#             Days_Since_Last_AM_Utarb, Days_To_Deadline, Alert_Count
#    Sort: Days_To_Deadline ASC
#    Use: triage view — cases closest to (or past) their individual deadlines.
#
# 7. TREND OVER TIME (kpi_snapshot):
#    X axis: Snapshot_Date
#    Y axis: stacked bar — On Track / At Risk / Off Track
#    Shows: is the portfolio improving or deteriorating over time?
