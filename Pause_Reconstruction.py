# =============================================================================
# Pause reconstruction — deadline clock per fase and weekly flow history.
# Runs nightly after the main data pipeline.
#
# Why:
#   Many cases sit on hold while the applicant completes documentation. That
#   time is outside the department's control and stops the deadline clock. The
#   fact table already carries the clock for TODAY — `tidsbruk` (our time) and
#   `bransjetid` (the applicant's time) are live accumulators — so "how many
#   faser are with us and at risk right now" is plain live DAX
#   (At_Risk_Open_Cases_POWERBI_DAX.md). What a live measure can't give is
#   HISTORY: how many were with us, waiting, and at risk last week, last year.
#   The "asked for / received completion" milestones let us rebuild that for
#   every week back to START_YEAR, so no ever-growing snapshot table is needed.
#
# Method:
#   1. Pause/resume milestones per fase are paired into pause intervals
#      (pair_pauses). A pause opens on the first request and closes on the
#      next receipt; repeated requests while open, and receipts without an
#      open request, are counted and ignored.
#   2. Per fase, the time between start and end (or today) is split into
#      active segments (with us) and pause segments (waiting), and the clock =
#      active days (phase_segments, klokke_dager).
#   3. The day the clock reaches FARESONE_TERSKEL x frist_dager, and the day it
#      passes frist_dager, are found by walking the active segments
#      (crossing_day). Once crossed, a fase stays in that state while with us.
#   4. Weekly stocks (with us / waiting / at risk / past deadline at each ISO
#      week's Sunday) are counted from those intervals (count_covering), and
#      weekly flows (received, completed, withdrawn, put on hold) are binned by
#      ISO week.
#
# Validation, printed every run (read it after the first run):
#   - reconstructed clock vs the system's own `tidsbruk`, and pause days vs
#     `bransjetid`, for closed faser — share within VALIDERING_TOLERANSE_DAGER;
#   - reconstructed current state vs `Status = "Under behandling"` for open faser.
#   If agreement is low, the milestone names below are probably incomplete or
#   the system uses a different clock rule — the history in flyt_ukentlig is
#   then approximate. The at-risk page (live tidsbruk) is unaffected either way.
#
# Output tables (overwritten nightly — rebuilt from source, not appended):
#   analyser.fase_klokke    — one row per fase: clock, pause days, state
#   analyser.flyt_ukentlig  — one row per ISO week x enhet x indikator
# Power BI/DAX: see Pause_Reconstruction_POWERBI_DAX.md
#
# Requires: nothing to install (pandas/numpy/pyspark only).
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, TimestampType, DateType,
)
import pandas as pd
import numpy as np
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()  # pyright: ignore[reportAttributeAccessIssue]
BATCH_ID   = datetime.now().strftime("%Y%m%dT%H%M%S")
START_YEAR = 2015

# -----------------------------------------------------------------------------
# Milestone source — ADJUST HERE if the first run stops with a message.
# Table/column names follow the repo's earlier milestone scripts
# (Milepeler: Prosess_id, Milepel, Sluttdato), with Prosess_id = today's
# pk_faser / fk_faser. Milestone names are the case system's own strings.
# -----------------------------------------------------------------------------
FASER_TABELL         = "saksbehandling.faser"
MILEPEL_TABELL       = "saksbehandling.milepeler"
MILEPEL_FASE_KOLONNE = "fk_faser"     # -> faser.pk_faser
MILEPEL_NAVN_KOLONNE = "milepel"
MILEPEL_DATO_KOLONNE = "sluttdato"

PAUSE_MILEPELER = [
    "Behov for tilleggsdokumentasjon",
    "Anmodning om tilleggsdokumentasjon",
]
GJENOPPTA_MILEPELER = [
    "Mottatt tilleggsdokumentasjon",
    "Komplett søknad",
]
TRUKKET_MILEPELER = [
    "00e Sak trukket/utgått/uaktuell",
]

STATUS_HOS_OSS            = "Under behandling"
FARESONE_TERSKEL          = 0.75   # same default as the at-risk page's what-if parameter
VALIDERING_TOLERANSE_DAGER = 2     # |reconstructed - system| within this counts as agreement

UENDELIG = 10 ** 9                 # "no end yet" in day numbers


# =============================================================================
# Pure helpers (tested in tests/test_pause_reconstruction.py)
# =============================================================================

def to_day(value):
    """date/Timestamp/str -> integer day number (days since 1970-01-01); None for missing."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int((pd.Timestamp(value).normalize() - pd.Timestamp("1970-01-01")).days)


def from_day(day):
    """Integer day number -> datetime.date."""
    return (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(day))).date()


def week_sunday(day):
    """Day number -> day number of the Sunday ending its ISO week (Mon-Sun).
    1970-01-01 was a Thursday, so Monday-based weekday = (day + 3) % 7."""
    return day + 6 - (day + 3) % 7


def iso_week_key(day):
    """Day number -> ISO year * 100 + ISO week (same key as PELT_Changepoint.py)."""
    iso = from_day(day).isocalendar()
    return iso[0] * 100 + iso[1]


def pair_pauses(events):
    """
    Pair pause/resume events per fase into pause intervals.

    events: DataFrame with columns pk_faser, dag (int day), hendelse
            ('pause' or 'gjenoppta').
    Rules: a pause opens on the first 'pause' while none is open; repeated
    'pause' while open is ignored (counted); 'gjenoppta' closes the open pause;
    'gjenoppta' with nothing open is ignored (counted). On the same day a
    resume is processed before a pause, so "received" + "new request" on one
    day closes one pause and opens the next.

    Returns (intervals, stats): intervals has pk_faser, start, slutt (slutt =
    UENDELIG if still open); stats counts the ignored events.
    """
    stats = {"gjentatt_anmodning": 0, "mottatt_uten_anmodning": 0}
    rows = []
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["pk_faser", "start", "slutt"]), stats

    order = events.assign(
        _o=events["hendelse"].map({"gjenoppta": 0, "pause": 1})
    ).sort_values(["pk_faser", "dag", "_o"])

    current_fase, open_start = None, None
    for fase, dag, hendelse in order[["pk_faser", "dag", "hendelse"]].itertuples(index=False):
        if fase != current_fase:
            if current_fase is not None and open_start is not None:
                rows.append((current_fase, open_start, UENDELIG))
            current_fase, open_start = fase, None
        if hendelse == "pause":
            if open_start is None:
                open_start = int(dag)
            else:
                stats["gjentatt_anmodning"] += 1
        else:
            if open_start is None:
                stats["mottatt_uten_anmodning"] += 1
            else:
                rows.append((fase, open_start, int(dag)))
                open_start = None
    if current_fase is not None and open_start is not None:
        rows.append((current_fase, open_start, UENDELIG))

    return pd.DataFrame(rows, columns=["pk_faser", "start", "slutt"]), stats


def phase_segments(start, end, pauses):
    """
    Split a fase's lifetime [start, end) into active (with us) and pause
    (waiting) segments. end=None means still open (runs to UENDELIG).
    pauses: iterable of (pause_start, pause_end); clipped to the fase and
    merged where they overlap. Returns (active, paused) as lists of (a, b)
    with a < b.
    """
    stop = UENDELIG if end is None else end
    clipped = sorted(
        (max(ps, start), min(pe, stop)) for ps, pe in pauses
        if min(pe, stop) > max(ps, start)
    )
    merged = []
    for a, b in clipped:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    active, cursor = [], start
    for a, b in merged:
        if a > cursor:
            active.append((cursor, a))
        cursor = max(cursor, b)
    if stop > cursor:
        active.append((cursor, stop))
    return active, merged


def overlap_days(segments, upto):
    """Total days of segments falling before day `upto`."""
    return int(sum(max(0, min(b, upto) - a) for a, b in segments if a < upto))


def crossing_day(active, need_days):
    """
    First day W at which the clock (active days in [start, W)) is >= need_days.
    None if the fase's active time never reaches it.
    """
    if need_days is None or not np.isfinite(need_days):
        return None
    need = float(need_days)
    if need <= 0:
        return active[0][0] if active else None
    done = 0.0
    for a, b in active:
        length = b - a
        if done + length >= need:
            return int(a + np.ceil(need - done))
        done += length
    return None


def clip_from(segments, from_day_):
    """Part of segments at or after day from_day_."""
    if from_day_ is None:
        return []
    return [(max(a, from_day_), b) for a, b in segments if b > max(a, from_day_)]


def count_covering(starts, ends, points):
    """Number of intervals [start, end) covering each point (start <= p < end)."""
    starts = np.sort(np.asarray(starts, dtype="int64"))
    ends = np.sort(np.asarray(ends, dtype="int64"))
    points = np.asarray(points, dtype="int64")
    return (np.searchsorted(starts, points, side="right")
            - np.searchsorted(ends, points, side="right"))


def build_phase_rows(faser, pauses_by_fase, trukket_faser, today,
                     terskel=FARESONE_TERSKEL):
    """
    Per fase: clock, pause days, state, and the labelled intervals the weekly
    stocks are counted from.

    faser: DataFrame with pk_faser, indikator, enhet, fasetittel, start (day),
           slutt (day or None), frist_dager, tidsbruk, bransjetid, status.
    pauses_by_fase: dict pk_faser -> list of (start, slutt).
    Returns (phase_rows: list of dict, intervals: DataFrame enhet, indikator,
    etikett, a, b).
    """
    phase_rows, iv = [], []
    for r in faser.itertuples(index=False):
        start = r.start
        end = None if r.slutt is None or (isinstance(r.slutt, float) and np.isnan(r.slutt)) else int(r.slutt)
        if start is None or (end is not None and end < start):
            continue
        start = int(start)
        active, paused = phase_segments(start, end, pauses_by_fase.get(r.pk_faser, []))

        ref = end if end is not None else today
        pause_dager = overlap_days(paused, ref)
        klokke = max(0, ref - start) - pause_dager
        first_pause = paused[0][0] if paused else None

        if end is not None:
            tilstand = "Avsluttet"
        elif any(a <= today < b for a, b in paused):
            tilstand = "Venter på søker"
        else:
            tilstand = "Hos oss"

        frist = r.frist_dager
        has_frist = frist is not None and not (isinstance(frist, float) and np.isnan(frist)) and frist > 0
        c_risk = crossing_day(active, terskel * frist) if has_frist else None
        c_over = crossing_day(active, frist + 1) if has_frist else None

        phase_rows.append({
            "pk_faser": r.pk_faser, "indikator": r.indikator, "enhet": r.enhet,
            "fasetittel": r.fasetittel, "start": start, "slutt": end,
            "frist_dager": float(frist) if has_frist else None,
            "klokke_dager": int(klokke), "pause_dager": int(pause_dager),
            "antall_pauser": len(paused),
            "dager_til_foerste_anmodning": (first_pause - start) if first_pause is not None else None,
            "tidsbruk": r.tidsbruk, "bransjetid": r.bransjetid, "status": r.status,
            "tilstand": tilstand, "trukket": r.pk_faser in trukket_faser,
            "faresone_dag": c_risk, "over_frist_dag": c_over,
        })
        for a, b in active:
            iv.append((r.enhet, r.indikator, "hos_oss", a, b))
        for a, b in paused:
            iv.append((r.enhet, r.indikator, "venter_paa_soeker", a, b))
        for a, b in clip_from(active, c_risk):
            iv.append((r.enhet, r.indikator, "i_faresonen", a, b))
        for a, b in clip_from(active, c_over):
            iv.append((r.enhet, r.indikator, "over_frist", a, b))

    intervals = pd.DataFrame(iv, columns=["enhet", "indikator", "etikett", "a", "b"])
    return phase_rows, intervals


STOCK_LABELS = ["hos_oss", "venter_paa_soeker", "i_faresonen", "over_frist"]
FLOW_LABELS  = ["mottatt", "ferdigstilt", "trukket", "satt_paa_vent"]


def weekly_table(intervals, flow_events, week_days):
    """
    One row per (week Sunday, enhet, indikator) with the four stocks counted
    at that Sunday and the four flows binned to that week. Rows where every
    count is zero are dropped.

    flow_events: DataFrame enhet, indikator, etikett (one of FLOW_LABELS), dag.
    """
    week_days = np.asarray(sorted(week_days), dtype="int64")
    parts = []
    for (enhet, indikator, etikett), g in intervals.groupby(["enhet", "indikator", "etikett"], sort=False):
        counts = count_covering(g["a"], g["b"], week_days)
        parts.append(pd.DataFrame({"uke_dag": week_days, "enhet": enhet,
                                   "indikator": indikator, "etikett": etikett,
                                   "antall": counts}))
    if len(flow_events):
        fe = flow_events.copy()
        fe["uke_dag"] = fe["dag"].astype("int64").map(week_sunday)
        fe = fe[fe["uke_dag"].isin(set(week_days.tolist()))]
        parts.append(fe.groupby(["uke_dag", "enhet", "indikator", "etikett"])
                       .size().rename("antall").reset_index())

    cols = ["uke_dag", "enhet", "indikator"] + STOCK_LABELS + FLOW_LABELS
    if not parts:
        return pd.DataFrame(columns=cols)
    long = pd.concat(parts, ignore_index=True)
    wide = (long.pivot_table(index=["uke_dag", "enhet", "indikator"], columns="etikett",
                             values="antall", aggfunc="sum", fill_value=0)
                .reindex(columns=STOCK_LABELS + FLOW_LABELS, fill_value=0)
                .reset_index())
    wide = wide[(wide[STOCK_LABELS + FLOW_LABELS] != 0).any(axis=1)]
    return wide[cols].astype({c: "int64" for c in STOCK_LABELS + FLOW_LABELS})


def agreement(reconstructed, system, tolerance=VALIDERING_TOLERANSE_DAGER):
    """(share within tolerance, median difference, n) over pairs where both are present."""
    r = pd.to_numeric(pd.Series(reconstructed), errors="coerce")
    s = pd.to_numeric(pd.Series(system), errors="coerce")
    mask = r.notna() & s.notna()
    if not mask.any():
        return None, None, 0
    diff = (r[mask] - s[mask]).to_numpy()
    return float(np.mean(np.abs(diff) <= tolerance)), float(np.median(diff)), int(mask.sum())


# =============================================================================
# CELL 1 — Check the milestone source before doing anything else
# =============================================================================
# A wrong table or milestone name would silently produce "nobody ever waits".
# Stop early with a message that says exactly what to change.

if not spark.catalog.tableExists(MILEPEL_TABELL):
    schema = MILEPEL_TABELL.split(".")[0] if "." in MILEPEL_TABELL else "default"
    kandidater = [t.name for t in spark.catalog.listTables(schema) if "mile" in t.name.lower()]
    raise ValueError(
        f"Fant ikke {MILEPEL_TABELL}. Sett MILEPEL_TABELL øverst i skriptet. "
        f"Tabeller i '{schema}' med 'mile' i navnet: {kandidater}"
    )

milepel_kolonner = {c.lower() for c in spark.table(MILEPEL_TABELL).columns}
mangler = [c for c in (MILEPEL_FASE_KOLONNE, MILEPEL_NAVN_KOLONNE, MILEPEL_DATO_KOLONNE)
           if c.lower() not in milepel_kolonner]
if mangler:
    raise ValueError(
        f"{MILEPEL_TABELL} mangler kolonne(ne) {mangler}. Juster MILEPEL_*_KOLONNE øverst. "
        f"Kolonner som finnes: {sorted(milepel_kolonner)}"
    )

alle_navn = PAUSE_MILEPELER + GJENOPPTA_MILEPELER + TRUKKET_MILEPELER
_navn_sql = ", ".join("'" + n.replace("'", "''") + "'" for n in alle_navn)
funnet = {
    r[0]: r[1] for r in spark.sql(f"""
        SELECT {MILEPEL_NAVN_KOLONNE} AS navn, COUNT(*) AS antall
        FROM {MILEPEL_TABELL}
        WHERE {MILEPEL_NAVN_KOLONNE} IN ({_navn_sql})
        GROUP BY {MILEPEL_NAVN_KOLONNE}
    """).collect()
}
print("Milepæler funnet (navn: antall):")
for n in alle_navn:
    print(f"  {n}: {funnet.get(n, 0):,}")

if not any(funnet.get(n) for n in PAUSE_MILEPELER) or not any(funnet.get(n) for n in GJENOPPTA_MILEPELER):
    lignende = [r[0] for r in spark.sql(f"""
        SELECT DISTINCT {MILEPEL_NAVN_KOLONNE}
        FROM {MILEPEL_TABELL}
        WHERE LOWER({MILEPEL_NAVN_KOLONNE}) LIKE '%tillegg%'
           OR LOWER({MILEPEL_NAVN_KOLONNE}) LIKE '%komplett%'
           OR LOWER({MILEPEL_NAVN_KOLONNE}) LIKE '%mangel%'
           OR LOWER({MILEPEL_NAVN_KOLONNE}) LIKE '%trukket%'
    """).collect()]
    raise ValueError(
        "Ingen pause- eller gjenopptakelses-milepæler funnet med navnene i "
        "PAUSE_MILEPELER/GJENOPPTA_MILEPELER. Lignende navn i tabellen: "
        f"{sorted(lignende)}"
    )

faser_kolonner = {c.lower() for c in spark.table(FASER_TABELL).columns}
HAR_BRANSJETID = "bransjetid" in faser_kolonner
print(f"bransjetid på {FASER_TABELL}: {'ja' if HAR_BRANSJETID else 'nei — hopper over den kontrollen'}")


# =============================================================================
# CELL 2 — Create output tables
# =============================================================================

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.fase_klokke (
    pk_faser                     STRING      NOT NULL,
    indikator                    STRING      NOT NULL,
    enhet                        STRING      NOT NULL,
    fasetittel                   STRING      NOT NULL,
    startdato                    DATE        NOT NULL,
    sluttdato                    DATE,
    frist_dager                  DOUBLE,
    klokke_dager                 INT         NOT NULL,
    pause_dager                  INT         NOT NULL,
    antall_pauser                INT         NOT NULL,
    dager_til_foerste_anmodning  INT,
    tilstand                     STRING      NOT NULL,
    trukket                      BOOLEAN     NOT NULL,
    avvik_mot_tidsbruk           DOUBLE,
    kjoert_tidspunkt             TIMESTAMP   NOT NULL,
    kjoere_id                    STRING      NOT NULL
)
USING DELTA
COMMENT 'Fristklokke rekonstruert fra milepæler per fase. klokke_dager = dager hos oss (kalenderdager fra start til slutt/i dag minus pause_dager). pause_dager = dager på vent etter anmodning om tilleggsdokumentasjon. tilstand = Hos oss / Venter på søker / Avsluttet. dager_til_foerste_anmodning = dager fra start til første anmodning. avvik_mot_tidsbruk = klokke_dager minus systemets tidsbruk (kontroll; nær 0 betyr at rekonstruksjonen stemmer). trukket = fasen har milepæl for trukket/utgått/uaktuell. For dagens tall: bruk saksbehandling.faser[tidsbruk] direkte.'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS analyser.flyt_ukentlig (
    uke_slutt           DATE        NOT NULL,
    iso_uke             INT         NOT NULL,
    enhet               STRING      NOT NULL,
    indikator           STRING      NOT NULL,
    hos_oss             INT         NOT NULL,
    venter_paa_soeker   INT         NOT NULL,
    i_faresonen         INT         NOT NULL,
    over_frist          INT         NOT NULL,
    mottatt             INT         NOT NULL,
    ferdigstilt         INT         NOT NULL,
    trukket             INT         NOT NULL,
    satt_paa_vent       INT         NOT NULL,
    kjoert_tidspunkt    TIMESTAMP   NOT NULL,
    kjoere_id           STRING      NOT NULL
)
USING DELTA
COMMENT 'Ukentlig flythistorikk per enhet og indikator, rekonstruert fra milepæler (ingen snapshot). Beholdninger telles søndag i ISO-uken: hos_oss (åpen, ikke på vent), venter_paa_soeker, i_faresonen (hos oss og fristklokke >= 75 % av frist_dager), over_frist (hos oss og klokke > frist_dager). Strømmer i uken: mottatt (faser startet), ferdigstilt (avsluttet, ikke trukket), trukket, satt_paa_vent (anmodninger sendt). Bare hele uker.'
""")

print("fase_klokke og flyt_ukentlig er klare")


# =============================================================================
# CELL 3 — Load faser and milestones
# =============================================================================

TODAY = to_day(date.today())
_bransjetid_sql = "CAST(pr.bransjetid AS DOUBLE)" if HAR_BRANSJETID else "CAST(NULL AS DOUBLE)"

faser_pd = spark.sql(f"""
    SELECT
        CAST(pr.pk_faser AS STRING)                                  AS pk_faser,
        CAST(pr.indikator AS STRING)                                 AS indikator,
        COALESCE(NULLIF(TRIM(pr.enhet), ''), 'Ukjent')               AS enhet,
        COALESCE(NULLIF(TRIM(pr.fasetittel), ''), 'Ukjent fase')     AS fasetittel,
        CAST(pr.startmilepaeldato AS DATE)                           AS startdato,
        CAST(pr.sluttmilepaeldato AS DATE)                           AS sluttdato,
        CAST(pr.frist_dager AS DOUBLE)                               AS frist_dager,
        CAST(pr.tidsbruk AS DOUBLE)                                  AS tidsbruk,
        {_bransjetid_sql}                                            AS bransjetid,
        CAST(pr.status AS STRING)                                    AS status
    FROM {FASER_TABELL} pr
    INNER JOIN felles.indikator indikator
        ON indikator.pk_indikator = pr.indikator
    WHERE pr.startmilepaeldato IS NOT NULL
      AND indikator.fagomraade IN ('Byggesak', 'Eiendomssak', 'Plansak')
      AND COALESCE(pr.sluttmilepaeldato, CURRENT_DATE()) >= DATE'{START_YEAR}-01-01'
""").toPandas()

milepeler_pd = spark.sql(f"""
    SELECT
        CAST(m.{MILEPEL_FASE_KOLONNE} AS STRING) AS pk_faser,
        CAST(m.{MILEPEL_DATO_KOLONNE} AS DATE)   AS dato,
        m.{MILEPEL_NAVN_KOLONNE}                 AS milepel
    FROM {MILEPEL_TABELL} m
    WHERE m.{MILEPEL_NAVN_KOLONNE} IN ({_navn_sql})
      AND m.{MILEPEL_DATO_KOLONNE} IS NOT NULL
      AND m.{MILEPEL_FASE_KOLONNE} IS NOT NULL
""").toPandas()

faser_pd["start"] = faser_pd["startdato"].map(to_day)
faser_pd["slutt"] = faser_pd["sluttdato"].map(to_day).astype(object)
faser_pd["slutt"] = faser_pd["slutt"].where(faser_pd["slutt"].notna(), None)
for col in ("frist_dager", "tidsbruk", "bransjetid"):
    faser_pd[col] = pd.to_numeric(faser_pd[col], errors="coerce")

milepeler_pd = milepeler_pd[milepeler_pd["pk_faser"].isin(set(faser_pd["pk_faser"]))].copy()
milepeler_pd["dag"] = milepeler_pd["dato"].map(to_day)
milepeler_pd["hendelse"] = np.select(
    [milepeler_pd["milepel"].isin(PAUSE_MILEPELER), milepeler_pd["milepel"].isin(GJENOPPTA_MILEPELER)],
    ["pause", "gjenoppta"], default="trukket",
)

print(f"Faser lastet:     {len(faser_pd):,}")
print(f"Milepæler lastet: {len(milepeler_pd):,}")


# =============================================================================
# CELL 4 — Reconstruct pauses, clock and intervals
# =============================================================================

pauser, pause_stats = pair_pauses(milepeler_pd[milepeler_pd["hendelse"] != "trukket"])
print(f"Pauseintervaller: {len(pauser):,}  "
      f"(åpne nå: {(pauser['slutt'] == UENDELIG).sum():,}; "
      f"ignorert — gjentatt anmodning: {pause_stats['gjentatt_anmodning']:,}, "
      f"mottatt uten anmodning: {pause_stats['mottatt_uten_anmodning']:,})")

pauses_by_fase = {}
for fase, s, e in pauser[["pk_faser", "start", "slutt"]].itertuples(index=False):
    pauses_by_fase.setdefault(fase, []).append((int(s), int(e)))

trukket_faser = set(milepeler_pd.loc[milepeler_pd["hendelse"] == "trukket", "pk_faser"])

phase_rows, intervals = build_phase_rows(faser_pd, pauses_by_fase, trukket_faser, TODAY)
fase_df = pd.DataFrame(phase_rows)
print(f"Faser rekonstruert: {len(fase_df):,}")


# =============================================================================
# CELL 5 — Validate against the system's own clock and status
# =============================================================================
# Read this output after the first run. High agreement = the milestone names
# and the pause rule match the case system; the weekly history is trustworthy.

lukket = fase_df[fase_df["tilstand"] == "Avsluttet"]
andel, median, n = agreement(lukket["klokke_dager"], lukket["tidsbruk"])
print("\nKONTROLL 1 — rekonstruert klokke mot systemets tidsbruk (avsluttede faser):")
if n:
    print(f"  {andel:.1%} innenfor ±{VALIDERING_TOLERANSE_DAGER} dager, median avvik {median:+.1f} dager, n={n:,}")
    per_ind = (
        lukket.assign(ok=(lukket["klokke_dager"] - lukket["tidsbruk"]).abs() <= VALIDERING_TOLERANSE_DAGER)
        .dropna(subset=["tidsbruk"]).groupby("indikator")["ok"].agg(["mean", "size"])
        .sort_values("mean").head(10)
    )
    print("  Svakest samsvar per indikator (andel, antall):")
    for ind, r in per_ind.iterrows():
        print(f"    {ind}: {r['mean']:.1%} ({int(r['size']):,})")
else:
    print("  Ingen avsluttede faser med tidsbruk — kan ikke kontrollere")

if HAR_BRANSJETID:
    andel_b, median_b, n_b = agreement(lukket["pause_dager"], lukket["bransjetid"])
    print("KONTROLL 2 — rekonstruerte pausedager mot systemets bransjetid (avsluttede faser):")
    if n_b:
        print(f"  {andel_b:.1%} innenfor ±{VALIDERING_TOLERANSE_DAGER} dager, median avvik {median_b:+.1f} dager, n={n_b:,}")

aapne = fase_df[fase_df["tilstand"] != "Avsluttet"]
if len(aapne):
    hos_oss_status = aapne["status"].fillna("") == STATUS_HOS_OSS
    hos_oss_rek = aapne["tilstand"] == "Hos oss"
    print("KONTROLL 3 — rekonstruert tilstand mot Status (åpne faser i dag):")
    print(f"  Begge 'hos oss':            {int((hos_oss_status & hos_oss_rek).sum()):,}")
    print(f"  Begge 'ikke hos oss':       {int((~hos_oss_status & ~hos_oss_rek).sum()):,}")
    print(f"  Status hos oss, vi: venter: {int((hos_oss_status & ~hos_oss_rek).sum()):,}")
    print(f"  Status venter, vi: hos oss: {int((~hos_oss_status & hos_oss_rek).sum()):,}")
    print(f"  Samsvar: {float((hos_oss_status == hos_oss_rek).mean()):.1%}")
    print("  (Status kan også være 'venter på ekstern part' — det rekonstrueres ikke fra disse milepælene.)")


# =============================================================================
# CELL 6 — Weekly history
# =============================================================================

first_sunday = week_sunday(to_day(date(START_YEAR, 1, 1)))
last_complete_sunday = week_sunday(TODAY) - 7
week_days = list(range(first_sunday, last_complete_sunday + 1, 7))

# Once phase rows are a DataFrame, a missing end date is NaN, not None.
fase_df["slutt"] = fase_df["slutt"].astype(object).where(fase_df["slutt"].notna(), None)

flow_rows = []
for r in fase_df.itertuples(index=False):
    flow_rows.append((r.enhet, r.indikator, "mottatt", int(r.start)))
    if r.slutt is not None:
        flow_rows.append((r.enhet, r.indikator, "trukket" if r.trukket else "ferdigstilt", int(r.slutt)))
flow_events = pd.DataFrame(flow_rows, columns=["enhet", "indikator", "etikett", "dag"])
paa_vent = pauser.merge(fase_df[["pk_faser", "enhet", "indikator"]], on="pk_faser", how="inner")
if len(paa_vent):
    flow_events = pd.concat([
        flow_events,
        pd.DataFrame({"enhet": paa_vent["enhet"], "indikator": paa_vent["indikator"],
                      "etikett": "satt_paa_vent", "dag": paa_vent["start"].astype("int64")}),
    ], ignore_index=True)

uke_df = weekly_table(intervals, flow_events, week_days)
print(f"\nUkerader: {len(uke_df):,} ({len(week_days):,} uker, "
      f"{from_day(first_sunday)} – {from_day(last_complete_sunday)})")


# =============================================================================
# CELL 7 — Write to Lakehouse
# =============================================================================

kjoert = datetime.now()

KLOKKE_SCHEMA = StructType([
    StructField("pk_faser", StringType(), False),
    StructField("indikator", StringType(), False),
    StructField("enhet", StringType(), False),
    StructField("fasetittel", StringType(), False),
    StructField("startdato", DateType(), False),
    StructField("sluttdato", DateType(), True),
    StructField("frist_dager", DoubleType(), True),
    StructField("klokke_dager", IntegerType(), False),
    StructField("pause_dager", IntegerType(), False),
    StructField("antall_pauser", IntegerType(), False),
    StructField("dager_til_foerste_anmodning", IntegerType(), True),
    StructField("tilstand", StringType(), False),
    StructField("trukket", BooleanType(), False),
    StructField("avvik_mot_tidsbruk", DoubleType(), True),
    StructField("kjoert_tidspunkt", TimestampType(), False),
    StructField("kjoere_id", StringType(), False),
])

UKE_SCHEMA = StructType([
    StructField("uke_slutt", DateType(), False),
    StructField("iso_uke", IntegerType(), False),
    StructField("enhet", StringType(), False),
    StructField("indikator", StringType(), False),
] + [StructField(c, IntegerType(), False) for c in STOCK_LABELS + FLOW_LABELS] + [
    StructField("kjoert_tidspunkt", TimestampType(), False),
    StructField("kjoere_id", StringType(), False),
])


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


klokke_records = [
    (
        r.pk_faser, str(r.indikator), str(r.enhet), str(r.fasetittel),
        from_day(r.start), None if r.slutt is None else from_day(r.slutt),
        None if _num(r.frist_dager) is None else float(r.frist_dager),
        int(r.klokke_dager), int(r.pause_dager), int(r.antall_pauser),
        None if _num(r.dager_til_foerste_anmodning) is None else int(r.dager_til_foerste_anmodning),
        r.tilstand, bool(r.trukket),
        None if _num(r.tidsbruk) is None else float(r.klokke_dager - r.tidsbruk),
        kjoert, BATCH_ID,
    )
    for r in fase_df.itertuples(index=False)
]

uke_records = [
    (from_day(r.uke_dag), int(iso_week_key(r.uke_dag)), str(r.enhet), str(r.indikator))
    + tuple(int(getattr(r, c)) for c in STOCK_LABELS + FLOW_LABELS)
    + (kjoert, BATCH_ID)
    for r in uke_df.itertuples(index=False)
]

if klokke_records:
    spark.createDataFrame(klokke_records, schema=KLOKKE_SCHEMA) \
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("analyser.fase_klokke")
    print(f"fase_klokke skrevet: {len(klokke_records):,} rader")

if uke_records:
    spark.createDataFrame(uke_records, schema=UKE_SCHEMA) \
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("analyser.flyt_ukentlig")
    print(f"flyt_ukentlig skrevet: {len(uke_records):,} rader")

    spark.sql(f"""
        SELECT uke_slutt,
               SUM(hos_oss) AS hos_oss, SUM(venter_paa_soeker) AS venter,
               SUM(i_faresonen) AS i_faresonen, SUM(over_frist) AS over_frist,
               SUM(mottatt) AS mottatt, SUM(ferdigstilt) AS ferdigstilt
        FROM analyser.flyt_ukentlig
        WHERE kjoere_id = '{BATCH_ID}'
        GROUP BY uke_slutt
        ORDER BY uke_slutt DESC
        LIMIT 12
    """).show(12, truncate=False)
