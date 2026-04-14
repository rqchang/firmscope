"""
assign_loc.py
-------------
Constructs a time-varying firm location panel from Compustat auxiliary data.

For each firm-year this produces two location fields:

  hq_state     – US state of headquarters / physical operations
  incorp_state – US state of incorporation

Primary source:  compustat_histnames.rds  (crsp.comphist)
  Each row is a snapshot valid from hchgdt through hchgenddt (null = still
  current).  hstate gives the HQ state; hincorp the incorporation state.

Fallback:  compustat_conml.rds  (comp.company)
  comp.company.state is a static (current) US HQ state.  It is used to fill
  hq_state for firm-years that fall outside every histnames validity window
  (e.g., very early years before crsp.comphist coverage begins).
  There is no analogous static fallback for incorp_state because comp.company.fic
  is a country-level field, not a US state abbreviation.

Output: data/processed/Compustat/firm_location_panel.csv
  gvkey | year | hq_state | incorp_state
"""

import os
import warnings
import numpy as np
import pandas as pd
import pyreadr

warnings.filterwarnings("ignore", category=RuntimeWarning)
from utils.set_paths import RAW_DIR, PROC_DIR

RAW_CS  = os.path.join(RAW_DIR,  "Compustat")
PROC_CS = os.path.join(PROC_DIR, "Compustat")

YEAR_MIN = 1985
YEAR_MAX = 2023

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}


# ── 1. Load raw tables ────────────────────────────────────────────────────────
print("Loading compustat_histnames.rds …")
histnames = list(
    pyreadr.read_r(os.path.join(RAW_CS, "compustat_histnames.rds")).values()
)[0]

print("Loading compustat_conml.rds …")
conml = list(
    pyreadr.read_r(os.path.join(RAW_CS, "compustat_conml.rds")).values()
)[0]


# ── 2. Filter histnames to US-involved firms ──────────────────────────────────
# hfic = country of incorporation, hloc = country of HQ
# Keep firms where either is USA so we have all domestically operating entities.
histnames = histnames[
    histnames["hfic"].eq("USA") | histnames["hloc"].eq("USA")
].copy()
print(f"histnames after US filter: {len(histnames):,} rows, "
      f"{histnames['gvkey'].nunique():,} firms")


# ── 3. Parse validity-window dates to integer years ───────────────────────────
# hchgdt    – start of record (datetime.date, always present)
# hchgenddt – end of record   (datetime.date | None; None means still current)
histnames["start_year"] = (
    pd.to_datetime(histnames["hchgdt"], errors="coerce")
    .dt.year
    .fillna(YEAR_MIN)
    .astype(int)
)
histnames["end_year"] = (
    pd.to_datetime(histnames["hchgenddt"], errors="coerce")
    .dt.year
    .fillna(YEAR_MAX)       # null end → record is still current
    .astype(int)
    .clip(upper=YEAR_MAX)
)

locs = (
    histnames[["gvkey", "start_year", "end_year", "hstate", "hincorp"]]
    .sort_values(["gvkey", "start_year"])
    .reset_index(drop=True)
)


# ── 4. Build firm × year skeleton ────────────────────────────────────────────
# Expand each firm over only the years it actually has histnames coverage to
# keep the panel a manageable size before the merge.
firm_ranges = (
    locs.groupby("gvkey", sort=False)
    .agg(firm_start=("start_year", "min"), firm_end=("end_year", "max"))
    .reset_index()
)

panel_rows = [
    pd.DataFrame({
        "gvkey": r.gvkey,
        "year":  np.arange(max(r.firm_start, YEAR_MIN),
                           min(r.firm_end,   YEAR_MAX) + 1),
    })
    for r in firm_ranges.itertuples(index=False)
    if max(r.firm_start, YEAR_MIN) <= min(r.firm_end, YEAR_MAX)
]
panel = pd.concat(panel_rows, ignore_index=True)
print(f"Firm-year skeleton: {len(panel):,} rows")


# ── 5. Range-join: assign the active histnames record to each firm-year ───────
# Cross-merge on gvkey then filter to rows where start_year ≤ year ≤ end_year.
# This is the standard interval-overlap join in pandas.
# Intermediate size ≈ panel_rows × avg_records_per_firm (~8) → ~5 M rows max.
merged = pd.merge(
    panel,
    locs[["gvkey", "start_year", "end_year", "hstate", "hincorp"]],
    on="gvkey",
    how="left",
)
merged = merged[
    (merged["year"] >= merged["start_year"]) &
    (merged["year"] <= merged["end_year"])
].copy()

# If validity windows ever overlap (shouldn't in well-formed data), keep
# the record with the latest start_year.
merged = (
    merged.sort_values(["gvkey", "year", "start_year"])
    .drop_duplicates(subset=["gvkey", "year"], keep="last")
)

# Left-join back so firm-years outside every window are retained as NaN rows
# (those will be handled by the conml fallback in step 6).
panel = pd.merge(
    panel,
    merged[["gvkey", "year", "hstate", "hincorp"]],
    on=["gvkey", "year"],
    how="left",
)
panel = panel.rename(columns={"hstate": "hq_state", "hincorp": "incorp_state"})


# ── 6. Fallback: fill missing hq_state from conml static snapshot ─────────────
# comp.company.state is the current HQ US-state abbreviation.
# Apply only where histnames has no coverage (hq_state still NaN).
# We do NOT fall back for incorp_state: conml.fic is a country code, not a
# US state abbreviation.
conml_us = (
    conml[conml["fic"].eq("USA") | conml["loc"].eq("USA")]
    [["gvkey", "state"]]
    .rename(columns={"state": "_conml_hq"})
)
panel = pd.merge(panel, conml_us, on="gvkey", how="left")
panel["hq_state"] = panel["hq_state"].where(
    panel["hq_state"].notna(), panel["_conml_hq"]
)
panel = panel.drop(columns=["_conml_hq"])


# ── 7. Restrict location fields to valid US-state codes ───────────────────────
# Nullify any non-US-state values that slipped through (e.g. territory codes,
# Canadian provinces from cross-listed firms).
print(panel.count())
panel["hq_state"]     = panel["hq_state"].where(panel["hq_state"].isin(US_STATES))
panel["incorp_state"] = panel["incorp_state"].where(panel["incorp_state"].isin(US_STATES))


# ── 8. Final dedup & sort ─────────────────────────────────────────────────────
# merge_asof should already produce unique gvkey-year pairs, but guard against
# edge cases where duplicate start_year records exist in histnames.
panel = (
    panel
    .drop_duplicates(subset=["gvkey", "year"], keep="last")
    .sort_values(["gvkey", "year"])
    .reset_index(drop=True)
)
print(panel.count())


# ── 9. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(PROC_CS, "firm_location_fy.csv")
panel.to_csv(out_path, index=False)
