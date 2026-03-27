"""
import_funding.py
-----------------
Pulls state-level innovation-oriented public funding data for EIP Component 3:
Innovation Funding Intensity (Fund_{s,t}).

Conceptual basis
----------------
Public R&D resource commitments are measured following the OECD Frascati Manual
(2015) character-of-work taxonomy (basic research / applied research /
experimental development). CFDA program selection follows:
  - Howell (2017, AER): SBIR/STTR awards are definitionally R&D by statute.
  - Azoulay, Graff Zivin, Li, Sampat (2019, REStud): NIH IC/study-section
    structure as the unit of R&D classification.
  - NSF Survey of Federal Funds for R&D (NCSES): agency-level budget data.
Innovation domain tags follow Hall, Jaffe, Trajtenberg (2001, NBER WP 8498)
HJT patent technology taxonomy (6 categories).

Data sources (all public)
-----------------------------------
Three non-overlapping federal R&D funding streams are combined to avoid
double-counting. NSF and NIH are pulled from their own authoritative APIs;
USASpending covers the remaining agencies (DOE, DARPA) not in Steps 1-2.

  1. NSF Award Search API v1  – all NSF directorates (CFDA 47.*); pulled
                                 per-state in parallel; ~$9B/year
  2. NIH RePORTER API v2      – all NIH programs (CFDA 93.*); pulled
                                 per-state in parallel; ~$40B/year
  3. USASpending.gov API      – DOE Office of Science + DARPA only
                                 (CFDA 81.*, 12.910); NSF/NIH excluded to
                                 prevent double-counting; optional, slow;
                                 API coverage: FY2008 onwards only
  4. BEA Regional API         – Gross State Product for GSP normalization
  5. U.S. Census PEP API      – State population for per-capita normalization

Directory layout
----------------
  RAW_DIR/funding/
    NSF/                nsf_{year}.csv          one row per award
    NIH/                nih_{year}.csv          one row per project
    USASpending/        usaspending_{year}.csv  one row per grant (all states)
    BEA/                gsp_panel.csv           state x year GSP
    Census/             population_panel.csv    state x year population

  PROC_DIR/eip/
    funding_panel.csv            primary (state, year) panel -- see schema below
    import_funding_{ts}.log      run log with fetch stats and errors

Output schemas
--------------
  RAW_DIR/funding/NSF/nsf_{year}.csv
    nsf_id          str    NSF award ID
    state           str    2-letter state abbreviation
    year            int    calendar year of award
    title           str    award title
    awardee         str    recipient organization
    cfda            str    CFDA program code (e.g. "47.070")
    domain          str    HJT innovation domain (see INNOVATION_DOMAINS)
    amount_usd      float  obligated amount in USD
    start_date      str    award start date (MM/DD/YYYY)

  RAW_DIR/funding/NIH/nih_{year}.csv
    nih_id          str    NIH project number (e.g. R01CA123456)
    state           str    2-letter state abbreviation
    year            int    fiscal year
    activity_code   str    NIH activity code (R01, R21, P01, U01, ...)
    title           str    project title (truncated to 200 chars)
    domain          str    HJT innovation domain
    amount_usd      float  award amount in USD

  RAW_DIR/funding/USASpending/usaspending_{year}.csv
                              Note: FY2008+ only (API limit: start_date >= 2007-10-01)
    state           str    2-letter state abbreviation
    year            int    fiscal year (Oct-Sep); FY2008-FY2025 only
    award_id        str    USASpending award ID
    recipient       str    recipient organization name
    amount_usd      float  award amount in USD
    cfda_number     str    CFDA program number (DOE/DARPA codes only)
    cfda_title      str    CFDA program title
    award_type      str    award type code (02-05 = grants/cooperative agreements)
    start_date      str    award start date
    rd_tier         str    always "direct" (NSF/NIH excluded; all codes are R&D by statute)

  RAW_DIR/funding/BEA/gsp_panel.csv
    state           str    2-letter state abbreviation
    year            int    calendar year
    gsp_millions    float  real GDP in chained dollars (millions) -- BEA SAGDP2N Line 1

  RAW_DIR/funding/Census/population_panel.csv
    state           str    2-letter state abbreviation
    year            int    calendar year
    population      int    resident population estimate (Census PEP)

  PROC_DIR/eip/funding_panel.csv  <- primary output for EIP construction
    state                     str    2-letter state abbreviation
    year                      int    calendar/fiscal year
    nsf_grants_usd            float  total NSF obligations, all directorates ($)
    nih_grants_usd            float  total NIH awards, all ICs ($)
    usaspending_rd_grants_usd float  DOE + DARPA grants only ($); 0 if skipped
    nsf_{domain}_usd          float  NSF grants by HJT domain (9 columns)
    nih_{domain}_usd          float  NIH grants by HJT domain (9 columns)
    gsp_millions              float  real GSP in chained dollars (millions); NaN if no BEA key
    population                int    resident population; NaN if no Census key
    total_rd_funding_usd      float  NSF + NIH + DOE/DARPA, non-overlapping ($)
    rd_funding_pct_gsp        float  total_rd_funding_usd / (gsp_millions * 1e6)
    rd_funding_per_capita     float  total_rd_funding_usd / population ($)
      HJT domains (9): sbir_sttr | it_digital | clean_energy | biotech_health |
                       materials_mfg | aerospace_space | basic_science |
                       social_science | other

Usage
-----
  python import_funding.py --start 1990 --end 2025
  python import_funding.py --bea_key YOUR_KEY --census_key YOUR_KEY --workers 8
  python import_funding.py --start 2020 --end 2020 --force
  python data_import/import_funding.py --start 1990 --end 2025 --panel_only --usaspending --force

API keys (all free)
-------------------
  BEA:    https://apps.bea.gov/API/signup/        -> env var BEA_API_KEY
  Census: https://api.census.gov/data/key_signup.html -> env var CENSUS_API_KEY
  NSF / NIH / USASpending: no key required

Requirements
------------
  pip install requests pandas tqdm
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup – resolve project root and load shared path config
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.set_paths import RAW_DIR, PROC_DIR

logger = logging.getLogger("import_funding")

SLEEP = 0.5  # seconds between requests

# ---------------------------------------------------------------------------
# CFDA (Catalog of Federal Domestic Assistance) program codes for R&D funding.
#
# Split into two tiers following the methodology in:
#   Howell (2017, AER) — SBIR awards are R&D by statute; no keyword needed.
#   NSF Survey of Federal Funds for R&D (NCSES) — agency-level R&D budgets.
#   Azoulay, Graff Zivin, Li, Sampat (2019, REStud) — NIH program structure.
#
# Tier 1 (RD_CFDA_DIRECT): specific codes where every award is definitionally
#   R&D by statute or agency mandate. No keyword filtering applied.
# Tier 2 (RD_CFDA_BROAD_PREFIXES): agency-wide prefixes where R&D grants
#   must be separated from non-R&D assistance using RD_KEYWORD_FILTER.
# ---------------------------------------------------------------------------

# Tier 1 — definitionally R&D by statute (no keyword filter needed)
RD_CFDA_DIRECT = [
    # NSF — all directorates are R&D by statute (NSF Act of 1950)
    "47.041",  # Engineering
    "47.049",  # Mathematical and Physical Sciences
    "47.050",  # Geosciences
    "47.070",  # Computer and Information Science and Engineering (CISE)
    "47.074",  # Biological Sciences
    "47.075",  # Social, Behavioral and Economic Sciences
    "47.076",  # STEM Education
    "47.079",  # International Science and Engineering
    "47.083",  # Office of Integrative Activities
    # DOE — core science and energy R&D offices
    "81.049",  # Office of Science Financial Assistance
    "81.051",  # Energy Efficiency and Renewable Energy (Howell 2017 AER)
    "81.057",  # Fossil Energy R&D
    "81.135",  # DOE SBIR Phase I
    "81.136",  # DOE SBIR Phase II
    # NIH — key extramural research grant programs (NIH IC structure per
    #   Azoulay, Graff Zivin, Li, Sampat 2019 REStud DST framework)
    "93.242",  # Mental Health Research (NIMH)
    "93.286",  # Translational Research (NCATS)
    "93.361",  # NIH SBIR/STTR (all ICs)
    "93.837",  # Heart and Vascular Diseases (NHLBI)
    "93.838",  # Lung Diseases (NHLBI)
    "93.839",  # Blood Diseases (NHLBI)
    "93.853",  # Cancer Research (NCI)
    "93.855",  # Allergy and Infectious Diseases (NIAID)
    "93.859",  # Pharmacology and Biological Chemistry (NIGMS)
    "93.865",  # Child Health and Human Development (NICHD)
    "93.867",  # Vision Research (NEI)
    # DARPA — Defense Advanced Research Projects Agency
    "12.910",  # Research and Technology Development
]

# Tier 2 — broad agency prefixes: apply RD_KEYWORD_FILTER to discriminate
# R&D grants from non-R&D assistance within the same agency
RD_CFDA_BROAD_PREFIXES = [
    "12.",   # DoD (ARO, ONR, AFOSR — excluding DARPA direct above)
    "43.",   # NASA
    "11.",   # Commerce / NIST / EDA
    "59.",   # SBA (SBIR/STTR across all participating agencies)
    "20.",   # DOT (autonomous vehicles, broadband infrastructure)
    "10.",   # USDA (agricultural research, biotech)
    "81.",   # DOE (broader programs beyond direct codes above)
    "93.",   # HHS/NIH (broader programs beyond direct codes above)
]

# ---------------------------------------------------------------------------
# Keyword filter for Tier 2 CFDA programs.
#
# Grounded in:
#   OECD Frascati Manual 2015 (Ch.2) — five R&D criteria and activity types.
#   NSF Survey of Federal Funds for R&D (NCSES) — character-of-work taxonomy.
#   Hall, Jaffe, Trajtenberg (2001, NBER WP 8498) — HJT patent technology
#     taxonomy (6 main categories: Chemical, Computers & Comms, Drugs &
#     Medical, Electrical & Electronic, Mechanical, Other).
#   Bellstam, Bhagat, Cookson (2021, Management Science) — LDA innovation
#     topic top words: service, system, technology, product, solution.
#   Howell (2017, AER) — SBIR phase terminology and DOE program areas.
# ---------------------------------------------------------------------------
RD_KEYWORD_FILTER = [
    # --- Frascati Manual R&D activity types (OECD 2015, Ch.2) ---
    "basic research", "applied research", "experimental development",
    "research and development", "r&d", "r & d",
    # --- Core R&D signals ---
    "research", "innovation", "technology", "science",
    "discovery", "invention", "novel", "novelty", "breakthrough",
    "frontier", "state of the art", "knowledge creation",
    # --- Frascati project-stage terms (Howell 2017 SBIR phase vocabulary) ---
    "proof of concept", "prototype", "feasibility", "pilot",
    "demonstration", "scale-up", "technology readiness",
    "commercialization", "translation", "technology transfer",
    "phase i", "phase ii", "phase iii",
    # --- SBIR / STTR (statutory R&D programs) ---
    "sbir", "sttr", "small business innovation",
    # --- HJT Computers & Communications category ---
    "computer science", "software", "information technology",
    "artificial intelligence", "machine learning", "algorithm",
    "autonomous", "robotics", "data center", "cybersecurity",
    "broadband", "5g", "fiber optic", "internet of things",
    # --- HJT Electrical & Electronic category ---
    "semiconductor", "microelectronics", "integrated circuit",
    "quantum", "photonics", "nanotechnology", "advanced materials",
    "superconductor", "composites",
    # --- HJT Drugs & Medical category ---
    "biotechnology", "genomics", "life science", "precision medicine",
    "biomedical", "clinical research", "translational research",
    "drug discovery", "bioinformatics", "synthetic biology",
    # --- HJT Chemical category ---
    "chemistry", "chemical", "materials science",
    # --- HJT Mechanical / Environmental & Energy category ---
    "advanced manufacturing", "clean energy", "renewable energy",
    "electric vehicle", "battery storage", "hydrogen", "nuclear",
    "grid modernization", "carbon capture", "energy efficiency",
    # --- HJT Other: aerospace/space ---
    "aerospace", "space", "satellite", "launch vehicle",
    # --- NSF directorate areas (NCSES character-of-work taxonomy) ---
    "mathematics", "physics", "biology", "engineering", "geosciences",
    "social science", "behavioral science",
    # --- DOD R&D budget activities (BA 1–5, NSF Federal Funds survey) ---
    "advanced technology development", "prototype development",
    "system development", "technology maturation",
    # --- Workforce / STEM ---
    "stem", "workforce development", "training",
    # --- Policy programs ---
    "chips", "inflation reduction act", "industrial policy",
    # --- Bellstam et al. (2021) LDA innovation topic core words ---
    "solution", "system", "product innovation", "process innovation",
]

# ---------------------------------------------------------------------------
# Innovation domain taxonomy
# Follows Hall, Jaffe, Trajtenberg (2001, NBER WP 8498) HJT classification.
# Used to tag each grant with a domain for per-category panel variables.
# ---------------------------------------------------------------------------

# NSF CFDA code -> domain (primary classifier; exact, no text needed)
NSF_CFDA_DOMAIN: dict[str, str] = {
    "47.041": "materials_mfg",   # Engineering
    "47.049": "basic_science",   # Mathematical & Physical Sciences
    "47.050": "basic_science",   # Geosciences
    "47.070": "it_digital",      # Computer & Information Science (CISE)
    "47.074": "biotech_health",  # Biological Sciences
    "47.075": "social_science",  # Social, Behavioral & Economic Sciences
    "47.076": "social_science",  # STEM Education
    "47.079": "other",           # International Science & Engineering
    "47.083": "other",           # Office of Integrative Activities
}

# Title-keyword classifier (fallback for NSF; primary for NIH/USASpending).
# Ordered: first match wins. Follows HJT 6-category structure.
INNOVATION_DOMAINS: list[tuple[str, list[str]]] = [
    ("sbir_sttr",      ["sbir", "sttr", "small business innovation"]),
    ("it_digital",     ["computer", "software", "artificial intelligence",
                        "machine learning", "cybersecurity", "broadband",
                        "digital", "computing", "internet", "algorithm",
                        "quantum computing", "data science", "autonomous",
                        "information technology", "robotics"]),
    ("clean_energy",   ["renewable", "solar", "wind energy", "electric vehicle",
                        "battery", "hydrogen fuel", "grid modernization",
                        "carbon capture", "climate", "nuclear energy",
                        "fuel cell", "photovoltaic", "bioenergy",
                        "energy efficiency", "decarbonization"]),
    ("biotech_health", ["biology", "genomics", "biotechnology", "life science",
                        "medicine", "health", "biomedical", "clinical",
                        "pharmaceutical", "cancer", "disease", "neuroscience",
                        "genetics", "immunology", "biochemistry", "molecular",
                        "public health", "epidemiology", "drug discovery",
                        "precision medicine", "crispr"]),
    ("materials_mfg",  ["materials science", "semiconductor", "nanotechnology",
                        "photonics", "manufacturing", "additive manufacturing",
                        "composite", "polymer", "superconductor", "3d printing",
                        "advanced materials", "metallurgy"]),
    ("aerospace_space",["aerospace", "space", "satellite", "aviation",
                        "launch vehicle", "propulsion", "rocket"]),
    ("basic_science",  ["physics", "chemistry", "mathematics", "geoscience",
                        "earth science", "astronomy", "oceanography",
                        "atmospheric", "astrophysics", "geology"]),
    ("social_science", ["social science", "economics", "behavioral science",
                        "education", "workforce development", "stem education",
                        "cognitive science"]),
]

ALL_DOMAINS = [d for d, _ in INNOVATION_DOMAINS] + ["other"]


def classify_domain(text: str, cfda: str = "") -> str:
    """Return the first matching HJT innovation domain, or 'other'."""
    if cfda and cfda in NSF_CFDA_DOMAIN:
        return NSF_CFDA_DOMAIN[cfda]
    t = text.lower()
    for domain, keywords in INNOVATION_DOMAINS:
        if any(kw in t for kw in keywords):
            return domain
    return "other"


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def setup_dirs() -> dict[str, Path]:
    """
    Create and return all required raw and processed directories.
    Checks RAW_DIR and PROC_DIR from set_paths.py; creates subdirs if absent.
    """
    dirs = {
        "raw_nsf":    Path(RAW_DIR) / "funding" / "NSF",
        "raw_nih":    Path(RAW_DIR) / "funding" / "NIH",
        "raw_usa":    Path(RAW_DIR) / "funding" / "USASpending",
        "raw_bea":    Path(RAW_DIR) / "funding" / "BEA",
        "raw_census": Path(RAW_DIR) / "funding" / "Census",
        "proc_eip":   Path(PROC_DIR) / "eip",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        status = "exists" if path.exists() else "created"
        print(f"  [dir] {name:12s}: {path}  ({status})")
        logger.info("[dir] %s: %s (%s)", name, path, status)
    return dirs


def setup_logging(log_dir: Path) -> None:
    """
    Configure the module logger to write to both the console and a timestamped
    log file in log_dir (PROC_DIR/eip/).  Captures all key fetch stats and
    errors so every run is fully reproducible from the log alone.

    Log file: import_funding_{YYYYMMDD_HHMMSS}.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"import_funding_{ts}.log"

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)   # only warnings+ to console (prints handle INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log started: %s", log_path)


STATES_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13",
    "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45",
    "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
    "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
    "DC": "11",
}

FIPS_TO_STATE = {v: k for k, v in STATES_FIPS.items()}


# ---------------------------------------------------------------------------
# Generic HTTP helper
# ---------------------------------------------------------------------------

def get_json(
    url: str,
    params: Optional[dict] = None,
    retries: int = 4,
    timeout: int = 60,
) -> Optional[dict | list]:
    headers = {"Accept": "application/json", "User-Agent": "firmscope-research-bot/1.0"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  [rate limit] sleeping {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  [warn] {url}: {exc}. Retry in {wait}s.")
            time.sleep(wait)
    return None


def post_json(
    url: str,
    payload: dict,
    retries: int = 4,
    timeout: int = 60,
) -> Optional[dict]:
    headers = {"Content-Type": "application/json", "User-Agent": "firmscope-research-bot/1.0"}
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            body = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.json()
                except Exception:
                    body = exc.response.text[:500]
            print(f"  [warn] POST {url}: {exc}. Body: {body}. Retry in {wait}s.")
            if attempt == 0 and body:
                logger.error("POST %s body: %s  payload: %s", url, body, payload)
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# 1. USASpending.gov  (federal grants by state and fiscal year)
# ---------------------------------------------------------------------------
USASPENDING_BASE = "https://api.usaspending.gov/api/v2"


USA_SPENDING_MIN_YEAR = 2008   # API earliest: start_date 2007-10-01 = FY2008


def fetch_usaspending_by_year(
    year: int, raw_dir: Path, force: bool = False
) -> pd.DataFrame:
    """
    Fetch all USASpending financial assistance grants nationally for *year*,
    post-filter to DOE (81.*) and DARPA (12.*) only — excluding NSF (47.*)
    and NIH (93.*) already captured in Steps 1-2. State is extracted from the
    'recipient_location_state_code' response field. One paginated call stream
    per year instead of 51 per-state calls. Coverage: FY2008 onwards.
    """
    out_path = raw_dir / f"usaspending_{year}.csv"
    if out_path.exists() and not force:
        print(f"  USASpending {year}: cached ({out_path.name})")
        return pd.read_csv(out_path)

    if year < USA_SPENDING_MIN_YEAR:
        return pd.DataFrame()

    KEEP_PREFIXES = ("81.", "12.")
    EXCLUDE_PREFIXES = ("47.", "93.")

    all_rows: list[dict] = []
    seen_ids: set[str] = set()
    payload: dict = {
        "filters": {
            "time_period": [{"start_date": f"{year-1}-10-01", "end_date": f"{year}-09-30"}],
            "award_type_codes": ["02", "03", "04", "05", "F001", "F002"],
        },
        "fields": [
            "Award ID", "Recipient Name", "Award Amount",
            "CFDA Number", "CFDA Title", "Award Type",
            "Start Date", "recipient_location_state_code",
        ],
        "limit": 100,
        "page": 1,
        "sort": "Award Amount",
        "order": "desc",
    }
    while True:
        data = post_json(f"{USASPENDING_BASE}/search/spending_by_award/", payload)
        time.sleep(SLEEP)
        if data is None:
            break
        results = data.get("results", [])
        if not results:
            break
        for row in results:
            cfda = (row.get("CFDA Number") or "")
            if not any(cfda.startswith(p) for p in KEEP_PREFIXES):
                continue
            if any(cfda.startswith(p) for p in EXCLUDE_PREFIXES):
                continue
            aid = row.get("Award ID") or ""
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            all_rows.append({
                "state":       row.get("recipient_location_state_code", ""),
                "year":        year,
                "award_id":    aid,
                "recipient":   row.get("Recipient Name"),
                "amount_usd":  row.get("Award Amount"),
                "cfda_number": cfda,
                "cfda_title":  row.get("CFDA Title"),
                "award_type":  row.get("Award Type"),
                "start_date":  row.get("Start Date"),
                "rd_tier":     "direct",
            })
        pagination = data.get("page_metadata", {})
        if payload["page"] >= pagination.get("last_page", 1):
            break
        payload["page"] += 1

    df = pd.DataFrame(all_rows)
    n_states = df["state"].nunique() if not df.empty else 0
    print(f"  USASpending {year}: {len(df):,} awards | {n_states} states")
    logger.info("USASpending %d: %d awards | %d states -> %s",
                year, len(df), n_states, out_path)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. NSF Award Search API
# ---------------------------------------------------------------------------
NSF_API = "https://api.nsf.gov/services/v1/awards.json"


def _fetch_nsf_state(state: str, year: int) -> list[dict]:
    """Fetch NSF awards for one state-year. Called in parallel across states."""
    rows: list[dict] = []
    offset = 0
    rpp = 100
    params_base = {
        "awardeeStateCode": state,
        "dateStart": f"01/01/{year}",
        "dateEnd": f"12/31/{year}",
        "printFields": "id,title,awardeeName,awardeeStateCode,fundsObligatedAmt,startDate,cfdaNumber",
        "rpp": rpp,
    }
    while True:
        data = get_json(NSF_API, {**params_base, "offset": offset})
        time.sleep(SLEEP)
        if data is None:
            break
        awards = (data.get("response", {}) or {}).get("award", [])
        if not awards:
            break
        for award in awards:
            cfda = award.get("cfdaNumber", "")
            title = award.get("title", "")
            rows.append({
                "nsf_id":    award.get("id"),
                "state":     state,
                "year":      year,
                "title":     title,
                "awardee":   award.get("awardeeName", ""),
                "cfda":      cfda,
                "domain":    classify_domain(title, cfda),
                "amount_usd": pd.to_numeric(award.get("fundsObligatedAmt", 0), errors="coerce"),
                "start_date": award.get("startDate", ""),
            })
        if len(awards) < rpp:
            break
        offset += rpp
        if offset >= 10000:
            break
    return rows


def fetch_nsf_by_year(
    year: int, raw_dir: Path, workers: int = 4, force: bool = False
) -> pd.DataFrame:
    """
    Fetch NSF awards for *year* by querying each state separately in parallel.
    Saves raw award-level CSV to raw_dir (RAW_DIR/funding/NSF/).
    Skips fetch if output CSV already exists unless force=True.
    """
    out_path = raw_dir / f"nsf_{year}.csv"
    if out_path.exists() and not force:
        print(f"  NSF {year}: cached ({out_path.name})")
        return pd.read_csv(out_path)

    print(f"  NSF {year}: fetching ({workers} workers)...", end="", flush=True)
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_nsf_state, s, year): s for s in STATES_FIPS}
        for future in as_completed(futures):
            all_rows.extend(future.result())

    df = pd.DataFrame(all_rows)
    n_states = df["state"].nunique() if not df.empty else 0
    total_usd = df["amount_usd"].sum() if not df.empty else 0
    print(f" {len(df):,} awards | {n_states} states | ${total_usd/1e9:.2f}B")
    logger.info("NSF %d: %d awards | %d states | $%.2fB -> %s",
                year, len(df), n_states, total_usd / 1e9, out_path)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. NIH RePORTER API
# ---------------------------------------------------------------------------
NIH_REPORTER = "https://api.reporter.nih.gov/v2/projects/search"


def _fetch_nih_state(state: str, year: int) -> list[dict]:
    """
    Fetch NIH projects for one state-year using org_states filter.
    Most states have <500 projects/year so this completes in one page.
    activity_code (R01, R21, P01, U01, etc.) is pulled for grant-type tagging.
    """
    rows: list[dict] = []
    offset = 0
    limit = 500
    while True:
        payload = {
            "criteria": {"fiscal_years": [year], "org_states": [state]},
            "offset": offset,
            "limit": limit,
        }
        data = post_json(NIH_REPORTER, payload)
        time.sleep(SLEEP)
        if data is None:
            break
        results = data.get("results", [])
        if not results:
            break
        for proj in results:
            title = (proj.get("project_title") or proj.get("ProjectTitle") or "")
            rows.append({
                "nih_id":       proj.get("project_num") or proj.get("ProjectNum"),
                "state":        state,
                "year":         year,
                "activity_code": proj.get("activity_code", ""),
                "title":        title[:200],
                "domain":       classify_domain(title),
                "amount_usd":   pd.to_numeric(
                    proj.get("award_amount") or proj.get("AwardAmount") or 0,
                    errors="coerce",
                ),
            })
        if len(results) < limit:
            break
        offset += limit
        if offset > 10000:
            break
    return rows


def fetch_nih_by_year(
    year: int, raw_dir: Path, workers: int = 4, force: bool = False
) -> pd.DataFrame:
    """
    Fetch NIH grants for *year* by querying each state separately in parallel
    via the org_states criterion. Saves raw project-level CSV to raw_dir
    (RAW_DIR/funding/NIH/). NIH RePORTER history: FY1985 onwards.
    Skips fetch if output CSV already exists unless force=True.
    """
    out_path = raw_dir / f"nih_{year}.csv"
    if out_path.exists() and not force:
        print(f"  NIH {year}: cached ({out_path.name})")
        return pd.read_csv(out_path)

    print(f"  NIH {year}: fetching ({workers} workers)...", end="", flush=True)
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_nih_state, s, year): s for s in STATES_FIPS}
        for future in as_completed(futures):
            all_rows.extend(future.result())

    df = pd.DataFrame(all_rows)
    n_states = df["state"].nunique() if not df.empty else 0
    total_usd = df["amount_usd"].sum() if not df.empty else 0
    print(f" {len(df):,} projects | {n_states} states | ${total_usd/1e9:.2f}B")
    logger.info("NIH  %d: %d projects | %d states | $%.2fB -> %s",
                year, len(df), n_states, total_usd / 1e9, out_path)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 4. BEA API – Gross State Product
# ---------------------------------------------------------------------------
BEA_API = "https://apps.bea.gov/api/data"


def fetch_gsp(years: list[int], bea_key: str, raw_dir: Path) -> pd.DataFrame:
    """
    Fetch real Gross State Product (chained dollars) from BEA for all states.
    Table: SAGDP2N, Line 1 = All industry total. Saves to RAW_DIR/funding/BEA/.
    """
    year_str = ",".join(str(y) for y in years)
    params = {
        "UserID": bea_key,
        "method": "GetData",
        "DataSetName": "Regional",
        "TableName": "SAGDP2N",
        "LineCode": "1",
        "GeoFips": "STATE",
        "Year": year_str,
        "ResultFormat": "JSON",
    }
    print("  BEA: fetching GSP...", end="", flush=True)
    data = get_json(BEA_API, params)
    if data is None:
        print(" FAILED")
        return pd.DataFrame()

    try:
        results = data["BEAAPI"]["Results"]["Data"]
    except (KeyError, TypeError):
        print(f" Error parsing BEA response: {data}")
        return pd.DataFrame()

    rows = []
    for r in results:
        state_fips = r.get("GeoFips", "")
        state_abbr = FIPS_TO_STATE.get(state_fips.zfill(2))
        if state_abbr is None:
            continue
        try:
            value = float(r["DataValue"].replace(",", ""))
        except (ValueError, AttributeError):
            continue
        rows.append({
            "state": state_abbr,
            "year": int(r["TimePeriod"]),
            "gsp_millions": value,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        out_path = raw_dir / "gsp_panel.csv"
        df.to_csv(out_path, index=False)
        print(f" {len(df)} rows | years {df['year'].min()}-{df['year'].max()}")
    return df


# ---------------------------------------------------------------------------
# 5. Census API – State Population
# ---------------------------------------------------------------------------
CENSUS_POP_API = "https://api.census.gov/data"


def fetch_population(years: list[int], census_key: str, raw_dir: Path) -> pd.DataFrame:
    """
    Fetch state population estimates from Census Bureau (PEP series).
    Saves to RAW_DIR/funding/Census/population_panel.csv.
    """
    rows: list[dict] = []
    print("  Census: fetching population...", end="", flush=True)

    for year in years:
        if year >= 2020:
            url = f"{CENSUS_POP_API}/{year}/pep/population"
            params = {
                "get": "NAME,POP_2021" if year >= 2021 else "NAME,POP",
                "for": "state:*",
                "key": census_key,
            }
        elif year >= 2010:
            url = f"{CENSUS_POP_API}/{year}/pep/population"
            params = {"get": "NAME,POP", "for": "state:*", "key": census_key}
        else:
            url = f"{CENSUS_POP_API}/2000/pep/int_population"
            params = {"get": "POP,DATE_DESC,NAME", "for": "state:*", "key": census_key}

        data = get_json(url, params)
        time.sleep(SLEEP)
        if data is None or not isinstance(data, list) or len(data) < 2:
            continue
        header = data[0]
        for row in data[1:]:
            row_dict = dict(zip(header, row))
            fips = row_dict.get("state", "")
            state = FIPS_TO_STATE.get(fips)
            if state is None:
                continue
            pop_val = row_dict.get("POP") or row_dict.get("POP_2021")
            if pop_val is None:
                continue
            try:
                rows.append({"state": state, "year": year, "population": int(pop_val)})
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        out_path = raw_dir / "population_panel.csv"
        df.to_csv(out_path, index=False)
        print(f" {len(df)} rows | years {df['year'].min()}-{df['year'].max()}")
    return df


# ---------------------------------------------------------------------------
# Panel assembler  (saves intermediary output to PROC_DIR/eip/)
# ---------------------------------------------------------------------------

def build_funding_panel(
    usaspending_dfs: list[pd.DataFrame],
    nsf_dfs: list[pd.DataFrame],
    nih_dfs: list[pd.DataFrame],
    gsp_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    proc_dir: Path,
) -> pd.DataFrame:
    """
    Aggregate all funding streams into a (state, year) panel with:
      - Total grants per source (NSF, NIH, USASpending)
      - Per-domain breakdowns following HJT taxonomy (nsf_{domain}_usd,
        nih_{domain}_usd) for each domain in INNOVATION_DOMAINS
      - GSP and per-capita normalizations (when available)
    Saves final panel to proc_dir (PROC_DIR/eip/).
    """
    print("\n  [1/4] Aggregating totals by source...")

    def agg_total(dfs: list[pd.DataFrame], col: str) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame(columns=["state", "year", col])
        combined = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
        combined["amount_usd"] = pd.to_numeric(combined["amount_usd"], errors="coerce")
        return (
            combined.groupby(["state", "year"])["amount_usd"]
            .sum().reset_index().rename(columns={"amount_usd": col})
        )

    def agg_by_domain(dfs: list[pd.DataFrame], prefix: str) -> pd.DataFrame:
        """Pivot amount_usd by domain into wide columns: {prefix}_{domain}_usd."""
        if not dfs:
            return pd.DataFrame(columns=["state", "year"])
        combined = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
        if "domain" not in combined.columns:
            return pd.DataFrame(columns=["state", "year"])
        combined["amount_usd"] = pd.to_numeric(combined["amount_usd"], errors="coerce")
        pivoted = (
            combined.groupby(["state", "year", "domain"])["amount_usd"]
            .sum().unstack(fill_value=0).reset_index()
        )
        pivoted.columns = [
            f"{prefix}_{c}_usd" if c not in ("state", "year") else c
            for c in pivoted.columns
        ]
        # Ensure all domains present even if zero for this source
        for domain in ALL_DOMAINS:
            col = f"{prefix}_{domain}_usd"
            if col not in pivoted.columns:
                pivoted[col] = 0.0
        return pivoted

    usa_total = agg_total(usaspending_dfs, "usaspending_rd_grants_usd")
    nsf_total = agg_total(nsf_dfs, "nsf_grants_usd")
    nih_total = agg_total(nih_dfs, "nih_grants_usd")

    print("  [2/4] Building per-domain breakdowns (HJT taxonomy)...")
    nsf_domain = agg_by_domain(nsf_dfs, "nsf")
    nih_domain = agg_by_domain(nih_dfs, "nih")

    # Report domain distribution for a quick sanity check
    for label, dfs in [("NSF", nsf_dfs), ("NIH", nih_dfs)]:
        if dfs:
            all_data = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
            if "domain" in all_data.columns:
                dist = all_data.groupby("domain")["amount_usd"].sum().sort_values(ascending=False)
                print(f"    {label} domain distribution (all years, $B):")
                for dom, val in dist.items():
                    print(f"      {dom:20s}: ${val/1e9:.2f}B")

    print("  [3/4] Merging all sources into panel...")
    panel = usa_total
    for df in [nsf_total, nih_total, nsf_domain, nih_domain]:
        panel = panel.merge(df, on=["state", "year"], how="outer")
    if not gsp_df.empty:
        panel = panel.merge(gsp_df, on=["state", "year"], how="left")
    if not pop_df.empty:
        panel = panel.merge(pop_df, on=["state", "year"], how="left")

    # Coerce and fill funding columns
    usd_cols = [c for c in panel.columns if c.endswith("_usd")]
    for col in usd_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)

    # Total across sources
    source_total_cols = [c for c in usd_cols if c in
                         ("usaspending_rd_grants_usd", "nsf_grants_usd", "nih_grants_usd")]
    panel["total_rd_funding_usd"] = panel[source_total_cols].sum(axis=1)

    # Normalizations
    if "gsp_millions" in panel.columns:
        panel["rd_funding_pct_gsp"] = (
            panel["total_rd_funding_usd"] / (panel["gsp_millions"] * 1e6)
        ).replace([float("inf"), -float("inf")], None)
    if "population" in panel.columns:
        panel["rd_funding_per_capita"] = (
            panel["total_rd_funding_usd"] / panel["population"]
        ).replace([float("inf"), -float("inf")], None)

    panel = panel.sort_values(["state", "year"]).reset_index(drop=True)

    print("  [4/4] Saving to PROC_DIR/eip/funding_panel.csv...")
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / "funding_panel.csv"
    panel.to_csv(out_path, index=False)

    # Summary stats
    print(f"\n  Panel shape  : {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"  States       : {panel['state'].nunique()}")
    print(f"  Years        : {int(panel['year'].min())} - {int(panel['year'].max())}")
    print(f"  Columns      : {list(panel.columns)}")
    logger.info("Panel: %d rows x %d cols | %d states | years %d-%d",
                panel.shape[0], panel.shape[1], panel["state"].nunique(),
                int(panel["year"].min()), int(panel["year"].max()))
    logger.info("Panel columns: %s", list(panel.columns))
    if "total_rd_funding_usd" in panel.columns:
        total_t = panel["total_rd_funding_usd"].sum() / 1e12
        print(f"  Total funding: ${total_t:.3f}T across all state-years")
        top5 = (
            panel.groupby("state")["total_rd_funding_usd"].sum()
            .sort_values(ascending=False).head(5)
        )
        print(f"  Top 5 states (cumulative): {top5.to_dict()}")
        logger.info("Total funding: $%.3fT | top 5 states: %s", total_t, top5.to_dict())
    print(f"  Saved to     : {out_path}")
    logger.info("Panel saved to: %s", out_path)

    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pull innovation funding data for EIP.")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end",   type=int, default=2025)
    parser.add_argument(
        "--bea_key", type=str, default=os.environ.get("BEA_API_KEY", ""),
        help="BEA API key (free: https://apps.bea.gov/API/signup/ or set BEA_API_KEY)",
    )
    parser.add_argument(
        "--census_key", type=str, default=os.environ.get("CENSUS_API_KEY", ""),
        help="Census API key (free: https://api.census.gov/data/key_signup.html or set CENSUS_API_KEY)",
    )
    parser.add_argument("--states", nargs="*", default=list(STATES_FIPS.keys()))
    parser.add_argument(
        "--usaspending", action="store_true",
        help="Enable USASpending fetching (disabled by default; slow: ~51 states x N years)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel threads for per-state NSF/NIH fetching (default: 4)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if per-year CSV cache already exists",
    )
    parser.add_argument(
        "--panel_only", action="store_true",
        help="Skip Steps 1-3 (fetch); load existing raw CSVs and run Steps 4-6 only",
    )
    args = parser.parse_args()
    years = list(range(args.start, args.end + 1))

    # ------------------------------------------------------------------ dirs
    print("=" * 60)
    print("=== EIP Component 3: Innovation Funding Intensity ===")
    print("=" * 60)
    print(f"\n[Step 0] Setting up directories ({args.start}-{args.end}, "
          f"{len(years)} years, {len(args.states)} states)...")
    dirs = setup_dirs()
    setup_logging(dirs["proc_eip"])
    logger.info("Run started: years=%d-%d  states=%d  workers=%d  "
                "usaspending=%s  force=%s  panel_only=%s",
                args.start, args.end, len(args.states), args.workers,
                args.usaspending, args.force, args.panel_only)

    # ------------------------------------------------------------------ NSF
    nsf_dfs: list[pd.DataFrame] = []
    if args.panel_only:
        print("\n[Step 1] NSF grants: loading from cache...")
        for p in sorted(dirs["raw_nsf"].glob("nsf_*.csv")):
            df = pd.read_csv(p)
            if not df.empty:
                nsf_dfs.append(df)
        print(f"  Loaded {len(nsf_dfs)} year files from {dirs['raw_nsf']}")
    else:
        print(f"\n[Step 1] NSF grants  ({len(years)} years x {len(STATES_FIPS)} states, "
              f"{args.workers} workers)")
        for year in tqdm(years, desc="  NSF"):
            df = fetch_nsf_by_year(year, dirs["raw_nsf"], workers=args.workers, force=args.force)
            if not df.empty:
                nsf_dfs.append(df)
    if nsf_dfs:
        nsf_all = pd.concat(nsf_dfs, ignore_index=True)
        print(f"  NSF total: {len(nsf_all):,} awards | "
              f"${nsf_all['amount_usd'].sum()/1e9:.1f}B | "
              f"domains: {nsf_all['domain'].value_counts().to_dict()}")

    # ------------------------------------------------------------------ NIH
    nih_dfs: list[pd.DataFrame] = []
    if args.panel_only:
        print("\n[Step 2] NIH grants: loading from cache...")
        for p in sorted(dirs["raw_nih"].glob("nih_*.csv")):
            df = pd.read_csv(p)
            if not df.empty:
                nih_dfs.append(df)
        print(f"  Loaded {len(nih_dfs)} year files from {dirs['raw_nih']}")
    else:
        print(f"\n[Step 2] NIH grants  ({len(years)} years x {len(STATES_FIPS)} states, "
              f"{args.workers} workers)")
        for year in tqdm(years, desc="  NIH"):
            df = fetch_nih_by_year(year, dirs["raw_nih"], workers=args.workers, force=args.force)
            if not df.empty:
                nih_dfs.append(df)
    if nih_dfs:
        nih_all = pd.concat(nih_dfs, ignore_index=True)
        print(f"  NIH total: {len(nih_all):,} projects | "
              f"${nih_all['amount_usd'].sum()/1e9:.1f}B | "
              f"domains: {nih_all['domain'].value_counts().to_dict()}")

    # ---------------------------------------------------------- USASpending
    usaspending_dfs: list[pd.DataFrame] = []
    if args.panel_only and not args.usaspending:
        print("\n[Step 3] USASpending: skipped (--panel_only)")
    elif args.usaspending:
        usa_years = [y for y in years if y >= USA_SPENDING_MIN_YEAR]
        print(f"\n[Step 3] USASpending  ({len(usa_years)} years, "
              f"FY{USA_SPENDING_MIN_YEAR}-{years[-1]}; national fetch, DOE+DARPA only)")
        for year in tqdm(usa_years, desc="  USASpending"):
            df = fetch_usaspending_by_year(year, dirs["raw_usa"], force=args.force)
            if not df.empty:
                usaspending_dfs.append(df)
        if usaspending_dfs:
            usa_all = pd.concat(usaspending_dfs, ignore_index=True)
            print(f"  USASpending total: {len(usa_all):,} awards | "
                  f"${usa_all['amount_usd'].sum()/1e9:.1f}B")
    else:
        print("\n[Step 3] USASpending: skipped (pass --usaspending to enable)")

    # --------------------------------------------------------------- BEA GSP
    gsp_df = pd.DataFrame()
    if args.bea_key:
        print("\n[Step 4] BEA Gross State Product")
        gsp_df = fetch_gsp(years, args.bea_key, dirs["raw_bea"])
    else:
        print("\n[Step 4] BEA: skipped (no key — set BEA_API_KEY or pass --bea_key)")

    # ------------------------------------------------------------ Population
    pop_df = pd.DataFrame()
    if args.census_key:
        print("\n[Step 5] Census population")
        pop_df = fetch_population(years, args.census_key, dirs["raw_census"])
    else:
        print("\n[Step 5] Census: skipped (no key — set CENSUS_API_KEY or pass --census_key)")

    # --------------------------------------------------------------- Panel
    print("\n[Step 6] Building processed panel -> PROC_DIR/eip/funding_panel.csv")
    panel = build_funding_panel(
        usaspending_dfs, nsf_dfs, nih_dfs, gsp_df, pop_df, dirs["proc_eip"]
    )
    print("\n  Preview (first 5 rows, key columns):")
    preview_cols = [c for c in ["state", "year", "nsf_grants_usd", "nih_grants_usd",
                                "total_rd_funding_usd", "rd_funding_per_capita",
                                "rd_funding_pct_gsp"] if c in panel.columns]
    print(panel[preview_cols].head().to_string(index=False))
    print("\nDone.")
    logger.info("Run complete.")


if __name__ == "__main__":
    main()

