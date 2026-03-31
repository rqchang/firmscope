"""
import_legislative.py
---------------------
Pulls state-level legislative activity related to innovation for every U.S.
state and year, using the OpenStates API (free tier, registration required).

OpenStates (https://openstates.org/api/) tracks all 50 state legislatures plus
DC and Puerto Rico. The v3 API returns bill metadata including title, summary,
subjects, and sponsors. We search for bills whose title or subject tags match
an innovation keyword list, then classify them by innovation domain.

Output
------
  data_import/raw/legislative/
    {state_abbr}_{year}_bills.csv    – one row per bill with metadata
    legislative_panel.csv            – (state, year, n_bills, n_domains, ...) summary

Usage
-----
  python data_import/import_legislative.py --start 2007 --end 2025
  python import_legislative.py --api_key YOUR_KEY --states CA TX --start 2020
  python data_import/import_legislative.py --start 2010 --end 2025 --states AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO
  python data_import/import_legislative.py --start 2010 --end 2025 --states MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY

Get a free API key at: https://openstates.org/accounts/profile/

Requirements
------------
  pip install requests pandas tqdm
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.set_paths import RAW_DIR

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

OPENSTATES_BASE = "https://v3.openstates.org"
DEFAULT_PAGE_SIZE = 20          # max allowed by free tier
SLEEP_BETWEEN_PAGES = 7.0       # free tier ~10 req/min; 7s keeps us safe


""" Innovation keyword groups """

# for search queries and domain tagging)
INNOVATION_DOMAINS: dict[str, list[str]] = {
    "rd_tax_credit":     ["research tax credit", "r&d tax credit", "research and development tax",
                          "qualified research expense", "research expenditure credit",
                          "innovation tax credit", "research investment credit",
                          "incremental research", "research activities credit",
                          "technology tax incentive", "r&d incentive", "research deduction",
                          "qualified research activity", "research credit"],
    "general_rd":        ["research and development", "r&d", "research funding",
                          "innovation fund", "technology transfer", "tech transfer",
                          "technology commercialization", "intellectual property",
                          "university research", "research university",
                          "national laboratory", "federal research",
                          "applied research", "basic research",
                          "patent box", "inventor credit", "trade secret",
                          "technology licensing", "research park",
                          "cooperative research", "research consortium"],
    "clean_energy":      ["clean energy", "renewable energy", "solar", "wind energy",
                          "electric vehicle", "battery storage", "carbon capture",
                          "green hydrogen", "offshore wind", "energy efficiency",
                          "nuclear energy", "hydrogen fuel", "grid modernization",
                          "smart grid", "energy storage", "ev charging",
                          "zero emission", "net zero", "decarbonization",
                          "climate technology", "clean tech"],
    "semiconductor":     ["semiconductor", "chips act", "microchip", "advanced manufacturing",
                          "fab", "foundry", "integrated circuit", "wafer",
                          "chip manufacturing", "printed circuit", "photonics",
                          "nanotechnology", "materials science",
                          "advanced materials", "composites", "graphene",
                          "rare earth", "critical minerals"],
    "ai_ml":             ["artificial intelligence", "machine learning", "deep learning",
                          "autonomous vehicle", "robotics", "data center",
                          "large language model", "generative ai", "computer vision",
                          "natural language processing", "automation", "drone",
                          "unmanned system", "predictive analytics", "algorithm"],
    "biotech":           ["biotechnology", "biomanufacturing", "genomics", "life sciences",
                          "pharmaceutical", "medical device", "health technology",
                          "precision medicine", "crispr", "gene editing",
                          "cell therapy", "clinical trial", "telemedicine",
                          "digital health", "health informatics", "drug discovery",
                          "bioinformatics", "synthetic biology",
                          "medical research", "biomedical research",
                          "translational research", "biologics", "medical innovation"],
    "quantum":           ["quantum computing", "quantum information", "quantum technology",
                          "quantum sensor", "quantum communication",
                          "quantum encryption", "quantum network"],
    "broadband":         ["broadband", "5g", "fiber optic", "digital infrastructure",
                          "internet access", "6g", "rural broadband",
                          "digital divide", "wireless network", "spectrum",
                          "smart city", "internet of things", "connected device"],
    "cybersecurity":     ["cybersecurity", "cyber security", "information security",
                          "data privacy", "data security", "network security",
                          "critical infrastructure protection", "zero trust",
                          "identity verification"],
    "workforce":         ["stem education", "workforce development", "coding", "computer science",
                          "vocational training", "apprenticeship", "tech talent",
                          "reskilling", "upskilling", "digital literacy",
                          "coding bootcamp", "engineering education",
                          "mathematics education", "technical training",
                          "community college technology"],
    "venture_startup":   ["venture capital", "startup", "entrepreneur", "accelerator",
                          "incubator", "small business innovation", "angel investor",
                          "seed funding", "innovation hub", "tech district",
                          "early stage company", "new business formation",
                          "spinoff", "technology company"],
    "advanced_mfg":      ["additive manufacturing", "3d printing", "robotics manufacturing",
                          "precision manufacturing", "industrial automation",
                          "industrial internet", "smart factory", "industry 4.0",
                          "digital manufacturing", "supply chain technology"],
    "space":             ["space technology", "aerospace", "satellite",
                          "space launch", "commercial space", "earth observation",
                          "launch vehicle", "space economy"],
    "climate_tech":      ["climate change", "carbon neutral", "sustainability",
                          "green economy", "circular economy", "clean water technology",
                          "sustainable agriculture", "environmental technology",
                          "emissions reduction"],
    "fintech":           ["financial technology", "fintech", "digital payment",
                          "cryptocurrency", "blockchain", "digital currency",
                          "open banking", "regtech", "insurtech",
                          "digital asset", "payment innovation"],
    "defense_tech":      ["defense technology", "dual use technology", "defense research",
                          "national security technology", "military technology",
                          "defense innovation", "defense advanced research",
                          "defense manufacturing", "defense industrial base"],
}

# Flat list for broad API queries
ALL_KEYWORDS = [kw for kws in INNOVATION_DOMAINS.values() for kw in kws]

# OpenStates subject tags that imply innovation relevance
INNOVATION_SUBJECTS = {
    "Science, Technology, Communications",
    "Energy",
    "Education",
    "Environment",
    "Business and Economic Development",
    "Labor and Employment",
    "Health",
    "Agriculture and Food",
}

STATE_ABBR_TO_OPENSTATES: dict[str, str] = {
    "AL": "al", "AK": "ak", "AZ": "az", "AR": "ar", "CA": "ca",
    "CO": "co", "CT": "ct", "DE": "de", "FL": "fl", "GA": "ga",
    "HI": "hi", "ID": "id", "IL": "il", "IN": "in", "IA": "ia",
    "KS": "ks", "KY": "ky", "LA": "la", "ME": "me", "MD": "md",
    "MA": "ma", "MI": "mi", "MN": "mn", "MS": "ms", "MO": "mo",
    "MT": "mt", "NE": "ne", "NV": "nv", "NH": "nh", "NJ": "nj",
    "NM": "nm", "NY": "ny", "NC": "nc", "ND": "nd", "OH": "oh",
    "OK": "ok", "OR": "or", "PA": "pa", "RI": "ri", "SC": "sc",
    "SD": "sd", "TN": "tn", "TX": "tx", "UT": "ut", "VT": "vt",
    "VA": "va", "WA": "wa", "WV": "wv", "WI": "wi", "WY": "wy",
    "DC": "dc",
}



""" Helpers """

def tag_domains(text: str) -> list[str]:
    """Return innovation domain labels that match text."""
    t = text.lower()
    matched = [domain for domain, kws in INNOVATION_DOMAINS.items() if any(kw in t for kw in kws)]
    return matched if matched else ["general_innovation"]


def openstates_get(
    endpoint: str,
    params: dict,
    api_key: str,
    retries: int = 4,
) -> Optional[dict]:
    headers = {"X-API-KEY": api_key}
    url = f"{OPENSTATES_BASE}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                logging.getLogger(__name__).warning(f"[rate limit] sleeping {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            logging.getLogger(__name__).warning(f"[warn] {url} failed: {exc}. Retrying in {wait}s.")
            time.sleep(wait)
    return None


""" Fetchers """

# Cache sessions per state so the jurisdictions endpoint is only called once
_session_cache: dict[str, list[dict]] = {}


def fetch_sessions_for_state(state: str, api_key: str) -> list[dict]:
    """Return all legislative sessions for a state (cached)."""
    if state in _session_cache:
        return _session_cache[state]
    os_state = STATE_ABBR_TO_OPENSTATES.get(state)
    if not os_state:
        _session_cache[state] = []
        return []
    data = openstates_get(
        f"/jurisdictions/ocd-jurisdiction/country:us/state:{os_state}/government",
        {"include": "legislative_sessions"},
        api_key,
    )
    sessions = data.get("legislative_sessions", []) if data else []
    # Only cache successful non-empty results — don't poison the cache on API failure
    if sessions:
        _session_cache[state] = sessions
    return sessions


def year_to_session_id(sessions: list[dict], year: int) -> Optional[str]:
    """Return the session identifier whose date range contains *year*."""
    for s in sessions:
        start = (s.get("start_date") or "")[:4]
        end   = (s.get("end_date")   or start)[:4]
        try:
            if int(start) <= year <= int(end):
                return s["identifier"]
        except (ValueError, TypeError, KeyError):
            continue
    # fallback: identifier string contains the year
    for s in sessions:
        if str(year) in s.get("identifier", ""):
            return s["identifier"]
    return None


def fetch_bills_for_state_year(
    state: str,
    year: int,
    api_key: str,
) -> list[dict]:
    """
    Fetch all innovation-relevant bills for *state* in *year* from OpenStates.
    Returns a list of normalized bill dicts.
    """
    os_state = STATE_ABBR_TO_OPENSTATES.get(state)
    if os_state is None:
        return []

    # Resolve the correct session identifier for this year
    sessions   = fetch_sessions_for_state(state, api_key)
    session_id = year_to_session_id(sessions, year)
    if session_id is None:
        logging.getLogger(__name__).info(f"[skip] {state} {year}: no matching session found")
        return []

    params_base = {
        "jurisdiction": os_state,
        "session":      session_id,
        "per_page":     DEFAULT_PAGE_SIZE,
    }

    search_queries = [
        # --- Frascati R&D activity types ---
        "research", "basic research", "applied research", "experimental development",
        "research and development",
        # --- Core R&D signals (Bellstam et al. 2021) ---
        "innovation", "discovery", "invention", "novel", "breakthrough", "frontier",
        # --- Frascati project-stage terms ---
        "proof of concept", "prototype", "feasibility", "demonstration",
        "commercialization", "technology transfer", "tech transfer",
        "phase i", "phase ii",
        # --- SBIR / STTR ---
        "SBIR", "STTR", "small business innovation",
        # --- General R&D infrastructure ---
        "technology", "science", "engineering", "laboratory", "university",
        "patent", "intellectual property", "research park",
        # --- Fiscal / funding mechanisms ---
        "appropriation", "grant", "fund", "tax credit", "tax incentive",
        # --- HJT Computers & Communications ---
        "software", "information technology", "computer science",
        "artificial intelligence", "machine learning", "algorithm",
        "autonomous", "robotics", "automation", "data center", "cybersecurity",
        "broadband", "5g", "fiber optic", "internet of things",
        "cloud computing", "drone", "unmanned",
        # --- HJT Electrical & Electronic ---
        "semiconductor", "microelectronics", "integrated circuit",
        "quantum", "photonics", "nanotechnology", "advanced materials",
        "superconductor", "composite", "rare earth",
        # --- HJT Drugs & Medical ---
        "biotechnology", "genomics", "life science", "precision medicine",
        "biomedical", "clinical research", "translational research",
        "drug discovery", "bioinformatics", "synthetic biology",
        "gene", "pharmaceutical", "medical device", "biomanufacturing",
        "telemedicine", "health technology",
        # --- HJT Chemical ---
        "chemistry", "materials science",
        # --- HJT Mechanical / Environmental & Energy ---
        "advanced manufacturing", "additive manufacturing", "3d printing",
        "clean energy", "renewable energy", "renewable",
        "electric vehicle", "battery storage", "hydrogen", "nuclear",
        "grid modernization", "carbon capture", "energy efficiency",
        "solar", "wind", "decarbonization", "carbon", "climate", "sustainability",
        # --- HJT Aerospace / Space ---
        "aerospace", "space", "satellite", "launch vehicle",
        # --- NSF directorate areas ---
        "mathematics", "physics", "biology", "geosciences",
        "social science", "behavioral science",
        # --- DOD R&D budget activities ---
        "advanced technology development", "technology maturation",
        "prototype development",
        # --- Policy programs ---
        "chips", "industrial policy",
        # --- Startup / commercialization ecosystem ---
        "startup", "entrepreneur", "venture capital", "incubator", "accelerator",
        "spinoff",
        # --- Workforce / STEM ---
        "stem", "workforce development", "apprenticeship", "vocational", "coding",
        # --- Digital infrastructure ---
        "fiber", "spectrum", "smart city", "digital infrastructure",
        # --- Ag-tech ---
        "precision agriculture", "agriculture technology",
        # --- Fintech ---
        "fintech", "blockchain", "cryptocurrency",
        # --- Defense ---
        "defense technology",
    ]

    all_bills: dict[str, dict] = {}   # bill_id → bill dict, deduplicated

    for query in search_queries:
        page = 1
        while True:
            params = {**params_base, "q": query, "page": page}
            data = openstates_get("/bills", params, api_key)
            time.sleep(SLEEP_BETWEEN_PAGES)
            if data is None:
                break

            results = data.get("results", [])
            if not results:
                break

            for bill in results:
                bill_id = bill.get("id", "")
                if bill_id in all_bills:
                    continue

                title = bill.get("title", "") or ""
                # abstracts and subjects require paid include= params; fall back to title
                full_text = title
                subjects   = [s.get("name", "") if isinstance(s, dict) else str(s)
                               for s in bill.get("subjects", [])]

                # Trust the search API query — accept all returned bills.
                # No client-side keyword filter: title-only text is too short for phrase matching.
                domains = tag_domains(full_text)

                # Status: enacted/passed bills are more meaningful as instruments
                classification = bill.get("classification", [])
                classification_str = " ".join(str(c) for c in classification).lower()
                latest_action = (bill.get("latest_action_description") or "").lower()
                passed = int(
                    "signed" in latest_action
                    or "enacted" in latest_action
                    or "chaptered" in latest_action
                    or "effective" in latest_action
                    or "became law" in latest_action
                    or "passed" in latest_action
                    or ("bill" in classification_str and "law" in latest_action)
                )

                # Appropriation signal: title mentions fiscal/funding terms
                # (limited recall since full text unavailable on free tier)
                has_appropriation = int(
                    any(w in full_text.lower() for w in
                        ["appropriat", "million", "billion", "fund", "grant",
                         "fiscal", "budget", "allocation", "tax credit"])
                )

                all_bills[bill_id] = {
                    "state": state,
                    "year": year,
                    "bill_id": bill_id,
                    "identifier": bill.get("identifier", ""),
                    "title": title,
                    "abstract": "",
                    "subjects": "; ".join(subjects),
                    "domains": "; ".join(domains),
                    "classification": classification_str,
                    "first_action_date": bill.get("first_action_date", ""),
                    "latest_action_date": bill.get("latest_action_date", ""),
                    "latest_action_description": bill.get("latest_action_description", ""),
                    "passed": passed,
                    "has_appropriation": has_appropriation,
                    "session": bill.get("session", ""),
                    "n_sponsors": len(bill.get("sponsorships", [])),
                    "url": bill.get("openstates_url", ""),
                }

            # Pagination
            pagination = data.get("pagination", {})
            total_pages = pagination.get("max_page", 1)
            if page >= total_pages or page >= 10:   # cap at 10 pages per query
                break
            page += 1

    return list(all_bills.values())


""" Panel builder """

def build_panel(bills_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bill-level data into a (state, year) panel."""
    if bills_df.empty:
        return pd.DataFrame()

    # Explode domain tags so we can count per domain
    bills_df = bills_df.copy()
    bills_df["domain_list"] = bills_df["domains"].str.split("; ")

    agg = (
        bills_df
        .groupby(["state", "year"])
        .agg(
            n_innovation_bills=("bill_id", "count"),
            n_passed=("passed", "sum"),
            n_appropriation=("has_appropriation", "sum"),
            n_unique_domains=("domains", lambda x: len(
                {d for s in x.str.split("; ") for d in s}
            )),
            n_sponsors_total=("n_sponsors", "sum"),
        )
        .reset_index()
    )
    # Pass rate
    agg["pass_rate"] = agg["n_passed"] / agg["n_innovation_bills"].clip(lower=1)

    # Domain-specific counts
    for domain in INNOVATION_DOMAINS:
        mask = bills_df["domains"].str.contains(domain, na=False)
        counts = bills_df[mask].groupby(["state", "year"]).size().rename(f"n_{domain}")
        agg = agg.merge(counts, on=["state", "year"], how="left")
        agg[f"n_{domain}"] = agg[f"n_{domain}"].fillna(0).astype(int)

    return agg


""" Main """

def main():
    parser = argparse.ArgumentParser(description="Pull state legislative innovation data for EIP.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENSTATES_API_KEY", ""),
        help="OpenStates API key (or set OPENSTATES_API_KEY env var). "
             "Register free at https://openstates.org/accounts/profile/",
    )
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2023)
    parser.add_argument(
        "--states",
        nargs="*",
        default=list(STATE_ABBR_TO_OPENSTATES.keys()),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to RAW_DIR/legislative (from set_paths.py)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-fetch state-years already saved (default: skip them).",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "ERROR: OpenStates API key required.\n"
            "  Register free at https://openstates.org/accounts/profile/\n"
            "  Then pass --api_key YOUR_KEY or set OPENSTATES_API_KEY=YOUR_KEY"
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(RAW_DIR) / "legislative"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging — writes to both console and a timestamped log file
    log_path = out_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Run started — states={args.states}  years={args.start}-{args.end}")
    log.info(f"Output directory: {out_dir}")
    log.info(f"Log file: {log_path}")

    all_bills: list[dict] = []
    state_year_pairs = [
        (s, y)
        for s in args.states
        for y in range(args.start, args.end + 1)
    ]

    # Load any previously saved bills so the panel rebuild is always complete
    existing_files = sorted(out_dir.glob('*_bills.csv'))
    for f in existing_files:
        if f.name == 'legislative_bills_all.csv':
            continue
        try:
            all_bills.extend(pd.read_csv(f).to_dict('records'))
        except Exception:
            pass
    already_done = {f.stem for f in existing_files}  # e.g. CA_2019

    pending = [
        (s, y) for s, y in state_year_pairs
        if args.force or f"{s}_{y}" not in already_done
    ]
    skipped = len(state_year_pairs) - len(pending)
    log.info(f"Fetching legislative data: {len(pending)} pending, {skipped} already cached.")

    for state, year in tqdm(pending, desc='state-years'):
        out_path = out_dir / f"{state}_{year}_bills.csv"
        bills = fetch_bills_for_state_year(state, year, args.api_key)
        if bills:
            df = pd.DataFrame(bills)
            df.to_csv(out_path, index=False)
            log.info(f"[ok]    {state} {year}: {len(bills)} bills saved")
            all_bills.extend(bills)
        else:
            # Write empty sentinel so we do not re-fetch on next run
            pd.DataFrame(columns=["state", "year", "bill_id"]).to_csv(out_path, index=False)
            log.info(f"[empty] {state} {year}")

    if all_bills:
        bills_df = pd.DataFrame(all_bills).drop_duplicates(subset=["bill_id"])
        all_path = out_dir / "legislative_bills_all.csv"
        bills_df.to_csv(all_path, index=False)
        log.info(f"All bills: {len(bills_df)} rows -> {all_path}")

        panel = build_panel(bills_df)
        panel_path = out_dir / "legislative_panel.csv"
        panel.to_csv(panel_path, index=False)
        log.info(f"Panel: {len(panel)} rows -> {panel_path}")
    else:
        log.warning("No bills collected.")

    log.info("Run complete.")



if __name__ == "__main__":
    main()
