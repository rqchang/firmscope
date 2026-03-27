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
  python import_legislative.py --api_key YOUR_KEY --start 2010 --end 2023
  python import_legislative.py --api_key YOUR_KEY --states CA TX --start 2020

Get a free API key at: https://openstates.org/accounts/profile/

Requirements
------------
  pip install requests pandas tqdm
"""

import argparse
import os
import time
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
from utils.set_paths import RAW_DIR

# ---------------------------------------------------------------------------
# OpenStates API settings
# ---------------------------------------------------------------------------
OPENSTATES_BASE = "https://v3.openstates.org"
DEFAULT_PAGE_SIZE = 20          # max allowed by free tier
SLEEP_BETWEEN_PAGES = 0.5       # seconds

# ---------------------------------------------------------------------------
# Innovation keyword groups (for search queries and domain tagging)
# ---------------------------------------------------------------------------
INNOVATION_DOMAINS: dict[str, list[str]] = {
    "general_rd":        ["research and development", "r&d", "research funding",
                          "innovation fund", "technology transfer", "tech transfer",
                          "technology commercialization", "intellectual property",
                          "university research", "research university",
                          "national laboratory", "federal research",
                          "applied research", "basic research"],
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
                          "nanotechnology", "materials science"],
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
                          "bioinformatics", "synthetic biology"],
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
                print(f"  [rate limit] sleeping {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  [warn] {url} failed: {exc}. Retrying in {wait}s.")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_bills_for_state_year(
    state: str,
    year: int,
    api_key: str,
    query_keywords: Optional[list[str]] = None,
) -> list[dict]:
    """
    Fetch all innovation-relevant bills for *state* in *year* from OpenStates.
    Returns a list of normalized bill dicts.
    """
    os_state = STATE_ABBR_TO_OPENSTATES.get(state)
    if os_state is None:
        return []

    # OpenStates sessions are identified by state+session label.
    # We filter by updated_since / created_before to approximate calendar year.
    params_base = {
        "jurisdiction": os_state,
        "session": str(year),          # works for most states; fallback below
        "per_page": DEFAULT_PAGE_SIZE,
        "include": ["abstracts", "subjects", "sponsorships"],
    }

    # Build search queries: we run one query per broad domain keyword group
    # to stay within free-tier result limits.
    search_queries = [
        "innovation", "technology", "research", "clean energy",
        "artificial intelligence", "semiconductor", "biotech", "broadband",
        "cybersecurity", "startup", "workforce STEM",
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
                abstract = " ".join(
                    a.get("abstract", "") for a in bill.get("abstracts", [])
                )
                full_text = f"{title} {abstract}"
                subjects = [s.get("name", "") for s in bill.get("subjects", [])]

                # Keep only innovation-relevant bills
                subject_hit = bool(INNOVATION_SUBJECTS & set(subjects))
                kw_hit = any(kw in full_text.lower() for kw in ALL_KEYWORDS)
                if not (subject_hit or kw_hit):
                    continue

                domains = tag_domains(full_text)

                all_bills[bill_id] = {
                    "state": state,
                    "year": year,
                    "bill_id": bill_id,
                    "identifier": bill.get("identifier", ""),
                    "title": title,
                    "abstract": abstract[:500],
                    "subjects": "; ".join(subjects),
                    "domains": "; ".join(domains),
                    "classification": "; ".join(
                        c for c in bill.get("classification", [])
                    ),
                    "first_action_date": bill.get("first_action_date", ""),
                    "latest_action_date": bill.get("latest_action_date", ""),
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


# ---------------------------------------------------------------------------
# Summary panel builder
# ---------------------------------------------------------------------------

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
            n_unique_domains=("domains", lambda x: len(
                {d for s in x.str.split("; ") for d in s}
            )),
            n_sponsors_total=("n_sponsors", "sum"),
        )
        .reset_index()
    )

    # Domain-specific counts
    for domain in INNOVATION_DOMAINS:
        mask = bills_df["domains"].str.contains(domain, na=False)
        counts = bills_df[mask].groupby(["state", "year"]).size().rename(f"n_{domain}")
        agg = agg.merge(counts, on=["state", "year"], how="left")
        agg[f"n_{domain}"] = agg[f"n_{domain}"].fillna(0).astype(int)

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        help="Output directory. Defaults to RAW_DIR/eip/legislative (from set_paths.py)",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "ERROR: OpenStates API key required.\n"
            "  Register free at https://openstates.org/accounts/profile/\n"
            "  Then pass --api_key YOUR_KEY or set OPENSTATES_API_KEY=YOUR_KEY"
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(RAW_DIR) / "eip" / "legislative"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_bills: list[dict] = []
    state_year_pairs = [
        (s, y)
        for s in args.states
        for y in range(args.start, args.end + 1)
    ]

    print(f"Fetching legislative data for {len(state_year_pairs)} (state, year) pairs...")
    for state, year in tqdm(state_year_pairs, desc="state-years"):
        bills = fetch_bills_for_state_year(state, year, args.api_key)
        if bills:
            df = pd.DataFrame(bills)
            out_path = out_dir / f"{state}_{year}_bills.csv"
            df.to_csv(out_path, index=False)
            print(f"  [ok] {state} {year}: {len(bills)} bills → {out_path}")
            all_bills.extend(bills)
        else:
            print(f"  [empty] {state} {year}")

    if all_bills:
        bills_df = pd.DataFrame(all_bills)
        all_path = out_dir / "legislative_bills_all.csv"
        bills_df.to_csv(all_path, index=False)
        print(f"\nAll bills: {len(bills_df)} rows → {all_path}")

        panel = build_panel(bills_df)
        panel_path = out_dir / "legislative_panel.csv"
        panel.to_csv(panel_path, index=False)
        print(f"Panel: {len(panel)} rows → {panel_path}")
    else:
        print("\nNo bills collected.")


if __name__ == "__main__":
    main()
