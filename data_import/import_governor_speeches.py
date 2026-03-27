"""
import_governor_speeches.py
---------------------------
Pulls gubernatorial speech / press-release text for every U.S. state and year
so that downstream NLP can score each (state, year) pair on innovation rhetoric.

Data strategy (two complementary layers):
  1. Internet Archive CDX API  – retrieves archived snapshots of each state
     governor's "newsroom" / press-release sub-page and stores the raw HTML for
     later parsing.
  2. State-specific direct URLs – a curated map of known press-release endpoints
     for states whose archives are well-indexed.

Output
------
  data_import/raw/governor_speeches/
    {state_abbr}_{year}_urls.csv    – list of archived page URLs
    {state_abbr}_{year}_texts.csv   – (url, date, title, body_text) rows

Usage
-----
  python import_governor_speeches.py --start 2010 --end 2023 --out_dir raw/governor_speeches
  python import_governor_speeches.py --states CA TX NY --start 2018 --end 2022

Requirements
------------
  pip install requests beautifulsoup4 lxml pandas tqdm
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup – resolve project root and load shared path config
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.set_paths import RAW_DIR

# ---------------------------------------------------------------------------
# State press-release URL templates
# Each value is a Python format string; {year} is substituted at runtime.
# Where no year filter exists in the URL, we fetch the main page and filter by
# date after the fact.
# ---------------------------------------------------------------------------
STATE_PRESS_URLS: dict[str, str] = {
    "AL": "https://governor.alabama.gov/newsroom/press-releases/",
    "AK": "https://gov.alaska.gov/newsroom/press-releases/",
    "AZ": "https://azgovernor.gov/governor/news",
    "AR": "https://governor.arkansas.gov/news/",
    "CA": "https://www.gov.ca.gov/newsroom/",
    "CO": "https://www.colorado.gov/governor/news",
    "CT": "https://portal.ct.gov/Office-of-the-Governor/News",
    "DE": "https://news.delaware.gov/tag/governor/",
    "FL": "https://www.flgov.com/media-center/press-releases/",
    "GA": "https://gov.georgia.gov/press-releases",
    "HI": "https://governor.hawaii.gov/newsroom/",
    "ID": "https://gov.idaho.gov/pressrelease/",
    "IL": "https://gov.illinois.gov/news/press-releases.html",
    "IN": "https://www.in.gov/gov/newsroom/",
    "IA": "https://governor.iowa.gov/press-releases",
    "KS": "https://governor.kansas.gov/newsroom/press-releases/",
    "KY": "https://governor.ky.gov/news/press-releases/",
    "LA": "https://gov.louisiana.gov/index.cfm/newsroom/category/3",
    "ME": "https://www.maine.gov/governor/mills/news/press-releases",
    "MD": "https://governor.maryland.gov/press-releases/",
    "MA": "https://www.mass.gov/governor",
    "MI": "https://www.michigan.gov/whitmer/news/press-releases",
    "MN": "https://mn.gov/governor/newsroom/",
    "MS": "https://governorreeves.ms.gov/news/press-releases/",
    "MO": "https://governor.mo.gov/press-releases",
    "MT": "https://governor.mt.gov/news/press-releases/",
    "NE": "https://governor.nebraska.gov/press",
    "NV": "https://gov.nv.gov/News/Press_Releases/",
    "NH": "https://www.governor.nh.gov/news-and-media/press-releases",
    "NJ": "https://www.nj.gov/governor/news/news/",
    "NM": "https://www.governor.state.nm.us/category/press-releases/",
    "NY": "https://www.governor.ny.gov/news",
    "NC": "https://governor.nc.gov/news/press-releases",
    "ND": "https://www.governor.nd.gov/news",
    "OH": "https://governor.ohio.gov/media-center/press-releases",
    "OK": "https://www.governor.ok.gov/news.html",
    "OR": "https://www.oregon.gov/gov/news/pages/default.aspx",
    "PA": "https://www.governor.pa.gov/newsroom/press-releases/",
    "RI": "https://www.ri.gov/governor/",
    "SC": "https://governor.sc.gov/news/press-releases",
    "SD": "https://news.sd.gov/news.aspx?Show=1&Agency=OOG",
    "TN": "https://www.tn.gov/governor/news.html",
    "TX": "https://gov.texas.gov/news",
    "UT": "https://governor.utah.gov/news/",
    "VT": "https://governor.vermont.gov/newsroom",
    "VA": "https://www.governor.virginia.gov/newsroom/news-releases/",
    "WA": "https://www.governor.wa.gov/news-media/news-releases",
    "WV": "https://governor.wv.gov/News/press-releases/",
    "WI": "https://www.wisconsin.gov/Pages/newsroom.aspx",
    "WY": "https://governor.wyo.gov/news/press-releases",
    "DC": "https://mayor.dc.gov/node/1496301",
}

# ---------------------------------------------------------------------------
# Innovation-related keyword list (used for lightweight pre-filtering so we
# don't store every press release — only those relevant to innovation).
# ---------------------------------------------------------------------------
INNOVATION_KEYWORDS = [
    # Core R&D concepts (stem-matched — trailing chars intentionally dropped)
    "innovat", "technolog", "research", "r&d", "r & d",
    "commercializ", "tech transfer", "intellectual propert", "patent",
    "university research", "national lab",
    # AI / software
    "artificial intelligence", "machine learning", "deep learning",
    "large language model", "generative ai", "computer vision",
    "natural language processing", "algorithm", "autonomous", "robotics",
    "drone", "unmanned system",
    # Hardware / semiconductors
    "semiconductor", "microchip", "chip manufactur", "integrated circuit",
    "advanced manufacturing", "additive manufactur", "3d printing",
    "photonic", "nanotechnolog", "nanotech",
    # Energy / climate
    "clean energy", "renewable", "solar", "wind energy", "offshore wind",
    "electric vehicle", "battery storage", "hydrogen", "nuclear energy",
    "grid moderniz", "smart grid", "carbon capture", "net zero",
    "decarboniz", "climate technolog",
    # Life sciences / health
    "biotech", "biomanufactur", "genomic", "life science", "crispr",
    "gene editing", "precision medicine", "drug discovery", "medtech",
    "digital health", "telemedicine", "health informatic",
    # Quantum / space
    "quantum", "space technolog", "aerospace", "commercial space",
    "satellite", "launch vehicle",
    # Digital infrastructure
    "broadband", "5g", "6g", "fiber optic", "cloud computing",
    "data center", "cybersecurity", "cyber security", "internet of things",
    "digital infrastructure", "smart city", "spectrum",
    # Ecosystem / finance
    "startup", "venture capital", "incubator", "accelerator", "spinoff",
    "angel investor", "seed fund", "tech hub", "innovation district",
    "innovation corridor", "innovation ecosystem",
    # Workforce / education
    "stem", "workforce development", "reskill", "upskill",
    "coding bootcamp", "digital literac", "apprenticeship",
    # Policy instruments
    "chips act", "inflation reduction", "sbir", "sttr",
    "industrial policy", "economic competitiveness", "reshoring",
    "onshoring", "supply chain", "subsidy", "grant", "tax credit",
    "incentive program", "public investment",
    # Broad signals
    "digital transform", "emerging technolog", "industry 4.0",
    "fourth industrial", "future of work",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research bot; firmscope academic project; "
        "contact: researcher@university.edu)"
    )
}

SLEEP_BETWEEN_REQUESTS = 1.5  # seconds — be polite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_innovation_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in INNOVATION_KEYWORDS)


def fetch_with_retry(url: str, retries: int = 3, timeout: int = 20) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  [warn] {url} failed (attempt {attempt+1}/{retries}): {exc}. Retrying in {wait}s.")
            time.sleep(wait)
    return None


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


# ---------------------------------------------------------------------------
# Internet Archive CDX API
# ---------------------------------------------------------------------------
CDX_API = "https://web.archive.org/cdx/search/cdx"


def get_archive_snapshots(url: str, year: int) -> list[dict]:
    """Return a list of {timestamp, original, statuscode} for snapshots in *year*."""
    params = {
        "url": url,
        "output": "json",
        "from": f"{year}0101",
        "to": f"{year}1231",
        "fl": "timestamp,original,statuscode,mimetype",
        "filter": "statuscode:200",
        "filter2": "mimetype:text/html",
        "collapse": "timestamp:8",   # one snapshot per day
        "limit": 365,
    }
    resp = fetch_with_retry(CDX_API + "?" + "&".join(f"{k}={v}" for k, v in params.items()))
    if resp is None:
        return []
    data = resp.json()
    if len(data) < 2:
        return []
    keys = data[0]
    return [dict(zip(keys, row)) for row in data[1:]]


def fetch_archived_page(timestamp: str, original_url: str) -> Optional[str]:
    archive_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
    resp = fetch_with_retry(archive_url)
    return resp.text if resp else None


# ---------------------------------------------------------------------------
# Per-state scraper
# ---------------------------------------------------------------------------

def scrape_state_year(
    state: str,
    year: int,
    out_dir: Path,
    use_archive: bool = True,
    use_live: bool = True,
) -> pd.DataFrame:
    """
    Scrape governor press releases for *state* in *year*.
    Returns a DataFrame with columns [state, year, date, url, title, body].
    """
    records: list[dict] = []
    base_url = STATE_PRESS_URLS.get(state)
    if base_url is None:
        print(f"  [skip] No URL configured for {state}")
        return pd.DataFrame()

    urls_to_fetch: list[tuple[str, str]] = []  # (date_str, url)

    # -- Layer 1: Internet Archive snapshots --------------------------------
    if use_archive:
        snapshots = get_archive_snapshots(base_url, year)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        for snap in snapshots[:30]:   # cap at 30 archived snapshots per state-year
            archive_url = f"https://web.archive.org/web/{snap['timestamp']}/{snap['original']}"
            date_str = snap["timestamp"][:8]
            urls_to_fetch.append((date_str, archive_url))

    # -- Layer 2: live site (current year or recent years) ------------------
    if use_live and year >= datetime.now().year - 2:
        urls_to_fetch.append(("live", base_url))

    seen_urls: set[str] = set()
    for date_str, page_url in urls_to_fetch:
        if page_url in seen_urls:
            continue
        seen_urls.add(page_url)

        resp = fetch_with_retry(page_url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if resp is None:
            continue

        html = resp.text
        body_text = extract_text_from_html(html)

        if not is_innovation_relevant(body_text):
            continue

        # Try to extract individual release links from listing pages
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if any(kw in (href + text).lower() for kw in ["press", "release", "news", "speech"]):
                if href.startswith("http"):
                    links.append((text, href))
                elif href.startswith("/"):
                    from urllib.parse import urlparse
                    parsed = urlparse(page_url)
                    links.append((text, f"{parsed.scheme}://{parsed.netloc}{href}"))

        if links:
            for title, link_url in links[:20]:
                if link_url in seen_urls:
                    continue
                seen_urls.add(link_url)
                link_resp = fetch_with_retry(link_url)
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                if link_resp is None:
                    continue
                link_text = extract_text_from_html(link_resp.text)
                if is_innovation_relevant(link_text):
                    records.append({
                        "state": state,
                        "year": year,
                        "date": date_str,
                        "url": link_url,
                        "title": title,
                        "body": link_text[:5000],   # truncate for storage
                    })
        else:
            # The page itself is a release
            records.append({
                "state": state,
                "year": year,
                "date": date_str,
                "url": page_url,
                "title": "",
                "body": body_text[:5000],
            })

    df = pd.DataFrame(records)
    if not df.empty:
        out_path = out_dir / f"{state}_{year}_texts.csv"
        df.to_csv(out_path, index=False)
        print(f"  [ok] {state} {year}: {len(df)} records → {out_path}")
    else:
        print(f"  [empty] {state} {year}: no innovation-relevant releases found")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape governor speeches for EIP component 1.")
    parser.add_argument("--start", type=int, default=2010, help="Start year (inclusive)")
    parser.add_argument("--end", type=int, default=2023, help="End year (inclusive)")
    parser.add_argument(
        "--states",
        nargs="*",
        default=list(STATE_PRESS_URLS.keys()),
        help="State abbreviations to process (default: all 50 + DC)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to RAW_DIR/eip/governor_speeches (from set_paths.py)",
    )
    parser.add_argument(
        "--no_archive",
        action="store_true",
        help="Skip Internet Archive lookups (faster, live-only)",
    )
    parser.add_argument(
        "--no_live",
        action="store_true",
        help="Skip live site fetches (archive-only)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(RAW_DIR) / "eip" / "governor_speeches"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    state_year_pairs = [
        (s, y)
        for s in args.states
        for y in range(args.start, args.end + 1)
    ]

    print(f"Processing {len(state_year_pairs)} (state, year) pairs...")
    for state, year in tqdm(state_year_pairs, desc="state-years"):
        df = scrape_state_year(
            state=state,
            year=year,
            out_dir=out_dir,
            use_archive=not args.no_archive,
            use_live=not args.no_live,
        )
        if not df.empty:
            all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = out_dir / "governor_speeches_all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nDone. Combined dataset: {len(combined)} rows → {combined_path}")
    else:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()
