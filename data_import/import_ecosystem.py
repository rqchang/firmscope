"""
import_ecosystem.py
-------------------
Pulls state-level local innovation ecosystem activity data for EIP component 4
(Local Innovation Ecosystem Activity).

Data sources (all free/public):
  1. Census Bureau Business Formation Statistics (BFS) API
       → high-propensity business application formation by state and month
       → proxy for startup activity / entrepreneurial dynamism
  2. SBA SBIR/STTR Award Data (public database)
       → Small Business Innovation Research awards by state and year
       → direct measure of innovation-oriented private-sector activity
  3. GDELT 2.0 Global Knowledge Graph (GKG) API
       → news event counts mentioning innovation/tech themes by state
       → proxy for "innovation-oriented local news intensity"
  4. USPTO PatentsView API
       → patent counts by state and year (inventor location)
       → proxy for realized private innovation output

Output
------
  data_import/raw/ecosystem/
    bfs_panel.csv              – business formation by state-year
    sbir_panel.csv             – SBIR/STTR awards by state-year
    gdelt_news_panel.csv       – innovation news intensity by state-year
    patents_panel.csv          – patent counts by state-year
    ecosystem_panel.csv        – combined (state, year) panel

Usage
-----
  python import_ecosystem.py --start 2010 --end 2023
  python import_ecosystem.py --states CA TX NY --start 2018

Requirements
------------
  pip install requests pandas tqdm
"""

import argparse
import io
import time
import zipfile
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

SLEEP = 0.5

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

# State full names → abbreviation (for matching in non-FIPS sources)
STATE_NAME_TO_ABBR: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def get_json(url: str, params: Optional[dict] = None, retries: int = 4) -> Optional[dict | list]:
    headers = {"User-Agent": "firmscope-research-bot/1.0", "Accept": "application/json"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  [warn] {url}: {exc}. Retry {attempt+1}/{retries} in {wait}s.")
            time.sleep(wait)
    return None


def get_text(url: str, params: Optional[dict] = None) -> Optional[str]:
    headers = {"User-Agent": "firmscope-research-bot/1.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        print(f"  [warn] {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. Census Bureau Business Formation Statistics (BFS)
#    https://www.census.gov/econ/bfs/index.html
# ---------------------------------------------------------------------------
BFS_API = "https://api.census.gov/data/timeseries/eits/bfs"


def fetch_bfs(years: list[int], out_dir: Path, census_key: str = "") -> pd.DataFrame:
    """
    Fetch monthly high-propensity business applications (HBA) by state from BFS.
    Aggregate to annual totals.
    """
    print("  BFS: fetching business formation stats...", end="", flush=True)
    all_rows: list[dict] = []

    for year in years:
        params: dict = {
            "get": "cell_value,seasonally_adj,data_type_code,geo_level_code,category_code",
            "for": "state:*",
            "time": f"{year}",
            "data_type_code": "BA_HBA",  # High-propensity business applications
        }
        if census_key:
            params["key"] = census_key

        data = get_json(BFS_API, params)
        time.sleep(SLEEP)
        if data is None or not isinstance(data, list) or len(data) < 2:
            continue

        header = data[0]
        for row in data[1:]:
            d = dict(zip(header, row))
            fips = d.get("state", "")
            state = FIPS_TO_STATE.get(fips)
            if state is None:
                continue
            try:
                value = float(d.get("cell_value", 0) or 0)
            except ValueError:
                continue
            all_rows.append({
                "state": state,
                "year": year,
                "hba_applications": value,
            })

    if not all_rows:
        print(" no data")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Aggregate monthly to annual (BFS returns monthly)
    panel = (
        df.groupby(["state", "year"])["hba_applications"]
        .sum()
        .reset_index()
        .rename(columns={"hba_applications": "hba_annual"})
    )
    out_path = out_dir / "bfs_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f" {len(panel)} state-year rows → {out_path}")
    return panel


# ---------------------------------------------------------------------------
# 2. SBA SBIR/STTR Awards
#    https://www.sbir.gov/sbirsearch/award/all  (bulk download available)
# ---------------------------------------------------------------------------
SBIR_BULK_URL = "https://www.sbir.gov/sites/default/files/awards_latest.zip"
SBIR_API = "https://api.www.sbir.gov/public/api/awards"


def fetch_sbir_awards(years: list[int], out_dir: Path) -> pd.DataFrame:
    """
    Fetch SBIR/STTR awards from SBA, aggregate by state and year.
    Tries bulk download first; falls back to paginated API.
    """
    print("  SBIR: fetching awards...", end="", flush=True)

    # Try bulk CSV download
    try:
        resp = requests.get(SBIR_BULK_URL, timeout=120,
                            headers={"User-Agent": "firmscope-research-bot/1.0"})
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                df_raw = pd.read_csv(f, low_memory=False)

        # Normalize columns
        df_raw.columns = [c.lower().strip().replace(" ", "_") for c in df_raw.columns]
        year_col = next((c for c in df_raw.columns if "year" in c or "date" in c), None)
        state_col = next((c for c in df_raw.columns if "state" in c), None)
        amount_col = next((c for c in df_raw.columns if "amount" in c or "award" in c.lower()), None)

        if year_col and state_col:
            df_raw["_year"] = pd.to_numeric(
                df_raw[year_col].astype(str).str[:4], errors="coerce"
            )
            df_filt = df_raw[df_raw["_year"].isin(years)].copy()
            df_filt["state"] = df_filt[state_col].astype(str).str.upper().str.strip()
            df_filt["state"] = df_filt["state"].apply(
                lambda s: STATE_NAME_TO_ABBR.get(s.lower(), s if len(s) == 2 else None)
            )
            df_filt = df_filt[df_filt["state"].isin(STATES_FIPS)]

            agg_cols: dict = {"count": ("state", "count")}
            if amount_col:
                agg_cols["total_amount"] = (amount_col, "sum")

            panel = (
                df_filt.rename(columns={"_year": "year"})
                .groupby(["state", "year"])
                .agg(
                    sbir_n_awards=("state", "count"),
                    **({} if amount_col is None else {"sbir_total_usd": (amount_col, "sum")}),
                )
                .reset_index()
            )
            out_path = out_dir / "sbir_panel.csv"
            panel.to_csv(out_path, index=False)
            print(f" {len(panel)} rows (bulk) → {out_path}")
            return panel

    except Exception as exc:
        print(f"\n  [warn] bulk SBIR download failed: {exc}. Trying API...")

    # Fallback: paginated API
    all_rows: list[dict] = []
    for year in years:
        page = 1
        while True:
            params = {
                "rows": 500,
                "start": (page - 1) * 500,
                "fiscal_year": year,
                "fields": "firm_state,award_amount,contract_start_date,program",
            }
            data = get_json(SBIR_API, params)
            time.sleep(SLEEP)
            if data is None:
                break
            items = data if isinstance(data, list) else data.get("data", [])
            if not items:
                break
            for item in items:
                state = (item.get("firm_state") or "").upper().strip()
                state = STATE_NAME_TO_ABBR.get(state.lower(), state if len(state) == 2 else None)
                if state not in STATES_FIPS:
                    continue
                all_rows.append({
                    "state": state,
                    "year": year,
                    "amount_usd": item.get("award_amount", 0) or 0,
                })
            if len(items) < 500:
                break
            page += 1

    if not all_rows:
        print(" no data")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    panel = (
        df.groupby(["state", "year"])
        .agg(sbir_n_awards=("state", "count"), sbir_total_usd=("amount_usd", "sum"))
        .reset_index()
    )
    out_path = out_dir / "sbir_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f" {len(panel)} rows (API) → {out_path}")
    return panel


# ---------------------------------------------------------------------------
# 3. GDELT – Innovation-related news intensity by state
#    GDELT GKG v2 allows full-text search via BigQuery, but for free/no-auth
#    access we use the GDELT 2.0 DOC API (limited) and the GKG summary API.
# ---------------------------------------------------------------------------
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

INNOVATION_THEMES = [
    "INNOVATION", "TECHNOLOGY", "STARTUP", "VENTURE_CAPITAL",
    "ARTIFICIAL_INTELLIGENCE", "CLEAN_ENERGY", "RENEWABLE_ENERGY",
    "SEMICONDUCTOR", "BIOTECH", "CYBERSECURITY",
]

# GDELT FIPS-based location codes for US states
# GDELT uses ISO 3166-2 codes like "US-CA", "US-TX", etc.
GDELT_STATE_CODES = {abbr: f"US-{abbr}" for abbr in STATES_FIPS}


def fetch_gdelt_news(years: list[int], out_dir: Path) -> pd.DataFrame:
    """
    Estimate innovation-news intensity per state per year using GDELT DOC API.
    Note: GDELT free API returns aggregate article counts for queries.
    We iterate over years and themes to build a frequency panel.
    """
    print("  GDELT: fetching innovation news counts...")
    rows: list[dict] = []

    for year in tqdm(years, desc="GDELT years"):
        # GDELT DOC API: timespan is limited to ~3 months for free tier;
        # we use quarterly queries and sum them.
        quarters = [
            (f"{year}0101000000", f"{year}0331235959"),
            (f"{year}0401000000", f"{year}0630235959"),
            (f"{year}0701000000", f"{year}0930235959"),
            (f"{year}1001000000", f"{year}1231235959"),
        ]

        year_counts: dict[str, int] = {abbr: 0 for abbr in STATES_FIPS}

        for start_dt, end_dt in quarters:
            for theme in INNOVATION_THEMES[:5]:   # limit to avoid rate limits
                params = {
                    "query": f"sourceloc:US theme:{theme}",
                    "mode": "artlist",
                    "maxrecords": 250,
                    "startdatetime": start_dt,
                    "enddatetime": end_dt,
                    "format": "json",
                }
                data = get_json(GDELT_DOC_API, params)
                time.sleep(SLEEP * 2)  # be extra polite with GDELT
                if data is None:
                    continue

                articles = data.get("articles", [])
                for art in articles:
                    # Extract state from source country/location fields
                    locations = art.get("socialimage", "") or ""
                    url = art.get("url", "") or ""
                    # GDELT article metadata often has state in seendate or sourcecountry
                    # We use a heuristic: match known state patterns in URL domains
                    for state_abbr in STATES_FIPS:
                        state_lower = state_abbr.lower()
                        if (
                            f".{state_lower}." in url.lower()
                            or f"/{state_lower}/" in url.lower()
                        ):
                            year_counts[state_abbr] += 1
                            break

        # Append annual counts
        for state_abbr, count in year_counts.items():
            if count > 0:
                rows.append({"state": state_abbr, "year": year, "gdelt_innovation_articles": count})

    if not rows:
        print("  GDELT: no data collected (API may be rate-limiting)")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_path = out_dir / "gdelt_news_panel.csv"
    df.to_csv(out_path, index=False)
    print(f"  GDELT: {len(df)} rows → {out_path}")
    return df


# ---------------------------------------------------------------------------
# 4. USPTO PatentsView API – patents by state and year
# ---------------------------------------------------------------------------
PATENTSVIEW_API = "https://search.patentsview.org/api/v1/patent/"


def fetch_patents(years: list[int], out_dir: Path) -> pd.DataFrame:
    """
    Fetch patent counts by state and year from PatentsView.
    Uses the inventor location to assign patents to states.
    """
    print("  PatentsView: fetching patent counts by state-year...")
    rows: list[dict] = []

    for year in tqdm(years, desc="PatentsView years"):
        for state_abbr in STATES_FIPS:
            params = {
                "q": (
                    f'{{"_and":[{{"_gte":{{"patent_date":"{year}-01-01"}}}},'
                    f'{{"_lte":{{"patent_date":"{year}-12-31"}}}},'
                    f'{{"inventor_state":"{state_abbr}"}}]}}'
                ),
                "f": '["patent_id","patent_date"]',
                "o": '{"per_page":1}',   # we only need the count
            }
            data = get_json(PATENTSVIEW_API, params)
            time.sleep(SLEEP)
            if data is None:
                continue

            total = data.get("total_patent_count", 0) or 0
            if total > 0:
                rows.append({
                    "state": state_abbr,
                    "year": year,
                    "n_patents": total,
                })

    if not rows:
        print("  PatentsView: no data")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_path = out_dir / "patents_panel.csv"
    df.to_csv(out_path, index=False)
    print(f"  PatentsView: {len(df)} rows → {out_path}")
    return df


# ---------------------------------------------------------------------------
# Ecosystem panel assembler
# ---------------------------------------------------------------------------

def build_ecosystem_panel(
    bfs_df: pd.DataFrame,
    sbir_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    patents_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all ecosystem streams into a single (state, year) panel.
    Computes a raw composite ecosystem score via z-score standardization.
    """
    frames = [
        df for df in [bfs_df, sbir_df, gdelt_df, patents_df] if not df.empty
    ]
    if not frames:
        return pd.DataFrame()

    panel = frames[0]
    for df in frames[1:]:
        panel = panel.merge(df, on=["state", "year"], how="outer")

    panel = panel.sort_values(["state", "year"]).reset_index(drop=True)

    # Z-score each component (within-variable standardization)
    component_cols = [
        c for c in panel.columns
        if c not in ("state", "year")
    ]
    for col in component_cols:
        if col in panel.columns:
            mean = panel[col].mean()
            std = panel[col].std()
            if std > 0:
                panel[f"{col}_z"] = (panel[col] - mean) / std

    # Composite ecosystem score = mean of available z-scores
    z_cols = [c for c in panel.columns if c.endswith("_z")]
    if z_cols:
        panel["ecosystem_score_raw"] = panel[z_cols].mean(axis=1)

    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pull innovation ecosystem data for EIP.")
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2023)
    parser.add_argument(
        "--states",
        nargs="*",
        default=list(STATES_FIPS.keys()),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to RAW_DIR/eip/ecosystem (from set_paths.py)",
    )
    parser.add_argument(
        "--census_key",
        type=str,
        default="",
        help="Optional Census API key (for BFS rate limit relief)",
    )
    parser.add_argument("--skip_gdelt", action="store_true",
                        help="Skip GDELT (slow, rate-limited)")
    parser.add_argument("--skip_patents", action="store_true",
                        help="Skip PatentsView (many API calls)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(RAW_DIR) / "eip" / "ecosystem"
    out_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(args.start, args.end + 1))

    print("=== Business Formation Statistics (Census BFS) ===")
    bfs_df = fetch_bfs(years, out_dir, census_key=args.census_key)

    print("=== SBIR/STTR Awards (SBA) ===")
    sbir_df = fetch_sbir_awards(years, out_dir)

    gdelt_df = pd.DataFrame()
    if not args.skip_gdelt:
        print("=== GDELT Innovation News ===")
        gdelt_df = fetch_gdelt_news(years, out_dir)
    else:
        print("[skip] GDELT")

    patents_df = pd.DataFrame()
    if not args.skip_patents:
        print("=== USPTO PatentsView ===")
        patents_df = fetch_patents(years, out_dir)
    else:
        print("[skip] PatentsView")

    print("=== Building ecosystem panel ===")
    panel = build_ecosystem_panel(bfs_df, sbir_df, gdelt_df, patents_df)
    if not panel.empty:
        panel_path = out_dir / "ecosystem_panel.csv"
        panel.to_csv(panel_path, index=False)
        print(f"Ecosystem panel: {len(panel)} rows → {panel_path}")
        print(panel.head(10).to_string(index=False))
    else:
        print("No ecosystem data collected.")


if __name__ == "__main__":
    main()
