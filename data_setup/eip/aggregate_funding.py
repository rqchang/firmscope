"""
aggregate_funding.py
--------------------
Reads grant-level raw CSVs produced by import_funding.py and builds the
research panel and Bartik instrument used in the empirical analysis of
external innovation pressure on firm scope.

Pipeline position
-----------------
  import_funding.py  -->  [RAW_DIR/funding/]  -->  aggregate_funding.py
                                                     -->  funding_panel.csv
                                                     -->  bartik_shares.csv

Design rationale
----------------
Aggregation is deliberately separated from fetching (import_funding.py) so
that research-design choices — fiscal-year alignment, grant-type filtering,
Bartik base period — can be varied quickly without re-hitting the APIs.

Responsibilities
----------------
  1. Fiscal-year alignment
       NIH reports by fiscal year (Oct 1 – Sep 30); NSF reports by calendar
       year. To align sources on the same time axis, shift NIH year by
       --fy_shift (default: 1), so that FY2024 (Oct 2023 – Sep 2024) is
       attributed to CY2023. This is the standard treatment in the NIH-funding
       literature (Azoulay et al. 2019).

  2. Research-grant filter (--research_only)
       NIH K-series (career awards) and T-series (training grants) do not
       represent frontier research pressure — they fund human capital formation
       rather than new knowledge production. Including them inflates NIH totals
       at elite research universities (MA, CA, MD), creating a confound between
       innovation pressure and institutional prestige. When --research_only is
       set, only R/P/U/DP-series grants (is_research_grant==True) are kept.
       The flag has no effect on NSF, which is definitionally research by statute.

  3. USASpending exclusion (--no_usaspending)
       USASpending (DOE + DARPA) coverage begins only in 2008, making it
       impossible to compute pre-determined Bartik shares from a pre-sample
       base period (1990–2000). Including it would require shares estimated
       within the sample period, violating the pre-determination requirement
       for IV validity (Goldsmith-Pinkham, Sorkin, Swift 2020). The main
       specification therefore excludes USASpending and relies on NSF + NIH
       alone (1990–2024). USASpending can be added for post-2008 robustness
       checks or domain-specific heterogeneity tests (e.g., clean energy).

  4. Panel aggregation
       Produces a (state, year) panel with source-level totals and per-domain
       breakdowns following the HJT-extended taxonomy (12 domains). Domain
       breakdowns allow construction of the technology-weighted spillover
       measure Fund~_{i,t} = sum_d w_{i,d} * Fund_{s,d,t} used in the
       firm-level regressions.

  5. Bartik shift-share IV
       Computes pre-determined state shares theta_{s,a} = state s's cumulative
       share of agency a's grants over a pre-sample base period. The instrument
       Z_{s,t} = sum_a theta_{s,a} * B_{a,t} (constructed separately using
       national agency budgets B_{a,t}) isolates plausibly exogenous variation
       in local funding from congressional appropriation shocks. Agencies:
       NSF and NIH (treated separately for over-identification tests).

Output files (PROC_DIR/eip/)
-----------------------------
  logs/
    aggregate_funding_{ts}.log   run log with load counts, panel shape, and errors

  funding_panel.csv
    state                     str    2-letter state abbreviation
    year                      int    calendar year (after FY alignment)
    nsf_grants_usd            float  total NSF ($)
    nih_grants_usd            float  total NIH ($); filtered if --research_only
    {src}_{domain}_usd        float  per-source per-domain (12 domains x 2 sources)
    gsp_millions              float  real GSP chained 2017$ (BEA SAGDP9)
    population                int    resident population (BEA SAINC1)
    total_rd_funding_usd      float  sum across included sources
    rd_funding_pct_gsp        float  total / (gsp_millions * 1e6)
    rd_funding_per_capita     float  total / population

  bartik_shares.csv
    state   str    2-letter state abbreviation
    agency  str    nsf | nih
    share   float  state's avg share of agency grants in base period [0, 1]
    Note: shares sum to 1.0 within each agency across states.

Usage
-----
  # Main specification (NSF + NIH only, research grants, 1990–2024)
  python data_setup/eip/aggregate_funding.py --research_only --bartik_base_end 2000 --no_usaspending

  # Include USASpending for robustness (post-2008 subsample only)
  python data_setup/eip/aggregate_funding.py --research_only --bartik_base_end 2000
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

# Project root is three levels up from data_setup/eip/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.set_paths import RAW_DIR, PROC_DIR
from data_import.import_funding import ALL_DOMAINS, classify_domain

RAW_FUNDING  = Path(RAW_DIR)  / "funding"
PROC_EIP     = Path(PROC_DIR) / "eip"
LOG_DIR      = PROC_EIP / "logs"

logger = logging.getLogger("aggregate_funding")


def setup_logging() -> None:
    """Write timestamped log to PROC_DIR/eip/logs/aggregate_funding_{ts}.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"aggregate_funding_{ts}.log"
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log started: %s", log_path)

AGENCIES = ["nsf", "nih", "doe", "darpa"]


# ---------------------------------------------------------------------------
# Load raw CSVs
# ---------------------------------------------------------------------------

def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read one CSV; return empty DataFrame on error."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] Could not read {path.name}: {e}")
        return pd.DataFrame()


def _load_parallel(paths: list[Path], workers: int = 8) -> pd.DataFrame:
    """Load a list of CSV files in parallel and concatenate."""
    if not paths:
        return pd.DataFrame()
    dfs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_read_csv_safe, p): p for p in paths}
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_nsf(workers: int = 8) -> pd.DataFrame:
    paths = sorted((RAW_FUNDING / "NSF").glob("nsf_*.csv"))
    nsf = _load_parallel(paths, workers)
    if nsf.empty:
        return nsf
    nsf["amount_usd"] = pd.to_numeric(nsf["amount_usd"], errors="coerce").fillna(0)
    nsf["agency"] = "nsf"
    if "domain" not in nsf.columns:
        # Vectorize: build lookup key then map — avoids slow row-wise apply
        nsf["_key"] = nsf.get("cfda", pd.Series("", index=nsf.index)).fillna("")
        nsf["domain"] = nsf.apply(
            lambda r: classify_domain(r.get("title", ""), r["_key"]), axis=1
        )
        nsf.drop(columns=["_key"], inplace=True)
    return nsf


def load_nih(fy_shift: int, research_only: bool, workers: int = 8) -> pd.DataFrame:
    paths = sorted((RAW_FUNDING / "NIH").glob("nih_*.csv"))
    nih = _load_parallel(paths, workers)
    if nih.empty:
        return nih
    nih["amount_usd"] = pd.to_numeric(nih["amount_usd"], errors="coerce").fillna(0)
    nih["year"] = nih["year"] - fy_shift  # FY -> CY alignment
    nih["agency"] = "nih"

    if research_only and "is_research_grant" in nih.columns:
        before = len(nih)
        nih = nih[nih["is_research_grant"].astype(bool)]
        print(f"  NIH research_only filter: {before:,} -> {len(nih):,} "
              f"(dropped {before - len(nih):,} training/career awards)")

    if "domain" not in nih.columns:
        nih["domain"] = nih["title"].fillna("").apply(classify_domain)
    return nih


def load_usaspending(fy_shift: int, workers: int = 8) -> pd.DataFrame:
    paths = sorted((RAW_FUNDING / "USASpending").glob("usaspending_*.csv"))
    usa = _load_parallel(paths, workers)
    if usa.empty:
        return usa
    usa["amount_usd"] = pd.to_numeric(usa["amount_usd"], errors="coerce").fillna(0)
    usa["year"] = usa["year"] - fy_shift  # FY -> CY alignment
    if "domain" not in usa.columns:
        usa["domain"] = usa["cfda_title"].fillna("").apply(classify_domain)
    usa["agency"] = usa["cfda_number"].apply(
        lambda c: "darpa" if str(c).startswith("12.") else "doe"
    )
    return usa


def load_bea() -> tuple[pd.DataFrame, pd.DataFrame]:
    gsp_path = RAW_FUNDING / "BEA" / "gsp_panel.csv"
    pop_path = RAW_FUNDING / "BEA" / "population_panel.csv"
    gsp = pd.read_csv(gsp_path) if gsp_path.exists() else pd.DataFrame()
    pop = pd.read_csv(pop_path) if pop_path.exists() else pd.DataFrame()
    return gsp, pop


# ---------------------------------------------------------------------------
# Panel aggregation
# ---------------------------------------------------------------------------

def _agg_total(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby(["state", "year"])["amount_usd"]
        .sum().reset_index().rename(columns={"amount_usd": col})
    )


def _agg_by_domain(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty or "domain" not in df.columns:
        return pd.DataFrame(columns=["state", "year"])
    pivoted = (
        df.groupby(["state", "year", "domain"])["amount_usd"]
        .sum().unstack(fill_value=0).reset_index()
    )
    pivoted.columns = [
        f"{prefix}_{c}_usd" if c not in ("state", "year") else c
        for c in pivoted.columns
    ]
    for domain in ALL_DOMAINS:
        col = f"{prefix}_{domain}_usd"
        if col not in pivoted.columns:
            pivoted[col] = 0.0
    return pivoted


def build_panel(
    nsf: pd.DataFrame,
    nih: pd.DataFrame,
    usa: pd.DataFrame,
    gsp: pd.DataFrame,
    pop: pd.DataFrame,
) -> pd.DataFrame:
    print("  [1/3] Aggregating totals and domain breakdowns...")
    nsf_total  = _agg_total(nsf, "nsf_grants_usd")  if not nsf.empty else pd.DataFrame(columns=["state","year","nsf_grants_usd"])
    nih_total  = _agg_total(nih, "nih_grants_usd")  if not nih.empty else pd.DataFrame(columns=["state","year","nih_grants_usd"])
    usa_total  = _agg_total(usa, "usaspending_rd_grants_usd") if not usa.empty else pd.DataFrame(columns=["state","year","usaspending_rd_grants_usd"])
    nsf_domain = _agg_by_domain(nsf, "nsf")
    nih_domain = _agg_by_domain(nih, "nih")
    usa_domain = _agg_by_domain(usa, "usa")

    print("  [2/3] Merging sources into panel...")
    panel = nsf_total
    for df in [nih_total, usa_total, nsf_domain, nih_domain, usa_domain]:
        if not df.empty:
            panel = panel.merge(df, on=["state", "year"], how="outer")
    if not gsp.empty:
        panel = panel.merge(gsp, on=["state", "year"], how="left")
    if not pop.empty:
        panel = panel.merge(pop, on=["state", "year"], how="left")

    usd_cols = [c for c in panel.columns if c.endswith("_usd")]
    for c in usd_cols:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)

    src_totals = [c for c in usd_cols if c in
                  ("nsf_grants_usd", "nih_grants_usd", "usaspending_rd_grants_usd")]
    panel["total_rd_funding_usd"] = panel[src_totals].sum(axis=1)

    if "gsp_millions" in panel.columns:
        panel["rd_funding_pct_gsp"] = (
            panel["total_rd_funding_usd"] / (panel["gsp_millions"] * 1e6)
        ).replace([float("inf"), -float("inf")], None)
    if "population" in panel.columns:
        panel["rd_funding_per_capita"] = (
            panel["total_rd_funding_usd"] / panel["population"]
        ).replace([float("inf"), -float("inf")], None)

    return panel.sort_values(["state", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bartik shift-share: pre-determined state shares theta_{s,a}
# ---------------------------------------------------------------------------

def build_bartik_shares(
    nsf: pd.DataFrame,
    nih: pd.DataFrame,
    usa: pd.DataFrame,
    base_start: int,
    base_end: int,
) -> pd.DataFrame:
    """
    Compute theta_{s,a} = state s's average share of agency a's total grants
    over the base period [base_start, base_end].

    These pre-determined shares are the "shift" component of the Bartik IV:
        Z_{s,t} = sum_a theta_{s,a} * B_{a,t}
    where B_{a,t} is the national budget of agency a in year t.
    Using a pre-sample base period ensures shares are orthogonal to
    post-base-period changes in states' economic conditions.
    """
    rows = []
    sources = [
        ("nsf",   nsf),
        ("nih",   nih),
        ("doe",   usa[usa["agency"] == "doe"]  if not usa.empty else pd.DataFrame()),
        ("darpa", usa[usa["agency"] == "darpa"] if not usa.empty else pd.DataFrame()),
    ]
    for agency, df in sources:
        if df.empty:
            continue
        base = df[df["year"].between(base_start, base_end)]
        if base.empty:
            print(f"  [warn] No base-period data for agency={agency} "
                  f"({base_start}-{base_end}); shares will be missing.")
            continue
        state_totals   = base.groupby("state")["amount_usd"].sum()
        national_total = state_totals.sum()
        if national_total == 0:
            continue
        for state, total in state_totals.items():
            rows.append({
                "state":  state,
                "agency": agency,
                "share":  total / national_total,
            })
    shares = pd.DataFrame(rows)
    # Verify shares sum to ~1 per agency
    for agency in shares["agency"].unique():
        s = shares[shares["agency"] == agency]["share"].sum()
        print(f"  Bartik shares sum check: {agency} -> {s:.4f} (should be 1.0)")
    return shares


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate raw funding CSVs into research panel and Bartik shares."
    )
    parser.add_argument(
        "--fy_shift", type=int, default=1,
        help="Shift NIH/USASpending fiscal year by this many years to align with "
             "NSF calendar year. Default 1 (FY2024 -> CY2023). Use 0 to disable.",
    )
    parser.add_argument(
        "--research_only", action="store_true",
        help="Restrict NIH to is_research_grant==True (R/P/U/DP series); "
             "exclude training (T) and career (K) awards.",
    )
    parser.add_argument(
        "--bartik_base_start", type=int, default=1990,
        help="Start year of Bartik base period (default: 1990).",
    )
    parser.add_argument(
        "--bartik_base_end", type=int, default=2000,
        help="End year of Bartik base period (default: 2000). "
             "Should precede your sample period to ensure pre-determination.",
    )
    parser.add_argument(
        "--no_usaspending", action="store_true",
        help="Exclude USASpending (DOE + DARPA) from the panel. "
             "Recommended when Bartik pre-determination cannot be satisfied "
             "due to USASpending coverage starting only in 2008.",
    )
    args = parser.parse_args()

    PROC_EIP.mkdir(parents=True, exist_ok=True)
    setup_logging()
    print("=" * 60)
    print("=== EIP Aggregation: funding_panel + bartik_shares ===")
    print("=" * 60)
    print(f"  FY shift      : {args.fy_shift} year(s)")
    print(f"  Research only : {args.research_only}")
    print(f"  Bartik base   : {args.bartik_base_start}-{args.bartik_base_end}")
    print(f"  USASpending   : {'excluded' if args.no_usaspending else 'included'}")

    logger.info("FY shift=%d  research_only=%s  bartik_base=%d-%d  no_usaspending=%s",
                args.fy_shift, args.research_only,
                args.bartik_base_start, args.bartik_base_end, args.no_usaspending)

    print("\n[Step 1] Loading raw CSVs...")
    nsf = load_nsf()
    nih = load_nih(args.fy_shift, args.research_only)
    usa = pd.DataFrame() if args.no_usaspending else load_usaspending(args.fy_shift)
    gsp, pop = load_bea()
    print(f"  NSF : {len(nsf):,} awards")
    print(f"  NIH : {len(nih):,} projects")
    print(f"  USA : {'excluded' if args.no_usaspending else f'{len(usa):,} grants'}")
    print(f"  GSP : {len(gsp):,} state-years | Pop: {len(pop):,} state-years")
    logger.info("Loaded  NSF=%d  NIH=%d  USA=%s  GSP=%d  Pop=%d",
                len(nsf), len(nih),
                "excluded" if args.no_usaspending else len(usa),
                len(gsp), len(pop))

    print("\n[Step 2] Building funding panel...")
    panel = build_panel(nsf, nih, usa, gsp, pop)
    out_panel = PROC_EIP / "funding_panel.csv"
    panel.to_csv(out_panel, index=False)
    logger.info("Panel saved: %s  (%d rows, %d cols)", out_panel, *panel.shape)
    print(f"  [3/3] Saved: {out_panel}")
    print(f"  Shape  : {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"  States : {panel['state'].nunique()} | "
          f"Years  : {int(panel['year'].min())}-{int(panel['year'].max())}")
    preview_cols = [c for c in ["state", "year", "nsf_grants_usd", "nih_grants_usd",
                                "total_rd_funding_usd", "rd_funding_per_capita",
                                "rd_funding_pct_gsp"] if c in panel.columns]
    print("\n  Preview:")
    print(panel[preview_cols].head().to_string(index=False))

    print("\n[Step 3] Computing Bartik pre-determined shares...")
    shares = build_bartik_shares(
        nsf, nih, usa, args.bartik_base_start, args.bartik_base_end
    )
    out_shares = PROC_EIP / "bartik_shares.csv"
    shares.to_csv(out_shares, index=False)
    logger.info("Bartik shares saved: %s  (%d rows)", out_shares, len(shares))
    logger.info("Done.")
    print(f"  Saved: {out_shares}  ({len(shares)} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
