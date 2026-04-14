"""
scope_funding_reg.py
--------------------
OLS and IV (Bartik shift-share) regressions of firm scope on state R&D funding.

Specification (firm i, state s, year t):
    d2vscope_{i,s,t} = beta * ln(rd_pc_{s,t}) + alpha_i + gamma_t + X_{i,t} + e

    X_{i,t}: ln(sale), xrd/sale, oibdp/at, fdat/at, che/at, rd_active

IV  Bartik shift-share [Goldsmith-Pinkham et al. 2020, leave-one-out]
    Z_{s,t} = sum_a  theta_{s,a} * B_{a,t}^{-s}
    theta_{s,a}       state s pre-determined share of national agency-a funding
    B_{a,t}^{-s}      national agency-a budget in year t, excluding state s

FEs:  firm (gvkey) + industry x year (2-digit NAICS from naicsh_filled)
      Industry x year FE absorbs standalone year FE and industry-specific
      technology cycles that may correlate with state funding patterns.
SEs:  clustered by state (treatment variation is at state-year level)

State assignment:
    Firms are linked to states via their time-varying HQ/operational state
    (crsp.comphist hstate), NOT the Compustat state of incorporation.
    Compustat's state field on funda is the incorporation state — ~65% of
    large public firms are incorporated in Delaware regardless of operations.
    Using the HQ state eliminates the DE/NV incorporation-haven misclassification
    that would attenuate estimates and introduce systematic bias.
    Source: data_setup/compustat/assign_loc.py → firm_location_panel.csv

Data inputs:
    data/raw/scope/FirmScope.txt                        Hoberg-Phillips scope
    data/processed/Compustat/compustat_annual.csv       financials
    data/processed/Compustat/firm_location_panel.csv    time-varying HQ + incorp state
    data/processed/eip/funding_panel.csv                state-year R&D panel
    data/processed/eip/bartik_shares.csv                pre-determined Bartik shares
"""

import os
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from linearmodels.iv.absorbing import AbsorbingLS
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyreadr")
from utils.set_paths import RAW_DIR, PROC_DIR, TABLES_DIR, PLOTS_DIR

RAW_SCOPE = os.path.join(RAW_DIR,  "scope")
PROC_EIP  = os.path.join(PROC_DIR, "eip")
PROC_CS   = os.path.join(PROC_DIR, "Compustat")


""" Read in data """
" Scope data (Hoberg-Phillips d2vscope) "
scope = pd.read_csv(os.path.join(RAW_SCOPE, "FirmScope.txt"), sep="\t")
scope["gvkey"] = scope["gvkey"].astype(str).str.zfill(6)
# year == first 4 digits of Compustat datadate (fiscal year end year)

" Compustat annual financials "
comp = pd.read_csv(os.path.join(PROC_CS, "compustat_annual.csv"), dtype={"gvkey": str})
comp["gvkey"] = comp["gvkey"].str.zfill(6)
comp = comp.rename(columns={"fyear": "year"})
comp["year"] = comp["year"].astype("Int64")
comp = comp[["gvkey", "year", "sale", "at", "xrd", "oibdp", "fdat", "che", "sic", "naicsh_filled"]].copy()

" Time-varying firm location: HQ state (operational) + state of incorporation "
loc = pd.read_csv(os.path.join(PROC_CS, "firm_location_panel.csv"), dtype={"gvkey": str})
loc["gvkey"] = loc["gvkey"].str.zfill(6)
loc["year"]  = loc["year"].astype("Int64")

" Segment-based firm scope (Compustat seg_ann) "
seg = pd.read_csv(os.path.join(PROC_CS, "compustat_segments_annual.csv"), dtype={"gvkey": str})
seg["gvkey"] = seg["gvkey"].astype(str).str.zfill(6)
seg["year"]  = seg["fyear"].astype("Int64")
seg = seg[["gvkey", "year", "n_segments", "seg_hhi", "seg_diversity"]].copy()

" State-year funding panel "
fund = pd.read_csv(os.path.join(PROC_EIP, "funding_panel.csv"))
fund = fund[fund["state"].apply(lambda x: str(x).isalpha() and len(str(x)) == 2)].copy()
fund = fund[fund["state"] != "DC"].copy()

" Bartik shares IV "
shares = pd.read_csv(os.path.join(PROC_EIP, "bartik_shares.csv"))


""" Construct regression data """
" Build agency-level Bartik shift-share instrument "
# Z_{s,t} = sum_{a in {NSF,NIH}}  theta_{s,a}  *  B_{a,t}^{-s}
#
# theta_{s,a}: pre-determined share of state s in agency a's total grants,
#   averaged over pre-sample years 1990-1996 (before firm panel starts 1997).
#   Satisfies the predetermined condition: captures historical research
#   infrastructure rather than forward-looking grant-seeking behavior.
#
# B_{a,t}^{-s}: leave-one-out national budget (Goldsmith-Pinkham et al. 2020),
#   removing state s's own grants so idiosyncratic local shocks don't feed
#   back into the instrument.
#
# Scale-matching: Z_{s,t} predicts total R&D dollars.  Using it to instrument
# a ratio treatment (R&D/pop or R&D/GSP) causes a sign flip because large
# states have high Z but moderate funding intensity.  Dividing by the same
# denominator as the treatment removes this bias:
#   treat = ln(R&D/pop)  → instrument = ln(Z / pop)
#   treat = ln(R&D/GSP)  → instrument = ln(Z / GSP)

# Pre-sample shares: 1990-1996 average
fund_base  = fund[fund["year"] <= 1996]
base_state = (
    fund_base.groupby("state")[["nsf_grants_usd", "nih_grants_usd"]]
    .mean().fillna(0)
)
base_state["share_nsf"] = base_state["nsf_grants_usd"] / base_state["nsf_grants_usd"].sum()
base_state["share_nih"] = base_state["nih_grants_usd"] / base_state["nih_grants_usd"].sum()

# National totals by year
nat_yr = (
    fund.groupby("year")[["nsf_grants_usd", "nih_grants_usd"]]
    .sum()
    .rename(columns={"nsf_grants_usd": "nat_nsf", "nih_grants_usd": "nat_nih"})
    .reset_index()
)

# State-year panel with LOO shifts
state_yr = fund[["state", "year", "nsf_grants_usd", "nih_grants_usd",
                  "rd_funding_per_capita", "rd_funding_pct_gsp",
                  "gsp_millions", "population"]].copy()
state_yr = state_yr.merge(nat_yr, on="year", how="left")
state_yr["loo_nsf"] = state_yr["nat_nsf"] - state_yr["nsf_grants_usd"].fillna(0)
state_yr["loo_nih"] = state_yr["nat_nih"] - state_yr["nih_grants_usd"].fillna(0)

# Merge pre-sample shares
state_yr = state_yr.merge(
    base_state[["share_nsf", "share_nih"]].reset_index(),
    on="state", how="left"
)

# Z_{s,t} = theta_{s,NSF} * LOO_NSF_{t}^{-s} + theta_{s,NIH} * LOO_NIH_{t}^{-s}
state_yr["bartik_z"] = (
    state_yr["share_nsf"] * state_yr["loo_nsf"] +
    state_yr["share_nih"] * state_yr["loo_nih"]
)

# Scale-matched composite instruments
_pop = state_yr["population"].clip(lower=1)
_gsp = state_yr["gsp_millions"].clip(lower=1e-3) * 1e6
state_yr["bartik_z_pc"]      = state_yr["bartik_z"] / _pop
state_yr["bartik_z_pct_gsp"] = state_yr["bartik_z"] / _gsp

# Agency-level per-capita instruments (for over-identified IV and Rotemberg weights)
state_yr["bartik_z_nsf_pc"]  = state_yr["share_nsf"] * state_yr["loo_nsf"] / _pop
state_yr["bartik_z_nih_pc"]  = state_yr["share_nih"] * state_yr["loo_nih"] / _pop


" Merge to firm-year panel "
df = pd.merge(scope, comp, on=["gvkey", "year"], how="inner")

# Attach time-varying HQ state and state of incorporation.
# hq_state  → renamed to 'state' so all downstream code is unchanged.
# incorp_state → retained for robustness checks (DE/NV haven exclusion etc.).
df = pd.merge(df, loc[["gvkey", "year", "hq_state", "incorp_state"]],
              on=["gvkey", "year"], how="left")
df = df.rename(columns={"hq_state": "state"})
df = df.dropna(subset=["state"]).copy()   # drop firms with no HQ-state record

df = pd.merge(df, state_yr[["state", "year", "rd_funding_per_capita", "rd_funding_pct_gsp",
                            "bartik_z", "bartik_z_pc", "bartik_z_pct_gsp",
                            "bartik_z_nsf_pc", "bartik_z_nih_pc",
                            "gsp_millions", "population"]].drop_duplicates(["state", "year"]),
                        on=["state", "year"], how="inner")
df = pd.merge(df, seg, on=["gvkey", "year"], how="left")

" Sample restrictions & variable construction "
df = df[df["year"].between(2000, 2021)].copy()   # scope: 1988-2021; base period ends 1996; sample 2000-2021
df = df[df["rd_funding_per_capita"] > 0].copy()
df = df[df["rd_funding_pct_gsp"]    > 0].copy()
df = df[df["bartik_z"] > 0].copy()
df = df[df["sale"] > 0].copy()
df = df[df["at"]   > 0].copy()

df["ln_rd_pc"]          = np.log(df["rd_funding_per_capita"])
df["ln_rd_pct_gsp"]     = np.log(df["rd_funding_pct_gsp"])
df["ln_z_pc"]           = np.log(df["bartik_z_pc"].clip(lower=1e-6))
df["ln_z_pct_gsp"]      = np.log(df["bartik_z_pct_gsp"].clip(lower=1e-12))
df["ln_z_nsf"]          = np.log(df["bartik_z_nsf_pc"].clip(lower=1e-12))
df["ln_z_nih"]          = np.log(df["bartik_z_nih_pc"].clip(lower=1e-12))
df["ln_sale"]       = np.log(df["sale"])
df["rd_intensity"]  = (df["xrd"].fillna(0) / df["sale"]).clip(0, 1)
df["roa"]           = (df["oibdp"] / df["at"]).clip(-1, 1)
df["leverage"]      = (df["fdat"]  / df["at"]).clip(0, 2)
df["cash_ratio"]    = (df["che"]   / df["at"]).clip(0, 1)
df["naics2"]        = df["naicsh_filled"].astype(str).str[:2]
df["ln_gsp_pc"]     = np.log(df["gsp_millions"] * 1e6 / df["population"].clip(lower=1))

# State GDP growth (robustness control)
df = df.sort_values(["state", "year"])
df["gsp_growth"] = df.groupby("state")["gsp_millions"].pct_change()

# Lag treatment, instruments, and all controls by one year within firm
df = df.sort_values(["gvkey", "year"])
_to_lag = ["ln_rd_pc", "ln_rd_pct_gsp", "ln_z_pc", "ln_z_pct_gsp",
           "ln_z_nsf", "ln_z_nih",
           "ln_sale", "rd_intensity", "roa", "leverage", "cash_ratio",
           "ln_gsp_pc", "gsp_growth"]
for col in _to_lag:
    df[f"{col}_lag"] = df.groupby("gvkey")[col].shift(1)

# Binary dependent variable: 1[Δd2vscope > 0] (LPM)
df["d2vscope_prev"]  = df.groupby("gvkey")["d2vscope"].shift(1)
df["scope_increase"] = (df["d2vscope"] - df["d2vscope_prev"] > 0).astype(float)

df = df.dropna(subset=["scope_increase",
                        "ln_rd_pc_lag", "ln_rd_pct_gsp_lag",
                        "ln_z_pc_lag", "ln_z_pct_gsp_lag",
                        "ln_z_nsf_lag", "ln_z_nih_lag",
                        "ln_sale_lag", "rd_intensity_lag", "roa_lag",
                        "leverage_lag", "cash_ratio_lag", "ln_gsp_pc_lag",
                        "gsp_growth_lag", "naics2"]).copy()
df = df.reset_index(drop=True)

print(
    f"Sample: {df['gvkey'].nunique():,} firms  "
    f"{df['state'].nunique()} states  "
    f"{df['year'].nunique()} years  "
    f"{len(df):,} firm-year obs"
)


""" Run regressions """
CTRL_COLS        = ["ln_sale_lag", "rd_intensity_lag", "roa_lag",
                    "leverage_lag", "cash_ratio_lag", "ln_gsp_pc_lag"]
CTRL_COLS_ROBUST = CTRL_COLS + ["gsp_growth_lag"]  # + lagged state GDP growth

# Panel structure for PanelOLS (entity=gvkey, time=year)
df_p       = df.set_index(["gvkey", "year"])
clusters_p = df_p["state"]

# Industry×year categorical for PanelOLS other_effects (Spec 2)
indyr_fe = pd.DataFrame(
    {"indyr": pd.Categorical(df["naics2"] + "_" + df["year"].astype(str))},
    index=df_p.index,
)

# Absorb DataFrames for AbsorbingLS IV
absorb_yr    = pd.DataFrame({"firm": pd.Categorical(df["gvkey"]),
                              "year": pd.Categorical(df["year"].astype(str))})
absorb_indyr = pd.DataFrame({"firm":  pd.Categorical(df["gvkey"]),
                              "indyr": pd.Categorical(df["naics2"] + "_" + df["year"].astype(str))})
clusters_abs = df["state"].astype("category")


" Regression helpers "
def run_panel_ols(df_panel, depvar, exog_cols, other_effects=None):
    """OLS with firm + year FE via PanelOLS, clustered by state."""
    oe = {"other_effects": other_effects} if other_effects is not None else {}
    return PanelOLS(
        df_panel[depvar],
        df_panel[exog_cols],
        entity_effects=True,
        time_effects=(other_effects is None),
        **oe,
    ).fit(cov_type="clustered", clusters=clusters_p)


def run_absorbing_iv(df_reg, depvar, endog_col, instrument_cols, exog_cols, absorb_df):
    """
    2SLS with absorbed FEs, clustered by state.
    Partials out FEs via AbsorbingLS (OLS) then runs IV2SLS on residuals (FWL).
    instrument_cols: str (just-identified) or list of str (over-identified).
    """
    if isinstance(instrument_cols, str):
        instrument_cols = [instrument_cols]

    def _resid(col):
        ones = pd.Series(np.ones(len(df_reg)), index=df_reg.index, name="_c")
        r = AbsorbingLS(df_reg[col], ones, absorb=absorb_df).fit()
        return pd.Series(r.resids.values.ravel(), index=df_reg.index, name=col)

    y_r = _resid(depvar)
    x_r = _resid(endog_col)
    Z_r = pd.concat([_resid(z) for z in instrument_cols], axis=1)
    C_r = pd.concat([_resid(c) for c in exog_cols], axis=1)

    return IV2SLS(y_r, C_r, x_r.to_frame(), Z_r
                  ).fit(cov_type="clustered", clusters=clusters_abs)


def print_result(res, title):
    """Print regression summary in the style of mps_baseline.py."""
    def stars(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "  "
    ci = res.conf_int()
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  {'-' * 66}")
    print(f"  {'Variable':28s}  {'Coef':>9}  {'[95% CI]':>20}  {'t':>7}  Sig")
    print(f"  {'-' * 66}")
    for var in res.params.index:
        p  = res.pvalues[var]
        lo = ci.loc[var, "lower"]
        hi = ci.loc[var, "upper"]
        print(f"  {var:28s}  {res.params[var]:9.4f}  "
              f"[{lo:8.4f}, {hi:8.4f}]  {res.tstats[var]:7.2f}  {stars(p)}")
    print(f"  {'-' * 66}")
    try:
        print(f"  R² (within): {res.rsquared_within:.4f}   N: {res.nobs:,}")
    except AttributeError:
        print(f"  R²: {res.rsquared:.4f}   N: {res.nobs:,}")
    print(f"{'=' * 70}")


def to_coef_df(res, label):
    """Return tidy coefficient DataFrame for output."""
    ci = res.conf_int()
    return pd.DataFrame({
        "spec":     label,
        "variable": list(res.params.index),
        "coef":     res.params.values,
        "se":       res.std_errors.values,
        "tstat":    res.tstats.values,
        "pval":     res.pvalues.values,
        "ci_lo":    ci["lower"].values,
        "ci_hi":    ci["upper"].values,
    })


def run_all_specs(treat_lag, instr_lag="ln_z_lag", ctrl_cols=None, depvar="scope_increase"):
    """Run OLS(1,2), FS(1,2), IV(1,2), RF(1,2) for a given lagged treatment variable.

    ctrl_cols: override default CTRL_COLS (e.g. drop ln_gsp_pc_lag when treatment
               already embeds GSP in its denominator, to avoid exact collinearity).
    depvar: outcome variable for OLS/IV/RF (default: scope_increase).
    """
    cc = ctrl_cols if ctrl_cols is not None else CTRL_COLS
    r_ols1 = run_panel_ols(df_p, depvar, [treat_lag] + cc)
    r_ols2 = run_panel_ols(df_p, depvar, [treat_lag] + cc, other_effects=indyr_fe)
    r_fs1  = run_panel_ols(df_p, treat_lag,  [instr_lag] + cc)
    fs_F1  = float(r_fs1.tstats[instr_lag]) ** 2
    r_fs2  = run_panel_ols(df_p, treat_lag,  [instr_lag] + cc, other_effects=indyr_fe)
    fs_F2  = float(r_fs2.tstats[instr_lag]) ** 2
    r_iv1  = run_absorbing_iv(df, depvar, treat_lag, instr_lag, cc, absorb_yr)
    r_iv2  = run_absorbing_iv(df, depvar, treat_lag, instr_lag, cc, absorb_indyr)
    # Reduced form: regress outcome directly on instrument (no first stage)
    r_rf1  = run_panel_ols(df_p, depvar, [instr_lag] + cc)
    r_rf2  = run_panel_ols(df_p, depvar, [instr_lag] + cc, other_effects=indyr_fe)
    return r_ols1, r_ols2, r_fs1, r_fs2, r_iv1, r_iv2, r_rf1, r_rf2, fs_F1, fs_F2


def stars_tex(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""

def fmt_tex(res, var):
    if var not in res.params.index:
        return "", ""
    c, s, p = float(res.params[var]), float(res.std_errors[var]), float(res.pvalues[var])
    return f"{c:.3f}{stars_tex(p)}", f"({s:.3f})"

def tex_row(label, cells):
    coef_line = label + " & " + " & ".join(c[0] for c in cells) + " \\\\\n"
    se_line   = "       & " + " & ".join(c[1] for c in cells) + " \\\\\n"
    return coef_line + se_line

CTRL_LABELS = {
    "ln_sale_lag":      "ln(Sale)$_{t-1}$",
    "rd_intensity_lag": "R\\&D intensity$_{t-1}$",
    "roa_lag":          "ROA$_{t-1}$",
    "leverage_lag":     "Leverage$_{t-1}$",
    "cash_ratio_lag":   "Cash ratio$_{t-1}$",
    "ln_gsp_pc_lag":    "ln(GSP/capita)$_{t-1}$",
}
INSTR_LABELS = {
    "ln_z_pc_lag":      "ln(Z/pop)$_{t-1}$",
    "ln_z_pct_gsp_lag": "ln(Z/GSP)$_{t-1}$",
}


" Panel A: Treatment = ln(R&D / population), instrument = ln(Z / population) "
print("\n=== Panel A: Treatment = ln(R&D/pop)_{t-1}, IV = ln(Z/pop)_{t-1} ===")
res_A = run_all_specs("ln_rd_pc_lag", instr_lag="ln_z_pc_lag", depvar="scope_increase")
r_ols1, r_ols2, r_fs1, r_fs2, r_iv1, r_iv2, r_rf1, r_rf2, fs_F1, fs_F2 = res_A
for res, title in [
    (r_ols1, "OLS [Firm+Year FE]"),
    (r_fs1,  "First Stage [Firm+Year FE]"),
    (r_iv1,  "2SLS [Firm+Year FE]"),
    (r_rf1,  "Reduced Form [Firm+Year FE]"),
    (r_ols2, "OLS [Firm+Ind*Year FE]"),
    (r_fs2,  "First Stage [Firm+Ind*Year FE]"),
    (r_iv2,  "2SLS [Firm+Ind*Year FE]"),
    (r_rf2,  "Reduced Form [Firm+Ind*Year FE]"),
]:
    print_result(res, title)
print(f"  First-stage F — Spec 1: {fs_F1:.1f}   Spec 2: {fs_F2:.1f}")


" Panel B: Treatment = ln(R&D / GSP), instrument = ln(Z / GSP) "
# Drop ln_gsp_pc_lag from controls: GSP is already in the treatment denominator,
# so including ln_gsp_pc_lag would create exact collinearity (ln_rd_pc = ln_rd_pct_gsp + ln_gsp_pc).
CTRL_COLS_B = [c for c in CTRL_COLS if c != "ln_gsp_pc_lag"]
print("\n=== Panel B: Treatment = ln(R&D/%%GSP)_{t-1}, IV = ln(Z/GSP)_{t-1} ===")
res_B = run_all_specs("ln_rd_pct_gsp_lag", instr_lag="ln_z_pct_gsp_lag", ctrl_cols=CTRL_COLS_B, depvar="scope_increase")
r_ols1b, r_ols2b, r_fs1b, r_fs2b, r_iv1b, r_iv2b, r_rf1b, r_rf2b, fs_F1b, fs_F2b = res_B
for res, title in [
    (r_ols1b, "OLS [Firm+Year FE]"),
    (r_fs1b,  "First Stage [Firm+Year FE]"),
    (r_iv1b,  "2SLS [Firm+Year FE]"),
    (r_rf1b,  "Reduced Form [Firm+Year FE]"),
    (r_ols2b, "OLS [Firm+Ind*Year FE]"),
    (r_fs2b,  "First Stage [Firm+Ind*Year FE]"),
    (r_iv2b,  "2SLS [Firm+Ind*Year FE]"),
    (r_rf2b,  "Reduced Form [Firm+Ind*Year FE]"),
]:
    print_result(res, title)
print(f"  First-stage F — Spec 1: {fs_F1b:.1f}   Spec 2: {fs_F2b:.1f}")


""" Panel C: Over-identified IV — separate NSF and NIH instruments
    Exploits differential timing: NIH doubled 2000-2003; NSF was flat.
    States with different NSF vs NIH shares have differential within-state
    variation after two-way FEs — restores first-stage power.
    Hansen J-test checks instrument consistency.
"""
print("\n=== Panel C: Over-ID IV — Z_NSF + Z_NIH as separate instruments ===")

IV2_COLS = ["ln_z_nsf_lag", "ln_z_nih_lag"]

# First-stage strength of each agency separately
r_fs_nsf = run_panel_ols(df_p, "ln_rd_pc_lag", ["ln_z_nsf_lag"] + CTRL_COLS)
r_fs_nih = run_panel_ols(df_p, "ln_rd_pc_lag", ["ln_z_nih_lag"] + CTRL_COLS)
r_fs_ovid = run_panel_ols(df_p, "ln_rd_pc_lag", IV2_COLS + CTRL_COLS)
fs_F_nsf = float(r_fs_nsf.tstats["ln_z_nsf_lag"]) ** 2
fs_F_nih = float(r_fs_nih.tstats["ln_z_nih_lag"]) ** 2
print_result(r_fs_ovid, "First Stage (both) — ln(R&D/pop)_{t-1} ~ Z_NSF + Z_NIH  [Firm+Year FE]")
print(f"\n  F(NSF only): {fs_F_nsf:.1f}   F(NIH only): {fs_F_nih:.1f}")

r_iv_ovid1 = run_absorbing_iv(df, "scope_increase", "ln_rd_pc_lag", IV2_COLS, CTRL_COLS, absorb_yr)
r_iv_ovid2 = run_absorbing_iv(df, "scope_increase", "ln_rd_pc_lag", IV2_COLS, CTRL_COLS, absorb_indyr)
print_result(r_iv_ovid1, "2SLS (over-ID) — scope_increase ~ ln(R&D/pop)  [Firm+Year FE]")
print_result(r_iv_ovid2, "2SLS (over-ID) — scope_increase ~ ln(R&D/pop)  [Firm+Ind×Year FE]")
try:
    print(f"  Hansen J p-val (Spec 1): {r_iv_ovid1.j_statistic.pval:.3f}  "
          f"(Spec 2): {r_iv_ovid2.j_statistic.pval:.3f}  (p>0.1 → consistent)")
except Exception:
    pass


""" Panel D: Robustness — state linear trends + extended controls """
print("\n=== Panel D: Robustness — state linear trends + extended controls ===")

def partial_state_trends(df_in, cols):
    """Remove state-specific linear time trend from each column."""
    df_out = df_in.copy()
    year_f = df_in["year"].astype(float).values
    for col in cols:
        y   = df_in[col].values.astype(float)
        out = y.copy()
        for st in df_in["state"].unique():
            mask = (df_in["state"] == st).values
            x, yy = year_f[mask], y[mask]
            valid = np.isfinite(x) & np.isfinite(yy)
            if valid.sum() >= 3:
                coef = np.polyfit(x[valid], yy[valid], 1)
                out[mask] = yy - np.polyval(coef, x)
        df_out[col] = out
    return df_out

trend_cols = ["scope_increase", "ln_rd_pc_lag", "ln_z_pc_lag"] + CTRL_COLS_ROBUST
df_dt      = partial_state_trends(df, trend_cols)
df_dt_p    = df_dt.set_index(["gvkey", "year"])
cl_dt      = df_dt_p["state"]

r_ols_dt = PanelOLS(
    df_dt_p["scope_increase"], df_dt_p[["ln_rd_pc_lag"] + CTRL_COLS_ROBUST],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", clusters=cl_dt)

r_fs_dt = PanelOLS(
    df_dt_p["ln_rd_pc_lag"], df_dt_p[["ln_z_pc_lag"] + CTRL_COLS_ROBUST],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", clusters=cl_dt)
fs_F_dt = float(r_fs_dt.tstats["ln_z_pc_lag"]) ** 2

absorb_yr_dt = pd.DataFrame({"firm": pd.Categorical(df_dt["gvkey"]),
                              "year": pd.Categorical(df_dt["year"].astype(str))})
r_iv_dt = run_absorbing_iv(df_dt, "scope_increase", "ln_rd_pc_lag", "ln_z_pc_lag",
                           CTRL_COLS_ROBUST, absorb_yr_dt)

print_result(r_ols_dt, "OLS  [state-detrended + GSP growth, Firm+Year FE]")
print_result(r_fs_dt,  f"First Stage  [state-detrended]  F={fs_F_dt:.1f}")
print_result(r_iv_dt,  "2SLS  [state-detrended + GSP growth, Firm+Year FE]")


""" Panel E: Alternative FE structures — loosen FEs to recover Bartik variation
    The composite Z_{s,t} = theta_s * LOO_t loses nearly all within variation
    under two-way FE because year FE absorbs the national LOO_t component
    and corr(theta_NSF, theta_NIH) = 0.917 leaves little differential exposure.

    E-1: Aggregate to state-year; state + year FEs (proper Bartik aggregation level)
         Drops firm-level noise, uses full state-level R&D variation.
    E-2: Firm FE only, no year FE — lets national budget shocks in LOO_t
         contribute to identification (at cost of absorbing macro confounds).
    E-3: Over-identified at firm level, firm + year FEs but Z_NSF ⊥ Z_NIH as
         separate instruments — differential NIH doubling (2000-03) vs flat NSF.
"""
print("\n=== Panel E: Alternative FE structures ===")

# ── E-1: Aggregate to state-year ──────────────────────────────────────────────
# Use lagged firm controls averaged to state-year (one-period lag already baked in)
CTRL_SY = CTRL_COLS  # same names: ln_sale_lag etc — still valid as state-year means
_agg_cols = (["scope_increase", "ln_rd_pc", "ln_z_pc", "ln_z_nsf", "ln_z_nih"]
             + CTRL_SY)
df_sy = (df.groupby(["state", "year"])[_agg_cols].mean().reset_index().dropna().copy())
df_sy_p = df_sy.set_index(["state", "year"])
cl_sy   = pd.Series(df_sy["state"].values, name="state")

r_sy_ols = PanelOLS(
    df_sy_p["scope_increase"], df_sy_p[["ln_rd_pc"] + CTRL_SY],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", clusters=cl_sy)

r_sy_fs = PanelOLS(
    df_sy_p["ln_rd_pc"], df_sy_p[["ln_z_pc"] + CTRL_SY],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", clusters=cl_sy)
sy_F = float(r_sy_fs.tstats["ln_z_pc"]) ** 2

r_sy_fs_ovid = PanelOLS(
    df_sy_p["ln_rd_pc"], df_sy_p[["ln_z_nsf", "ln_z_nih"] + CTRL_SY],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", clusters=cl_sy)
sy_F_nsf = float(r_sy_fs_ovid.tstats["ln_z_nsf"]) ** 2
sy_F_nih = float(r_sy_fs_ovid.tstats["ln_z_nih"]) ** 2

# FWL-IV at state-year level
absorb_sy   = pd.DataFrame({"state": pd.Categorical(df_sy["state"]),
                             "year":  pd.Categorical(df_sy["year"].astype(str))})
cl_abs_sy   = df_sy["state"].astype("category")

def _resid_sy(col):
    ones = pd.Series(np.ones(len(df_sy)), index=df_sy.index, name="_c")
    r = AbsorbingLS(df_sy[col], ones, absorb=absorb_sy).fit()
    return pd.Series(r.resids.values.ravel(), index=df_sy.index, name=col)

y_sy  = _resid_sy("scope_increase")
x_sy  = _resid_sy("ln_rd_pc")
z_sy  = _resid_sy("ln_z_pc")
C_sy  = pd.concat([_resid_sy(c) for c in CTRL_SY], axis=1)
r_sy_iv = IV2SLS(y_sy, C_sy, x_sy.to_frame(), z_sy.to_frame()
                 ).fit(cov_type="clustered", clusters=cl_abs_sy)

zn_sy  = _resid_sy("ln_z_nsf")
zh_sy  = _resid_sy("ln_z_nih")
Z2_sy  = pd.concat([zn_sy, zh_sy], axis=1)
r_sy_iv_ovid = IV2SLS(y_sy, C_sy, x_sy.to_frame(), Z2_sy
                      ).fit(cov_type="clustered", clusters=cl_abs_sy)

print_result(r_sy_ols,      "E-1 OLS   [State+Year FE, state-year level]")
print_result(r_sy_fs,       f"E-1 FS    [State+Year FE]  F={sy_F:.1f}")
print_result(r_sy_iv,       "E-1 2SLS  [State+Year FE, just-ID composite Z]")
print_result(r_sy_fs_ovid,  f"E-1 FS    [State+Year FE, NSF+NIH]  F_NSF={sy_F_nsf:.1f}  F_NIH={sy_F_nih:.1f}")
print_result(r_sy_iv_ovid,  "E-1 2SLS  [State+Year FE, over-ID NSF+NIH]")
try:
    print(f"  Hansen J p-val (E-1 over-ID): {r_sy_iv_ovid.j_statistic.pval:.3f}")
except Exception:
    pass

# ── E-2: Firm FE only, no year FE ─────────────────────────────────────────────
r_noy_ols = PanelOLS(
    df_p["scope_increase"], df_p[["ln_rd_pc_lag"] + CTRL_COLS],
    entity_effects=True, time_effects=False,
).fit(cov_type="clustered", clusters=clusters_p)

r_noy_fs = PanelOLS(
    df_p["ln_rd_pc_lag"], df_p[["ln_z_pc_lag"] + CTRL_COLS],
    entity_effects=True, time_effects=False,
).fit(cov_type="clustered", clusters=clusters_p)
noy_F = float(r_noy_fs.tstats["ln_z_pc_lag"]) ** 2

absorb_firm = pd.DataFrame({"firm": pd.Categorical(df["gvkey"])})
r_noy_iv = run_absorbing_iv(df, "scope_increase", "ln_rd_pc_lag", "ln_z_pc_lag",
                             CTRL_COLS, absorb_firm)

print_result(r_noy_ols, "E-2 OLS  [Firm FE only — no Year FE]")
print_result(r_noy_fs,  f"E-2 FS   [Firm FE only]  F={noy_F:.1f}")
print_result(r_noy_iv,  "E-2 2SLS [Firm FE only]")

print(f"\n  Summary — first-stage F statistics:")
print(f"  Panel A (Firm+Year FE):           {fs_F1:.1f}")
print(f"  Panel E-1 (State+Year, just-ID):  {sy_F:.1f}")
print(f"  Panel E-1 (State+Year, NSF+NIH):  F_NSF={sy_F_nsf:.1f}  F_NIH={sy_F_nih:.1f}")
print(f"  Panel E-2 (Firm FE only):         {noy_F:.1f}")


" Save coefficient tables "
coef_tables = pd.concat([
    to_coef_df(r_ols1,  "A_ols_firm_year"),
    to_coef_df(r_fs1,   "A_fs_firm_year"),
    to_coef_df(r_iv1,   "A_iv_firm_year"),
    to_coef_df(r_rf1,   "A_rf_firm_year"),
    to_coef_df(r_ols2,  "A_ols_firm_indyr"),
    to_coef_df(r_fs2,   "A_fs_firm_indyr"),
    to_coef_df(r_iv2,   "A_iv_firm_indyr"),
    to_coef_df(r_rf2,   "A_rf_firm_indyr"),
    to_coef_df(r_ols1b, "B_ols_firm_year"),
    to_coef_df(r_fs1b,  "B_fs_firm_year"),
    to_coef_df(r_iv1b,  "B_iv_firm_year"),
    to_coef_df(r_rf1b,  "B_rf_firm_year"),
    to_coef_df(r_ols2b, "B_ols_firm_indyr"),
    to_coef_df(r_fs2b,  "B_fs_firm_indyr"),
    to_coef_df(r_iv2b,  "B_iv_firm_indyr"),
    to_coef_df(r_rf2b,  "B_rf_firm_indyr"),
], ignore_index=True)

out_csv = os.path.join(TABLES_DIR, "reg_scope_funding.csv")
coef_tables.to_csv(out_csv, index=False)
print(f"\nCoefficient table -> {out_csv}")

print("\nKey treatment coefficients:")
key = coef_tables[coef_tables["variable"].isin(["ln_rd_pc_lag", "ln_rd_pct_gsp_lag"])]
print(key[["spec", "variable", "coef", "se", "tstat", "pval"]].to_string(index=False))


def write_reg_tex(res_tuple, treat_var, instr_var, treat_tex, dv_fs_tex, tex_path):
    """Write 8-column regression table: OLS(1,2), FS(3,4), 2SLS(5,6), RF(7,8)."""
    r_ols1, r_ols2, r_fs1, r_fs2, r_iv1, r_iv2, r_rf1, r_rf2, fsF1, fsF2 = res_tuple
    RES = [r_ols1, r_ols2, r_fs1, r_fs2, r_iv1, r_iv2, r_rf1, r_rf2]
    instr_tex = INSTR_LABELS.get(instr_var, f"{instr_var}")
    ROW_VARS = {treat_var: treat_tex, instr_var: instr_tex, **CTRL_LABELS}
    with open(tex_path, "w") as fh:
        fh.write(
            f"% {os.path.basename(tex_path)} -- auto-generated\n"
            "\\begin{tabular}{lcccccccc}\n"
            "\\toprule\n"
            " & \\multicolumn{2}{c}{OLS} & \\multicolumn{2}{c}{First Stage}"
            " & \\multicolumn{2}{c}{2SLS} & \\multicolumn{2}{c}{Reduced Form} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            " & (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8) \\\\\n"
            f"\\textit{{Dep. var.}}"
            f" & $\\mathbf{{1}}$[ScopeInc] & $\\mathbf{{1}}$[ScopeInc]"
            f" & {dv_fs_tex} & {dv_fs_tex}"
            f" & $\\mathbf{{1}}$[ScopeInc] & $\\mathbf{{1}}$[ScopeInc]"
            f" & $\\mathbf{{1}}$[ScopeInc] & $\\mathbf{{1}}$[ScopeInc] \\\\\n"
            "\\midrule\n"
        )
        for var, label in ROW_VARS.items():
            fh.write(tex_row(label, [fmt_tex(r, var) for r in RES]))
        fh.write(
            "\\midrule\n"
            f"Observations  & {int(r_ols1.nobs):,} & {int(r_ols2.nobs):,}"
            f" & {int(r_fs1.nobs):,} & {int(r_fs2.nobs):,}"
            f" & {int(r_iv1.nobs):,} & {int(r_iv2.nobs):,}"
            f" & {int(r_rf1.nobs):,} & {int(r_rf2.nobs):,} \\\\\n"
            f"Within-R$^2$  & {r_ols1.rsquared:.3f} & {r_ols2.rsquared:.3f}"
            f" & {r_fs1.rsquared:.3f} & {r_fs2.rsquared:.3f} & &"
            f" & {r_rf1.rsquared:.3f} & {r_rf2.rsquared:.3f} \\\\\n"
            f"First-stage $F$ & & & & & {fsF1:.1f} & {fsF2:.1f} & & \\\\\n"
            "Firm FE & \\checkmark & \\checkmark & \\checkmark & \\checkmark"
            " & \\checkmark & \\checkmark & \\checkmark & \\checkmark \\\\\n"
            "Year FE & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark & \\\\\n"
            "Industry $\\times$ Year FE & & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark \\\\\n"
            "Clustered SE & State & State & State & State & State & State & State & State \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )
    print(f"LaTeX table -> {tex_path}")


write_reg_tex(
    res_A, "ln_rd_pc_lag", "ln_z_pc_lag",
    "ln(R\\&D/pop)$_{t-1}$",
    "ln(R\\&D/pop)$_{t-1}$",
    os.path.join(TABLES_DIR, "reg_scope_rdpc.tex"),
)
write_reg_tex(
    res_B, "ln_rd_pct_gsp_lag", "ln_z_pct_gsp_lag",
    "ln(R\\&D/\\%GSP)$_{t-1}$",
    "ln(R\\&D/\\%GSP)$_{t-1}$",
    os.path.join(TABLES_DIR, "reg_scope_rdpctgsp.tex"),
)


""" Segment-based scope: alternative outcome variables
    Panel C: Y = n_segments    (number of reported business segments)
    Panel D: Y = seg_diversity (Berry index = 1 - HHI of revenue shares)
    Treatment = ln_rd_pc_lag;  same IV and controls as Panel A.
    Firms missing segment data are dropped (left-join, then dropna on Y).
"""

SEG_OUTCOMES = [
    ("n_segments",   "Num.\\ segments",    "reg_scope_nseg.tex"),
    ("seg_diversity","Seg.\\ diversity",   "reg_scope_segdiv.tex"),
]

for seg_dep, seg_tex, seg_fname in SEG_OUTCOMES:
    seg_avail = df[seg_dep].notna().sum()
    print(f"\n=== Segment outcome: {seg_dep}  ({seg_avail:,} non-missing obs) ===")
    if seg_avail < 100:
        print(f"  Skipping {seg_dep}: too few observations (segment data not yet imported).")
        continue

    # Build sub-panel: drop rows where segment outcome is missing
    df_seg = df.dropna(subset=[seg_dep]).copy().reset_index(drop=True)
    df_seg_p = df_seg.set_index(["gvkey", "year"])
    cl_seg   = df_seg_p["state"]

    absorb_yr_seg    = pd.DataFrame({"firm": pd.Categorical(df_seg["gvkey"]),
                                     "year": pd.Categorical(df_seg["year"].astype(str))})
    absorb_indyr_seg = pd.DataFrame({"firm":  pd.Categorical(df_seg["gvkey"]),
                                     "indyr": pd.Categorical(df_seg["naics2"] + "_" + df_seg["year"].astype(str))})
    cl_abs_seg = df_seg["state"].astype("category")
    indyr_seg  = pd.DataFrame(
        {"indyr": pd.Categorical(df_seg["naics2"] + "_" + df_seg["year"].astype(str))},
        index=df_seg_p.index,
    )

    def _run_seg_ols(dep, exog_cols, other_effects=None):
        oe = {"other_effects": other_effects} if other_effects is not None else {}
        return PanelOLS(df_seg_p[dep], df_seg_p[exog_cols],
                        entity_effects=True,
                        time_effects=(other_effects is None),
                        **oe).fit(cov_type="clustered", clusters=cl_seg)

    def _run_seg_iv(dep, endog_col, instr_col, exog_cols, absorb_df, cl_abs):
        def _resid(col):
            ones = pd.Series(np.ones(len(df_seg)), index=df_seg.index, name="_c")
            r = AbsorbingLS(df_seg[col], ones, absorb=absorb_df).fit()
            return pd.Series(r.resids.values.ravel(), index=df_seg.index, name=col)
        y_r = _resid(dep);  x_r = _resid(endog_col);  z_r = _resid(instr_col)
        C_r = pd.concat([_resid(c) for c in exog_cols], axis=1)
        return IV2SLS(y_r, C_r, x_r.to_frame(), z_r.to_frame()
                      ).fit(cov_type="clustered", clusters=cl_abs)

    treat, instr = "ln_rd_pc_lag", "ln_z_pc_lag"
    cc = CTRL_COLS

    r_s_ols1 = _run_seg_ols(seg_dep, [treat] + cc)
    r_s_ols2 = _run_seg_ols(seg_dep, [treat] + cc, other_effects=indyr_seg)
    r_s_fs1  = _run_seg_ols(treat,   [instr] + cc)
    r_s_fs2  = _run_seg_ols(treat,   [instr] + cc, other_effects=indyr_seg)
    fsF_s1   = float(r_s_fs1.tstats[instr]) ** 2
    fsF_s2   = float(r_s_fs2.tstats[instr]) ** 2
    r_s_iv1  = _run_seg_iv(seg_dep, treat, instr, cc, absorb_yr_seg,    cl_abs_seg)
    r_s_iv2  = _run_seg_iv(seg_dep, treat, instr, cc, absorb_indyr_seg, cl_abs_seg)
    r_s_rf1  = _run_seg_ols(seg_dep, [instr] + cc)
    r_s_rf2  = _run_seg_ols(seg_dep, [instr] + cc, other_effects=indyr_seg)

    for res, title in [
        (r_s_ols1, f"{seg_dep} OLS [Firm+Year FE]"),
        (r_s_fs1,  f"{seg_dep} First Stage [Firm+Year FE]"),
        (r_s_iv1,  f"{seg_dep} 2SLS [Firm+Year FE]"),
        (r_s_rf1,  f"{seg_dep} Reduced Form [Firm+Year FE]"),
    ]:
        print_result(res, title)
    print(f"  First-stage F — Spec 1: {fsF_s1:.1f}   Spec 2: {fsF_s2:.1f}")

    # LaTeX table
    res_seg = (r_s_ols1, r_s_ols2, r_s_fs1, r_s_fs2,
               r_s_iv1, r_s_iv2, r_s_rf1, r_s_rf2, fsF_s1, fsF_s2)
    write_reg_tex(
        res_seg, treat, instr,
        "ln(R\\&D/pop)$_{t-1}$",
        "ln(R\\&D/pop)$_{t-1}$",
        os.path.join(TABLES_DIR, seg_fname),
    )


""" Local Projections: OLS impulse response at horizons h = 1 to 5
    For each h, estimate:
        scope_increase_{i,t+h} = beta_h * ln(rd_pc)_{i,t} + X_{i,t} + alpha_i + gamma_t + e
    Treatment is ln_rd_pc at year t (not lagged); outcomes are h-year-ahead scope_increase.
    Spec 1: Firm + Year FE  |  Spec 2: Firm + Industry×Year FE
"""
HORIZONS = range(1, 6)

# LP controls are contemporaneous (treatment and controls both at time t)
LP_CTRL_COLS = ["ln_sale", "rd_intensity", "roa", "leverage", "cash_ratio", "ln_gsp_pc"]

# Create cumulative scope increase at each horizon: 1[d2vscope_{t+h} > d2vscope_t]
# This is the LP analog of a cumulative impulse response — did scope end up higher
# at t+h than at t, rather than whether it happened to increase in that specific year.
df_lp = df.copy()
for h in HORIZONS:
    df_lp[f"scope_fwd_{h}"] = (
        df_lp.groupby("gvkey")["d2vscope"].shift(-h) > df_lp["d2vscope"]
    ).astype(float)

lp_res1 = {}   # Spec 1: Firm + Year FE
lp_res2 = {}   # Spec 2: Firm + Ind×Year FE

print("\n=== Local Projections: Scope ~ ln(R&D/pop)_t at horizon h ===")
print(f"  {'h':>2}  {'β (Firm+Year)':>14}  {'SE':>7}  {'95% CI':>22}  "
      f"{'β (Firm+IndYr)':>15}  {'SE':>7}  {'95% CI':>22}  N")
print("  " + "-" * 105)

for h in HORIZONS:
    dep = f"scope_fwd_{h}"
    needed = [dep, "ln_rd_pc"] + LP_CTRL_COLS + ["naics2", "state"]
    df_h = df_lp.dropna(subset=needed).copy()
    df_h_p = df_h.set_index(["gvkey", "year"])
    cl_h   = df_h_p["state"]

    # Spec 1: Firm + Year FE
    r1 = PanelOLS(
        df_h_p[dep], df_h_p[["ln_rd_pc"] + LP_CTRL_COLS],
        entity_effects=True, time_effects=True,
    ).fit(cov_type="clustered", clusters=cl_h)
    lp_res1[h] = r1

    # Spec 2: Firm + Industry×Year FE
    indyr_h = pd.DataFrame(
        {"indyr": pd.Categorical(df_h["naics2"] + "_" + df_h["year"].astype(str))},
        index=df_h_p.index,
    )
    r2 = PanelOLS(
        df_h_p[dep], df_h_p[["ln_rd_pc"] + LP_CTRL_COLS],
        entity_effects=True, time_effects=False, other_effects=indyr_h,
    ).fit(cov_type="clustered", clusters=cl_h)
    lp_res2[h] = r2

    def _fmt(res):
        c  = res.params["ln_rd_pc"]
        s  = res.std_errors["ln_rd_pc"]
        p  = res.pvalues["ln_rd_pc"]
        ci = res.conf_int()
        lo, hi = ci.loc["ln_rd_pc", "lower"], ci.loc["ln_rd_pc", "upper"]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "  "
        return f"{c:.4f}{sig}", f"{s:.4f}", f"[{lo:.4f},{hi:.4f}]"

    c1, s1, ci1 = _fmt(r1)
    c2, s2, ci2 = _fmt(r2)
    print(f"  {h:>2}  {c1:>14}  {s1:>7}  {ci1:>22}  {c2:>15}  {s2:>7}  {ci2:>22}  {int(r1.nobs):,}")


" Plot impulse response (both specs) "
fig, ax = plt.subplots(figsize=(7, 4))
hs = list(HORIZONS)

for lp_res, label, color, ls in [
    (lp_res1, "Firm + Year FE",         "steelblue",  "-"),
    (lp_res2, "Firm + Ind×Year FE",     "darkorange", "--"),
]:
    coefs = [lp_res[h].params["ln_rd_pc"]                       for h in hs]
    lo95  = [lp_res[h].conf_int().loc["ln_rd_pc", "lower"]      for h in hs]
    hi95  = [lp_res[h].conf_int().loc["ln_rd_pc", "upper"]      for h in hs]
    ax.plot(hs, coefs, f"o{ls}", color=color, label=label, linewidth=1.8, markersize=5)
    ax.fill_between(hs, lo95, hi95, alpha=0.12, color=color)

ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax.set_xlabel("Horizon h (years)")
ax.set_ylabel(r"$\beta_h$: effect of $\ln$(R\&D/pop)$_t$ on Pr(scope$_{t+h}$ > scope$_t$)")
ax.set_title("Local Projection — R&D Funding → Pr(Scope Increase) (OLS)")
ax.set_xticks(hs)
ax.legend(frameon=False)
fig.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "lp_scope_funding.pdf")
fig.savefig(fig_path)
import os; os.startfile(fig_path)  # open in default PDF viewer (Windows)


" LaTeX table for local projections "
lp_tex = os.path.join(TABLES_DIR, "lp_scope_funding.tex")
with open(lp_tex, "w") as fh:
    fh.write(
        "% lp_scope_funding.tex -- auto-generated by analyses/scope_funding_reg.py\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        " & \\multicolumn{5}{c}{Horizon $h$ (years ahead)} \\\\\n"
        "\\cmidrule(lr){2-6}\n"
        "Dep.\\ var: $\\mathbf{1}$[scope$_{t+h}>$scope$_t$] & $h=1$ & $h=2$ & $h=3$ & $h=4$ & $h=5$ \\\\\n"
        "\\midrule\n"
    )
    for spec_label, lp_res in [("\\textit{Firm + Year FE}", lp_res1),
                                ("\\textit{Firm + Ind$\\times$Year FE}", lp_res2)]:
        fh.write(f"\\multicolumn{{6}}{{l}}{{{spec_label}}} \\\\\n")
        coef_line = "\\quad ln(R\\&D/pop)$_t$"
        se_line   = "       "
        for h in hs:
            c1, s1, _ = _fmt(lp_res[h])
            coef_line += f" & {c1}"
            se_line   += f" & ({s1})"
        fh.write(coef_line + " \\\\\n")
        fh.write(se_line   + " \\\\\n")
        obs_line = "\\quad $N$"
        for h in hs:
            obs_line += f" & {int(lp_res[h].nobs):,}"
        fh.write(obs_line + " \\\\\n")
        fh.write("\\addlinespace\n")
    fh.write(
        "\\midrule\n"
        "Controls & \\checkmark & \\checkmark & \\checkmark & \\checkmark & \\checkmark \\\\\n"
        "Firm FE  & \\checkmark & \\checkmark & \\checkmark & \\checkmark & \\checkmark \\\\\n"
        "Clustered SE & \\multicolumn{5}{c}{State} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
print(f"Local projection LaTeX -> {lp_tex}")
