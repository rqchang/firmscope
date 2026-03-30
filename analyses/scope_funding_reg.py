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

Data inputs:
    data/raw/scope/FirmScope.txt                      Hoberg-Phillips scope
    data/processed/Compustat/compustat_annual.csv     financials + state abbreviation
    data/processed/eip/funding_panel.csv              state-year R&D panel
    data/processed/eip/bartik_shares.csv              pre-determined Bartik shares
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

" Compustat annual financials + state "
comp = pd.read_csv(os.path.join(PROC_CS, "compustat_annual.csv"), dtype={"gvkey": str})
comp["gvkey"] = comp["gvkey"].str.zfill(6)
comp = comp.rename(columns={"fyear": "year"})
comp["year"] = comp["year"].astype("Int64")
comp = comp[["gvkey", "year", "sale", "at", "xrd", "oibdp", "fdat", "che", "sic", "naicsh_filled", "state"]].dropna(subset=["state"]).copy()

" State-year funding panel "
fund = pd.read_csv(os.path.join(PROC_EIP, "funding_panel.csv"))
fund = fund[fund["state"].apply(lambda x: str(x).isalpha() and len(str(x)) == 2)].copy()
fund = fund[fund["state"] != "DC"].copy()

" Bartik shares IV "
shares = pd.read_csv(os.path.join(PROC_EIP, "bartik_shares.csv"))


""" Construct regression data """
" Build state-year Bartik instrument "
nat = (
    fund.groupby("year")[["nsf_grants_usd", "nih_grants_usd"]]
    .sum()
    .rename(columns={"nsf_grants_usd": "nat_nsf", "nih_grants_usd": "nat_nih"})
    .reset_index()
)
state_yr = fund[["state", "year", "nsf_grants_usd", "nih_grants_usd",
                  "rd_funding_per_capita", "rd_funding_pct_gsp",
                  "gsp_millions", "population"]].copy()
state_yr = state_yr.merge(nat, on="year", how="left")
state_yr["loo_nsf"] = state_yr["nat_nsf"] - state_yr["nsf_grants_usd"].fillna(0)
state_yr["loo_nih"] = state_yr["nat_nih"] - state_yr["nih_grants_usd"].fillna(0)

shares_w = (
    shares.pivot(index="state", columns="agency", values="share")
    .rename(columns={"nsf": "share_nsf", "nih": "share_nih"})
    .reset_index()
)
state_yr = state_yr.merge(shares_w, on="state", how="left")
state_yr["bartik_z"] = (state_yr["share_nsf"] * state_yr["loo_nsf"] +
                        state_yr["share_nih"] * state_yr["loo_nih"])


" Merge to firm-year panel "
df = pd.merge(scope, comp, on=["gvkey", "year"], how="inner")
df = pd.merge(df, state_yr[["state", "year", "rd_funding_per_capita", "rd_funding_pct_gsp",
                            "bartik_z", "gsp_millions", "population"]],
                        on=["state", "year"], how="inner")

" Sample restrictions & variable construction "
df = df[df["year"].between(1997, 2021)].copy()   # scope: 1988-2021; funding: 1997-2024
df = df[df["rd_funding_per_capita"] > 0].copy()
df = df[df["bartik_z"] > 0].copy()
df = df[df["sale"] > 0].copy()
df = df[df["at"]   > 0].copy()

df["ln_rd_pc"]     = np.log(df["rd_funding_per_capita"])
df["ln_z"]         = np.log(df["bartik_z"])
df["ln_sale"]      = np.log(df["sale"])
df["rd_intensity"] = (df["xrd"].fillna(0) / df["sale"]).clip(0, 1)   # winsorized at 100%
df["roa"]          = (df["oibdp"] / df["at"]).clip(-1, 1)
df["leverage"]     = (df["fdat"]  / df["at"]).clip(0, 2)
df["cash_ratio"]   = (df["che"]   / df["at"]).clip(0, 1)
df["naics2"]       = df["naicsh_filled"].astype(str).str[:2]

# State-level economic control: ln(GSP per capita) absorbs state business cycles
df["ln_gsp_pc"]    = np.log(df["gsp_millions"] * 1e6 / df["population"].clip(lower=1))

# Lag funding and instrument by one year within firm (economic mechanism takes time)
df = df.sort_values(["gvkey", "year"])
df["ln_rd_pc_lag"] = df.groupby("gvkey")["ln_rd_pc"].shift(1)
df["ln_z_lag"]     = df.groupby("gvkey")["ln_z"].shift(1)

df = df.dropna(subset=["d2vscope", "ln_rd_pc_lag", "ln_z_lag", "ln_sale",
                        "rd_intensity", "roa", "leverage", "cash_ratio",
                        "ln_gsp_pc", "naics2"]).copy()
df = df.reset_index(drop=True)

print(
    f"Sample: {df['gvkey'].nunique():,} firms  "
    f"{df['state'].nunique()} states  "
    f"{df['year'].nunique()} years  "
    f"{len(df):,} firm-year obs"
)


""" Run regressions """
CTRL_COLS = ["ln_sale", "rd_intensity", "roa", "leverage", "cash_ratio", "ln_gsp_pc"]

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


def run_absorbing_iv(df_reg, depvar, endog_col, instrument_col, exog_cols, absorb_df):
    """
    2SLS with absorbed FEs, clustered by state.
    Partials out FEs via AbsorbingLS (OLS) then runs IV2SLS on residuals (FWL).
    Equivalent to full 2SLS with FE dummies; compatible with linearmodels < 0.60.
    """
    def _resid(col):
        ones = pd.Series(np.ones(len(df_reg)), index=df_reg.index, name="_c")
        r = AbsorbingLS(df_reg[col], ones, absorb=absorb_df).fit()
        return pd.Series(r.resids.values.ravel(), index=df_reg.index, name=col)

    y_r = _resid(depvar)
    x_r = _resid(endog_col)
    z_r = _resid(instrument_col)
    C_r = pd.concat([_resid(c) for c in exog_cols], axis=1)

    return IV2SLS(y_r, C_r, x_r.to_frame(), z_r.to_frame()
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


" Spec 1: Firm FE + Year FE (baseline) "
print("\n=== Spec 1: Firm FE + Year FE ===")

res_ols1 = run_panel_ols(df_p, "d2vscope", ["ln_rd_pc_lag"] + CTRL_COLS)
print_result(res_ols1, "OLS  —  d2vscope ~ ln(R&D/pop)_{t-1} + controls  [Firm + Year FE]")

res_fs1 = run_panel_ols(df_p, "ln_rd_pc_lag", ["ln_z_lag"] + CTRL_COLS)
fs_F1   = float(res_fs1.tstats["ln_z_lag"]) ** 2
print_result(res_fs1, "First Stage  —  ln(R&D/pop)_{t-1} ~ ln(Bartik Z)_{t-1} + controls  [Firm + Year FE]")
print(f"\n  First-stage F (KP approx): {fs_F1:.1f}")

res_iv1 = run_absorbing_iv(df, "d2vscope", "ln_rd_pc_lag", "ln_z_lag", CTRL_COLS, absorb_yr)
print_result(res_iv1, "2SLS  —  d2vscope ~ ln(R&D/pop)_{t-1} [IV: ln(Bartik Z)_{t-1}]  [Firm + Year FE]")
print(f"\n  Key: β = {res_iv1.params['ln_rd_pc_lag']:.4f}  "
      f"(t={res_iv1.tstats['ln_rd_pc_lag']:.2f})  First-stage F={fs_F1:.1f}")


" Spec 2: Firm FE + Industry × Year FE (preferred) "
print("\n=== Spec 2: Firm FE + Industry × Year FE ===")

res_ols2 = run_panel_ols(df_p, "d2vscope", ["ln_rd_pc_lag"] + CTRL_COLS, other_effects=indyr_fe)
print_result(res_ols2, "OLS  —  d2vscope ~ ln(R&D/pop)_{t-1} + controls  [Firm + Ind×Year FE]")

res_fs2 = run_panel_ols(df_p, "ln_rd_pc_lag", ["ln_z_lag"] + CTRL_COLS, other_effects=indyr_fe)
fs_F2   = float(res_fs2.tstats["ln_z_lag"]) ** 2
print_result(res_fs2, "First Stage  —  ln(R&D/pop)_{t-1} ~ ln(Bartik Z)_{t-1} + controls  [Firm + Ind×Year FE]")
print(f"\n  First-stage F (KP approx): {fs_F2:.1f}")

res_iv2 = run_absorbing_iv(df, "d2vscope", "ln_rd_pc_lag", "ln_z_lag", CTRL_COLS, absorb_indyr)
print_result(res_iv2, "2SLS  —  d2vscope ~ ln(R&D/pop)_{t-1} [IV: ln(Bartik Z)_{t-1}]  [Firm + Ind×Year FE]")
print(f"\n  Key: β = {res_iv2.params['ln_rd_pc_lag']:.4f}  "
      f"(t={res_iv2.tstats['ln_rd_pc_lag']:.2f})  First-stage F={fs_F2:.1f}")


" Save coefficient tables "
coef_tables = pd.concat([
    to_coef_df(res_ols1, "ols_firm_year"),
    to_coef_df(res_fs1,  "fs_firm_year"),
    to_coef_df(res_iv1,  "iv_firm_year"),
    to_coef_df(res_ols2, "ols_firm_indyr"),
    to_coef_df(res_fs2,  "fs_firm_indyr"),
    to_coef_df(res_iv2,  "iv_firm_indyr"),
], ignore_index=True)

out_csv = os.path.join(TABLES_DIR, "reg_scope_funding.csv")
coef_tables.to_csv(out_csv, index=False)
print(f"\nCoefficient table -> {out_csv}")

key_rows = coef_tables[coef_tables["variable"] == "ln_rd_pc_lag"]
print("\nKey coefficients (ln_rd_pc_lag):")
print(key_rows[["spec", "coef", "se", "tstat", "pval"]].to_string(index=False))


""" Local Projections: OLS impulse response at horizons h = 1 to 5
    For each h, estimate:
        d2vscope_{i,t+h} = beta_h * ln(rd_pc)_{i,t} + X_{i,t} + alpha_i + gamma_t + e
    Treatment is ln_rd_pc at year t (not lagged); outcomes are h-year-ahead scope.
    Spec 1: Firm + Year FE  |  Spec 2: Firm + Industry×Year FE
"""
HORIZONS = range(1, 6)

# Create forward leads of d2vscope within firm
df_lp = df.copy()
for h in HORIZONS:
    df_lp[f"scope_fwd_{h}"] = df_lp.groupby("gvkey")["d2vscope"].shift(-h)

lp_res1 = {}   # Spec 1: Firm + Year FE
lp_res2 = {}   # Spec 2: Firm + Ind×Year FE

print("\n=== Local Projections: Scope ~ ln(R&D/pop)_t at horizon h ===")
print(f"  {'h':>2}  {'β (Firm+Year)':>14}  {'SE':>7}  {'95% CI':>22}  "
      f"{'β (Firm+IndYr)':>15}  {'SE':>7}  {'95% CI':>22}  N")
print("  " + "-" * 105)

for h in HORIZONS:
    dep = f"scope_fwd_{h}"
    needed = [dep, "ln_rd_pc"] + CTRL_COLS + ["naics2", "state"]
    df_h = df_lp.dropna(subset=needed).copy()
    df_h_p = df_h.set_index(["gvkey", "year"])
    cl_h   = df_h_p["state"]

    # Spec 1: Firm + Year FE
    r1 = PanelOLS(
        df_h_p[dep], df_h_p[["ln_rd_pc"] + CTRL_COLS],
        entity_effects=True, time_effects=True,
    ).fit(cov_type="clustered", clusters=cl_h)
    lp_res1[h] = r1

    # Spec 2: Firm + Industry×Year FE
    indyr_h = pd.DataFrame(
        {"indyr": pd.Categorical(df_h["naics2"] + "_" + df_h["year"].astype(str))},
        index=df_h_p.index,
    )
    r2 = PanelOLS(
        df_h_p[dep], df_h_p[["ln_rd_pc"] + CTRL_COLS],
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
ax.set_ylabel(r"$\beta_h$: effect of $\ln$(R\&D/pop)$_t$ on d2vscope$_{t+h}$")
ax.set_title("Local Projection — R&D Funding → Firm Scope (OLS)")
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
        "Dep.\\ var: d2vscope$_{t+h}$ & $h=1$ & $h=2$ & $h=3$ & $h=4$ & $h=5$ \\\\\n"
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


" LaTeX table: OLS(1) OLS(2) | FS(1) FS(2) | 2SLS(1) 2SLS(2) — full coefficients "
def stars_tex(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""

def fmt_tex(res, var):
    """Return (coef_str, se_str); empty strings if var not in result."""
    if var not in res.params.index:
        return "", ""
    c, s, p = float(res.params[var]), float(res.std_errors[var]), float(res.pvalues[var])
    return f"{c:.3f}{stars_tex(p)}", f"({s:.3f})"

def tex_row(label, cells):
    """Two-line LaTeX row: coefficient line then SE line."""
    coef_line = label + " & " + " & ".join(c[0] for c in cells) + " \\\\\n"
    se_line   = "       & " + " & ".join(c[1] for c in cells) + " \\\\\n"
    return coef_line + se_line

# All six result objects in column order
RES = [res_ols1, res_ols2, res_fs1, res_fs2, res_iv1, res_iv2]

# Variable → display label mapping (in desired row order)
ROW_VARS = {
    "ln_rd_pc_lag":  "ln(R\\&D/pop)$_{t-1}$",
    "ln_z_lag":      "ln(Bartik Z)$_{t-1}$",
    "ln_sale":       "ln(Sale)",
    "rd_intensity":  "R\\&D intensity",
    "roa":           "ROA",
    "leverage":      "Leverage",
    "cash_ratio":    "Cash ratio",
    "ln_gsp_pc":     "ln(GSP/capita)",
}

out_tex = os.path.join(TABLES_DIR, "reg_scope_funding.tex")
with open(out_tex, "w") as fh:
    fh.write(
        "% reg_scope_funding.tex -- auto-generated by analyses/scope_funding_reg.py\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{OLS} & \\multicolumn{2}{c}{First Stage}"
        " & \\multicolumn{2}{c}{2SLS} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        " & (1) & (2) & (3) & (4) & (5) & (6) \\\\\n"
        "\\textit{Dep. var.}"
        " & d2vscope & d2vscope"
        " & ln(R\\&D/pop)$_{t-1}$ & ln(R\\&D/pop)$_{t-1}$"
        " & d2vscope & d2vscope \\\\\n"
        "\\midrule\n"
    )
    for var, label in ROW_VARS.items():
        fh.write(tex_row(label, [fmt_tex(r, var) for r in RES]))
    fh.write(
        "\\midrule\n"
        f"Observations  & {int(res_ols1.nobs):,} & {int(res_ols2.nobs):,}"
        f" & {int(res_fs1.nobs):,} & {int(res_fs2.nobs):,}"
        f" & {int(res_iv1.nobs):,} & {int(res_iv2.nobs):,} \\\\\n"
        f"Within-R$^2$  & {res_ols1.rsquared:.3f} & {res_ols2.rsquared:.3f}"
        f" & {res_fs1.rsquared:.3f} & {res_fs2.rsquared:.3f} & & \\\\\n"
        f"First-stage $F$ & & & & & {fs_F1:.1f} & {fs_F2:.1f} \\\\\n"
        "Firm FE & \\checkmark & \\checkmark & \\checkmark & \\checkmark"
        " & \\checkmark & \\checkmark \\\\\n"
        "Year FE & \\checkmark & & \\checkmark & & \\checkmark & \\\\\n"
        "Industry $\\times$ Year FE & & \\checkmark & & \\checkmark & & \\checkmark \\\\\n"
        "Clustered SE & State & State & State & State & State & State \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
print(f"LaTeX table -> {out_tex}")
