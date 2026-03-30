"""
scope_funding_reg.py
--------------------
OLS and IV (Bartik shift-share) regressions of firm scope on state R&D funding.

Specification (firm i, state s, year t):
    d2vscope_{i,s,t} = beta * ln(rd_pc_{s,t}) + alpha_i + gamma_t + X_{i,t} + e

    X_{i,t}: ln(sale), ln(at), rd_active indicator

IV  Bartik shift-share [Goldsmith-Pinkham et al. 2020, leave-one-out]
    Z_{s,t} = sum_a  theta_{s,a} * B_{a,t}^{-s}
    theta_{s,a}       state s pre-determined share of national agency-a funding
    B_{a,t}^{-s}      national agency-a budget in year t, excluding state s

FEs:  firm (gvkey) + year
SEs:  clustered by state (treatment variation is at state-year level)

Data inputs:
    data/raw/scope/FirmScope.txt                   Hoberg-Phillips scope
    data/processed/Compustat/compustat_annual.rds  financials (via pyreadr)
    data/processed/Compustat/gvkey_state.csv       gvkey->stateabbr crosswalk
        (generate via data_import/compustat/import_comp_state.R)
    data/processed/eip/funding_panel.csv           state-year R&D panel
    data/processed/eip/bartik_shares.csv           pre-determined Bartik shares
"""

import sys
import warnings
import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyreadr")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.set_paths import RAW_DIR, PROC_DIR, TABLES_DIR

RAW_SCOPE = Path(RAW_DIR)  / "scope"
PROC_EIP  = Path(PROC_DIR) / "eip"
PROC_CS   = Path(PROC_DIR) / "Compustat"


# ── 1. Scope data (Hoberg-Phillips d2vscope) ──────────────────────────────────
scope = pd.read_csv(RAW_SCOPE / "FirmScope.txt", sep="\t")
scope["gvkey"] = scope["gvkey"].astype(str).str.zfill(6)
# year == first 4 digits of Compustat datadate (fiscal year end year)

# ── 2. Compustat annual financials ────────────────────────────────────────────
_rds  = pyreadr.read_r(str(PROC_CS / "compustat_annual.rds"))
comp  = list(_rds.values())[0]
comp["gvkey"] = comp["gvkey"].astype(str).str.zfill(6)
comp  = comp.rename(columns={"fyear": "year"})
comp["year"] = comp["year"].astype("Int64")
comp  = comp[["gvkey", "year", "sale", "at", "xrd", "sic"]].copy()

# ── 3. gvkey -> state crosswalk ───────────────────────────────────────────────
state_path = PROC_CS / "gvkey_state.csv"
if not state_path.exists():
    raise FileNotFoundError(
        f"{state_path} not found.\n"
        "Run data_import/compustat/import_comp_state.R to pull from WRDS."
    )
gvkey_state = pd.read_csv(state_path, dtype={"gvkey": str})
gvkey_state["gvkey"] = gvkey_state["gvkey"].str.zfill(6)
gvkey_state = gvkey_state[gvkey_state["loc"] == "USA"][["gvkey", "stateabbr"]].dropna()
gvkey_state = gvkey_state.rename(columns={"stateabbr": "state"})

# ── 4. State-year funding panel ───────────────────────────────────────────────
fund = pd.read_csv(PROC_EIP / "funding_panel.csv")
fund = fund[fund["state"].apply(lambda x: str(x).isalpha() and len(str(x)) == 2)].copy()
fund = fund[fund["state"] != "DC"].copy()

# ── 5. Bartik shares ──────────────────────────────────────────────────────────
shares = pd.read_csv(PROC_EIP / "bartik_shares.csv")


# ── 6. Build state-year Bartik instrument ─────────────────────────────────────
nat = (
    fund.groupby("year")[["nsf_grants_usd", "nih_grants_usd"]]
    .sum()
    .rename(columns={"nsf_grants_usd": "nat_nsf", "nih_grants_usd": "nat_nih"})
    .reset_index()
)
state_yr = fund[["state", "year", "nsf_grants_usd", "nih_grants_usd",
                  "rd_funding_per_capita", "rd_funding_pct_gsp"]].copy()
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


# ── 7. Merge to firm-year panel ───────────────────────────────────────────────
df = scope.merge(comp, on=["gvkey", "year"], how="inner")
df = df.merge(gvkey_state, on="gvkey", how="inner")
df = df.merge(
    state_yr[["state", "year", "rd_funding_per_capita", "rd_funding_pct_gsp", "bartik_z"]],
    on=["state", "year"],
    how="inner",
)

# ── 8. Sample restrictions & variable construction ────────────────────────────
df = df[df["year"].between(1997, 2021)].copy()   # scope: 1988-2021; funding: 1997-2024
df = df[df["rd_funding_per_capita"] > 0].copy()
df = df[df["bartik_z"] > 0].copy()
df = df[df["sale"] > 0].copy()
df = df[df["at"]   > 0].copy()

df["ln_rd_pc"]  = np.log(df["rd_funding_per_capita"])
df["ln_z"]      = np.log(df["bartik_z"])
df["ln_sale"]   = np.log(df["sale"])
df["ln_at"]     = np.log(df["at"])
df["rd_active"] = (df["xrd"].notna() & (df["xrd"] > 0)).astype(float)

df = df.dropna(subset=["d2vscope", "ln_rd_pc", "ln_z", "ln_sale", "ln_at"]).copy()
df = df.reset_index(drop=True)

print(
    f"Sample: {df['gvkey'].nunique():,} firms  "
    f"{df['state'].nunique()} states  "
    f"{df['year'].nunique()} years  "
    f"{len(df):,} firm-year obs"
)

# ── 9. Regression setup ───────────────────────────────────────────────────────
firm_fe  = pd.get_dummies(df["gvkey"], prefix="f", drop_first=True)
year_fe  = pd.get_dummies(df["year"],  prefix="y", drop_first=True)
FE       = pd.concat([firm_fe, year_fe], axis=1).astype(float).values
controls = np.column_stack([df["ln_sale"].values, df["ln_at"].values,
                            df["rd_active"].values])

y      = df["d2vscope"].values.astype(float)
x      = df["ln_rd_pc"].values
z      = df["ln_z"].values
groups = df["state"].values     # cluster by state (treatment is state-level)


def _Xmat(*regressors) -> np.ndarray:
    """Stack [const | regressors | controls | firm FEs | year FEs]."""
    return sm.add_constant(np.column_stack([*regressors, controls, FE]))


def _stars(t: float) -> str:
    a = abs(t)
    return "***" if a > 3.29 else ("**" if a > 1.96 else ("*" if a > 1.645 else ""))


def _clust_vcov_iv(X_proj: np.ndarray, resid: np.ndarray,
                   cl: np.ndarray) -> np.ndarray:
    """
    2SLS sandwich variance with small-sample correction:
        V = (Xh'Xh)^-1 [sum_c Xh_c' e_c e_c' Xh_c] (Xh'Xh)^-1
    """
    n, k     = X_proj.shape
    G        = len(np.unique(cl))
    XhXh_inv = np.linalg.inv(X_proj.T @ X_proj)
    meat     = np.zeros((k, k))
    for c in np.unique(cl):
        idx = cl == c
        Xc, ec = X_proj[idx], resid[idx]
        meat  += Xc.T @ np.outer(ec, ec) @ Xc
    meat *= G / (G - 1) * (n - 1) / (n - k)
    return XhXh_inv @ meat @ XhXh_inv


# ── 10. OLS ───────────────────────────────────────────────────────────────────
ols = sm.OLS(y, _Xmat(x)).fit(
    cov_type="cluster", cov_kwds={"groups": groups}
)

# ── 11. First stage ───────────────────────────────────────────────────────────
fs    = sm.OLS(x, _Xmat(z)).fit(
    cov_type="cluster", cov_kwds={"groups": groups}
)
x_hat = fs.fittedvalues
fs_F  = (fs.params[1] / fs.bse[1]) ** 2    # cluster-robust KP-F approximation

# ── 12. 2SLS ─────────────────────────────────────────────────────────────────
X_hat   = _Xmat(x_hat)
beta_iv = np.linalg.solve(X_hat.T @ X_hat, X_hat.T @ y)
e_iv    = y - _Xmat(x) @ beta_iv            # structural residuals
vcov_iv = _clust_vcov_iv(X_hat, e_iv, groups)
se_iv   = np.sqrt(np.diag(vcov_iv))
t_iv    = beta_iv[1] / se_iv[1]

# ── 13. Console output ────────────────────────────────────────────────────────
DIV = "-" * 70

print(f"\n{DIV}")
print("  OLS  --  d2vscope ~ ln(R&D/pop) + controls + firm FE + year FE")
print(DIV)
print(
    f"  b ln(R&D/pop)  = {ols.params[1]:+.4f}{_stars(ols.tvalues[1])}"
    f"  se={ols.bse[1]:.4f}  t={ols.tvalues[1]:.2f}  p={ols.pvalues[1]:.3f}"
)
print(f"  N={int(ols.nobs):,}   R2={ols.rsquared:.4f}   clustered SE: state")

print(f"\n{DIV}")
print("  First Stage  --  ln(R&D/pop) ~ ln(Bartik Z) + controls + firm FE + year FE")
print(DIV)
print(
    f"  b ln(Bartik Z) = {fs.params[1]:+.4f}{_stars(fs.tvalues[1])}"
    f"  se={fs.bse[1]:.4f}  t={fs.tvalues[1]:.2f}  p={fs.pvalues[1]:.3f}"
)
print(f"  N={int(fs.nobs):,}   R2={fs.rsquared:.4f}   First-stage F (KP approx) = {fs_F:.1f}")

print(f"\n{DIV}")
print("  2SLS  --  d2vscope ~ ln(R&D/pop) [IV: ln(Bartik Z)]")
print("            + controls + firm FE + year FE")
print(DIV)
print(
    f"  b ln(R&D/pop)  = {beta_iv[1]:+.4f}{_stars(t_iv)}"
    f"  se={se_iv[1]:.4f}  t={t_iv:.2f}"
)
print(f"  N={len(y):,}   First-stage F={fs_F:.1f}   clustered SE: state")
print()

# ── 14. LaTeX table ───────────────────────────────────────────────────────────
def _fmt(c, s, t) -> tuple[str, str]:
    return f"{c:+.3f}{_stars(t)}", f"({s:.3f})"

ols_c, ols_s = _fmt(ols.params[1],  ols.bse[1],  ols.tvalues[1])
fs_c,  fs_s  = _fmt(fs.params[1],   fs.bse[1],   fs.tvalues[1])
iv_c,  iv_s  = _fmt(beta_iv[1],     se_iv[1],    t_iv)

out_tex = Path(TABLES_DIR) / "reg_scope_funding.tex"
with open(out_tex, "w") as fh:
    fh.write(
        "% reg_scope_funding.tex -- auto-generated by analyses/scope_funding_reg.py\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        " & OLS & First Stage & 2SLS \\\\\n"
        " & (1) & (2) & (3) \\\\\n"
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-3}\\cmidrule(lr){4-4}\n"
        "\\textit{Dep. var.} & d2vscope & ln(R\\&D/pop) & d2vscope \\\\\n"
        "\\midrule\n"
        f"ln(R\\&D/pop)  & {ols_c} & & {iv_c} \\\\\n"
        f"              & {ols_s} & & {iv_s} \\\\\n"
        f"ln(Bartik Z)  & & {fs_c} & \\\\\n"
        f"              & & {fs_s} & \\\\\n"
        "ln(Sale), ln(Assets), R\\&D active"
        " & \\checkmark & \\checkmark & \\checkmark \\\\\n"
        "\\midrule\n"
        f"Observations  & {int(ols.nobs):,} & {int(fs.nobs):,} & {len(y):,} \\\\\n"
        f"R$^2$         & {ols.rsquared:.3f} & {fs.rsquared:.3f} & \\\\\n"
        f"First-stage $F$ & & & {fs_F:.1f} \\\\\n"
        "Firm FE       & \\checkmark & \\checkmark & \\checkmark \\\\\n"
        "Year FE       & \\checkmark & \\checkmark & \\checkmark \\\\\n"
        "Clustered SE  & State & State & State \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

print(f"LaTeX table -> {out_tex}")
