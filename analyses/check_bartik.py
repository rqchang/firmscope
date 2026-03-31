"""
check_bartik.py
---------------
Diagnostic plots for the Bartik shift-share instrument.

Checks:
  1. Shares sanity  — do shares sum to ~1 per agency?
  2. National budgets over time — variation in the "shocks"
  3. Raw scatter: bartik_z_total vs rd_pc — shows the population-size bias
  4. Raw scatter: bartik_z_pc vs rd_pc   — after per-capita correction
  5. First-stage (within): ln(bartik_z_pc) vs ln(rd_pc) after state+year demeaning
  6. State-level mean: which states drive the instrument?
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.set_paths import RAW_DIR, PROC_DIR, PLOTS_DIR

PROC_EIP = os.path.join(PROC_DIR, "eip")

# ── Load data ──────────────────────────────────────────────────────────────────
fund   = pd.read_csv(os.path.join(PROC_EIP, "funding_panel.csv"))
shares = pd.read_csv(os.path.join(PROC_EIP, "bartik_shares.csv"))

fund = fund[fund["state"].apply(lambda x: str(x).isalpha() and len(str(x)) == 2)].copy()
fund = fund[fund["state"] != "DC"].copy()

# Build instrument (mirrors scope_funding_reg.py)
nat = (
    fund.groupby("year")[["nsf_grants_usd", "nih_grants_usd"]]
    .sum()
    .rename(columns={"nsf_grants_usd": "nat_nsf", "nih_grants_usd": "nat_nih"})
    .reset_index()
)
state_yr = fund[["state", "year", "nsf_grants_usd", "nih_grants_usd",
                  "rd_funding_per_capita", "population"]].copy()
state_yr = state_yr.merge(nat, on="year", how="left")
state_yr["loo_nsf"] = state_yr["nat_nsf"] - state_yr["nsf_grants_usd"].fillna(0)
state_yr["loo_nih"] = state_yr["nat_nih"] - state_yr["nih_grants_usd"].fillna(0)

shares_w = (
    shares.pivot(index="state", columns="agency", values="share")
    .rename(columns={"nsf": "share_nsf", "nih": "share_nih"})
    .reset_index()
)
state_yr = state_yr.merge(shares_w, on="state", how="left")
state_yr["bartik_z"]    = (state_yr["share_nsf"] * state_yr["loo_nsf"] +
                            state_yr["share_nih"] * state_yr["loo_nih"])
state_yr["bartik_z_pc"] = state_yr["bartik_z"] / state_yr["population"].clip(lower=1)

state_yr = state_yr[state_yr["year"].between(1997, 2021)].copy()
state_yr = state_yr[state_yr["rd_funding_per_capita"] > 0].copy()
state_yr = state_yr[state_yr["bartik_z_pc"] > 0].copy()

state_yr["ln_rd_pc"] = np.log(state_yr["rd_funding_per_capita"])
state_yr["ln_z_pc"]  = np.log(state_yr["bartik_z_pc"])
state_yr["ln_z_tot"] = np.log(state_yr["bartik_z"].clip(lower=1e-10))


# ── Helper: demean by state and year ──────────────────────────────────────────
def demean(df, col):
    x = df[col].copy()
    x -= df.groupby("state")[col].transform("mean")
    x -= df.groupby("year")[col].transform("mean")
    return x


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ─── Plot 1: Shares sum per agency ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
share_sums = shares.groupby("agency")["share"].sum().reset_index()
bars = ax1.bar(share_sums["agency"], share_sums["share"],
               color=["steelblue", "darkorange"], width=0.5)
ax1.axhline(1.0, color="red", linestyle="--", linewidth=1, label="1.0")
for bar, val in zip(bars, share_sums["share"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax1.set_ylim(0, 1.3)
ax1.set_title("Shares sum per agency\n(should equal 1.0)", fontsize=10)
ax1.set_ylabel("Sum of shares across states")
ax1.legend(fontsize=8)

# ─── Plot 2: Top-10 states by NSF share ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
nsf_shares = (shares[shares["agency"] == "nsf"]
              .nlargest(10, "share")
              .sort_values("share"))
ax2.barh(nsf_shares["state"], nsf_shares["share"], color="steelblue")
ax2.set_title("Top-10 states: NSF base-period share", fontsize=10)
ax2.set_xlabel("Share θ_{s,NSF}")

# ─── Plot 3: National budgets over time ──────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
nat_plot = nat[(nat["year"] >= 1997) & (nat["year"] <= 2021)]
ax3.plot(nat_plot["year"], nat_plot["nat_nsf"] / 1e9, "o-",
         color="steelblue", label="NSF", markersize=3)
ax3.plot(nat_plot["year"], nat_plot["nat_nih"] / 1e9, "s--",
         color="darkorange", label="NIH", markersize=3)
ax3.set_title("National budgets over time\n(Bartik 'shocks')", fontsize=10)
ax3.set_xlabel("Year")
ax3.set_ylabel("$B (billions)")
ax3.legend(fontsize=8, frameon=False)

# ─── Plot 4: Raw scatter — total Z vs rd_pc (the biased version) ─────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(state_yr["ln_z_tot"], state_yr["ln_rd_pc"],
            alpha=0.15, s=8, color="firebrick")
m4 = np.polyfit(state_yr["ln_z_tot"].dropna(),
                state_yr["ln_rd_pc"][state_yr["ln_z_tot"].notna()], 1)
xr = np.linspace(state_yr["ln_z_tot"].min(), state_yr["ln_z_tot"].max(), 100)
ax4.plot(xr, np.polyval(m4, xr), "k-", linewidth=1.5)
ax4.set_title(f"ln(Z_total) vs ln(rd_pc)\n(raw, slope={m4[0]:.3f})", fontsize=10)
ax4.set_xlabel("ln(Bartik Z — total $)")
ax4.set_ylabel("ln(R&D/pop)")

# ─── Plot 5: Raw scatter — Z_pc vs rd_pc (the corrected version) ─────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(state_yr["ln_z_pc"], state_yr["ln_rd_pc"],
            alpha=0.15, s=8, color="steelblue")
m5 = np.polyfit(state_yr["ln_z_pc"], state_yr["ln_rd_pc"], 1)
xr = np.linspace(state_yr["ln_z_pc"].min(), state_yr["ln_z_pc"].max(), 100)
ax5.plot(xr, np.polyval(m5, xr), "k-", linewidth=1.5)
ax5.set_title(f"ln(Z_pc) vs ln(rd_pc)\n(raw, slope={m5[0]:.3f})", fontsize=10)
ax5.set_xlabel("ln(Bartik Z — per capita)")
ax5.set_ylabel("ln(R&D/pop)")

# ─── Plot 6: Within-variation scatter (state + year demeaned) ─────────────────
ax6 = fig.add_subplot(gs[1, 2])
state_yr_w = state_yr.copy()
state_yr_w["ln_z_pc_w"]  = demean(state_yr_w, "ln_z_pc")
state_yr_w["ln_rd_pc_w"] = demean(state_yr_w, "ln_rd_pc")
ax6.scatter(state_yr_w["ln_z_pc_w"], state_yr_w["ln_rd_pc_w"],
            alpha=0.15, s=8, color="steelblue")
m6 = np.polyfit(state_yr_w["ln_z_pc_w"], state_yr_w["ln_rd_pc_w"], 1)
xr = np.linspace(state_yr_w["ln_z_pc_w"].min(), state_yr_w["ln_z_pc_w"].max(), 100)
ax6.plot(xr, np.polyval(m6, xr), "k-", linewidth=1.5)
ax6.set_title(f"Within first stage: ln(Z_pc) → ln(rd_pc)\n"
              f"(state+year demeaned, slope={m6[0]:.3f})", fontsize=10)
ax6.set_xlabel("ln(Z_pc) within")
ax6.set_ylabel("ln(R&D/pop) within")

# ─── Plot 7: Year-by-year correlation ─────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
yr_corr = (state_yr.groupby("year")
           .apply(lambda g: g["ln_z_pc"].corr(g["ln_rd_pc"]))
           .reset_index(name="corr"))
ax7.bar(yr_corr["year"], yr_corr["corr"],
        color=["steelblue" if c >= 0 else "firebrick" for c in yr_corr["corr"]])
ax7.axhline(0, color="black", linewidth=0.8)
ax7.set_title("Year-by-year corr:\nln(Z_pc) vs ln(rd_pc)", fontsize=10)
ax7.set_xlabel("Year")
ax7.set_ylabel("Pearson r")

# ─── Plot 8: LOO check — how much does removing each state change national budget?
ax8 = fig.add_subplot(gs[2, 1])
loo_check = state_yr.groupby("state").agg(
    share_nsf=("share_nsf", "first"),
    mean_loo_frac_nsf=("loo_nsf", lambda x: (x / state_yr.loc[x.index, "nat_nsf"]).mean()),
).reset_index().nlargest(10, "share_nsf")
ax8.barh(loo_check["state"], 1 - loo_check["mean_loo_frac_nsf"], color="steelblue")
ax8.set_title("LOO correction size (NSF)\n= own share removed from national total",
              fontsize=10)
ax8.set_xlabel("Own share of national total")

# ─── Plot 9: State means — instrument vs treatment ───────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
state_means = state_yr.groupby("state")[["ln_z_pc", "ln_rd_pc"]].mean().reset_index()
ax9.scatter(state_means["ln_z_pc"], state_means["ln_rd_pc"],
            s=40, color="steelblue")
for _, row in state_means.iterrows():
    ax9.annotate(row["state"], (row["ln_z_pc"], row["ln_rd_pc"]),
                 fontsize=5, alpha=0.7)
m9 = np.polyfit(state_means["ln_z_pc"], state_means["ln_rd_pc"], 1)
xr = np.linspace(state_means["ln_z_pc"].min(), state_means["ln_z_pc"].max(), 100)
ax9.plot(xr, np.polyval(m9, xr), "k-", linewidth=1.5)
ax9.set_title(f"State means: Z_pc vs rd_pc\n(cross-sectional, slope={m9[0]:.3f})",
              fontsize=10)
ax9.set_xlabel("ln(Z_pc) — state mean")
ax9.set_ylabel("ln(R&D/pop) — state mean")

fig.suptitle("Bartik Instrument Diagnostics", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.show()

fig_path = os.path.join(PLOTS_DIR, "check_bartik.pdf")
fig.savefig(fig_path, bbox_inches="tight")
os.startfile(fig_path)
print(f"Saved: {fig_path}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n=== Share sums (should be 1.0 per agency) ===")
print(share_sums.to_string(index=False))

print("\n=== First-stage correlations ===")
print(f"  Raw:   corr(ln_z_tot, ln_rd_pc) = {state_yr['ln_z_tot'].corr(state_yr['ln_rd_pc']):+.4f}")
print(f"  Raw:   corr(ln_z_pc,  ln_rd_pc) = {state_yr['ln_z_pc'].corr(state_yr['ln_rd_pc']):+.4f}")
print(f"  Within: corr(ln_z_pc_w, ln_rd_pc_w) = {state_yr_w['ln_z_pc_w'].corr(state_yr_w['ln_rd_pc_w']):+.4f}")
print(f"  Within slope = {m6[0]:.4f}")
