"""
check_bartik.py
---------------
Diagnostic plots for the Bartik shift-share instrument.

Row 1: Share sanity | NSF top-10 states | National budgets (NIH doubled 2000-03)
Row 2: Within composite Z_pc | Within Z_NSF | Within Z_NIH
Row 3: NIH vs NSF share scatter | Year-by-year corr per agency | State means
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

# ── Load and build instrument — mirrors scope_funding_reg.py exactly ──────────
fund = pd.read_csv(os.path.join(PROC_EIP, "funding_panel.csv"))
fund = fund[fund["state"].apply(lambda x: str(x).isalpha() and len(str(x)) == 2)].copy()
fund = fund[fund["state"] != "DC"].copy()

# Pre-sample shares: 1990-1996 average (same as main code)
fund_base  = fund[fund["year"] <= 1996]
base_state = fund_base.groupby("state")[["nsf_grants_usd", "nih_grants_usd"]].mean().fillna(0)
base_state["share_nsf"] = base_state["nsf_grants_usd"] / base_state["nsf_grants_usd"].sum()
base_state["share_nih"] = base_state["nih_grants_usd"] / base_state["nih_grants_usd"].sum()

# National LOO totals
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
state_yr = state_yr.merge(base_state[["share_nsf", "share_nih"]].reset_index(),
                          on="state", how="left")

_pop = state_yr["population"].clip(lower=1)
state_yr["bartik_z_pc"]     = (state_yr["share_nsf"] * state_yr["loo_nsf"] +
                                state_yr["share_nih"] * state_yr["loo_nih"]) / _pop
state_yr["bartik_z_nsf_pc"] = state_yr["share_nsf"] * state_yr["loo_nsf"] / _pop
state_yr["bartik_z_nih_pc"] = state_yr["share_nih"] * state_yr["loo_nih"] / _pop

state_yr = state_yr[state_yr["year"].between(1997, 2021)].copy()
state_yr = state_yr[state_yr["rd_funding_per_capita"] > 0].copy()
state_yr = state_yr[state_yr["bartik_z_pc"] > 0].copy()

state_yr["ln_rd_pc"]    = np.log(state_yr["rd_funding_per_capita"])
state_yr["ln_z_pc"]     = np.log(state_yr["bartik_z_pc"])
state_yr["ln_z_nsf"]    = np.log(state_yr["bartik_z_nsf_pc"].clip(lower=1e-12))
state_yr["ln_z_nih"]    = np.log(state_yr["bartik_z_nih_pc"].clip(lower=1e-12))

# ── Helper: two-way demean ─────────────────────────────────────────────────────
def demean(df, col):
    x = df[col].copy()
    x -= df.groupby("state")[col].transform("mean")
    x -= df.groupby("year")[col].transform("mean")
    return x

def within_scatter(ax, df, x_col, y_col, color, title):
    xw = demean(df, x_col)
    yw = demean(df, y_col)
    valid = np.isfinite(xw) & np.isfinite(yw)
    m = np.polyfit(xw[valid], yw[valid], 1)
    xr = np.linspace(xw[valid].min(), xw[valid].max(), 100)
    ax.scatter(xw, yw, alpha=0.12, s=6, color=color)
    ax.plot(xr, np.polyval(m, xr), "k-", linewidth=1.5)
    ax.set_title(f"{title}\n(state+year demeaned, slope={m[0]:.3f})", fontsize=9)
    ax.set_xlabel(f"{x_col} within")
    ax.set_ylabel("ln(R&D/pop) within")
    return m[0], xw[valid].std()

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

# ─── Row 1: Instrument construction checks ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
share_sums = (base_state[["share_nsf", "share_nih"]]
              .sum().rename({"share_nsf": "nsf", "share_nih": "nih"}))
bars = ax1.bar(share_sums.index, share_sums.values,
               color=["steelblue", "darkorange"], width=0.5)
ax1.axhline(1.0, color="red", linestyle="--", linewidth=1, label="1.0")
for bar, val in zip(bars, share_sums.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax1.set_ylim(0, 1.3)
ax1.set_title("Pre-sample shares sum (1990-96)\n(should equal 1.0 per agency)", fontsize=9)
ax1.set_ylabel("Sum across states")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
top_nsf = base_state.nlargest(10, "share_nsf")[["share_nsf"]].sort_values("share_nsf")
top_nih = base_state.nlargest(10, "share_nih")[["share_nih"]].sort_values("share_nih")
ax2.barh(top_nsf.index, top_nsf["share_nsf"], color="steelblue", alpha=0.8, label="NSF")
ax2.barh(top_nih.index, top_nih["share_nih"], color="darkorange", alpha=0.6, label="NIH")
ax2.set_title("Top-10 states by pre-sample share\n(NSF vs NIH)", fontsize=9)
ax2.set_xlabel("Share θ_{s,a}")
ax2.legend(fontsize=8, frameon=False)

ax3 = fig.add_subplot(gs[0, 2])
nat_plot = nat[(nat["year"] >= 1997) & (nat["year"] <= 2021)]
ax3.plot(nat_plot["year"], nat_plot["nat_nsf"] / 1e9, "o-",
         color="steelblue", label="NSF", markersize=3)
ax3.plot(nat_plot["year"], nat_plot["nat_nih"] / 1e9, "s--",
         color="darkorange", label="NIH", markersize=3)
ax3.axvspan(2000, 2003, alpha=0.12, color="darkorange", label="NIH doubling")
ax3.set_title("National budgets (Bartik shocks)\nNIH doubled 2000-03; NSF flat", fontsize=9)
ax3.set_xlabel("Year"); ax3.set_ylabel("$B (billions)")
ax3.legend(fontsize=8, frameon=False)

# ─── Row 2: Within-variation — the key diagnostic ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
slope_comp, std_comp = within_scatter(ax4, state_yr, "ln_z_pc",  "ln_rd_pc",
                                      "gray",       "Composite Z_pc within")

ax5 = fig.add_subplot(gs[1, 1])
slope_nsf, std_nsf = within_scatter(ax5, state_yr, "ln_z_nsf", "ln_rd_pc",
                                    "steelblue",  "NSF-only Z_pc within")

ax6 = fig.add_subplot(gs[1, 2])
slope_nih, std_nih = within_scatter(ax6, state_yr, "ln_z_nih", "ln_rd_pc",
                                    "darkorange", "NIH-only Z_pc within")

# ─── Row 3: Differential exposure, correlation, state means ───────────────────
ax7 = fig.add_subplot(gs[2, 0])
# NSF vs NIH share scatter — states with very different mixes get differential exposure
ax7.scatter(base_state["share_nsf"], base_state["share_nih"],
            s=30, color="steelblue", alpha=0.7)
for st, row in base_state.iterrows():
    ax7.annotate(st, (row["share_nsf"], row["share_nih"]), fontsize=5, alpha=0.7)
# Regression line (if states were equally weighted, this is the 45-degree mix)
ax7.set_title("Pre-sample NSF vs NIH shares\n(spread = differential exposure)", fontsize=9)
ax7.set_xlabel("θ_{s,NSF}")
ax7.set_ylabel("θ_{s,NIH}")
corr_shares = base_state["share_nsf"].corr(base_state["share_nih"])
ax7.text(0.05, 0.92, f"corr={corr_shares:.3f}", transform=ax7.transAxes, fontsize=8)

ax8 = fig.add_subplot(gs[2, 1])
yr_corr_nsf = (state_yr.groupby("year")
               .apply(lambda g: g["ln_z_nsf"].corr(g["ln_rd_pc"]))
               .reset_index(name="corr_nsf"))
yr_corr_nih = (state_yr.groupby("year")
               .apply(lambda g: g["ln_z_nih"].corr(g["ln_rd_pc"]))
               .reset_index(name="corr_nih"))
ax8.plot(yr_corr_nsf["year"], yr_corr_nsf["corr_nsf"], "o-",
         color="steelblue", label="NSF", markersize=3)
ax8.plot(yr_corr_nih["year"], yr_corr_nih["corr_nih"], "s--",
         color="darkorange", label="NIH", markersize=3)
ax8.axhline(0, color="black", linewidth=0.8)
ax8.set_title("Year-by-year corr: Z_agency vs rd_pc\n(divergence → within variation)", fontsize=9)
ax8.set_xlabel("Year"); ax8.set_ylabel("Pearson r")
ax8.legend(fontsize=8, frameon=False)

ax9 = fig.add_subplot(gs[2, 2])
state_means = state_yr.groupby("state")[["ln_z_pc", "ln_rd_pc"]].mean().reset_index()
ax9.scatter(state_means["ln_z_pc"], state_means["ln_rd_pc"], s=40, color="steelblue")
for _, row in state_means.iterrows():
    ax9.annotate(row["state"], (row["ln_z_pc"], row["ln_rd_pc"]), fontsize=5, alpha=0.7)
m9 = np.polyfit(state_means["ln_z_pc"], state_means["ln_rd_pc"], 1)
xr = np.linspace(state_means["ln_z_pc"].min(), state_means["ln_z_pc"].max(), 100)
ax9.plot(xr, np.polyval(m9, xr), "k-", linewidth=1.5)
ax9.set_title(f"State means: composite Z_pc vs rd_pc\n(cross-sect. slope={m9[0]:.3f})", fontsize=9)
ax9.set_xlabel("ln(Z_pc) state mean"); ax9.set_ylabel("ln(R&D/pop) state mean")

fig.suptitle("Bartik Instrument Diagnostics — Pre-sample shares 1990–96",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.show()

fig_path = os.path.join(PLOTS_DIR, "check_bartik.pdf")
fig.savefig(fig_path, bbox_inches="tight")
os.startfile(fig_path)
print(f"Saved: {fig_path}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n=== Pre-sample share sums (should be 1.0 per agency) ===")
print(f"  NSF: {base_state['share_nsf'].sum():.4f}   NIH: {base_state['share_nih'].sum():.4f}")

print("\n=== Within-variation (state+year demeaned) ===")
print(f"  Composite Z_pc : slope={slope_comp:+.4f}  within-SD={std_comp:.4f}")
print(f"  NSF-only Z_pc  : slope={slope_nsf:+.4f}  within-SD={std_nsf:.4f}")
print(f"  NIH-only Z_pc  : slope={slope_nih:+.4f}  within-SD={std_nih:.4f}")
print(f"\n  NSF vs NIH share correlation: {corr_shares:.3f}")
print(f"  (lower corr → more differential exposure → more within variation)")
