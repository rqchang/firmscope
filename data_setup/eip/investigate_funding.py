"""
investigate_funding.py
----------------------
Exploratory plots for the EIP innovation funding panel (funding_panel.csv).

Plots generated:
  1. National R&D funding by source ($B), 1990-2025
  2. R&D intensity (% of GSP) over time, national aggregate
  3. R&D per capita over time, national aggregate
  4. Top 20 states by total R&D funding, most recent year
  5. NSF domain breakdown (HJT), national cumulative
  6. NIH domain breakdown (HJT), national cumulative
  7. State-level rd_funding_pct_gsp distribution (box by decade)
  8. Heatmap: top 15 states x year (rd_funding_per_capita)
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Works in both script and interactive (REPL) contexts
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       if "__file__" in dir() else Path.cwd().parents[1]))
from utils.set_paths import PROC_DIR, PLOTS_DIR

# ---------------------------------------------------------------------------
# Load and clean
# ---------------------------------------------------------------------------
panel_path = Path(PROC_DIR) / "eip" / "funding_panel.csv"
df = pd.read_csv(panel_path)

# Drop invalid state codes (e.g. "00" = ungeocoded)
df = df[df["state"].apply(lambda x: str(x).isalpha() and len(str(x)) == 2)].copy()

DOMAINS = [
    "sbir_sttr", "ai_ml", "it_digital", "clean_energy", "neuroscience",
    "biotech_health", "clinical_health", "materials_mfg",
    "aerospace_space", "basic_science", "social_science", "other",
]
DOMAIN_LABELS = {
    "sbir_sttr":       "SBIR/STTR",
    "ai_ml":           "AI & ML",
    "it_digital":      "IT & Digital",
    "clean_energy":    "Clean Energy",
    "neuroscience":    "Neuroscience",
    "biotech_health":  "Biotech & Health",
    "clinical_health": "Clinical & Pop. Health",
    "materials_mfg":   "Materials & Mfg",
    "aerospace_space": "Aerospace & Space",
    "basic_science":   "Basic Science",
    "social_science":  "Social Science",
    "other":           "Other",
}
DOMAIN_COLORS = sns.color_palette("tab10", len(DOMAINS))

Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.edgecolor":    "black",
    "axes.linewidth":    1.2,
    "axes.spines.top":   True,
    "axes.spines.right": True,
})

# National aggregates by year (51 states summed)
nat = (
    df.groupby("year")
    .agg(
        nsf=("nsf_grants_usd", "sum"),
        nih=("nih_grants_usd", "sum"),
        usa=("usaspending_rd_grants_usd", "sum"),
        total=("total_rd_funding_usd", "sum"),
        gsp=("gsp_millions", "sum"),
        pop=("population", "sum"),
    )
    .reset_index()
)
nat["rd_pct_gsp"] = nat["total"] / (nat["gsp"] * 1e6) * 100
nat["rd_per_cap"] = nat["total"] / nat["pop"]

# ---------------------------------------------------------------------------
# Plot 1 – National R&D by source
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(
    nat["year"],
    nat["nsf"] / 1e9,
    nat["nih"] / 1e9,
    nat["usa"] / 1e9,
    labels=["NSF", "NIH", "DOE + DARPA"],
    colors=["#4C72B0", "#55A868", "#C44E52"],
    alpha=0.85,
)
ax.set_xlabel("Year")
ax.set_ylabel("Funding ($B)")
#ax.set_title("National Federal R&D Grants by Source, 1990–2025")
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "tot_funding_by_source.pdf")
plt.show()

# ---------------------------------------------------------------------------
# Plot 2 & 3 – R&D intensity (% GSP) + R&D per capita (2×1 subplots)
# ---------------------------------------------------------------------------
nat_gsp = nat.dropna(subset=["rd_pct_gsp"])
nat_pc  = nat.dropna(subset=["rd_per_cap"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(nat_gsp["year"], nat_gsp["rd_pct_gsp"], color="#4C72B0", linewidth=2)
ax1.fill_between(nat_gsp["year"], nat_gsp["rd_pct_gsp"], alpha=0.15, color="#4C72B0")
ax1.set_ylabel("% of GSP")
ax1.set_title("National R&D Grants Intensity (% of GSP)")
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

ax2.plot(nat_pc["year"], nat_pc["rd_per_cap"], color="#C44E52", linewidth=2)
ax2.fill_between(nat_pc["year"], nat_pc["rd_per_cap"], alpha=0.15, color="#C44E52")
ax2.set_xlabel("Year")
ax2.set_ylabel("$ per capita")
ax2.set_title("National R&D Grants per Capita ($)")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "funding_intensity_per_capita.pdf")
plt.show()

# ---------------------------------------------------------------------------
# Plot 4 – Top 20 states, most recent full year
# ---------------------------------------------------------------------------
latest_year = 2024
all_states = (
    df[df["year"] == latest_year][["state", "total_rd_funding_usd", "rd_funding_per_capita"]]
    .sort_values("total_rd_funding_usd")
)
top20_total = (
    df[df["year"] == latest_year]
    .nlargest(20, "total_rd_funding_usd")[["state", "total_rd_funding_usd"]]
    .sort_values("total_rd_funding_usd")
)
top20_pc = (
    df[df["year"] == latest_year]
    .nlargest(20, "rd_funding_per_capita")[["state", "rd_funding_per_capita"]]
    .sort_values("rd_funding_per_capita")
)
top20_gsp = (
    df[df["year"] == latest_year]
    .nlargest(20, "rd_funding_pct_gsp")[["state", "rd_funding_pct_gsp"]]
    .sort_values("rd_funding_pct_gsp")
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

ax1.barh(top20_total["state"], top20_total["total_rd_funding_usd"] / 1e9, color="#4C72B0")
ax1.set_xlabel("Total R&D Grants ($B)")
ax1.set_title(f"Top 20 States – Total R&D ({latest_year})")
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))

ax2.barh(top20_pc["state"], top20_pc["rd_funding_per_capita"], color="#C44E52")
ax2.set_xlabel("R&D Grants per Capita ($)")
ax2.set_title(f"Top 20 States – R&D per Capita ({latest_year})")
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

ax3.barh(top20_gsp["state"], top20_gsp["rd_funding_pct_gsp"] * 100, color="#55A868")
ax3.set_xlabel("R&D Grants (% of GSP)")
ax3.set_title(f"Top 20 States – R&D % of GSP ({latest_year})")
ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "top20_states_24.pdf")
plt.show()

# ---------------------------------------------------------------------------
# Plot 5 – NSF domain breakdown
# ---------------------------------------------------------------------------
nsf_dom = nat.copy()
for d in DOMAINS:
    col = f"nsf_{d}_usd"
    if col in df.columns:
        nsf_dom[d] = df.groupby("year")[col].sum().reindex(nat["year"]).values / 1e9
    else:
        nsf_dom[d] = 0.0

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(
    nsf_dom["year"],
    *[nsf_dom[d] for d in DOMAINS],
    labels=[DOMAIN_LABELS[d] for d in DOMAINS],
    colors=DOMAIN_COLORS,
    alpha=0.85,
)
ax.set_xlabel("Year")
ax.set_ylabel("Funding ($B)")
ax.set_title("NSF Grants by HJT Innovation Domain")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "nsf_by_domain.pdf")
plt.show()

# ---------------------------------------------------------------------------
# Plot 6 – NIH domain breakdown
# ---------------------------------------------------------------------------
nih_dom = nat.copy()
for d in DOMAINS:
    col = f"nih_{d}_usd"
    if col in df.columns:
        nih_dom[d] = df.groupby("year")[col].sum().reindex(nat["year"]).values / 1e9
    else:
        nih_dom[d] = 0.0

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(
    nih_dom["year"],
    *[nih_dom[d] for d in DOMAINS],
    labels=[DOMAIN_LABELS[d] for d in DOMAINS],
    colors=DOMAIN_COLORS,
    alpha=0.85,
)
ax.set_xlabel("Year")
ax.set_ylabel("Funding ($B)")
ax.set_title("NIH Grants by HJT Innovation Domain")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "nih_by_domain.pdf")
plt.show()

# ---------------------------------------------------------------------------
# Plot 7 – DOE + DARPA (USASpending) total by year
# Note: no domain-level breakdown available in usaspending_rd_grants_usd
# ---------------------------------------------------------------------------
usa_nat = (
    df.groupby("year")["usaspending_rd_grants_usd"]
    .sum()
    .reset_index()
    .rename(columns={"usaspending_rd_grants_usd": "usa"})
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(usa_nat["year"], usa_nat["usa"] / 1e9, color="#C44E52", linewidth=2)
ax.fill_between(usa_nat["year"], usa_nat["usa"] / 1e9, alpha=0.15, color="#C44E52")
ax.set_xlabel("Year")
ax.set_ylabel("Funding ($B)")
ax.set_title("DOE + DARPA R&D Grants (USASpending)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
fig.tight_layout()
#fig.savefig(Path(PLOTS_DIR) / "07_doe_darpa_total.pdf")
plt.show()

# ---------------------------------------------------------------------------
# Plot 8 – Heatmap: top 25 states x domain (% share of state total)
# Averaged over all available years; rows sorted by total funding.
# ---------------------------------------------------------------------------
PLOT_DOMAINS = [d for d in DOMAINS if d != "other"]  # exclude other for clarity

# Sum each domain across NSF + NIH + USA sources per state-year
df_heat = df.copy()
for d in PLOT_DOMAINS:
    cols = [f"{src}_{d}_usd" for src in ("nsf", "nih", "usa") if f"{src}_{d}_usd" in df.columns]
    df_heat[f"total_{d}_usd"] = df_heat[cols].sum(axis=1) if cols else 0.0

# Average over years, keep top 25 states by total funding
top25_states = (
    df_heat.groupby("state")["total_rd_funding_usd"].sum()
    .nlargest(25).index.tolist()
)
state_domain = (
    df_heat[df_heat["state"].isin(top25_states)]
    .groupby("state")[[f"total_{d}_usd" for d in PLOT_DOMAINS]]
    .mean()
)
# Normalize to row share (%) so heatmap shows domain mix, not size
state_domain_pct = state_domain.div(state_domain.sum(axis=1), axis=0) * 100
state_domain_pct.columns = [DOMAIN_LABELS[d] for d in PLOT_DOMAINS]
# Sort rows by total funding descending
state_domain_pct = state_domain_pct.loc[
    df_heat[df_heat["state"].isin(top25_states)]
    .groupby("state")["total_rd_funding_usd"].sum()
    .sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    state_domain_pct, ax=ax, cmap="YlOrRd", linewidths=0.4,
    annot=True, fmt=".1f",
    cbar_kws={"label": "% of state R&D funding"},
)
ax.set_title("Domain Specialization by State – Share of Total R&D Funding (%, avg. all years)")
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=40)
fig.tight_layout()
#fig.savefig(Path(PLOTS_DIR) / "08_heatmap_state_domain.pdf")
plt.show()

# ---------------------------------------------------------------------------
# Plot 9 – Funding density time series: state lines + national avg
#   Two panels: (a) rd_funding_pct_gsp  (b) rd_funding_per_capita
# ---------------------------------------------------------------------------
TOP5_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

top5_states = (
    df.groupby("state")["total_rd_funding_usd"].sum()
    .nlargest(5).index.tolist()
)
top5_colors = {s: c for s, c in zip(top5_states, TOP5_COLORS)}

# National averages: unweighted cross-state mean for each density measure
nat_ts = (
    df.groupby("year")
    .agg(
        total=("total_rd_funding_usd", "sum"),
        pop=("population", "sum"),
        nat_pct_gsp=("rd_funding_pct_gsp", "mean"),
        nat_per_cap=("rd_funding_per_capita", "mean"),
    )
    .reset_index()
)

PANELS = [
    ("rd_funding_pct_gsp", "nat_pct_gsp", "R&D / GSP (%)"),
    ("rd_funding_per_capita", "nat_per_cap", "R&D per Capita ($)"),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor="white", sharex=True)
for ax in axes:
    ax.set_facecolor("white")

for ax, (col, nat_col, ylabel) in zip(axes, PANELS):
    df_plot = df.dropna(subset=[col])
    nat_plot = nat_ts.dropna(subset=[nat_col])
    for state, grp in df_plot.groupby("state"):
        grp_s = grp.sort_values("year")
        if state in top5_colors:
            ax.plot(grp_s["year"], grp_s[col],
                    color=top5_colors[state], linewidth=1.5, alpha=0.85,
                    marker="o", markersize=3, label=state)
        else:
            ax.plot(grp_s["year"], grp_s[col],
                    color="grey", linewidth=0.5, alpha=0.2, label="_nolegend_")
    ax.plot(nat_plot["year"], nat_plot[nat_col],
            color="#d62728", linewidth=2.5, marker="o", markersize=4,
            label="National mean")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, color="lightgrey")
    ax.legend(ncol=2, fontsize=9)

axes[1].set_xlabel("Year")
fig.suptitle("State R&D Funding Density: All States vs. Top 5", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(Path(PLOTS_DIR) / "funding_density_state_ts.pdf")
plt.show()


