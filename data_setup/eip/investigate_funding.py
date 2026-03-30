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
import numpy as np

# Works in both script and interactive (REPL) contexts
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       if "__file__" in dir() else Path.cwd().parents[1]))
from utils.set_paths import PROC_DIR, PLOTS_DIR


""" Load and clean data """
panel_path = Path(PROC_DIR) / "eip" / "funding_panel.csv"
df = pd.read_csv(panel_path)

# Drop invalid state codes (e.g. "00" = ungeocoded)
df = df[df["state"].apply(lambda x: str(x).isalpha() and len(str(x)) == 2)].copy()
df = df[df["year"] <= 2024].copy()
# Exclude DC: federal agency overhead inflates per-capita and per-GSP metrics
df = df[df["state"] != "DC"].copy()

# All 12 original domains from import_funding.py (preserved in panel columns)
_ALL12 = [
    "sbir_sttr", "ai_ml", "it_digital", "clean_energy", "neuroscience",
    "biotech_health", "clinical_health", "materials_mfg",
    "aerospace_space", "basic_science", "social_science", "other",
]
ALL12_LABELS = {
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
ALL12_COLORS = [
    "#8c564b",  # sbir_sttr       – brown
    "#1f77b4",  # ai_ml           – strong blue
    "#aec7e8",  # it_digital      – light blue
    "#2ca02c",  # clean_energy    – green
    "#9467bd",  # neuroscience    – purple
    "#d62728",  # biotech_health  – red
    "#ff9896",  # clinical_health – light red/pink
    "#7f7f7f",  # materials_mfg   – gray
    "#17becf",  # aerospace_space – cyan
    "#bcbd22",  # basic_science   – olive
    "#e377c2",  # social_science  – pink
    "#c7c7c7",  # other           – light gray
]

# ---------------------------------------------------------------------------
# NSF  –  HJT-inspired 8-bucket consolidation
#   ai_ml → it_digital  |  aerospace_space → basic_science
#   sbir_sttr, social_science → other
# ---------------------------------------------------------------------------
NSF_MERGES = {
    "ai_ml":           "it_digital",
    "aerospace_space": "basic_science",
    "sbir_sttr":       "other",
    "social_science":  "other",
}
NSF_DOMAINS = [
    "it_digital", "clean_energy", "neuroscience",
    "biotech_health", "clinical_health", "materials_mfg",
    "basic_science", "other",
]
NSF_LABELS = {
    "it_digital":      "IT, Digital & AI",
    "clean_energy":    "Clean Energy",
    "neuroscience":    "Neuroscience",
    "biotech_health":  "Biotech & Health",
    "clinical_health": "Clinical & Pop. Health",
    "materials_mfg":   "Materials & Mfg",
    "basic_science":   "Basic & Space Science",
    "other":           "Other",
}
NSF_COLORS = [
    "#1f77b4",  # it_digital      – blue (absorbs ai_ml)
    "#2ca02c",  # clean_energy    – green
    "#9467bd",  # neuroscience    – purple
    "#d62728",  # biotech_health  – red
    "#ff9896",  # clinical_health – light red/pink
    "#7f7f7f",  # materials_mfg   – gray
    "#bcbd22",  # basic_science   – olive (absorbs aerospace_space)
    "#c7c7c7",  # other           – light gray (absorbs sbir_sttr, social_science)
]

# ---------------------------------------------------------------------------
# NIH  –  biomedical sub-domain breakdown + IT as a separate visible band
#   Keeps neuroscience | biotech_health | clinical_health | it_digital separate;
#   collapses energy and physical-science grants into "other_stem".
# ---------------------------------------------------------------------------
NIH_MERGES = {
    "ai_ml":           "it_digital",    # AI absorbed into IT
    "clean_energy":    "other_stem",
    "materials_mfg":   "other_stem",
    "aerospace_space": "other_stem",
    "basic_science":   "other_stem",
    "social_science":  "other_stem",
    "sbir_sttr":       "other_stem",
    "other":           "other_stem",
}
NIH_DOMAINS = ["neuroscience", "biotech_health", "clinical_health", "it_digital", "other_stem"]
NIH_LABELS = {
    "neuroscience":    "Neuroscience",
    "biotech_health":  "Genomics & Drug Discovery",
    "clinical_health": "Clinical & Pop. Health",
    "it_digital":      "IT & AI",
    "other_stem":      "Other (Energy, Phys. Sci.)",
}
NIH_COLORS = [
    "#9467bd",  # neuroscience    – purple
    "#d62728",  # biotech_health  – red
    "#ff9896",  # clinical_health – light red/pink
    "#1f77b4",  # it_digital      – blue
    "#c7c7c7",  # other_stem      – light gray
]

# ---------------------------------------------------------------------------
# DOE + DARPA  –  energy / physical-science breakdown
#   clean_energy and basic_science are DOE's two core missions;
#   DARPA drives it_digital (absorbs ai_ml) and materials_mfg.
# ---------------------------------------------------------------------------
USA_MERGES = {
    "ai_ml":           "it_digital",
    "aerospace_space": "other",
    "sbir_sttr":       "other",
    "neuroscience":    "other",
    "biotech_health":  "other",
    "clinical_health": "other",
    "social_science":  "other",
}
USA_DOMAINS = ["clean_energy", "basic_science", "it_digital", "materials_mfg", "other"]
USA_LABELS = {
    "clean_energy":  "Clean Energy (DOE EERE/FE)",
    "basic_science": "Basic Science (DOE Office of Science)",
    "it_digital":    "IT & AI (DARPA Cyber)",
    "materials_mfg": "Materials & Mfg",
    "other":         "Other",
}
USA_COLORS = [
    "#2ca02c",  # clean_energy  – green
    "#bcbd22",  # basic_science – olive
    "#1f77b4",  # it_digital    – blue
    "#7f7f7f",  # materials_mfg – gray
    "#c7c7c7",  # other         – light gray
]

# DOMAINS/DOMAIN_LABELS/DOMAIN_COLORS = NSF set; reused for state heatmap (Plot 8)
DOMAINS       = NSF_DOMAINS
DOMAIN_LABELS = NSF_LABELS
DOMAIN_COLORS = NSF_COLORS

# Reverse map: NSF merged domain → list of original _ALL12 domains that feed it
# (used by the heatmap to correctly sum all contributing columns)
_NSF_REV: dict[str, list[str]] = {d: [d] for d in NSF_DOMAINS}
for _orig, _tgt in NSF_MERGES.items():
    _NSF_REV[_tgt].append(_orig)


def _dom_agg(df: pd.DataFrame, prefix: str, domains: list[str],
             merges: dict[str, str], years) -> pd.DataFrame:
    """Sum `{prefix}_{orig}_usd` columns into `domains` (in $B) via `merges`."""
    acc = {d: pd.Series(0.0, index=range(len(years))) for d in domains}
    for orig in _ALL12:
        tgt = merges.get(orig, orig)
        if tgt not in acc:
            continue
        col = f"{prefix}_{orig}_usd"
        if col in df.columns:
            vals = df.groupby("year")[col].sum().reindex(years).fillna(0).values / 1e9
            acc[tgt] = acc[tgt] + pd.Series(vals)
    result = pd.DataFrame(acc)
    result.insert(0, "year", years)
    return result

Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.edgecolor":    "black",
    "axes.linewidth":    1.2,
    "axes.spines.top":   True,
    "axes.spines.right": True,
})

# National aggregates by year (51 states summed)
_has_usa = "usaspending_rd_grants_usd" in df.columns
nat = (
    df.groupby("year")
    .agg(
        nsf=("nsf_grants_usd", "sum"),
        nih=("nih_grants_usd", "sum"),
        total=("total_rd_funding_usd", "sum"),
        gsp=("gsp_millions", "sum"),
        pop=("population", "sum"),
        **({} if not _has_usa else {"usa": ("usaspending_rd_grants_usd", "sum")}),
    )
    .reset_index()
)
if "usa" not in nat.columns:
    nat["usa"] = 0.0
nat["rd_pct_gsp"] = nat["total"] / (nat["gsp"] * 1e6) * 100
nat["rd_per_cap"] = nat["total"] / nat["pop"]


""" National R&D by source """
" Total R&D "
fig, ax = plt.subplots(figsize=(10, 5))
# ax.stackplot(
#     nat["year"],
#     nat["nsf"] / 1e9,
#     nat["nih"] / 1e9,
#     nat["usa"] / 1e9,
#     labels=["NSF", "NIH", "DOE + DARPA"],
#     colors=["#4C72B0", "#55A868", "#C44E52"],
#     alpha=0.85,
# )
ax.stackplot(
    nat["year"],
    nat["nsf"] / 1e9,
    nat["nih"] / 1e9,
    nat["usa"] / 1e9,
    labels=["NSF", "NIH"],
    colors=["#4C72B0", "#55A868"],
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


" R&D intensity (% GSP) + R&D per capita "
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


""" Domain breakdowns """
_years = nat["year"].values

" NSF – all 12 domains "
nsf_dom = _dom_agg(df, "nsf", _ALL12, {}, _years)

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(
    nsf_dom["year"],
    *[nsf_dom[d] for d in _ALL12],
    labels=[ALL12_LABELS[d] for d in _ALL12],
    colors=ALL12_COLORS,
    alpha=0.85,
)
ax.set_xlabel("Year")
ax.set_ylabel("Funding ($B)")
ax.set_title("NSF Grants by Innovation Domain")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "nsf_by_domain12.pdf")
plt.show()

" NSF – 8 HJT domains "
nsf_dom = _dom_agg(df, "nsf", NSF_DOMAINS, NSF_MERGES, _years)

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(
    nsf_dom["year"],
    *[nsf_dom[d] for d in NSF_DOMAINS],
    labels=[NSF_LABELS[d] for d in NSF_DOMAINS],
    colors=NSF_COLORS,
    alpha=0.85,
)
ax.set_xlabel("Year")
ax.set_ylabel("Funding ($B)")
#ax.set_title("NSF Grants by HJT Innovation Domain")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "nsf_by_domain8.pdf")
plt.show()


" NIH – biomedical sub-domain breakdown "
nih_dom = _dom_agg(df, "nih", NIH_DOMAINS, NIH_MERGES, _years)

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(
    nih_dom["year"],
    *[nih_dom[d] for d in NIH_DOMAINS],
    labels=[NIH_LABELS[d] for d in NIH_DOMAINS],
    colors=NIH_COLORS,
    alpha=0.85,
)
ax.set_xlabel("Year")
ax.set_ylabel("Funding ($B)")
#ax.set_title("NIH Grants by Biomedical Sub-Domain")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "nih_by_domain.pdf")
plt.show()


""" State-level """
" Top 20 states, 2024 "
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
#ax1.set_title(f"Top 20 States – Total R&D ({latest_year})")
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))

ax2.barh(top20_pc["state"], top20_pc["rd_funding_per_capita"], color="#C44E52")
ax2.set_xlabel("R&D Grants per Capita ($)")
#ax2.set_title(f"Top 20 States – R&D per Capita ({latest_year})")
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

ax3.barh(top20_gsp["state"], top20_gsp["rd_funding_pct_gsp"] * 100, color="#55A868")
ax3.set_xlabel("R&D Grants (% of GSP)")
#ax3.set_title(f"Top 20 States – R&D % of GSP ({latest_year})")
ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

fig.tight_layout()
fig.savefig(Path(PLOTS_DIR) / "top20_states_24.pdf")
plt.show()


" Location-quotient heatmap: top 25 states x domain, three sub-periods "
# LQ_{s,d} = state_share_{s,d} / national_share_d.
# log(LQ) is symmetric around 0: positive = over-specialised, negative = under-specialised.

# Periods reflect federal R&D regime shifts:
#   1990–2000  Bartik base period; pre-NIH-doubling, pre-genomics
#   2001–2012  NIH doubling + post-9/11 DARPA surge + genomics/ARRA era
#   2013–2024  AI/ML emergence, clean energy (IRA), COVID biomedical surge
PERIODS = [
    ("1990–2000", 1990, 2000),
    ("2001–2012", 2001, 2012),
    ("2013–2024", 2013, 2024),
]
PLOT_DOMAINS = DOMAINS  # all 8 HJT domains including "other"

# Build total_{d}_usd per state-year, resolving merged domains via _NSF_REV.
df_heat = df.copy()
for d in PLOT_DOMAINS:
    cols = [
        f"{src}_{orig}_usd"
        for src in ("nsf", "nih", "usa")
        for orig in _NSF_REV.get(d, [d])
        if f"{src}_{orig}_usd" in df.columns
    ]
    df_heat[f"total_{d}_usd"] = df_heat[cols].sum(axis=1) if cols else 0.0

# Fix row order: top-25 states by full-sample total, sorted descending
top25_states = (
    df_heat.groupby("state")["total_rd_funding_usd"].sum()
    .nlargest(25).index.tolist()
)
row_order = (
    df_heat[df_heat["state"].isin(top25_states)]
    .groupby("state")["total_rd_funding_usd"].sum()
    .sort_values(ascending=False).index
)
col_labels = [DOMAIN_LABELS[d] for d in PLOT_DOMAINS]


def _log_lq(df_sub, states, domains):
    """Compute log-LQ matrix (states × domains) for a sub-period slice.

    National share uses ALL states so large research states (CA, MA) are not
    compared against a reference they themselves dominate.
    """
    dom_cols = [f"total_{d}_usd" for d in domains]
    rename   = {f"total_{d}_usd": d for d in domains}
    all_domain   = df_sub.groupby("state")[dom_cols].mean().rename(columns=rename)
    nat_share    = all_domain.sum(axis=0) / all_domain.sum(axis=0).sum()
    state_domain = all_domain.reindex(states)
    state_share  = state_domain.div(state_domain.sum(axis=1), axis=0)
    lq = state_share.div(nat_share, axis=1)
    return np.log(lq.replace(0, np.nan)).clip(-2, 2)


import matplotlib as mpl

# Leave room on the right for a shared colorbar
fig, axes = plt.subplots(1, 3, figsize=(26, 10), sharey=True,
                         gridspec_kw={"wspace": 0.04})

for ax, (label, y0, y1) in zip(axes, PERIODS):
    sub = df_heat[df_heat["year"].between(y0, y1)]
    mat = _log_lq(sub, row_order, PLOT_DOMAINS)
    mat.columns = col_labels
    sns.heatmap(
        mat, ax=ax, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
        linewidths=0.4, annot=True, fmt=".2f",
        cbar=False,  # shared colorbar drawn separately
    )
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40)

# Shared colorbar on a dedicated axis so all three panels stay equal width
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.013, 0.7])
sm = mpl.cm.ScalarMappable(cmap="RdBu_r", norm=mpl.colors.Normalize(vmin=-2, vmax=2))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="log LQ  (0 = national avg)")

fig.savefig(Path(PLOTS_DIR) / "heatmap_state_domain_lq.pdf", bbox_inches="tight")
plt.show()


" Funding density time series: state + national avg "
#   Two panels: (a) rd_funding_pct_gsp  (b) rd_funding_per_capita
TOP5_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

top5_states = (
    df.groupby("state")["total_rd_funding_usd"].sum()
    .nlargest(5).index.tolist()
)
top5_colors = {s: c for s, c in zip(top5_states, TOP5_COLORS)}

# National averages: unweighted cross-state mean for each density measure
# rd_funding_pct_gsp is stored as a fraction in the panel; multiply by 100 for display.
nat_ts = (
    df.groupby("year")
    .agg(
        total=("total_rd_funding_usd", "sum"),
        nat_per_cap=("rd_funding_per_capita", "mean"),
    )
    .reset_index()
)
nat_ts["nat_pct_gsp"] = (
    df.groupby("year")["rd_funding_pct_gsp"].mean().reindex(nat_ts["year"]).values * 100
)

PANELS = [
    ("rd_funding_pct_gsp", "nat_pct_gsp", "R&D / GSP (%)"),
    ("rd_funding_per_capita", "nat_per_cap", "R&D per Capita ($)"),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor="white", sharex=True)
for ax in axes:
    ax.set_facecolor("white")

for ax, (col, nat_col, ylabel) in zip(axes, PANELS):
    # rd_funding_pct_gsp is a fraction in the panel; scale to % for display
    scale = 100 if col == "rd_funding_pct_gsp" else 1
    df_plot = df.dropna(subset=[col])
    nat_plot = nat_ts.dropna(subset=[nat_col])
    for state, grp in df_plot.groupby("state"):
        grp_s = grp.sort_values("year")
        if state in top5_colors:
            ax.plot(grp_s["year"], grp_s[col] * scale,
                    color=top5_colors[state], linewidth=1.5, alpha=0.85,
                    marker="o", markersize=3, label=state)
        else:
            ax.plot(grp_s["year"], grp_s[col] * scale,
                    color="grey", linewidth=0.5, alpha=0.2, label="_nolegend_")
    ax.plot(nat_plot["year"], nat_plot[nat_col],
            color="#d62728", linewidth=2.5, marker="o", markersize=4,
            label="National mean")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, color="lightgrey")
    ax.legend(ncol=2, fontsize=9)

axes[1].set_xlabel("Year")
#fig.suptitle("State R&D Funding Density: All States vs. Top 5", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(Path(PLOTS_DIR) / "funding_density_state_ts.pdf")
plt.show()


