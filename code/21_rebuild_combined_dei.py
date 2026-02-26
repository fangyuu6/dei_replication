"""
21_rebuild_combined_dei.py — Rebuild combined_dish_DEI.csv with pairwise H
=========================================================================
After pairwise expansion (script 20), updates the 334-dish combined dataset
to use pairwise Bradley-Terry H scores instead of BERT absolute H.

Also updates:
  - dish_hedonic_scores.csv (158 original → already done by script 19)
  - expanded_dish_hedonic.csv → expanded_dish_hedonic_pairwise.csv
  - combined_dish_DEI.csv → rebuilt with pairwise H
  - Variance decomposition, Pareto frontier, OLS, cuisine summary

Input:
  data/dish_h_pairwise_all.csv   (337 dishes, unified BT scale)
  data/dish_environmental_costs.csv (158 original E)
  data/expanded_dish_env_costs.csv  (177 expanded E)

Output:
  data/combined_dish_DEI.csv     (334 dishes, pairwise H)
  results/figures/combined_variance_decomposition.png
  results/figures/combined_h_vs_e_scatter.png
  results/figures/combined_dei_by_cuisine.png
  results/tables/combined_variance_decomposition.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, FIGURES_DIR, TABLES_DIR, DEI_TIERS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.1)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("21: Rebuild combined DEI with pairwise H")
print("=" * 70)

# ── Load pairwise H for all dishes ──────────────────────────────
pw = pd.read_csv(DATA_DIR / "dish_h_pairwise_all.csv", index_col="dish_id")
print(f"Pairwise H: {len(pw)} dishes, range [{pw['H_pairwise'].min():.2f}, {pw['H_pairwise'].max():.2f}]")
print(f"  CV: {pw['H_pairwise'].std()/pw['H_pairwise'].mean()*100:.1f}%")
print(f"  Source: {pw['source'].value_counts().to_dict()}")

# ── Load E for all dishes ───────────────────────────────────────
env_orig = pd.read_csv(DATA_DIR / "dish_environmental_costs.csv", index_col="dish_id")
env_exp = pd.read_csv(DATA_DIR / "expanded_dish_env_costs.csv", index_col="dish_id")

# Ensure no overlap
overlap = set(env_orig.index) & set(env_exp.index)
if overlap:
    env_exp = env_exp.drop(list(overlap))

env_all = pd.concat([env_orig, env_exp])
print(f"\nEnvironmental costs: {len(env_all)} dishes")

# ── Load cuisine/cook_method ────────────────────────────────────
# From original DEI
dei_orig = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv", index_col="dish_id")
# From expanded
dei_exp = pd.read_csv(DATA_DIR / "expanded_dish_DEI.csv", index_col="dish_id")

# Merge cuisine info
cuisine_map = {}
cook_map = {}
for idx, row in dei_orig.iterrows():
    cuisine_map[idx] = row.get("cuisine", "Unknown")
    cook_map[idx] = row.get("cook_method", "unknown")
if "cuisine" in env_exp.columns:
    for idx, row in env_exp.iterrows():
        cuisine_map[idx] = row.get("cuisine", "Unknown")
if "cook_method" in env_exp.columns:
    for idx, row in env_exp.iterrows():
        cook_map[idx] = row.get("cook_method", "unknown")

# ── Build combined dataset ──────────────────────────────────────
# Join pairwise H with E
common = sorted(set(pw.index) & set(env_all.index))
print(f"\nDishes with both H and E: {len(common)}")

df = pd.DataFrame(index=common)
df["H_mean"] = pw.loc[common, "H_pairwise"]
df["H_bert"] = pw.loc[common, "H_bert"]

# E components
for col in ["E_composite", "E_carbon", "E_water", "E_energy"]:
    if col in env_all.columns:
        df[col] = env_all.loc[common, col]

# Handle missing E_composite by computing from components if possible
if df["E_composite"].isna().sum() > 0:
    print(f"  Warning: {df['E_composite'].isna().sum()} dishes missing E_composite")
    df = df.dropna(subset=["E_composite"])
    print(f"  After dropping: {len(df)} dishes")

# Log transforms
df["log_H"] = np.log(df["H_mean"])
df["log_E"] = np.log(df["E_composite"])
df["log_DEI"] = df["log_H"] - df["log_E"]

# Z-scores
df["Z_H"] = (df["H_mean"] - df["H_mean"].mean()) / df["H_mean"].std()
df["Z_E"] = (df["E_composite"] - df["E_composite"].mean()) / df["E_composite"].std()
df["DEI_z"] = df["Z_H"] - df["Z_E"]

# Metadata
df["cuisine"] = [cuisine_map.get(d, "Unknown") for d in df.index]
df["cook_method"] = [cook_map.get(d, "unknown") for d in df.index]
df["source"] = pw.loc[df.index, "source"]

# DEI tier
df["DEI_tier"] = pd.qcut(df["log_DEI"], q=5, labels=DEI_TIERS, duplicates="drop")

# Pareto frontier
points = df[["E_composite", "H_mean"]].values
n = len(points)
is_pareto = np.ones(n, dtype=bool)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
            if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                is_pareto[i] = False
                break
df["is_pareto"] = is_pareto

# Ranks
df["rank_DEI"] = df["log_DEI"].rank(ascending=False).astype(int)
df["rank_invE"] = (-df["E_composite"]).rank(ascending=False).astype(int)
df["rank_shift"] = df["rank_invE"] - df["rank_DEI"]

df.index.name = "dish_id"

# ══════════════════════════════════════════════════════════════════
# VARIANCE DECOMPOSITION
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("VARIANCE DECOMPOSITION")
print("="*70)

log_h = df["log_H"]
log_e = df["log_E"]

var_log_h = log_h.var()
var_log_e = log_e.var()
cov_he = np.cov(log_h, log_e)[0, 1]
var_log_dei = df["log_DEI"].var()
var_theoretical = var_log_h + var_log_e - 2 * cov_he

share_h = (var_log_h - cov_he) / var_theoretical * 100
share_e = (var_log_e - cov_he) / var_theoretical * 100

cv_h = df["H_mean"].std() / df["H_mean"].mean() * 100
cv_e = df["E_composite"].std() / df["E_composite"].mean() * 100
r_he = np.corrcoef(log_h, log_e)[0, 1]

print(f"  N dishes: {len(df)}")
print(f"  H CV = {cv_h:.1f}%, E CV = {cv_e:.1f}%, ratio = {cv_e/cv_h:.1f}×")
print(f"  Var(log H)   = {var_log_h:.6f}")
print(f"  Var(log E)   = {var_log_e:.6f}")
print(f"  Cov(log H,E) = {cov_he:.6f}")
print(f"  Var(log DEI)  = {var_log_dei:.6f}")
print(f"  H contribution: {share_h:.1f}%")
print(f"  E contribution: {share_e:.1f}%")
print(f"  Cor(log H, log E): {r_he:.3f}")

# Rank correlation with 1/E
rho_invE, _ = sp_stats.spearmanr(df["log_DEI"], -df["log_E"])
print(f"  ρ(log DEI, 1/E): {rho_invE:.3f}")

# Save variance table
var_df = pd.DataFrame([{
    "n_dishes": len(df),
    "H_CV_pct": cv_h, "E_CV_pct": cv_e,
    "Var_logH": var_log_h, "Var_logE": var_log_e,
    "Cov_logHE": cov_he, "Var_logDEI": var_log_dei,
    "H_pct": share_h, "E_pct": share_e,
    "r_HE": r_he, "rho_DEI_invE": rho_invE,
}])
var_df.to_csv(TABLES_DIR / "combined_variance_decomposition.csv", index=False)

# ══════════════════════════════════════════════════════════════════
# PARETO FRONTIER
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PARETO FRONTIER")
print("="*70)

pareto = df[df["is_pareto"]].sort_values("E_composite")
print(f"  {len(pareto)} Pareto-optimal dishes:")
for idx, row in pareto.iterrows():
    src = "★" if row["source"] == "expanded" else " "
    print(f"    {src} {idx:<25s} H={row['H_mean']:.2f}  E={row['E_composite']:.4f}  "
          f"log_DEI={row['log_DEI']:.2f}  {row['cuisine']}")

# ══════════════════════════════════════════════════════════════════
# TOP / BOTTOM 10
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TOP 10 by log DEI")
print("="*70)
for i, (idx, row) in enumerate(df.nlargest(10, "log_DEI").iterrows(), 1):
    src = "★" if row["source"] == "expanded" else " "
    print(f"  {i:2d}. {src} {idx:<25s} H={row['H_mean']:.2f}  E={row['E_composite']:.4f}  "
          f"log_DEI={row['log_DEI']:.2f}  {row['cuisine']}")

print(f"\nBOTTOM 10 by log DEI")
for i, (idx, row) in enumerate(df.nsmallest(10, "log_DEI").iterrows(), 1):
    src = "★" if row["source"] == "expanded" else " "
    print(f"  {i:2d}. {src} {idx:<25s} H={row['H_mean']:.2f}  E={row['E_composite']:.4f}  "
          f"log_DEI={row['log_DEI']:.2f}  {row['cuisine']}")

# ══════════════════════════════════════════════════════════════════
# CUISINE SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CUISINE SUMMARY")
print("="*70)

cuisine_stats = df.groupby("cuisine").agg(
    n=("log_DEI", "count"),
    H_mean=("H_mean", "mean"),
    E_mean=("E_composite", "mean"),
    DEI_mean=("log_DEI", "mean"),
    DEI_std=("log_DEI", "std"),
).sort_values("DEI_mean", ascending=False)

for cuisine, row in cuisine_stats.iterrows():
    print(f"  {cuisine:<20s} n={row['n']:3.0f}  H={row['H_mean']:.2f}  "
          f"E={row['E_mean']:.3f}  DEI={row['DEI_mean']:.2f} ± {row['DEI_std']:.2f}")

cuisine_stats.to_csv(TABLES_DIR / "combined_cuisine_summary.csv")

# ══════════════════════════════════════════════════════════════════
# PLANT-BASED COMPARISON
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PLANT-BASED COMPARISON")
print("="*70)

plant_mask = df["E_composite"] < 0.1
n_plant = plant_mask.sum()
n_other = (~plant_mask).sum()
h_plant = df.loc[plant_mask, "H_mean"]
h_other = df.loc[~plant_mask, "H_mean"]
dei_plant = df.loc[plant_mask, "log_DEI"]
dei_other = df.loc[~plant_mask, "log_DEI"]

t_h, p_h = sp_stats.ttest_ind(h_plant, h_other)
t_dei, p_dei = sp_stats.ttest_ind(dei_plant, dei_other)

print(f"  Plant-based (E<0.1): n={n_plant}, H={h_plant.mean():.2f}, DEI={dei_plant.mean():.2f}")
print(f"  Other:               n={n_other}, H={h_other.mean():.2f}, DEI={dei_other.mean():.2f}")
print(f"  H difference: t={t_h:.2f}, p={p_h:.4f}")
print(f"  DEI difference: t={t_dei:.2f}, p={p_dei:.2e}")

# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print(f"\n── Plotting ──")

# 1. Variance decomposition bar chart
fig, ax = plt.subplots(figsize=(6, 4))
x = [0]
w = 0.35
ax.bar([0-w/2], [share_h], w, label="H contribution", color="#2196F3")
ax.bar([0+w/2], [share_e], w, label="E contribution", color="#FF5722")
ax.text(0-w/2, share_h+1, f"{share_h:.1f}%", ha="center", fontsize=11)
ax.text(0+w/2, share_e+1, f"{share_e:.1f}%", ha="center", fontsize=11)
ax.set_xticks([0])
ax.set_xticklabels([f"Pairwise H\n({len(df)} dishes)"])
ax.set_ylabel("% of Var(log DEI)")
ax.set_title("Variance Decomposition (Pairwise Bradley-Terry H)")
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "combined_variance_decomposition.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: combined_variance_decomposition.png")

# 2. H vs E scatter with Pareto
fig, ax = plt.subplots(figsize=(10, 7))
cuisines = df["cuisine"].dropna().unique()
palette = sns.color_palette("husl", len(cuisines))
for cuisine, color in zip(sorted(cuisines), palette):
    mask = df["cuisine"] == cuisine
    ax.scatter(df.loc[mask, "E_composite"], df.loc[mask, "H_mean"],
               alpha=0.5, s=35, color=color, label=cuisine,
               edgecolors="white", linewidth=0.3)
# Pareto frontier
front = pareto.sort_values("E_composite")
ax.plot(front["E_composite"], front["H_mean"], "k-", linewidth=2, zorder=5)
ax.scatter(front["E_composite"], front["H_mean"],
           color="gold", s=120, zorder=6, edgecolors="black", linewidth=1.5)
for idx, row in front.iterrows():
    name = idx.replace("_", " ").title()
    ax.annotate(name, (row["E_composite"], row["H_mean"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, fontweight="bold")
ax.set_xlabel("Environmental Cost (E)")
ax.set_ylabel("Hedonic Score (H, Pairwise)")
ax.set_title(f"H vs E: {len(df)} Dishes (Pairwise Bradley-Terry)")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "combined_h_vs_e_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: combined_h_vs_e_scatter.png")

# 3. DEI by cuisine boxplot
fig, ax = plt.subplots(figsize=(14, 6))
cuisine_order = cuisine_stats.index.tolist()
sns.boxplot(data=df, x="cuisine", y="log_DEI", order=cuisine_order,
            palette="viridis", ax=ax, fliersize=3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Cuisine")
ax.set_ylabel("log(DEI)")
ax.set_title(f"DEI by Cuisine ({len(df)} dishes, pairwise H)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "combined_dei_by_cuisine.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: combined_dei_by_cuisine.png")

# ── Save combined dataset ──────────────────────────────────────
df.to_csv(DATA_DIR / "combined_dish_DEI.csv")
print(f"\nSaved: {DATA_DIR / 'combined_dish_DEI.csv'} ({len(df)} dishes)")

print(f"\n{'='*70}")
print("DONE")
print("="*70)
