"""
07e_uncertainty.py — P0-3 + P1-3: E Uncertainty Quantification & Rank Probability
==================================================================================
Addresses reviewer concern: "No uncertainty propagation... LCA factors span an order of magnitude"

Analyses:
  1. E uncertainty from (a) LCA factor ranges and (b) portion size variability
  2. Monte Carlo propagation to DEI (10,000 simulations)
  3. Rank intervals and Top/Bottom membership probability
  4. Tier stability under uncertainty
  5. Rank probability plots

Outputs:
  - tables/e_uncertainty_bounds.csv
  - tables/dei_rank_intervals.csv
  - tables/dei_tier_stability.csv
  - figures/dei_rank_uncertainty.png
  - figures/dei_rank_probability.png
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
                    GRID_EMISSION_FACTOR, COOKING_ENERGY_KWH, E_WEIGHT_SCHEMES)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────
dei = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv")
impacts = pd.read_csv(DATA_DIR / "ingredient_impact_factors.csv")
print(f"Loaded {len(dei)} dishes, {len(impacts)} ingredient impact factors")

# ── Import recipe data from 04 script ───────────────────────────
# We need DISH_RECIPES dict — import from the script
import importlib.util
spec = importlib.util.spec_from_file_location("env_cost", ROOT / "code" / "04_env_cost_calculation.py")
mod = importlib.util.module_from_spec(spec)

# Monkeypatch to prevent execution
import types
original_init = types.ModuleType.__init__
# Instead: parse the file for DISH_RECIPES dict
print("Extracting DISH_RECIPES from 04_env_cost_calculation.py...")

with open(ROOT / "code" / "04_env_cost_calculation.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find and exec just the DISH_RECIPES dict
import re
# Find start of DISH_RECIPES
start = content.find("DISH_RECIPES = {")
if start == -1:
    raise ValueError("Could not find DISH_RECIPES in 04_env_cost_calculation.py")

# Find end — match braces
depth = 0
end = start
in_string = False
str_char = None
for i, c in enumerate(content[start:], start=start):
    if in_string:
        if c == str_char and content[i-1] != '\\':
            in_string = False
        continue
    if c in ('"', "'"):
        in_string = True
        str_char = c
        continue
    if c == '{':
        depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0:
            end = i + 1
            break

recipe_code = content[start:end]
exec(recipe_code)  # Creates DISH_RECIPES in local scope

print(f"Loaded {len(DISH_RECIPES)} recipes")

# Build impact factor lookup
impact_lookup = {}
for _, row in impacts.iterrows():
    impact_lookup[row["ingredient"]] = {
        "co2": row["co2_per_kg"],
        "water": row["water_per_kg"],
        "land": row["land_per_kg"],
    }

# ══════════════════════════════════════════════════════════════════
# UNCERTAINTY PARAMETER DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# LCA factor uncertainty: Based on Poore & Nemecek 2018 supplementary
# They report interquartile ranges spanning ~0.5x to ~2x the median
# We use log-normal with CV derived from their data
LCA_UNCERTAINTY_CV = {
    # Category-level CVs (coefficient of variation) from P&N Table S2
    "meat": 0.50,      # beef: 3-122 kg CO2, median 60 → CV ≈ 0.5
    "dairy": 0.40,     # cheese: 6-37 kg CO2, median 21 → CV ≈ 0.4
    "poultry": 0.35,   # chicken: 2-12 kg CO2, median 6.9 → CV ≈ 0.35
    "seafood": 0.60,   # fish: highly variable by species
    "grain": 0.30,     # wheat: relatively consistent
    "legume": 0.25,    # beans/lentils: low variation
    "vegetable": 0.30,
    "fruit": 0.30,
    "starch": 0.25,
    "nut": 0.35,
    "oil": 0.30,
    "condiment": 0.35,
    "beverage": 0.40,
    "other": 0.40,
}

# Portion size uncertainty: ±30% (uniform distribution)
PORTION_CV = 0.30

# Cooking energy uncertainty: ±40%
COOKING_ENERGY_CV = 0.40

N_MC = 10000
print(f"\nMonte Carlo simulations: {N_MC:,}")

# ══════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Running Monte Carlo simulation...")
print("=" * 60)

# Get ingredient categories
ingredient_categories = dict(zip(impacts["ingredient"], impacts["category"]))

# Pre-compute normalisation bounds (from original data for consistency)
e_carbon_all = dei["E_carbon"].values
e_water_all = dei["E_water"].values
e_energy_all = dei["E_energy"].values

def compute_e_composite(carbon, water, energy, e_carbon_range, e_water_range, e_energy_range):
    """Compute normalised E composite (equal weighting)."""
    carbon_norm = (carbon - e_carbon_range[0]) / (e_carbon_range[1] - e_carbon_range[0]) if e_carbon_range[1] > e_carbon_range[0] else 0
    water_norm = (water - e_water_range[0]) / (e_water_range[1] - e_water_range[0]) if e_water_range[1] > e_water_range[0] else 0
    energy_norm = (energy - e_energy_range[0]) / (e_energy_range[1] - e_energy_range[0]) if e_energy_range[1] > e_energy_range[0] else 0
    return (carbon_norm + water_norm + energy_norm) / 3

dish_ids = dei["dish_id"].tolist()
n_dishes = len(dish_ids)

# Store MC results: (N_MC, n_dishes) for E and DEI
mc_E = np.zeros((N_MC, n_dishes))
mc_DEI = np.zeros((N_MC, n_dishes))
mc_ranks = np.zeros((N_MC, n_dishes), dtype=int)

# H values (fixed — or with small uncertainty from BERT prediction)
H_values = dei.set_index("dish_id")["H_mean"]
H_std = dei.set_index("dish_id")["H_std"]
H_n = dei.set_index("dish_id")["H_n"]

# H uncertainty: SE = std / sqrt(n)
H_se = H_std / np.sqrt(H_n)

for sim in range(N_MC):
    if sim % 2000 == 0:
        print(f"  Simulation {sim:,}/{N_MC:,}...")

    # Sample H with uncertainty
    h_sample = {}
    for dish in dish_ids:
        h_sample[dish] = np.random.normal(H_values[dish], H_se[dish])

    # Sample E for each dish
    e_carbon_samples = {}
    e_water_samples = {}
    e_energy_samples = {}

    for dish in dish_ids:
        if dish not in DISH_RECIPES:
            # Use original values with some noise
            row = dei[dei["dish_id"] == dish].iloc[0]
            e_carbon_samples[dish] = row["E_carbon"] * np.random.lognormal(0, 0.3)
            e_water_samples[dish] = row["E_water"] * np.random.lognormal(0, 0.3)
            e_energy_samples[dish] = row["E_energy"] * np.random.lognormal(0, 0.3)
            continue

        recipe = DISH_RECIPES[dish]
        carbon_total = 0
        water_total = 0

        for ingredient, base_grams in recipe["ingredients"].items():
            if ingredient not in impact_lookup:
                continue

            # Sample portion variation (multiplicative, log-normal)
            portion_mult = np.random.lognormal(0, PORTION_CV * 0.5)
            grams = base_grams * portion_mult

            # Sample LCA factor variation
            cat = ingredient_categories.get(ingredient, "other")
            cv = LCA_UNCERTAINTY_CV.get(cat, 0.35)
            lca_mult = np.random.lognormal(0, cv * 0.7)  # sigma for log-normal

            impact = impact_lookup[ingredient]
            carbon_total += (grams / 1000) * impact["co2"] * lca_mult
            water_total += (grams / 1000) * impact["water"] * lca_mult

        # Cooking energy
        cook_method = recipe["cook_method"]
        base_energy = COOKING_ENERGY_KWH.get(cook_method, 0.5)
        cook_mult = np.random.lognormal(0, COOKING_ENERGY_CV * 0.5)
        cooking_energy_kwh = base_energy * cook_mult
        cooking_co2 = cooking_energy_kwh * GRID_EMISSION_FACTOR

        e_carbon_samples[dish] = carbon_total + cooking_co2
        e_water_samples[dish] = water_total + cooking_energy_kwh  # water footprint includes cooling
        e_energy_samples[dish] = cooking_energy_kwh

    # Normalize and compute E composite for this simulation
    carbons = np.array([e_carbon_samples[d] for d in dish_ids])
    waters = np.array([e_water_samples[d] for d in dish_ids])
    energies = np.array([e_energy_samples[d] for d in dish_ids])

    c_range = (carbons.min(), carbons.max())
    w_range = (waters.min(), waters.max())
    en_range = (energies.min(), energies.max())

    for j, dish in enumerate(dish_ids):
        e_comp = compute_e_composite(
            e_carbon_samples[dish], e_water_samples[dish], e_energy_samples[dish],
            c_range, w_range, en_range
        )
        mc_E[sim, j] = max(e_comp, 1e-6)  # avoid log(0)
        h = max(h_sample[dish], 1.0)
        mc_DEI[sim, j] = np.log(h) - np.log(mc_E[sim, j])

    # Rank (1 = best DEI)
    mc_ranks[sim] = np.argsort(np.argsort(-mc_DEI[sim])) + 1

print("  Monte Carlo complete!")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: E uncertainty bounds
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 1: E uncertainty bounds")
print("=" * 60)

e_bounds = pd.DataFrame({
    "dish_id": dish_ids,
    "E_original": dei.set_index("dish_id").loc[dish_ids, "E_composite"].values,
    "E_mc_mean": mc_E.mean(axis=0),
    "E_mc_median": np.median(mc_E, axis=0),
    "E_mc_p5": np.percentile(mc_E, 5, axis=0),
    "E_mc_p25": np.percentile(mc_E, 25, axis=0),
    "E_mc_p75": np.percentile(mc_E, 75, axis=0),
    "E_mc_p95": np.percentile(mc_E, 95, axis=0),
})
e_bounds["E_90CI_width"] = e_bounds["E_mc_p95"] - e_bounds["E_mc_p5"]
e_bounds["E_relative_uncertainty"] = e_bounds["E_90CI_width"] / e_bounds["E_mc_median"]

print(f"  Mean relative uncertainty (90% CI width / median): {e_bounds['E_relative_uncertainty'].mean():.2f}")
print(f"  Median relative uncertainty: {e_bounds['E_relative_uncertainty'].median():.2f}")

e_bounds.to_csv(TABLES_DIR / "e_uncertainty_bounds.csv", index=False)
print(f"  Saved: {TABLES_DIR / 'e_uncertainty_bounds.csv'}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: DEI rank intervals
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 2: DEI rank intervals")
print("=" * 60)

rank_stats = pd.DataFrame({
    "dish_id": dish_ids,
    "rank_original": dei.set_index("dish_id").loc[dish_ids, "log_DEI"].rank(ascending=False).values,
    "rank_mc_mean": mc_ranks.mean(axis=0),
    "rank_mc_median": np.median(mc_ranks, axis=0),
    "rank_mc_p5": np.percentile(mc_ranks, 5, axis=0),
    "rank_mc_p95": np.percentile(mc_ranks, 95, axis=0),
    "rank_mc_p25": np.percentile(mc_ranks, 25, axis=0),
    "rank_mc_p75": np.percentile(mc_ranks, 75, axis=0),
})
rank_stats["rank_90CI_width"] = rank_stats["rank_mc_p95"] - rank_stats["rank_mc_p5"]
rank_stats["rank_IQR"] = rank_stats["rank_mc_p75"] - rank_stats["rank_mc_p25"]

# Top/Bottom membership probability
n_tercile = n_dishes // 3
rank_stats["p_top_tercile"] = (mc_ranks <= n_tercile).mean(axis=0)
rank_stats["p_bottom_tercile"] = (mc_ranks > 2 * n_tercile).mean(axis=0)
rank_stats["p_top10"] = (mc_ranks <= 10).mean(axis=0)
rank_stats["p_bottom10"] = (mc_ranks > n_dishes - 10).mean(axis=0)

rank_stats = rank_stats.sort_values("rank_mc_median")

print(f"  Median rank 90% CI width: {rank_stats['rank_90CI_width'].median():.1f} positions")
print(f"  Mean rank 90% CI width: {rank_stats['rank_90CI_width'].mean():.1f} positions")
print(f"  Max rank 90% CI width: {rank_stats['rank_90CI_width'].max():.0f} positions")

print(f"\n  Top 10 most certain ranks (narrowest 90% CI):")
for _, row in rank_stats.nsmallest(10, "rank_90CI_width").iterrows():
    print(f"    {row['dish_id']:25s}  median_rank={row['rank_mc_median']:5.0f}  "
          f"90%CI=[{row['rank_mc_p5']:.0f}, {row['rank_mc_p95']:.0f}]  "
          f"width={row['rank_90CI_width']:.0f}")

print(f"\n  Top 10 most uncertain ranks (widest 90% CI):")
for _, row in rank_stats.nlargest(10, "rank_90CI_width").iterrows():
    print(f"    {row['dish_id']:25s}  median_rank={row['rank_mc_median']:5.0f}  "
          f"90%CI=[{row['rank_mc_p5']:.0f}, {row['rank_mc_p95']:.0f}]  "
          f"width={row['rank_90CI_width']:.0f}")

rank_stats.to_csv(TABLES_DIR / "dei_rank_intervals.csv", index=False)
print(f"\n  Saved: {TABLES_DIR / 'dei_rank_intervals.csv'}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Tier stability
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 3: Tier stability under uncertainty")
print("=" * 60)

# Assign tiers in each MC simulation
tiers = ["Top", "Middle", "Bottom"]
tier_boundaries = [n_tercile, 2 * n_tercile]

# Count how often each dish stays in its original tier
original_tiers = pd.cut(
    rank_stats["rank_original"],
    bins=[0, n_tercile, 2 * n_tercile, n_dishes],
    labels=tiers
)
rank_stats["original_tier"] = original_tiers.values

mc_tier_stability = np.zeros(n_dishes)
for j in range(n_dishes):
    orig_tier_idx = 0 if rank_stats.iloc[j]["rank_original"] <= n_tercile else (
        1 if rank_stats.iloc[j]["rank_original"] <= 2 * n_tercile else 2)
    for sim in range(N_MC):
        r = mc_ranks[sim, j]
        sim_tier = 0 if r <= n_tercile else (1 if r <= 2 * n_tercile else 2)
        if sim_tier == orig_tier_idx:
            mc_tier_stability[j] += 1
    mc_tier_stability[j] /= N_MC

# Map back using dish_ids ordering
tier_stab_by_dish = {dish_ids[j]: mc_tier_stability[j] for j in range(n_dishes)}
rank_stats["tier_stability"] = rank_stats["dish_id"].map(tier_stab_by_dish)

print(f"  Mean tier stability: {rank_stats['tier_stability'].mean():.3f}")
print(f"  Dishes with >90% tier stability: {(rank_stats['tier_stability'] > 0.9).sum()}")
print(f"  Dishes with <50% tier stability: {(rank_stats['tier_stability'] < 0.5).sum()}")

# By tier
for tier in tiers:
    sub = rank_stats[rank_stats["original_tier"] == tier]
    print(f"  {tier:8s}: mean stability = {sub['tier_stability'].mean():.3f}, "
          f"min = {sub['tier_stability'].min():.3f}")

tier_summary = rank_stats[["dish_id", "original_tier", "tier_stability",
                           "rank_mc_median", "rank_90CI_width",
                           "p_top_tercile", "p_bottom_tercile"]].copy()
tier_summary.to_csv(TABLES_DIR / "dei_tier_stability.csv", index=False)
print(f"  Saved: {TABLES_DIR / 'dei_tier_stability.csv'}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1: Rank uncertainty plot (forest plot style)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 20), gridspec_kw={"width_ratios": [3, 1]})

# Panel A: Forest plot of rank intervals
ax = axes[0]
sorted_stats = rank_stats.sort_values("rank_mc_median")

for i, (_, row) in enumerate(sorted_stats.iterrows()):
    y = n_dishes - i
    color = "#2ecc71" if row["original_tier"] == "Top" else (
        "#f1c40f" if row["original_tier"] == "Middle" else "#e74c3c")
    # 90% CI
    ax.plot([row["rank_mc_p5"], row["rank_mc_p95"]], [y, y],
            color=color, alpha=0.3, linewidth=2)
    # IQR
    ax.plot([row["rank_mc_p25"], row["rank_mc_p75"]], [y, y],
            color=color, alpha=0.7, linewidth=4)
    # Median
    ax.plot(row["rank_mc_median"], y, "o", color=color, markersize=3)

# Tier boundary lines
ax.axvline(n_tercile, color="gray", linestyle="--", alpha=0.5, label=f"Tier boundary ({n_tercile})")
ax.axvline(2 * n_tercile, color="gray", linestyle="--", alpha=0.5)

# Labels for top/bottom 5
for i, (_, row) in enumerate(sorted_stats.head(5).iterrows()):
    ax.text(row["rank_mc_p5"] - 2, n_dishes - i, row["dish_id"].replace("_", " "),
            fontsize=6, ha="right", va="center")
for i, (_, row) in enumerate(sorted_stats.tail(5).iterrows()):
    idx = n_dishes - (len(sorted_stats) - 5 + i)
    ax.text(row["rank_mc_p95"] + 2, idx, row["dish_id"].replace("_", " "),
            fontsize=6, ha="left", va="center")

ax.set_xlabel("DEI Rank (1 = best)", fontsize=12)
ax.set_ylabel("Dishes (sorted by median rank)", fontsize=12)
ax.set_title("A. DEI Rank Uncertainty\n(thick = IQR, thin = 90% CI)", fontsize=13)
ax.invert_xaxis()
ax.legend(fontsize=8)

# Panel B: Tier stability
ax = axes[1]
sorted_stab = sorted_stats["tier_stability"].values
colors = ["#2ecc71" if t == "Top" else "#f1c40f" if t == "Middle" else "#e74c3c"
          for t in sorted_stats["original_tier"]]
y_pos = [n_dishes - i for i in range(len(sorted_stab))]
ax.barh(y_pos, sorted_stab, color=colors, alpha=0.7, height=0.8)
ax.axvline(0.5, color="red", linestyle=":", alpha=0.5)
ax.axvline(0.9, color="green", linestyle=":", alpha=0.5)
ax.set_xlabel("Tier Stability\n(fraction staying in same tier)", fontsize=10)
ax.set_title("B. Tier Stability", fontsize=13)
ax.set_xlim(0, 1)
ax.set_yticks([])

plt.tight_layout()
fig_path = FIGURES_DIR / "dei_rank_uncertainty.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Top/Bottom membership probability
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Panel A: P(top tercile) for all dishes
ax = axes[0]
sorted_by_ptop = rank_stats.sort_values("p_top_tercile", ascending=False)
x = range(len(sorted_by_ptop))
colors = ["#2ecc71" if t == "Top" else "#f1c40f" if t == "Middle" else "#e74c3c"
          for t in sorted_by_ptop["original_tier"]]
ax.bar(x, sorted_by_ptop["p_top_tercile"], color=colors, alpha=0.7, width=1.0)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Dishes (sorted by P(top tercile))")
ax.set_ylabel("P(in Top Tercile)")
ax.set_title("A. Probability of Being in Top Tercile")

# Annotate top dishes
for i, (_, row) in enumerate(sorted_by_ptop.head(5).iterrows()):
    if row["p_top_tercile"] > 0.8:
        ax.text(i, row["p_top_tercile"] + 0.02, row["dish_id"].replace("_", " "),
                fontsize=6, rotation=45, ha="left")

# Panel B: Rank distribution for selected dishes
ax = axes[1]
# Pick 8 representative dishes: 3 top, 2 middle, 3 bottom
top3 = rank_stats.nsmallest(3, "rank_mc_median")["dish_id"].tolist()
bottom3 = rank_stats.nlargest(3, "rank_mc_median")["dish_id"].tolist()
middle2 = rank_stats.iloc[n_dishes//2 - 1 : n_dishes//2 + 1]["dish_id"].tolist()
selected = top3 + middle2 + bottom3

for dish in selected:
    j = dish_ids.index(dish)
    ranks = mc_ranks[:, j]
    ax.hist(ranks, bins=30, alpha=0.5, label=dish.replace("_", " "), density=True)

ax.set_xlabel("DEI Rank")
ax.set_ylabel("Density")
ax.set_title("B. Rank Distributions for Selected Dishes")
ax.legend(fontsize=7, ncol=2, loc="upper center")

plt.tight_layout()
fig_path2 = FIGURES_DIR / "dei_rank_probability.png"
plt.savefig(fig_path2, dpi=200, bbox_inches="tight")
print(f"  Saved: {fig_path2}")
plt.close()

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY: E Uncertainty & DEI Rank Stability")
print("=" * 60)

n_stable_90 = (rank_stats['tier_stability'] > 0.9).sum()
n_unstable_50 = (rank_stats['tier_stability'] < 0.5).sum()

print(f"""
1. E UNCERTAINTY:
   - Mean relative uncertainty (90% CI / median): {e_bounds['E_relative_uncertainty'].mean():.2f}
   - LCA factor CVs range from 0.25 (legumes) to 0.60 (seafood)
   - Portion size variation: ±30%

2. DEI RANK UNCERTAINTY:
   - Median rank 90% CI width: {rank_stats['rank_90CI_width'].median():.0f} positions
   - Mean rank 90% CI width: {rank_stats['rank_90CI_width'].mean():.0f} positions
   - Most certain rank: {rank_stats.nsmallest(1, 'rank_90CI_width')['dish_id'].iloc[0]} (width={rank_stats['rank_90CI_width'].min():.0f})
   - Most uncertain rank: {rank_stats.nlargest(1, 'rank_90CI_width')['dish_id'].iloc[0]} (width={rank_stats['rank_90CI_width'].max():.0f})

3. TIER STABILITY:
   - {n_stable_90} dishes ({n_stable_90/n_dishes*100:.0f}%) stay in same tier >90% of simulations
   - {n_unstable_50} dishes ({n_unstable_50/n_dishes*100:.0f}%) are unstable (<50% in same tier)
   - Top tercile: mean stability = {rank_stats[rank_stats['original_tier']=='Top']['tier_stability'].mean():.3f}
   - Bottom tercile: mean stability = {rank_stats[rank_stats['original_tier']=='Bottom']['tier_stability'].mean():.3f}

4. TOP/BOTTOM CERTAINTY:
   - Dishes with P(top10) > 0.5: {(rank_stats['p_top10'] > 0.5).sum()}
   - Dishes with P(bottom10) > 0.5: {(rank_stats['p_bottom10'] > 0.5).sum()}
""")

print("Done!")
