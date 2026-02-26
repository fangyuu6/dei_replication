"""
25e_e_uncertainty_v2.py — E Uncertainty Propagation for 2,563-dish dataset
==========================================================================
Monte Carlo perturbation of recipe weights (±20%) and LCA impact factors
(log-normal with category-specific CVs from Poore & Nemecek 2018).

Key outputs:
  - Rank correlation ρ(perturbed E, point-estimate E) across MC draws
  - Fraction of dishes within ±2 rank-decile positions
  - Var(log DEI) decomposition stability across draws
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TABLES_DIR = ROOT / "results" / "tables"

# ── Load data ────────────────────────────────────────────────────
dei = pd.read_csv(DATA_DIR / "combined_dish_DEI_v2.csv")
print(f"Loaded {len(dei)} dishes")

# Original E and H
E_orig = dei["E_composite"].values
logE_orig = dei["log_E"].values
logH = dei["log_H"].values
n_dishes = len(dei)

# ── MC parameters ────────────────────────────────────────────────
N_MC = 10000

# Recipe weight uncertainty: ±20% → log-normal sigma
RECIPE_SIGMA = 0.20  # ~20% CV

# LCA factor uncertainty by category (from P&N supplementary)
# We assign each dish a category-level CV based on its dominant ingredients
# Approximated by E_composite magnitude: high-E dishes are meat-heavy (higher CV)
# For simplicity, we perturb the three E sub-components independently
LCA_CV_CARBON = 0.35   # GHG: median CV across food categories
LCA_CV_WATER  = 0.40   # Water footprint: slightly more variable
LCA_CV_ENERGY = 0.30   # Cooking energy: more consistent

print(f"Monte Carlo: {N_MC:,} draws")
print(f"  Recipe weight sigma: {RECIPE_SIGMA}")
print(f"  LCA CVs: carbon={LCA_CV_CARBON}, water={LCA_CV_WATER}, energy={LCA_CV_ENERGY}")

# ── Monte Carlo ──────────────────────────────────────────────────
# For each draw, perturb the three E sub-components independently
# Then re-normalize and compute E_composite

E_carbon = dei["E_carbon"].values
E_water  = dei["E_water"].values
E_energy = dei["E_energy"].values

rank_orig = stats.rankdata(-np.log(dei["E_composite"].values + 1e-9))  # DEI rank (high=good)
# Actually use log_DEI for ranking
logDEI_orig = dei["log_DEI"].values
rank_orig_dei = stats.rankdata(-logDEI_orig)

rho_E_list = []
rho_DEI_list = []
e_dominance_list = []
within_2decile_list = []

decile_width = n_dishes / 10.0

for sim in range(N_MC):
    if sim % 2000 == 0:
        print(f"  Sim {sim:,}/{N_MC:,}...")

    # Perturb each sub-component with log-normal noise
    carbon_pert = E_carbon * np.random.lognormal(0, LCA_CV_CARBON, n_dishes)
    water_pert  = E_water  * np.random.lognormal(0, LCA_CV_WATER,  n_dishes)
    energy_pert = E_energy * np.random.lognormal(0, LCA_CV_ENERGY, n_dishes)

    # Also perturb recipe weights (multiplicative, correlated across sub-components)
    recipe_mult = np.random.lognormal(0, RECIPE_SIGMA, n_dishes)
    carbon_pert *= recipe_mult
    water_pert  *= recipe_mult
    # energy_pert is cooking-based, less affected by recipe weight
    energy_pert *= np.random.lognormal(0, RECIPE_SIGMA * 0.5, n_dishes)

    # Re-normalize (same method as original: divide by cross-dish max)
    c_max = carbon_pert.max()
    w_max = water_pert.max()
    en_max = energy_pert.max()

    E_pert = (carbon_pert / c_max + water_pert / w_max + energy_pert / en_max) / 3.0
    E_pert = np.maximum(E_pert, 1e-9)

    # Spearman ρ between perturbed and original E
    rho_e, _ = stats.spearmanr(E_orig, E_pert)
    rho_E_list.append(rho_e)

    # Compute perturbed log DEI
    logE_pert = np.log(E_pert)
    logDEI_pert = logH - logE_pert

    # Spearman ρ for DEI ranking
    rho_dei, _ = stats.spearmanr(logDEI_orig, logDEI_pert)
    rho_DEI_list.append(rho_dei)

    # Variance decomposition: E dominance
    var_logH = np.var(logH)
    var_logE_pert = np.var(logE_pert)
    var_logDEI_pert = np.var(logDEI_pert)
    e_share = var_logE_pert / (var_logH + var_logE_pert)
    e_dominance_list.append(e_share * 100)

    # Fraction within ±2 decile positions
    rank_pert = stats.rankdata(-logDEI_pert)
    rank_diff = np.abs(rank_orig_dei - rank_pert)
    within_2dec = np.mean(rank_diff <= 2 * decile_width)
    within_2decile_list.append(within_2dec)

rho_E_arr = np.array(rho_E_list)
rho_DEI_arr = np.array(rho_DEI_list)
e_dom_arr = np.array(e_dominance_list)
w2d_arr = np.array(within_2decile_list)

# ── Results ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("E UNCERTAINTY PROPAGATION RESULTS")
print("=" * 60)

print(f"\n1. E rank stability:")
print(f"   Median ρ(E_orig, E_pert): {np.median(rho_E_arr):.3f}")
print(f"   95% interval: [{np.percentile(rho_E_arr, 2.5):.3f}, {np.percentile(rho_E_arr, 97.5):.3f}]")

print(f"\n2. DEI rank stability:")
print(f"   Median ρ(DEI_orig, DEI_pert): {np.median(rho_DEI_arr):.3f}")
print(f"   95% interval: [{np.percentile(rho_DEI_arr, 2.5):.3f}, {np.percentile(rho_DEI_arr, 97.5):.3f}]")

print(f"\n3. Fraction within ±2 decile positions:")
print(f"   Median: {np.median(w2d_arr):.3f} ({np.median(w2d_arr)*100:.1f}%)")
print(f"   95% interval: [{np.percentile(w2d_arr, 2.5):.3f}, {np.percentile(w2d_arr, 97.5):.3f}]")

print(f"\n4. E-dominance across MC draws:")
print(f"   Median E share of Var(log DEI): {np.median(e_dom_arr):.1f}%")
print(f"   95% interval: [{np.percentile(e_dom_arr, 2.5):.1f}%, {np.percentile(e_dom_arr, 97.5):.1f}%]")
print(f"   Min E share: {e_dom_arr.min():.1f}%")
print(f"   E > 90% in all draws: {(e_dom_arr > 90).all()}")

# Save summary
summary = pd.DataFrame({
    "metric": [
        "rho_E_median", "rho_E_p2.5", "rho_E_p97.5",
        "rho_DEI_median", "rho_DEI_p2.5", "rho_DEI_p97.5",
        "within_2decile_median", "within_2decile_p2.5", "within_2decile_p97.5",
        "e_dominance_median", "e_dominance_p2.5", "e_dominance_p97.5",
        "e_dominance_min",
    ],
    "value": [
        np.median(rho_E_arr), np.percentile(rho_E_arr, 2.5), np.percentile(rho_E_arr, 97.5),
        np.median(rho_DEI_arr), np.percentile(rho_DEI_arr, 2.5), np.percentile(rho_DEI_arr, 97.5),
        np.median(w2d_arr), np.percentile(w2d_arr, 2.5), np.percentile(w2d_arr, 97.5),
        np.median(e_dom_arr), np.percentile(e_dom_arr, 2.5), np.percentile(e_dom_arr, 97.5),
        e_dom_arr.min(),
    ],
})
summary.to_csv(TABLES_DIR / "e_uncertainty_mc_v2.csv", index=False)
print(f"\nSaved: {TABLES_DIR / 'e_uncertainty_mc_v2.csv'}")
print("Done!")
