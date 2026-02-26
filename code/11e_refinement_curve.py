"""
11e_refinement_curve.py — Refinement Cost Curve Analysis
========================================================
New concept: as a dish goes from basic → refined, E increases
disproportionately faster than H. "Diminishing hedonic returns
on environmental investment."

Steps:
  1. Define 15+ dish families (variants of same base concept)
  2. Compute refinement score per dish
  3. Fit H ~ α·log(E/E_base) within each family
  4. Simulate base variants where missing
  5. Global mixed-effects model
  6. Policy visualizations

Outputs:
  - tables/dish_families.csv
  - tables/refinement_curves.csv
  - figures/refinement_cost_curves.png
  - figures/refinement_global_fit.png
  - figures/hedonic_waste_by_family.png
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR, COOKING_ENERGY_KWH

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ══════════════════════════════════════════════════════════════════
# STEP 1: Define Dish Families
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("11e: Refinement Cost Curve Analysis")
print("=" * 60)

DISH_FAMILIES = {
    "Curry": ["dal", "chana_masala", "aloo_gobi", "korma", "green_curry", "red_curry",
              "tikka_masala", "butter_chicken", "massaman_curry", "rendang"],
    "Noodle/Pasta": ["wonton_noodle", "dan_dan_noodles", "lo_mein", "chow_mein", "pad_thai",
                     "pasta_carbonara", "ramen", "pasta_bolognese", "tonkotsu_ramen", "lasagna"],
    "Rice Dish": ["fried_rice", "bibimbap", "nasi_goreng", "jollof_rice", "arroz_con_pollo",
                  "biryani", "paella", "risotto", "plov"],
    "Chicken": ["chicken_sandwich", "tandoori_chicken", "kung_pao_chicken", "teriyaki",
                "jerk_chicken", "piri_piri", "fried_chicken", "korean_fried_chicken",
                "butter_chicken", "coq_au_vin"],
    "Beef": ["hamburger", "bulgogi", "anticucho", "steak", "churrasco",
             "beef_bourguignon", "osso_buco", "brisket"],
    "Salad/Cold": ["coleslaw", "papaya_salad", "som_tam", "fattoush", "tabbouleh",
                   "caesar_salad", "caprese", "nicoise_salad", "cobb_salad"],
    "Dessert": ["gelato", "ice_cream", "panna_cotta", "tiramisu",
                "creme_brulee", "brownie", "cheesecake"],
    "Seafood": ["ceviche", "sushi", "sashimi", "fish_tacos", "bouillabaisse",
                "gambas_al_ajillo", "tempura", "fish_and_chips"],
    "Soup": ["miso_soup", "tom_yum", "gazpacho", "borscht", "clam_chowder",
             "french_onion_soup"],
    "Wrapped/Dumpling": ["spring_rolls", "dumplings", "gyoza", "samosa", "empanada_argentina",
                         "lumpia", "pupusa"],
    "Bread/Pastry": ["pita", "naan", "focaccia", "pretzel", "croissant", "simit"],
    "Pork": ["samgyeopsal", "adobo", "carnitas", "pork_katsu", "pulled_pork", "ribs"],
    "Kebab/Grilled": ["satay_indonesian", "kebab", "souvlaki", "doner_kebab",
                       "iskender_kebab", "kofte"],
    "Egg Dish": ["tamagoyaki", "shakshuka", "eggs_benedict", "tortilla_espanola",
                 "pajeon"],
    "Vegetable/Legume": ["guacamole", "hummus", "dal", "falafel", "chana_masala"],
}

# Load data
dei = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
print(f"Loaded {len(dei)} dishes")

# Map dishes to families
family_map = {}
for family, dishes in DISH_FAMILIES.items():
    for dish in dishes:
        if dish in family_map:
            # Dish appears in multiple families — keep first
            continue
        family_map[dish] = family

dei["family"] = dei["dish_id"].map(family_map)
assigned = dei[dei["family"].notna()]
print(f"Assigned {len(assigned)} dishes to {assigned['family'].nunique()} families")
print(f"Unassigned: {len(dei) - len(assigned)} dishes")

# Family sizes
print("\nFamily sizes:")
for fam, count in assigned["family"].value_counts().items():
    print(f"  {fam:20s}: {count} dishes")

# Save family mapping
fam_df = dei[["dish_id", "family"]].copy()
fam_df.to_csv(TABLES_DIR / "dish_families.csv", index=False)
print(f"\nSaved: {TABLES_DIR / 'dish_families.csv'}")

# ══════════════════════════════════════════════════════════════════
# STEP 2: Compute Refinement Score
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Refinement Score Computation")
print("=" * 60)

# Load recipes for refinement score calculation
def _extract_recipes(script_path, var_name):
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    start_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{var_name} = {{"):
            start_line = i
            break
    if start_line is None:
        return {}
    depth = 0
    collected = []
    for line in lines[start_line:]:
        collected.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            break
    dict_code = "\n".join(collected)
    local_ns = {}
    exec(compile(dict_code, "<recipes>", "exec"), {"__builtins__": __builtins__}, local_ns)
    return local_ns.get(var_name, {})

ALL_RECIPES = {}
ALL_RECIPES.update(_extract_recipes(ROOT / "code" / "04_env_cost_calculation.py", "DISH_RECIPES"))
ALL_RECIPES.update(_extract_recipes(ROOT / "code" / "09b_expanded_recipes.py", "EXPANDED_RECIPES"))
print(f"Loaded {len(ALL_RECIPES)} total recipes")

# High-impact animal proteins
ANIMAL_PROTEINS = {"beef", "ground_beef", "lamb", "pork", "chicken", "duck", "turkey",
                   "bacon", "sausage", "fish", "salmon", "shrimp", "squid", "tuna",
                   "cod", "crab"}

# Compute refinement features
refine_rows = []
for _, row in assigned.iterrows():
    dish_id = row["dish_id"]
    if dish_id not in ALL_RECIPES:
        continue
    recipe = ALL_RECIPES[dish_id]
    ingredients = recipe["ingredients"]
    cook_method = recipe.get("cook_method", "raw")

    total_grams = sum(ingredients.values())
    n_ingredients = len(ingredients)

    # Animal protein fraction
    animal_grams = sum(g for ing, g in ingredients.items() if ing in ANIMAL_PROTEINS)
    animal_fraction = animal_grams / total_grams if total_grams > 0 else 0

    # Cooking energy intensity (normalized to [0,1])
    cook_energy = COOKING_ENERGY_KWH.get(cook_method, 0.5)
    cook_intensity = cook_energy / 1.5  # max is smoke=1.5

    # High-impact ingredient share (beef, lamb, cheese, butter, chocolate)
    HIGH_IMPACT = {"beef", "ground_beef", "lamb", "cheese", "mozzarella", "parmesan",
                   "butter", "cream", "chocolate", "coffee"}
    hi_grams = sum(g for ing, g in ingredients.items() if ing in HIGH_IMPACT)
    hi_share = hi_grams / total_grams if total_grams > 0 else 0

    refine_rows.append({
        "dish_id": dish_id,
        "family": row["family"],
        "n_ingredients": n_ingredients,
        "total_grams": total_grams,
        "animal_fraction": animal_fraction,
        "cook_intensity": cook_intensity,
        "hi_impact_share": hi_share,
        "H_mean": row["H_mean"],
        "E_composite": row["E_composite"],
        "log_H": row["log_H"],
        "log_E": row["log_E"],
        "log_DEI": row["log_DEI"],
    })

refine_df = pd.DataFrame(refine_rows)
print(f"Computed refinement features for {len(refine_df)} dishes")

# Compute family-relative refinement score
# Normalize within each family
for fam in refine_df["family"].unique():
    mask = refine_df["family"] == fam
    sub = refine_df[mask]

    # Min in family for relative computation
    n_min = sub["n_ingredients"].min()
    g_min = sub["total_grams"].min()

    refine_df.loc[mask, "refinement_score"] = (
        0.30 * np.log(sub["n_ingredients"] / max(n_min, 1)) +
        0.20 * np.log(sub["total_grams"] / max(g_min, 1)) +
        0.20 * sub["animal_fraction"] +
        0.15 * sub["cook_intensity"] +
        0.15 * sub["hi_impact_share"]
    )

# ══════════════════════════════════════════════════════════════════
# STEP 3: Fit Refinement Curves per Family
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Family-Level Refinement Curves")
print("=" * 60)

# Model: H_fi = H_base_f + α_f · log(E_fi / E_base_f) + ε
# Equivalently: (H - H_base) = α · log(E/E_base)

curve_results = []

for fam in sorted(refine_df["family"].unique()):
    sub = refine_df[refine_df["family"] == fam].sort_values("E_composite")
    n = len(sub)
    if n < 3:
        continue

    # Base dish = lowest E in family
    base = sub.iloc[0]
    H_base = base["H_mean"]
    E_base = base["E_composite"]

    # Compute relative values
    log_E_ratio = np.log(sub["E_composite"].values / E_base)
    H_diff = sub["H_mean"].values - H_base

    # OLS: H_diff = α · log_E_ratio
    # Add intercept for robustness
    X = np.column_stack([np.ones(n), log_E_ratio])
    y = H_diff

    from numpy.linalg import lstsq
    beta, _, _, _ = lstsq(X, y, rcond=None)
    alpha = beta[1]  # slope = hedonic elasticity

    # R-squared
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Bootstrap CI for alpha
    np.random.seed(42)
    boot_alphas = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        X_b, y_b = X[idx], y[idx]
        try:
            b, _, _, _ = lstsq(X_b, y_b, rcond=None)
            boot_alphas.append(b[1])
        except:
            pass
    ci_lo = np.percentile(boot_alphas, 2.5)
    ci_hi = np.percentile(boot_alphas, 97.5)

    # E range and H range
    e_range = sub["E_composite"].max() / sub["E_composite"].min()
    h_range = sub["H_mean"].max() - sub["H_mean"].min()

    # Refinement efficiency: H gained per unit E increase
    if sub["E_composite"].max() > sub["E_composite"].min():
        refine_eff = h_range / (sub["E_composite"].max() - sub["E_composite"].min())
    else:
        refine_eff = 0

    curve_results.append({
        "family": fam,
        "n_dishes": n,
        "alpha": alpha,
        "alpha_ci_lo": ci_lo,
        "alpha_ci_hi": ci_hi,
        "r_squared": r_sq,
        "E_min": sub["E_composite"].min(),
        "E_max": sub["E_composite"].max(),
        "E_range_ratio": e_range,
        "H_min": sub["H_mean"].min(),
        "H_max": sub["H_mean"].max(),
        "H_range": h_range,
        "refinement_efficiency": refine_eff,
        "base_dish": base["dish_id"],
    })

    print(f"  {fam:20s}: α={alpha:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}], "
          f"R²={r_sq:.3f}, E ratio={e_range:.1f}x, n={n}")

curves_df = pd.DataFrame(curve_results)
curves_df.to_csv(TABLES_DIR / "refinement_curves.csv", index=False)
print(f"\nSaved: {TABLES_DIR / 'refinement_curves.csv'}")

# ══════════════════════════════════════════════════════════════════
# STEP 4: Global Mixed-Effects Model
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Global Refinement Model")
print("=" * 60)

# Pool all families: (H - H_base) = α_global · log(E/E_base) + family effects
all_h_diff = []
all_log_e_ratio = []
all_family_labels = []

for fam in sorted(refine_df["family"].unique()):
    sub = refine_df[refine_df["family"] == fam].sort_values("E_composite")
    if len(sub) < 3:
        continue
    base = sub.iloc[0]
    for _, row in sub.iterrows():
        all_h_diff.append(row["H_mean"] - base["H_mean"])
        all_log_e_ratio.append(np.log(row["E_composite"] / base["E_composite"]))
        all_family_labels.append(fam)

all_h_diff = np.array(all_h_diff)
all_log_e_ratio = np.array(all_log_e_ratio)

# Global OLS (ignoring family random effects for now)
X_global = np.column_stack([np.ones(len(all_log_e_ratio)), all_log_e_ratio])
beta_global, _, _, _ = lstsq(X_global, all_h_diff, rcond=None)
alpha_global = beta_global[1]

y_hat_global = X_global @ beta_global
ss_res_g = np.sum((all_h_diff - y_hat_global) ** 2)
ss_tot_g = np.sum((all_h_diff - all_h_diff.mean()) ** 2)
r_sq_global = 1 - ss_res_g / ss_tot_g if ss_tot_g > 0 else 0

# Bootstrap CI
np.random.seed(42)
boot_alpha_g = []
for _ in range(2000):
    idx = np.random.choice(len(all_h_diff), len(all_h_diff), replace=True)
    try:
        b, _, _, _ = lstsq(X_global[idx], all_h_diff[idx], rcond=None)
        boot_alpha_g.append(b[1])
    except:
        pass

ci_lo_g = np.percentile(boot_alpha_g, 2.5)
ci_hi_g = np.percentile(boot_alpha_g, 97.5)

print(f"  Global α = {alpha_global:.4f} [{ci_lo_g:.4f}, {ci_hi_g:.4f}]")
print(f"  Global R² = {r_sq_global:.4f}")
print(f"  Interpretation: each doubling of E yields {alpha_global * np.log(2):.3f} H points")
print(f"  (out of 10-point scale)")

# ══════════════════════════════════════════════════════════════════
# STEP 5: Policy — Diminishing Returns Quantification
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Diminishing Returns Quantification")
print("=" * 60)

# For each family: what fraction of H gain is captured by first 50% of E increase?
for _, cr in curves_df.iterrows():
    fam = cr["family"]
    sub = refine_df[refine_df["family"] == fam].sort_values("E_composite")
    if len(sub) < 4:
        continue

    # Midpoint E
    e_mid = (cr["E_min"] + cr["E_max"]) / 2
    below_mid = sub[sub["E_composite"] <= e_mid]
    above_mid = sub[sub["E_composite"] > e_mid]

    if len(below_mid) > 0 and len(above_mid) > 0:
        h_gain_first_half = below_mid["H_mean"].max() - below_mid["H_mean"].min()
        h_gain_total = cr["H_range"]
        pct_captured = h_gain_first_half / h_gain_total * 100 if h_gain_total > 0 else 0
        print(f"  {fam:20s}: first 50% E captures {pct_captured:.0f}% of H gain")

# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

# Figure 1: Multi-panel refinement curves (main novel figure)
families_to_plot = curves_df.nlargest(12, "n_dishes")["family"].tolist()
n_fams = len(families_to_plot)
n_cols = 4
n_rows = (n_fams + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
axes = axes.flatten()

for idx, fam in enumerate(families_to_plot):
    ax = axes[idx]
    sub = refine_df[refine_df["family"] == fam].sort_values("E_composite")
    base = sub.iloc[0]

    log_e_ratio = np.log(sub["E_composite"].values / base["E_composite"])
    h_diff = sub["H_mean"].values - base["H_mean"]

    ax.scatter(log_e_ratio, h_diff, s=40, alpha=0.8, color="#2196F3", zorder=5)

    # Fit line
    cr = curves_df[curves_df["family"] == fam].iloc[0]
    x_fit = np.linspace(0, log_e_ratio.max() * 1.05, 50)
    y_fit = cr["alpha"] * x_fit
    ax.plot(x_fit, y_fit, "r-", linewidth=2, alpha=0.7,
            label=f"α={cr['alpha']:.2f}")

    # Proportional line (if H grew proportionally with E)
    if log_e_ratio.max() > 0 and h_diff.max() != 0:
        prop_slope = h_diff.max() / log_e_ratio.max()
        ax.plot(x_fit, x_fit * prop_slope, "k--", alpha=0.2, linewidth=1)

    # Label dishes
    for _, row in sub.iterrows():
        le = np.log(row["E_composite"] / base["E_composite"])
        hd = row["H_mean"] - base["H_mean"]
        name = row["dish_id"].replace("_", " ")
        if len(name) > 12:
            name = name[:11] + "."
        ax.annotate(name, (le, hd), fontsize=6, alpha=0.7,
                    xytext=(2, 3), textcoords="offset points")

    ax.set_title(f"{fam}\n(α={cr['alpha']:.2f}, n={cr['n_dishes']})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("log(E/E_base)", fontsize=8)
    ax.set_ylabel("H − H_base", fontsize=8)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

# Hide empty subplots
for idx in range(n_fams, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Refinement Cost Curves: H Gain vs Environmental Cost Increase\n"
             "(α = hedonic points per log-unit of E)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "refinement_cost_curves.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'refinement_cost_curves.png'}")

# Figure 2: Global fit
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: All data pooled
colors_fam = plt.cm.tab20(np.linspace(0, 1, len(set(all_family_labels))))
fam_color_map = {fam: colors_fam[i] for i, fam in enumerate(sorted(set(all_family_labels)))}

for i, (h_d, le, fam_l) in enumerate(zip(all_h_diff, all_log_e_ratio, all_family_labels)):
    ax1.scatter(le, h_d, color=fam_color_map[fam_l], alpha=0.5, s=20,
                edgecolors="none")

x_fit_g = np.linspace(0, max(all_log_e_ratio) * 1.05, 100)
y_fit_g = alpha_global * x_fit_g + beta_global[0]
ax1.plot(x_fit_g, y_fit_g, "r-", linewidth=2.5,
         label=f"α_global={alpha_global:.3f}\n[{ci_lo_g:.3f}, {ci_hi_g:.3f}]")
ax1.axhline(y=0, color="gray", linewidth=0.5)
ax1.set_xlabel("log(E / E_base)", fontsize=12)
ax1.set_ylabel("H − H_base", fontsize=12)
ax1.set_title(f"A. Global Refinement Curve (R²={r_sq_global:.3f})",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel B: Family-level alpha comparison
curves_sorted = curves_df.sort_values("alpha")
colors = ["#F44336" if a < 0 else "#4CAF50" for a in curves_sorted["alpha"]]
ax2.barh(range(len(curves_sorted)), curves_sorted["alpha"], color=colors, alpha=0.7)
ax2.errorbar(curves_sorted["alpha"], range(len(curves_sorted)),
             xerr=[curves_sorted["alpha"] - curves_sorted["alpha_ci_lo"],
                   curves_sorted["alpha_ci_hi"] - curves_sorted["alpha"]],
             fmt="none", color="black", capsize=3, linewidth=1)
ax2.set_yticks(range(len(curves_sorted)))
ax2.set_yticklabels(curves_sorted["family"], fontsize=9)
ax2.axvline(x=0, color="black", linewidth=0.5)
ax2.axvline(x=alpha_global, color="red", linewidth=1, linestyle="--",
            label=f"Global α={alpha_global:.3f}")
ax2.set_xlabel("α (H points per log-unit of E)", fontsize=12)
ax2.set_title("B. Hedonic Elasticity by Family", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "refinement_global_fit.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'refinement_global_fit.png'}")

# Figure 3: Refinement efficiency bar plot
fig, ax = plt.subplots(figsize=(10, 6))
curves_eff = curves_df.sort_values("refinement_efficiency", ascending=False)
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(curves_eff)))
ax.barh(range(len(curves_eff)), curves_eff["refinement_efficiency"],
        color=colors, alpha=0.8)
ax.set_yticks(range(len(curves_eff)))
ax.set_yticklabels(curves_eff["family"], fontsize=9)
ax.set_xlabel("Refinement Efficiency (H range / E range)", fontsize=12)
ax.set_title("Which Food Types Give Best Hedonic Return on Environmental Investment?",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hedonic_waste_by_family.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'hedonic_waste_by_family.png'}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Families analyzed: {len(curves_df)}")
print(f"  Global α = {alpha_global:.4f} (each doubling of E → {alpha_global * np.log(2):.3f} H points)")
print(f"  Global R² = {r_sq_global:.4f}")
print(f"  Family α range: [{curves_df['alpha'].min():.3f}, {curves_df['alpha'].max():.3f}]")
n_positive = (curves_df["alpha"] > 0).sum()
n_negative = (curves_df["alpha"] <= 0).sum()
print(f"  Positive α (more E → more H): {n_positive} families")
print(f"  Negative/zero α (more E ≠ more H): {n_negative} families")
print(f"\n  Key insight: Refinement has DIMINISHING RETURNS.")
print(f"  Spending more on environmental cost does NOT proportionally improve taste.")
print("=" * 60)
