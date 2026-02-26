"""
11d_within_category_analysis.py — Within-Category DEI Analysis
===============================================================
Addresses criticism C4: Comparing across categories (salad vs main)
inflates apparent substitutability. brisket → ceviche ignores
caloric/protein adequacy.

Steps:
  1. Recipe-based functional category assignment
  2. Within-category DEI rankings and variance decomposition
  3. Nutritionally-constrained substitution matrix
  4. Category-aggregate DEI summary

Depends on: 11c (dish_nutritional_profiles.csv)

Outputs:
  - tables/comprehensive_category_assignment.csv
  - tables/within_category_dei_rankings.csv
  - tables/within_category_variance.csv
  - tables/nutrition_constrained_substitutions_full.csv
  - figures/within_category_dei_panels.png
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ── Load data ────────────────────────────────────────────────────
print("=" * 60)
print("11d: Within-Category DEI Analysis")
print("=" * 60)

dei = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
nutr = pd.read_csv(DATA_DIR / "dish_nutritional_profiles.csv")
print(f"Loaded {len(dei)} dishes, {len(nutr)} nutritional profiles")

merged = dei.merge(nutr, on="dish_id", how="inner")
print(f"Merged: {len(merged)} dishes with both DEI and nutrition data")

# ── Load recipes for classification ──
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
print(f"Loaded {len(ALL_RECIPES)} recipes")

# ══════════════════════════════════════════════════════════════════
# STEP 1: Recipe-Based Functional Category Assignment
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Recipe-Based Category Assignment")
print("=" * 60)

RED_MEAT = {"beef", "ground_beef", "lamb", "veal"}
POULTRY = {"chicken", "duck", "turkey"}
SEAFOOD = {"fish", "salmon", "shrimp", "squid", "tuna", "cod", "crab"}
LEGUME = {"black_bean", "chickpea", "lentil", "soybean", "tofu", "peanut"}
DAIRY_HEAVY = {"cheese", "mozzarella", "parmesan", "cream", "butter", "yogurt"}
DESSERT_INDICATORS = {"sugar", "chocolate", "honey", "maple_syrup", "vanilla"}
BEVERAGE_DISHES = {"thai_iced_tea", "ca_phe_sua_da", "coquito", "halo_halo"}

def classify_dish(dish_id, recipe, cal):
    """Classify dish by dominant protein source and function."""
    ingredients = recipe["ingredients"]
    cook_method = recipe.get("cook_method", "raw")

    # Check known beverages
    if dish_id in BEVERAGE_DISHES:
        return "Beverage"

    # Sum weights by category
    red_g = sum(g for ing, g in ingredients.items() if ing in RED_MEAT)
    poultry_g = sum(g for ing, g in ingredients.items() if ing in POULTRY)
    seafood_g = sum(g for ing, g in ingredients.items() if ing in SEAFOOD)
    pork_g = sum(g for ing, g in ingredients.items() if ing == "pork" or ing == "bacon" or ing == "sausage")
    legume_g = sum(g for ing, g in ingredients.items() if ing in LEGUME)
    dairy_g = sum(g for ing, g in ingredients.items() if ing in DAIRY_HEAVY)
    dessert_g = sum(g for ing, g in ingredients.items() if ing in DESSERT_INDICATORS)
    total_g = sum(ingredients.values())

    # Dessert detection: high sugar content or known dessert names
    dessert_names = {"brownie", "cheesecake", "gelato", "ice_cream", "panna_cotta",
                     "tiramisu", "creme_brulee", "baklava", "churro", "churros_mexican",
                     "eclair", "profiterole", "macarons", "cannoli", "gulab_jamun",
                     "jalebi", "galaktoboureko", "kunefe", "pasteis_de_nata",
                     "loukoumades", "mango_sticky_rice", "dan_tat"}
    if dish_id in dessert_names or (dessert_g / total_g > 0.15 if total_g > 0 else False):
        return "Dessert"

    # Soup detection
    soup_names = {"miso_soup", "tom_yum", "tom_kha", "clam_chowder", "french_onion_soup",
                  "gazpacho", "hot_and_sour_soup", "borscht", "caldo_verde",
                  "sancocho", "sinigang", "callaloo", "egusi_soup"}
    if dish_id in soup_names:
        return "Soup/Stew"

    # Raw/cold salad detection
    if cook_method in ("raw", "cold") and cal < 300:
        return "Salad/Cold"

    # Protein-based classification
    if red_g >= 100:
        return "Red Meat Main"
    if pork_g >= 100:
        return "Pork Main"
    if poultry_g >= 100:
        return "Poultry Main"
    if seafood_g >= 80:
        return "Seafood Main"
    if legume_g >= 60 and red_g < 50 and poultry_g < 50 and pork_g < 50:
        return "Plant Protein"

    # Starch/carb dominant
    grain_items = {"rice", "pasta_dry", "rice_noodle", "wheat_flour", "bread",
                   "tortilla", "corn", "oats", "pita"}
    grain_g = sum(g for ing, g in ingredients.items() if ing in grain_items)
    if grain_g / total_g > 0.4 if total_g > 0 else False:
        return "Starch/Carb"

    # Egg dominant
    egg_g = ingredients.get("egg", 0)
    if egg_g >= 80 and red_g < 50 and poultry_g < 50:
        return "Egg Dish"

    # Dairy dominant
    if dairy_g / total_g > 0.3 if total_g > 0 else False:
        return "Dairy Main"

    return "Mixed/Other"

# Apply classification
cat_rows = []
for _, row in merged.iterrows():
    dish_id = row["dish_id"]
    cal = row.get("calorie_kcal", 500)
    if dish_id in ALL_RECIPES:
        cat = classify_dish(dish_id, ALL_RECIPES[dish_id], cal)
    else:
        cat = "Mixed/Other"
    cat_rows.append({"dish_id": dish_id, "category_recipe": cat})

cat_df = pd.DataFrame(cat_rows)
merged = merged.merge(cat_df, on="dish_id")

print("\nCategory distribution:")
for cat, count in merged["category_recipe"].value_counts().items():
    pct = count / len(merged) * 100
    print(f"  {cat:20s}: {count:3d} dishes ({pct:.1f}%)")

other_pct = (merged["category_recipe"] == "Mixed/Other").sum() / len(merged) * 100
print(f"\n  'Mixed/Other' percentage: {other_pct:.1f}% (target: <15%)")

# Save
merged[["dish_id", "category_recipe"]].to_csv(
    TABLES_DIR / "comprehensive_category_assignment.csv", index=False)

# ══════════════════════════════════════════════════════════════════
# STEP 2: Within-Category DEI Rankings & Variance
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Within-Category DEI Rankings & Variance Decomposition")
print("=" * 60)

var_rows = []
rank_rows = []

# Global H CV for comparison
global_h_cv = merged["H_mean"].std() / merged["H_mean"].mean() * 100
global_h_var_pct = merged["log_H"].var() / merged["log_DEI"].var() * 100

print(f"\n  Global: H CV={global_h_cv:.2f}%, H contributes {global_h_var_pct:.2f}% of Var(log DEI)")

for cat in sorted(merged["category_recipe"].unique()):
    sub = merged[merged["category_recipe"] == cat]
    n = len(sub)
    if n < 3:
        continue

    # Within-category stats
    h_cv = sub["H_mean"].std() / sub["H_mean"].mean() * 100
    e_cv = sub["E_composite"].std() / sub["E_composite"].mean() * 100

    var_log_h = sub["log_H"].var()
    var_log_e = sub["log_E"].var()
    var_log_dei = sub["log_DEI"].var()
    h_pct = var_log_h / var_log_dei * 100 if var_log_dei > 0 else 0

    var_rows.append({
        "category": cat,
        "n_dishes": n,
        "H_cv_pct": h_cv,
        "E_cv_pct": e_cv,
        "H_contribution_pct": h_pct,
        "H_mean": sub["H_mean"].mean(),
        "E_mean": sub["E_composite"].mean(),
        "log_DEI_mean": sub["log_DEI"].mean(),
        "protein_mean": sub["protein_g"].mean(),
        "cal_mean": sub["calorie_kcal"].mean(),
    })

    # Within-category ranking
    sub_ranked = sub.copy()
    sub_ranked["within_rank"] = sub_ranked["log_DEI"].rank(ascending=False)
    for _, row in sub_ranked.iterrows():
        rank_rows.append({
            "dish_id": row["dish_id"],
            "category_recipe": cat,
            "H_mean": row["H_mean"],
            "E_composite": row["E_composite"],
            "log_DEI": row["log_DEI"],
            "within_rank": int(row["within_rank"]),
            "n_in_category": n,
            "protein_g": row["protein_g"],
            "calorie_kcal": row["calorie_kcal"],
        })

    print(f"  {cat:20s}: n={n:3d}, H CV={h_cv:.2f}%, E CV={e_cv:.2f}%, "
          f"H contrib={h_pct:.1f}%")

var_df = pd.DataFrame(var_rows)
var_df.to_csv(TABLES_DIR / "within_category_variance.csv", index=False)

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(TABLES_DIR / "within_category_dei_rankings.csv", index=False)

print(f"\n  Key finding: within-category H CV is "
      f"{'HIGHER' if var_df['H_cv_pct'].mean() > global_h_cv else 'LOWER'} "
      f"than global ({var_df['H_cv_pct'].mean():.2f}% vs {global_h_cv:.2f}%)")

# ══════════════════════════════════════════════════════════════════
# STEP 3: Nutrition-Constrained Substitution Matrix
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Nutrition-Constrained Substitutions")
print("=" * 60)

# For each high-E dish, find within-category substitutes that meet nutritional criteria
sub_rows = []
n_total_subs = 0

for cat in merged["category_recipe"].unique():
    cat_dishes = merged[merged["category_recipe"] == cat]
    if len(cat_dishes) < 2:
        continue

    for _, from_dish in cat_dishes.iterrows():
        for _, to_dish in cat_dishes.iterrows():
            if from_dish["dish_id"] == to_dish["dish_id"]:
                continue
            if to_dish["E_composite"] >= from_dish["E_composite"]:
                continue

            e_reduction = (from_dish["E_composite"] - to_dish["E_composite"]) / from_dish["E_composite"]
            h_loss = from_dish["H_mean"] - to_dish["H_mean"]

            # Nutritional constraints
            protein_ok = to_dish["protein_g"] >= from_dish["protein_g"] * 0.5
            cal_ok = abs(to_dish["calorie_kcal"] - from_dish["calorie_kcal"]) / max(from_dish["calorie_kcal"], 1) < 0.5
            e_ok = e_reduction >= 0.30  # ≥30% E reduction
            h_ok = h_loss < 1.0  # <1 point H loss

            if e_ok and h_ok and protein_ok and cal_ok:
                sub_rows.append({
                    "from_dish": from_dish["dish_id"],
                    "to_dish": to_dish["dish_id"],
                    "category": cat,
                    "E_reduction_pct": e_reduction * 100,
                    "H_change": -h_loss,
                    "from_protein_g": from_dish["protein_g"],
                    "to_protein_g": to_dish["protein_g"],
                    "from_cal": from_dish["calorie_kcal"],
                    "to_cal": to_dish["calorie_kcal"],
                    "protein_ratio": to_dish["protein_g"] / max(from_dish["protein_g"], 0.1),
                })
                n_total_subs += 1

subs_df = pd.DataFrame(sub_rows)
if len(subs_df) > 0:
    subs_df = subs_df.sort_values("E_reduction_pct", ascending=False)
    subs_df.to_csv(TABLES_DIR / "nutrition_constrained_substitutions_full.csv", index=False)
    print(f"  Found {n_total_subs} viable substitutions across all categories")
    print(f"  Mean E reduction: {subs_df['E_reduction_pct'].mean():.1f}%")
    print(f"  Mean H change: {subs_df['H_change'].mean():.3f}")

    # Summary by category
    print(f"\n  Substitutions by category:")
    for cat, group in subs_df.groupby("category"):
        unique_from = group["from_dish"].nunique()
        print(f"    {cat:20s}: {len(group):4d} options for {unique_from} dishes, "
              f"mean E reduction={group['E_reduction_pct'].mean():.1f}%")

    # Showcase top substitutions
    print(f"\n  Top 10 substitutions (highest E reduction, nutrition-constrained):")
    for _, row in subs_df.head(10).iterrows():
        print(f"    {row['from_dish']:25s} → {row['to_dish']:25s}: "
              f"E-{row['E_reduction_pct']:.0f}%, H{row['H_change']:+.2f}, "
              f"protein {row['from_protein_g']:.0f}→{row['to_protein_g']:.0f}g")
else:
    print("  No substitutions found meeting all constraints")

print(f"\n  Saved: {TABLES_DIR / 'nutrition_constrained_substitutions_full.csv'}")

# ══════════════════════════════════════════════════════════════════
# FIGURE: Multi-panel within-category scatter plots
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

cats_to_plot = var_df.nlargest(9, "n_dishes")["category"].tolist()
n_cats = len(cats_to_plot)
n_cols = 3
n_rows = (n_cats + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for idx, cat in enumerate(cats_to_plot):
    ax = axes[idx]
    sub = merged[merged["category_recipe"] == cat]

    ax.scatter(sub["E_composite"], sub["H_mean"], s=40, alpha=0.7,
               c="#2196F3", edgecolors="none")

    # Label dishes
    for _, row in sub.iterrows():
        name = row["dish_id"].replace("_", " ")
        if len(name) > 15:
            name = name[:14] + "."
        ax.annotate(name, (row["E_composite"], row["H_mean"]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    vr = var_df[var_df["category"] == cat].iloc[0]
    ax.set_title(f"{cat} (n={int(vr['n_dishes'])})\n"
                 f"H contrib={vr['H_contribution_pct']:.1f}%, "
                 f"H CV={vr['H_cv_pct']:.1f}%",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("E (environmental cost)", fontsize=9)
    ax.set_ylabel("H (hedonic score)", fontsize=9)
    ax.grid(True, alpha=0.2)

for idx in range(n_cats, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Within-Category H vs E: Is H More Discriminating Within Categories?",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "within_category_dei_panels.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'within_category_dei_panels.png'}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Categories: {len(var_df)} (with ≥3 dishes)")
print(f"  'Mixed/Other': {other_pct:.1f}%")
print(f"  Global H CV: {global_h_cv:.2f}%")
print(f"  Mean within-category H CV: {var_df['H_cv_pct'].mean():.2f}%")
print(f"  Mean within-category H contribution: {var_df['H_contribution_pct'].mean():.1f}%")
print(f"  Nutrition-constrained substitutions: {n_total_subs}")
if len(subs_df) > 0:
    print(f"  Mean E reduction in viable subs: {subs_df['E_reduction_pct'].mean():.1f}%")
print("=" * 60)
