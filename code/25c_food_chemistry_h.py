#!/usr/bin/env python3
"""
25c_food_chemistry_h.py — Food Chemistry H + Three-Layer Reality
================================================================
Core analysis for the "perceptual decoupling" narrative:
  1. Compute H_chem from ingredient chemistry (fat, umami, sodium, sugar, spice diversity)
  2. Build three-layer correlation matrix (H_chem, H_text, E)
  3. Decompose H_text variance (chemistry vs non-chemistry)
  4. Counterfactual sensitivity: what if H_true = w*H_chem + (1-w)*H_nonchemical

Input:  data/combined_dish_DEI_v2.csv, data/ingredient_nutrients.csv,
        data/ingredient_impact_factors.csv, data/llm_generated_recipes.csv
Output: results/tables/food_chemistry_h_scores.csv
        results/tables/three_layer_correlation_matrix.csv
        results/tables/h_text_variance_sources.csv
        results/figures/three_layer_reality.png
        results/figures/h_chemistry_sensitivity.png
"""

import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import (DATA_DIR, TABLES_DIR, FIGURES_DIR,
                    COOKING_ENERGY_KWH)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Umami scores per ingredient (relative 0-10 scale) ──
# Based on glutamate/nucleotide content from food science literature
UMAMI_SCORES = {
    "parmesan": 10, "soy_sauce": 9, "fish_sauce": 9, "miso": 8,
    "mushroom": 7, "tomato": 6, "tomato_sauce": 6, "anchovy": 8,
    "beef": 5, "pork": 4, "chicken": 4, "lamb": 5, "duck": 5,
    "ground_beef": 5, "bacon": 6, "sausage": 4, "turkey": 3,
    "fish": 5, "salmon": 5, "cod": 4, "shrimp": 5, "squid": 4, "crab": 5,
    "cheese": 6, "mozzarella": 5, "cream": 2, "butter": 2,
    "egg": 3, "milk": 2, "yogurt": 2,
    "tofu": 3, "soybean": 4, "edamame": 3,
    "seaweed": 8, "kelp": 9, "bonito": 9,
    "hoisin_sauce": 6, "oyster_sauce": 7,
}

SODIUM_MG_PER_G = {
    "salt": 388, "soy_sauce": 57, "fish_sauce": 73, "miso": 38,
    "hoisin_sauce": 25, "oyster_sauce": 30, "vinegar": 0.3,
    "cheese": 6, "parmesan": 15, "mozzarella": 5, "bacon": 11,
    "bread": 5, "tortilla": 5, "pita": 5, "butter": 6,
    "mayonnaise": 6, "tomato_sauce": 3, "curry_paste": 15,
}


def compute_h_chem(dish_id, recipe, nutrients_df):
    """Compute food chemistry hedonic score for a dish from its recipe."""
    ingredients = recipe.get("ingredients", {})
    if not ingredients:
        return None

    total_g = sum(ingredients.values())
    if total_g == 0:
        return None

    # 1. Fat content (per serving)
    fat_g = 0
    for ing, grams in ingredients.items():
        if ing in nutrients_df.index:
            fat_g += (grams / 1000) * nutrients_df.loc[ing, "fat_g"]

    # 2. Umami composite (weighted by grams)
    umami = 0
    for ing, grams in ingredients.items():
        umami += (grams / total_g) * UMAMI_SCORES.get(ing, 0)

    # 3. Sodium (mg total)
    sodium_mg = 0
    for ing, grams in ingredients.items():
        sodium_mg += grams * SODIUM_MG_PER_G.get(ing, 0.1)  # default trace

    # 4. Sugar content (crude proxy from nutrients)
    sugar_g = 0
    for ing, grams in ingredients.items():
        if ing in nutrients_df.index:
            # Use carb_g as proxy; sugar-heavy items have high carb
            carb = nutrients_df.loc[ing, "carb_g"]
            # Adjust: fruits/sugar/honey are high-sugar carbs
            if ing in {"sugar", "honey", "chocolate", "mango", "pineapple",
                       "banana", "apple", "pear", "coconut_milk"}:
                sugar_g += (grams / 1000) * carb * 0.8
            else:
                sugar_g += (grams / 1000) * carb * 0.15

    # 5. Spice diversity (number of distinct spice/herb ingredients)
    spice_herbs = {"chili", "cumin", "turmeric", "ginger", "garlic", "black_pepper",
                   "coriander", "basil", "cinnamon", "cardamom", "clove", "nutmeg",
                   "oregano", "thyme", "rosemary", "mint", "dill", "paprika",
                   "curry_paste", "star_anise", "lemongrass", "saffron", "vanilla"}
    spice_count = sum(1 for ing in ingredients if ing in spice_herbs)

    # 6. Protein (for reference, not in H_chem)
    protein_g = 0
    for ing, grams in ingredients.items():
        if ing in nutrients_df.index:
            protein_g += (grams / 1000) * nutrients_df.loc[ing, "protein_g"]

    # 7. Calorie
    calorie = 0
    for ing, grams in ingredients.items():
        if ing in nutrients_df.index:
            calorie += (grams / 1000) * nutrients_df.loc[ing, "calorie_kcal"]

    return {
        "dish_id": dish_id,
        "fat_g": fat_g,
        "umami_score": umami,
        "sodium_mg": sodium_mg,
        "sugar_g": sugar_g,
        "spice_count": spice_count,
        "protein_g": protein_g,
        "calorie_kcal": calorie,
        "total_g": total_g,
    }


def load_all_recipes():
    """Load both human and LLM recipes into a unified dict."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location("env_cost", ROOT / "code" / "04_env_cost_calculation.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DISH_RECIPES = _mod.DISH_RECIPES

    # Start with human recipes
    recipes = {}
    for dish_id, recipe in DISH_RECIPES.items():
        recipes[dish_id] = recipe

    # Add LLM recipes (don't overwrite human ones)
    llm_path = DATA_DIR / "llm_generated_recipes.csv"
    if llm_path.exists():
        llm_df = pd.read_csv(llm_path)
        for _, row in llm_df.iterrows():
            dish_id = row["dish_id"]
            if dish_id in recipes:
                continue
            try:
                ings_list = json.loads(row["ingredients_json"])
                ing_dict = {item["name"]: item["grams"] for item in ings_list
                            if item.get("name") and item.get("grams", 0) > 0}
                cook = row.get("cook_method", "unknown")
                recipes[dish_id] = {"ingredients": ing_dict, "cook_method": cook}
            except Exception:
                pass

    return recipes


def main():
    print("=" * 60, flush=True)
    print("25c — Food Chemistry H + Three-Layer Reality", flush=True)
    print("=" * 60, flush=True)

    # Load data
    dei = pd.read_csv(DATA_DIR / "combined_dish_DEI_v2.csv")
    nutrients = pd.read_csv(DATA_DIR / "ingredient_nutrients.csv").set_index("ingredient")
    recipes = load_all_recipes()

    print(f"  DEI dishes: {len(dei):,}", flush=True)
    print(f"  Nutrient ingredients: {len(nutrients)}", flush=True)
    print(f"  Recipes loaded: {len(recipes)}", flush=True)

    # ── Step 1: Compute H_chem for all dishes with recipes ──
    print(f"\n── Step 1: Computing food chemistry attributes ──", flush=True)
    chem_records = []
    for dish_id, recipe in recipes.items():
        result = compute_h_chem(dish_id, recipe, nutrients)
        if result:
            chem_records.append(result)

    chem_df = pd.DataFrame(chem_records).set_index("dish_id")
    print(f"  Chemistry scores for {len(chem_df)} dishes", flush=True)

    # Standardize and combine into H_chem composite
    chem_cols = ["fat_g", "umami_score", "sodium_mg", "sugar_g", "spice_count"]
    scaler = StandardScaler()
    chem_z = pd.DataFrame(scaler.fit_transform(chem_df[chem_cols]),
                          index=chem_df.index, columns=[f"{c}_z" for c in chem_cols])

    # Equal-weight composite (each dimension contributes equally)
    chem_df["H_chem"] = chem_z.mean(axis=1)
    # Rescale to 1-10 for interpretability
    hc = chem_df["H_chem"]
    if hc.max() > hc.min():
        chem_df["H_chem_scaled"] = 1 + 9 * (hc - hc.min()) / (hc.max() - hc.min())
    else:
        chem_df["H_chem_scaled"] = 5.5

    # Save
    chem_df.to_csv(TABLES_DIR / "food_chemistry_h_scores.csv")
    print(f"  Saved: {TABLES_DIR / 'food_chemistry_h_scores.csv'}", flush=True)

    # ── Step 2: Three-layer correlation matrix ──
    print(f"\n── Step 2: Three-layer correlation matrix ──", flush=True)

    # Merge
    dei_idx = dei.set_index("dish_id")
    # Select chem columns, avoiding overlap with DEI columns
    chem_join_cols = ["H_chem_scaled", "fat_g", "umami_score",
                      "sodium_mg", "sugar_g", "spice_count"]
    # Only add protein_g/calorie_kcal from chem if not already in DEI
    for c in ["protein_g", "calorie_kcal"]:
        if c not in dei_idx.columns:
            chem_join_cols.append(c)
    merged = dei_idx.join(chem_df[chem_join_cols], how="inner")
    # Ensure protein_g and calorie_kcal exist (from either source)
    if "protein_g" not in merged.columns and "protein_g" in chem_df.columns:
        merged["protein_g"] = chem_df.loc[merged.index, "protein_g"]
    if "calorie_kcal" not in merged.columns and "calorie_kcal" in chem_df.columns:
        merged["calorie_kcal"] = chem_df.loc[merged.index, "calorie_kcal"]
    merged = merged.dropna(subset=["H_chem_scaled", "H_mean", "E_composite"])
    print(f"  Matched dishes: {len(merged)}", flush=True)

    H_text = merged["H_mean"]
    H_chem = merged["H_chem_scaled"]
    E = merged["E_composite"]
    log_E = merged["log_E"]

    # Pearson correlations
    pairs = [
        ("H_text", "E", H_text, E),
        ("H_chem", "E", H_chem, E),
        ("H_text", "H_chem", H_text, H_chem),
        ("H_text", "log_E", H_text, log_E),
        ("H_chem", "log_E", H_chem, log_E),
    ]

    print(f"\n  {'Pair':<25s} {'Pearson r':>10s} {'Spearman ρ':>12s} {'p-value':>12s}", flush=True)
    print(f"  {'-'*60}", flush=True)
    corr_records = []
    for name, name2, x, y in pairs:
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        print(f"  {name+' vs '+name2:<25s} {r_p:>10.4f} {r_s:>12.4f} {p_p:>12.2e}", flush=True)
        corr_records.append({"var1": name, "var2": name2,
                             "pearson_r": r_p, "spearman_rho": r_s,
                             "p_pearson": p_p, "p_spearman": p_s, "n": len(x)})

    # Individual chemistry components vs E
    print(f"\n  Chemistry components vs E:", flush=True)
    for col in ["fat_g", "protein_g", "calorie_kcal", "umami_score", "sodium_mg",
                "sugar_g", "spice_count"]:
        if col in merged.columns:
            r, p = stats.pearsonr(merged[col], E)
            rho, _ = stats.spearmanr(merged[col], E)
            print(f"    {col:<18s} r={r:+.4f} ρ={rho:+.4f} (p={p:.2e})", flush=True)
            corr_records.append({"var1": col, "var2": "E",
                                 "pearson_r": r, "spearman_rho": rho,
                                 "p_pearson": p, "n": len(merged)})

    # Individual chemistry vs H_text
    print(f"\n  Chemistry components vs H_text:", flush=True)
    for col in ["fat_g", "protein_g", "calorie_kcal", "umami_score", "sodium_mg",
                "sugar_g", "spice_count"]:
        if col in merged.columns:
            r, p = stats.pearsonr(merged[col], H_text)
            rho, _ = stats.spearmanr(merged[col], H_text)
            print(f"    {col:<18s} r={r:+.4f} ρ={rho:+.4f} (p={p:.2e})", flush=True)
            corr_records.append({"var1": col, "var2": "H_text",
                                 "pearson_r": r, "spearman_rho": rho,
                                 "p_pearson": p, "n": len(merged)})

    corr_df = pd.DataFrame(corr_records)
    corr_df.to_csv(TABLES_DIR / "three_layer_correlation_matrix.csv", index=False)
    print(f"\n  Saved: {TABLES_DIR / 'three_layer_correlation_matrix.csv'}", flush=True)

    # ── Step 3: H_text variance decomposition ──
    print(f"\n── Step 3: H_text variance decomposition ──", flush=True)

    from sklearn.linear_model import LinearRegression

    var_records = []

    # Model 1: H_text ~ chemistry only
    chem_features = ["fat_g", "protein_g", "calorie_kcal"]
    X1 = merged[chem_features].values
    y = H_text.values
    lr1 = LinearRegression().fit(X1, y)
    r2_chem = lr1.score(X1, y)
    print(f"  R²(H_text ~ fat+protein+cal) = {r2_chem:.4f} ({r2_chem*100:.2f}%)", flush=True)
    var_records.append({"model": "chemistry_basic", "features": "fat+protein+calorie",
                        "R2": r2_chem, "n": len(merged)})

    # Model 2: H_text ~ full chemistry
    full_chem = ["fat_g", "protein_g", "calorie_kcal", "umami_score", "sodium_mg",
                 "sugar_g", "spice_count"]
    X2 = merged[full_chem].values
    lr2 = LinearRegression().fit(X2, y)
    r2_fullchem = lr2.score(X2, y)
    print(f"  R²(H_text ~ full chemistry) = {r2_fullchem:.4f} ({r2_fullchem*100:.2f}%)", flush=True)
    var_records.append({"model": "chemistry_full", "features": "+".join(full_chem),
                        "R2": r2_fullchem, "n": len(merged)})

    # Model 3: H_text ~ cook_method (one-hot)
    cook_dummies = pd.get_dummies(merged["cook_method"], prefix="cook", drop_first=True)
    X3 = cook_dummies.values
    if X3.shape[1] > 0:
        lr3 = LinearRegression().fit(X3, y)
        r2_cook = lr3.score(X3, y)
    else:
        r2_cook = 0.0
    print(f"  R²(H_text ~ cook_method) = {r2_cook:.4f} ({r2_cook*100:.2f}%)", flush=True)
    var_records.append({"model": "cook_method", "features": "cook_method_dummies",
                        "R2": r2_cook, "n": len(merged)})

    # Model 4: Combined
    X4 = np.hstack([X2, X3])
    lr4 = LinearRegression().fit(X4, y)
    r2_combined = lr4.score(X4, y)
    print(f"  R²(H_text ~ chemistry+cooking) = {r2_combined:.4f} ({r2_combined*100:.2f}%)", flush=True)
    var_records.append({"model": "combined", "features": "chemistry+cook_method",
                        "R2": r2_combined, "n": len(merged)})

    # Model 5: H_text ~ E alone
    X5 = E.values.reshape(-1, 1)
    lr5 = LinearRegression().fit(X5, y)
    r2_e = lr5.score(X5, y)
    print(f"  R²(H_text ~ E) = {r2_e:.4f} ({r2_e*100:.2f}%)", flush=True)
    var_records.append({"model": "E_only", "features": "E_composite",
                        "R2": r2_e, "n": len(merged)})

    unexplained = 1 - r2_combined
    print(f"\n  Unexplained by chemistry+cooking: {unexplained*100:.1f}%", flush=True)

    var_df = pd.DataFrame(var_records)
    var_df.to_csv(TABLES_DIR / "h_text_variance_sources.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'h_text_variance_sources.csv'}", flush=True)

    # ── Step 4: Counterfactual sensitivity ──
    print(f"\n── Step 4: Counterfactual sensitivity ──", flush=True)

    # H_nonchemical = H_text residualized against chemistry
    H_text_arr = H_text.values
    H_chem_arr = H_chem.values
    E_arr = E.values

    # Residualize H_text against H_chem
    from numpy.polynomial.polynomial import polyfit
    slope, intercept = np.polyfit(H_chem_arr, H_text_arr, 1)
    H_nonchemical = H_text_arr - (slope * H_chem_arr + intercept)
    # Standardize
    H_nonchemical_z = (H_nonchemical - H_nonchemical.mean()) / H_nonchemical.std()
    H_chem_z = (H_chem_arr - H_chem_arr.mean()) / H_chem_arr.std()

    weights = np.arange(0, 1.01, 0.05)
    sensitivity = []
    for w in weights:
        H_true = w * H_chem_z + (1 - w) * H_nonchemical_z
        r_true_e, _ = stats.pearsonr(H_true, E_arr)
        rho_true_e, _ = stats.spearmanr(H_true, E_arr)

        # Variance decomposition with H_true
        log_H_true = np.log(np.clip(H_true - H_true.min() + 1, 0.01, None))
        log_E_arr = np.log(np.clip(E_arr, 0.001, None))
        var_lh = np.var(log_H_true)
        var_le = np.var(log_E_arr)
        h_pct = var_lh / (var_lh + var_le) * 100

        sensitivity.append({
            "w_chem": w,
            "r_H_true_E": r_true_e,
            "rho_H_true_E": rho_true_e,
            "H_contribution_pct": h_pct,
        })

    sens_df = pd.DataFrame(sensitivity)
    print(f"  Key checkpoints:", flush=True)
    for w in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        row = sens_df[sens_df["w_chem"].round(2) == w]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"    w_chem={w:.1f}: r(H_true,E)={r['r_H_true_E']:+.4f}, "
                  f"ρ={r['rho_H_true_E']:+.4f}", flush=True)

    # ── Step 5: Visualization ──
    print(f"\n── Step 5: Visualization ──", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: Three-layer reality (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: H_text vs E
    ax = axes[0]
    ax.scatter(E, H_text, alpha=0.15, s=8, c="#2077B4")
    r_te, _ = stats.pearsonr(H_text, E)
    ax.set_xlabel("E (environmental cost)", fontsize=11)
    ax.set_ylabel("H_text (consumer rating)", fontsize=11)
    ax.set_title(f"Consumer: r = {r_te:.3f} (decoupled)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel B: H_chem vs E
    ax = axes[1]
    ax.scatter(E, H_chem, alpha=0.15, s=8, c="#E24A33")
    r_ce, _ = stats.pearsonr(H_chem, E)
    sl, ic = np.polyfit(E, H_chem, 1)
    xr = np.linspace(E.min(), E.max(), 50)
    ax.plot(xr, sl * xr + ic, "k--", lw=1.5, alpha=0.5)
    ax.set_xlabel("E (environmental cost)", fontsize=11)
    ax.set_ylabel("H_chem (food chemistry)", fontsize=11)
    ax.set_title(f"Chemistry: r = {r_ce:.3f} (coupled)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel C: H_text vs H_chem
    ax = axes[2]
    ax.scatter(H_chem, H_text, alpha=0.15, s=8, c="#8EBA42")
    r_tc, _ = stats.pearsonr(H_text, H_chem)
    ax.set_xlabel("H_chem (food chemistry)", fontsize=11)
    ax.set_ylabel("H_text (consumer rating)", fontsize=11)
    ax.set_title(f"Text↔Chem: r = {r_tc:.3f} (weak)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Three-Layer Reality: Chemistry Couples with E, Consumers Don't",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "three_layer_reality.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {FIGURES_DIR / 'three_layer_reality.png'}", flush=True)
    plt.close()

    # Figure 2: Counterfactual sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(sens_df["w_chem"], sens_df["r_H_true_E"], "o-", color="#2077B4", lw=2, ms=4)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0.3, color="red", ls="--", lw=1, alpha=0.5, label="Plausible range")
    ax.axvline(0.5, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Weight on chemistry (w_chem)", fontsize=11)
    ax.set_ylabel("r(H_true, E)", fontsize=11)
    ax.set_title("How r(H,E) Changes with Chemistry Weight", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sens_df["w_chem"], sens_df["H_contribution_pct"], "o-", color="#E87D2F", lw=2, ms=4)
    ax.set_xlabel("Weight on chemistry (w_chem)", fontsize=11)
    ax.set_ylabel("H contribution to Var(log DEI) (%)", fontsize=11)
    ax.set_title("Variance Decomposition Sensitivity", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Counterfactual: What If H Captured More Chemistry?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "h_chemistry_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {FIGURES_DIR / 'h_chemistry_sensitivity.png'}", flush=True)
    plt.close()

    # ── Top/Bottom examples ──
    print(f"\n── Top/Bottom examples ──", flush=True)
    merged_full = merged.join(chem_df[["H_chem_scaled"]], rsuffix="_chem2")
    print(f"\n  Top 10 H_chem (most 'chemically delicious'):", flush=True)
    top_chem = chem_df.nlargest(10, "H_chem_scaled")
    for dish, row in top_chem.iterrows():
        e_val = merged.loc[dish, "E_composite"] if dish in merged.index else np.nan
        h_text = merged.loc[dish, "H_mean"] if dish in merged.index else np.nan
        print(f"    {dish:30s} H_chem={row['H_chem_scaled']:.2f} "
              f"H_text={h_text:.2f} E={e_val:.3f}", flush=True)

    print(f"\n  Top 10 H_text but LOW H_chem (perception >> chemistry):", flush=True)
    if len(merged) > 0:
        merged_with_chem = merged.copy()
        merged_with_chem["H_chem_scaled"] = chem_df.loc[merged.index, "H_chem_scaled"]
        merged_with_chem = merged_with_chem.dropna(subset=["H_chem_scaled"])
        merged_with_chem["gap"] = (merged_with_chem["H_mean"] -
                                   merged_with_chem["H_mean"].mean()) - \
                                  (merged_with_chem["H_chem_scaled"] -
                                   merged_with_chem["H_chem_scaled"].mean())
        top_gap = merged_with_chem.nlargest(10, "gap")
        for dish, row in top_gap.iterrows():
            print(f"    {dish:30s} H_text={row['H_mean']:.2f} "
                  f"H_chem={row['H_chem_scaled']:.2f} E={row['E_composite']:.3f}",
                  flush=True)

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
