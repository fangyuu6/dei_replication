#!/usr/bin/env python3
"""
25a_recipe_validation.py — Validate LLM-generated recipes against human-authored ones
======================================================================================
For the 158 dishes with human recipes (DISH_RECIPES in 04_env_cost_calculation.py),
regenerate recipes via DeepSeek v3.2 and compare:
  1. Ingredient overlap (Jaccard index)
  2. Gram-weight MAE per shared ingredient
  3. E score Spearman ρ (LLM E vs human E)
  4. Systematic bias by protein source category
  5. Impact on variance decomposition when substituting LLM E

Input:  DISH_RECIPES (hardcoded), data/ingredient_impact_factors.csv
Output: results/tables/recipe_validation_summary.csv
        results/figures/recipe_validation_scatter.png
"""

import sys, os, json, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from config import DATA_DIR, TABLES_DIR, FIGURES_DIR, GRID_EMISSION_FACTOR, COOKING_ENERGY_KWH

# Import from 04_env_cost_calculation using importlib (filename starts with digit)
import importlib.util
_spec = importlib.util.spec_from_file_location("env_cost", ROOT / "code" / "04_env_cost_calculation.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DISH_RECIPES = _mod.DISH_RECIPES
compute_dish_environmental_cost = _mod.compute_dish_environmental_cost

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"
WORKERS = 30

# Load ingredient list for LLM prompt
ENV_INGREDIENTS = pd.read_csv(DATA_DIR / "ingredient_impact_factors.csv")["ingredient"].tolist()
ENV_INGREDIENTS_STR = ", ".join(sorted(ENV_INGREDIENTS))

SYSTEM_PROMPT = f"""You are a culinary expert. Given a dish name, produce a standardised recipe as JSON.
Map all ingredients to the closest match from this list:
{ENV_INGREDIENTS_STR}

If no close match, use the closest substitute (e.g. "ghee"→"butter", "paneer"→"cheese").

Output ONLY valid JSON:
{{"ingredients": {{"ingredient_name": grams_per_serving, ...}},
  "cook_method": "raw|boil|steam|stir_fry|saute|pan_fry|grill|bake|roast|deep_fry|braise|simmer|slow_cook|smoke|cold"}}

Rules: Use ONLY names from the list. 3-12 ingredients. Realistic single-serving grams."""


def call_llm(dish_id):
    """Generate a recipe for a single dish via LLM."""
    import requests
    dish_name = dish_id.replace("_", " ").title()
    user_msg = f"Dish: {dish_name}\n\nGenerate the recipe JSON."

    for attempt in range(3):
        try:
            resp = requests.post(API_URL,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [
                          {"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                      "temperature": 0.1, "max_tokens": 500},
                timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                recipe = json.loads(m.group())
                return dish_id, recipe, "ok"
            return dish_id, None, "parse_error"
        except Exception as e:
            if attempt == 2:
                return dish_id, None, f"error: {str(e)[:80]}"
            time.sleep(1)
    return dish_id, None, "max_retries"


def normalize_llm_recipe(raw_recipe):
    """Convert LLM output to standard {ingredients: {name: grams}, cook_method: str} format."""
    ings = raw_recipe.get("ingredients", {})
    cook = raw_recipe.get("cook_method", "unknown")

    # Handle both dict and list-of-dict formats
    if isinstance(ings, list):
        ing_dict = {}
        for item in ings:
            if isinstance(item, dict):
                name = item.get("name", "").lower().strip()
                grams = item.get("grams", 0)
                if name and grams > 0:
                    ing_dict[name] = grams
        ings = ing_dict

    # Normalize ingredient names
    env_set = {e.lower() for e in ENV_INGREDIENTS}
    clean = {}
    for k, v in ings.items():
        k_clean = k.lower().strip()
        if k_clean in env_set and v > 0:
            clean[k_clean] = v

    return {"ingredients": clean, "cook_method": cook}


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    print("=" * 60, flush=True)
    print("25a — LLM Recipe Validation", flush=True)
    print("=" * 60, flush=True)

    # ── Step 0: Load impact factors ──
    impact_df = pd.read_csv(DATA_DIR / "ingredient_impact_factors.csv").set_index("ingredient")

    # ── Step 1: Compute human E for all DISH_RECIPES ──
    print(f"\n  Human recipes: {len(DISH_RECIPES)} dishes", flush=True)
    human_e = {}
    for dish_id, recipe in DISH_RECIPES.items():
        env = compute_dish_environmental_cost(recipe, impact_df)
        human_e[dish_id] = env["E_carbon"]

    # ── Step 2: Generate LLM recipes ──
    dish_ids = sorted(DISH_RECIPES.keys())

    # Check cache
    cache_path = DATA_DIR / "recipe_validation_llm_cache.json"
    llm_recipes = {}
    if cache_path.exists():
        with open(cache_path) as f:
            llm_recipes = json.load(f)
        print(f"  Cached LLM recipes: {len(llm_recipes)}", flush=True)

    remaining = [d for d in dish_ids if d not in llm_recipes]

    if remaining and API_KEY:
        print(f"  Generating {len(remaining)} LLM recipes ({WORKERS} workers)...", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(call_llm, d): d for d in remaining}
            for fut in as_completed(futures):
                dish_id, recipe, status = fut.result()
                done += 1
                if status == "ok" and recipe:
                    llm_recipes[dish_id] = recipe
                if done % 20 == 0:
                    print(f"    {done}/{len(remaining)} done", flush=True)

        # Save cache
        with open(cache_path, "w") as f:
            json.dump(llm_recipes, f)
        print(f"  LLM recipes generated: {len(llm_recipes)}", flush=True)
    elif not API_KEY:
        print("  WARNING: No API key — using cached recipes only", flush=True)

    # ── Step 3: Compare ──
    print(f"\n  Comparing {len(llm_recipes)} matched dishes...", flush=True)

    records = []
    for dish_id in dish_ids:
        if dish_id not in llm_recipes:
            continue

        human_recipe = DISH_RECIPES[dish_id]
        llm_raw = llm_recipes[dish_id]
        llm_recipe = normalize_llm_recipe(llm_raw)

        human_ings = set(human_recipe["ingredients"].keys())
        llm_ings = set(llm_recipe["ingredients"].keys())

        # Jaccard
        jac = jaccard(human_ings, llm_ings)

        # Gram MAE for shared ingredients
        shared = human_ings & llm_ings
        if shared:
            gram_diffs = [abs(human_recipe["ingredients"][i] - llm_recipe["ingredients"].get(i, 0))
                          for i in shared]
            gram_mae = np.mean(gram_diffs)
        else:
            gram_mae = np.nan

        # E comparison
        llm_env = compute_dish_environmental_cost(llm_recipe, impact_df)
        llm_e_val = llm_env["E_carbon"]
        human_e_val = human_e[dish_id]

        # Cook method match
        cook_match = human_recipe["cook_method"] == llm_recipe["cook_method"]

        # Protein source category
        meat_ings = {"beef", "pork", "chicken", "lamb", "duck", "turkey", "ground_beef",
                     "bacon", "sausage"}
        seafood_ings = {"fish", "shrimp", "salmon", "cod", "squid", "crab"}
        plant_ings = {"tofu", "lentil", "chickpea", "soybean", "black_bean"}

        human_has_meat = bool(human_ings & meat_ings)
        human_has_seafood = bool(human_ings & seafood_ings)
        human_has_plant_prot = bool(human_ings & plant_ings)

        if human_has_meat:
            prot_cat = "meat"
        elif human_has_seafood:
            prot_cat = "seafood"
        elif human_has_plant_prot:
            prot_cat = "plant"
        else:
            prot_cat = "other"

        records.append({
            "dish_id": dish_id,
            "jaccard": jac,
            "gram_mae": gram_mae,
            "n_human_ings": len(human_ings),
            "n_llm_ings": len(llm_ings),
            "n_shared": len(shared),
            "human_E_carbon": human_e_val,
            "llm_E_carbon": llm_e_val,
            "E_ratio": llm_e_val / human_e_val if human_e_val > 0 else np.nan,
            "cook_method_match": cook_match,
            "protein_category": prot_cat,
        })

    comp = pd.DataFrame(records)
    print(f"  Matched dishes for comparison: {len(comp)}", flush=True)

    if len(comp) < 10:
        print("  ERROR: Too few matched dishes for meaningful analysis", flush=True)
        return

    # ── Step 4: Summary statistics ──
    print(f"\n{'='*60}", flush=True)
    print("RECIPE VALIDATION RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n  Ingredient overlap (Jaccard):", flush=True)
    print(f"    Mean: {comp['jaccard'].mean():.3f}", flush=True)
    print(f"    Median: {comp['jaccard'].median():.3f}", flush=True)
    print(f"    Q25-Q75: [{comp['jaccard'].quantile(.25):.3f}, {comp['jaccard'].quantile(.75):.3f}]", flush=True)

    print(f"\n  Gram MAE (shared ingredients):", flush=True)
    valid_mae = comp['gram_mae'].dropna()
    print(f"    Mean: {valid_mae.mean():.1f}g", flush=True)
    print(f"    Median: {valid_mae.median():.1f}g", flush=True)

    print(f"\n  Cook method agreement: {comp['cook_method_match'].mean()*100:.1f}%", flush=True)

    # E correlation
    rho_e, p_e = stats.spearmanr(comp["human_E_carbon"], comp["llm_E_carbon"])
    r_e, p_r = stats.pearsonr(comp["human_E_carbon"], comp["llm_E_carbon"])
    print(f"\n  E_carbon correlation:", flush=True)
    print(f"    Spearman ρ: {rho_e:.3f} (p={p_e:.2e})", flush=True)
    print(f"    Pearson r: {r_e:.3f} (p={p_r:.2e})", flush=True)
    print(f"    Mean E ratio (LLM/human): {comp['E_ratio'].mean():.3f}", flush=True)
    print(f"    Median E ratio: {comp['E_ratio'].median():.3f}", flush=True)

    # ── Step 5: By protein category ──
    print(f"\n  Systematic bias by protein source:", flush=True)
    for cat in ["meat", "seafood", "plant", "other"]:
        sub = comp[comp["protein_category"] == cat]
        if len(sub) >= 3:
            r_sub, _ = stats.spearmanr(sub["human_E_carbon"], sub["llm_E_carbon"])
            print(f"    {cat:10s} (n={len(sub):3d}): ρ={r_sub:.3f}, "
                  f"mean ratio={sub['E_ratio'].mean():.3f}, "
                  f"Jaccard={sub['jaccard'].mean():.3f}", flush=True)

    # ── Step 6: Impact on variance decomposition ──
    # Load DEI data
    dei = pd.read_csv(DATA_DIR / "combined_dish_DEI_v2.csv")
    dei_matched = dei[dei["dish_id"].isin(comp["dish_id"])].copy()
    comp_idx = comp.set_index("dish_id")

    if len(dei_matched) > 20:
        dei_matched = dei_matched.set_index("dish_id")
        dei_matched["E_llm"] = comp_idx["llm_E_carbon"]
        dei_matched = dei_matched.dropna(subset=["E_llm"])

        # Recompute log DEI with LLM E
        dei_matched["log_E_llm"] = np.log(dei_matched["E_llm"].clip(lower=0.001))
        dei_matched["log_DEI_llm"] = dei_matched["log_H"] - dei_matched["log_E_llm"]

        # Variance decomposition: original vs LLM
        var_logH = dei_matched["log_H"].var()
        var_logE_orig = dei_matched["log_E"].var()
        var_logE_llm = dei_matched["log_E_llm"].var()
        var_logDEI_orig = dei_matched["log_DEI"].var()
        var_logDEI_llm = dei_matched["log_DEI_llm"].var()

        pct_H_orig = var_logH / (var_logH + var_logE_orig) * 100
        pct_H_llm = var_logH / (var_logH + var_logE_llm) * 100

        rho_dei, _ = stats.spearmanr(dei_matched["log_DEI"], dei_matched["log_DEI_llm"])

        print(f"\n  Variance decomposition impact (n={len(dei_matched)} matched):", flush=True)
        print(f"    H contribution (original E): {pct_H_orig:.1f}%", flush=True)
        print(f"    H contribution (LLM E):      {pct_H_llm:.1f}%", flush=True)
        print(f"    DEI rank correlation (original vs LLM E): ρ={rho_dei:.3f}", flush=True)

    # ── Step 7: Save ──
    comp.to_csv(TABLES_DIR / "recipe_validation_summary.csv", index=False)
    print(f"\n  Saved: {TABLES_DIR / 'recipe_validation_summary.csv'}", flush=True)

    # ── Step 8: Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: E scatter
    ax = axes[0]
    colors = {"meat": "#E24A33", "seafood": "#348ABD", "plant": "#8EBA42", "other": "#988ED5"}
    for cat in ["meat", "seafood", "plant", "other"]:
        sub = comp[comp["protein_category"] == cat]
        ax.scatter(sub["human_E_carbon"], sub["llm_E_carbon"],
                   alpha=0.6, s=30, c=colors[cat], label=f"{cat} (n={len(sub)})")
    lim = max(comp["human_E_carbon"].max(), comp["llm_E_carbon"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Human recipe E (kg CO₂)", fontsize=11)
    ax.set_ylabel("LLM recipe E (kg CO₂)", fontsize=11)
    ax.set_title(f"ρ = {rho_e:.3f}, r = {r_e:.3f}", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Jaccard distribution
    ax = axes[1]
    ax.hist(comp["jaccard"], bins=20, alpha=0.7, color="#2077B4", edgecolor="white")
    ax.axvline(comp["jaccard"].mean(), color="red", ls="--", lw=1.5,
               label=f"Mean={comp['jaccard'].mean():.2f}")
    ax.set_xlabel("Jaccard ingredient overlap", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Ingredient Agreement", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: E ratio by category
    ax = axes[2]
    cats = ["meat", "seafood", "plant", "other"]
    cat_data = [comp[comp["protein_category"] == c]["E_ratio"].dropna() for c in cats]
    cat_data_valid = [(c, d) for c, d in zip(cats, cat_data) if len(d) >= 2]
    if cat_data_valid:
        bp = ax.boxplot([d for _, d in cat_data_valid],
                        labels=[c for c, _ in cat_data_valid],
                        patch_artist=True)
        for patch, (c, _) in zip(bp['boxes'], cat_data_valid):
            patch.set_facecolor(colors.get(c, "#888"))
            patch.set_alpha(0.6)
    ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel("E ratio (LLM / Human)", fontsize=11)
    ax.set_title("Systematic Bias by Protein Source", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle("LLM Recipe Validation (25a)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recipe_validation_scatter.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {FIGURES_DIR / 'recipe_validation_scatter.png'}", flush=True)
    plt.close()

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
