#!/usr/bin/env python3
"""
22c_merge_and_compute_E.py — Merge LLM recipes with WorldCuisines metadata,
compute environmental costs, and produce final expanded dish list.

Inputs:
  - data/worldcuisines_matched.csv   (2,230 dish metadata)
  - data/llm_generated_recipes.csv   (2,229 LLM recipes with ingredients)
  - data/ingredient_impact_factors.csv (101 impact factors)
  - data/combined_dish_DEI.csv       (existing 334 dishes)

Outputs:
  - data/expanded_dish_env_costs_v2.csv  (E scores for new dishes)
  - data/all_dishes_master.csv           (334 + new, with E scores)
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COOKING_ENERGY_KWH, GRID_EMISSION_FACTOR

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ── Load data ────────────────────────────────────────────────────
print("Loading data...", flush=True)

meta = pd.read_csv(DATA / "worldcuisines_matched.csv")
recipes = pd.read_csv(DATA / "llm_generated_recipes.csv")
impacts = pd.read_csv(DATA / "ingredient_impact_factors.csv")
existing = pd.read_csv(DATA / "combined_dish_DEI.csv")

print(f"  WorldCuisines metadata: {len(meta)} dishes")
print(f"  LLM recipes: {len(recipes)} dishes")
print(f"  Impact factors: {len(impacts)} ingredients")
print(f"  Existing DEI: {len(existing)} dishes")

# Build impact lookup
impact_lookup = {}
for _, row in impacts.iterrows():
    impact_lookup[row["ingredient"].lower().strip()] = {
        "co2": row["co2_per_kg"],
        "water": row["water_per_kg"],
        "land": row["land_per_kg"],
    }

# Add common substitutes not in the 101 list
for sub, proxy in [("wine", {"co2": 1.5, "water": 800, "land": 3.0}),
                   ("ice_cream", {"co2": 5.0, "water": 2500, "land": 15.0}),
                   ("saffron", {"co2": 5.0, "water": 8000, "land": 3.0}),
                   ("water", {"co2": 0.0, "water": 1, "land": 0.0})]:
    if sub not in impact_lookup:
        impact_lookup[sub] = proxy

# ── Merge recipes with metadata ─────────────────────────────────
print("\nMerging recipes with metadata...", flush=True)
merged = meta.merge(recipes, on="dish_id", how="inner", suffixes=("_meta", "_recipe"))
print(f"  Merged: {len(merged)} dishes with both metadata and recipe")

# Dishes with metadata but no recipe
no_recipe = meta[~meta["dish_id"].isin(recipes["dish_id"])]
print(f"  No recipe: {len(no_recipe)} dishes (will use extracted ingredients)")

# For dishes without LLM recipe, try to use extracted ingredients from metadata
extra_rows = []
for _, row in no_recipe.iterrows():
    ings_str = row.get("ingredients_matched", "[]")
    try:
        ings = eval(ings_str) if isinstance(ings_str, str) else []
    except:
        ings = []
    ings_known = [i for i in ings if i.lower().strip() in impact_lookup]
    if len(ings_known) >= 2:
        # Assign equal weight, 50g each as rough estimate
        ing_list = [{"name": i.lower().strip(), "grams": 50} for i in ings_known]
        extra_rows.append({
            "dish_id": row["dish_id"],
            "ingredients_json": json.dumps(ing_list),
            "n_ingredients_recipe": len(ing_list),
            "cook_method_recipe": "unknown",
            "total_grams": 50 * len(ing_list),
            "calories_approx": 0,
            "protein_g_approx": 0,
        })

if extra_rows:
    extra_df = pd.DataFrame(extra_rows)
    extra_meta = no_recipe.merge(extra_df, on="dish_id", how="inner")
    # Rename columns to match merged
    for col in ["n_ingredients", "cook_method"]:
        if col + "_meta" not in extra_meta.columns and col in extra_meta.columns:
            extra_meta.rename(columns={col: col + "_meta"}, inplace=True)
    merged = pd.concat([merged, extra_meta], ignore_index=True)
    print(f"  Added {len(extra_rows)} dishes from extracted ingredients")

print(f"  Total with ingredients: {len(merged)} dishes")

# ── Compute E for each dish ──────────────────────────────────────
print("\nComputing environmental costs...", flush=True)

rows = []
missing_all = set()

for _, dish in merged.iterrows():
    dish_id = dish["dish_id"]

    # Parse ingredients
    try:
        ingredients = json.loads(dish["ingredients_json"])
    except:
        continue

    # Determine cook method
    cook_method = str(dish.get("cook_method_recipe",
                               dish.get("cook_method", "unknown"))).lower().strip()

    total_grams = sum(i.get("grams", 0) for i in ingredients)
    carbon = 0.0
    water = 0.0
    land = 0.0
    missing = []
    matched_count = 0

    for ing in ingredients:
        name = ing["name"].lower().strip()
        grams = ing.get("grams", 0)
        if grams <= 0:
            continue
        if name not in impact_lookup:
            missing.append(name)
            missing_all.add(name)
            continue
        kg = grams / 1000.0
        imp = impact_lookup[name]
        carbon += kg * imp["co2"]
        water += kg * imp["water"]
        land += kg * imp["land"]
        matched_count += 1

    # Cooking energy
    cook_kwh = COOKING_ENERGY_KWH.get(cook_method, 0.5)
    cook_co2 = cook_kwh * GRID_EMISSION_FACTOR

    # Dishwashing (same as 04 script)
    vessel_count = 2
    water_cleaning = vessel_count * 0.5
    co2_cleaning = vessel_count * 0.01 * GRID_EMISSION_FACTOR

    rows.append({
        "dish_id": dish_id,
        "name": dish.get("name", dish_id),
        "primary_cuisine": dish.get("primary_cuisine", ""),
        "primary_coarse": dish.get("primary_coarse", ""),
        "countries": dish.get("countries", ""),
        "region": dish.get("region", ""),
        "E_carbon": round(carbon + cook_co2 + co2_cleaning, 4),
        "E_carbon_ingredients": round(carbon, 4),
        "E_carbon_cooking": round(cook_co2, 4),
        "E_water": round(water + water_cleaning, 1),
        "E_water_ingredients": round(water, 1),
        "E_energy": round(cook_kwh, 2),
        "E_land": round(land, 4),
        "cook_method": cook_method,
        "n_ingredients": matched_count,
        "total_grams": total_grams,
        "ingredients_json": dish["ingredients_json"],
        "calories_approx": dish.get("calories_approx", 0),
        "protein_g_approx": dish.get("protein_g_approx", 0),
        "missing_ingredients": ",".join(missing) if missing else "",
        "source": "worldcuisines_llm",
    })

env_df = pd.DataFrame(rows)
print(f"  Computed E for {len(env_df)} dishes")

if missing_all:
    print(f"  Unique missing ingredients: {len(missing_all)}")
    # Show top missing by frequency
    from collections import Counter
    miss_counts = Counter()
    for r in rows:
        for m in r["missing_ingredients"].split(","):
            if m:
                miss_counts[m] += 1
    print(f"  Top 10 missing: {miss_counts.most_common(10)}")

# ── Normalize E using combined range (existing + new) ────────────
print("\nNormalizing E scores...", flush=True)

all_carbon = pd.concat([existing["E_carbon"], env_df["E_carbon"]])
all_water = pd.concat([existing["E_water"], env_df["E_water"]])
all_energy = pd.concat([existing["E_energy"], env_df["E_energy"]])

c_min, c_max = all_carbon.min(), all_carbon.max()
w_min, w_max = all_water.min(), all_water.max()
e_min, e_max = all_energy.min(), all_energy.max()

env_df["E_carbon_norm"] = (env_df["E_carbon"] - c_min) / (c_max - c_min)
env_df["E_water_norm"] = (env_df["E_water"] - w_min) / (w_max - w_min)
env_df["E_energy_norm"] = (env_df["E_energy"] - e_min) / (e_max - e_min)
env_df["E_composite"] = (env_df["E_carbon_norm"] + env_df["E_water_norm"] + env_df["E_energy_norm"]) / 3

# Also renormalize existing dishes on same scale
existing["E_carbon_norm"] = (existing["E_carbon"] - c_min) / (c_max - c_min)
existing["E_water_norm"] = (existing["E_water"] - w_min) / (w_max - w_min)
existing["E_energy_norm"] = (existing["E_energy"] - e_min) / (e_max - e_min)
existing["E_composite_renorm"] = (existing["E_carbon_norm"] + existing["E_water_norm"] + existing["E_energy_norm"]) / 3

print(f"  E_composite range (new): [{env_df['E_composite'].min():.4f}, {env_df['E_composite'].max():.4f}]")
print(f"  E_composite range (existing renormed): [{existing['E_composite_renorm'].min():.4f}, {existing['E_composite_renorm'].max():.4f}]")

# Save new dish E scores
env_df.to_csv(DATA / "expanded_dish_env_costs_v2.csv", index=False)
print(f"\n  Saved: data/expanded_dish_env_costs_v2.csv ({len(env_df)} dishes)")

# ── Build master dish list ───────────────────────────────────────
print("\nBuilding master dish list...", flush=True)

# Existing dishes: keep all original columns, update E_composite
existing_out = existing.copy()
existing_out["E_composite"] = existing_out["E_composite_renorm"]

# New dishes: placeholder H until pairwise ranking is done
env_df["H_mean"] = np.nan  # Will be filled by pairwise ranking
env_df["H_bert"] = np.nan
env_df["log_E"] = np.log(env_df["E_composite"].clip(lower=1e-6))

# Common columns for master list
master_cols = ["dish_id", "name", "primary_cuisine", "E_composite", "E_carbon",
               "E_water", "E_energy", "E_land", "log_E", "cook_method",
               "n_ingredients", "total_grams", "source"]

# Prepare existing
exist_master = existing_out[["dish_id"]].copy()
exist_master["name"] = existing_out["dish_id"]  # original dishes use dish_id as name
exist_master["primary_cuisine"] = existing_out.get("cuisine", "")
exist_master["E_composite"] = existing_out["E_composite"]
exist_master["E_carbon"] = existing_out["E_carbon"]
exist_master["E_water"] = existing_out["E_water"]
exist_master["E_energy"] = existing_out["E_energy"]
exist_master["E_land"] = existing_out.get("E_land", np.nan)
exist_master["log_E"] = np.log(existing_out["E_composite"].clip(lower=1e-6))
exist_master["cook_method"] = existing_out.get("cook_method", "")
exist_master["n_ingredients"] = existing_out.get("n_ingredients", np.nan)
exist_master["total_grams"] = existing_out.get("total_grams", np.nan)
exist_master["source"] = existing_out.get("source", "original")
exist_master["H_mean"] = existing_out.get("H_mean", np.nan)
exist_master["H_bert"] = existing_out.get("H_bert", np.nan)

new_master = env_df[["dish_id", "name", "primary_cuisine", "E_composite", "E_carbon",
                      "E_water", "E_energy", "E_land", "log_E", "cook_method",
                      "n_ingredients", "total_grams", "source"]].copy()
new_master["H_mean"] = np.nan
new_master["H_bert"] = np.nan

master = pd.concat([exist_master, new_master], ignore_index=True)
master = master.drop_duplicates("dish_id", keep="first")
master.to_csv(DATA / "all_dishes_master.csv", index=False)

print(f"  Master dish list: {len(master)} dishes")
print(f"    Existing (with H): {master['H_mean'].notna().sum()}")
print(f"    New (need H): {master['H_mean'].isna().sum()}")

# ── Cuisine coverage ─────────────────────────────────────────────
print("\nCuisine coverage:")
cuisine_counts = master["primary_cuisine"].value_counts()
for cuisine in cuisine_counts.head(30).index:
    n = cuisine_counts[cuisine]
    print(f"  {cuisine:25s}: {n}")
print(f"  ... Total cuisines: {cuisine_counts.nunique()}")

# ── Summary statistics ───────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total dishes: {len(master)}")
print(f"  With H scores: {master['H_mean'].notna().sum()}")
print(f"  Needing H scores: {master['H_mean'].isna().sum()}")
print(f"  Cuisines: {cuisine_counts.nunique()}")
print(f"  E_composite range: [{master['E_composite'].min():.4f}, {master['E_composite'].max():.4f}]")
print(f"  Mean E_composite: {master['E_composite'].mean():.4f}")

# Top/Bottom by E
print(f"\n  Top 10 lowest E (most eco-friendly):")
for _, row in master.nsmallest(10, "E_composite").iterrows():
    print(f"    {row['dish_id']:30s} E={row['E_composite']:.4f} ({row['primary_cuisine']})")

print(f"\n  Top 10 highest E (most eco-costly):")
for _, row in master.nlargest(10, "E_composite").iterrows():
    print(f"    {row['dish_id']:30s} E={row['E_composite']:.4f} ({row['primary_cuisine']})")

print("\nDone!", flush=True)
