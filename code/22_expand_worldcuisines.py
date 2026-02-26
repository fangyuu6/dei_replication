#!/usr/bin/env python3
"""
22_expand_worldcuisines.py
Expand dish dataset using WorldCuisines food-kb + CulinaryDB + LLM

Pipeline:
1. Load 2,230 new dishes from WorldCuisines (already deduplicated)
2. Match to CulinaryDB for ingredient lists
3. Extract ingredients from WorldCuisines text descriptions
4. Map ingredients to our 101 environmental factor ingredients
5. Generate standardised recipes via LLM for unmapped dishes
6. Output: data/expanded_dishes_v2.csv with dish_id, cuisine, ingredients, cook_method
"""

import pandas as pd
import numpy as np
import re
import json
import os
import sys
from collections import Counter
from difflib import SequenceMatcher

# ── Paths ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW  = os.path.join(ROOT, "raw")

# ── Load data ──────────────────────────────────────────────────────
print("Loading datasets...")
wc_new = pd.read_csv(os.path.join(DATA, "worldcuisines_new_dishes.csv"))
cdb_recipes = pd.read_csv(os.path.join(RAW, "culinarydb", "01_Recipe_Details.csv"))
cdb_ing_map = pd.read_csv(os.path.join(RAW, "culinarydb", "04_Recipe-Ingredients_Aliases.csv"))
env_factors = pd.read_csv(os.path.join(DATA, "ingredient_impact_factors.csv"))
existing = pd.read_csv(os.path.join(DATA, "combined_dish_DEI.csv"))

# Our 101 known environmental factor ingredients
ENV_INGREDIENTS = set(env_factors["ingredient"].str.lower().str.strip().tolist())

print(f"  WorldCuisines new dishes: {len(wc_new)}")
print(f"  CulinaryDB recipes: {len(cdb_recipes)}")
print(f"  Environmental factor ingredients: {len(ENV_INGREDIENTS)}")
print(f"  Existing dishes: {len(existing)}")

# ── Step 1: Normalize names ───────────────────────────────────────
def normalize_name(name):
    """Normalize dish name to snake_case ID."""
    s = str(name).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s

wc_new["dish_id"] = wc_new["Name"].apply(normalize_name)

# ── Step 2: Match to CulinaryDB ──────────────────────────────────
print("\nMatching to CulinaryDB...")

def normalize_title(t):
    return re.sub(r'[^a-z0-9\s]', '', str(t).lower()).strip()

# Build CulinaryDB title -> recipe ID index
cdb_title_idx = {}
for _, row in cdb_recipes.iterrows():
    t = normalize_title(row["Title"])
    cdb_title_idx.setdefault(t, []).append(row["Recipe ID"])

# Build recipe ID -> ingredient list
cdb_recipe_ings = {}
for rid, group in cdb_ing_map.groupby("Recipe ID"):
    ings = group["Aliased Ingredient Name"].str.strip().str.lower().unique().tolist()
    cdb_recipe_ings[rid] = ings

# Match WorldCuisines dishes to CulinaryDB
cdb_matches = {}
for _, row in wc_new.iterrows():
    dish_name = normalize_title(row["Name"])
    # Exact match
    if dish_name in cdb_title_idx:
        rid = cdb_title_idx[dish_name][0]
        cdb_matches[row["dish_id"]] = cdb_recipe_ings.get(rid, [])
        continue
    # Substring match (dish name in CDB title)
    for title, rids in cdb_title_idx.items():
        if len(dish_name) >= 4 and (dish_name in title or title.startswith(dish_name)):
            cdb_matches[row["dish_id"]] = cdb_recipe_ings.get(rids[0], [])
            break

print(f"  CulinaryDB matches: {len(cdb_matches)} / {len(wc_new)} ({len(cdb_matches)/len(wc_new)*100:.1f}%)")

# ── Step 3: Extract ingredients from WC text descriptions ─────────
print("\nExtracting ingredients from text descriptions...")

# Common food ingredient keywords to look for in descriptions
INGREDIENT_PATTERNS = {
    # Proteins
    "beef": r"\bbeef\b", "chicken": r"\bchicken\b", "pork": r"\bpork\b",
    "lamb": r"\blamb\b", "fish": r"\bfish\b", "shrimp": r"\bshrimp\b|prawn",
    "egg": r"\beggs?\b", "tofu": r"\btofu\b", "duck": r"\bduck\b",
    "turkey": r"\bturkey\b", "crab": r"\bcrab\b", "squid": r"\bsquid\b|calamari",
    "salmon": r"\bsalmon\b", "tuna": r"\btuna\b", "cod": r"\bcod\b",
    "sausage": r"\bsausage\b", "bacon": r"\bbacon\b",
    "ground_beef": r"\bground\s*(beef|meat)\b|mince\s*meat",
    # Dairy
    "cheese": r"\bcheese\b", "butter": r"\bbutter\b", "cream": r"\bcream\b",
    "milk": r"\bmilk\b", "yogurt": r"\byogu?rt\b", "mozzarella": r"\bmozzarella\b",
    "parmesan": r"\bparmesan\b",
    # Grains & starches
    "rice": r"\brice\b", "wheat_flour": r"\b(wheat\s*)?flour\b|dough",
    "bread": r"\bbread\b", "pasta_dry": r"\bpasta\b|noodle|spaghetti|macaroni",
    "rice_noodle": r"\brice\s*noodle\b|vermicelli",
    "corn": r"\bcorn\b|maize", "potato": r"\bpotato\b",
    "oats": r"\boats?\b|oatmeal", "tortilla": r"\btortilla\b",
    "sweet_potato": r"\bsweet\s*potato\b|yam",
    "pita": r"\bpita\b|flatbread",
    # Vegetables
    "onion": r"\bonion\b|shallot", "garlic": r"\bgarlic\b",
    "tomato": r"\btomato\b", "pepper": r"\bpepper\b|capsicum|bell pepper",
    "carrot": r"\bcarrot\b", "cabbage": r"\bcabbage\b",
    "spinach": r"\bspinach\b", "eggplant": r"\beggplant\b|aubergine",
    "mushroom": r"\bmushroom\b", "broccoli": r"\bbroccoli\b",
    "celery": r"\bcelery\b", "cucumber": r"\bcucumber\b",
    "lettuce": r"\blettuce\b", "zucchini": r"\bzucchini\b|courgette",
    "bamboo_shoot": r"\bbamboo\s*shoot\b",
    "bean_sprout": r"\bbean\s*sprout\b",
    # Legumes
    "lentil": r"\blentil\b|dal\b", "chickpea": r"\bchickpea\b|garbanzo",
    "black_bean": r"\bblack\s*bean\b", "soybean": r"\bsoybean\b|soy\s*bean",
    "peanut": r"\bpeanut\b",
    # Fruits
    "lemon": r"\blemon\b", "lime": r"\blime\b", "mango": r"\bmango\b",
    "avocado": r"\bavocado\b", "banana": r"\bbanana\b|plantain",
    "apple": r"\bapple\b", "pineapple": r"\bpineapple\b",
    "coconut": r"\bcoconut\b", "coconut_milk": r"\bcoconut\s*milk\b|coconut\s*cream",
    # Oils & fats
    "olive_oil": r"\bolive\s*oil\b", "vegetable_oil": r"\bvegetable\s*oil\b|cooking\s*oil|oil",
    "sesame_oil": r"\bsesame\s*oil\b", "palm_oil": r"\bpalm\s*oil\b",
    "coconut_oil": r"\bcoconut\s*oil\b",
    # Condiments & sauces
    "soy_sauce": r"\bsoy\s*sauce\b", "fish_sauce": r"\bfish\s*sauce\b",
    "vinegar": r"\bvinegar\b", "sugar": r"\bsugar\b",
    "salt": r"\bsalt\b", "honey": r"\bhoney\b",
    "ginger": r"\bginger\b", "chili": r"\bchil[li]\b|chili",
    "cumin": r"\bcumin\b", "turmeric": r"\bturmeric\b",
    "coriander": r"\bcoriander\b|cilantro", "basil": r"\bbasil\b",
    "black_pepper": r"\bblack\s*pepper\b",
    "curry_paste": r"\bcurry\b",
    "chocolate": r"\bchocolate\b|cocoa",
    "vanilla": r"\bvanilla\b",
    "coffee": r"\bcoffee\b", "tea": r"\btea\b(?!\s*leaf\b|\s*spoon)",
    # Nuts
    "almond": r"\balmond\b", "cashew": r"\bcashew\b",
    "walnut": r"\bwalnut\b", "pine_nut": r"\bpine\s*nut\b",
    "sesame_seed": r"\bsesame\b",
}

def extract_ingredients_from_text(text):
    """Extract ingredient mentions from description text."""
    if pd.isna(text):
        return []
    text = text.lower()
    found = []
    for ing, pattern in INGREDIENT_PATTERNS.items():
        if re.search(pattern, text):
            found.append(ing)
    return found

wc_new["text_ingredients"] = wc_new["Text Description"].apply(extract_ingredients_from_text)

# ── Step 4: Merge CulinaryDB + text-extracted ingredients ─────────
print("\nMerging ingredient sources...")

# Map CulinaryDB ingredient names to our 101 env factor ingredients
CDB_TO_ENV = {}
for cdb_name in cdb_ing_map["Aliased Ingredient Name"].str.strip().str.lower().unique():
    # Direct match
    if cdb_name in ENV_INGREDIENTS:
        CDB_TO_ENV[cdb_name] = cdb_name
        continue
    # Common mappings
    mappings = {
        "chicken ": "chicken", "beef ": "beef", "pork ": "pork",
        "lamb ": "lamb", "egg ": "egg", "butter ": "butter",
        "sugar ": "sugar", "salt ": "salt", "cream ": "cream",
        "milk ": "milk", "rice ": "rice", "onion ": "onion",
        "garlic ": "garlic", "tomato ": "tomato", "potato ": "potato",
        "carrot ": "carrot", "ginger ": "ginger", "lemon ": "lemon",
        "honey ": "honey", "vinegar ": "vinegar",
        "olive oil ": "olive_oil", "vegetable oil ": "vegetable_oil",
        "soy sauce ": "soy_sauce", "fish sauce ": "fish_sauce",
        "coconut milk ": "coconut_milk", "all purpose flour": "wheat_flour",
        "flour ": "wheat_flour", "pepper ": "pepper", "cumin ": "cumin",
        "coriander ": "coriander", "turmeric ": "turmeric",
        "cayenne ": "chili", "chili ": "chili", "paprika ": "chili",
        "yogurt ": "yogurt", "cucumber ": "cucumber", "spinach ": "spinach",
        "mushroom ": "mushroom", "cabbage ": "cabbage", "celery ": "celery",
        "corn ": "corn", "avocado ": "avocado", "pineapple ": "pineapple",
        "mango ": "mango", "banana ": "banana", "apple ": "apple",
        "lentil ": "lentil", "chickpea ": "chickpea", "tofu ": "tofu",
        "shrimp ": "shrimp", "salmon ": "salmon", "tuna ": "tuna",
        "squid ": "squid", "crab ": "crab", "cod ": "cod",
        "duck ": "duck", "turkey ": "turkey", "bacon ": "bacon",
        "sausage ": "sausage", "cheese ": "cheese",
        "sesame oil ": "sesame_oil", "peanut ": "peanut",
        "almond ": "almond", "walnut ": "walnut", "cashew ": "cashew",
        "chocolate ": "chocolate", "vanilla ": "vanilla",
        "coffee ": "coffee", "basil ": "basil",
        "broccoli ": "broccoli", "eggplant ": "eggplant",
        "lettuce ": "lettuce", "zucchini ": "zucchini",
        "sweet potato ": "sweet_potato", "oats ": "oats",
        "black bean ": "black_bean", "lime ": "lime",
        "tomato paste ": "tomato", "tomato sauce ": "tomato_sauce",
        "parmesan ": "parmesan", "mozzarella ": "mozzarella",
        "mayonnaise ": "mayonnaise", "ketchup ": "ketchup",
        "mustard ": "mustard", "hoisin sauce ": "hoisin_sauce",
        "oyster sauce ": "oyster_sauce",
    }
    for k, v in mappings.items():
        if k in cdb_name or cdb_name.startswith(k.strip()):
            CDB_TO_ENV[cdb_name] = v
            break

def map_cdb_ingredients(cdb_list):
    """Map CulinaryDB ingredient list to env factor ingredients."""
    mapped = set()
    for ing in cdb_list:
        ing = ing.strip().lower()
        if ing in CDB_TO_ENV:
            mapped.add(CDB_TO_ENV[ing])
        elif ing in ENV_INGREDIENTS:
            mapped.add(ing)
    return sorted(mapped)

# Combine ingredients from both sources
results = []
for _, row in wc_new.iterrows():
    dish_id = row["dish_id"]
    ingredients = set()

    # From CulinaryDB
    if dish_id in cdb_matches:
        cdb_ings = map_cdb_ingredients(cdb_matches[dish_id])
        ingredients.update(cdb_ings)

    # From text description
    text_ings = row["text_ingredients"]
    # Only add text ingredients that are in our env factor list
    for ing in text_ings:
        if ing in ENV_INGREDIENTS:
            ingredients.add(ing)

    results.append({
        "dish_id": dish_id,
        "name": row["Name"],
        "primary_cuisine": row["primary_cuisine"],
        "primary_coarse": row["primary_coarse"],
        "countries": row["Countries"],
        "region": row.get("Region1", ""),
        "description": row.get("Text Description", ""),
        "ingredients_matched": sorted(ingredients),
        "n_ingredients": len(ingredients),
        "has_cdb_match": dish_id in cdb_matches,
        "source": "worldcuisines",
    })

df_results = pd.DataFrame(results)

# ── Step 5: Assess coverage ───────────────────────────────────────
print("\n=== Coverage assessment ===")
print(f"Total new dishes: {len(df_results)}")
print(f"  With CulinaryDB match: {df_results['has_cdb_match'].sum()}")
print(f"  With ≥1 mapped ingredient: {(df_results['n_ingredients'] >= 1).sum()}")
print(f"  With ≥3 mapped ingredients: {(df_results['n_ingredients'] >= 3).sum()}")
print(f"  With ≥5 mapped ingredients: {(df_results['n_ingredients'] >= 5).sum()}")
print(f"  With 0 ingredients (need LLM): {(df_results['n_ingredients'] == 0).sum()}")

print(f"\nIngredient count distribution:")
for n in range(0, 11):
    count = (df_results['n_ingredients'] == n).sum()
    if count > 0:
        print(f"  {n} ingredients: {count} dishes")
count = (df_results['n_ingredients'] > 10).sum()
if count > 0:
    print(f"  >10 ingredients: {count} dishes")

# Save intermediate results
df_results.to_csv(os.path.join(DATA, "worldcuisines_matched.csv"), index=False)
print(f"\nSaved to data/worldcuisines_matched.csv")

# ── Step 6: Identify dishes needing LLM supplement ────────────────
need_llm = df_results[df_results["n_ingredients"] < 3].copy()
print(f"\nDishes needing LLM recipe generation: {len(need_llm)}")
print(f"  0 ingredients: {(need_llm['n_ingredients'] == 0).sum()}")
print(f"  1-2 ingredients: {((need_llm['n_ingredients'] >= 1) & (need_llm['n_ingredients'] <= 2)).sum()}")

# Save list for LLM processing
need_llm[["dish_id", "name", "primary_cuisine", "primary_coarse", "description"]].to_csv(
    os.path.join(DATA, "dishes_needing_llm_recipes.csv"), index=False
)
print(f"Saved dishes needing LLM to data/dishes_needing_llm_recipes.csv")

# ── Summary by cuisine ───────────────────────────────────────────
print("\n=== Coverage by cuisine (top 20) ===")
cuisine_stats = df_results.groupby("primary_cuisine").agg(
    total=("dish_id", "count"),
    has_ingredients=("n_ingredients", lambda x: (x >= 3).sum()),
    needs_llm=("n_ingredients", lambda x: (x < 3).sum()),
).sort_values("total", ascending=False)

for cuisine, row in cuisine_stats.head(20).iterrows():
    pct = row["has_ingredients"] / row["total"] * 100
    print(f"  {cuisine:25s}: {int(row['total']):4d} total, {int(row['has_ingredients']):4d} have ≥3 ings ({pct:.0f}%), {int(row['needs_llm']):4d} need LLM")

print("\n✓ Step 1 complete. Run next: LLM recipe generation for uncovered dishes.")
