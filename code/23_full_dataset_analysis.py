"""
23_full_dataset_analysis.py — Full 2,563-dish DEI Analysis
==========================================================
Re-runs all extensible analyses on the combined v2 dataset.
Enriches combined_dish_DEI_v2.csv with category, meal_role, NDI, protein_g, calorie_kcal.
Then runs analyses A–G producing updated tables and figures.

Runtime: ~1–2 min, no API calls.
"""

import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "figure.max_open_warning": 50})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

# ══════════════════════════════════════════════════════════════════
# SECTION 0: Data Loading & Recipe Unification
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("23: Full 2,563-dish DEI Analysis")
print("=" * 60)

def _extract_recipes(script_path, var_name):
    """Extract recipe dict from a script file by parsing the assignment."""
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

# Load DEI v2
dei = pd.read_csv(DATA_DIR / "combined_dish_DEI_v2.csv")
if "dish_id" in dei.columns:
    dei = dei.set_index("dish_id")
print(f"Loaded combined_dish_DEI_v2: {len(dei)} dishes")

# Load expanded env costs (has ingredients_json, primary_coarse)
expanded = pd.read_csv(DATA_DIR / "expanded_dish_env_costs_v2.csv")
expanded = expanded.set_index("dish_id")
print(f"Loaded expanded_dish_env_costs_v2: {len(expanded)} new dishes")

# Build unified recipe dict
ALL_RECIPES = {}

# Original 334 dishes from hardcoded recipe scripts
ALL_RECIPES.update(_extract_recipes(ROOT / "code" / "04_env_cost_calculation.py", "DISH_RECIPES"))
ALL_RECIPES.update(_extract_recipes(ROOT / "code" / "09b_expanded_recipes.py", "EXPANDED_RECIPES"))
print(f"Legacy recipes: {len(ALL_RECIPES)}")

# New 2,230 dishes from ingredients_json
n_new = 0
for dish_id, row in expanded.iterrows():
    if dish_id not in ALL_RECIPES and pd.notna(row.get("ingredients_json")):
        try:
            ing_list = json.loads(row["ingredients_json"])
            ingredients = {item["name"]: item["grams"] for item in ing_list}
            ALL_RECIPES[dish_id] = {
                "ingredients": ingredients,
                "cook_method": row.get("cook_method", "no_cook"),
            }
            n_new += 1
        except (json.JSONDecodeError, KeyError):
            pass
print(f"New recipes parsed: {n_new}")
print(f"Total unified recipes: {len(ALL_RECIPES)}")

# ══════════════════════════════════════════════════════════════════
# SECTION 1: USDA Nutrient Database (from 11c)
# ══════════════════════════════════════════════════════════════════
# Values per kg (1000g), sourced from USDA FoodData Central SR28
NUTRIENT_DATA = {
    "coffee":       {"protein_g": 1, "fat_g": 0, "carb_g": 0, "fiber_g": 0, "iron_mg": 2, "zinc_mg": 0.5, "b12_ug": 0, "calcium_mg": 20, "vitamin_c_mg": 0, "calorie_kcal": 10},
    "tea":          {"protein_g": 0, "fat_g": 0, "carb_g": 3, "fiber_g": 0, "iron_mg": 0, "zinc_mg": 0.2, "b12_ug": 0, "calcium_mg": 5, "vitamin_c_mg": 0, "calorie_kcal": 10},
    "basil":        {"protein_g": 32, "fat_g": 6, "carb_g": 27, "fiber_g": 16, "iron_mg": 32, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 1770, "vitamin_c_mg": 180, "calorie_kcal": 230},
    "black_pepper": {"protein_g": 104, "fat_g": 33, "carb_g": 639, "fiber_g": 253, "iron_mg": 97, "zinc_mg": 12, "b12_ug": 0, "calcium_mg": 4430, "vitamin_c_mg": 0, "calorie_kcal": 2510},
    "chili":        {"protein_g": 120, "fat_g": 60, "carb_g": 500, "fiber_g": 280, "iron_mg": 75, "zinc_mg": 25, "b12_ug": 0, "calcium_mg": 1480, "vitamin_c_mg": 760, "calorie_kcal": 2820},
    "coconut_milk": {"protein_g": 22, "fat_g": 235, "carb_g": 57, "fiber_g": 22, "iron_mg": 16, "zinc_mg": 7, "b12_ug": 0, "calcium_mg": 160, "vitamin_c_mg": 28, "calorie_kcal": 2300},
    "coriander":    {"protein_g": 21, "fat_g": 5, "carb_g": 37, "fiber_g": 28, "iron_mg": 18, "zinc_mg": 5, "b12_ug": 0, "calcium_mg": 670, "vitamin_c_mg": 270, "calorie_kcal": 230},
    "cumin":        {"protein_g": 178, "fat_g": 222, "carb_g": 441, "fiber_g": 106, "iron_mg": 663, "zinc_mg": 48, "b12_ug": 0, "calcium_mg": 9310, "vitamin_c_mg": 77, "calorie_kcal": 3750},
    "curry_paste":  {"protein_g": 30, "fat_g": 100, "carb_g": 100, "fiber_g": 20, "iron_mg": 30, "zinc_mg": 5, "b12_ug": 0, "calcium_mg": 500, "vitamin_c_mg": 30, "calorie_kcal": 1400},
    "fish_sauce":   {"protein_g": 51, "fat_g": 0, "carb_g": 36, "fiber_g": 0, "iron_mg": 8, "zinc_mg": 2, "b12_ug": 3, "calcium_mg": 440, "vitamin_c_mg": 0, "calorie_kcal": 350},
    "ginger":       {"protein_g": 18, "fat_g": 8, "carb_g": 179, "fiber_g": 20, "iron_mg": 6, "zinc_mg": 3, "b12_ug": 0, "calcium_mg": 160, "vitamin_c_mg": 50, "calorie_kcal": 800},
    "hoisin_sauce": {"protein_g": 33, "fat_g": 36, "carb_g": 444, "fiber_g": 40, "iron_mg": 15, "zinc_mg": 5, "b12_ug": 0, "calcium_mg": 350, "vitamin_c_mg": 0, "calorie_kcal": 2200},
    "ketchup":      {"protein_g": 13, "fat_g": 1, "carb_g": 258, "fiber_g": 4, "iron_mg": 5, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 140, "vitamin_c_mg": 150, "calorie_kcal": 1010},
    "mayonnaise":   {"protein_g": 10, "fat_g": 750, "carb_g": 8, "fiber_g": 0, "iron_mg": 5, "zinc_mg": 3, "b12_ug": 1, "calcium_mg": 120, "vitamin_c_mg": 0, "calorie_kcal": 6800},
    "mustard":      {"protein_g": 41, "fat_g": 40, "carb_g": 58, "fiber_g": 40, "iron_mg": 23, "zinc_mg": 6, "b12_ug": 0, "calcium_mg": 630, "vitamin_c_mg": 17, "calorie_kcal": 600},
    "oyster_sauce": {"protein_g": 18, "fat_g": 3, "carb_g": 110, "fiber_g": 0, "iron_mg": 10, "zinc_mg": 3, "b12_ug": 1, "calcium_mg": 250, "vitamin_c_mg": 0, "calorie_kcal": 510},
    "salt":         {"protein_g": 0, "fat_g": 0, "carb_g": 0, "fiber_g": 0, "iron_mg": 3, "zinc_mg": 0.1, "b12_ug": 0, "calcium_mg": 240, "vitamin_c_mg": 0, "calorie_kcal": 0},
    "soy_sauce":    {"protein_g": 53, "fat_g": 1, "carb_g": 48, "fiber_g": 0, "iron_mg": 20, "zinc_mg": 5, "b12_ug": 0, "calcium_mg": 200, "vitamin_c_mg": 0, "calorie_kcal": 530},
    "sugar":        {"protein_g": 0, "fat_g": 0, "carb_g": 1000, "fiber_g": 0, "iron_mg": 1, "zinc_mg": 0.1, "b12_ug": 0, "calcium_mg": 10, "vitamin_c_mg": 0, "calorie_kcal": 3870},
    "tomato_sauce": {"protein_g": 13, "fat_g": 4, "carb_g": 72, "fiber_g": 15, "iron_mg": 10, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 180, "vitamin_c_mg": 120, "calorie_kcal": 290},
    "turmeric":     {"protein_g": 79, "fat_g": 100, "carb_g": 678, "fiber_g": 213, "iron_mg": 417, "zinc_mg": 41, "b12_ug": 0, "calcium_mg": 1830, "vitamin_c_mg": 258, "calorie_kcal": 3120},
    "vinegar":      {"protein_g": 0, "fat_g": 0, "carb_g": 9, "fiber_g": 0, "iron_mg": 3, "zinc_mg": 0.5, "b12_ug": 0, "calcium_mg": 60, "vitamin_c_mg": 5, "calorie_kcal": 180},
    "butter":       {"protein_g": 9, "fat_g": 810, "carb_g": 1, "fiber_g": 0, "iron_mg": 0.2, "zinc_mg": 0.9, "b12_ug": 1.7, "calcium_mg": 240, "vitamin_c_mg": 0, "calorie_kcal": 7170},
    "cheese":       {"protein_g": 250, "fat_g": 330, "carb_g": 13, "fiber_g": 0, "iron_mg": 7, "zinc_mg": 33, "b12_ug": 15, "calcium_mg": 7210, "vitamin_c_mg": 0, "calorie_kcal": 4030},
    "cream":        {"protein_g": 21, "fat_g": 365, "carb_g": 28, "fiber_g": 0, "iron_mg": 0.4, "zinc_mg": 2.2, "b12_ug": 1.1, "calcium_mg": 650, "vitamin_c_mg": 6, "calorie_kcal": 3450},
    "egg":          {"protein_g": 126, "fat_g": 99, "carb_g": 7, "fiber_g": 0, "iron_mg": 17, "zinc_mg": 13, "b12_ug": 11, "calcium_mg": 560, "vitamin_c_mg": 0, "calorie_kcal": 1430},
    "milk":         {"protein_g": 33, "fat_g": 33, "carb_g": 48, "fiber_g": 0, "iron_mg": 0.3, "zinc_mg": 4, "b12_ug": 4.5, "calcium_mg": 1130, "vitamin_c_mg": 0, "calorie_kcal": 610},
    "mozzarella":   {"protein_g": 224, "fat_g": 224, "carb_g": 22, "fiber_g": 0, "iron_mg": 4, "zinc_mg": 28, "b12_ug": 22, "calcium_mg": 5050, "vitamin_c_mg": 0, "calorie_kcal": 3000},
    "parmesan":     {"protein_g": 358, "fat_g": 259, "carb_g": 32, "fiber_g": 0, "iron_mg": 8, "zinc_mg": 29, "b12_ug": 12, "calcium_mg": 11840, "vitamin_c_mg": 0, "calorie_kcal": 3920},
    "yogurt":       {"protein_g": 100, "fat_g": 6, "carb_g": 36, "fiber_g": 0, "iron_mg": 1, "zinc_mg": 6, "b12_ug": 7.5, "calcium_mg": 1100, "vitamin_c_mg": 5, "calorie_kcal": 590},
    "apple":        {"protein_g": 3, "fat_g": 2, "carb_g": 138, "fiber_g": 24, "iron_mg": 1.2, "zinc_mg": 0.4, "b12_ug": 0, "calcium_mg": 60, "vitamin_c_mg": 46, "calorie_kcal": 520},
    "avocado":      {"protein_g": 20, "fat_g": 147, "carb_g": 86, "fiber_g": 67, "iron_mg": 5.5, "zinc_mg": 6.4, "b12_ug": 0, "calcium_mg": 120, "vitamin_c_mg": 100, "calorie_kcal": 1600},
    "banana":       {"protein_g": 11, "fat_g": 3, "carb_g": 228, "fiber_g": 26, "iron_mg": 2.6, "zinc_mg": 1.5, "b12_ug": 0, "calcium_mg": 50, "vitamin_c_mg": 87, "calorie_kcal": 890},
    "coconut":      {"protein_g": 33, "fat_g": 334, "carb_g": 153, "fiber_g": 90, "iron_mg": 24, "zinc_mg": 11, "b12_ug": 0, "calcium_mg": 140, "vitamin_c_mg": 33, "calorie_kcal": 3540},
    "lemon":        {"protein_g": 11, "fat_g": 3, "carb_g": 93, "fiber_g": 28, "iron_mg": 6, "zinc_mg": 0.6, "b12_ug": 0, "calcium_mg": 260, "vitamin_c_mg": 530, "calorie_kcal": 290},
    "lime":         {"protein_g": 7, "fat_g": 2, "carb_g": 107, "fiber_g": 28, "iron_mg": 6, "zinc_mg": 1.1, "b12_ug": 0, "calcium_mg": 330, "vitamin_c_mg": 290, "calorie_kcal": 300},
    "mango":        {"protein_g": 8, "fat_g": 4, "carb_g": 150, "fiber_g": 16, "iron_mg": 1.6, "zinc_mg": 0.9, "b12_ug": 0, "calcium_mg": 110, "vitamin_c_mg": 365, "calorie_kcal": 600},
    "pineapple":    {"protein_g": 5, "fat_g": 1, "carb_g": 132, "fiber_g": 14, "iron_mg": 2.9, "zinc_mg": 1.2, "b12_ug": 0, "calcium_mg": 130, "vitamin_c_mg": 479, "calorie_kcal": 500},
    "bread":        {"protein_g": 90, "fat_g": 33, "carb_g": 490, "fiber_g": 27, "iron_mg": 36, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 2600, "vitamin_c_mg": 0, "calorie_kcal": 2650},
    "corn":         {"protein_g": 32, "fat_g": 12, "carb_g": 190, "fiber_g": 20, "iron_mg": 5, "zinc_mg": 5, "b12_ug": 0, "calcium_mg": 20, "vitamin_c_mg": 68, "calorie_kcal": 860},
    "oats":         {"protein_g": 169, "fat_g": 69, "carb_g": 661, "fiber_g": 106, "iron_mg": 47, "zinc_mg": 40, "b12_ug": 0, "calcium_mg": 540, "vitamin_c_mg": 0, "calorie_kcal": 3890},
    "pasta_dry":    {"protein_g": 130, "fat_g": 15, "carb_g": 750, "fiber_g": 31, "iron_mg": 35, "zinc_mg": 14, "b12_ug": 0, "calcium_mg": 210, "vitamin_c_mg": 0, "calorie_kcal": 3710},
    "rice":         {"protein_g": 27, "fat_g": 3, "carb_g": 282, "fiber_g": 4, "iron_mg": 8, "zinc_mg": 11, "b12_ug": 0, "calcium_mg": 280, "vitamin_c_mg": 0, "calorie_kcal": 1300},
    "rice_noodle":  {"protein_g": 32, "fat_g": 2, "carb_g": 832, "fiber_g": 10, "iron_mg": 2, "zinc_mg": 4, "b12_ug": 0, "calcium_mg": 80, "vitamin_c_mg": 0, "calorie_kcal": 3600},
    "tortilla":     {"protein_g": 85, "fat_g": 75, "carb_g": 480, "fiber_g": 35, "iron_mg": 30, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 1500, "vitamin_c_mg": 0, "calorie_kcal": 3060},
    "wheat_flour":  {"protein_g": 100, "fat_g": 10, "carb_g": 763, "fiber_g": 27, "iron_mg": 46, "zinc_mg": 7, "b12_ug": 0, "calcium_mg": 150, "vitamin_c_mg": 0, "calorie_kcal": 3640},
    "black_bean":   {"protein_g": 215, "fat_g": 9, "carb_g": 627, "fiber_g": 156, "iron_mg": 50, "zinc_mg": 35, "b12_ug": 0, "calcium_mg": 1230, "vitamin_c_mg": 0, "calorie_kcal": 3410},
    "chickpea":     {"protein_g": 209, "fat_g": 61, "carb_g": 610, "fiber_g": 174, "iron_mg": 62, "zinc_mg": 34, "b12_ug": 0, "calcium_mg": 1050, "vitamin_c_mg": 40, "calorie_kcal": 3640},
    "lentil":       {"protein_g": 254, "fat_g": 11, "carb_g": 601, "fiber_g": 307, "iron_mg": 76, "zinc_mg": 33, "b12_ug": 0, "calcium_mg": 350, "vitamin_c_mg": 44, "calorie_kcal": 3520},
    "peanut":       {"protein_g": 258, "fat_g": 493, "carb_g": 161, "fiber_g": 85, "iron_mg": 46, "zinc_mg": 33, "b12_ug": 0, "calcium_mg": 920, "vitamin_c_mg": 0, "calorie_kcal": 5670},
    "soybean":      {"protein_g": 365, "fat_g": 198, "carb_g": 302, "fiber_g": 92, "iron_mg": 157, "zinc_mg": 49, "b12_ug": 0, "calcium_mg": 2770, "vitamin_c_mg": 60, "calorie_kcal": 4460},
    "tofu":         {"protein_g": 80, "fat_g": 48, "carb_g": 19, "fiber_g": 3, "iron_mg": 54, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 3500, "vitamin_c_mg": 1, "calorie_kcal": 760},
    "bacon":        {"protein_g": 370, "fat_g": 420, "carb_g": 13, "fiber_g": 0, "iron_mg": 12, "zinc_mg": 30, "b12_ug": 11, "calcium_mg": 110, "vitamin_c_mg": 0, "calorie_kcal": 5410},
    "beef":         {"protein_g": 260, "fat_g": 150, "carb_g": 0, "fiber_g": 0, "iron_mg": 26, "zinc_mg": 45, "b12_ug": 25, "calcium_mg": 180, "vitamin_c_mg": 0, "calorie_kcal": 2500},
    "ground_beef":  {"protein_g": 260, "fat_g": 150, "carb_g": 0, "fiber_g": 0, "iron_mg": 26, "zinc_mg": 45, "b12_ug": 25, "calcium_mg": 180, "vitamin_c_mg": 0, "calorie_kcal": 2500},
    "lamb":         {"protein_g": 253, "fat_g": 210, "carb_g": 0, "fiber_g": 0, "iron_mg": 17, "zinc_mg": 38, "b12_ug": 26, "calcium_mg": 170, "vitamin_c_mg": 0, "calorie_kcal": 2940},
    "pork":         {"protein_g": 273, "fat_g": 140, "carb_g": 0, "fiber_g": 0, "iron_mg": 8, "zinc_mg": 19, "b12_ug": 7, "calcium_mg": 80, "vitamin_c_mg": 6, "calorie_kcal": 2420},
    "sausage":      {"protein_g": 190, "fat_g": 280, "carb_g": 20, "fiber_g": 0, "iron_mg": 12, "zinc_mg": 20, "b12_ug": 13, "calcium_mg": 130, "vitamin_c_mg": 0, "calorie_kcal": 3390},
    "almond":       {"protein_g": 212, "fat_g": 494, "carb_g": 217, "fiber_g": 125, "iron_mg": 37, "zinc_mg": 31, "b12_ug": 0, "calcium_mg": 2690, "vitamin_c_mg": 0, "calorie_kcal": 5790},
    "cashew":       {"protein_g": 183, "fat_g": 438, "carb_g": 305, "fiber_g": 33, "iron_mg": 67, "zinc_mg": 58, "b12_ug": 0, "calcium_mg": 370, "vitamin_c_mg": 5, "calorie_kcal": 5530},
    "pine_nut":     {"protein_g": 137, "fat_g": 681, "carb_g": 131, "fiber_g": 37, "iron_mg": 55, "zinc_mg": 64, "b12_ug": 0, "calcium_mg": 160, "vitamin_c_mg": 8, "calorie_kcal": 6730},
    "sesame_seed":  {"protein_g": 179, "fat_g": 496, "carb_g": 234, "fiber_g": 118, "iron_mg": 146, "zinc_mg": 78, "b12_ug": 0, "calcium_mg": 9750, "vitamin_c_mg": 0, "calorie_kcal": 5730},
    "walnut":       {"protein_g": 152, "fat_g": 654, "carb_g": 139, "fiber_g": 67, "iron_mg": 29, "zinc_mg": 31, "b12_ug": 0, "calcium_mg": 980, "vitamin_c_mg": 13, "calorie_kcal": 6540},
    "coconut_oil":  {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0.4, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 10, "vitamin_c_mg": 0, "calorie_kcal": 8620},
    "olive_oil":    {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 5.6, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 10, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    "palm_oil":     {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0.1, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    "sesame_oil":   {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    "vegetable_oil":{"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0.6, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    "chocolate":    {"protein_g": 76, "fat_g": 430, "carb_g": 460, "fiber_g": 109, "iron_mg": 118, "zinc_mg": 33, "b12_ug": 3, "calcium_mg": 730, "vitamin_c_mg": 0, "calorie_kcal": 5460},
    "honey":        {"protein_g": 3, "fat_g": 0, "carb_g": 824, "fiber_g": 2, "iron_mg": 4, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 60, "vitamin_c_mg": 5, "calorie_kcal": 3040},
    "maple_syrup":  {"protein_g": 0, "fat_g": 1, "carb_g": 670, "fiber_g": 0, "iron_mg": 11, "zinc_mg": 14, "b12_ug": 0, "calcium_mg": 1020, "vitamin_c_mg": 0, "calorie_kcal": 2600},
    "chicken":      {"protein_g": 239, "fat_g": 73, "carb_g": 0, "fiber_g": 0, "iron_mg": 9, "zinc_mg": 16, "b12_ug": 3, "calcium_mg": 150, "vitamin_c_mg": 0, "calorie_kcal": 1650},
    "duck":         {"protein_g": 192, "fat_g": 284, "carb_g": 0, "fiber_g": 0, "iron_mg": 25, "zinc_mg": 18, "b12_ug": 30, "calcium_mg": 110, "vitamin_c_mg": 28, "calorie_kcal": 3370},
    "turkey":       {"protein_g": 294, "fat_g": 15, "carb_g": 0, "fiber_g": 0, "iron_mg": 13, "zinc_mg": 21, "b12_ug": 10, "calcium_mg": 140, "vitamin_c_mg": 0, "calorie_kcal": 1350},
    "cod":          {"protein_g": 179, "fat_g": 8, "carb_g": 0, "fiber_g": 0, "iron_mg": 4, "zinc_mg": 5, "b12_ug": 9, "calcium_mg": 160, "vitamin_c_mg": 10, "calorie_kcal": 820},
    "crab":         {"protein_g": 184, "fat_g": 11, "carb_g": 0, "fiber_g": 0, "iron_mg": 7, "zinc_mg": 37, "b12_ug": 91, "calcium_mg": 590, "vitamin_c_mg": 30, "calorie_kcal": 830},
    "fish":         {"protein_g": 200, "fat_g": 33, "carb_g": 0, "fiber_g": 0, "iron_mg": 5, "zinc_mg": 5, "b12_ug": 10, "calcium_mg": 120, "vitamin_c_mg": 0, "calorie_kcal": 1100},
    "salmon":       {"protein_g": 208, "fat_g": 127, "carb_g": 0, "fiber_g": 0, "iron_mg": 8, "zinc_mg": 6, "b12_ug": 32, "calcium_mg": 120, "vitamin_c_mg": 0, "calorie_kcal": 2080},
    "shrimp":       {"protein_g": 241, "fat_g": 17, "carb_g": 2, "fiber_g": 0, "iron_mg": 22, "zinc_mg": 15, "b12_ug": 15, "calcium_mg": 700, "vitamin_c_mg": 20, "calorie_kcal": 990},
    "squid":        {"protein_g": 158, "fat_g": 12, "carb_g": 31, "fiber_g": 0, "iron_mg": 7, "zinc_mg": 15, "b12_ug": 13, "calcium_mg": 320, "vitamin_c_mg": 49, "calorie_kcal": 920},
    "tuna":         {"protein_g": 236, "fat_g": 49, "carb_g": 0, "fiber_g": 0, "iron_mg": 10, "zinc_mg": 6, "b12_ug": 98, "calcium_mg": 80, "vitamin_c_mg": 0, "calorie_kcal": 1440},
    "potato":       {"protein_g": 20, "fat_g": 1, "carb_g": 170, "fiber_g": 22, "iron_mg": 8, "zinc_mg": 3, "b12_ug": 0, "calcium_mg": 120, "vitamin_c_mg": 198, "calorie_kcal": 770},
    "sweet_potato": {"protein_g": 16, "fat_g": 1, "carb_g": 201, "fiber_g": 30, "iron_mg": 6, "zinc_mg": 3, "b12_ug": 0, "calcium_mg": 300, "vitamin_c_mg": 24, "calorie_kcal": 860},
    "bamboo_shoot": {"protein_g": 26, "fat_g": 3, "carb_g": 52, "fiber_g": 22, "iron_mg": 5, "zinc_mg": 11, "b12_ug": 0, "calcium_mg": 130, "vitamin_c_mg": 40, "calorie_kcal": 270},
    "bean_sprout":  {"protein_g": 31, "fat_g": 2, "carb_g": 60, "fiber_g": 18, "iron_mg": 9, "zinc_mg": 4, "b12_ug": 0, "calcium_mg": 130, "vitamin_c_mg": 132, "calorie_kcal": 310},
    "broccoli":     {"protein_g": 28, "fat_g": 4, "carb_g": 67, "fiber_g": 26, "iron_mg": 7, "zinc_mg": 4, "b12_ug": 0, "calcium_mg": 470, "vitamin_c_mg": 893, "calorie_kcal": 340},
    "cabbage":      {"protein_g": 13, "fat_g": 1, "carb_g": 58, "fiber_g": 25, "iron_mg": 5, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 400, "vitamin_c_mg": 365, "calorie_kcal": 250},
    "carrot":       {"protein_g": 9, "fat_g": 2, "carb_g": 96, "fiber_g": 28, "iron_mg": 3, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 330, "vitamin_c_mg": 59, "calorie_kcal": 410},
    "celery":       {"protein_g": 7, "fat_g": 2, "carb_g": 30, "fiber_g": 16, "iron_mg": 2, "zinc_mg": 1, "b12_ug": 0, "calcium_mg": 400, "vitamin_c_mg": 31, "calorie_kcal": 160},
    "cucumber":     {"protein_g": 7, "fat_g": 1, "carb_g": 36, "fiber_g": 5, "iron_mg": 3, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 160, "vitamin_c_mg": 28, "calorie_kcal": 150},
    "eggplant":     {"protein_g": 10, "fat_g": 2, "carb_g": 57, "fiber_g": 30, "iron_mg": 2, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 90, "vitamin_c_mg": 22, "calorie_kcal": 250},
    "garlic":       {"protein_g": 64, "fat_g": 5, "carb_g": 331, "fiber_g": 21, "iron_mg": 17, "zinc_mg": 12, "b12_ug": 0, "calcium_mg": 1810, "vitamin_c_mg": 312, "calorie_kcal": 1490},
    "lettuce":      {"protein_g": 14, "fat_g": 2, "carb_g": 29, "fiber_g": 13, "iron_mg": 10, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 360, "vitamin_c_mg": 92, "calorie_kcal": 150},
    "mushroom":     {"protein_g": 31, "fat_g": 3, "carb_g": 33, "fiber_g": 10, "iron_mg": 5, "zinc_mg": 5, "b12_ug": 0.4, "calcium_mg": 30, "vitamin_c_mg": 21, "calorie_kcal": 220},
    "onion":        {"protein_g": 11, "fat_g": 1, "carb_g": 93, "fiber_g": 17, "iron_mg": 2, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 230, "vitamin_c_mg": 74, "calorie_kcal": 400},
    "pepper":       {"protein_g": 10, "fat_g": 3, "carb_g": 62, "fiber_g": 17, "iron_mg": 3, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 70, "vitamin_c_mg": 1277, "calorie_kcal": 260},
    "spinach":      {"protein_g": 29, "fat_g": 4, "carb_g": 36, "fiber_g": 22, "iron_mg": 27, "zinc_mg": 5, "b12_ug": 0, "calcium_mg": 990, "vitamin_c_mg": 282, "calorie_kcal": 230},
    "tomato":       {"protein_g": 9, "fat_g": 2, "carb_g": 39, "fiber_g": 12, "iron_mg": 3, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 100, "vitamin_c_mg": 139, "calorie_kcal": 180},
    "zucchini":     {"protein_g": 12, "fat_g": 3, "carb_g": 31, "fiber_g": 10, "iron_mg": 4, "zinc_mg": 3, "b12_ug": 0, "calcium_mg": 160, "vitamin_c_mg": 179, "calorie_kcal": 170},
    "pita":         {"protein_g": 90, "fat_g": 12, "carb_g": 558, "fiber_g": 22, "iron_mg": 26, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 860, "vitamin_c_mg": 0, "calorie_kcal": 2750},
    "vanilla":      {"protein_g": 1, "fat_g": 1, "carb_g": 130, "fiber_g": 0, "iron_mg": 1, "zinc_mg": 1, "b12_ug": 0, "calcium_mg": 110, "vitamin_c_mg": 0, "calorie_kcal": 510},
    "ice_cream":    {"protein_g": 35, "fat_g": 110, "carb_g": 240, "fiber_g": 0, "iron_mg": 1, "zinc_mg": 5, "b12_ug": 4, "calcium_mg": 1280, "vitamin_c_mg": 6, "calorie_kcal": 2070},
    "wine":         {"protein_g": 1, "fat_g": 0, "carb_g": 26, "fiber_g": 0, "iron_mg": 5, "zinc_mg": 1, "b12_ug": 0, "calcium_mg": 80, "vitamin_c_mg": 0, "calorie_kcal": 830},
    "saffron":      {"protein_g": 114, "fat_g": 59, "carb_g": 651, "fiber_g": 38, "iron_mg": 114, "zinc_mg": 11, "b12_ug": 0, "calcium_mg": 1110, "vitamin_c_mg": 808, "calorie_kcal": 3100},
    "water":        {"protein_g": 0, "fat_g": 0, "carb_g": 0, "fiber_g": 0, "iron_mg": 0, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 0},
}

DRV = {
    "protein_g": 50, "fiber_g": 25, "iron_mg": 18,
    "zinc_mg": 11, "b12_ug": 2.4, "calcium_mg": 1000, "vitamin_c_mg": 90,
}

# ══════════════════════════════════════════════════════════════════
# SECTION 2: Compute Nutrition, Meal Role, Category for all 2,563
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ENRICHMENT: Computing nutrition, meal role, category...")
print("=" * 60)

# --- 2a. Nutritional profiles ---
RED_MEAT = {"beef", "ground_beef", "lamb", "veal"}
POULTRY = {"chicken", "duck", "turkey"}
SEAFOOD = {"fish", "salmon", "shrimp", "squid", "tuna", "cod", "crab"}
LEGUME = {"black_bean", "chickpea", "lentil", "soybean", "tofu", "peanut"}
DAIRY_HEAVY = {"cheese", "mozzarella", "parmesan", "cream", "butter", "yogurt"}
DESSERT_INDICATORS = {"sugar", "chocolate", "honey", "maple_syrup", "vanilla"}

# Coarse-to-category fallback mapping for WorldCuisines dishes
COARSE_TO_CATEGORY = {
    "meat": "Red Meat Main", "lamb": "Red Meat Main", "mutton": "Red Meat Main",
    "chicken": "Poultry Main",
    "fish": "Seafood Main", "seafood": "Seafood Main",
    "sausage": "Pork Main",
    "beans": "Plant Protein", "lentil": "Plant Protein", "peas": "Plant Protein",
    "tofu": "Plant Protein", "soybeans": "Plant Protein",
    "rice": "Starch/Carb", "noodle": "Starch/Carb", "pasta": "Starch/Carb",
    "bread": "Starch/Carb", "flatbread": "Starch/Carb", "pizza": "Starch/Carb",
    "corn": "Starch/Carb", "cereal": "Starch/Carb", "porridge": "Starch/Carb",
    "grain": "Starch/Carb", "potato": "Starch/Carb", "biscuit": "Starch/Carb",
    "egg": "Egg Dish", "omelette": "Egg Dish",
    "cheese": "Dairy Main", "dairy": "Dairy Main",
    "soup": "Soup/Stew", "stew": "Soup/Stew", "hot pot": "Soup/Stew", "curry": "Soup/Stew", "casserole": "Soup/Stew",
    "salad": "Salad/Cold", "vegetable": "Salad/Cold",
    "dessert": "Dessert", "cake": "Dessert", "candy": "Dessert",
    "confectionery": "Dessert", "cookies": "Dessert", "pudding": "Dessert",
    "sweets": "Dessert", "pastry": "Dessert", "doughnut": "Dessert", "fruit": "Dessert",
    "beverages": "Beverage",
    "snack": "Starch/Carb", "fritter": "Starch/Carb", "pancake": "Starch/Carb",
    "crepe": "Starch/Carb", "roll": "Starch/Carb", "wrap": "Starch/Carb",
    "sandwich": "Starch/Carb", "dough": "Starch/Carb",
    "dim sum": "Mixed/Other", "dumpling": "Mixed/Other",
    "cutlet": "Mixed/Other", "skewer": "Mixed/Other", "stir fry": "Mixed/Other",
    "side dish": "Mixed/Other", "platter": "Mixed/Other",
    "nuts": "Mixed/Other", "caviar": "Mixed/Other", "banana": "Dessert",
}

BEVERAGE_DISHES = {"thai_iced_tea", "ca_phe_sua_da", "coquito", "halo_halo"}

def classify_dish(dish_id, recipe, cal):
    """Classify dish by dominant protein source and function (from 11d)."""
    ingredients = recipe["ingredients"]
    cook_method = recipe.get("cook_method", "raw")
    if dish_id in BEVERAGE_DISHES:
        return "Beverage"
    red_g = sum(g for ing, g in ingredients.items() if ing in RED_MEAT)
    poultry_g = sum(g for ing, g in ingredients.items() if ing in POULTRY)
    seafood_g = sum(g for ing, g in ingredients.items() if ing in SEAFOOD)
    pork_g = sum(g for ing, g in ingredients.items() if ing in {"pork", "bacon", "sausage"})
    legume_g = sum(g for ing, g in ingredients.items() if ing in LEGUME)
    dairy_g = sum(g for ing, g in ingredients.items() if ing in DAIRY_HEAVY)
    dessert_g = sum(g for ing, g in ingredients.items() if ing in DESSERT_INDICATORS)
    total_g = sum(ingredients.values())
    # Dessert detection
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
    if cook_method in ("raw", "cold", "no_cook") and cal < 300:
        return "Salad/Cold"
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
    grain_items = {"rice", "pasta_dry", "rice_noodle", "wheat_flour", "bread",
                   "tortilla", "corn", "oats", "pita"}
    grain_g = sum(g for ing, g in ingredients.items() if ing in grain_items)
    if total_g > 0 and grain_g / total_g > 0.4:
        return "Starch/Carb"
    egg_g = ingredients.get("egg", 0)
    if egg_g >= 80 and red_g < 50 and poultry_g < 50:
        return "Egg Dish"
    if total_g > 0 and dairy_g / total_g > 0.3:
        return "Dairy Main"
    return "Mixed/Other"

# Compute for all dishes
cal_list, prot_list, ndi_list, role_list, cat_list = [], [], [], [], []
n_recipe_found = 0

for dish_id in dei.index:
    if dish_id in ALL_RECIPES:
        recipe = ALL_RECIPES[dish_id]
        ingredients = recipe["ingredients"]
        n_recipe_found += 1
        # Nutrient totals
        totals = {k: 0.0 for k in ["protein_g", "calorie_kcal"]}
        ndi_nutrients = {k: 0.0 for k in DRV}
        for ing, grams in ingredients.items():
            if ing in NUTRIENT_DATA:
                kg = grams / 1000
                totals["protein_g"] += kg * NUTRIENT_DATA[ing]["protein_g"]
                totals["calorie_kcal"] += kg * NUTRIENT_DATA[ing]["calorie_kcal"]
                for nutrient in DRV:
                    ndi_nutrients[nutrient] += kg * NUTRIENT_DATA[ing][nutrient]
        cal = totals["calorie_kcal"]
        prot = totals["protein_g"]
        # NDI
        if cal > 0:
            ndi_score = sum(min(ndi_nutrients[n] / drv * 100, 100) for n, drv in DRV.items()) / len(DRV)
            ndi = ndi_score / (cal / 100)
        else:
            ndi = 0
        # Meal role
        if cal < 200:
            role = "Side/Snack"
        elif cal < 400:
            role = "Light Main"
        elif cal < 700:
            role = "Full Main"
        else:
            role = "Heavy Main"
        # Category
        cat = classify_dish(dish_id, recipe, cal)
        # Fallback for Mixed/Other using primary_coarse
        if cat == "Mixed/Other" and dish_id in expanded.index:
            coarse = expanded.loc[dish_id].get("primary_coarse", "")
            if pd.notna(coarse) and coarse in COARSE_TO_CATEGORY:
                cat = COARSE_TO_CATEGORY[coarse]
    else:
        # No recipe — use LLM estimates if available
        if dish_id in expanded.index:
            row_e = expanded.loc[dish_id]
            cal = row_e.get("calories_approx", 400)
            prot = row_e.get("protein_g_approx", 10)
        else:
            cal, prot = 400, 10
        ndi = 0
        role = "Full Main" if cal >= 400 else ("Light Main" if cal >= 200 else "Side/Snack")
        coarse = expanded.loc[dish_id].get("primary_coarse", "") if dish_id in expanded.index else ""
        cat = COARSE_TO_CATEGORY.get(coarse, "Mixed/Other") if pd.notna(coarse) else "Mixed/Other"

    cal_list.append(cal)
    prot_list.append(prot)
    ndi_list.append(ndi)
    role_list.append(role)
    cat_list.append(cat)

dei["calorie_kcal"] = cal_list
dei["protein_g"] = prot_list
dei["NDI"] = ndi_list
dei["meal_role"] = role_list
dei["category"] = cat_list

print(f"Recipes found: {n_recipe_found}/{len(dei)}")
print(f"\nCategory distribution:")
for cat, n in dei["category"].value_counts().items():
    print(f"  {cat}: {n}")
mixed_pct = (dei["category"] == "Mixed/Other").mean() * 100
print(f"Mixed/Other: {mixed_pct:.1f}%")
print(f"\nMeal role distribution:")
for role, n in dei["meal_role"].value_counts().items():
    print(f"  {role}: {n}")
print(f"\nNDI: range [{dei['NDI'].min():.1f}, {dei['NDI'].max():.1f}], mean {dei['NDI'].mean():.1f}")

# Save enriched dataset
dei.to_csv(DATA_DIR / "combined_dish_DEI_v2.csv")
print(f"\nSaved enriched combined_dish_DEI_v2.csv ({len(dei)} dishes, {len(dei.columns)} columns)")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS A: Within-Category Variance Decomposition
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS A: Within-Category Variance Decomposition")
print("=" * 60)

cat_rows = []
for cat_name, grp in dei.groupby("category"):
    if len(grp) < 3:
        continue
    var_logH = grp["log_H"].var()
    var_logE = grp["log_E"].var()
    cov_HE = np.cov(grp["log_H"], grp["log_E"])[0, 1]
    sh_H = var_logH - cov_HE
    sh_E = var_logE - cov_HE
    total = sh_H + sh_E
    h_pct = sh_H / total * 100 if total > 0 else 0
    cat_rows.append({
        "category": cat_name,
        "n_dishes": len(grp),
        "H_cv_pct": grp["H_mean"].std() / grp["H_mean"].mean() * 100,
        "E_cv_pct": grp["E_composite"].std() / grp["E_composite"].mean() * 100,
        "H_contribution_pct": h_pct,
        "mean_logDEI": grp["log_DEI"].mean(),
    })

cat_df = pd.DataFrame(cat_rows).sort_values("H_contribution_pct", ascending=False)
cat_df.to_csv(TABLES_DIR / "within_category_variance_v2.csv", index=False)
print(cat_df.to_string(index=False))
mean_h_contrib = cat_df["H_contribution_pct"].mean()
print(f"\nMean within-category H contribution: {mean_h_contrib:.1f}%")

# Nutrition-constrained substitutions
print("\nComputing nutrition-constrained substitutions...")
subs = []
for cat_name, grp in dei.groupby("category"):
    if len(grp) < 2:
        continue
    dishes = grp.reset_index()
    for i in range(len(dishes)):
        for j in range(len(dishes)):
            if i == j:
                continue
            d_from = dishes.iloc[i]
            d_to = dishes.iloc[j]
            e_reduction = (d_from["E_composite"] - d_to["E_composite"]) / d_from["E_composite"]
            h_loss = d_from["H_mean"] - d_to["H_mean"]
            if e_reduction < 0.30:
                continue
            if h_loss > 1.0:
                continue
            prot_from = d_from["protein_g"]
            prot_to = d_to["protein_g"]
            if prot_from > 0 and prot_to / prot_from < 0.5:
                continue
            cal_from = d_from["calorie_kcal"]
            cal_to = d_to["calorie_kcal"]
            if cal_from > 0 and (cal_to / cal_from < 0.5 or cal_to / cal_from > 1.5):
                continue
            subs.append({
                "from": d_from["dish_id"],
                "to": d_to["dish_id"],
                "category": cat_name,
                "E_reduction_pct": e_reduction * 100,
                "H_change": -h_loss,
            })

subs_df = pd.DataFrame(subs)
subs_df.to_csv(TABLES_DIR / "nutrition_constrained_substitutions_v2.csv", index=False)
print(f"Viable substitutions: {len(subs_df)}")
if len(subs_df) > 0:
    print(f"Mean E reduction: {subs_df['E_reduction_pct'].mean():.1f}%")
    print(f"Mean H change: {subs_df['H_change'].mean():.2f}")

# Figure: within-category panels (top 9 categories)
top_cats = cat_df.head(9)["category"].values
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
for ax, cat_name in zip(axes.flat, top_cats):
    grp = dei[dei["category"] == cat_name]
    ax.scatter(grp["E_composite"], grp["H_mean"], alpha=0.5, s=15)
    ax.set_title(f"{cat_name} (n={len(grp)})", fontsize=9)
    ax.set_xlabel("E", fontsize=8)
    ax.set_ylabel("H", fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "within_category_dei_panels_v2.png", dpi=150)
plt.close()
print(f"Saved: within_category_dei_panels_v2.png")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS B: Meal-Level Analysis
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS B: Meal-Level Analysis")
print("=" * 60)

# Within-role variance decomposition
role_rows = []
for role_name, grp in dei.groupby("meal_role"):
    if len(grp) < 5:
        continue
    var_logH = grp["log_H"].var()
    var_logE = grp["log_E"].var()
    cov_HE = np.cov(grp["log_H"], grp["log_E"])[0, 1]
    sh_H = var_logH - cov_HE
    sh_E = var_logE - cov_HE
    total = sh_H + sh_E
    h_pct = sh_H / total * 100 if total > 0 else 0
    e_pct = sh_E / total * 100 if total > 0 else 0
    role_rows.append({
        "meal_role": role_name,
        "n": len(grp),
        "H_cv_pct": grp["H_mean"].std() / grp["H_mean"].mean() * 100,
        "E_cv_pct": grp["E_composite"].std() / grp["E_composite"].mean() * 100,
        "H_pct_var": h_pct,
        "E_pct_var": e_pct,
    })

role_df = pd.DataFrame(role_rows)
role_df.to_csv(TABLES_DIR / "within_role_variance_v2.csv", index=False)
print(role_df.to_string(index=False))

# Meal-level combinations
mains = dei[dei["meal_role"].isin(["Full Main", "Heavy Main"])].reset_index()
sides = dei[dei["meal_role"].isin(["Side/Snack", "Light Main"])].reset_index()
print(f"\nMains: {len(mains)}, Sides: {len(sides)}")

# Vectorized meal construction
m_cal = mains["calorie_kcal"].values
m_prot = mains["protein_g"].values
m_H = mains["H_mean"].values
m_E = mains["E_composite"].values
m_logH = mains["log_H"].values
m_logE = mains["log_E"].values

s_cal = sides["calorie_kcal"].values
s_prot = sides["protein_g"].values
s_H = sides["H_mean"].values
s_E = sides["E_composite"].values
s_logH = sides["log_H"].values
s_logE = sides["log_E"].values

# Use broadcasting for large cartesian product
total_cal = m_cal[:, None] + s_cal[None, :]  # (M, S)
total_prot = m_prot[:, None] + s_prot[None, :]
mask = (total_cal >= 500) & (total_cal <= 1500) & (total_prot >= 20)
valid_i, valid_j = np.where(mask)
n_valid = len(valid_i)
print(f"Valid meal combinations: {n_valid}")

if n_valid > 0:
    # Calorie-weighted H
    mc = m_cal[valid_i]
    sc = s_cal[valid_j]
    total_c = mc + sc
    meal_H = (mc * m_H[valid_i] + sc * s_H[valid_j]) / total_c
    meal_E = m_E[valid_i] + s_E[valid_j]
    meal_logDEI = np.log(meal_H) - np.log(meal_E)

    var_meal_logH = np.var(np.log(meal_H))
    var_meal_logE = np.var(np.log(meal_E))
    cov_meal = np.cov(np.log(meal_H), np.log(meal_E))[0, 1]
    sh_meal_H = var_meal_logH - cov_meal
    sh_meal_E = var_meal_logE - cov_meal
    total_meal = sh_meal_H + sh_meal_E
    meal_h_pct = sh_meal_H / total_meal * 100 if total_meal > 0 else 0

    dei_range = np.exp(meal_logDEI.max() - meal_logDEI.min())

    print(f"Meal-level H contribution: {meal_h_pct:.1f}%")
    print(f"Meal-level DEI range: {dei_range:.0f}x")

    # Calorie-equivalent substitutions within roles
    cal_eq_subs = 0
    e_reductions = []
    h_changes = []
    for role_name in ["Full Main", "Heavy Main", "Light Main", "Side/Snack"]:
        grp = dei[dei["meal_role"] == role_name].reset_index()
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(len(grp)):
                if i == j:
                    continue
                cal_ratio = grp.iloc[j]["calorie_kcal"] / grp.iloc[i]["calorie_kcal"] if grp.iloc[i]["calorie_kcal"] > 0 else 0
                prot_ratio = grp.iloc[j]["protein_g"] / grp.iloc[i]["protein_g"] if grp.iloc[i]["protein_g"] > 0 else 0
                e_red = (grp.iloc[i]["E_composite"] - grp.iloc[j]["E_composite"]) / grp.iloc[i]["E_composite"] if grp.iloc[i]["E_composite"] > 0 else 0
                if 0.75 <= cal_ratio <= 1.33 and 0.5 <= prot_ratio <= 2.0 and e_red > 0.20:
                    cal_eq_subs += 1
                    e_reductions.append(e_red * 100)
                    h_changes.append(grp.iloc[j]["H_mean"] - grp.iloc[i]["H_mean"])
    print(f"Calorie-equivalent substitution pairs: {cal_eq_subs}")
    if cal_eq_subs > 0:
        print(f"Mean E reduction: {np.mean(e_reductions):.1f}%")
        print(f"Mean H change: {np.mean(h_changes):.2f}")

    # --- Figure: meal_dei_distribution_v2.png ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(meal_logDEI, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(np.median(meal_logDEI), color="red", ls="--", lw=1.5, label=f"Median = {np.median(meal_logDEI):.2f}")
    ax.set_xlabel("log(DEI) of meal combination")
    ax.set_ylabel("Density")
    ax.set_title(f"Meal-level DEI distribution ({n_valid:,} combinations)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "meal_dei_distribution_v2.png", dpi=150)
    plt.close()
    print("Saved: meal_dei_distribution_v2.png")

    # --- Figure: like_for_like_comparison_v2.png ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: within-role variance decomposition bar chart
    role_order = ["Side/Snack", "Light Main", "Full Main", "Heavy Main"]
    role_plot = role_df.set_index("meal_role").reindex([r for r in role_order if r in role_df["meal_role"].values])
    axes[0].barh(role_plot.index, role_plot["H_pct_var"], color="#DD8452", label="H contribution")
    axes[0].barh(role_plot.index, role_plot["E_pct_var"], left=role_plot["H_pct_var"], color="#4C72B0", label="E contribution")
    axes[0].set_xlabel("% of Var(log DEI)")
    axes[0].set_title("Within-role variance decomposition")
    axes[0].legend(loc="lower right")
    for idx, row in role_plot.iterrows():
        axes[0].text(row["H_pct_var"] / 2, idx, f'{row["H_pct_var"]:.1f}%', ha="center", va="center", fontsize=9, fontweight="bold")

    # Right: DEI distributions by meal role
    for role_name in role_order:
        grp = dei[dei["meal_role"] == role_name]
        if len(grp) > 0:
            axes[1].hist(grp["log_DEI"], bins=30, alpha=0.5, label=f"{role_name} (n={len(grp)})", density=True)
    axes[1].set_xlabel("log(DEI)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("DEI distribution by meal role")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "like_for_like_comparison_v2.png", dpi=150)
    plt.close()
    print("Saved: like_for_like_comparison_v2.png")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS C: NDI / Nutritional Dimension
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS C: NDI / Nutritional Dimension")
print("=" * 60)

# Filter dishes with valid NDI
dei_ndi = dei[dei["NDI"] > 0].copy()
print(f"Dishes with valid NDI: {len(dei_ndi)}")
print(f"NDI range: [{dei_ndi['NDI'].min():.1f}, {dei_ndi['NDI'].max():.1f}]")

# DEI-N computation
alpha = 0.5
dei_ndi["log_NDI"] = np.log(dei_ndi["NDI"].clip(lower=0.01))
dei_ndi["log_DEIN"] = dei_ndi["log_H"] + alpha * dei_ndi["log_NDI"] - dei_ndi["log_E"]

rho_dein, _ = sp_stats.spearmanr(dei_ndi["log_DEI"], dei_ndi["log_DEIN"])
print(f"Spearman rho(DEI, DEI-N): {rho_dein:.3f}")

# 2D Pareto count
pareto_2d = sum(dei_ndi["is_pareto"])
print(f"2D Pareto-optimal: {pareto_2d}")

# 3D Pareto: (H, 1/E, NDI) — no dish dominated in all 3
H_arr = dei_ndi["H_mean"].values
invE_arr = 1 / dei_ndi["E_composite"].values
NDI_arr = dei_ndi["NDI"].values
n_d = len(dei_ndi)
is_pareto_3d = np.ones(n_d, dtype=bool)
for i in range(n_d):
    for j in range(n_d):
        if i == j:
            continue
        if H_arr[j] >= H_arr[i] and invE_arr[j] >= invE_arr[i] and NDI_arr[j] >= NDI_arr[i]:
            if H_arr[j] > H_arr[i] or invE_arr[j] > invE_arr[i] or NDI_arr[j] > NDI_arr[i]:
                is_pareto_3d[i] = False
                break
n_pareto_3d = is_pareto_3d.sum()
print(f"3D Pareto-optimal (H, 1/E, NDI): {n_pareto_3d}")

# --- Figure: dei_vs_dein_comparison_v2.png ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Left: DEI rank vs DEI-N rank
dei_ndi["rank_DEI"] = dei_ndi["log_DEI"].rank(ascending=False)
dei_ndi["rank_DEIN"] = dei_ndi["log_DEIN"].rank(ascending=False)
axes[0].scatter(dei_ndi["rank_DEI"], dei_ndi["rank_DEIN"], s=5, alpha=0.3, c="#4C72B0")
axes[0].plot([0, len(dei_ndi)], [0, len(dei_ndi)], "r--", lw=1, alpha=0.5)
axes[0].set_xlabel("DEI rank")
axes[0].set_ylabel("DEI-N rank (α=0.5)")
axes[0].set_title(f"DEI vs DEI-N rankings (ρ = {rho_dein:.3f})")
# Right: log(DEI) vs log(DEI-N) scatter
axes[1].scatter(dei_ndi["log_DEI"], dei_ndi["log_DEIN"], s=5, alpha=0.3, c="#4C72B0")
lims = [min(dei_ndi["log_DEI"].min(), dei_ndi["log_DEIN"].min()),
        max(dei_ndi["log_DEI"].max(), dei_ndi["log_DEIN"].max())]
axes[1].plot(lims, lims, "r--", lw=1, alpha=0.5)
# Highlight 3D Pareto dishes
pareto_3d_names = dei_ndi.index[is_pareto_3d]
axes[1].scatter(dei_ndi.loc[pareto_3d_names, "log_DEI"],
                dei_ndi.loc[pareto_3d_names, "log_DEIN"],
                s=30, c="red", zorder=5, label=f"3D Pareto ({n_pareto_3d})")
axes[1].set_xlabel("log(DEI)")
axes[1].set_ylabel("log(DEI-N)")
axes[1].set_title(f"DEI vs DEI-N (n={len(dei_ndi)})")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "dei_vs_dein_comparison_v2.png", dpi=150)
plt.close()
print("Saved: dei_vs_dein_comparison_v2.png")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS D: OLS Regression
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS D: OLS Regression (n=2,563)")
print("=" * 60)

try:
    import statsmodels.api as sm
    # Filter valid E sub-components (E > 0)
    reg_df = dei[(dei["E_carbon"] > 0) & (dei["E_water"] > 0) & (dei["E_energy"] > 0)].copy()
    reg_df["log_E_carbon"] = np.log(reg_df["E_carbon"])
    reg_df["log_E_water"] = np.log(reg_df["E_water"])
    reg_df["log_E_energy"] = np.log(reg_df["E_energy"])

    X = reg_df[["log_E_carbon", "log_E_water", "log_E_energy"]]
    X = sm.add_constant(X)
    y = reg_df["log_DEI"]

    model = sm.OLS(y, X).fit(cov_type="HC3")
    print(model.summary())

    with open(TABLES_DIR / "dei_regression_v2.txt", "w") as f:
        f.write(str(model.summary()))
    print(f"\nSaved: dei_regression_v2.txt")
    print(f"R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}, n = {len(reg_df)}")
except ImportError:
    print("statsmodels not installed, skipping OLS regression")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS E: Refinement Cost Curves
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS E: Refinement Cost Curves")
print("=" * 60)

# Family keyword matching
FAMILY_KEYWORDS = {
    "Curry": ["curry", "masala", "korma", "tikka", "rendang", "vindaloo", "rogan_josh", "aloo_gobi", "palak_paneer", "butter_chicken"],
    "Noodle/Pasta": ["noodle", "pasta", "ramen", "pad_thai", "lo_mein", "spaghetti", "udon", "soba", "lasagna", "pho", "laksa", "carbonara", "fettuccine", "linguine", "penne", "macaroni"],
    "Rice Dish": ["rice", "biryani", "risotto", "paella", "plov", "pilaf", "pilau", "nasi", "congee", "jollof"],
    "Chicken": ["chicken", "ayam", "pollo", "poulet"],
    "Beef": ["beef", "steak", "brisket", "pot_roast", "bourguignon", "rendang"],
    "Salad/Cold": ["salad", "coleslaw", "ceviche", "tabouleh", "fattoush", "rojak"],
    "Dessert": ["brownie", "cheesecake", "gelato", "ice_cream", "panna_cotta", "tiramisu", "baklava", "cake", "pudding", "mousse", "tart", "pie"],
    "Seafood": ["fish", "shrimp", "crab", "lobster", "squid", "sushi", "sashimi", "ceviche"],
    "Soup": ["soup", "chowder", "broth", "stew", "bisque", "pozole", "pho", "tom_yum", "borscht"],
    "Dumpling": ["dumpling", "momo", "gyoza", "pierogi", "empanada", "samosa", "wonton", "ravioli", "mandu"],
    "Bread": ["bread", "naan", "pita", "focaccia", "baguette", "flatbread", "roti", "chapati", "tortilla"],
    "Pork": ["pork", "bacon", "carnitas", "pulled_pork", "tonkatsu"],
    "Kebab/Grill": ["kebab", "kebap", "satay", "yakitori", "grilled", "bbq", "barbecue", "skewer"],
    "Egg": ["egg", "omelette", "frittata", "shakshuka", "quiche"],
    "Vegetable": ["vegetable", "broccoli", "spinach", "eggplant", "stir_fry"],
}

family_assignments = {}
for dish_id in dei.index:
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(kw in dish_id.lower() for kw in keywords):
            if dish_id not in family_assignments:
                family_assignments[dish_id] = family
            break

# Fit refinement curves per family
family_rows = []
for family_name, keywords in FAMILY_KEYWORDS.items():
    members = [d for d, f in family_assignments.items() if f == family_name and d in dei.index]
    if len(members) < 3:
        continue
    grp = dei.loc[members]
    base_dish = grp["E_composite"].idxmin()
    E_base = grp.loc[base_dish, "E_composite"]
    H_base = grp.loc[base_dish, "H_mean"]
    # Avoid log(0)
    grp_valid = grp[grp["E_composite"] > E_base * 0.01]
    if len(grp_valid) < 3:
        continue
    x = np.log(grp_valid["E_composite"] / E_base)
    y = grp_valid["H_mean"] - H_base
    if x.std() == 0:
        continue
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
    family_rows.append({
        "family": family_name,
        "n_members": len(grp_valid),
        "alpha": slope,
        "alpha_se": std_err,
        "r_squared": r_value**2,
        "p_value": p_value,
        "base_dish": base_dish,
    })

family_df = pd.DataFrame(family_rows).sort_values("alpha")
family_df.to_csv(TABLES_DIR / "refinement_curves_v2.csv", index=False)
print(family_df.to_string(index=False))
n_families = len(family_df)
total_members = family_df["n_members"].sum()
mean_alpha = family_df["alpha"].mean()
n_negative = (family_df["alpha"] <= 0).sum()
print(f"\n{n_families} families, {total_members} dishes")
print(f"Mean alpha: {mean_alpha:.2f}")
print(f"Negative/zero alpha: {n_negative}/{n_families}")

# Global pooled fit
all_x, all_y = [], []
for family_name, keywords in FAMILY_KEYWORDS.items():
    members = [d for d, f in family_assignments.items() if f == family_name and d in dei.index]
    if len(members) < 3:
        continue
    grp = dei.loc[members]
    base_dish = grp["E_composite"].idxmin()
    E_base = grp.loc[base_dish, "E_composite"]
    H_base = grp.loc[base_dish, "H_mean"]
    grp_valid = grp[grp["E_composite"] > E_base * 0.01]
    all_x.extend(np.log(grp_valid["E_composite"] / E_base))
    all_y.extend(grp_valid["H_mean"] - H_base)

if len(all_x) > 3:
    all_x, all_y = np.array(all_x), np.array(all_y)
    slope_g, intercept_g, r_g, p_g, se_g = sp_stats.linregress(all_x, all_y)
    ci95_lo = slope_g - 1.96 * se_g
    ci95_hi = slope_g + 1.96 * se_g
    print(f"\nGlobal alpha: {slope_g:.2f} (95% CI [{ci95_lo:.2f}, {ci95_hi:.2f}]), R²={r_g**2:.2f}")

# --- Figure: refinement_cost_curves_v2.png ---
n_fam_plot = len(family_df)
ncols = min(4, n_fam_plot)
nrows = int(np.ceil(n_fam_plot / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
if n_fam_plot == 1:
    axes = np.array([axes])
axes_flat = axes.flatten()
for idx, (_, row) in enumerate(family_df.iterrows()):
    ax = axes_flat[idx]
    fam_name = row["family"]
    members = [d for d, f in family_assignments.items() if f == fam_name and d in dei.index]
    grp = dei.loc[members]
    base_dish = grp["E_composite"].idxmin()
    E_base = grp.loc[base_dish, "E_composite"]
    H_base = grp.loc[base_dish, "H_mean"]
    grp_valid = grp[grp["E_composite"] > E_base * 0.01]
    x_plot = np.log(grp_valid["E_composite"] / E_base)
    y_plot = grp_valid["H_mean"] - H_base
    ax.scatter(x_plot, y_plot, s=15, alpha=0.6, c="#4C72B0")
    if x_plot.std() > 0:
        x_line = np.linspace(x_plot.min(), x_plot.max(), 50)
        slope_i, intercept_i, _, _, _ = sp_stats.linregress(x_plot, y_plot)
        ax.plot(x_line, intercept_i + slope_i * x_line, "r-", lw=1.5)
    ax.set_title(f"{fam_name} (n={len(grp_valid)}, α={row['alpha']:.2f})", fontsize=9)
    ax.set_xlabel("log(E/E_base)", fontsize=8)
    ax.set_ylabel("H − H_base", fontsize=8)
    ax.tick_params(labelsize=7)
for idx in range(n_fam_plot, len(axes_flat)):
    axes_flat[idx].set_visible(False)
plt.suptitle(f"Refinement cost curves: {n_fam_plot} families, {int(family_df['n_members'].sum())} dishes", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "refinement_cost_curves_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: refinement_cost_curves_v2.png")

# --- Figure: refinement_global_fit_v2.png ---
if len(all_x) > 3:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(all_x, all_y, s=8, alpha=0.3, c="#4C72B0", label="Individual dishes")
    x_line = np.linspace(all_x.min(), all_x.max(), 100)
    ax.plot(x_line, intercept_g + slope_g * x_line, "r-", lw=2,
            label=f"α = {slope_g:.2f} (95% CI [{ci95_lo:.2f}, {ci95_hi:.2f}])")
    ax.axhline(0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("log(E / E_base)")
    ax.set_ylabel("H − H_base")
    ax.set_title(f"Global refinement cost curve ({int(family_df['n_members'].sum())} dishes, {n_fam_plot} families)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "refinement_global_fit_v2.png", dpi=150)
    plt.close()
    print("Saved: refinement_global_fit_v2.png")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS F: Monte Carlo Rank Uncertainty
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS F: Monte Carlo Rank Uncertainty")
print("=" * 60)

n_sims = 1000
n_dishes = len(dei)
logH = dei["log_H"].values
logE = dei["log_E"].values
E_vals = dei["E_composite"].values

# E perturbation: CV category-specific (0.25-0.60)
rng = np.random.default_rng(42)
ranks_matrix = np.zeros((n_sims, n_dishes), dtype=int)

for sim in range(n_sims):
    # Perturb E by ingredient-category CV (average ~0.35)
    e_noise = rng.normal(1.0, 0.35, n_dishes)
    e_noise = np.clip(e_noise, 0.1, 3.0)
    E_perturbed = E_vals * e_noise
    logDEI_sim = logH - np.log(E_perturbed)
    ranks_matrix[sim] = sp_stats.rankdata(-logDEI_sim).astype(int)

# Compute rank CIs
median_rank = np.median(ranks_matrix, axis=0).astype(int)
rank_5 = np.percentile(ranks_matrix, 5, axis=0).astype(int)
rank_95 = np.percentile(ranks_matrix, 95, axis=0).astype(int)
rank_iqr = (np.percentile(ranks_matrix, 75, axis=0) - np.percentile(ranks_matrix, 25, axis=0)).astype(int)

mc_df = pd.DataFrame({
    "dish_id": dei.index,
    "point_rank": dei["rank_DEI"].values,
    "rank_5pct": rank_5,
    "rank_95pct": rank_95,
    "rank_iqr": rank_iqr,
})
mc_df = mc_df.sort_values("point_rank")
mc_df.to_csv(TABLES_DIR / "mc_rank_stability_v2.csv", index=False)

# Print top/bottom 10
print("\nTop 10:")
for _, row in mc_df.head(10).iterrows():
    print(f"  {row['dish_id']:25s} rank={row['point_rank']:4d} 90%CI=[{row['rank_5pct']}, {row['rank_95pct']}] IQR={row['rank_iqr']}")
print("\nBottom 10:")
for _, row in mc_df.tail(10).iterrows():
    print(f"  {row['dish_id']:25s} rank={row['point_rank']:4d} 90%CI=[{row['rank_5pct']}, {row['rank_95pct']}] IQR={row['rank_iqr']}")

median_ci_width = np.median(rank_95 - rank_5)
print(f"\nMedian 90% CI width: {median_ci_width:.0f} positions (out of {n_dishes})")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS G: Survivorship Bias Bounds
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS G: Survivorship Bias Bounds")
print("=" * 60)

H_mean_global = dei["H_mean"].mean()
E_observed = dei["E_composite"].values

K_values = [100, 250, 500, 1000, 2000, 2563]
Delta_values = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
n_reps = 200

surv_rows = []
for K in K_values:
    for Delta in Delta_values:
        h_contribs = []
        for _ in range(n_reps):
            # Ghost H: uniform from [H_mean - Delta, H_mean]
            ghost_H = rng.uniform(H_mean_global - Delta, H_mean_global, K)
            ghost_H = np.clip(ghost_H, 1.0, 10.0)
            ghost_E = rng.choice(E_observed, K, replace=True)
            all_H = np.concatenate([dei["H_mean"].values, ghost_H])
            all_E = np.concatenate([E_observed, ghost_E])
            all_logH = np.log(all_H)
            all_logE = np.log(all_E)
            var_lH = np.var(all_logH)
            var_lE = np.var(all_logE)
            cov_lHE = np.cov(all_logH, all_logE)[0, 1]
            sh_H = var_lH - cov_lHE
            sh_E = var_lE - cov_lHE
            total = sh_H + sh_E
            h_contribs.append(sh_H / total * 100 if total > 0 else 0)
        surv_rows.append({
            "K": K, "Delta": Delta,
            "H_contribution_mean": np.mean(h_contribs),
            "H_contribution_std": np.std(h_contribs),
        })

surv_df = pd.DataFrame(surv_rows)
surv_df.to_csv(TABLES_DIR / "survivorship_bounds_v2.csv", index=False)

# Print key scenarios
print("Ghost-dish bounds (H contribution %):")
pivot = surv_df.pivot(index="K", columns="Delta", values="H_contribution_mean")
print(pivot.round(1).to_string())

# Heatmap figure
fig, ax = plt.subplots(figsize=(8, 5))
pivot_plot = surv_df.pivot(index="K", columns="Delta", values="H_contribution_mean")
im = ax.imshow(pivot_plot.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(Delta_values)))
ax.set_xticklabels(Delta_values)
ax.set_yticks(range(len(K_values)))
ax.set_yticklabels(K_values)
ax.set_xlabel("Hedonic deficit Δ")
ax.set_ylabel("Number of ghost dishes K")
ax.set_title("H contribution (%) under survivorship bias scenarios")
for i in range(len(K_values)):
    for j in range(len(Delta_values)):
        ax.text(j, i, f"{pivot_plot.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="H contribution (%)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "survivorship_heatmap_v2.png", dpi=150)
plt.close()
print(f"Saved: survivorship_heatmap_v2.png")

# ══════════════════════════════════════════════════════════════════
# SUMMARY: All key numbers for paper update
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PAPER UPDATE SUMMARY")
print("=" * 60)

print(f"""
=== Section 2.3: Within-Category ===
  Classified {len(dei)} dishes into {len(cat_df)} categories
  Mean within-category H contribution: {mean_h_contrib:.1f}%
  Viable nutrition-constrained substitutions: {len(subs_df)}
  Mean E reduction: {subs_df['E_reduction_pct'].mean():.1f}% (if subs exist)

=== Section 2.7: Meal-Level ===
  Meal roles: {dei['meal_role'].value_counts().to_dict()}
  Valid meal combinations: {n_valid}
  Meal-level H contribution: {meal_h_pct:.1f}%
  Meal-level DEI range: {dei_range:.0f}x
  Calorie-equivalent substitutions: {cal_eq_subs}
  Mean E reduction (cal-eq): {np.mean(e_reductions):.1f}%

=== Section 2.8: NDI ===
  NDI range: [{dei_ndi['NDI'].min():.1f}, {dei_ndi['NDI'].max():.1f}] across {len(dei_ndi)} dishes
  rho(DEI, DEI-N): {rho_dein:.3f}
  3D Pareto-optimal: {n_pareto_3d}

=== Section 2.9: Refinement Curves ===
  {n_families} families, {total_members} dishes
  Global alpha: {slope_g:.2f} (95% CI [{ci95_lo:.2f}, {ci95_hi:.2f}])
  Negative/zero alpha: {n_negative}/{n_families}

=== Extended Data Table 1: OLS ===
  R² = {model.rsquared:.3f}, n = {len(reg_df)}

=== Extended Data Table 3: MC Rank ===
  Median 90% CI width: {median_ci_width:.0f} positions

=== Section 3.8: Survivorship ===
  K={n_dishes}, Delta=2.0: H contrib = {surv_df[(surv_df['K']==2563) & (surv_df['Delta']==2.0)]['H_contribution_mean'].values[0]:.1f}%
""")

print("DONE.")
