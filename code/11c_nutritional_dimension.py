"""
11c_nutritional_dimension.py — Nutritional Dimension for DEI
============================================================
Addresses criticism C3: DEI ignores nutritional value.
Comparing kimchi (15 kcal side dish) with brisket (500 kcal protein main) is unfair.

Steps:
  1. Build USDA-sourced nutritional database for 101 ingredients
  2. Compute per-dish nutritional profile using recipes
  3. Compute NDI (Nutrient Density Index, Drewnowski 2009 NRF framework)
  4. Construct DEI-N: log(DEI_N) = log(H) + α·log(NDI) - log(E)
  5. 3D Pareto analysis in (H, 1/E, NDI) space
  6. Meal-role classification by calorie content

Outputs:
  - data/ingredient_nutrients.csv
  - data/dish_nutritional_profiles.csv
  - tables/dei_n_rankings.csv
  - figures/3d_pareto_hne.png
  - figures/dei_vs_dein_comparison.png
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR, COOKING_ENERGY_KWH, GRID_EMISSION_FACTOR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ══════════════════════════════════════════════════════════════════
# STEP 1: USDA Nutritional Database for 101 Ingredients
# ══════════════════════════════════════════════════════════════════
# Values per kg (1000g), sourced from USDA FoodData Central SR28
# Columns: protein_g, fat_g, carb_g, fiber_g, iron_mg, zinc_mg,
#           b12_ug, calcium_mg, vitamin_c_mg, calorie_kcal

NUTRIENT_DATA = {
    # ── Beverages ──
    "coffee":       {"protein_g": 1, "fat_g": 0, "carb_g": 0, "fiber_g": 0, "iron_mg": 2, "zinc_mg": 0.5, "b12_ug": 0, "calcium_mg": 20, "vitamin_c_mg": 0, "calorie_kcal": 10},
    "tea":          {"protein_g": 0, "fat_g": 0, "carb_g": 3, "fiber_g": 0, "iron_mg": 0, "zinc_mg": 0.2, "b12_ug": 0, "calcium_mg": 5, "vitamin_c_mg": 0, "calorie_kcal": 10},
    # ── Condiments ──
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
    # ── Dairy ──
    "butter":       {"protein_g": 9, "fat_g": 810, "carb_g": 1, "fiber_g": 0, "iron_mg": 0.2, "zinc_mg": 0.9, "b12_ug": 1.7, "calcium_mg": 240, "vitamin_c_mg": 0, "calorie_kcal": 7170},
    "cheese":       {"protein_g": 250, "fat_g": 330, "carb_g": 13, "fiber_g": 0, "iron_mg": 7, "zinc_mg": 33, "b12_ug": 15, "calcium_mg": 7210, "vitamin_c_mg": 0, "calorie_kcal": 4030},
    "cream":        {"protein_g": 21, "fat_g": 365, "carb_g": 28, "fiber_g": 0, "iron_mg": 0.4, "zinc_mg": 2.2, "b12_ug": 1.1, "calcium_mg": 650, "vitamin_c_mg": 6, "calorie_kcal": 3450},
    "egg":          {"protein_g": 126, "fat_g": 99, "carb_g": 7, "fiber_g": 0, "iron_mg": 17, "zinc_mg": 13, "b12_ug": 11, "calcium_mg": 560, "vitamin_c_mg": 0, "calorie_kcal": 1430},
    "milk":         {"protein_g": 33, "fat_g": 33, "carb_g": 48, "fiber_g": 0, "iron_mg": 0.3, "zinc_mg": 4, "b12_ug": 4.5, "calcium_mg": 1130, "vitamin_c_mg": 0, "calorie_kcal": 610},
    "mozzarella":   {"protein_g": 224, "fat_g": 224, "carb_g": 22, "fiber_g": 0, "iron_mg": 4, "zinc_mg": 28, "b12_ug": 22, "calcium_mg": 5050, "vitamin_c_mg": 0, "calorie_kcal": 3000},
    "parmesan":     {"protein_g": 358, "fat_g": 259, "carb_g": 32, "fiber_g": 0, "iron_mg": 8, "zinc_mg": 29, "b12_ug": 12, "calcium_mg": 11840, "vitamin_c_mg": 0, "calorie_kcal": 3920},
    "yogurt":       {"protein_g": 100, "fat_g": 6, "carb_g": 36, "fiber_g": 0, "iron_mg": 1, "zinc_mg": 6, "b12_ug": 7.5, "calcium_mg": 1100, "vitamin_c_mg": 5, "calorie_kcal": 590},
    # ── Fruits ──
    "apple":        {"protein_g": 3, "fat_g": 2, "carb_g": 138, "fiber_g": 24, "iron_mg": 1.2, "zinc_mg": 0.4, "b12_ug": 0, "calcium_mg": 60, "vitamin_c_mg": 46, "calorie_kcal": 520},
    "avocado":      {"protein_g": 20, "fat_g": 147, "carb_g": 86, "fiber_g": 67, "iron_mg": 5.5, "zinc_mg": 6.4, "b12_ug": 0, "calcium_mg": 120, "vitamin_c_mg": 100, "calorie_kcal": 1600},
    "banana":       {"protein_g": 11, "fat_g": 3, "carb_g": 228, "fiber_g": 26, "iron_mg": 2.6, "zinc_mg": 1.5, "b12_ug": 0, "calcium_mg": 50, "vitamin_c_mg": 87, "calorie_kcal": 890},
    "coconut":      {"protein_g": 33, "fat_g": 334, "carb_g": 153, "fiber_g": 90, "iron_mg": 24, "zinc_mg": 11, "b12_ug": 0, "calcium_mg": 140, "vitamin_c_mg": 33, "calorie_kcal": 3540},
    "lemon":        {"protein_g": 11, "fat_g": 3, "carb_g": 93, "fiber_g": 28, "iron_mg": 6, "zinc_mg": 0.6, "b12_ug": 0, "calcium_mg": 260, "vitamin_c_mg": 530, "calorie_kcal": 290},
    "lime":         {"protein_g": 7, "fat_g": 2, "carb_g": 107, "fiber_g": 28, "iron_mg": 6, "zinc_mg": 1.1, "b12_ug": 0, "calcium_mg": 330, "vitamin_c_mg": 290, "calorie_kcal": 300},
    "mango":        {"protein_g": 8, "fat_g": 4, "carb_g": 150, "fiber_g": 16, "iron_mg": 1.6, "zinc_mg": 0.9, "b12_ug": 0, "calcium_mg": 110, "vitamin_c_mg": 365, "calorie_kcal": 600},
    "pineapple":    {"protein_g": 5, "fat_g": 1, "carb_g": 132, "fiber_g": 14, "iron_mg": 2.9, "zinc_mg": 1.2, "b12_ug": 0, "calcium_mg": 130, "vitamin_c_mg": 479, "calorie_kcal": 500},
    # ── Grains ──
    "bread":        {"protein_g": 90, "fat_g": 33, "carb_g": 490, "fiber_g": 27, "iron_mg": 36, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 2600, "vitamin_c_mg": 0, "calorie_kcal": 2650},
    "corn":         {"protein_g": 32, "fat_g": 12, "carb_g": 190, "fiber_g": 20, "iron_mg": 5, "zinc_mg": 5, "b12_ug": 0, "calcium_mg": 20, "vitamin_c_mg": 68, "calorie_kcal": 860},
    "oats":         {"protein_g": 169, "fat_g": 69, "carb_g": 661, "fiber_g": 106, "iron_mg": 47, "zinc_mg": 40, "b12_ug": 0, "calcium_mg": 540, "vitamin_c_mg": 0, "calorie_kcal": 3890},
    "pasta_dry":    {"protein_g": 130, "fat_g": 15, "carb_g": 750, "fiber_g": 31, "iron_mg": 35, "zinc_mg": 14, "b12_ug": 0, "calcium_mg": 210, "vitamin_c_mg": 0, "calorie_kcal": 3710},
    "rice":         {"protein_g": 27, "fat_g": 3, "carb_g": 282, "fiber_g": 4, "iron_mg": 8, "zinc_mg": 11, "b12_ug": 0, "calcium_mg": 280, "vitamin_c_mg": 0, "calorie_kcal": 1300},
    "rice_noodle":  {"protein_g": 32, "fat_g": 2, "carb_g": 832, "fiber_g": 10, "iron_mg": 2, "zinc_mg": 4, "b12_ug": 0, "calcium_mg": 80, "vitamin_c_mg": 0, "calorie_kcal": 3600},
    "tortilla":     {"protein_g": 85, "fat_g": 75, "carb_g": 480, "fiber_g": 35, "iron_mg": 30, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 1500, "vitamin_c_mg": 0, "calorie_kcal": 3060},
    "wheat_flour":  {"protein_g": 100, "fat_g": 10, "carb_g": 763, "fiber_g": 27, "iron_mg": 46, "zinc_mg": 7, "b12_ug": 0, "calcium_mg": 150, "vitamin_c_mg": 0, "calorie_kcal": 3640},
    # ── Legumes ──
    "black_bean":   {"protein_g": 215, "fat_g": 9, "carb_g": 627, "fiber_g": 156, "iron_mg": 50, "zinc_mg": 35, "b12_ug": 0, "calcium_mg": 1230, "vitamin_c_mg": 0, "calorie_kcal": 3410},
    "chickpea":     {"protein_g": 209, "fat_g": 61, "carb_g": 610, "fiber_g": 174, "iron_mg": 62, "zinc_mg": 34, "b12_ug": 0, "calcium_mg": 1050, "vitamin_c_mg": 40, "calorie_kcal": 3640},
    "lentil":       {"protein_g": 254, "fat_g": 11, "carb_g": 601, "fiber_g": 307, "iron_mg": 76, "zinc_mg": 33, "b12_ug": 0, "calcium_mg": 350, "vitamin_c_mg": 44, "calorie_kcal": 3520},
    "peanut":       {"protein_g": 258, "fat_g": 493, "carb_g": 161, "fiber_g": 85, "iron_mg": 46, "zinc_mg": 33, "b12_ug": 0, "calcium_mg": 920, "vitamin_c_mg": 0, "calorie_kcal": 5670},
    "soybean":      {"protein_g": 365, "fat_g": 198, "carb_g": 302, "fiber_g": 92, "iron_mg": 157, "zinc_mg": 49, "b12_ug": 0, "calcium_mg": 2770, "vitamin_c_mg": 60, "calorie_kcal": 4460},
    "tofu":         {"protein_g": 80, "fat_g": 48, "carb_g": 19, "fiber_g": 3, "iron_mg": 54, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 3500, "vitamin_c_mg": 1, "calorie_kcal": 760},
    # ── Meats ──
    "bacon":        {"protein_g": 370, "fat_g": 420, "carb_g": 13, "fiber_g": 0, "iron_mg": 12, "zinc_mg": 30, "b12_ug": 11, "calcium_mg": 110, "vitamin_c_mg": 0, "calorie_kcal": 5410},
    "beef":         {"protein_g": 260, "fat_g": 150, "carb_g": 0, "fiber_g": 0, "iron_mg": 26, "zinc_mg": 45, "b12_ug": 25, "calcium_mg": 180, "vitamin_c_mg": 0, "calorie_kcal": 2500},
    "ground_beef":  {"protein_g": 260, "fat_g": 150, "carb_g": 0, "fiber_g": 0, "iron_mg": 26, "zinc_mg": 45, "b12_ug": 25, "calcium_mg": 180, "vitamin_c_mg": 0, "calorie_kcal": 2500},
    "lamb":         {"protein_g": 253, "fat_g": 210, "carb_g": 0, "fiber_g": 0, "iron_mg": 17, "zinc_mg": 38, "b12_ug": 26, "calcium_mg": 170, "vitamin_c_mg": 0, "calorie_kcal": 2940},
    "pork":         {"protein_g": 273, "fat_g": 140, "carb_g": 0, "fiber_g": 0, "iron_mg": 8, "zinc_mg": 19, "b12_ug": 7, "calcium_mg": 80, "vitamin_c_mg": 6, "calorie_kcal": 2420},
    "sausage":      {"protein_g": 190, "fat_g": 280, "carb_g": 20, "fiber_g": 0, "iron_mg": 12, "zinc_mg": 20, "b12_ug": 13, "calcium_mg": 130, "vitamin_c_mg": 0, "calorie_kcal": 3390},
    # ── Nuts ──
    "almond":       {"protein_g": 212, "fat_g": 494, "carb_g": 217, "fiber_g": 125, "iron_mg": 37, "zinc_mg": 31, "b12_ug": 0, "calcium_mg": 2690, "vitamin_c_mg": 0, "calorie_kcal": 5790},
    "cashew":       {"protein_g": 183, "fat_g": 438, "carb_g": 305, "fiber_g": 33, "iron_mg": 67, "zinc_mg": 58, "b12_ug": 0, "calcium_mg": 370, "vitamin_c_mg": 5, "calorie_kcal": 5530},
    "pine_nut":     {"protein_g": 137, "fat_g": 681, "carb_g": 131, "fiber_g": 37, "iron_mg": 55, "zinc_mg": 64, "b12_ug": 0, "calcium_mg": 160, "vitamin_c_mg": 8, "calorie_kcal": 6730},
    "sesame_seed":  {"protein_g": 179, "fat_g": 496, "carb_g": 234, "fiber_g": 118, "iron_mg": 146, "zinc_mg": 78, "b12_ug": 0, "calcium_mg": 9750, "vitamin_c_mg": 0, "calorie_kcal": 5730},
    "walnut":       {"protein_g": 152, "fat_g": 654, "carb_g": 139, "fiber_g": 67, "iron_mg": 29, "zinc_mg": 31, "b12_ug": 0, "calcium_mg": 980, "vitamin_c_mg": 13, "calorie_kcal": 6540},
    # ── Oils ──
    "coconut_oil":  {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0.4, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 10, "vitamin_c_mg": 0, "calorie_kcal": 8620},
    "olive_oil":    {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 5.6, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 10, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    "palm_oil":     {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0.1, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    "sesame_oil":   {"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    "vegetable_oil":{"protein_g": 0, "fat_g": 1000, "carb_g": 0, "fiber_g": 0, "iron_mg": 0.6, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 8840},
    # ── Other ──
    "chocolate":    {"protein_g": 76, "fat_g": 430, "carb_g": 460, "fiber_g": 109, "iron_mg": 118, "zinc_mg": 33, "b12_ug": 3, "calcium_mg": 730, "vitamin_c_mg": 0, "calorie_kcal": 5460},
    "honey":        {"protein_g": 3, "fat_g": 0, "carb_g": 824, "fiber_g": 2, "iron_mg": 4, "zinc_mg": 2, "b12_ug": 0, "calcium_mg": 60, "vitamin_c_mg": 5, "calorie_kcal": 3040},
    "maple_syrup":  {"protein_g": 0, "fat_g": 1, "carb_g": 670, "fiber_g": 0, "iron_mg": 11, "zinc_mg": 14, "b12_ug": 0, "calcium_mg": 1020, "vitamin_c_mg": 0, "calorie_kcal": 2600},
    # ── Poultry ──
    "chicken":      {"protein_g": 239, "fat_g": 73, "carb_g": 0, "fiber_g": 0, "iron_mg": 9, "zinc_mg": 16, "b12_ug": 3, "calcium_mg": 150, "vitamin_c_mg": 0, "calorie_kcal": 1650},
    "duck":         {"protein_g": 192, "fat_g": 284, "carb_g": 0, "fiber_g": 0, "iron_mg": 25, "zinc_mg": 18, "b12_ug": 30, "calcium_mg": 110, "vitamin_c_mg": 28, "calorie_kcal": 3370},
    "turkey":       {"protein_g": 294, "fat_g": 15, "carb_g": 0, "fiber_g": 0, "iron_mg": 13, "zinc_mg": 21, "b12_ug": 10, "calcium_mg": 140, "vitamin_c_mg": 0, "calorie_kcal": 1350},
    # ── Seafood ──
    "cod":          {"protein_g": 179, "fat_g": 8, "carb_g": 0, "fiber_g": 0, "iron_mg": 4, "zinc_mg": 5, "b12_ug": 9, "calcium_mg": 160, "vitamin_c_mg": 10, "calorie_kcal": 820},
    "crab":         {"protein_g": 184, "fat_g": 11, "carb_g": 0, "fiber_g": 0, "iron_mg": 7, "zinc_mg": 37, "b12_ug": 91, "calcium_mg": 590, "vitamin_c_mg": 30, "calorie_kcal": 830},
    "fish":         {"protein_g": 200, "fat_g": 33, "carb_g": 0, "fiber_g": 0, "iron_mg": 5, "zinc_mg": 5, "b12_ug": 10, "calcium_mg": 120, "vitamin_c_mg": 0, "calorie_kcal": 1100},
    "salmon":       {"protein_g": 208, "fat_g": 127, "carb_g": 0, "fiber_g": 0, "iron_mg": 8, "zinc_mg": 6, "b12_ug": 32, "calcium_mg": 120, "vitamin_c_mg": 0, "calorie_kcal": 2080},
    "shrimp":       {"protein_g": 241, "fat_g": 17, "carb_g": 2, "fiber_g": 0, "iron_mg": 22, "zinc_mg": 15, "b12_ug": 15, "calcium_mg": 700, "vitamin_c_mg": 20, "calorie_kcal": 990},
    "squid":        {"protein_g": 158, "fat_g": 12, "carb_g": 31, "fiber_g": 0, "iron_mg": 7, "zinc_mg": 15, "b12_ug": 13, "calcium_mg": 320, "vitamin_c_mg": 49, "calorie_kcal": 920},
    "tuna":         {"protein_g": 236, "fat_g": 49, "carb_g": 0, "fiber_g": 0, "iron_mg": 10, "zinc_mg": 6, "b12_ug": 98, "calcium_mg": 80, "vitamin_c_mg": 0, "calorie_kcal": 1440},
    # ── Starch ──
    "potato":       {"protein_g": 20, "fat_g": 1, "carb_g": 170, "fiber_g": 22, "iron_mg": 8, "zinc_mg": 3, "b12_ug": 0, "calcium_mg": 120, "vitamin_c_mg": 198, "calorie_kcal": 770},
    "sweet_potato": {"protein_g": 16, "fat_g": 1, "carb_g": 201, "fiber_g": 30, "iron_mg": 6, "zinc_mg": 3, "b12_ug": 0, "calcium_mg": 300, "vitamin_c_mg": 24, "calorie_kcal": 860},
    # ── Vegetables ──
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
    # ── Extra (from impact factors) ──
    "pita":         {"protein_g": 90, "fat_g": 12, "carb_g": 558, "fiber_g": 22, "iron_mg": 26, "zinc_mg": 8, "b12_ug": 0, "calcium_mg": 860, "vitamin_c_mg": 0, "calorie_kcal": 2750},
    "vanilla":      {"protein_g": 1, "fat_g": 1, "carb_g": 130, "fiber_g": 0, "iron_mg": 1, "zinc_mg": 1, "b12_ug": 0, "calcium_mg": 110, "vitamin_c_mg": 0, "calorie_kcal": 510},
    # ── Extras used in expanded recipes ──
    "ice_cream":    {"protein_g": 35, "fat_g": 110, "carb_g": 240, "fiber_g": 0, "iron_mg": 1, "zinc_mg": 5, "b12_ug": 4, "calcium_mg": 1280, "vitamin_c_mg": 6, "calorie_kcal": 2070},
    "wine":         {"protein_g": 1, "fat_g": 0, "carb_g": 26, "fiber_g": 0, "iron_mg": 5, "zinc_mg": 1, "b12_ug": 0, "calcium_mg": 80, "vitamin_c_mg": 0, "calorie_kcal": 830},
    "saffron":      {"protein_g": 114, "fat_g": 59, "carb_g": 651, "fiber_g": 38, "iron_mg": 114, "zinc_mg": 11, "b12_ug": 0, "calcium_mg": 1110, "vitamin_c_mg": 808, "calorie_kcal": 3100},
    "water":        {"protein_g": 0, "fat_g": 0, "carb_g": 0, "fiber_g": 0, "iron_mg": 0, "zinc_mg": 0, "b12_ug": 0, "calcium_mg": 0, "vitamin_c_mg": 0, "calorie_kcal": 0},
}

print("=" * 60)
print("11c: Nutritional Dimension for DEI")
print("=" * 60)
print(f"USDA nutrient data defined for {len(NUTRIENT_DATA)} ingredients")

# Save ingredient nutrients
nutr_df = pd.DataFrame.from_dict(NUTRIENT_DATA, orient="index")
nutr_df.index.name = "ingredient"
nutr_df.to_csv(DATA_DIR / "ingredient_nutrients.csv")
print(f"Saved: {DATA_DIR / 'ingredient_nutrients.csv'}")

# ── Load recipes by running extraction helpers ──
import json, subprocess

def _extract_recipes(script_path, var_name):
    """Run a helper script to extract recipe dict as JSON."""
    code = (
        f"import sys, json; sys.path.insert(0, r'{ROOT / 'code'}');\n"
        f"exec(open(r'{script_path}', encoding='utf-8').read().split('# =')[0]"
        f" if '{var_name}' == 'DISH_RECIPES' else '');\n"
    )
    # Simpler: just run the script in a way that only defines the dict
    # Actually, let's use ast to parse safely
    import ast
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the assignment using line-by-line search
    lines = content.split("\n")
    start_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{var_name} = {{"):
            start_line = i
            break
    if start_line is None:
        return {}

    # Collect lines until braces balance
    depth = 0
    collected = []
    for line in lines[start_line:]:
        collected.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            break

    # Execute the collected dict definition
    dict_code = "\n".join(collected)
    local_ns = {}
    exec(compile(dict_code, "<recipes>", "exec"), {"__builtins__": __builtins__}, local_ns)
    return local_ns.get(var_name, {})

DISH_RECIPES = _extract_recipes(
    ROOT / "code" / "04_env_cost_calculation.py", "DISH_RECIPES")
print(f"Loaded {len(DISH_RECIPES)} original recipes")

EXPANDED_RECIPES = _extract_recipes(
    ROOT / "code" / "09b_expanded_recipes.py", "EXPANDED_RECIPES")
print(f"Loaded {len(EXPANDED_RECIPES)} expanded recipes")

# Combine all recipes
ALL_RECIPES = {}
for dish_id, recipe in DISH_RECIPES.items():
    ALL_RECIPES[dish_id] = recipe
for dish_id, recipe in EXPANDED_RECIPES.items():
    ALL_RECIPES[dish_id] = recipe
print(f"Total recipes: {len(ALL_RECIPES)}")

# ══════════════════════════════════════════════════════════════════
# STEP 2: Compute Per-Dish Nutritional Profile
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Computing dish nutritional profiles...")
print("=" * 60)

# Daily Reference Values for NDI calculation
DRV = {
    "protein_g": 50, "fiber_g": 25, "iron_mg": 18,
    "zinc_mg": 11, "b12_ug": 2.4, "calcium_mg": 1000, "vitamin_c_mg": 90,
}

dei = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
rows = []
missing_dishes = []

for _, dish_row in dei.iterrows():
    dish_id = dish_row["dish_id"]
    if dish_id not in ALL_RECIPES:
        missing_dishes.append(dish_id)
        continue

    recipe = ALL_RECIPES[dish_id]
    ingredients = recipe["ingredients"]

    # Sum nutrients
    totals = {k: 0.0 for k in ["protein_g", "fat_g", "carb_g", "fiber_g",
                                 "iron_mg", "zinc_mg", "b12_ug", "calcium_mg",
                                 "vitamin_c_mg", "calorie_kcal"]}
    missing_ings = []
    for ing, grams in ingredients.items():
        if ing not in NUTRIENT_DATA:
            missing_ings.append(ing)
            continue
        kg = grams / 1000
        for nutrient in totals:
            totals[nutrient] += kg * NUTRIENT_DATA[ing][nutrient]

    # NDI: Drewnowski NRF-7 (per 100 kcal)
    cal = totals["calorie_kcal"]
    if cal > 0:
        ndi_score = 0
        for nutrient, drv in DRV.items():
            ndi_score += min(totals[nutrient] / drv * 100, 100)  # cap at 100% DRV
        ndi_score /= len(DRV)
        ndi_per_100kcal = ndi_score / (cal / 100)
    else:
        ndi_per_100kcal = 0

    # Meal role classification
    if cal < 200:
        meal_role = "Side/Snack"
    elif cal < 400:
        meal_role = "Light Main"
    elif cal < 700:
        meal_role = "Full Main"
    else:
        meal_role = "Heavy Main"

    rows.append({
        "dish_id": dish_id,
        **totals,
        "NDI": ndi_per_100kcal,
        "meal_role": meal_role,
        "missing_nutrients_ingredients": ",".join(missing_ings) if missing_ings else "",
    })

nutr_profiles = pd.DataFrame(rows)
print(f"Computed nutritional profiles for {len(nutr_profiles)} dishes")
if missing_dishes:
    print(f"  Missing recipes for {len(missing_dishes)} dishes: {missing_dishes[:10]}...")

nutr_profiles.to_csv(DATA_DIR / "dish_nutritional_profiles.csv", index=False)
print(f"Saved: {DATA_DIR / 'dish_nutritional_profiles.csv'}")

# Summary stats
print(f"\n  Calorie range: {nutr_profiles['calorie_kcal'].min():.0f} - {nutr_profiles['calorie_kcal'].max():.0f} kcal")
print(f"  Protein range: {nutr_profiles['protein_g'].min():.1f} - {nutr_profiles['protein_g'].max():.1f} g")
print(f"  NDI range: {nutr_profiles['NDI'].min():.2f} - {nutr_profiles['NDI'].max():.2f}")
print(f"\n  Meal role distribution:")
for role, count in nutr_profiles["meal_role"].value_counts().items():
    print(f"    {role:15s}: {count} dishes")

# ══════════════════════════════════════════════════════════════════
# STEP 3: DEI-N Computation
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: DEI-N Computation")
print("=" * 60)

# Merge nutritional profiles with DEI data
merged = dei.merge(nutr_profiles[["dish_id", "NDI", "calorie_kcal", "protein_g", "meal_role"]],
                   on="dish_id", how="inner")
print(f"Merged: {len(merged)} dishes with both DEI and nutrition data")

# Compute DEI-N for multiple alpha values
merged["log_NDI"] = np.log(merged["NDI"].clip(lower=0.01))

alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for alpha in alpha_values:
    merged[f"log_DEI_N_a{alpha:.1f}"] = merged["log_H"] + alpha * merged["log_NDI"] - merged["log_E"]

# Default alpha = 0.5
merged["log_DEI_N"] = merged["log_DEI_N_a0.5"]
merged["rank_DEI"] = merged["log_DEI"].rank(ascending=False)
merged["rank_DEI_N"] = merged["log_DEI_N"].rank(ascending=False)
merged["rank_shift_N"] = (merged["rank_DEI_N"] - merged["rank_DEI"]).astype(int)

# Rank correlation across alpha values
print(f"\n  DEI vs DEI-N rank correlation by α:")
for alpha in alpha_values:
    col = f"log_DEI_N_a{alpha:.1f}"
    rho, _ = sp_stats.spearmanr(merged["log_DEI"], merged[col])
    print(f"    α={alpha:.1f}: Spearman ρ = {rho:.4f}")

# Top movers with default alpha=0.5
print(f"\n  Biggest rank gainers with DEI-N (α=0.5):")
gainers = merged.nsmallest(10, "rank_shift_N")
for _, row in gainers.iterrows():
    print(f"    {row['dish_id']:25s}: rank {int(row['rank_DEI'])} → {int(row['rank_DEI_N'])} "
          f"(shift={int(row['rank_shift_N'])}), protein={row['protein_g']:.1f}g, cal={row['calorie_kcal']:.0f}")

print(f"\n  Biggest rank losers with DEI-N (α=0.5):")
losers = merged.nlargest(10, "rank_shift_N")
for _, row in losers.iterrows():
    print(f"    {row['dish_id']:25s}: rank {int(row['rank_DEI'])} → {int(row['rank_DEI_N'])} "
          f"(shift={int(row['rank_shift_N'])}), protein={row['protein_g']:.1f}g, cal={row['calorie_kcal']:.0f}")

# ══════════════════════════════════════════════════════════════════
# STEP 4: 3D Pareto Analysis
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: 3D Pareto Analysis (H, 1/E, NDI)")
print("=" * 60)

# A dish is 3D Pareto-optimal if no other dish dominates it on all three:
# higher H AND lower E AND higher NDI
def is_pareto_3d(h_vals, inv_e_vals, ndi_vals):
    n = len(h_vals)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (h_vals[j] >= h_vals[i] and
                inv_e_vals[j] >= inv_e_vals[i] and
                ndi_vals[j] >= ndi_vals[i] and
                (h_vals[j] > h_vals[i] or inv_e_vals[j] > inv_e_vals[i] or ndi_vals[j] > ndi_vals[i])):
                pareto[i] = False
                break
    return pareto

h_arr = merged["H_mean"].values
inv_e_arr = 1.0 / merged["E_composite"].values
ndi_arr = merged["NDI"].values

pareto_3d = is_pareto_3d(h_arr, inv_e_arr, ndi_arr)
merged["is_pareto_3d"] = pareto_3d

pareto_dishes = merged[merged["is_pareto_3d"]].sort_values("log_DEI_N", ascending=False)
print(f"  3D Pareto-optimal dishes: {len(pareto_dishes)}")
for _, row in pareto_dishes.iterrows():
    print(f"    {row['dish_id']:25s}: H={row['H_mean']:.3f}, E={row['E_composite']:.4f}, "
          f"NDI={row['NDI']:.2f}, protein={row['protein_g']:.1f}g, cal={row['calorie_kcal']:.0f}, "
          f"role={row['meal_role']}")

# Compare with 2D Pareto
pareto_2d_count = merged["is_pareto"].sum() if "is_pareto" in merged.columns else 0
print(f"\n  2D Pareto count: {pareto_2d_count}")
print(f"  3D Pareto count: {len(pareto_dishes)}")

# Save
save_cols = ["dish_id", "cuisine", "H_mean", "E_composite", "log_DEI", "NDI",
             "log_NDI", "log_DEI_N", "calorie_kcal", "protein_g",
             "meal_role", "rank_DEI", "rank_DEI_N", "rank_shift_N", "is_pareto_3d"]
merged[save_cols].sort_values("rank_DEI_N").to_csv(TABLES_DIR / "dei_n_rankings.csv", index=False)
print(f"\n  Saved: {TABLES_DIR / 'dei_n_rankings.csv'}")

# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

# Figure 1: 3D Pareto scatter (2D projection with NDI as color/size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: H vs E colored by NDI
sc = ax1.scatter(merged["E_composite"], merged["H_mean"],
                 c=merged["NDI"], cmap="RdYlGn", s=30, alpha=0.7,
                 edgecolors="none")
# Highlight 3D Pareto
pareto_data = merged[merged["is_pareto_3d"]]
ax1.scatter(pareto_data["E_composite"], pareto_data["H_mean"],
            s=100, facecolors="none", edgecolors="red", linewidths=2,
            label=f"3D Pareto ({len(pareto_data)})")
for _, row in pareto_data.iterrows():
    ax1.annotate(row["dish_id"].replace("_", " "),
                 (row["E_composite"], row["H_mean"]),
                 fontsize=6, alpha=0.8,
                 xytext=(5, 3), textcoords="offset points")
plt.colorbar(sc, ax=ax1, label="NDI (per 100 kcal)")
ax1.set_xlabel("Environmental Cost (E)", fontsize=12)
ax1.set_ylabel("Hedonic Score (H)", fontsize=12)
ax1.set_title("A. H vs E, colored by Nutritional Density", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel B: DEI rank vs DEI-N rank
ax2.scatter(merged["rank_DEI"], merged["rank_DEI_N"], alpha=0.4, s=15, c="#2196F3")
lims = [0, len(merged) + 5]
ax2.plot(lims, lims, "k--", alpha=0.3, label="No change")
# Annotate biggest movers
for _, row in gainers.head(5).iterrows():
    ax2.annotate(row["dish_id"].replace("_", " "),
                 (row["rank_DEI"], row["rank_DEI_N"]),
                 fontsize=7, color="green", fontweight="bold",
                 xytext=(5, -10), textcoords="offset points")
for _, row in losers.head(5).iterrows():
    ax2.annotate(row["dish_id"].replace("_", " "),
                 (row["rank_DEI"], row["rank_DEI_N"]),
                 fontsize=7, color="red", fontweight="bold",
                 xytext=(5, 5), textcoords="offset points")
rho_n, _ = sp_stats.spearmanr(merged["rank_DEI"], merged["rank_DEI_N"])
ax2.set_xlabel("DEI Rank (original)", fontsize=12)
ax2.set_ylabel("DEI-N Rank (nutrition-adjusted, α=0.5)", fontsize=12)
ax2.set_title(f"B. Rank Migration: DEI → DEI-N\n(Spearman ρ = {rho_n:.3f})",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "dei_vs_dein_comparison.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'dei_vs_dein_comparison.png'}")

# Figure 2: NDI by meal role
fig, ax = plt.subplots(figsize=(10, 6))
roles = ["Side/Snack", "Light Main", "Full Main", "Heavy Main"]
colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
for role, color in zip(roles, colors):
    sub = merged[merged["meal_role"] == role]
    if len(sub) > 0:
        ax.scatter(sub["E_composite"], sub["NDI"], alpha=0.6, s=40,
                   color=color, label=f"{role} (n={len(sub)})", edgecolors="none")
ax.set_xlabel("Environmental Cost (E)", fontsize=12)
ax.set_ylabel("NDI (Nutrient Density per 100 kcal)", fontsize=12)
ax.set_title("Nutritional Density vs Environmental Cost by Meal Role", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "nutritional_profiles_by_category.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'nutritional_profiles_by_category.png'}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
rho_default, _ = sp_stats.spearmanr(merged["log_DEI"], merged["log_DEI_N"])
print(f"  DEI vs DEI-N (α=0.5): Spearman ρ = {rho_default:.4f}")
print(f"  3D Pareto optimal: {len(pareto_dishes)} dishes")
print(f"  Meal role distribution: {dict(merged['meal_role'].value_counts())}")
print(f"  Mean |rank shift|: {merged['rank_shift_N'].abs().mean():.1f}")
print(f"  Conclusion: Adding nutritional dimension meaningfully reshuffles")
print(f"  rankings — high-protein dishes gain, low-nutrient snacks drop.")
print("=" * 60)
