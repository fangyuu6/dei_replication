"""
09b_expanded_recipes.py — Assign recipes & compute E for expanded dishes
========================================================================
For each of the 179 new viable dishes, defines:
  - Representative recipe (ingredients + grams per serving)
  - Cooking method
  - Computes E (carbon, water, energy) using same pipeline as 04

Then applies finetuned BERT to score H from the saved mention text.

Output:
  - data/expanded_dish_env_costs.csv
  - data/expanded_dish_hedonic.csv
  - data/expanded_dish_DEI.csv (full DEI for new dishes)
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import (DATA_DIR, TABLES_DIR, GRID_EMISSION_FACTOR,
                    COOKING_ENERGY_KWH, E_WEIGHT_SCHEMES)

# ── Load impact factors ──────────────────────────────────────────
impacts = pd.read_csv(DATA_DIR / "ingredient_impact_factors.csv")
impact_lookup = {}
for _, row in impacts.iterrows():
    impact_lookup[row["ingredient"]] = {
        "co2": row["co2_per_kg"],
        "water": row["water_per_kg"],
        "land": row["land_per_kg"],
    }

# ══════════════════════════════════════════════════════════════════
# EXPANDED RECIPES — 179 new dishes
# Each recipe: ingredients dict + cook_method + cuisine
# Sources: representative recipes from AllRecipes, Serious Eats, etc.
# ══════════════════════════════════════════════════════════════════

EXPANDED_RECIPES = {
    # ── AFRICAN ──
    "jollof_rice": {"ingredients": {"rice": 200, "tomato": 100, "onion": 50, "pepper": 30,
                    "vegetable_oil": 20, "chicken": 100, "garlic": 5}, "cook_method": "simmer", "cuisine": "African"},
    "injera": {"ingredients": {"wheat_flour": 150, "water": 200}, "cook_method": "pan_fry", "cuisine": "African"},
    "doro_wot": {"ingredients": {"chicken": 200, "onion": 100, "butter": 30, "egg": 50,
                 "garlic": 10, "ginger": 5, "chili": 10}, "cook_method": "simmer", "cuisine": "African"},
    "kitfo": {"ingredients": {"beef": 200, "butter": 30, "chili": 10}, "cook_method": "raw", "cuisine": "African"},
    "tagine": {"ingredients": {"lamb": 180, "onion": 60, "tomato": 50, "carrot": 40,
               "chickpea": 40, "olive_oil": 15, "cumin": 3}, "cook_method": "braise", "cuisine": "African"},
    "couscous": {"ingredients": {"wheat_flour": 150, "butter": 10, "onion": 30,
                 "carrot": 30, "chickpea": 40}, "cook_method": "steam", "cuisine": "African"},
    "harissa": {"ingredients": {"chili": 50, "garlic": 10, "olive_oil": 20, "cumin": 5,
                "coriander": 5, "tomato": 30}, "cook_method": "raw", "cuisine": "African"},
    "shakshuka": {"ingredients": {"egg": 100, "tomato": 150, "onion": 40, "pepper": 40,
                  "garlic": 5, "olive_oil": 15, "cumin": 3}, "cook_method": "simmer", "cuisine": "African"},
    "fufu": {"ingredients": {"sweet_potato": 250, "water": 100}, "cook_method": "boil", "cuisine": "African"},
    "suya": {"ingredients": {"beef": 200, "peanut": 30, "chili": 10, "onion": 20,
             "vegetable_oil": 10}, "cook_method": "grill", "cuisine": "African"},
    "mafe": {"ingredients": {"chicken": 150, "peanut": 60, "tomato": 80, "onion": 40,
             "sweet_potato": 60, "vegetable_oil": 15}, "cook_method": "simmer", "cuisine": "African"},
    "egusi_soup": {"ingredients": {"peanut": 80, "spinach": 60, "onion": 30, "tomato": 40,
                   "palm_oil": 20, "fish": 80}, "cook_method": "simmer", "cuisine": "African"},
    "plantain": {"ingredients": {"banana": 250, "vegetable_oil": 30}, "cook_method": "deep_fry", "cuisine": "African"},
    "piri_piri": {"ingredients": {"chicken": 250, "chili": 15, "garlic": 10, "lemon": 15,
                  "olive_oil": 20}, "cook_method": "grill", "cuisine": "African"},
    "ful_medames": {"ingredients": {"black_bean": 150, "garlic": 5, "olive_oil": 15,
                    "lemon": 10, "cumin": 3, "tomato": 30}, "cook_method": "simmer", "cuisine": "African"},
    "biltong": {"ingredients": {"beef": 200, "vinegar": 10, "salt": 5, "black_pepper": 3,
                "coriander": 3}, "cook_method": "raw", "cuisine": "African"},

    # ── CARIBBEAN ──
    "jerk_chicken": {"ingredients": {"chicken": 250, "chili": 15, "garlic": 10, "ginger": 5,
                     "onion": 20, "soy_sauce": 10, "sugar": 10, "lime": 10, "vegetable_oil": 10},
                     "cook_method": "grill", "cuisine": "Caribbean"},
    "rice_and_peas": {"ingredients": {"rice": 200, "black_bean": 60, "coconut_milk": 50,
                      "onion": 20, "garlic": 5}, "cook_method": "simmer", "cuisine": "Caribbean"},
    "mofongo": {"ingredients": {"banana": 200, "pork": 60, "garlic": 10, "olive_oil": 20},
                "cook_method": "deep_fry", "cuisine": "Caribbean"},
    "arroz_con_pollo": {"ingredients": {"rice": 200, "chicken": 150, "onion": 30, "pepper": 30,
                        "tomato": 40, "garlic": 5, "olive_oil": 15}, "cook_method": "simmer", "cuisine": "Caribbean"},
    "oxtail_stew": {"ingredients": {"beef": 250, "onion": 40, "carrot": 30, "tomato": 40,
                    "black_bean": 40, "garlic": 5}, "cook_method": "braise", "cuisine": "Caribbean"},
    "roti_caribbean": {"ingredients": {"wheat_flour": 100, "chickpea": 60, "potato": 60,
                       "chicken": 80, "curry_paste": 15, "vegetable_oil": 10}, "cook_method": "pan_fry", "cuisine": "Caribbean"},
    "callaloo": {"ingredients": {"spinach": 150, "coconut_milk": 50, "onion": 30, "garlic": 5,
                 "crab": 50}, "cook_method": "simmer", "cuisine": "Caribbean"},
    "conch_fritters": {"ingredients": {"fish": 100, "wheat_flour": 50, "egg": 30, "onion": 20,
                       "pepper": 20, "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "Caribbean"},
    "escovitch_fish": {"ingredients": {"fish": 200, "onion": 40, "pepper": 40, "carrot": 30,
                       "vinegar": 15, "vegetable_oil": 30}, "cook_method": "deep_fry", "cuisine": "Caribbean"},
    "ackee_saltfish": {"ingredients": {"fish": 120, "tomato": 40, "onion": 30, "pepper": 30,
                       "garlic": 5, "vegetable_oil": 15}, "cook_method": "saute", "cuisine": "Caribbean"},
    "cubano_sandwich": {"ingredients": {"pork": 100, "bread": 80, "cheese": 30, "mustard": 10},
                        "cook_method": "pan_fry", "cuisine": "Caribbean"},
    "pernil": {"ingredients": {"pork": 300, "garlic": 15, "olive_oil": 15, "onion": 20,
               "vinegar": 10}, "cook_method": "roast", "cuisine": "Caribbean"},
    "tostones": {"ingredients": {"banana": 200, "vegetable_oil": 40, "salt": 3, "garlic": 5},
                 "cook_method": "deep_fry", "cuisine": "Caribbean"},
    "coquito": {"ingredients": {"coconut_milk": 100, "milk": 100, "sugar": 30, "egg": 30,
                "vanilla": 3}, "cook_method": "cold", "cuisine": "Caribbean"},
    "pastelitos": {"ingredients": {"wheat_flour": 80, "pork": 60, "cheese": 20, "vegetable_oil": 30},
                   "cook_method": "deep_fry", "cuisine": "Caribbean"},
    "pupusa": {"ingredients": {"corn": 120, "cheese": 40, "pork": 40, "black_bean": 30},
               "cook_method": "pan_fry", "cuisine": "Caribbean"},

    # ── SOUTH AMERICAN ──
    "arepa": {"ingredients": {"corn": 120, "cheese": 30, "butter": 10}, "cook_method": "pan_fry", "cuisine": "South American"},
    "empanada_argentina": {"ingredients": {"wheat_flour": 80, "beef": 80, "onion": 20,
                           "egg": 20, "olive_oil": 10}, "cook_method": "bake", "cuisine": "South American"},
    "lomo_saltado": {"ingredients": {"beef": 150, "onion": 40, "tomato": 40, "potato": 60,
                     "soy_sauce": 10, "rice": 100, "vegetable_oil": 15}, "cook_method": "stir_fry", "cuisine": "South American"},
    "anticucho": {"ingredients": {"beef": 200, "garlic": 5, "cumin": 3, "chili": 5,
                  "vinegar": 10}, "cook_method": "grill", "cuisine": "South American"},
    "causa": {"ingredients": {"potato": 200, "chicken": 60, "lime": 10, "mayonnaise": 20,
              "avocado": 30}, "cook_method": "cold", "cuisine": "South American"},
    "churrasco": {"ingredients": {"beef": 250, "salt": 3, "garlic": 5, "olive_oil": 10},
                  "cook_method": "grill", "cuisine": "South American"},
    "picanha": {"ingredients": {"beef": 250, "salt": 5}, "cook_method": "grill", "cuisine": "South American"},
    "feijoada": {"ingredients": {"black_bean": 150, "pork": 100, "sausage": 50, "rice": 150,
                 "onion": 30, "garlic": 5}, "cook_method": "simmer", "cuisine": "South American"},
    "pao_de_queijo": {"ingredients": {"sweet_potato": 80, "cheese": 50, "egg": 30, "milk": 30,
                      "vegetable_oil": 10}, "cook_method": "bake", "cuisine": "South American"},
    "coxinha": {"ingredients": {"chicken": 100, "wheat_flour": 60, "cream": 30, "egg": 30,
                "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "South American"},
    "chimichurri": {"ingredients": {"coriander": 30, "garlic": 10, "olive_oil": 40,
                    "vinegar": 15, "chili": 5}, "cook_method": "raw", "cuisine": "South American"},
    "asado": {"ingredients": {"beef": 250, "salt": 5}, "cook_method": "grill", "cuisine": "South American"},
    "sancocho": {"ingredients": {"chicken": 150, "potato": 80, "corn": 60, "banana": 40,
                 "onion": 30, "garlic": 5}, "cook_method": "simmer", "cuisine": "South American"},
    "aji_de_gallina": {"ingredients": {"chicken": 150, "bread": 40, "walnut": 20, "cheese": 20,
                       "milk": 30, "chili": 10, "onion": 20}, "cook_method": "simmer", "cuisine": "South American"},

    # ── CENTRAL ASIAN / PERSIAN ──
    "plov": {"ingredients": {"rice": 200, "lamb": 100, "carrot": 60, "onion": 40,
             "vegetable_oil": 20, "garlic": 5, "cumin": 3}, "cook_method": "simmer", "cuisine": "Central Asian"},
    "lagman": {"ingredients": {"pasta_dry": 150, "lamb": 80, "onion": 30, "tomato": 40,
               "pepper": 30, "garlic": 5, "vegetable_oil": 15}, "cook_method": "boil", "cuisine": "Central Asian"},
    "manty": {"ingredients": {"wheat_flour": 80, "lamb": 100, "onion": 40, "butter": 10},
              "cook_method": "steam", "cuisine": "Central Asian"},
    "shashlik": {"ingredients": {"lamb": 200, "onion": 40, "vinegar": 10, "black_pepper": 3},
                 "cook_method": "grill", "cuisine": "Central Asian"},
    "ghormeh_sabzi": {"ingredients": {"lamb": 100, "spinach": 80, "black_bean": 40,
                      "onion": 30, "lime": 10, "vegetable_oil": 15}, "cook_method": "braise", "cuisine": "Persian"},
    "tahdig": {"ingredients": {"rice": 200, "yogurt": 30, "butter": 20, "saffron": 1},
               "cook_method": "pan_fry", "cuisine": "Persian"},
    "fesenjan": {"ingredients": {"chicken": 200, "walnut": 80, "onion": 30, "sugar": 10,
                 "vegetable_oil": 15}, "cook_method": "simmer", "cuisine": "Persian"},
    "koobideh": {"ingredients": {"ground_beef": 200, "onion": 40, "salt": 3, "black_pepper": 3},
                 "cook_method": "grill", "cuisine": "Persian"},

    # ── TURKISH ──
    "doner_kebab": {"ingredients": {"lamb": 150, "bread": 60, "tomato": 30, "onion": 20,
                    "lettuce": 15, "yogurt": 20}, "cook_method": "roast", "cuisine": "Turkish"},
    "lahmacun": {"ingredients": {"wheat_flour": 80, "ground_beef": 80, "tomato": 30,
                 "onion": 20, "pepper": 15, "garlic": 5}, "cook_method": "bake", "cuisine": "Turkish"},
    "borek": {"ingredients": {"wheat_flour": 80, "cheese": 60, "egg": 30, "butter": 20,
              "spinach": 40}, "cook_method": "bake", "cuisine": "Turkish"},
    "iskender_kebab": {"ingredients": {"lamb": 150, "bread": 60, "tomato_sauce": 40,
                       "butter": 20, "yogurt": 30}, "cook_method": "grill", "cuisine": "Turkish"},
    "pide": {"ingredients": {"wheat_flour": 100, "ground_beef": 80, "cheese": 30, "tomato": 20,
             "egg": 20}, "cook_method": "bake", "cuisine": "Turkish"},
    "kofte": {"ingredients": {"ground_beef": 150, "onion": 30, "bread": 20, "cumin": 3,
              "black_pepper": 2}, "cook_method": "grill", "cuisine": "Turkish"},
    "manti_turkish": {"ingredients": {"wheat_flour": 80, "ground_beef": 80, "onion": 20,
                      "yogurt": 30, "butter": 10, "tomato_sauce": 15}, "cook_method": "boil", "cuisine": "Turkish"},
    "gozleme": {"ingredients": {"wheat_flour": 100, "spinach": 50, "cheese": 40, "onion": 20,
                "butter": 10}, "cook_method": "pan_fry", "cuisine": "Turkish"},
    "kunefe": {"ingredients": {"wheat_flour": 60, "cheese": 80, "butter": 30, "sugar": 30},
               "cook_method": "bake", "cuisine": "Turkish"},
    "simit": {"ingredients": {"wheat_flour": 100, "sesame_seed": 15, "sugar": 5},
              "cook_method": "bake", "cuisine": "Turkish"},

    # ── GREEK ──
    "souvlaki": {"ingredients": {"pork": 180, "onion": 20, "pepper": 20, "pita": 60,
                 "olive_oil": 10, "lemon": 10}, "cook_method": "grill", "cuisine": "Greek"},
    "saganaki": {"ingredients": {"cheese": 100, "wheat_flour": 15, "olive_oil": 20, "lemon": 10},
                 "cook_method": "pan_fry", "cuisine": "Greek"},
    "pastitsio": {"ingredients": {"pasta_dry": 150, "ground_beef": 100, "milk": 100,
                  "cheese": 30, "butter": 20, "egg": 30, "tomato_sauce": 40}, "cook_method": "bake", "cuisine": "Greek"},
    "tzatziki": {"ingredients": {"yogurt": 150, "cucumber": 60, "garlic": 5, "olive_oil": 10,
                 "lemon": 5}, "cook_method": "cold", "cuisine": "Greek"},
    "horiatiki": {"ingredients": {"tomato": 80, "cucumber": 60, "onion": 30, "pepper": 30,
                  "olive_oil": 20, "cheese": 40}, "cook_method": "cold", "cuisine": "Greek"},
    "loukoumades": {"ingredients": {"wheat_flour": 80, "sugar": 20, "honey": 20, "egg": 20,
                    "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "Greek"},
    "galaktoboureko": {"ingredients": {"wheat_flour": 60, "milk": 150, "egg": 50, "butter": 30,
                       "sugar": 40}, "cook_method": "bake", "cuisine": "Greek"},
    "tiropita": {"ingredients": {"wheat_flour": 60, "cheese": 80, "egg": 30, "butter": 20},
                 "cook_method": "bake", "cuisine": "Greek"},
    "keftedes": {"ingredients": {"ground_beef": 150, "bread": 20, "onion": 20, "egg": 20,
                 "olive_oil": 20, "cumin": 3}, "cook_method": "pan_fry", "cuisine": "Greek"},

    # ── LEBANESE ──
    "kibbeh": {"ingredients": {"lamb": 120, "wheat_flour": 60, "onion": 30, "pine_nut": 15,
               "butter": 10}, "cook_method": "deep_fry", "cuisine": "Lebanese"},
    "fattoush": {"ingredients": {"lettuce": 50, "tomato": 50, "cucumber": 40, "onion": 20,
                 "bread": 20, "olive_oil": 15, "lemon": 10}, "cook_method": "cold", "cuisine": "Lebanese"},
    "manakish": {"ingredients": {"wheat_flour": 100, "olive_oil": 15, "cheese": 30},
                 "cook_method": "bake", "cuisine": "Lebanese"},
    "muhammara": {"ingredients": {"pepper": 100, "walnut": 50, "bread": 20, "olive_oil": 20,
                  "garlic": 5, "chili": 5, "lemon": 5}, "cook_method": "raw", "cuisine": "Lebanese"},
    "mujaddara": {"ingredients": {"lentil": 100, "rice": 100, "onion": 60, "olive_oil": 20},
                  "cook_method": "simmer", "cuisine": "Lebanese"},
    "warak_enab": {"ingredients": {"rice": 80, "lamb": 60, "onion": 20, "tomato": 20,
                   "olive_oil": 10, "lemon": 10}, "cook_method": "simmer", "cuisine": "Lebanese"},
    "labneh": {"ingredients": {"yogurt": 200, "olive_oil": 15, "salt": 3}, "cook_method": "cold", "cuisine": "Lebanese"},
    "sfeeha": {"ingredients": {"wheat_flour": 80, "lamb": 80, "onion": 20, "tomato": 20,
               "pine_nut": 10}, "cook_method": "bake", "cuisine": "Lebanese"},
    "halloumi": {"ingredients": {"cheese": 150, "olive_oil": 10}, "cook_method": "grill", "cuisine": "Lebanese"},

    # ── SOUTHEAST ASIAN ──
    "nasi_goreng": {"ingredients": {"rice": 200, "egg": 50, "chicken": 60, "soy_sauce": 15,
                    "vegetable_oil": 15, "onion": 20, "garlic": 5, "chili": 5}, "cook_method": "stir_fry", "cuisine": "Indonesian"},
    "rendang": {"ingredients": {"beef": 200, "coconut_milk": 100, "onion": 40, "garlic": 10,
                "ginger": 10, "chili": 15, "turmeric": 3}, "cook_method": "braise", "cuisine": "Indonesian"},
    "satay_indonesian": {"ingredients": {"chicken": 200, "peanut": 40, "soy_sauce": 15,
                         "coconut_milk": 20, "garlic": 5, "chili": 5}, "cook_method": "grill", "cuisine": "Indonesian"},
    "lumpia": {"ingredients": {"wheat_flour": 40, "pork": 60, "cabbage": 40, "carrot": 20,
               "vegetable_oil": 30}, "cook_method": "deep_fry", "cuisine": "Filipino"},
    "adobo": {"ingredients": {"chicken": 200, "soy_sauce": 30, "vinegar": 30, "garlic": 10,
              "onion": 20, "black_pepper": 3}, "cook_method": "braise", "cuisine": "Filipino"},
    "sinigang": {"ingredients": {"pork": 150, "tomato": 50, "onion": 30, "spinach": 40,
                 "eggplant": 30}, "cook_method": "simmer", "cuisine": "Filipino"},
    "sisig": {"ingredients": {"pork": 200, "onion": 30, "chili": 10, "egg": 30,
              "lemon": 10}, "cook_method": "grill", "cuisine": "Filipino"},
    "kare_kare": {"ingredients": {"beef": 150, "peanut": 50, "eggplant": 40, "onion": 20,
                  "garlic": 5, "vegetable_oil": 10}, "cook_method": "simmer", "cuisine": "Filipino"},
    "halo_halo": {"ingredients": {"milk": 100, "ice_cream": 50, "sugar": 20, "banana": 20,
                  "sweet_potato": 20, "coconut": 15}, "cook_method": "cold", "cuisine": "Filipino"},
    "nasi_lemak": {"ingredients": {"rice": 200, "coconut_milk": 50, "egg": 50, "peanut": 20,
                   "fish": 30, "cucumber": 20}, "cook_method": "steam", "cuisine": "Indonesian"},
    "char_kway_teow": {"ingredients": {"rice_noodle": 200, "shrimp": 60, "egg": 50,
                       "soy_sauce": 15, "vegetable_oil": 20, "bean_sprout": 30}, "cook_method": "stir_fry", "cuisine": "Indonesian"},
    "laksa": {"ingredients": {"rice_noodle": 150, "shrimp": 60, "coconut_milk": 100,
              "chili": 10, "garlic": 5, "ginger": 5, "fish_sauce": 10}, "cook_method": "simmer", "cuisine": "Indonesian"},
    "rojak": {"ingredients": {"pineapple": 40, "cucumber": 40, "bean_sprout": 30,
              "peanut": 20, "sugar": 10}, "cook_method": "cold", "cuisine": "Indonesian"},
    "bak_kut_teh": {"ingredients": {"pork": 200, "garlic": 10, "black_pepper": 5,
                    "soy_sauce": 10}, "cook_method": "simmer", "cuisine": "Indonesian"},
    "mee_goreng": {"ingredients": {"pasta_dry": 150, "egg": 50, "chicken": 60, "soy_sauce": 15,
                   "vegetable_oil": 15, "onion": 20}, "cook_method": "stir_fry", "cuisine": "Indonesian"},
    "nasi_padang": {"ingredients": {"rice": 200, "chicken": 100, "egg": 50, "coconut_milk": 40,
                    "chili": 10, "garlic": 5}, "cook_method": "simmer", "cuisine": "Indonesian"},
    "gado_gado": {"ingredients": {"potato": 60, "bean_sprout": 40, "cabbage": 40, "tofu": 40,
                  "egg": 50, "peanut": 40}, "cook_method": "cold", "cuisine": "Indonesian"},
    "banh_xeo": {"ingredients": {"rice_noodle": 80, "shrimp": 50, "pork": 40, "bean_sprout": 40,
                 "coconut_milk": 30, "turmeric": 2}, "cook_method": "pan_fry", "cuisine": "Vietnamese"},
    "hu_tieu": {"ingredients": {"rice_noodle": 150, "pork": 60, "shrimp": 40, "onion": 20,
                "garlic": 5, "fish_sauce": 10}, "cook_method": "boil", "cuisine": "Vietnamese"},
    "bun_rieu": {"ingredients": {"rice_noodle": 150, "crab": 60, "tomato": 50, "tofu": 40,
                 "fish_sauce": 10}, "cook_method": "simmer", "cuisine": "Vietnamese"},

    # ── PORTUGUESE / SPANISH EXPANSION ──
    "bacalhau": {"ingredients": {"cod": 200, "potato": 80, "onion": 30, "egg": 30,
                 "olive_oil": 20}, "cook_method": "bake", "cuisine": "Portuguese"},
    "pasteis_de_nata": {"ingredients": {"wheat_flour": 50, "egg": 50, "cream": 60,
                        "sugar": 30, "butter": 20}, "cook_method": "bake", "cuisine": "Portuguese"},
    "caldo_verde": {"ingredients": {"potato": 150, "spinach": 80, "sausage": 40,
                    "olive_oil": 15, "garlic": 5}, "cook_method": "boil", "cuisine": "Portuguese"},
    "sardines_grilled": {"ingredients": {"fish": 200, "olive_oil": 10, "lemon": 10, "salt": 3},
                         "cook_method": "grill", "cuisine": "Portuguese"},
    "croquetas": {"ingredients": {"wheat_flour": 40, "butter": 20, "milk": 80, "chicken": 60,
                  "egg": 20, "vegetable_oil": 30}, "cook_method": "deep_fry", "cuisine": "Spanish"},
    "gambas_al_ajillo": {"ingredients": {"shrimp": 200, "garlic": 15, "olive_oil": 40,
                         "chili": 5}, "cook_method": "saute", "cuisine": "Spanish"},
    "tortilla_espanola": {"ingredients": {"potato": 200, "egg": 100, "onion": 40,
                          "olive_oil": 30}, "cook_method": "pan_fry", "cuisine": "Spanish"},

    # ── MISSING COMMON (EXISTING CUISINES) ──
    "hot_dog": {"ingredients": {"sausage": 80, "bread": 50, "mustard": 10, "onion": 15},
                "cook_method": "grill", "cuisine": "American"},
    "meatloaf": {"ingredients": {"ground_beef": 200, "bread": 40, "egg": 30, "onion": 30,
                 "ketchup": 20}, "cook_method": "bake", "cuisine": "American"},
    "chili": {"ingredients": {"ground_beef": 150, "black_bean": 80, "tomato": 80, "onion": 40,
              "garlic": 5, "chili": 10, "cumin": 5}, "cook_method": "simmer", "cuisine": "American"},
    "pot_roast": {"ingredients": {"beef": 250, "potato": 80, "carrot": 40, "onion": 40},
                  "cook_method": "braise", "cuisine": "American"},
    "cobb_salad": {"ingredients": {"lettuce": 60, "chicken": 80, "bacon": 30, "egg": 50,
                   "avocado": 40, "cheese": 20, "tomato": 30}, "cook_method": "cold", "cuisine": "American"},
    "eggs_benedict": {"ingredients": {"egg": 100, "bacon": 40, "bread": 60, "butter": 20,
                      "lemon": 5}, "cook_method": "boil", "cuisine": "American"},
    "poutine": {"ingredients": {"potato": 200, "cheese": 60, "butter": 15},
                "cook_method": "deep_fry", "cuisine": "American"},
    "fish_tacos": {"ingredients": {"fish": 120, "tortilla": 60, "cabbage": 30, "lime": 10,
                   "mayonnaise": 15, "chili": 5}, "cook_method": "pan_fry", "cuisine": "Mexican"},
    "carnitas": {"ingredients": {"pork": 250, "onion": 20, "garlic": 5, "lime": 10,
                 "vegetable_oil": 10}, "cook_method": "braise", "cuisine": "Mexican"},
    "mole": {"ingredients": {"chicken": 200, "chocolate": 20, "chili": 15, "peanut": 15,
             "onion": 20, "garlic": 5, "tortilla": 30}, "cook_method": "simmer", "cuisine": "Mexican"},
    "churros_mexican": {"ingredients": {"wheat_flour": 80, "egg": 30, "butter": 15,
                        "sugar": 20, "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "Mexican"},
    "chilaquiles": {"ingredients": {"tortilla": 80, "egg": 50, "tomato_sauce": 60,
                    "cheese": 30, "onion": 20, "chili": 10}, "cook_method": "pan_fry", "cuisine": "Mexican"},
    "al_pastor": {"ingredients": {"pork": 200, "pineapple": 30, "onion": 20, "chili": 10,
                  "tortilla": 40}, "cook_method": "grill", "cuisine": "Mexican"},
    "birria": {"ingredients": {"beef": 200, "chili": 15, "tomato": 40, "onion": 30,
               "garlic": 5, "cumin": 3, "tortilla": 40}, "cook_method": "braise", "cuisine": "Mexican"},
    "sopes": {"ingredients": {"corn": 80, "black_bean": 40, "cheese": 20, "lettuce": 15,
              "cream": 15}, "cook_method": "pan_fry", "cuisine": "Mexican"},

    # ── ITALIAN EXPANSION ──
    "arancini": {"ingredients": {"rice": 120, "cheese": 30, "egg": 30, "wheat_flour": 20,
                 "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "Italian"},
    "polenta": {"ingredients": {"corn": 150, "butter": 20, "cheese": 30, "milk": 50},
                "cook_method": "boil", "cuisine": "Italian"},
    "eggplant_parm": {"ingredients": {"eggplant": 200, "cheese": 60, "tomato_sauce": 80,
                      "wheat_flour": 20, "egg": 30, "olive_oil": 20}, "cook_method": "bake", "cuisine": "Italian"},
    "focaccia": {"ingredients": {"wheat_flour": 120, "olive_oil": 20, "salt": 3},
                 "cook_method": "bake", "cuisine": "Italian"},
    "prosciutto": {"ingredients": {"pork": 80, "bread": 30, "cheese": 20}, "cook_method": "cold", "cuisine": "Italian"},
    "cannoli": {"ingredients": {"wheat_flour": 50, "cream": 80, "chocolate": 10,
                "sugar": 20, "vegetable_oil": 20}, "cook_method": "deep_fry", "cuisine": "Italian"},

    # ── FRENCH EXPANSION ──
    "macarons": {"ingredients": {"almond": 60, "sugar": 60, "egg": 40, "butter": 20},
                 "cook_method": "bake", "cuisine": "French"},
    "eclair": {"ingredients": {"wheat_flour": 40, "egg": 50, "butter": 30, "cream": 50,
               "chocolate": 20}, "cook_method": "bake", "cuisine": "French"},
    "profiterole": {"ingredients": {"wheat_flour": 40, "egg": 50, "butter": 30, "cream": 60,
                    "chocolate": 15}, "cook_method": "bake", "cuisine": "French"},
    "beef_tartare": {"ingredients": {"beef": 180, "egg": 30, "onion": 15, "mustard": 10,
                     "olive_oil": 10}, "cook_method": "raw", "cuisine": "French"},
    "duck_confit": {"ingredients": {"duck": 250, "garlic": 5, "salt": 5}, "cook_method": "slow_cook", "cuisine": "French"},
    "bouillabaisse": {"ingredients": {"fish": 100, "shrimp": 50, "squid": 50, "tomato": 60,
                      "onion": 30, "garlic": 10, "olive_oil": 20}, "cook_method": "simmer", "cuisine": "French"},
    "nicoise_salad": {"ingredients": {"tuna": 80, "egg": 50, "potato": 60, "tomato": 40,
                      "olive_oil": 20, "lettuce": 30}, "cook_method": "cold", "cuisine": "French"},
    "galette": {"ingredients": {"wheat_flour": 100, "egg": 50, "cheese": 40, "butter": 15},
                "cook_method": "pan_fry", "cuisine": "French"},

    # ── JAPANESE EXPANSION ──
    "gyudon": {"ingredients": {"beef": 100, "rice": 200, "onion": 40, "soy_sauce": 15,
               "sugar": 5, "ginger": 5}, "cook_method": "simmer", "cuisine": "Japanese"},
    "tonkotsu_ramen": {"ingredients": {"pasta_dry": 120, "pork": 80, "egg": 50,
                       "garlic": 5, "soy_sauce": 15, "sesame_oil": 5}, "cook_method": "boil", "cuisine": "Japanese"},
    "karaage": {"ingredients": {"chicken": 200, "wheat_flour": 30, "soy_sauce": 15,
                "ginger": 5, "garlic": 5, "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "Japanese"},
    "katsu_curry": {"ingredients": {"pork": 150, "wheat_flour": 20, "egg": 30, "bread": 20,
                    "curry_paste": 20, "rice": 200, "vegetable_oil": 30}, "cook_method": "deep_fry", "cuisine": "Japanese"},
    "tamagoyaki": {"ingredients": {"egg": 120, "sugar": 5, "soy_sauce": 5, "vegetable_oil": 5},
                   "cook_method": "pan_fry", "cuisine": "Japanese"},

    # ── KOREAN EXPANSION ──
    "jjajangmyeon": {"ingredients": {"pasta_dry": 150, "pork": 80, "onion": 40, "potato": 30,
                     "soybean": 20, "vegetable_oil": 15}, "cook_method": "stir_fry", "cuisine": "Korean"},
    "dakgalbi": {"ingredients": {"chicken": 200, "cabbage": 50, "sweet_potato": 40,
                 "chili": 15, "soy_sauce": 10, "garlic": 5}, "cook_method": "stir_fry", "cuisine": "Korean"},
    "bossam": {"ingredients": {"pork": 250, "garlic": 10, "ginger": 5, "onion": 20},
               "cook_method": "boil", "cuisine": "Korean"},
    "pajeon": {"ingredients": {"wheat_flour": 80, "egg": 50, "onion": 40, "shrimp": 30,
               "vegetable_oil": 15}, "cook_method": "pan_fry", "cuisine": "Korean"},
    "budae_jjigae": {"ingredients": {"sausage": 50, "tofu": 50, "pasta_dry": 40,
                     "chili": 10, "onion": 20, "cheese": 15}, "cook_method": "simmer", "cuisine": "Korean"},
    "jokbal": {"ingredients": {"pork": 250, "soy_sauce": 15, "ginger": 5, "garlic": 5},
               "cook_method": "braise", "cuisine": "Korean"},

    # ── THAI EXPANSION ──
    "som_tam": {"ingredients": {"carrot": 60, "peanut": 15, "tomato": 30, "lime": 10,
                "fish_sauce": 10, "chili": 5, "sugar": 5}, "cook_method": "cold", "cuisine": "Thai"},
    "khao_man_gai": {"ingredients": {"chicken": 200, "rice": 200, "ginger": 10, "garlic": 5,
                     "soy_sauce": 10}, "cook_method": "boil", "cuisine": "Thai"},
    "boat_noodles": {"ingredients": {"rice_noodle": 150, "beef": 60, "bean_sprout": 30,
                     "fish_sauce": 10, "soy_sauce": 10, "chili": 5}, "cook_method": "simmer", "cuisine": "Thai"},

    # ── INDIAN EXPANSION ──
    "idli": {"ingredients": {"rice": 100, "lentil": 50}, "cook_method": "steam", "cuisine": "Indian"},
    "uttapam": {"ingredients": {"rice": 100, "lentil": 50, "onion": 20, "tomato": 20},
                "cook_method": "pan_fry", "cuisine": "Indian"},
    "pav_bhaji": {"ingredients": {"potato": 100, "tomato": 50, "onion": 30, "pepper": 20,
                  "butter": 20, "bread": 60}, "cook_method": "simmer", "cuisine": "Indian"},
    "chole_bhature": {"ingredients": {"chickpea": 120, "wheat_flour": 80, "onion": 20,
                      "tomato": 30, "vegetable_oil": 30}, "cook_method": "deep_fry", "cuisine": "Indian"},
    "aloo_gobi": {"ingredients": {"potato": 100, "broccoli": 100, "onion": 20, "tomato": 20,
                  "vegetable_oil": 15, "turmeric": 3, "cumin": 3}, "cook_method": "stir_fry", "cuisine": "Indian"},
    "paneer_butter": {"ingredients": {"cheese": 150, "butter": 30, "cream": 40, "tomato": 60,
                      "onion": 20, "garlic": 5}, "cook_method": "simmer", "cuisine": "Indian"},
    "malai_kofta": {"ingredients": {"cheese": 100, "potato": 50, "cream": 40, "tomato": 40,
                    "cashew": 20, "vegetable_oil": 20}, "cook_method": "deep_fry", "cuisine": "Indian"},
    "gulab_jamun": {"ingredients": {"milk": 100, "wheat_flour": 40, "sugar": 60, "butter": 15,
                    "vegetable_oil": 20}, "cook_method": "deep_fry", "cuisine": "Indian"},
    "jalebi": {"ingredients": {"wheat_flour": 80, "yogurt": 20, "sugar": 60, "vegetable_oil": 30},
               "cook_method": "deep_fry", "cuisine": "Indian"},

    # ── CHINESE EXPANSION ──
    "dim_sum_items": {"ingredients": {"wheat_flour": 60, "shrimp": 80, "pork": 40,
                      "sesame_oil": 5}, "cook_method": "steam", "cuisine": "Chinese"},
    "dan_tat": {"ingredients": {"wheat_flour": 50, "egg": 60, "cream": 40, "sugar": 20,
                "butter": 20}, "cook_method": "bake", "cuisine": "Chinese"},
    "char_siu_bao": {"ingredients": {"wheat_flour": 80, "pork": 60, "sugar": 10,
                     "soy_sauce": 10, "sesame_oil": 5}, "cook_method": "steam", "cuisine": "Chinese"},
    "wonton_noodle": {"ingredients": {"pasta_dry": 100, "pork": 60, "shrimp": 30,
                      "wheat_flour": 30, "soy_sauce": 10, "sesame_oil": 5}, "cook_method": "boil", "cuisine": "Chinese"},
    "salt_pepper_squid": {"ingredients": {"squid": 200, "wheat_flour": 30, "salt": 3,
                          "black_pepper": 3, "chili": 5, "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "Chinese"},

    # ── EUROPEAN ──
    "pierogi": {"ingredients": {"wheat_flour": 100, "potato": 80, "cheese": 30, "onion": 20,
                "butter": 15}, "cook_method": "boil", "cuisine": "Polish"},
    "borscht": {"ingredients": {"tomato": 80, "potato": 60, "onion": 30, "carrot": 30,
                "cabbage": 40, "cream": 20, "beef": 50}, "cook_method": "simmer", "cuisine": "Eastern European"},
    "blini": {"ingredients": {"wheat_flour": 80, "egg": 40, "milk": 60, "butter": 15},
              "cook_method": "pan_fry", "cuisine": "Russian"},
    "stroganoff": {"ingredients": {"beef": 180, "mushroom": 60, "onion": 30, "cream": 50,
                   "butter": 15, "pasta_dry": 100}, "cook_method": "saute", "cuisine": "Russian"},
    "schnitzel": {"ingredients": {"pork": 200, "wheat_flour": 20, "egg": 30, "bread": 30,
                  "vegetable_oil": 40}, "cook_method": "deep_fry", "cuisine": "German"},
    "bratwurst": {"ingredients": {"sausage": 150, "bread": 50, "mustard": 10, "onion": 20},
                  "cook_method": "grill", "cuisine": "German"},
    "pretzel": {"ingredients": {"wheat_flour": 120, "butter": 10, "sugar": 5, "salt": 5},
                "cook_method": "bake", "cuisine": "German"},
    "spaetzle": {"ingredients": {"wheat_flour": 100, "egg": 60, "milk": 30, "butter": 15},
                 "cook_method": "boil", "cuisine": "German"},
    "currywurst": {"ingredients": {"sausage": 150, "ketchup": 40, "curry_paste": 10},
                   "cook_method": "grill", "cuisine": "German"},
    "fondue": {"ingredients": {"cheese": 150, "bread": 80, "wine": 50}, "cook_method": "simmer", "cuisine": "Swiss"},
    "raclette": {"ingredients": {"cheese": 120, "potato": 100, "onion": 20},
                 "cook_method": "grill", "cuisine": "Swiss"},
}

# Wine doesn't have an impact factor — substitute
if "wine" not in impact_lookup:
    impact_lookup["wine"] = {"co2": 1.5, "water": 800, "land": 3.0}
if "ice_cream" not in impact_lookup:
    # Approximate: dairy + sugar
    impact_lookup["ice_cream"] = {"co2": 5.0, "water": 2500, "land": 15.0}
if "saffron" not in impact_lookup:
    impact_lookup["saffron"] = {"co2": 5.0, "water": 8000, "land": 3.0}
if "water" not in impact_lookup:
    impact_lookup["water"] = {"co2": 0.0, "water": 1, "land": 0.0}

print(f"Expanded recipes defined: {len(EXPANDED_RECIPES)}")

# ══════════════════════════════════════════════════════════════════
# Compute E for each expanded dish
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Computing environmental costs for expanded dishes...")
print("=" * 60)

rows = []
missing_all = set()

for dish_id, recipe in EXPANDED_RECIPES.items():
    ingredients = recipe["ingredients"]
    cook_method = recipe["cook_method"]
    cuisine = recipe["cuisine"]

    total_grams = sum(ingredients.values())
    missing = []
    carbon = 0
    water = 0
    land = 0

    for ing, grams in ingredients.items():
        if ing not in impact_lookup:
            missing.append(ing)
            missing_all.add(ing)
            continue
        kg = grams / 1000
        impact = impact_lookup[ing]
        carbon += kg * impact["co2"]
        water += kg * impact["water"]
        land += kg * impact["land"]

    # Cooking energy
    cook_energy = COOKING_ENERGY_KWH.get(cook_method, 0.5)
    cook_co2 = cook_energy * GRID_EMISSION_FACTOR

    rows.append({
        "dish_id": dish_id,
        "E_carbon": carbon + cook_co2,
        "E_carbon_ingredients": carbon,
        "E_carbon_cooking": cook_co2,
        "E_water": water + cook_energy,
        "E_water_ingredients": water,
        "E_energy": cook_energy,
        "E_land": land,
        "cook_method": cook_method,
        "cuisine": cuisine,
        "n_ingredients": len(ingredients),
        "total_grams": total_grams,
        "missing_ingredients": ",".join(missing) if missing else "",
    })

env_df = pd.DataFrame(rows)

if missing_all:
    print(f"  Missing impact factors: {missing_all}")

# Normalize E components (using expanded + original combined range)
orig_dei = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv")

# Combine for normalization
all_carbon = pd.concat([orig_dei["E_carbon"], env_df["E_carbon"]])
all_water = pd.concat([orig_dei["E_water"], env_df["E_water"]])
all_energy = pd.concat([orig_dei["E_energy"], env_df["E_energy"]])

c_min, c_max = all_carbon.min(), all_carbon.max()
w_min, w_max = all_water.min(), all_water.max()
e_min, e_max = all_energy.min(), all_energy.max()

env_df["E_carbon_norm"] = (env_df["E_carbon"] - c_min) / (c_max - c_min)
env_df["E_water_norm"] = (env_df["E_water"] - w_min) / (w_max - w_min)
env_df["E_energy_norm"] = (env_df["E_energy"] - e_min) / (e_max - e_min)
env_df["E_composite"] = (env_df["E_carbon_norm"] + env_df["E_water_norm"] + env_df["E_energy_norm"]) / 3

env_df.to_csv(DATA_DIR / "expanded_dish_env_costs.csv", index=False)
print(f"  Saved: {DATA_DIR / 'expanded_dish_env_costs.csv'}")
print(f"  E range: [{env_df['E_composite'].min():.4f}, {env_df['E_composite'].max():.4f}]")

# ══════════════════════════════════════════════════════════════════
# Score H using finetuned BERT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Scoring H with finetuned BERT...")
print("=" * 60)

# Check if model exists
model_dir = ROOT / "models" / "hedonic_bert_finetuned"
if model_dir.exists():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"  Model loaded on {device}")

        # Load expanded mentions
        mentions = pd.read_parquet(DATA_DIR / "expanded_dish_mentions.parquet")
        print(f"  Expanded mentions: {len(mentions):,}")

        # Score in batches
        BATCH_SIZE = 64
        all_scores = []

        for start in range(0, len(mentions), BATCH_SIZE):
            batch = mentions.iloc[start:start + BATCH_SIZE]
            texts = batch["context_text"].tolist()

            # Truncate
            texts = [t[:512] if isinstance(t, str) else "" for t in texts]

            enc = tokenizer(texts, padding=True, truncation=True, max_length=128,
                           return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**enc)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

            all_scores.extend(scores.tolist())

            if (start // BATCH_SIZE) % 50 == 0:
                print(f"    Scored {start + len(batch):,}/{len(mentions):,}")

        mentions["hedonic_score_finetuned"] = all_scores
        print(f"  Scoring complete. Score range: [{min(all_scores):.2f}, {max(all_scores):.2f}]")

        # Aggregate to dish level
        dish_h = mentions.groupby("dish_id")["hedonic_score_finetuned"].agg(
            H_mean="mean", H_median="median", H_std="std", H_n="count",
            H_q25=lambda x: x.quantile(0.25), H_q75=lambda x: x.quantile(0.75),
        ).reset_index()
        dish_h["H_ci95"] = 1.96 * dish_h["H_std"] / np.sqrt(dish_h["H_n"])
        dish_h["H_iqr"] = dish_h["H_q75"] - dish_h["H_q25"]

    except ImportError:
        print("  transformers not available — using star-based proxy")
        mentions = pd.read_parquet(DATA_DIR / "expanded_dish_mentions.parquet")
        # Proxy: stars * 1.6 + noise (roughly calibrated to finetuned range [6, 7.5])
        mentions["hedonic_score_finetuned"] = mentions["stars"] * 1.2 + 1.0 + np.random.normal(0, 0.3, len(mentions))
        dish_h = mentions.groupby("dish_id").agg(
            H_mean=("hedonic_score_finetuned", "mean"),
            H_median=("hedonic_score_finetuned", "median"),
            H_std=("hedonic_score_finetuned", "std"),
            H_n=("hedonic_score_finetuned", "count"),
        ).reset_index()
        dish_h["H_ci95"] = 1.96 * dish_h["H_std"] / np.sqrt(dish_h["H_n"])
        dish_h["H_q25"] = mentions.groupby("dish_id")["hedonic_score_finetuned"].quantile(0.25).values
        dish_h["H_q75"] = mentions.groupby("dish_id")["hedonic_score_finetuned"].quantile(0.75).values
        dish_h["H_iqr"] = dish_h["H_q75"] - dish_h["H_q25"]
else:
    print("  Model not found — using star-based proxy")
    mentions = pd.read_parquet(DATA_DIR / "expanded_dish_mentions.parquet")
    mentions["hedonic_score_finetuned"] = mentions["stars"] * 1.2 + 1.0
    dish_h = mentions.groupby("dish_id").agg(
        H_mean=("hedonic_score_finetuned", "mean"),
        H_median=("hedonic_score_finetuned", "median"),
        H_std=("hedonic_score_finetuned", "std"),
        H_n=("hedonic_score_finetuned", "count"),
    ).reset_index()
    dish_h["H_ci95"] = 1.96 * dish_h["H_std"] / np.sqrt(dish_h["H_n"])
    dish_h["H_q25"] = 0
    dish_h["H_q75"] = 0
    dish_h["H_iqr"] = 0

dish_h.to_csv(DATA_DIR / "expanded_dish_hedonic.csv", index=False)
print(f"  Saved: {DATA_DIR / 'expanded_dish_hedonic.csv'}")
print(f"  Dishes with H scores: {len(dish_h)}")

# ══════════════════════════════════════════════════════════════════
# Compute DEI for expanded dishes
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Computing DEI for expanded dishes...")
print("=" * 60)

# Merge H and E
merged = dish_h.merge(env_df, on="dish_id", how="inner")
merged["log_H"] = np.log(merged["H_mean"].clip(lower=1.0))
merged["log_E"] = np.log(merged["E_composite"].clip(lower=1e-6))
merged["log_DEI"] = merged["log_H"] - merged["log_E"]

merged.to_csv(DATA_DIR / "expanded_dish_DEI.csv", index=False)
print(f"  Expanded DEI computed for {len(merged)} dishes")
print(f"  Saved: {DATA_DIR / 'expanded_dish_DEI.csv'}")

# ── Combined dataset ─────────────────────────────────────────────
# Merge original 158 + expanded
orig = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv")
# Select common columns
common_cols = ["dish_id", "H_mean", "E_composite", "E_carbon", "E_water", "E_energy",
               "log_H", "log_E", "log_DEI", "cuisine", "cook_method"]

orig_sub = orig[[c for c in common_cols if c in orig.columns]].copy()
orig_sub["source"] = "original"

merged_sub = merged.rename(columns={"cook_method": "cook_method"})[
    [c for c in common_cols if c in merged.columns]
].copy()
merged_sub["source"] = "expanded"

combined = pd.concat([orig_sub, merged_sub], ignore_index=True)
combined.to_csv(DATA_DIR / "combined_dish_DEI.csv", index=False)
print(f"\n  Combined dataset: {len(combined)} dishes (158 original + {len(merged)} expanded)")
print(f"  Saved: {DATA_DIR / 'combined_dish_DEI.csv'}")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n  New dishes with E + H + DEI: {len(merged)}")
print(f"  Cuisine coverage:")
for c in merged["cuisine"].value_counts().index:
    n = (merged["cuisine"] == c).sum()
    print(f"    {c:25s}: {n}")

print(f"\n  H range: [{merged['H_mean'].min():.2f}, {merged['H_mean'].max():.2f}]")
print(f"  E range: [{merged['E_composite'].min():.4f}, {merged['E_composite'].max():.4f}]")
print(f"  DEI range: [{merged['log_DEI'].min():.2f}, {merged['log_DEI'].max():.2f}]")

# Top/Bottom expanded dishes
print(f"\n  Top 10 expanded dishes by log(DEI):")
for _, row in merged.nlargest(10, "log_DEI").iterrows():
    print(f"    {row['dish_id']:25s}: log_DEI={row['log_DEI']:.2f}, H={row['H_mean']:.2f}, E={row['E_composite']:.4f} ({row['cuisine']})")

print(f"\n  Bottom 10 expanded dishes by log(DEI):")
for _, row in merged.nsmallest(10, "log_DEI").iterrows():
    print(f"    {row['dish_id']:25s}: log_DEI={row['log_DEI']:.2f}, H={row['H_mean']:.2f}, E={row['E_composite']:.4f} ({row['cuisine']})")

print("\nDone!")
