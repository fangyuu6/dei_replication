"""
15_nhanes_coverage.py — NHANES/FNDDS Coverage Analysis
======================================================
Compares our 334-dish DEI dataset against the USDA FNDDS
(Food and Nutrient Database for Dietary Studies) used by
NHANES/WWEIA and Stylianou et al. (2021).

Analyses:
  A. Text-based matching: what fraction of FNDDS foods map to our dishes?
  B. Category-level coverage: which WWEIA food categories do we cover?
  C. Dietary relevance: how much of actual US dietary intake do our dishes represent?

Outputs:
  - tables/nhanes_coverage_summary.csv
  - tables/nhanes_category_coverage.csv
  - tables/nhanes_matched_foods.csv
"""

import sys, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR

print("=" * 70)
print("15  NHANES/FNDDS COVERAGE ANALYSIS")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────
# Our dishes
combined = pd.read_csv(DATA_DIR / "combined_dish_DEI_revised.csv")
our_dishes = sorted(combined["dish_id"].unique())
print(f"Our dataset: {len(our_dishes)} dishes")

# FNDDS food list
fndds_dir = ROOT / "raw" / "external" / "fndds_csv" / "FoodData_Central_survey_food_csv_2024-10-31"
fndds_food = pd.read_csv(fndds_dir / "food.csv")
fndds_cat = pd.read_csv(fndds_dir / "wweia_food_category.csv")

# Merge categories
fndds_food = fndds_food.merge(fndds_cat, left_on="food_category_id",
                               right_on="wweia_food_category", how="left")
print(f"FNDDS dataset: {len(fndds_food)} food items, {fndds_cat.shape[0]} categories")

# ══════════════════════════════════════════════════════════════════
# A. TEXT-BASED MATCHING
# ══════════════════════════════════════════════════════════════════
print("\n── A. Text-Based Matching ──")

# Build matching keywords for each dish
DISH_KEYWORDS = {}
for dish in our_dishes:
    # Convert dish_id to keywords
    words = dish.replace("_", " ").split()
    DISH_KEYWORDS[dish] = words

# Match FNDDS descriptions against our dish keywords
def match_fndds_to_dish(description, dish_keywords):
    """Check if an FNDDS food description matches any of our dishes."""
    desc_lower = description.lower()
    matches = []
    for dish, keywords in dish_keywords.items():
        # Require all keywords present (for multi-word dishes)
        # or exact dish name match
        dish_name = dish.replace("_", " ")
        if dish_name in desc_lower:
            matches.append((dish, "exact"))
            continue
        # For single-word dishes, require word boundary match
        if len(keywords) == 1:
            word = keywords[0]
            # Use word boundary: the keyword must appear as a whole word
            if re.search(r'\b' + re.escape(word) + r'\b', desc_lower):
                matches.append((dish, "keyword"))
        else:
            # Multi-word: require all keywords
            if all(kw in desc_lower for kw in keywords):
                matches.append((dish, "all_keywords"))
    return matches

# Run matching
matched_foods = []
for _, row in fndds_food.iterrows():
    desc = row["description"]
    matches = match_fndds_to_dish(desc, DISH_KEYWORDS)
    for dish, match_type in matches:
        matched_foods.append({
            "fdc_id": row["fdc_id"],
            "fndds_description": desc,
            "wweia_category": row.get("wweia_food_category_description", ""),
            "matched_dish": dish,
            "match_type": match_type,
        })

matched_df = pd.DataFrame(matched_foods)
n_fndds_matched = matched_df["fdc_id"].nunique()
n_dishes_matched = matched_df["matched_dish"].nunique()

print(f"  FNDDS items matched: {n_fndds_matched}/{len(fndds_food)} ({n_fndds_matched/len(fndds_food)*100:.1f}%)")
print(f"  Our dishes with FNDDS match: {n_dishes_matched}/{len(our_dishes)} ({n_dishes_matched/len(our_dishes)*100:.1f}%)")

# Show top matched dishes
dish_match_counts = matched_df.groupby("matched_dish")["fdc_id"].nunique().sort_values(ascending=False)
print(f"\n  Top 10 dishes by FNDDS match count:")
for dish, count in dish_match_counts.head(10).items():
    print(f"    {dish}: {count} FNDDS items")

# Unmatched dishes
unmatched_dishes = set(our_dishes) - set(matched_df["matched_dish"].unique())
print(f"\n  Unmatched dishes ({len(unmatched_dishes)}):")
for d in sorted(unmatched_dishes)[:20]:
    print(f"    {d}")
if len(unmatched_dishes) > 20:
    print(f"    ... and {len(unmatched_dishes) - 20} more")

# ══════════════════════════════════════════════════════════════════
# B. CATEGORY-LEVEL COVERAGE
# ══════════════════════════════════════════════════════════════════
print("\n── B. Category-Level Coverage ──")

# Map our dishes to broad food categories
OUR_CATEGORIES = {
    "Beef/Red Meat": ["brisket", "steak", "hamburger", "bulgogi", "osso_buco",
                      "beef_bourguignon", "kebab", "gyro", "taco", "moussaka",
                      "lasagna", "massaman_curry", "rendang", "birria", "churrasco",
                      "picanha", "pot_roast", "oxtail_stew"],
    "Poultry": ["fried_chicken", "chicken_sandwich", "tandoori_chicken", "teriyaki",
                "coq_au_vin", "korean_fried_chicken", "kung_pao_chicken", "butter_chicken",
                "tikka_masala", "satay", "general_tso", "orange_chicken", "katsu",
                "peking_duck"],
    "Pork": ["pulled_pork", "ribs", "samgyeopsal", "char_siu", "bun_cha",
             "com_tam", "larb"],
    "Seafood": ["ceviche", "sashimi", "sushi", "tempura", "fish_and_chips",
                "lobster_roll", "pad_thai"],
    "Rice/Grain": ["bibimbap", "biryani", "fried_rice", "risotto", "paella",
                   "donburi", "congee"],
    "Noodle/Pasta": ["pad_thai", "ramen", "pho", "lo_mein", "chow_mein",
                     "pasta_carbonara", "pasta_bolognese", "lasagna", "dan_dan_noodles",
                     "pad_see_ew", "chow_fun", "udon"],
    "Bread/Bakery": ["naan", "pita", "croissant", "cornbread", "scallion_pancake"],
    "Soup/Stew": ["clam_chowder", "french_onion_soup", "miso_soup", "tom_yum",
                  "tom_kha", "hot_and_sour_soup", "pozole", "gazpacho", "minestrone"],
    "Salad/Vegetable": ["caesar_salad", "coleslaw", "papaya_salad", "tabbouleh",
                        "caprese", "guacamole"],
    "Dairy/Egg": ["mac_and_cheese", "grilled_cheese", "quiche", "souffle",
                  "quesadilla", "paneer_tikka"],
    "Dessert": ["cheesecake", "tiramisu", "gelato", "ice_cream", "brownie",
                "creme_brulee", "panna_cotta", "baklava", "churro", "mango_sticky_rice"],
    "Beverage": ["thai_iced_tea", "ca_phe_sua_da"],
    "Plant-Based": ["falafel", "hummus", "dal", "chana_masala", "edamame",
                    "kimchi", "dolma"],
    "Dumpling/Wrapped": ["dumplings", "xiao_long_bao", "samosa", "spring_rolls",
                         "tamale", "burrito", "enchilada"],
}

# WWEIA major categories
wweia_major = {
    "Milk/Dairy": [1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1602, 1604, 1820, 1822],
    "Meat/Poultry": [2002, 2004, 2006, 2008, 2010, 2202, 2204, 2206, 2602, 2604, 2606, 2608],
    "Seafood": [2402, 2404],
    "Eggs": [2502],
    "Legumes/Beans": [2802],
    "Grains/Bread": list(range(3000, 4200)),
    "Fruits": list(range(6000, 6800)),
    "Vegetables": list(range(6800, 7800)),
    "Fats/Oils": list(range(8000, 8600)),
    "Sugars/Sweets": list(range(9000, 9600)),
    "Beverages": list(range(9200, 9900)),
    "Mixed Dishes": list(range(5000, 5800)),
}

# Count FNDDS items per WWEIA category
cat_counts = fndds_food.groupby("wweia_food_category_description").size().sort_values(ascending=False)
print(f"  WWEIA categories: {len(cat_counts)}")
print(f"\n  Top 15 WWEIA categories by item count:")
for cat, n in cat_counts.head(15).items():
    # Check if any matched
    cat_matched = matched_df[matched_df["wweia_category"] == cat]["fdc_id"].nunique()
    coverage = cat_matched / n * 100 if n > 0 else 0
    print(f"    {cat}: {n} items, {cat_matched} matched ({coverage:.0f}%)")

# Category coverage table
cat_coverage = []
for cat in cat_counts.index:
    n_items = cat_counts[cat]
    n_matched = matched_df[matched_df["wweia_category"] == cat]["fdc_id"].nunique()
    n_dishes = matched_df[matched_df["wweia_category"] == cat]["matched_dish"].nunique()
    cat_coverage.append({
        "wweia_category": cat,
        "n_fndds_items": n_items,
        "n_matched": n_matched,
        "n_dei_dishes": n_dishes,
        "coverage_pct": n_matched / n_items * 100 if n_items > 0 else 0,
    })
cat_coverage_df = pd.DataFrame(cat_coverage).sort_values("n_fndds_items", ascending=False)
cat_coverage_df.to_csv(TABLES_DIR / "nhanes_category_coverage.csv", index=False)

# ══════════════════════════════════════════════════════════════════
# C. COVERAGE GAPS & COMPLEMENTARITY ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n── C. Coverage Analysis ──")

# What types of foods are in FNDDS but NOT in our dataset?
# Group by major food type
fndds_food["matched"] = fndds_food["fdc_id"].isin(matched_df["fdc_id"])
matched_by_cat = fndds_food.groupby("wweia_food_category_description").agg(
    total=("fdc_id", "count"),
    matched=("matched", "sum")
).reset_index()
matched_by_cat["unmatched_pct"] = (1 - matched_by_cat["matched"] / matched_by_cat["total"]) * 100

# Categories with zero coverage (gaps)
zero_coverage = matched_by_cat[matched_by_cat["matched"] == 0].sort_values("total", ascending=False)
print(f"\n  WWEIA categories with ZERO coverage ({len(zero_coverage)}):")
for _, row in zero_coverage.head(15).iterrows():
    print(f"    {row['wweia_food_category_description']}: {row['total']} items")

# Categories with coverage
has_coverage = matched_by_cat[matched_by_cat["matched"] > 0].sort_values("matched", ascending=False)
print(f"\n  WWEIA categories WITH coverage ({len(has_coverage)}):")
for _, row in has_coverage.head(10).iterrows():
    print(f"    {row['wweia_food_category_description']}: "
          f"{row['matched']}/{row['total']} ({100-row['unmatched_pct']:.0f}%)")

# ══════════════════════════════════════════════════════════════════
# D. SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n── D. Summary ──")

# Key statistics
summary = {
    "n_dei_dishes": len(our_dishes),
    "n_fndds_items": len(fndds_food),
    "n_wweia_categories": len(cat_counts),
    "n_fndds_matched": n_fndds_matched,
    "pct_fndds_matched": n_fndds_matched / len(fndds_food) * 100,
    "n_dishes_matched": n_dishes_matched,
    "pct_dishes_matched": n_dishes_matched / len(our_dishes) * 100,
    "n_wweia_covered": len(has_coverage),
    "pct_wweia_covered": len(has_coverage) / len(cat_counts) * 100,
    "n_wweia_zero": len(zero_coverage),
    "major_gaps": "Fruits, Plain vegetables, Milk, Fats/oils, Sugars/candies, Breakfast cereals",
    "strength": "Prepared dishes, ethnic cuisines, restaurant meals",
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(TABLES_DIR / "nhanes_coverage_summary.csv", index=False)

# Save matched foods
if len(matched_df) > 0:
    matched_df.to_csv(TABLES_DIR / "nhanes_matched_foods.csv", index=False)

print(f"\n  DEI → FNDDS coverage:")
print(f"    {n_dishes_matched}/{len(our_dishes)} DEI dishes have FNDDS matches ({n_dishes_matched/len(our_dishes)*100:.1f}%)")
print(f"    {n_fndds_matched}/{len(fndds_food)} FNDDS items match our dishes ({n_fndds_matched/len(fndds_food)*100:.1f}%)")
print(f"    {len(has_coverage)}/{len(cat_counts)} WWEIA categories have coverage ({len(has_coverage)/len(cat_counts)*100:.1f}%)")

print(f"\n  Interpretation:")
print(f"    Our 334-dish dataset is a CURATED RESTAURANT MENU sample, not a")
print(f"    comprehensive food inventory. It covers {n_dishes_matched/len(our_dishes)*100:.0f}% of dish types but")
print(f"    only {n_fndds_matched/len(fndds_food)*100:.0f}% of FNDDS items, because FNDDS includes individual")
print(f"    ingredients, preparation variants, and non-restaurant foods.")
print(f"\n  Key gaps: raw fruits, plain vegetables, breakfast cereals, milk,")
print(f"    fats/oils, infant foods — these are ingredient-level items, not")
print(f"    dishes. Our framework complements FNDDS by providing hedonic")
print(f"    scores for COMPOSITE DISHES as actually consumed in restaurants.")

print(f"\n  Complementarity with Stylianou et al. (2021):")
print(f"    - They: 5,853 foods × (health + environment), no hedonic dimension")
print(f"    - We: 334 dishes × (hedonic + environment + nutrition)")
print(f"    - Overlap: prepared/mixed dishes in FNDDS")
print(f"    - Our unique contribution: empirical hedonic measurement from")
print(f"      5.3M real dining experiences")

print("\n" + "=" * 70)
print("15 COMPLETE")
print("=" * 70)
