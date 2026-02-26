"""
15b_fndds_validation.py — FNDDS Nutritional Cross-Validation
=============================================================
Uses USDA FNDDS 2021-2023 official nutrient composition data to
validate our recipe-based calorie, protein, and NDI estimates.

This provides external validation against a nationally representative
food composition database used by NHANES/WWEIA.

Analyses:
  A. Match our 334 dishes to FNDDS items (improved matching)
  B. Compare calorie and protein estimates
  C. Compare key nutrients (iron, calcium, vitamin C, etc.)
  D. Assess systematic bias direction

Outputs:
  - tables/fndds_nutrient_validation.csv
  - tables/fndds_validation_summary.csv
  - figures/fndds_calorie_validation.png
"""

import sys, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

print("=" * 70)
print("15b  FNDDS NUTRITIONAL CROSS-VALIDATION")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────
combined = pd.read_csv(DATA_DIR / "combined_dish_DEI_revised.csv")
print(f"Our dataset: {len(combined)} dishes")

fndds_dir = ROOT / "raw" / "external" / "fndds_csv" / "FoodData_Central_survey_food_csv_2024-10-31"
fndds_food = pd.read_csv(fndds_dir / "food.csv")
fndds_nutr = pd.read_csv(fndds_dir / "food_nutrient.csv")
fndds_nutr_def = pd.read_csv(fndds_dir / "nutrient.csv")
fndds_portion = pd.read_csv(fndds_dir / "food_portion.csv")
fndds_cat = pd.read_csv(fndds_dir / "wweia_food_category.csv")

fndds_food = fndds_food.merge(fndds_cat, left_on="food_category_id",
                               right_on="wweia_food_category", how="left")
print(f"FNDDS: {len(fndds_food)} foods, {len(fndds_nutr)} nutrient records")

# Key nutrient IDs — FNDDS uses nutrient_nbr (not nutrient.id)
NUTRIENT_IDS = {
    208: "energy_kcal",     # Energy (kcal)
    203: "protein_g",       # Protein (g)
    204: "fat_g",           # Total fat (g)
    205: "carb_g",          # Carbohydrate (g)
    291: "fiber_g",         # Fiber, total dietary (g)
    303: "iron_mg",         # Iron, Fe (mg)
    309: "zinc_mg",         # Zinc, Zn (mg)
    418: "vitamin_b12_ug",  # Vitamin B-12 (ug)
    301: "calcium_mg",      # Calcium, Ca (mg)
    401: "vitamin_c_mg",    # Vitamin C (mg)
    307: "sodium_mg",       # Sodium, Na (mg)
    306: "potassium_mg",    # Potassium, K (mg)
    601: "cholesterol_mg",  # Cholesterol (mg)
    606: "sat_fat_g",       # Fatty acids, total saturated (g)
}

# Pivot nutrients to wide format for each food
nutr_wide = fndds_nutr[fndds_nutr["nutrient_id"].isin(NUTRIENT_IDS.keys())].copy()
nutr_wide["nutrient_name"] = nutr_wide["nutrient_id"].map(NUTRIENT_IDS)
nutr_pivot = nutr_wide.pivot_table(index="fdc_id", columns="nutrient_name",
                                    values="amount", aggfunc="first")
nutr_pivot = nutr_pivot.reset_index()
print(f"Nutrient matrix: {len(nutr_pivot)} foods × {len(NUTRIENT_IDS)} nutrients")

# ══════════════════════════════════════════════════════════════════
# A. IMPROVED MATCHING
# ══════════════════════════════════════════════════════════════════
print("\n── A. Improved Dish-to-FNDDS Matching ──")

# Manual curated matches for key dishes
MANUAL_MATCHES = {
    "hamburger": ["Hamburger", "Cheeseburger"],
    "pizza": ["Pizza"],
    "fried_chicken": ["Chicken, fried"],
    "fried_rice": ["Rice, fried"],
    "burrito": ["Burrito"],
    "taco": ["Taco"],
    "sushi": ["Sushi"],
    "ramen": ["Ramen"],
    "lasagna": ["Lasagna"],
    "steak": ["Steak", "Beef steak"],
    "ice_cream": ["Ice cream"],
    "cheesecake": ["Cheesecake"],
    "brownie": ["Brownie"],
    "pancake": ["Pancake"],
    "waffle": ["Waffle"],
    "french_fries": ["French fries", "Potato, french fries"],
    "mac_and_cheese": ["Macaroni and cheese"],
    "nachos": ["Nachos"],
    "quesadilla": ["Quesadilla"],
    "enchilada": ["Enchilada"],
    "chicken_sandwich": ["Chicken sandwich", "Chicken patty sandwich"],
    "grilled_cheese": ["Grilled cheese"],
    "soup": ["Soup"],
    "salad": ["Salad"],
    "croissant": ["Croissant"],
    "gelato": ["Ice cream", "Gelato"],
    "pad_thai": ["Pad Thai", "Thai noodle"],
    "lo_mein": ["Lo mein", "Chow mein"],
    "chow_mein": ["Chow mein"],
    "bibimbap": ["Bibimbap", "Korean rice"],
    "egg_roll": ["Egg roll"],
    "spring_rolls": ["Egg roll", "Spring roll"],
    "dumplings": ["Dumpling", "Potsticker"],
    "teriyaki": ["Teriyaki"],
    "tempura": ["Tempura"],
    "clam_chowder": ["Clam chowder"],
    "french_onion_soup": ["French onion soup", "Onion soup"],
    "miso_soup": ["Miso soup"],
    "risotto": ["Risotto"],
    "pasta_carbonara": ["Pasta, carbonara", "Spaghetti, carbonara"],
    "pasta_bolognese": ["Pasta, meat sauce", "Spaghetti, meat sauce", "Bolognese"],
    "falafel": ["Falafel"],
    "hummus": ["Hummus"],
    "kebab": ["Kebab", "Kabob"],
    "gyro": ["Gyro"],
    "biryani": ["Biryani"],
    "tikka_masala": ["Tikka masala", "Curry, chicken"],
    "butter_chicken": ["Butter chicken", "Curry, chicken"],
    "naan": ["Naan"],
    "samosa": ["Samosa"],
    "pho": ["Pho", "Vietnamese soup"],
    "fish_and_chips": ["Fish, battered", "Fish and chips"],
    "ceviche": ["Ceviche"],
    "tamale": ["Tamale"],
    "cornbread": ["Cornbread"],
    "pulled_pork": ["Pork, pulled", "Pork, barbecued"],
    "ribs": ["Ribs"],
    "coleslaw": ["Coleslaw"],
    "guacamole": ["Guacamole"],
    "lobster_roll": ["Lobster roll", "Lobster sandwich"],
    "brisket": ["Beef, brisket"],
    "quiche": ["Quiche"],
    "crepe": ["Crepe"],
    "tiramisu": ["Tiramisu"],
}

def find_best_fndds_match(dish_id, fndds_food_df, nutr_df):
    """Find best FNDDS match for a dish and return averaged nutrients."""
    desc_lower = fndds_food_df["description"].str.lower()

    # Try manual matches first
    if dish_id in MANUAL_MATCHES:
        keywords = MANUAL_MATCHES[dish_id]
        mask = pd.Series(False, index=fndds_food_df.index)
        for kw in keywords:
            mask |= desc_lower.str.contains(kw.lower(), na=False)
        matched_ids = fndds_food_df[mask]["fdc_id"].tolist()
    else:
        # Auto match: dish name as keyword
        dish_name = dish_id.replace("_", " ")
        mask = desc_lower.str.contains(dish_name, na=False)
        matched_ids = fndds_food_df[mask]["fdc_id"].tolist()

    if not matched_ids:
        return None

    # Average nutrients across all matched FNDDS items
    matched_nutr = nutr_df[nutr_df["fdc_id"].isin(matched_ids)]
    if matched_nutr.empty:
        return None

    result = {"n_fndds_matches": len(matched_ids)}
    for col in matched_nutr.columns:
        if col != "fdc_id" and matched_nutr[col].dtype in ["float64", "int64"]:
            result[f"fndds_{col}"] = matched_nutr[col].mean()

    return result

# Run matching
matches = {}
for dish in combined["dish_id"].unique():
    result = find_best_fndds_match(dish, fndds_food, nutr_pivot)
    if result is not None:
        matches[dish] = result

print(f"  Matched {len(matches)}/{len(combined)} dishes to FNDDS")

# ══════════════════════════════════════════════════════════════════
# B. CALORIE & PROTEIN COMPARISON
# ══════════════════════════════════════════════════════════════════
print("\n── B. Calorie & Protein Comparison ──")

# Build comparison table
comparison = []
for dish_id, fndds_vals in matches.items():
    our_row = combined[combined["dish_id"] == dish_id].iloc[0]

    # FNDDS values are per 100g, ours are per serving
    # We'll compare per-100g if possible, or note the caveat
    fndds_cal = fndds_vals.get("fndds_energy_kcal", np.nan)
    fndds_prot = fndds_vals.get("fndds_protein_g", np.nan)
    fndds_fat = fndds_vals.get("fndds_fat_g", np.nan)
    fndds_iron = fndds_vals.get("fndds_iron_mg", np.nan)
    fndds_calcium = fndds_vals.get("fndds_calcium_mg", np.nan)
    fndds_vitc = fndds_vals.get("fndds_vitamin_c_mg", np.nan)
    fndds_fiber = fndds_vals.get("fndds_fiber_g", np.nan)

    our_cal = our_row.get("calorie_kcal", np.nan)
    our_prot = our_row.get("protein_g", np.nan)

    comparison.append({
        "dish_id": dish_id,
        "cuisine": our_row.get("cuisine", ""),
        "meal_role": our_row.get("meal_role", ""),
        "n_fndds_matches": fndds_vals["n_fndds_matches"],
        "our_cal_per_serving": our_cal,
        "fndds_cal_per_100g": fndds_cal,
        "our_protein_per_serving": our_prot,
        "fndds_protein_per_100g": fndds_prot,
        "fndds_fat_per_100g": fndds_fat,
        "fndds_iron_per_100g": fndds_iron,
        "fndds_calcium_per_100g": fndds_calcium,
        "fndds_vitc_per_100g": fndds_vitc,
        "fndds_fiber_per_100g": fndds_fiber,
        "fndds_zinc_per_100g": fndds_vals.get("fndds_zinc_mg", np.nan),
        "fndds_b12_per_100g": fndds_vals.get("fndds_vitamin_b12_ug", np.nan),
        "log_DEI": our_row.get("log_DEI", np.nan),
        "H_mean": our_row.get("H_mean", np.nan),
        "E_composite": our_row.get("E_composite", np.nan),
    })

comp_df = pd.DataFrame(comparison)

# Since FNDDS is per 100g and ours is per serving, we estimate serving weight
# Our recipes define grams, so calorie_kcal / (fndds_cal_per_100g/100) = estimated grams
comp_df["est_serving_g"] = comp_df["our_cal_per_serving"] / (comp_df["fndds_cal_per_100g"] / 100)
comp_df["est_serving_g"] = comp_df["est_serving_g"].clip(50, 2000)

# Scale FNDDS to estimated serving size for comparison
comp_df["fndds_cal_scaled"] = comp_df["fndds_cal_per_100g"] * comp_df["est_serving_g"] / 100
comp_df["fndds_prot_scaled"] = comp_df["fndds_protein_per_100g"] * comp_df["est_serving_g"] / 100

# Calorie agreement (should be ~1.0 by construction since we scale)
# More meaningful: protein ratio
valid_prot = comp_df.dropna(subset=["our_protein_per_serving", "fndds_prot_scaled"])
valid_prot = valid_prot[valid_prot["fndds_prot_scaled"] > 0]

if len(valid_prot) > 5:
    prot_ratio = valid_prot["our_protein_per_serving"] / valid_prot["fndds_prot_scaled"]
    prot_corr = sp_stats.pearsonr(valid_prot["our_protein_per_serving"],
                                   valid_prot["fndds_prot_scaled"])
    print(f"  Protein comparison ({len(valid_prot)} dishes):")
    print(f"    Pearson r = {prot_corr[0]:.3f} (p = {prot_corr[1]:.2e})")
    print(f"    Mean ratio (ours/FNDDS) = {prot_ratio.mean():.2f}")
    print(f"    Median ratio = {prot_ratio.median():.2f}")

# ══════════════════════════════════════════════════════════════════
# C. NUTRIENT DENSITY COMPARISON
# ══════════════════════════════════════════════════════════════════
print("\n── C. Nutrient Density from FNDDS ──")

# Compute FNDDS-based NDI (NRF-7 per 100kcal) for matched dishes
# DRVs from Drewnowski 2009
DRV = {
    "protein_g": 50, "fiber_g": 25, "iron_mg": 18,
    "zinc_mg": 15, "vitamin_b12_ug": 2.4,
    "calcium_mg": 1000, "vitamin_c_mg": 60,
}

# Map DRV nutrient names to actual comp_df column names
DRV_COL_MAP = {
    "protein_g": "fndds_protein_per_100g",
    "fiber_g": "fndds_fiber_per_100g",
    "iron_mg": "fndds_iron_per_100g",
    "zinc_mg": "fndds_zinc_per_100g",
    "vitamin_b12_ug": "fndds_b12_per_100g",
    "calcium_mg": "fndds_calcium_per_100g",
    "vitamin_c_mg": "fndds_vitc_per_100g",
}

fndds_ndis = []
for _, row in comp_df.iterrows():
    cal_100g = row.get("fndds_cal_per_100g", 0)
    if pd.isna(cal_100g) or cal_100g <= 0:
        continue

    ndi = 0
    for nutrient, drv in DRV.items():
        fndds_col = DRV_COL_MAP.get(nutrient)
        if fndds_col is None or fndds_col not in row.index:
            continue
        val = row[fndds_col]
        if pd.isna(val):
            continue
        # Per 100kcal: (val_per_100g / cal_per_100g) * 100
        per_100kcal = (val / cal_100g) * 100
        pct_drv = min(per_100kcal / drv * 100, 100)
        ndi += pct_drv

    fndds_ndis.append({
        "dish_id": row["dish_id"],
        "fndds_ndi": ndi,
        "fndds_cal_per_100g": cal_100g,
    })

ndi_df = pd.DataFrame(fndds_ndis)

# Merge with our NDI if available
if "NDI" in combined.columns:
    ndi_merged = ndi_df.merge(combined[["dish_id", "NDI"]], on="dish_id", how="inner")
    if len(ndi_merged) > 5:
        ndi_corr = sp_stats.pearsonr(ndi_merged["NDI"], ndi_merged["fndds_ndi"])
        ndi_spearman = sp_stats.spearmanr(ndi_merged["NDI"], ndi_merged["fndds_ndi"])
        print(f"  NDI comparison ({len(ndi_merged)} dishes):")
        print(f"    Our NDI range: [{ndi_merged['NDI'].min():.1f}, {ndi_merged['NDI'].max():.1f}]")
        print(f"    FNDDS NDI range: [{ndi_merged['fndds_ndi'].min():.1f}, {ndi_merged['fndds_ndi'].max():.1f}]")
        print(f"    Pearson r = {ndi_corr[0]:.3f} (p = {ndi_corr[1]:.2e})")
        print(f"    Spearman rho = {ndi_spearman[0]:.3f} (p = {ndi_spearman[1]:.2e})")

# ══════════════════════════════════════════════════════════════════
# D. DEI RANKING VALIDATION
# ══════════════════════════════════════════════════════════════════
print("\n── D. DEI Ranking Stability with FNDDS Nutrients ──")

# Key question: if we replace our nutrient estimates with FNDDS ones,
# do DEI rankings change?
# DEI = log(H) - log(E): nutrients don't directly affect DEI
# But NDI-adjusted DEI = log(H) + alpha*log(NDI) - log(E)
# If FNDDS NDI is different, DEI-N rankings might shift

if "NDI" in combined.columns and len(ndi_df) > 5:
    ndi_merged2 = ndi_df.merge(
        combined[["dish_id", "H_mean", "E_composite", "log_DEI", "NDI"]],
        on="dish_id", how="inner"
    )
    # Compute DEI-N with FNDDS NDI
    alpha = 0.5
    ndi_merged2["log_DEIN_ours"] = (np.log(ndi_merged2["H_mean"])
                                     + alpha * np.log(ndi_merged2["NDI"].clip(lower=0.1))
                                     - np.log(ndi_merged2["E_composite"]))
    ndi_merged2["log_DEIN_fndds"] = (np.log(ndi_merged2["H_mean"])
                                      + alpha * np.log(ndi_merged2["fndds_ndi"].clip(lower=0.1))
                                      - np.log(ndi_merged2["E_composite"]))

    dein_corr = sp_stats.spearmanr(ndi_merged2["log_DEIN_ours"],
                                    ndi_merged2["log_DEIN_fndds"])
    dei_corr = sp_stats.spearmanr(ndi_merged2["log_DEI"],
                                   ndi_merged2["log_DEIN_fndds"])
    print(f"  DEI-N ranking comparison ({len(ndi_merged2)} dishes):")
    print(f"    Our DEI-N vs FNDDS DEI-N: rho = {dein_corr[0]:.3f}")
    print(f"    Original DEI vs FNDDS DEI-N: rho = {dei_corr[0]:.3f}")

# ══════════════════════════════════════════════════════════════════
# E. WWEIA CATEGORY NUTRIENT PROFILES
# ══════════════════════════════════════════════════════════════════
print("\n── E. WWEIA Category Nutrient Profiles ──")

# For each matched dish, show its WWEIA category and how it compares
# to the category average
cat_profiles = comp_df.groupby("meal_role").agg(
    n=("dish_id", "count"),
    mean_cal=("fndds_cal_per_100g", "mean"),
    mean_prot=("fndds_protein_per_100g", "mean"),
    mean_fat=("fndds_fat_per_100g", "mean"),
    mean_fiber=("fndds_fiber_per_100g", "mean"),
).reset_index()
print(f"\n  FNDDS nutrient profile by meal role (per 100g):")
for _, row in cat_profiles.iterrows():
    print(f"    {row['meal_role']} (n={row['n']:.0f}): "
          f"cal={row['mean_cal']:.0f}, prot={row['mean_prot']:.1f}g, "
          f"fat={row['mean_fat']:.1f}g, fiber={row['mean_fiber']:.1f}g")

# ══════════════════════════════════════════════════════════════════
# F. SAVE & PLOT
# ══════════════════════════════════════════════════════════════════
print("\n── F. Save Results ──")

comp_df.to_csv(TABLES_DIR / "fndds_nutrient_validation.csv", index=False)

# Summary table
summary = {
    "n_matched_dishes": len(matches),
    "n_total_dishes": len(combined),
    "pct_matched": len(matches) / len(combined) * 100,
    "protein_pearson_r": prot_corr[0] if len(valid_prot) > 5 else np.nan,
    "protein_p_value": prot_corr[1] if len(valid_prot) > 5 else np.nan,
    "protein_mean_ratio": prot_ratio.mean() if len(valid_prot) > 5 else np.nan,
}
if "NDI" in combined.columns and len(ndi_df) > 5:
    summary["ndi_spearman_rho"] = ndi_spearman[0]
    summary["ndi_spearman_p"] = ndi_spearman[1]
    summary["dein_ranking_rho"] = dein_corr[0]

summary_df = pd.DataFrame([summary])
summary_df.to_csv(TABLES_DIR / "fndds_validation_summary.csv", index=False)

# Plot: protein comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Protein scatter
ax = axes[0]
if len(valid_prot) > 5:
    ax.scatter(valid_prot["fndds_prot_scaled"], valid_prot["our_protein_per_serving"],
               alpha=0.5, s=20, color="steelblue")
    max_val = max(valid_prot["fndds_prot_scaled"].max(),
                  valid_prot["our_protein_per_serving"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("FNDDS protein (g/serving)")
    ax.set_ylabel("Our protein estimate (g/serving)")
    ax.set_title(f"Protein: r = {prot_corr[0]:.3f}")
    ax.legend()

# 2. NDI comparison
ax = axes[1]
if "NDI" in combined.columns and len(ndi_merged) > 5:
    ax.scatter(ndi_merged["fndds_ndi"], ndi_merged["NDI"],
               alpha=0.5, s=20, color="coral")
    ax.set_xlabel("FNDDS-derived NDI")
    ax.set_ylabel("Our NDI")
    ax.set_title(f"NDI: rho = {ndi_spearman[0]:.3f}")

# 3. Calorie density by meal role
ax = axes[2]
roles = ["Side/Snack", "Light Main", "Full Main", "Heavy Main"]
role_cals = []
for role in roles:
    subset = comp_df[comp_df["meal_role"] == role]["fndds_cal_per_100g"].dropna()
    if len(subset) > 0:
        role_cals.append(subset.values)
    else:
        role_cals.append([])
positions = range(len(roles))
bp = ax.boxplot([rc for rc in role_cals if len(rc) > 0],
                positions=[i for i, rc in enumerate(role_cals) if len(rc) > 0],
                patch_artist=True, widths=0.6)
for patch in bp["boxes"]:
    patch.set_facecolor("lightblue")
    patch.set_alpha(0.7)
valid_roles = [roles[i] for i, rc in enumerate(role_cals) if len(rc) > 0]
ax.set_xticks(range(len(valid_roles)))
ax.set_xticklabels(valid_roles, rotation=20)
ax.set_ylabel("FNDDS cal/100g")
ax.set_title("Calorie density by meal role")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fndds_calorie_validation.png", dpi=150)
plt.close()
print("  Saved fndds_calorie_validation.png")

print(f"\n  Key validation results:")
print(f"    - {len(matches)} dishes matched to FNDDS ({len(matches)/len(combined)*100:.0f}%)")
if len(valid_prot) > 5:
    print(f"    - Protein correlation: r = {prot_corr[0]:.3f} (p = {prot_corr[1]:.2e})")
if "NDI" in combined.columns and len(ndi_merged) > 5:
    print(f"    - NDI correlation: rho = {ndi_spearman[0]:.3f}")
    print(f"    - DEI-N ranking stability: rho = {dein_corr[0]:.3f}")

print("\n" + "=" * 70)
print("15b COMPLETE")
print("=" * 70)
