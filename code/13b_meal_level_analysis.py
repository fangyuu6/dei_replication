"""
13b_meal_level_analysis.py — Meal-Level DEI Analysis
=====================================================
Addresses C7: Comparing kimchi (side) to brisket (main) is apples vs oranges.

Analyses:
  A. Within-role variance decomposition
  B. Meal-level DEI (main + side + drink combos)
  C. Calorie-equivalent substitution
  D. Reframed headline numbers

Outputs:
  - tables/within_role_variance.csv
  - tables/meal_level_dei.csv
  - tables/calorie_equivalent_subs.csv
  - figures/meal_dei_distribution.png
  - figures/like_for_like_comparison.png
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from itertools import product
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

print("=" * 70)
print("13b  MEAL-LEVEL DEI ANALYSIS")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────
combined = pd.read_csv(DATA_DIR / "combined_dish_DEI_revised.csv")
print(f"Dataset: {len(combined)} dishes")
print(f"Meal roles: {combined['meal_role'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════
# A. WITHIN-ROLE VARIANCE DECOMPOSITION
# ══════════════════════════════════════════════════════════════════
print("\n── A. Within-Role Variance Decomposition ──")

role_results = []
for role in combined["meal_role"].dropna().unique():
    subset = combined[combined["meal_role"] == role]
    if len(subset) < 5:
        continue

    log_h = np.log(subset["H_mean"].values)
    log_e = subset["log_E"].values

    var_lh = np.var(log_h)
    var_le = np.var(log_e)
    cov_lhe = np.cov(log_h, log_e)[0, 1]
    var_ldei = var_lh + var_le - 2 * cov_lhe

    h_pct = var_lh / var_ldei * 100 if var_ldei > 0 else 0
    e_pct = var_le / var_ldei * 100 if var_ldei > 0 else 0

    cv_h = subset["H_mean"].std() / subset["H_mean"].mean() * 100
    cv_e = subset["E_composite"].std() / subset["E_composite"].mean() * 100
    cv_ratio = cv_e / cv_h if cv_h > 0 else np.inf

    # Within-role DEI range
    dei_range = subset["log_DEI"].max() - subset["log_DEI"].min()
    dei_ratio = np.exp(dei_range)

    role_results.append({
        "meal_role": role,
        "n_dishes": len(subset),
        "H_cv_pct": cv_h,
        "E_cv_pct": cv_e,
        "cv_ratio_E_over_H": cv_ratio,
        "H_contribution_pct": h_pct,
        "E_contribution_pct": e_pct,
        "mean_cal": subset["calorie_kcal"].mean(),
        "mean_protein": subset["protein_g"].mean(),
        "log_DEI_range": dei_range,
        "DEI_fold_difference": dei_ratio,
        "top_dish": subset.loc[subset["log_DEI"].idxmax(), "dish_id"],
        "bottom_dish": subset.loc[subset["log_DEI"].idxmin(), "dish_id"],
    })

role_df = pd.DataFrame(role_results).sort_values("n_dishes", ascending=False)
role_df.to_csv(TABLES_DIR / "within_role_variance.csv", index=False)

print("\nWithin-role variance decomposition:")
for _, row in role_df.iterrows():
    print(f"  {row['meal_role']} (n={row['n_dishes']}): "
          f"H={row['H_contribution_pct']:.1f}%, E={row['E_contribution_pct']:.1f}%, "
          f"CV ratio={row['cv_ratio_E_over_H']:.1f}x, "
          f"DEI range={row['DEI_fold_difference']:.0f}x")

# ══════════════════════════════════════════════════════════════════
# B. MEAL-LEVEL DEI
# ══════════════════════════════════════════════════════════════════
print("\n── B. Meal-Level DEI ──")

# Define meal components
mains = combined[combined["meal_role"].isin(["Full Main", "Heavy Main"])].copy()
sides = combined[combined["meal_role"].isin(["Side/Snack", "Light Main"])].copy()
drinks = combined[combined["meal_role"] == "Side/Snack"].copy()  # use sides as proxy

# If not enough sides/drinks, also include Light Main
if len(sides) < 10:
    sides = combined[combined["calorie_kcal"] < 400].copy()

print(f"  Mains: {len(mains)}, Sides: {len(sides)}")

# Sample meal combos: main + side
meal_combos = []
for _, main in mains.iterrows():
    for _, side in sides.iterrows():
        if main["dish_id"] == side["dish_id"]:
            continue
        total_cal = main["calorie_kcal"] + side["calorie_kcal"]
        if total_cal < 500 or total_cal > 1500:
            continue
        total_protein = main["protein_g"] + side["protein_g"]
        if total_protein < 20:
            continue

        # Meal H = calorie-weighted average
        w_main = main["calorie_kcal"] / total_cal
        w_side = side["calorie_kcal"] / total_cal
        H_meal = w_main * main["H_mean"] + w_side * side["H_mean"]
        E_meal = main["E_composite"] + side["E_composite"]

        if E_meal <= 0 or H_meal <= 0:
            continue

        log_DEI_meal = np.log(H_meal) - np.log(E_meal)

        meal_combos.append({
            "main": main["dish_id"],
            "side": side["dish_id"],
            "H_meal": H_meal,
            "E_meal": E_meal,
            "log_DEI_meal": log_DEI_meal,
            "total_cal": total_cal,
            "total_protein": total_protein,
            "main_cuisine": main.get("cuisine", ""),
            "side_cuisine": side.get("cuisine", ""),
        })

meals_df = pd.DataFrame(meal_combos)
print(f"  Total meal combos: {len(meals_df):,}")

if len(meals_df) > 0:
    # Variance decomposition at meal level
    log_H_meal = np.log(meals_df["H_meal"].values)
    log_E_meal = np.log(meals_df["E_meal"].values)
    var_lh_meal = np.var(log_H_meal)
    var_le_meal = np.var(log_E_meal)
    cov_meal = np.cov(log_H_meal, log_E_meal)[0, 1]
    var_ldei_meal = var_lh_meal + var_le_meal - 2 * cov_meal
    h_pct_meal = var_lh_meal / var_ldei_meal * 100

    print(f"  Meal-level Var(log H) = {var_lh_meal:.4f} ({h_pct_meal:.1f}%)")
    print(f"  Meal-level Var(log E) = {var_le_meal:.4f}")
    print(f"  Meal-level Var(log DEI) = {var_ldei_meal:.4f}")

    # Top/bottom meals
    meals_sorted = meals_df.sort_values("log_DEI_meal", ascending=False)
    print(f"\n  Top 5 meals:")
    for _, row in meals_sorted.head(5).iterrows():
        print(f"    {row['main']} + {row['side']}: DEI={row['log_DEI_meal']:.2f}, "
              f"cal={row['total_cal']:.0f}, protein={row['total_protein']:.0f}g")
    print(f"  Bottom 5 meals:")
    for _, row in meals_sorted.tail(5).iterrows():
        print(f"    {row['main']} + {row['side']}: DEI={row['log_DEI_meal']:.2f}, "
              f"cal={row['total_cal']:.0f}, protein={row['total_protein']:.0f}g")

    # Save top/bottom 50
    top_bottom = pd.concat([meals_sorted.head(50), meals_sorted.tail(50)])
    top_bottom.to_csv(TABLES_DIR / "meal_level_dei.csv", index=False)

    # DEI range at meal level
    meal_dei_range = meals_sorted["log_DEI_meal"].max() - meals_sorted["log_DEI_meal"].min()
    print(f"\n  Meal-level DEI range: {meal_dei_range:.2f} log-units "
          f"({np.exp(meal_dei_range):.0f}x)")

# ══════════════════════════════════════════════════════════════════
# C. CALORIE-EQUIVALENT SUBSTITUTION
# ══════════════════════════════════════════════════════════════════
print("\n── C. Calorie-Equivalent Substitution ──")

# Within each meal_role, find calorie-matched pairs
cal_subs = []
for role in ["Full Main", "Heavy Main", "Light Main"]:
    role_dishes = combined[combined["meal_role"] == role].copy()
    if len(role_dishes) < 5:
        continue

    for i, d1 in role_dishes.iterrows():
        for j, d2 in role_dishes.iterrows():
            if i >= j:
                continue
            # Calorie match: within ±25%
            cal_ratio = d1["calorie_kcal"] / d2["calorie_kcal"]
            if cal_ratio < 0.75 or cal_ratio > 1.33:
                continue
            # Protein match: within ±50%
            prot_ratio = d1["protein_g"] / d2["protein_g"]
            if prot_ratio < 0.5 or prot_ratio > 2.0:
                continue

            e_diff_pct = (d1["E_composite"] - d2["E_composite"]) / max(
                d1["E_composite"], d2["E_composite"]
            ) * 100
            h_diff = d1["H_mean"] - d2["H_mean"]

            if abs(e_diff_pct) < 20:
                continue  # skip near-identical E

            # Assign direction: from high-E to low-E
            if d1["E_composite"] > d2["E_composite"]:
                cal_subs.append({
                    "from_dish": d1["dish_id"], "to_dish": d2["dish_id"],
                    "meal_role": role,
                    "E_reduction_pct": abs(e_diff_pct),
                    "H_change": d2["H_mean"] - d1["H_mean"],
                    "cal_from": d1["calorie_kcal"], "cal_to": d2["calorie_kcal"],
                    "protein_from": d1["protein_g"], "protein_to": d2["protein_g"],
                })
            else:
                cal_subs.append({
                    "from_dish": d2["dish_id"], "to_dish": d1["dish_id"],
                    "meal_role": role,
                    "E_reduction_pct": abs(e_diff_pct),
                    "H_change": d1["H_mean"] - d2["H_mean"],
                    "cal_from": d2["calorie_kcal"], "cal_to": d1["calorie_kcal"],
                    "protein_from": d2["protein_g"], "protein_to": d1["protein_g"],
                })

cal_subs_df = pd.DataFrame(cal_subs)
cal_subs_df = cal_subs_df.sort_values("E_reduction_pct", ascending=False)
cal_subs_df.to_csv(TABLES_DIR / "calorie_equivalent_subs.csv", index=False)

print(f"  Calorie-equivalent substitutions: {len(cal_subs_df):,}")
print(f"  Mean E reduction: {cal_subs_df['E_reduction_pct'].mean():.1f}%")
print(f"  Mean H change: {cal_subs_df['H_change'].mean():.3f}")

if len(cal_subs_df) > 0:
    print(f"\n  Top 5 calorie-equivalent swaps:")
    for _, row in cal_subs_df.head(5).iterrows():
        print(f"    {row['from_dish']} → {row['to_dish']} ({row['meal_role']}): "
              f"E↓{row['E_reduction_pct']:.0f}%, H Δ={row['H_change']:+.2f}, "
              f"cal {row['cal_from']:.0f}→{row['cal_to']:.0f}")

# ══════════════════════════════════════════════════════════════════
# D. REFRAMED HEADLINE NUMBERS
# ══════════════════════════════════════════════════════════════════
print("\n── D. Reframed Headlines ──")

# Within Full Main only
full_mains = combined[combined["meal_role"] == "Full Main"]
if len(full_mains) > 5:
    top_main = full_mains.loc[full_mains["log_DEI"].idxmax()]
    bot_main = full_mains.loc[full_mains["log_DEI"].idxmin()]
    main_dei_ratio = np.exp(top_main["log_DEI"] - bot_main["log_DEI"])
    print(f"  Within Full Mains:")
    print(f"    Best: {top_main['dish_id']} (DEI={top_main['log_DEI']:.2f})")
    print(f"    Worst: {bot_main['dish_id']} (DEI={bot_main['log_DEI']:.2f})")
    print(f"    Ratio: {main_dei_ratio:.0f}x (vs 345x global)")
    print(f"    H difference: {abs(top_main['H_mean'] - bot_main['H_mean']):.2f}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: within-role variance
ax = axes[0]
roles_sorted = role_df.sort_values("H_contribution_pct")
y_pos = range(len(roles_sorted))
ax.barh(y_pos, roles_sorted["E_contribution_pct"].values, color="coral",
        alpha=0.7, label="E contribution")
ax.barh(y_pos, roles_sorted["H_contribution_pct"].values, color="steelblue",
        alpha=0.7, label="H contribution")
ax.set_yticks(y_pos)
ax.set_yticklabels(roles_sorted["meal_role"].values)
ax.set_xlabel("% of Var(log DEI)")
ax.set_title("Variance decomposition by meal role")
ax.legend()

# Right: like-for-like DEI distributions
ax = axes[1]
role_order = ["Side/Snack", "Light Main", "Full Main", "Heavy Main"]
positions = []
for i, role in enumerate(role_order):
    subset = combined[combined["meal_role"] == role]["log_DEI"]
    if len(subset) > 3:
        bp = ax.boxplot(subset.values, positions=[i], widths=0.6,
                        patch_artist=True,
                        medianprops=dict(color="red"))
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        positions.append(i)
ax.set_xticks(range(len(role_order)))
ax.set_xticklabels(role_order, rotation=30)
ax.set_ylabel("log(DEI)")
ax.set_title("DEI distribution by meal role")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "like_for_like_comparison.png", dpi=150)
plt.close()
print("  Saved like_for_like_comparison.png")

# Meal distribution plot
if len(meals_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(meals_df["log_DEI_meal"], bins=50, color="steelblue", alpha=0.7,
            edgecolor="white")
    ax.axvline(meals_df["log_DEI_meal"].median(), color="red", linestyle="--",
               label=f"Median={meals_df['log_DEI_meal'].median():.2f}")
    ax.set_xlabel("log(DEI) at meal level")
    ax.set_ylabel("Count")
    ax.set_title(f"Meal-level DEI distribution ({len(meals_df):,} combinations)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "meal_dei_distribution.png", dpi=150)
    plt.close()
    print("  Saved meal_dei_distribution.png")

print("\n" + "=" * 70)
print("13b COMPLETE")
print("=" * 70)
