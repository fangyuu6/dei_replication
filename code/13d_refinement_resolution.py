"""
13d_refinement_resolution.py — Refinement Curve Resolution Validation
======================================================================
Addresses C9: The flat refinement curve (α≈0) might be an artifact of
BERT's inability to discriminate fine hedonic differences.

Analyses:
  A. Minimum Detectable Difference (MDD) per dish pair
  B. Cohen's d effect sizes within families
  C. Cross-platform refinement validation
  D. Price-tier stratified H analysis

Outputs:
  - tables/within_family_mdd.csv
  - tables/cross_platform_refinement.csv
  - tables/price_tier_h_analysis.csv
  - figures/refinement_resolution_diagnostic.png
"""

import sys, warnings, json
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
print("13d  REFINEMENT CURVE RESOLUTION VALIDATION")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
combined = pd.read_csv(DATA_DIR / "combined_dish_DEI_revised.csv")
families = pd.read_csv(TABLES_DIR / "dish_families.csv")
restaurants = pd.read_parquet(DATA_DIR / "restaurants.parquet")

try:
    cross_plat = pd.read_csv(DATA_DIR / "cross_platform_h_scores.csv")
    has_cross_plat = True
except FileNotFoundError:
    has_cross_plat = False

# Merge family info
combined = combined.merge(families, on="dish_id", how="left",
                          suffixes=("", "_fam"))
# Handle duplicate family columns
if "family_fam" in combined.columns:
    combined["family"] = combined["family_fam"].fillna(combined.get("family", ""))
    combined.drop(columns=["family_fam"], inplace=True, errors="ignore")

print(f"Mentions: {len(mentions):,}")
print(f"Dishes with family: {combined['family'].notna().sum()}")

# ══════════════════════════════════════════════════════════════════
# A. MINIMUM DETECTABLE DIFFERENCE (MDD)
# ══════════════════════════════════════════════════════════════════
print("\n── A. Minimum Detectable Difference ──")

# Compute per-dish stats
dish_mention_stats = mentions.groupby("dish_id").agg(
    n=("hedonic_score_finetuned", "count"),
    mean=("hedonic_score_finetuned", "mean"),
    std=("hedonic_score_finetuned", "std"),
).reset_index()
dish_mention_stats["se"] = dish_mention_stats["std"] / np.sqrt(dish_mention_stats["n"])
# MDD at 95% confidence (two-sided)
dish_mention_stats["mdd_95"] = 1.96 * dish_mention_stats["se"] * np.sqrt(2)

print(f"  Mean within-dish SD: {dish_mention_stats['std'].mean():.3f}")
print(f"  Median MDD (95%): {dish_mention_stats['mdd_95'].median():.3f}")
print(f"  Mean MDD (95%): {dish_mention_stats['mdd_95'].mean():.3f}")

# Within-family pairwise tests
family_pairs = []
for fam in combined["family"].dropna().unique():
    fam_dishes = combined[combined["family"] == fam]["dish_id"].tolist()
    fam_mentions = mentions[mentions["dish_id"].isin(fam_dishes)]

    for i, d1 in enumerate(fam_dishes):
        for d2 in fam_dishes[i + 1:]:
            h1 = fam_mentions[fam_mentions["dish_id"] == d1]["hedonic_score_finetuned"]
            h2 = fam_mentions[fam_mentions["dish_id"] == d2]["hedonic_score_finetuned"]

            if len(h1) < 10 or len(h2) < 10:
                continue

            # t-test
            t_stat, p_val = sp_stats.ttest_ind(h1, h2, equal_var=False)
            # Cohen's d
            pooled_std = np.sqrt((h1.var() + h2.var()) / 2)
            cohens_d = abs(h1.mean() - h2.mean()) / pooled_std if pooled_std > 0 else 0

            # Effect size category
            if cohens_d < 0.2:
                effect_cat = "negligible"
            elif cohens_d < 0.5:
                effect_cat = "small"
            elif cohens_d < 0.8:
                effect_cat = "medium"
            else:
                effect_cat = "large"

            family_pairs.append({
                "family": fam,
                "dish_1": d1, "dish_2": d2,
                "H_1": h1.mean(), "H_2": h2.mean(),
                "H_diff": abs(h1.mean() - h2.mean()),
                "n_1": len(h1), "n_2": len(h2),
                "t_stat": abs(t_stat), "p_value": p_val,
                "cohens_d": cohens_d, "effect_category": effect_cat,
                "significant_005": p_val < 0.05,
                "significant_bonf": p_val < 0.05 / max(1, len(fam_dishes) * (len(fam_dishes) - 1) / 2),
            })

pairs_df = pd.DataFrame(family_pairs)
if len(pairs_df) > 0:
    pairs_df.to_csv(TABLES_DIR / "within_family_mdd.csv", index=False)

    # Summary
    n_total = len(pairs_df)
    n_sig = pairs_df["significant_005"].sum()
    n_sig_bonf = pairs_df["significant_bonf"].sum()
    n_negl = (pairs_df["effect_category"] == "negligible").sum()
    n_small = (pairs_df["effect_category"] == "small").sum()
    n_med = (pairs_df["effect_category"] == "medium").sum()
    n_large = (pairs_df["effect_category"] == "large").sum()

    print(f"\n  Total within-family pairs: {n_total}")
    print(f"  Significant (p<0.05): {n_sig} ({n_sig/n_total*100:.0f}%)")
    print(f"  Significant (Bonferroni): {n_sig_bonf} ({n_sig_bonf/n_total*100:.0f}%)")
    print(f"  Effect sizes: negligible={n_negl} ({n_negl/n_total*100:.0f}%), "
          f"small={n_small} ({n_small/n_total*100:.0f}%), "
          f"medium={n_med} ({n_med/n_total*100:.0f}%), "
          f"large={n_large} ({n_large/n_total*100:.0f}%)")
    print(f"  Mean Cohen's d: {pairs_df['cohens_d'].mean():.3f}")
    print(f"  Mean H difference: {pairs_df['H_diff'].mean():.3f}")

    # Key finding: are significant pairs showing α>0?
    sig_pairs = pairs_df[pairs_df["significant_005"]]
    if len(sig_pairs) > 0:
        # Check if higher-E dish has higher H among significant pairs
        sig_with_e = sig_pairs.merge(
            combined[["dish_id", "E_composite"]].rename(columns={"dish_id": "dish_1", "E_composite": "E_1"}),
            on="dish_1", how="left"
        ).merge(
            combined[["dish_id", "E_composite"]].rename(columns={"dish_id": "dish_2", "E_composite": "E_2"}),
            on="dish_2", how="left"
        )
        # Does higher E dish have higher H?
        sig_with_e["higher_E_higher_H"] = (
            ((sig_with_e["E_1"] > sig_with_e["E_2"]) & (sig_with_e["H_1"] > sig_with_e["H_2"])) |
            ((sig_with_e["E_2"] > sig_with_e["E_1"]) & (sig_with_e["H_2"] > sig_with_e["H_1"]))
        )
        pct_refinement = sig_with_e["higher_E_higher_H"].mean() * 100
        print(f"\n  Among significant pairs: {pct_refinement:.0f}% show higher E → higher H")
        print(f"  (50% expected by chance; >50% supports refinement, <50% contradicts it)")

# ══════════════════════════════════════════════════════════════════
# B. CROSS-PLATFORM REFINEMENT VALIDATION
# ══════════════════════════════════════════════════════════════════
print("\n── B. Cross-Platform Refinement Validation ──")

if has_cross_plat:
    # For dishes with family assignments AND cross-platform data
    cross_fam = cross_plat.merge(
        combined[["dish_id", "family", "E_composite"]],
        on="dish_id", how="inner"
    )
    cross_fam = cross_fam[cross_fam["family"].notna()]

    platforms = {"Yelp": "H_yelp", "Google": "H_google", "TripAdvisor": "H_tripadvisor"}
    platform_results = []

    for fam in cross_fam["family"].unique():
        fam_data = cross_fam[cross_fam["family"] == fam]
        if len(fam_data) < 3:
            continue

        for plat_name, col in platforms.items():
            valid = fam_data[fam_data[col].notna()]
            if len(valid) < 3:
                continue

            # Fit refinement curve for this platform
            E_base = valid["E_composite"].min()
            if E_base <= 0:
                continue
            log_E_ratio = np.log(valid["E_composite"] / E_base)
            H_base = valid.loc[valid["E_composite"].idxmin(), col]
            H_diff = valid[col] - H_base

            if log_E_ratio.std() < 0.01:
                continue

            slope, intercept, r_val, p_val, se = sp_stats.linregress(log_E_ratio, H_diff)

            platform_results.append({
                "family": fam,
                "platform": plat_name,
                "n_dishes": len(valid),
                "alpha": slope,
                "r_squared": r_val ** 2,
                "p_value": p_val,
            })

    plat_df = pd.DataFrame(platform_results)
    if len(plat_df) > 0:
        plat_df.to_csv(TABLES_DIR / "cross_platform_refinement.csv", index=False)

        # Summary by platform
        plat_summary = plat_df.groupby("platform").agg(
            n_families=("family", "count"),
            mean_alpha=("alpha", "mean"),
            median_alpha=("alpha", "median"),
            pct_positive=("alpha", lambda x: (x > 0).mean() * 100),
        )
        print(plat_summary.to_string())
    else:
        print("  Insufficient cross-platform data for family-level analysis")
else:
    print("  No cross-platform data file found")

# ══════════════════════════════════════════════════════════════════
# C. PRICE-TIER STRATIFIED H ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n── C. Price-Tier Stratified H Analysis ──")

# Extract price range from restaurant attributes (dict or string)
def extract_price(attrs):
    if isinstance(attrs, dict):
        pr = attrs.get("RestaurantsPriceRange2")
        if pr is not None:
            try:
                return int(pr)
            except (ValueError, TypeError):
                return np.nan
        return np.nan
    if not isinstance(attrs, str):
        return np.nan
    import re
    m = re.search(r"RestaurantsPriceRange2.*?(\d)", str(attrs))
    if m:
        return int(m.group(1))
    return np.nan

restaurants["price_range"] = restaurants["attributes"].apply(extract_price)
n_priced = restaurants["price_range"].notna().sum()
print(f"  Restaurants with price range: {n_priced}/{len(restaurants)}")

# Merge with mentions
mentions_priced = mentions.merge(
    restaurants[["business_id", "price_range"]],
    on="business_id", how="left"
)
mentions_priced = mentions_priced[mentions_priced["price_range"].notna()]
print(f"  Mentions with price data: {len(mentions_priced):,}")

# For each dish, compute H by price tier
price_results = []
for dish_id in mentions_priced["dish_id"].unique():
    dish_m = mentions_priced[mentions_priced["dish_id"] == dish_id]
    for tier in sorted(dish_m["price_range"].unique()):
        tier_m = dish_m[dish_m["price_range"] == tier]
        if len(tier_m) < 5:
            continue
        price_results.append({
            "dish_id": dish_id,
            "price_tier": int(tier),
            "n_mentions": len(tier_m),
            "H_mean": tier_m["hedonic_score_finetuned"].mean(),
            "H_std": tier_m["hedonic_score_finetuned"].std(),
        })

price_df = pd.DataFrame(price_results)
if len(price_df) > 0:
    # Aggregate: does H increase with price tier?
    tier_summary = price_df.groupby("price_tier").agg(
        n_dishes=("dish_id", "nunique"),
        n_mentions=("n_mentions", "sum"),
        H_mean=("H_mean", "mean"),
        H_std=("H_mean", "std"),
    )
    print("\n  H by restaurant price tier:")
    print(tier_summary.to_string())

    # Regression: H ~ price_tier
    slope, intercept, r, p, se = sp_stats.linregress(
        price_df["price_tier"], price_df["H_mean"]
    )
    print(f"\n  H ~ price_tier: slope={slope:.4f}, r²={r**2:.4f}, p={p:.4e}")
    print(f"  Interpretation: {'Higher price → higher H' if slope > 0 else 'No price → H relationship'}")

    # Within-dish price effect
    dish_price_slopes = []
    for dish_id in price_df["dish_id"].unique():
        d = price_df[price_df["dish_id"] == dish_id]
        if len(d) < 2 or d["price_tier"].nunique() < 2:
            continue
        s, _, r, p, _ = sp_stats.linregress(d["price_tier"], d["H_mean"])
        dish_price_slopes.append({"dish_id": dish_id, "slope": s, "r": r, "p": p})

    if dish_price_slopes:
        slopes_df = pd.DataFrame(dish_price_slopes)
        print(f"\n  Within-dish price slopes: n={len(slopes_df)}")
        print(f"    Mean slope: {slopes_df['slope'].mean():.4f}")
        print(f"    % positive: {(slopes_df['slope'] > 0).mean()*100:.0f}%")
        print(f"    % significant (p<0.05): {(slopes_df['p'] < 0.05).mean()*100:.0f}%")

    price_df.to_csv(TABLES_DIR / "price_tier_h_analysis.csv", index=False)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOT
# ══════════════════════════════════════════════════════════════════
print("\n── Generating plots ──")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (0,0): Cohen's d distribution
if len(pairs_df) > 0:
    ax = axes[0, 0]
    ax.hist(pairs_df["cohens_d"], bins=30, color="steelblue", alpha=0.7,
            edgecolor="white")
    for thresh, label in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
        ax.axvline(thresh, color="red", linestyle="--", alpha=0.5)
        ax.text(thresh + 0.02, ax.get_ylim()[1] * 0.9, label, fontsize=8, color="red")
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("Count")
    ax.set_title("Effect sizes within dish families")

# (0,1): Significant vs non-significant pairs H diff
if len(pairs_df) > 0:
    ax = axes[0, 1]
    sig = pairs_df[pairs_df["significant_005"]]["H_diff"]
    nsig = pairs_df[~pairs_df["significant_005"]]["H_diff"]
    ax.boxplot([sig.values, nsig.values], labels=["Significant", "Not significant"])
    ax.set_ylabel("|H difference|")
    ax.set_title(f"H differences: sig ({len(sig)}) vs non-sig ({len(nsig)})")

# (1,0): Price tier H
if len(price_df) > 0:
    ax = axes[1, 0]
    for tier in sorted(price_df["price_tier"].unique()):
        subset = price_df[price_df["price_tier"] == tier]["H_mean"]
        ax.boxplot(subset.values, positions=[int(tier)], widths=0.5,
                   boxprops=dict(color="steelblue"),
                   medianprops=dict(color="red"))
    ax.set_xlabel("Restaurant price tier ($-$$$$)")
    ax.set_ylabel("Dish-level H")
    ax.set_title("H by restaurant price tier")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["1", "2", "3", "4"])

# (1,1): MDD distribution
ax = axes[1, 1]
ax.hist(dish_mention_stats["mdd_95"].dropna(), bins=30,
        color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(dish_mention_stats["mdd_95"].median(), color="red", linestyle="--",
           label=f"Median MDD={dish_mention_stats['mdd_95'].median():.3f}")
ax.set_xlabel("Minimum Detectable Difference (95%)")
ax.set_ylabel("Count")
ax.set_title("BERT score discrimination threshold per dish")
ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / "refinement_resolution_diagnostic.png", dpi=150)
plt.close()
print("  Saved refinement_resolution_diagnostic.png")

print("\n" + "=" * 70)
print("13d COMPLETE")
print("=" * 70)
