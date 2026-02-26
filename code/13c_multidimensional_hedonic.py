"""
13c_multidimensional_hedonic.py — Multi-dimensional Hedonic Analysis
=====================================================================
Addresses C8: "Deliciousness" is oversimplified. Missing satiety,
satisfaction, comfort, and social context dimensions.

Analyses:
  A. Satiety signal extraction from review text
  B. Comfort/emotional signal extraction
  C. Multi-dimensional H composite with weight sensitivity
  D. Scene-based analysis

Outputs:
  - tables/multidimensional_hedonic.csv
  - tables/satiety_by_dish.csv
  - figures/h_satiety_scatter.png
  - figures/dei_sensitivity_to_satiety.png
"""

import sys, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

print("=" * 70)
print("13c  MULTI-DIMENSIONAL HEDONIC ANALYSIS")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
combined = pd.read_csv(DATA_DIR / "combined_dish_DEI_revised.csv")
print(f"Mentions: {len(mentions):,}, Dishes: {len(combined)}")

# ══════════════════════════════════════════════════════════════════
# A. SATIETY SIGNAL EXTRACTION
# ══════════════════════════════════════════════════════════════════
print("\n── A. Satiety Signal Extraction ──")

# Define keyword lists
SATIETY_POS = [
    r"\bfilling\b", r"\bhearty\b", r"\bsatisfying\b", r"\bsubstantial\b",
    r"\bcomfort\b", r"\brich\b", r"\bheavy\b", r"\bgenerous portion",
    r"\bstuffed\b", r"\bfull\b(?!\s+of\s+flavor)", r"\bmeaty\b",
    r"\bsavory\b", r"\bindulgent\b", r"\bdecadent\b",
]
SATIETY_NEG = [
    r"\blight\b(?!\s+year)", r"\brefreshing\b", r"\bsmall portion",
    r"\bstill hungry\b", r"\btiny\b", r"\bsnack\b(?!\s+bar)",
    r"\bnot enough\b", r"\bwished.{0,20}more\b", r"\bcould eat more\b",
    r"\bdelicate\b", r"\bcrisp\b", r"\bclean\b(?!\s+up)",
]

COMFORT_WORDS = [
    r"\bcomfort food\b", r"\bcomforting\b", r"\bremind.{0,20}(?:mom|grandma|home|childhood)\b",
    r"\bnostalg\w*\b", r"\bsoul food\b", r"\bcraving\b",
    r"\bhit the spot\b", r"\bwarm.{0,10}(?:soul|heart|belly)\b",
]

OCCASION_WORDS = {
    "special": [r"\bdate night\b", r"\bspecial occasion\b", r"\bcelebrat\w*\b",
                r"\banniversary\b", r"\bbirthday\b", r"\bfine dining\b"],
    "everyday": [r"\bquick lunch\b", r"\beveryday\b", r"\bcasual\b",
                 r"\bgrab.{0,10}bite\b", r"\bweeknight\b", r"\beasy meal\b"],
}


def count_pattern_matches(text, patterns):
    """Count regex pattern matches in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text_lower))
    return count


# Score each mention
print("  Scoring mentions for satiety signals...")
mentions["sat_pos"] = mentions["context_text"].apply(
    lambda x: count_pattern_matches(x, SATIETY_POS)
)
mentions["sat_neg"] = mentions["context_text"].apply(
    lambda x: count_pattern_matches(x, SATIETY_NEG)
)
mentions["comfort"] = mentions["context_text"].apply(
    lambda x: count_pattern_matches(x, COMFORT_WORDS)
)
mentions["occasion_special"] = mentions["context_text"].apply(
    lambda x: count_pattern_matches(x, OCCASION_WORDS["special"])
)
mentions["occasion_everyday"] = mentions["context_text"].apply(
    lambda x: count_pattern_matches(x, OCCASION_WORDS["everyday"])
)

# Compute word count for normalization
mentions["word_count"] = mentions["context_text"].apply(
    lambda x: len(str(x).split()) if isinstance(x, str) else 0
)

# Satiety index: (pos - neg) / word_count * 100
mentions["satiety_raw"] = mentions["sat_pos"] - mentions["sat_neg"]
mentions["satiety_norm"] = np.where(
    mentions["word_count"] > 0,
    mentions["satiety_raw"] / mentions["word_count"] * 100,
    0
)

# Signal prevalence
total = len(mentions)
has_sat = (mentions["sat_pos"] > 0).sum()
has_neg = (mentions["sat_neg"] > 0).sum()
has_comfort = (mentions["comfort"] > 0).sum()
print(f"  Satiety positive signals: {has_sat:,} ({has_sat/total*100:.1f}%)")
print(f"  Satiety negative signals: {has_neg:,} ({has_neg/total*100:.1f}%)")
print(f"  Comfort signals: {has_comfort:,} ({has_comfort/total*100:.1f}%)")

# ── Aggregate to dish level ───────────────────────────────────────
dish_signals = mentions.groupby("dish_id").agg(
    n_mentions=("dish_id", "count"),
    H_taste=("hedonic_score_finetuned", "mean"),
    satiety_mean=("satiety_norm", "mean"),
    satiety_pos_rate=("sat_pos", lambda x: (x > 0).mean()),
    satiety_neg_rate=("sat_neg", lambda x: (x > 0).mean()),
    comfort_rate=("comfort", lambda x: (x > 0).mean()),
    occasion_special_rate=("occasion_special", lambda x: (x > 0).mean()),
    occasion_everyday_rate=("occasion_everyday", lambda x: (x > 0).mean()),
).reset_index()

# Merge with combined
dish_signals = dish_signals.merge(
    combined[["dish_id", "E_composite", "log_E", "log_DEI", "meal_role",
              "calorie_kcal", "protein_g", "cuisine", "category_recipe"]],
    on="dish_id", how="inner"
)

# Normalize satiety to 0-10 scale
sat_min = dish_signals["satiety_mean"].min()
sat_max = dish_signals["satiety_mean"].max()
dish_signals["H_satiety"] = 1 + 9 * (dish_signals["satiety_mean"] - sat_min) / (sat_max - sat_min)

dish_signals.to_csv(TABLES_DIR / "satiety_by_dish.csv", index=False)

print(f"\n  Dish-level satiety index range: [{dish_signals['H_satiety'].min():.2f}, "
      f"{dish_signals['H_satiety'].max():.2f}]")
print(f"  Satiety CV: {dish_signals['H_satiety'].std()/dish_signals['H_satiety'].mean()*100:.1f}%")

# Satiety by category
print("\n  Satiety by food category:")
cat_sat = dish_signals.groupby("category_recipe").agg(
    n=("dish_id", "count"),
    H_taste_mean=("H_taste", "mean"),
    H_satiety_mean=("H_satiety", "mean"),
    comfort_rate_mean=("comfort_rate", "mean"),
).sort_values("H_satiety_mean", ascending=False)
for _, row in cat_sat.head(8).iterrows():
    print(f"    {row.name} (n={row['n']}): "
          f"H_taste={row['H_taste_mean']:.2f}, H_sat={row['H_satiety_mean']:.2f}, "
          f"comfort={row['comfort_rate_mean']:.3f}")

# ══════════════════════════════════════════════════════════════════
# B. MULTI-DIMENSIONAL H COMPOSITE
# ══════════════════════════════════════════════════════════════════
print("\n── B. Multi-dimensional H Composite ──")

w2_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
sensitivity_results = []

for w2 in w2_values:
    w1 = 1 - w2
    dish_signals["H_composite"] = w1 * dish_signals["H_taste"] + w2 * dish_signals["H_satiety"]

    log_H_comp = np.log(dish_signals["H_composite"].values)
    log_E = dish_signals["log_E"].values

    var_lh = np.var(log_H_comp)
    var_le = np.var(log_E)
    cov_lhe = np.cov(log_H_comp, log_E)[0, 1]
    var_ldei = var_lh + var_le - 2 * cov_lhe
    h_pct = var_lh / var_ldei * 100 if var_ldei > 0 else 0

    # DEI with composite H
    log_dei_comp = log_H_comp - log_E
    rho_vs_orig = sp_stats.spearmanr(dish_signals["log_DEI"], log_dei_comp)[0]

    # H composite CV
    h_cv = dish_signals["H_composite"].std() / dish_signals["H_composite"].mean() * 100

    sensitivity_results.append({
        "w_satiety": w2,
        "H_composite_cv_pct": h_cv,
        "H_contribution_pct": h_pct,
        "rank_rho_vs_original": rho_vs_orig,
        "H_composite_range": dish_signals["H_composite"].max() - dish_signals["H_composite"].min(),
    })

sens_df = pd.DataFrame(sensitivity_results)
print(sens_df.to_string(index=False, float_format="%.3f"))

# ══════════════════════════════════════════════════════════════════
# C. COMBINED OUTPUT
# ══════════════════════════════════════════════════════════════════
print("\n── C. Combined Output ──")

# Best composite: w2=0.3
w2_best = 0.3
dish_signals["H_composite_best"] = (1 - w2_best) * dish_signals["H_taste"] + w2_best * dish_signals["H_satiety"]
dish_signals["log_DEI_composite"] = np.log(dish_signals["H_composite_best"]) - dish_signals["log_E"]
dish_signals["rank_original"] = dish_signals["log_DEI"].rank(ascending=False)
dish_signals["rank_composite"] = dish_signals["log_DEI_composite"].rank(ascending=False)
dish_signals["rank_shift_composite"] = dish_signals["rank_original"] - dish_signals["rank_composite"]

# Biggest rank gainers (high satiety dishes gaining)
gainers = dish_signals.nlargest(10, "rank_shift_composite")
losers = dish_signals.nsmallest(10, "rank_shift_composite")

print("  Top rank gainers (high satiety advantage):")
for _, row in gainers.iterrows():
    print(f"    {row['dish_id']}: rank {row['rank_original']:.0f}→{row['rank_composite']:.0f} "
          f"(+{row['rank_shift_composite']:.0f}), "
          f"sat={row['H_satiety']:.1f}, taste={row['H_taste']:.2f}")

print("  Top rank losers (low satiety penalty):")
for _, row in losers.iterrows():
    print(f"    {row['dish_id']}: rank {row['rank_original']:.0f}→{row['rank_composite']:.0f} "
          f"({row['rank_shift_composite']:.0f}), "
          f"sat={row['H_satiety']:.1f}, taste={row['H_taste']:.2f}")

dish_signals.to_csv(TABLES_DIR / "multidimensional_hedonic.csv", index=False)

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n── Generating plots ──")

# Plot 1: H_taste vs H_satiety
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
scatter = ax.scatter(dish_signals["H_taste"], dish_signals["H_satiety"],
                     c=dish_signals["log_DEI"], cmap="RdYlGn", s=30, alpha=0.7)
plt.colorbar(scatter, ax=ax, label="log(DEI)")
ax.set_xlabel("H (taste, BERT)")
ax.set_ylabel("H (satiety, text-mined)")
ax.set_title("Taste vs Satiety dimensions of hedonic value")
# Annotate corners
for _, row in dish_signals.nlargest(3, "H_satiety").iterrows():
    ax.annotate(row["dish_id"], (row["H_taste"], row["H_satiety"]),
                fontsize=6, alpha=0.8)
for _, row in dish_signals.nsmallest(3, "H_satiety").iterrows():
    ax.annotate(row["dish_id"], (row["H_taste"], row["H_satiety"]),
                fontsize=6, alpha=0.8)

# Correlation
r, p = sp_stats.pearsonr(dish_signals["H_taste"], dish_signals["H_satiety"])
ax.text(0.05, 0.95, f"r = {r:.3f}, p = {p:.3e}", transform=ax.transAxes,
        fontsize=9, va="top")

# Plot 2: Sensitivity of H contribution to satiety weight
ax = axes[1]
ax.plot(sens_df["w_satiety"], sens_df["H_contribution_pct"], "o-",
        color="steelblue", label="H contribution %")
ax2 = ax.twinx()
ax2.plot(sens_df["w_satiety"], sens_df["rank_rho_vs_original"], "s--",
         color="coral", label="Rank ρ vs original")
ax.set_xlabel("Weight on satiety (w₂)")
ax.set_ylabel("H contribution to Var(log DEI) (%)", color="steelblue")
ax2.set_ylabel("Spearman ρ vs original ranking", color="coral")
ax.set_title("DEI sensitivity to satiety weighting")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "h_satiety_scatter.png", dpi=150)
plt.close()
print("  Saved h_satiety_scatter.png")

# Plot 3: DEI sensitivity
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(sens_df["w_satiety"], 0, sens_df["H_contribution_pct"],
                color="steelblue", alpha=0.3, label="H contribution")
ax.fill_between(sens_df["w_satiety"], sens_df["H_contribution_pct"], 100,
                color="coral", alpha=0.3, label="E contribution")
ax.plot(sens_df["w_satiety"], sens_df["H_contribution_pct"], "o-k", markersize=5)
ax.set_xlabel("Weight on satiety dimension (w₂)")
ax.set_ylabel("% of Var(log DEI)")
ax.set_title("Variance decomposition under multi-dimensional H")
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "dei_sensitivity_to_satiety.png", dpi=150)
plt.close()
print("  Saved dei_sensitivity_to_satiety.png")

print("\n" + "=" * 70)
print("13c COMPLETE")
print("=" * 70)
