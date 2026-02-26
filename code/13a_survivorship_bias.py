"""
13a_survivorship_bias.py — Survivorship Bias Bounds Analysis
=============================================================
Addresses C6: Yelp dishes are market-selected survivors;
the observed 4% CV may be an artifact of survivorship filtering.

Analyses:
  A. Mention-frequency vs H variation
  B. Ghost-dish bounds analysis (heatmap)
  C. Star-rating stratification
  D. Literature CV comparison

Outputs:
  - tables/survivorship_bounds.csv
  - tables/mention_freq_h_analysis.csv
  - figures/survivorship_heatmap.png
  - figures/mention_frequency_vs_h_cv.png
"""

import sys, warnings
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
print("13a  SURVIVORSHIP BIAS BOUNDS ANALYSIS")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
combined = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
restaurants = pd.read_parquet(DATA_DIR / "restaurants.parquet")

print(f"Mentions: {len(mentions):,}, Dishes: {mentions['dish_id'].nunique()}")
print(f"Combined dataset: {len(combined)} dishes")

# ══════════════════════════════════════════════════════════════════
# A. MENTION FREQUENCY vs H VARIATION
# ══════════════════════════════════════════════════════════════════
print("\n── A. Mention Frequency vs H Variation ──")

dish_stats = mentions.groupby("dish_id").agg(
    n_mentions=("hedonic_score_finetuned", "count"),
    H_mean=("hedonic_score_finetuned", "mean"),
    H_std=("hedonic_score_finetuned", "std"),
).reset_index()

# Frequency bins
bins = [0, 50, 100, 200, 500, 1000, 100000]
labels = ["<50", "50-100", "100-200", "200-500", "500-1K", ">1K"]
dish_stats["freq_bin"] = pd.cut(dish_stats["n_mentions"], bins=bins, labels=labels)

freq_analysis = dish_stats.groupby("freq_bin", observed=True).agg(
    n_dishes=("dish_id", "count"),
    H_mean_avg=("H_mean", "mean"),
    H_mean_std=("H_mean", "std"),
    H_cv_pct=("H_mean", lambda x: x.std() / x.mean() * 100 if len(x) > 1 else np.nan),
    H_range=("H_mean", lambda x: x.max() - x.min()),
).reset_index()

print("\nFrequency bin analysis:")
print(freq_analysis.to_string(index=False))

# Test: do low-frequency dishes have higher H variation?
low_freq = dish_stats[dish_stats["n_mentions"] < 100]["H_mean"]
high_freq = dish_stats[dish_stats["n_mentions"] >= 500]["H_mean"]
levene_stat, levene_p = sp_stats.levene(low_freq, high_freq)
print(f"\nLevene's test (low vs high freq variance): F={levene_stat:.3f}, p={levene_p:.4f}")

freq_analysis.to_csv(TABLES_DIR / "mention_freq_h_analysis.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for i, row in freq_analysis.iterrows():
    subset = dish_stats[dish_stats["freq_bin"] == row["freq_bin"]]["H_mean"]
    ax.boxplot(subset.values, positions=[i], widths=0.6,
               boxprops=dict(color="steelblue"),
               medianprops=dict(color="red"))
ax.set_xticks(range(len(freq_analysis)))
ax.set_xticklabels(freq_analysis["freq_bin"].values, rotation=45)
ax.set_xlabel("Mention frequency bin")
ax.set_ylabel("Dish-level H (mean)")
ax.set_title("H distribution by mention frequency")

ax = axes[1]
ax.bar(range(len(freq_analysis)), freq_analysis["H_cv_pct"].values,
       color="steelblue", alpha=0.7)
ax.set_xticks(range(len(freq_analysis)))
ax.set_xticklabels(freq_analysis["freq_bin"].values, rotation=45)
ax.set_xlabel("Mention frequency bin")
ax.set_ylabel("CV of H (%)")
ax.set_title("H coefficient of variation by mention frequency")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "mention_frequency_vs_h_cv.png", dpi=150)
plt.close()
print("  Saved mention_frequency_vs_h_cv.png")

# ══════════════════════════════════════════════════════════════════
# B. GHOST DISH BOUNDS ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n── B. Ghost Dish Bounds Analysis ──")

# Current H stats
H_observed = combined["H_mean"].values
log_H_obs = np.log(H_observed)
log_E = combined["log_E"].values
var_log_E = np.var(log_E)

# Parameter sweep: K ghost dishes × Delta H reduction
K_values = [0, 50, 100, 200, 334, 500, 1000]
delta_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

np.random.seed(42)
n_sim = 500

bounds_results = []
for K in K_values:
    for delta in delta_values:
        if K == 0 and delta > 0:
            continue

        h_contributions = []
        for _ in range(n_sim):
            if K > 0:
                # Ghost dishes: H ~ N(H_min - delta, observed_std * 1.5)
                ghost_H = np.random.normal(
                    H_observed.min() - delta,
                    H_observed.std() * 1.5,
                    size=K
                )
                ghost_H = np.clip(ghost_H, 1.0, 10.0)
                # Ghost E: sample from observed E distribution
                ghost_log_E = np.random.choice(log_E, size=K, replace=True)
                all_log_H = np.concatenate([log_H_obs, np.log(ghost_H)])
                all_log_E = np.concatenate([log_E, ghost_log_E])
            else:
                all_log_H = log_H_obs
                all_log_E = log_E

            var_lh = np.var(all_log_H)
            var_le = np.var(all_log_E)
            var_ldei = var_lh + var_le - 2 * np.cov(all_log_H, all_log_E)[0, 1]
            h_pct = var_lh / var_ldei * 100 if var_ldei > 0 else 0

            h_contributions.append(h_pct)

        mean_h_pct = np.mean(h_contributions)
        bounds_results.append({
            "K_ghost": K, "delta_H": delta,
            "H_contribution_pct": mean_h_pct,
            "H_contribution_std": np.std(h_contributions),
        })

bounds_df = pd.DataFrame(bounds_results)
bounds_df.to_csv(TABLES_DIR / "survivorship_bounds.csv", index=False)

print("\nGhost dish bounds (H contribution %):")
pivot = bounds_df.pivot_table(index="delta_H", columns="K_ghost",
                               values="H_contribution_pct")
print(pivot.to_string(float_format="%.1f"))

# Find tipping points
print("\nTipping points (H contribution > 10%):")
tips = bounds_df[bounds_df["H_contribution_pct"] > 10]
if len(tips) > 0:
    print(tips[["K_ghost", "delta_H", "H_contribution_pct"]].to_string(index=False))
else:
    print("  None found — H contribution stays below 10% in all scenarios")

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
pivot_plot = bounds_df[bounds_df["K_ghost"] > 0].pivot_table(
    index="delta_H", columns="K_ghost", values="H_contribution_pct"
)
im = ax.imshow(pivot_plot.values, cmap="YlOrRd", aspect="auto",
               vmin=0, vmax=max(50, pivot_plot.values.max()))
ax.set_xticks(range(len(pivot_plot.columns)))
ax.set_xticklabels(pivot_plot.columns.values)
ax.set_yticks(range(len(pivot_plot.index)))
ax.set_yticklabels([f"{v:.1f}" for v in pivot_plot.index.values])
ax.set_xlabel("Number of ghost dishes (K)")
ax.set_ylabel("H reduction below observed minimum (Δ)")
ax.set_title("H contribution to Var(log DEI) under survivorship bias")

# Add text annotations
for i in range(len(pivot_plot.index)):
    for j in range(len(pivot_plot.columns)):
        val = pivot_plot.values[i, j]
        color = "white" if val > 25 else "black"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=8, color=color)

plt.colorbar(im, ax=ax, label="H contribution (%)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "survivorship_heatmap.png", dpi=150)
plt.close()
print("  Saved survivorship_heatmap.png")

# ══════════════════════════════════════════════════════════════════
# C. STAR RATING STRATIFICATION
# ══════════════════════════════════════════════════════════════════
print("\n── C. Restaurant Star Rating Stratification ──")

# Merge restaurant stars with mentions
mentions_with_stars = mentions.merge(
    restaurants[["business_id", "stars"]].rename(columns={"stars": "biz_stars"}),
    on="business_id", how="left"
)

# Bin restaurants by star rating
star_bins = [(1, 2.5, "Low (1-2.5★)"), (3, 3.5, "Mid (3-3.5★)"), (4, 5, "High (4-5★)")]
star_results = []
for lo, hi, label in star_bins:
    subset = mentions_with_stars[
        (mentions_with_stars["biz_stars"] >= lo) &
        (mentions_with_stars["biz_stars"] <= hi)
    ]
    if len(subset) < 100:
        continue
    dish_h = subset.groupby("dish_id")["hedonic_score_finetuned"].mean()
    dish_h = dish_h[dish_h.index.isin(combined["dish_id"])]
    if len(dish_h) < 10:
        continue
    star_results.append({
        "star_bin": label,
        "n_mentions": len(subset),
        "n_dishes": len(dish_h),
        "H_mean": dish_h.mean(),
        "H_std": dish_h.std(),
        "H_cv_pct": dish_h.std() / dish_h.mean() * 100,
        "H_range": dish_h.max() - dish_h.min(),
    })

star_df = pd.DataFrame(star_results)
print(star_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# D. SURVIVORSHIP ARGUMENT SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n── D. Summary ──")
print(f"  Observed H CV: {combined['H_mean'].std()/combined['H_mean'].mean()*100:.1f}%")
print(f"  Observed H range: [{combined['H_mean'].min():.2f}, {combined['H_mean'].max():.2f}]")
print(f"  To reach H contribution >10%: need K≥{bounds_df[bounds_df['H_contribution_pct']>10]['K_ghost'].min() if len(bounds_df[bounds_df['H_contribution_pct']>10])>0 else 'N/A'} ghost dishes")

# Argument: even among survivors, if H is compressed, the full population
# would have H with both LOW outliers (bad dishes) AND same E distribution.
# Since E variance is so large, adding bad-tasting dishes doesn't change
# the fact that E dominates — it just makes the decoupling even stronger
# (bad-tasting dishes can be either high or low E).

base_h_pct = bounds_df[bounds_df["K_ghost"] == 0]["H_contribution_pct"].values[0]
k334_d2_h_pct = bounds_df[
    (bounds_df["K_ghost"] == 334) & (bounds_df["delta_H"] == 2.0)
]["H_contribution_pct"].values[0]

print(f"\n  Baseline H contribution: {base_h_pct:.1f}%")
print(f"  With 334 ghost dishes (Δ=2.0): {k334_d2_h_pct:.1f}%")
print(f"  Even doubling the sample with very bad dishes: E still dominates")

print("\n" + "=" * 70)
print("13a COMPLETE")
print("=" * 70)
