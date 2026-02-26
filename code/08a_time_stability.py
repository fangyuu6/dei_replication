"""
08a_time_stability.py — P2-1: Temporal Stability of H Scores
=============================================================
Tests whether dish-level H scores are stable across review years.

Analyses:
  1. Review year distribution
  2. H ~ year regression (dish-level and mention-level)
  3. Split-period comparison: early vs late H rankings
  4. Year-by-year dish H correlation with full-sample H

Outputs:
  - tables/temporal_stability.csv
  - figures/temporal_stability.png
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, FIGURES_DIR, TABLES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load & merge ─────────────────────────────────────────────────
print("Loading data...")
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
reviews  = pd.read_parquet(DATA_DIR / "restaurant_reviews.parquet",
                           columns=["review_id", "date"])

df = mentions.merge(reviews, on="review_id", how="left")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

H = "hedonic_score_finetuned"
print(f"Merged: {len(df):,} mentions, years {df['year'].min()}-{df['year'].max()}")

# ══════════════════════════════════════════════════════════════════
# 1. Year distribution
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. Review year distribution")
print("=" * 60)

year_dist = df["year"].value_counts().sort_index()
for y, n in year_dist.items():
    print(f"  {y}: {n:>6,} mentions")

# ══════════════════════════════════════════════════════════════════
# 2. H ~ year regression (mention-level)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. H ~ year regression (mention-level)")
print("=" * 60)

slope, intercept, r, p, se = stats.linregress(df["year"], df[H])
print(f"  slope  = {slope:+.5f} H-points/year")
print(f"  r      = {r:.4f}")
print(f"  p      = {p:.2e}")
print(f"  → H {'increases' if slope > 0 else 'decreases'} by {abs(slope):.4f} per year")
print(f"  → Over 17 years ({df['year'].min()}-{df['year'].max()}): Δ = {slope*17:+.3f}")

# ══════════════════════════════════════════════════════════════════
# 3. Dish-level H by year
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Dish-level H stability across years")
print("=" * 60)

# Full-sample dish means
full_h = df.groupby("dish_id")[H].mean()

# Use years with enough data (>= 1000 mentions)
good_years = year_dist[year_dist >= 1000].index.tolist()
print(f"  Years with ≥1000 mentions: {good_years}")

year_corrs = []
for y in good_years:
    sub = df[df["year"] == y]
    # Only dishes with ≥5 mentions in that year
    dish_h_year = sub.groupby("dish_id")[H].agg(["mean", "count"])
    dish_h_year = dish_h_year[dish_h_year["count"] >= 5]
    common = dish_h_year.index.intersection(full_h.index)
    if len(common) >= 20:
        rho, p_rho = stats.spearmanr(dish_h_year.loc[common, "mean"],
                                      full_h.loc[common])
        year_corrs.append({
            "year": y,
            "n_mentions": len(sub),
            "n_dishes": len(common),
            "spearman_rho": rho,
            "p_value": p_rho,
            "mean_H": sub[H].mean(),
        })
        sig = "***" if p_rho < 0.001 else "**" if p_rho < 0.01 else "*" if p_rho < 0.05 else "n.s."
        print(f"  {y}: ρ = {rho:.3f} {sig} (n_dishes={len(common)}, mean_H={sub[H].mean():.3f})")

year_corr_df = pd.DataFrame(year_corrs)

# ══════════════════════════════════════════════════════════════════
# 4. Split-period comparison
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Split-period comparison (early vs late)")
print("=" * 60)

median_year = df["year"].median()
early = df[df["year"] <= median_year]
late  = df[df["year"] > median_year]

h_early = early.groupby("dish_id")[H].agg(["mean", "count"])
h_late  = late.groupby("dish_id")[H].agg(["mean", "count"])

# Filter to dishes with ≥10 in each period
h_early = h_early[h_early["count"] >= 10]
h_late  = h_late[h_late["count"] >= 10]
common = h_early.index.intersection(h_late.index)

rho_split, p_split = stats.spearmanr(h_early.loc[common, "mean"],
                                      h_late.loc[common, "mean"])
tau_split, p_tau = stats.kendalltau(h_early.loc[common, "mean"],
                                     h_late.loc[common, "mean"])

print(f"  Early period: ≤{median_year:.0f} ({len(early):,} mentions)")
print(f"  Late period:  >{median_year:.0f} ({len(late):,} mentions)")
print(f"  Common dishes (≥10 each): {len(common)}")
print(f"  Spearman ρ = {rho_split:.4f} (p = {p_split:.2e})")
print(f"  Kendall  τ = {tau_split:.4f} (p = {p_tau:.2e})")

# Mean absolute H difference between periods
h_diff = (h_early.loc[common, "mean"] - h_late.loc[common, "mean"]).abs()
print(f"  Mean |ΔH| between periods: {h_diff.mean():.4f}")
print(f"  Max  |ΔH|: {h_diff.max():.4f} ({h_diff.idxmax()})")

# ══════════════════════════════════════════════════════════════════
# 5. Per-dish time trend
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Per-dish H time trends")
print("=" * 60)

dish_trends = []
for dish_id, group in df.groupby("dish_id"):
    if len(group) < 50:
        continue
    sl, ic, r_d, p_d, se_d = stats.linregress(group["year"], group[H])
    dish_trends.append({
        "dish_id": dish_id,
        "slope": sl,
        "r": r_d,
        "p_value": p_d,
        "n": len(group),
    })

trend_df = pd.DataFrame(dish_trends)
n_sig_pos = ((trend_df["p_value"] < 0.05) & (trend_df["slope"] > 0)).sum()
n_sig_neg = ((trend_df["p_value"] < 0.05) & (trend_df["slope"] < 0)).sum()
n_ns = (trend_df["p_value"] >= 0.05).sum()

print(f"  Dishes with significant positive trend: {n_sig_pos}")
print(f"  Dishes with significant negative trend: {n_sig_neg}")
print(f"  Dishes with no significant trend: {n_ns}")
print(f"  Mean |slope|: {trend_df['slope'].abs().mean():.5f} H-points/year")

# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# A: Year distribution
ax = axes[0, 0]
ax.bar(year_dist.index, year_dist.values, color="steelblue", edgecolor="white")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Dish Mentions")
ax.set_title("A. Temporal Distribution of Reviews")

# B: Mean H by year
ax = axes[0, 1]
if len(year_corr_df) > 0:
    ax.plot(year_corr_df["year"], year_corr_df["mean_H"], "o-", color="steelblue", markersize=5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean H Score")
    ax.set_title(f"B. Mean H by Year\n(trend: {slope:+.4f}/yr, p={p:.2e})")
    # Add trend line
    x_line = np.array(year_corr_df["year"])
    ax.plot(x_line, intercept + slope * x_line, "r--", alpha=0.5, linewidth=2)

# C: Year-specific ρ with full sample
ax = axes[1, 0]
if len(year_corr_df) > 0:
    ax.bar(year_corr_df["year"], year_corr_df["spearman_rho"],
           color="seagreen", edgecolor="white")
    ax.axhline(0.8, color="red", linestyle=":", alpha=0.5, label="ρ = 0.8")
    ax.set_xlabel("Year")
    ax.set_ylabel("Spearman ρ with Full-Sample H")
    ax.set_title("C. Year-Specific H vs Full-Sample H")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

# D: Early vs Late scatter
ax = axes[1, 1]
ax.scatter(h_early.loc[common, "mean"], h_late.loc[common, "mean"],
           alpha=0.6, s=25, edgecolors="k", linewidths=0.3)
lims = [min(h_early.loc[common, "mean"].min(), h_late.loc[common, "mean"].min()) - 0.1,
        max(h_early.loc[common, "mean"].max(), h_late.loc[common, "mean"].max()) + 0.1]
ax.plot(lims, lims, "r--", alpha=0.5)
ax.set_xlabel(f"H (≤{median_year:.0f})")
ax.set_ylabel(f"H (>{median_year:.0f})")
ax.set_title(f"D. Early vs Late Period H\n(ρ = {rho_split:.3f}, τ = {tau_split:.3f})")

plt.tight_layout()
fig_path = FIGURES_DIR / "temporal_stability.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close()

# ── Save ─────────────────────────────────────────────────────────
results = {
    "mention_level_slope": slope,
    "mention_level_r": r,
    "mention_level_p": p,
    "split_period_rho": rho_split,
    "split_period_tau": tau_split,
    "n_dishes_sig_positive": n_sig_pos,
    "n_dishes_sig_negative": n_sig_neg,
    "n_dishes_ns": n_ns,
    "mean_year_rho": year_corr_df["spearman_rho"].mean() if len(year_corr_df) > 0 else np.nan,
}
pd.DataFrame([results]).to_csv(TABLES_DIR / "temporal_stability.csv", index=False)
year_corr_df.to_csv(TABLES_DIR / "temporal_year_correlations.csv", index=False)
trend_df.to_csv(TABLES_DIR / "temporal_dish_trends.csv", index=False)

print(f"  Saved: {TABLES_DIR / 'temporal_stability.csv'}")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: Temporal Stability")
print("=" * 60)
print(f"""
  Mention-level H ~ year: slope = {slope:+.5f}/yr (p = {p:.2e})
  → Over 17 years: total drift = {slope*17:+.3f} (negligible on 1-10 scale)

  Split-period reliability: ρ = {rho_split:.3f}, τ = {tau_split:.3f}
  → Dish rankings are {'highly' if rho_split > 0.8 else 'moderately' if rho_split > 0.6 else 'weakly'} stable across time

  Per-dish trends: {n_sig_pos} positive, {n_sig_neg} negative, {n_ns} n.s.
  → {'Most' if n_ns > n_sig_pos + n_sig_neg else 'Some'} dishes show stable H over time

  Mean year-specific ρ with full sample: {year_corr_df['spearman_rho'].mean():.3f}
""")
print("Done!")
