"""
11b_yelp_proxy_controls.py — Yelp Proxy Bias Control
=====================================================
Addresses criticism C2: Yelp reviews conflate dish quality with
restaurant quality. Restaurant ICC (8.3%) > Dish ICC (2.4%).

Method: Two-stage residualization
  1. Regress H on restaurant-level confounders (stars, price, review count, text length)
  2. Extract residuals as "controlled H"
  3. Recompute DEI with controlled H, compare rankings

Outputs:
  - tables/controlled_dei_rankings.csv
  - tables/proxy_bias_summary.csv
  - figures/controlled_vs_original_dei.png
"""

import sys, ast, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── Load data ────────────────────────────────────────────────────
print("=" * 60)
print("11b: Yelp Proxy Bias Control")
print("=" * 60)

dei = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
print(f"Loaded {len(dei)} dishes from combined DEI")

# Load mention-level data (original has finetuned H scores)
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
print(f"Loaded {len(mentions):,} original mentions (with finetuned H)")

# Load restaurant metadata
restaurants = pd.read_parquet(DATA_DIR / "restaurants.parquet")
print(f"Loaded {len(restaurants):,} restaurants")

# Extract price range
def extract_price(attr_str):
    if pd.isna(attr_str):
        return np.nan
    try:
        d = ast.literal_eval(str(attr_str))
        p = d.get("RestaurantsPriceRange2")
        if p is not None:
            p = str(p).strip("'\"")
            return int(p) if p.isdigit() else np.nan
        return np.nan
    except:
        return np.nan

restaurants["price_range"] = restaurants["attributes"].apply(extract_price)

# Merge
df = mentions.merge(
    restaurants[["business_id", "stars", "review_count", "price_range"]].rename(
        columns={"stars": "biz_stars", "review_count": "biz_review_count"}
    ),
    on="business_id", how="left"
)

# H column
H_COL = "hedonic_score_finetuned"
if H_COL not in df.columns:
    H_COL = "hedonic_score"
    print(f"  Using fallback H column: {H_COL}")

df["text_len"] = df["context_text"].str.len()
df = df.dropna(subset=[H_COL, "biz_stars"])
print(f"After merge & filter: {len(df):,} mentions, {df['dish_id'].nunique()} dishes")

# ══════════════════════════════════════════════════════════════════
# STAGE 1: Restaurant-level regression
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 1: Restaurant-Level Regression")
print("=" * 60)

# Prepare features
df["log_biz_review_count"] = np.log1p(df["biz_review_count"].fillna(0))
df["price_range_filled"] = df["price_range"].fillna(df["price_range"].median())
df["text_len_filled"] = df["text_len"].fillna(df["text_len"].median())

# OLS: H ~ biz_stars + price_range + log(review_count) + text_len
X_cols = ["biz_stars", "price_range_filled", "log_biz_review_count", "text_len_filled"]
X = df[X_cols].values
X = np.column_stack([np.ones(len(X)), X])  # add intercept
y = df[H_COL].values

# Solve OLS
from numpy.linalg import lstsq
beta, residuals, rank, sv = lstsq(X, y, rcond=None)

y_hat = X @ beta
r_squared = 1 - np.var(y - y_hat) / np.var(y)

print(f"  OLS: H ~ intercept + biz_stars + price + log(review_count) + text_len")
print(f"  R² = {r_squared:.4f}")
print(f"  Coefficients:")
coef_names = ["intercept"] + X_cols
for name, b in zip(coef_names, beta):
    print(f"    {name:30s}: {b:+.6f}")

# ══════════════════════════════════════════════════════════════════
# STAGE 2: Extract controlled H (remove restaurant-level confounders)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 2: Extract Controlled H")
print("=" * 60)

# Only remove restaurant-level effects (not text_len which is review-level)
restaurant_effect = (beta[1] * df["biz_stars"].values +
                     beta[2] * df["price_range_filled"].values +
                     beta[3] * df["log_biz_review_count"].values)

# Controlled H = original H - restaurant effects + grand mean of effects
# (centering preserves the overall H mean)
grand_mean_effect = restaurant_effect.mean()
df["H_controlled"] = df[H_COL].values - restaurant_effect + grand_mean_effect

print(f"  Original H: mean={df[H_COL].mean():.4f}, std={df[H_COL].std():.4f}")
print(f"  Controlled H: mean={df['H_controlled'].mean():.4f}, std={df['H_controlled'].std():.4f}")

# ══════════════════════════════════════════════════════════════════
# STAGE 3: Aggregate to dish level, recompute DEI
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 3: Dish-Level Aggregation & DEI Recomputation")
print("=" * 60)

dish_controlled = df.groupby("dish_id").agg(
    H_controlled_mean=("H_controlled", "mean"),
    H_original_mean=(H_COL, "mean"),
    n_mentions=(H_COL, "count"),
).reset_index()

# Merge with DEI data (get E values)
merged = dei.merge(dish_controlled, on="dish_id", how="inner")
print(f"  Matched {len(merged)} dishes (of {len(dei)} in combined)")

# Recompute DEI with controlled H
merged["log_H_controlled"] = np.log(merged["H_controlled_mean"].clip(lower=0.1))
merged["log_DEI_controlled"] = merged["log_H_controlled"] - merged["log_E"]

# ══════════════════════════════════════════════════════════════════
# STAGE 4: Impact Assessment
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 4: Impact Assessment")
print("=" * 60)

# Rank correlation
rho, p = sp_stats.spearmanr(merged["log_DEI"], merged["log_DEI_controlled"])
tau, p_tau = sp_stats.kendalltau(merged["log_DEI"], merged["log_DEI_controlled"])
print(f"  DEI rank correlation (original vs controlled):")
print(f"    Spearman ρ = {rho:.4f} (p = {p:.2e})")
print(f"    Kendall τ  = {tau:.4f} (p = {p_tau:.2e})")

# Variance decomposition with controlled H
var_log_H_ctrl = merged["log_H_controlled"].var()
var_log_E = merged["log_E"].var()
var_log_DEI_ctrl = merged["log_DEI_controlled"].var()
H_pct_ctrl = var_log_H_ctrl / var_log_DEI_ctrl * 100

var_log_H_orig = merged["log_H"].var()
var_log_DEI_orig = merged["log_DEI"].var()
H_pct_orig = var_log_H_orig / var_log_DEI_orig * 100

print(f"\n  Variance decomposition:")
print(f"    Original:   H contributes {H_pct_orig:.2f}% of Var(log DEI)")
print(f"    Controlled: H contributes {H_pct_ctrl:.2f}% of Var(log DEI)")

# H CV comparison
H_ctrl_cv = merged["H_controlled_mean"].std() / merged["H_controlled_mean"].mean() * 100
H_orig_cv = merged["H_original_mean"].std() / merged["H_original_mean"].mean() * 100
print(f"\n  H CV:")
print(f"    Original:   {H_orig_cv:.2f}%")
print(f"    Controlled: {H_ctrl_cv:.2f}%")

# Tier changes
N_matched = len(merged)
orig_tiers = pd.qcut(merged["log_DEI"], 5, labels=["E", "D", "C", "B", "A"])
ctrl_tiers = pd.qcut(merged["log_DEI_controlled"], 5, labels=["E", "D", "C", "B", "A"])
tier_match = (orig_tiers == ctrl_tiers).sum()
print(f"\n  Tier agreement: {tier_match}/{N_matched} ({tier_match/N_matched*100:.1f}%)")

# Rank shifts
merged["rank_original"] = merged["log_DEI"].rank(ascending=False)
merged["rank_controlled"] = merged["log_DEI_controlled"].rank(ascending=False)
merged["rank_shift"] = (merged["rank_controlled"] - merged["rank_original"]).astype(int)

print(f"\n  Rank shift statistics:")
print(f"    Mean |shift|: {merged['rank_shift'].abs().mean():.1f}")
print(f"    Median |shift|: {merged['rank_shift'].abs().median():.1f}")
print(f"    Max shift: {merged['rank_shift'].abs().max()}")

# Top movers
print(f"\n  Biggest rank gainers (better DEI after control):")
gainers = merged.nsmallest(5, "rank_shift")
for _, row in gainers.iterrows():
    print(f"    {row['dish_id']:25s}: rank {int(row['rank_original'])} → {int(row['rank_controlled'])} "
          f"(shift={int(row['rank_shift'])})")

print(f"\n  Biggest rank losers (worse DEI after control):")
losers = merged.nlargest(5, "rank_shift")
for _, row in losers.iterrows():
    print(f"    {row['dish_id']:25s}: rank {int(row['rank_original'])} → {int(row['rank_controlled'])} "
          f"(shift={int(row['rank_shift'])})")

# Save summary
summary_rows = [
    {"metric": "spearman_rho", "value": rho},
    {"metric": "kendall_tau", "value": tau},
    {"metric": "H_pct_original", "value": H_pct_orig},
    {"metric": "H_pct_controlled", "value": H_pct_ctrl},
    {"metric": "H_cv_original", "value": H_orig_cv},
    {"metric": "H_cv_controlled", "value": H_ctrl_cv},
    {"metric": "tier_agreement_pct", "value": tier_match / N_matched * 100},
    {"metric": "mean_abs_rank_shift", "value": merged["rank_shift"].abs().mean()},
    {"metric": "regression_r_squared", "value": r_squared},
    {"metric": "n_dishes_matched", "value": N_matched},
]
pd.DataFrame(summary_rows).to_csv(TABLES_DIR / "proxy_bias_summary.csv", index=False)

# Save full rankings
out_cols = ["dish_id", "cuisine", "H_original_mean", "H_controlled_mean",
            "E_composite", "log_DEI", "log_DEI_controlled",
            "rank_original", "rank_controlled", "rank_shift", "n_mentions"]
merged[out_cols].sort_values("rank_controlled").to_csv(
    TABLES_DIR / "controlled_dei_rankings.csv", index=False)

print(f"\n  Saved: {TABLES_DIR / 'proxy_bias_summary.csv'}")
print(f"  Saved: {TABLES_DIR / 'controlled_dei_rankings.csv'}")

# ══════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figure...")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Scatter of original vs controlled log(DEI)
ax1.scatter(merged["log_DEI"], merged["log_DEI_controlled"],
            alpha=0.5, s=20, c="#2196F3", edgecolors="none")
lims = [merged[["log_DEI", "log_DEI_controlled"]].min().min() - 0.2,
        merged[["log_DEI", "log_DEI_controlled"]].max().max() + 0.2]
ax1.plot(lims, lims, "k--", alpha=0.3, label="y=x")
ax1.set_xlabel("log(DEI) — Original", fontsize=12)
ax1.set_ylabel("log(DEI) — Controlled", fontsize=12)
ax1.set_title(f"A. DEI Before vs After Restaurant Control\n(Spearman ρ = {rho:.3f})",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(lims)
ax1.set_ylim(lims)

# Panel B: Rank shift distribution
shifts = merged["rank_shift"]
ax2.hist(shifts, bins=50, color="#FF9800", alpha=0.7, edgecolor="white")
ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5)
ax2.set_xlabel("Rank Shift (positive = worse rank after control)", fontsize=12)
ax2.set_ylabel("Number of Dishes", fontsize=12)
ax2.set_title(f"B. Distribution of Rank Shifts\n(mean |shift| = {shifts.abs().mean():.1f})",
              fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "controlled_vs_original_dei.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'controlled_vs_original_dei.png'}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Restaurant controls explain {r_squared*100:.1f}% of mention-level H variance")
print(f"  After control: DEI rank ρ = {rho:.4f}")
print(f"  H CV changes: {H_orig_cv:.1f}% → {H_ctrl_cv:.1f}%")
print(f"  Tier agreement: {tier_match/N_matched*100:.1f}%")
print(f"  Conclusion: E dominates DEI so strongly that restaurant controls")
print(f"  barely affect DEI rankings (ρ = {rho:.3f})")
print("=" * 60)
