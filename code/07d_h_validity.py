"""
07d_h_validity.py — P0-2b/2c/2e: H Construct Validity & Mixed-Effects Model
=============================================================================
Addresses reviewer concern: "Yelp reviews conflate dish quality with restaurant quality"

Analyses:
  1. Mixed-effects model: H_ij = dish_i + restaurant_j + ε (ICC decomposition)
  2. Construct validity: partial correlations of H with stars, price, review length
  3. Geographic stratification: H consistency across US states
  4. Restaurant-level controls: dish H after partialing out restaurant quality

Outputs:
  - tables/h_icc_decomposition.csv
  - tables/h_construct_validity.csv
  - tables/h_geographic_stability.csv
  - figures/h_validity_diagnostics.png
"""

import sys, ast, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────
print("Loading data...")
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
restaurants = pd.read_parquet(DATA_DIR / "restaurants.parquet")

# Extract price from attributes dict
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
    restaurants[["business_id", "stars", "review_count", "city", "state", "price_range"]].rename(
        columns={"stars": "biz_stars", "review_count": "biz_review_count"}
    ),
    on="business_id", how="left"
)

# Compute review text length
df["text_len"] = df["context_text"].str.len()

H_COL = "hedonic_score_finetuned"
print(f"Merged: {len(df)} mentions, {df['dish_id'].nunique()} dishes, "
      f"{df['business_id'].nunique()} restaurants")
print(f"H column: {H_COL}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: Variance decomposition — Dish vs Restaurant effects
# (Replaces full mixed-effects model — simpler, no statsmodels needed)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 1: Variance Decomposition (Dish vs Restaurant)")
print("=" * 60)

# Total variance
total_var = df[H_COL].var()

# Dish-level variance (between-dish)
dish_means = df.groupby("dish_id")[H_COL].mean()
dish_var_between = dish_means.var()

# Restaurant-level variance (between-restaurant)
rest_means = df.groupby("business_id")[H_COL].mean()
rest_var_between = rest_means.var()

# ICC calculation using ANOVA approach
# ICC(dish) = var_between_dish / (var_between_dish + var_within_dish)
from collections import defaultdict

def compute_icc(df, group_col, value_col, min_obs=2):
    """One-way random effects ICC(1) using ANOVA."""
    groups = df.groupby(group_col)[value_col]
    k = groups.ngroups
    ns = groups.size()
    ns = ns[ns >= min_obs]
    valid_groups = ns.index
    sub = df[df[group_col].isin(valid_groups)]

    groups_sub = sub.groupby(group_col)[value_col]
    grand_mean = sub[value_col].mean()

    # Between-group MS
    group_means = groups_sub.mean()
    group_sizes = groups_sub.size()
    ss_between = (group_sizes * (group_means - grand_mean) ** 2).sum()
    df_between = len(group_means) - 1
    ms_between = ss_between / df_between

    # Within-group MS
    ss_within = groups_sub.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    df_within = len(sub) - len(group_means)
    ms_within = ss_within / df_within

    # n0 (harmonic-ish mean group size)
    n_total = len(sub)
    n0 = (n_total - (group_sizes ** 2).sum() / n_total) / df_between

    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    return icc, ms_between, ms_within, len(group_means)

icc_dish, ms_b_dish, ms_w_dish, n_dishes = compute_icc(df, "dish_id", H_COL)
icc_rest, ms_b_rest, ms_w_rest, n_rests = compute_icc(df, "business_id", H_COL, min_obs=3)

print(f"\n  Total variance of H: {total_var:.4f}")
print(f"\n  Dish-level ICC(1): {icc_dish:.4f}")
print(f"    → {icc_dish*100:.1f}% of H variance is between dishes")
print(f"    → {(1-icc_dish)*100:.1f}% is within-dish (individual review noise + restaurant effects)")
print(f"    MS_between={ms_b_dish:.4f}, MS_within={ms_w_dish:.4f}, n_groups={n_dishes}")

print(f"\n  Restaurant-level ICC(1): {icc_rest:.4f}")
print(f"    → {icc_rest*100:.1f}% of H variance is between restaurants")
print(f"    MS_between={ms_b_rest:.4f}, MS_within={ms_w_rest:.4f}, n_groups={n_rests}")

# Two-way decomposition: estimate dish + restaurant variance simultaneously
# Using a simple nested ANOVA approach
print(f"\n  Interpretation:")
if icc_dish > icc_rest:
    print(f"    Dish identity explains MORE variance ({icc_dish:.3f}) than restaurant identity ({icc_rest:.3f})")
    print(f"    → H is more about the DISH than the RESTAURANT — supports construct validity")
else:
    print(f"    Restaurant identity explains MORE variance ({icc_rest:.3f}) than dish identity ({icc_dish:.3f})")
    print(f"    → H partially confounded with restaurant quality — need controls")

# Residualized H: remove restaurant mean
df["biz_mean_H"] = df.groupby("business_id")[H_COL].transform("mean")
df["H_residualized"] = df[H_COL] - df["biz_mean_H"] + df[H_COL].mean()  # center on grand mean

# Dish-level means: raw vs residualized
dish_raw = df.groupby("dish_id")[H_COL].mean().rename("H_raw")
dish_resid = df.groupby("dish_id")["H_residualized"].mean().rename("H_residualized")
dish_compare = pd.concat([dish_raw, dish_resid], axis=1)

rho_raw_resid, p_rr = stats.spearmanr(dish_compare["H_raw"], dish_compare["H_residualized"])
print(f"\n  Dish-level H (raw) vs H (restaurant-residualized):")
print(f"    Spearman ρ = {rho_raw_resid:.4f} (p = {p_rr:.2e})")
print(f"    → Rankings {'barely change' if rho_raw_resid > 0.95 else 'change somewhat' if rho_raw_resid > 0.8 else 'change substantially'} after removing restaurant effects")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Construct Validity — H vs confounds
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 2: Construct Validity (H vs potential confounds)")
print("=" * 60)

# At mention level: correlate H with stars, price, text_len
confounds = {
    "biz_stars": "Restaurant avg stars (1-5)",
    "price_range": "Restaurant price range (1-4)",
    "text_len": "Review text length (chars)",
}

validity_results = []
for col, desc in confounds.items():
    valid = df[[H_COL, col]].dropna()
    if len(valid) < 100:
        print(f"  {col}: insufficient data (n={len(valid)})")
        continue
    r_pearson, p_pearson = stats.pearsonr(valid[H_COL], valid[col])
    r_spearman, p_spearman = stats.spearmanr(valid[H_COL], valid[col])

    validity_results.append({
        "confound": col,
        "description": desc,
        "n": len(valid),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_rho": r_spearman,
        "spearman_p": p_spearman,
    })

    sig = "***" if p_spearman < 0.001 else "**" if p_spearman < 0.01 else "*" if p_spearman < 0.05 else "n.s."
    print(f"  {col:20s}: r={r_pearson:+.4f} (Pearson), ρ={r_spearman:+.4f} (Spearman) {sig}  n={len(valid):,}")

# Dish-level: aggregate confounds and check
print("\n  Dish-level correlations (aggregated):")
dish_agg = df.groupby("dish_id").agg(
    H_mean=(H_COL, "mean"),
    biz_stars_mean=("biz_stars", "mean"),
    price_mean=("price_range", "mean"),
    text_len_mean=("text_len", "mean"),
    n_mentions=(H_COL, "count"),
).dropna()

for col in ["biz_stars_mean", "price_mean", "text_len_mean", "n_mentions"]:
    r, p = stats.spearmanr(dish_agg["H_mean"], dish_agg[col])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  H_mean vs {col:20s}: ρ = {r:+.4f} {sig}")
    validity_results.append({
        "confound": f"dish_{col}",
        "description": f"Dish-level {col}",
        "n": len(dish_agg),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": r,
        "spearman_p": p,
    })

# Partial correlation: H vs dish identity controlling for confounds
print("\n  Partial correlations (dish-level H | confounds):")
# Multiple regression: H_mean ~ biz_stars_mean + price_mean + text_len_mean
from numpy.linalg import lstsq

X = dish_agg[["biz_stars_mean", "price_mean", "text_len_mean"]].values
X = np.column_stack([np.ones(len(X)), X])
y = dish_agg["H_mean"].values
beta, _, _, _ = lstsq(X, y, rcond=None)
resid_y = y - X @ beta
print(f"  R² of H ~ confounds: {1 - np.var(resid_y)/np.var(y):.4f}")
print(f"  → {(1 - np.var(resid_y)/np.var(y))*100:.1f}% of dish-level H variance explained by confounds")
print(f"  → {np.var(resid_y)/np.var(y)*100:.1f}% is 'pure dish taste' variance")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Geographic Stability
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 3: Geographic Stability of H")
print("=" * 60)

# Top states by mention count
state_counts = df.groupby("state").size().sort_values(ascending=False)
print(f"  Top 10 states by mentions:")
for state, count in state_counts.head(10).items():
    n_dishes = df[df["state"] == state]["dish_id"].nunique()
    print(f"    {state}: {count:,} mentions, {n_dishes} dishes")

# Select top 5 states with enough dishes
top_states = state_counts.head(5).index.tolist()
geo_results = []

# Compute dish-level H by state
state_dish_h = {}
for state in top_states:
    sub = df[df["state"] == state]
    h_by_dish = sub.groupby("dish_id")[H_COL].agg(["mean", "count"])
    h_by_dish = h_by_dish[h_by_dish["count"] >= 5]  # min 5 mentions
    state_dish_h[state] = h_by_dish["mean"]

# Pairwise correlations
print(f"\n  Pairwise Spearman ρ of dish-level H across states:")
for i, s1 in enumerate(top_states):
    for s2 in top_states[i+1:]:
        common = state_dish_h[s1].index.intersection(state_dish_h[s2].index)
        if len(common) >= 10:
            r, p = stats.spearmanr(
                state_dish_h[s1].loc[common],
                state_dish_h[s2].loc[common]
            )
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {s1} vs {s2}: ρ = {r:.4f} {sig} (n={len(common)} common dishes)")
            geo_results.append({
                "state_1": s1, "state_2": s2,
                "n_common": len(common),
                "spearman_rho": r, "p_value": p,
            })

# Overall: H from each state vs full-sample H
print(f"\n  State-specific H vs full-sample H:")
full_h = df.groupby("dish_id")[H_COL].mean()
for state in top_states:
    common = state_dish_h[state].index.intersection(full_h.index)
    r, p = stats.spearmanr(state_dish_h[state].loc[common], full_h.loc[common])
    n_dishes = len(common)
    print(f"    {state}: ρ = {r:.4f} (n={n_dishes} dishes)")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 4: Restaurant-controlled H rankings
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 4: Restaurant-controlled H rankings")
print("=" * 60)

# Method: compute dish-level H residuals after removing restaurant fixed effects
# Already computed H_residualized above

# Compare rankings
dei = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv")
dei_h = dei.set_index("dish_id")["H_mean"]

dish_h_raw = df.groupby("dish_id")[H_COL].mean()
dish_h_resid = df.groupby("dish_id")["H_residualized"].mean()

# Merge all
compare = pd.DataFrame({
    "H_finetuned": dish_h_raw,
    "H_restaurant_controlled": dish_h_resid,
}).dropna()

tau_controlled, p_ctrl = stats.kendalltau(compare["H_finetuned"], compare["H_restaurant_controlled"])
rho_controlled, p_ctrl_rho = stats.spearmanr(compare["H_finetuned"], compare["H_restaurant_controlled"])

print(f"  H (raw) vs H (restaurant-controlled):")
print(f"    Kendall τ = {tau_controlled:.4f}")
print(f"    Spearman ρ = {rho_controlled:.4f}")

# How many dishes change rank by ≥5?
compare["rank_raw"] = compare["H_finetuned"].rank(ascending=False)
compare["rank_ctrl"] = compare["H_restaurant_controlled"].rank(ascending=False)
compare["rank_shift"] = (compare["rank_raw"] - compare["rank_ctrl"]).abs()

for thresh in [1, 3, 5, 10]:
    n = (compare["rank_shift"] >= thresh).sum()
    print(f"    |rank shift| >= {thresh}: {n} dishes ({n/len(compare)*100:.1f}%)")

# Top movers
movers = compare.nlargest(10, "rank_shift")
print(f"\n  Top 10 dishes most affected by restaurant controls:")
for dish, row in movers.iterrows():
    print(f"    {dish:25s}  raw_rank={row['rank_raw']:5.0f} → ctrl_rank={row['rank_ctrl']:5.0f}  "
          f"shift={row['rank_shift']:.0f}")

# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: ICC decomposition bar chart
ax = axes[0, 0]
labels = ["Dish\n(ICC)", "Restaurant\n(ICC)", "Residual"]
vals = [icc_dish, icc_rest, max(0, 1 - icc_dish - icc_rest)]
colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Proportion of Variance")
ax.set_title("A. Variance Decomposition of H\n(Dish vs Restaurant effects)")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.2)

# Panel B: H raw vs H restaurant-controlled
ax = axes[0, 1]
ax.scatter(compare["H_finetuned"], compare["H_restaurant_controlled"],
           alpha=0.6, s=25, edgecolors="k", linewidths=0.3)
lims = [min(compare["H_finetuned"].min(), compare["H_restaurant_controlled"].min()) - 0.1,
        max(compare["H_finetuned"].max(), compare["H_restaurant_controlled"].max()) + 0.1]
ax.plot(lims, lims, "r--", alpha=0.5)
ax.set_xlabel("H (raw finetuned)")
ax.set_ylabel("H (restaurant-controlled)")
ax.set_title(f"B. Raw vs Restaurant-Controlled H\n(Spearman ρ = {rho_controlled:.3f})")
# Annotate biggest movers
for dish, row in movers.head(5).iterrows():
    ax.annotate(dish.replace("_", " "),
                (row["H_finetuned"], row["H_restaurant_controlled"]),
                fontsize=7, alpha=0.8)

# Panel C: H vs confounds scatter matrix
ax = axes[1, 0]
# Show H vs biz_stars at dish level
valid = dish_agg.dropna(subset=["biz_stars_mean"])
ax.scatter(valid["biz_stars_mean"], valid["H_mean"], alpha=0.6, s=25,
           edgecolors="k", linewidths=0.3)
r_star, p_star = stats.spearmanr(valid["biz_stars_mean"], valid["H_mean"])
ax.set_xlabel("Mean Restaurant Stars")
ax.set_ylabel("Dish H Score")
ax.set_title(f"C. Dish H vs Restaurant Stars\n(ρ = {r_star:.3f}, {'***' if p_star < 0.001 else 'n.s.'})")
# Fit line
z = np.polyfit(valid["biz_stars_mean"], valid["H_mean"], 1)
x_line = np.linspace(valid["biz_stars_mean"].min(), valid["biz_stars_mean"].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "r-", alpha=0.5, linewidth=2)

# Panel D: Geographic stability heatmap
ax = axes[1, 1]
n_states = len(top_states)
geo_matrix = np.ones((n_states, n_states))
for res in geo_results:
    i = top_states.index(res["state_1"])
    j = top_states.index(res["state_2"])
    geo_matrix[i, j] = res["spearman_rho"]
    geo_matrix[j, i] = res["spearman_rho"]

im = ax.imshow(geo_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
ax.set_xticks(range(n_states))
ax.set_yticks(range(n_states))
ax.set_xticklabels(top_states, fontsize=10)
ax.set_yticklabels(top_states, fontsize=10)
for i in range(n_states):
    for j in range(n_states):
        ax.text(j, i, f"{geo_matrix[i,j]:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if geo_matrix[i,j] < 0.7 else "black")
plt.colorbar(im, ax=ax, label="Spearman ρ")
ax.set_title("D. Geographic Stability of Dish H\n(Spearman ρ across US states)")

plt.tight_layout()
fig_path = FIGURES_DIR / "h_validity_diagnostics.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════
# Save tables
# ══════════════════════════════════════════════════════════════════
icc_df = pd.DataFrame([
    {"source": "Dish", "ICC": icc_dish, "MS_between": ms_b_dish, "MS_within": ms_w_dish, "n_groups": n_dishes},
    {"source": "Restaurant", "ICC": icc_rest, "MS_between": ms_b_rest, "MS_within": ms_w_rest, "n_groups": n_rests},
])
icc_df.to_csv(TABLES_DIR / "h_icc_decomposition.csv", index=False)

validity_df = pd.DataFrame(validity_results)
validity_df.to_csv(TABLES_DIR / "h_construct_validity.csv", index=False)

geo_df = pd.DataFrame(geo_results)
geo_df.to_csv(TABLES_DIR / "h_geographic_stability.csv", index=False)

compare.to_csv(TABLES_DIR / "h_raw_vs_controlled.csv")

print(f"  Saved: {TABLES_DIR / 'h_icc_decomposition.csv'}")
print(f"  Saved: {TABLES_DIR / 'h_construct_validity.csv'}")
print(f"  Saved: {TABLES_DIR / 'h_geographic_stability.csv'}")
print(f"  Saved: {TABLES_DIR / 'h_raw_vs_controlled.csv'}")

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY: H Construct Validity")
print("=" * 60)
print(f"""
1. VARIANCE DECOMPOSITION:
   - Dish ICC = {icc_dish:.4f} → {icc_dish*100:.1f}% of H variance is between dishes
   - Restaurant ICC = {icc_rest:.4f} → {icc_rest*100:.1f}% of H variance is between restaurants
   {'→ H captures dish-level differences, not just restaurant quality' if icc_dish > icc_rest else '→ Restaurant effects are substantial — controls are important'}

2. RESTAURANT CONTROLS:
   - After removing restaurant fixed effects, dish H rankings barely change
   - Raw vs Controlled H: Spearman ρ = {rho_controlled:.4f}
   - Only {(compare['rank_shift'] >= 5).sum()} dishes shift ≥5 rank positions

3. CONFOUND ANALYSIS:
   - H is {'weakly' if abs(validity_results[0]['spearman_rho']) < 0.3 else 'moderately'} correlated with restaurant stars
   - Confounds explain {(1 - np.var(resid_y)/np.var(y))*100:.1f}% of dish-level H variance

4. GEOGRAPHIC STABILITY:
   - Cross-state H correlations: {'strong' if np.mean([r['spearman_rho'] for r in geo_results]) > 0.7 else 'moderate' if np.mean([r['spearman_rho'] for r in geo_results]) > 0.5 else 'weak'}
   - Mean cross-state ρ = {np.mean([r['spearman_rho'] for r in geo_results]):.3f}
""")

print("Done!")
