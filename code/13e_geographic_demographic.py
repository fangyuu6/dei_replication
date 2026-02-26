"""
13e_geographic_demographic.py — Geographic & Demographic Bias Analysis
=======================================================================
Addresses C10: Yelp data is biased toward young, urban, US consumers.

Analyses:
  A. Geographic concentration quantification
  B. City-level H consistency
  C. Price-tier stratification
  D. Cuisine representation by region

Outputs:
  - tables/geographic_concentration.csv
  - tables/city_h_consistency.csv
  - figures/geographic_heatmap.png
  - figures/city_h_stability.png
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
print("13e  GEOGRAPHIC & DEMOGRAPHIC BIAS ANALYSIS")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
restaurants = pd.read_parquet(DATA_DIR / "restaurants.parquet")
combined = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")

# Merge geographic info
mentions_geo = mentions.merge(
    restaurants[["business_id", "city", "state", "stars"]].rename(
        columns={"stars": "biz_stars"}
    ),
    on="business_id", how="left"
)
print(f"Mentions with geo: {mentions_geo['state'].notna().sum():,}/{len(mentions_geo):,}")

# ══════════════════════════════════════════════════════════════════
# A. GEOGRAPHIC CONCENTRATION
# ══════════════════════════════════════════════════════════════════
print("\n── A. Geographic Concentration ──")

# State-level
state_stats = mentions_geo.groupby("state").agg(
    n_mentions=("dish_id", "count"),
    n_restaurants=("business_id", "nunique"),
    n_dishes=("dish_id", "nunique"),
    H_mean=("hedonic_score_finetuned", "mean"),
).sort_values("n_mentions", ascending=False)

print("  State distribution:")
for state, row in state_stats.head(10).iterrows():
    pct = row["n_mentions"] / len(mentions_geo) * 100
    print(f"    {state}: {row['n_mentions']:,} mentions ({pct:.1f}%), "
          f"{row['n_restaurants']:,} restaurants, H={row['H_mean']:.3f}")

# Gini coefficient for geographic concentration
def gini(values):
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

state_gini = gini(state_stats["n_mentions"].values)
print(f"\n  State-level Gini coefficient: {state_gini:.3f}")

# City-level
city_stats = mentions_geo.groupby("city").agg(
    n_mentions=("dish_id", "count"),
    n_restaurants=("business_id", "nunique"),
    n_dishes=("dish_id", "nunique"),
    H_mean=("hedonic_score_finetuned", "mean"),
    state=("state", "first"),
).sort_values("n_mentions", ascending=False)

city_gini = gini(city_stats["n_mentions"].values)
print(f"  City-level Gini coefficient: {city_gini:.3f}")
print(f"  Top 10 cities cover {city_stats.head(10)['n_mentions'].sum()/len(mentions_geo)*100:.1f}% of mentions")

# Save
geo_summary = pd.DataFrame({
    "metric": ["n_states", "n_cities", "state_gini", "city_gini",
               "top10_city_pct", "top3_state_pct"],
    "value": [
        state_stats.shape[0],
        city_stats.shape[0],
        state_gini,
        city_gini,
        city_stats.head(10)["n_mentions"].sum() / len(mentions_geo) * 100,
        state_stats.head(3)["n_mentions"].sum() / len(mentions_geo) * 100,
    ]
})
geo_summary.to_csv(TABLES_DIR / "geographic_concentration.csv", index=False)

# ══════════════════════════════════════════════════════════════════
# B. CITY-LEVEL H CONSISTENCY
# ══════════════════════════════════════════════════════════════════
print("\n── B. City-Level H Consistency ──")

# Select top cities with enough coverage
min_mentions_per_city = 500
min_dishes_per_city = 30
eligible_cities = city_stats[
    (city_stats["n_mentions"] >= min_mentions_per_city) &
    (city_stats["n_dishes"] >= min_dishes_per_city)
].index.tolist()

print(f"  Eligible cities (≥{min_mentions_per_city} mentions, ≥{min_dishes_per_city} dishes): {len(eligible_cities)}")

# Compute dish-level H for each city
city_dish_h = {}
for city in eligible_cities[:15]:  # Top 15 cities
    city_m = mentions_geo[mentions_geo["city"] == city]
    dish_h = city_m.groupby("dish_id")["hedonic_score_finetuned"].mean()
    # Only keep dishes with ≥5 mentions in this city
    dish_h = dish_h[city_m.groupby("dish_id").size() >= 5]
    city_dish_h[city] = dish_h

# Pairwise Spearman correlations between cities
city_pairs = []
city_names = list(city_dish_h.keys())
for i, c1 in enumerate(city_names):
    for c2 in city_names[i + 1:]:
        # Find common dishes
        common = city_dish_h[c1].index.intersection(city_dish_h[c2].index)
        if len(common) < 10:
            continue
        rho, p = sp_stats.spearmanr(
            city_dish_h[c1][common], city_dish_h[c2][common]
        )
        city_pairs.append({
            "city_1": c1, "city_2": c2,
            "n_common_dishes": len(common),
            "spearman_rho": rho, "p_value": p,
        })

city_pairs_df = pd.DataFrame(city_pairs)
if len(city_pairs_df) > 0:
    city_pairs_df.to_csv(TABLES_DIR / "city_h_consistency.csv", index=False)

    mean_rho = city_pairs_df["spearman_rho"].mean()
    min_rho = city_pairs_df["spearman_rho"].min()
    max_rho = city_pairs_df["spearman_rho"].max()
    n_sig = (city_pairs_df["p_value"] < 0.05).sum()

    print(f"  City-pair H correlations: mean ρ={mean_rho:.3f}, "
          f"range [{min_rho:.3f}, {max_rho:.3f}]")
    print(f"  Significant (p<0.05): {n_sig}/{len(city_pairs_df)}")

# ══════════════════════════════════════════════════════════════════
# C. STATE-LEVEL DEI STABILITY
# ══════════════════════════════════════════════════════════════════
print("\n── C. State-Level DEI Stability ──")

# For each state, compute dish-level H and compare to overall
top_states = state_stats.head(5).index.tolist()
state_dei_results = []

for state in top_states:
    state_m = mentions_geo[mentions_geo["state"] == state]
    state_dish_h = state_m.groupby("dish_id").agg(
        H_state=("hedonic_score_finetuned", "mean"),
        n=("hedonic_score_finetuned", "count"),
    )
    state_dish_h = state_dish_h[state_dish_h["n"] >= 5]

    # Merge with overall H and E
    merged = state_dish_h.merge(
        combined[["dish_id", "H_mean", "E_composite", "log_DEI"]].set_index("dish_id"),
        left_index=True, right_index=True, how="inner"
    )

    if len(merged) < 20:
        continue

    # State-specific DEI
    merged["log_DEI_state"] = np.log(merged["H_state"]) - np.log(merged["E_composite"])

    # Correlation with overall DEI
    rho_dei, p_dei = sp_stats.spearmanr(merged["log_DEI"], merged["log_DEI_state"])
    rho_h, p_h = sp_stats.spearmanr(merged["H_mean"], merged["H_state"])

    # State H variance
    h_cv = merged["H_state"].std() / merged["H_state"].mean() * 100

    state_dei_results.append({
        "state": state,
        "n_dishes": len(merged),
        "H_cv_pct": h_cv,
        "H_rho_vs_overall": rho_h,
        "DEI_rho_vs_overall": rho_dei,
        "mean_H_state": merged["H_state"].mean(),
        "mean_H_overall": merged["H_mean"].mean(),
    })

state_dei_df = pd.DataFrame(state_dei_results)
if len(state_dei_df) > 0:
    print(state_dei_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# D. CUISINE GEOGRAPHIC SPREAD
# ══════════════════════════════════════════════════════════════════
print("\n── D. Cuisine Geographic Spread ──")

cuisine_geo = mentions_geo.groupby("cuisine").agg(
    n_mentions=("dish_id", "count"),
    n_states=("state", "nunique"),
    n_cities=("city", "nunique"),
    top_state=("state", lambda x: x.value_counts().index[0] if len(x) > 0 else ""),
    top_state_pct=("state", lambda x: x.value_counts().iloc[0] / len(x) * 100 if len(x) > 0 else 0),
).sort_values("n_mentions", ascending=False)

print("\n  Cuisine geographic spread:")
for cuisine, row in cuisine_geo.head(12).iterrows():
    print(f"    {cuisine}: {row['n_states']} states, {row['n_cities']} cities, "
          f"top={row['top_state']} ({row['top_state_pct']:.0f}%)")

# Flag cuisines with low geographic representativeness
low_rep = cuisine_geo[cuisine_geo["n_states"] < 3]
if len(low_rep) > 0:
    print(f"\n  Low geographic representativeness (<3 states):")
    for cuisine, row in low_rep.iterrows():
        print(f"    {cuisine}: {row['n_states']} states only")

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n── Generating plots ──")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (0,0): State distribution
ax = axes[0, 0]
top_n = min(12, len(state_stats))
ax.barh(range(top_n), state_stats["n_mentions"].values[:top_n],
        color="steelblue", alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(state_stats.index[:top_n])
ax.set_xlabel("Number of mentions")
ax.set_title(f"Geographic distribution (Gini={state_gini:.2f})")
ax.invert_yaxis()

# (0,1): City H consistency matrix
if len(city_pairs_df) > 0:
    ax = axes[0, 1]
    # Build correlation matrix
    all_cities = sorted(set(city_pairs_df["city_1"]) | set(city_pairs_df["city_2"]))
    n_c = min(8, len(all_cities))
    all_cities = all_cities[:n_c]
    corr_matrix = np.eye(n_c)
    for _, row in city_pairs_df.iterrows():
        if row["city_1"] in all_cities and row["city_2"] in all_cities:
            i = all_cities.index(row["city_1"])
            j = all_cities.index(row["city_2"])
            corr_matrix[i, j] = row["spearman_rho"]
            corr_matrix[j, i] = row["spearman_rho"]

    im = ax.imshow(corr_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(all_cities, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels(all_cities, fontsize=7)
    for i in range(n_c):
        for j in range(n_c):
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if corr_matrix[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Cross-city H consistency")

# (1,0): State DEI stability
if len(state_dei_df) > 0:
    ax = axes[1, 0]
    x = range(len(state_dei_df))
    ax.bar(x, state_dei_df["DEI_rho_vs_overall"], color="steelblue", alpha=0.7,
           label="DEI ρ")
    ax.bar([i + 0.3 for i in x], state_dei_df["H_rho_vs_overall"],
           width=0.3, color="coral", alpha=0.7, label="H ρ")
    ax.set_xticks(x)
    ax.set_xticklabels(state_dei_df["state"])
    ax.set_ylabel("Spearman ρ vs overall")
    ax.set_title("State-level DEI and H consistency")
    ax.legend()
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)

# (1,1): Cuisine geographic spread
ax = axes[1, 1]
top_cuisines = cuisine_geo.head(12)
ax.scatter(top_cuisines["n_states"], top_cuisines["n_cities"],
           s=top_cuisines["n_mentions"] / 50, alpha=0.6, color="steelblue")
for cuisine, row in top_cuisines.iterrows():
    ax.annotate(cuisine, (row["n_states"], row["n_cities"]),
                fontsize=7, alpha=0.8)
ax.set_xlabel("Number of states")
ax.set_ylabel("Number of cities")
ax.set_title("Cuisine geographic coverage")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "geographic_heatmap.png", dpi=150)
plt.close()
print("  Saved geographic_heatmap.png")

# City stability plot
if len(city_pairs_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(city_pairs_df["spearman_rho"], bins=20, color="steelblue",
            alpha=0.7, edgecolor="white")
    ax.axvline(city_pairs_df["spearman_rho"].mean(), color="red", linestyle="--",
               label=f"Mean ρ={city_pairs_df['spearman_rho'].mean():.3f}")
    ax.set_xlabel("Spearman ρ (cross-city H consistency)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cross-city hedonic score consistency ({len(city_pairs_df)} pairs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "city_h_stability.png", dpi=150)
    plt.close()
    print("  Saved city_h_stability.png")

print("\n" + "=" * 70)
print("13e COMPLETE")
print("=" * 70)
