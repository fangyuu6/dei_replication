"""
08b_policy_weights.py — P2-4: Policy-Oriented E Weighting Sensitivity
======================================================================
Tests how DEI rankings change under different policy-motivated E weights.

Weight schemes:
  1. Equal weights (baseline)
  2. Carbon shadow price ($200/tCO2)
  3. Water scarcity adjusted (water stress multiplier)
  4. Planetary boundary weights

Outputs:
  - tables/policy_weight_sensitivity.csv
  - tables/policy_weight_rank_changes.csv
  - figures/policy_weight_sensitivity.png
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

# ── Load ─────────────────────────────────────────────────────────
dei = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv")
print(f"Loaded {len(dei)} dishes")

# ── Define policy weight schemes ─────────────────────────────────
# Baseline: equal weights (1/3 each)
# Carbon social cost: ~$200/tCO2 (Rennert et al. 2022, Nature)
# Water scarcity: global average WSI ≈ 0.4, but ag water is critical
# Energy: grid decarbonisation makes this less urgent

POLICY_SCHEMES = {
    "equal": {"carbon": 1/3, "water": 1/3, "energy": 1/3,
              "label": "Equal weights (baseline)"},
    "carbon_200": {"carbon": 0.60, "water": 0.25, "energy": 0.15,
                   "label": "Carbon priority ($200/tCO₂)"},
    "water_scarce": {"carbon": 0.20, "water": 0.60, "energy": 0.20,
                     "label": "Water scarcity priority"},
    "planetary": {"carbon": 0.50, "water": 0.35, "energy": 0.15,
                  "label": "Planetary boundary (Rockström)"},
    "energy_only": {"carbon": 0.0, "water": 0.0, "energy": 1.0,
                    "label": "Energy only"},
    "carbon_only": {"carbon": 1.0, "water": 0.0, "energy": 0.0,
                    "label": "Carbon only"},
    "water_only": {"carbon": 0.0, "water": 1.0, "energy": 0.0,
                   "label": "Water only"},
}

# ── Compute DEI under each scheme ────────────────────────────────
print("\n" + "=" * 60)
print("Computing DEI under different policy weight schemes")
print("=" * 60)

results = {}
for scheme_name, weights in POLICY_SCHEMES.items():
    w_c, w_w, w_e = weights["carbon"], weights["water"], weights["energy"]
    e_comp = w_c * dei["E_carbon_norm"] + w_w * dei["E_water_norm"] + w_e * dei["E_energy_norm"]
    e_comp = e_comp.clip(lower=1e-6)
    log_dei = np.log(dei["H_mean"]) - np.log(e_comp)
    rank = log_dei.rank(ascending=False)
    results[scheme_name] = {
        "E_composite": e_comp,
        "log_DEI": log_dei,
        "rank": rank,
    }
    print(f"  {weights['label']:40s}: top3 = {dei.loc[rank.nsmallest(3).index, 'dish_id'].tolist()}")

# ── Rank change analysis ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Rank changes across weight schemes")
print("=" * 60)

baseline_rank = results["equal"]["rank"]
rank_changes = pd.DataFrame({"dish_id": dei["dish_id"]})

for scheme_name, res in results.items():
    rank_changes[f"rank_{scheme_name}"] = res["rank"].values
    shift = (res["rank"] - baseline_rank).abs()
    rank_changes[f"shift_{scheme_name}"] = shift.values

# Pairwise rank correlations
schemes = list(POLICY_SCHEMES.keys())
corr_matrix = pd.DataFrame(index=schemes, columns=schemes, dtype=float)
for s1 in schemes:
    for s2 in schemes:
        rho, _ = stats.spearmanr(results[s1]["rank"], results[s2]["rank"])
        corr_matrix.loc[s1, s2] = rho

print("\n  Pairwise Spearman ρ between weight schemes:")
print(corr_matrix.to_string(float_format=lambda x: f"{x:.3f}"))

# Max rank shift from baseline
for scheme_name in schemes:
    if scheme_name == "equal":
        continue
    col = f"shift_{scheme_name}"
    max_shift = rank_changes[col].max()
    mean_shift = rank_changes[col].mean()
    n_shift5 = (rank_changes[col] >= 5).sum()
    print(f"\n  {POLICY_SCHEMES[scheme_name]['label']}:")
    print(f"    Mean |shift|: {mean_shift:.1f}, Max: {max_shift:.0f}, ≥5: {n_shift5}")
    # Top movers
    top = rank_changes.nlargest(3, col)
    for _, row in top.iterrows():
        print(f"    {row['dish_id']:25s}: equal_rank={row['rank_equal']:.0f} → "
              f"{scheme_name}_rank={row[f'rank_{scheme_name}']:.0f} (shift={row[col]:.0f})")

# ── Save ─────────────────────────────────────────────────────────
rank_changes.to_csv(TABLES_DIR / "policy_weight_rank_changes.csv", index=False)
corr_matrix.to_csv(TABLES_DIR / "policy_weight_correlations.csv")

# ── Figures ──────────────────────────────────────────────────────
print("\nGenerating figures...")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# A: Correlation heatmap
ax = axes[0]
n = len(schemes)
im = ax.imshow(corr_matrix.values.astype(float), cmap="RdYlGn", vmin=0.7, vmax=1.0)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
labels = [POLICY_SCHEMES[s]["label"][:20] for s in schemes]
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr_matrix.values[i,j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if float(corr_matrix.values[i,j]) < 0.85 else "black")
plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
ax.set_title("A. Rank Correlation Across\nWeight Schemes", fontsize=11)

# B: Rank shift distribution (carbon_200 vs equal)
ax = axes[1]
for scheme_name, color in [("carbon_200", "#e74c3c"), ("water_scarce", "#3498db"),
                            ("planetary", "#2ecc71")]:
    shifts = rank_changes[f"shift_{scheme_name}"]
    ax.hist(shifts, bins=20, alpha=0.5, color=color, edgecolor="white",
            label=POLICY_SCHEMES[scheme_name]["label"][:25])
ax.set_xlabel("|Rank Shift| from Equal Weights")
ax.set_ylabel("Number of Dishes")
ax.set_title("B. Rank Shift Distributions")
ax.legend(fontsize=8)

# C: Top/Bottom stability across schemes
ax = axes[2]
# For each dish, count how many schemes put it in top 20
top20_counts = pd.Series(0, index=dei.index)
for scheme_name, res in results.items():
    top20 = res["rank"].nsmallest(20).index
    top20_counts[top20] += 1

always_top = dei.loc[top20_counts[top20_counts == len(schemes)].index, "dish_id"].tolist()
mostly_top = dei.loc[top20_counts[top20_counts >= len(schemes)-1].index, "dish_id"].tolist()

top20_summary = top20_counts.value_counts().sort_index(ascending=False)
ax.bar(top20_summary.index, top20_summary.values, color="seagreen", edgecolor="white")
ax.set_xlabel(f"Number of Schemes in Top 20\n(out of {len(schemes)})")
ax.set_ylabel("Number of Dishes")
ax.set_title("C. How Often Dishes Appear in Top 20")
ax.set_xticks(range(len(schemes)+1))

plt.tight_layout()
fig_path = FIGURES_DIR / "policy_weight_sensitivity.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close()

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: Policy Weight Sensitivity")
print("=" * 60)
print(f"""
  Weight schemes tested: {len(POLICY_SCHEMES)}
  Mean pairwise ρ: {corr_matrix.values[np.triu_indices(n, k=1)].astype(float).mean():.3f}

  Always in top 20 (all schemes): {always_top[:10]}
  Mostly in top 20 (≥{len(schemes)-1} schemes): {len(mostly_top)} dishes

  Key finding: Rankings are {'highly robust' if corr_matrix.values[np.triu_indices(n, k=1)].astype(float).mean() > 0.9 else 'moderately robust' if corr_matrix.values[np.triu_indices(n, k=1)].astype(float).mean() > 0.8 else 'sensitive'} to weight scheme choice.
""")
print("Done!")
