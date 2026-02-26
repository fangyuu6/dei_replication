#!/usr/bin/env python3
"""
22e_rebuild_dei.py — Rebuild combined DEI dataset with ~2,500+ dishes
=====================================================================
Merges:
  - Pairwise H scores (data/dish_h_pairwise_v2.csv)
  - Environmental costs for new dishes (data/expanded_dish_env_costs_v2.csv)
  - Environmental costs for original dishes (data/combined_dish_DEI.csv)

Produces:
  - data/combined_dish_DEI_v2.csv  (full DEI for all dishes)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 70, flush=True)
print("22e — Rebuild Combined DEI Dataset", flush=True)
print("=" * 70, flush=True)

# ── Load H scores ────────────────────────────────────────────────
print("\n── Loading H scores ──", flush=True)
h_df = pd.read_csv(DATA / "dish_h_pairwise_v2.csv", index_col="dish_id")
print(f"  H scores: {len(h_df)} dishes", flush=True)
print(f"  H range: [{h_df['H_pairwise'].min():.2f}, {h_df['H_pairwise'].max():.2f}]",
      flush=True)

# ── Load E scores ────────────────────────────────────────────────
print("\n── Loading E scores ──", flush=True)

# Existing dishes
existing = pd.read_csv(DATA / "combined_dish_DEI.csv")
print(f"  Existing DEI: {len(existing)} dishes", flush=True)

# New dishes
new_env = pd.read_csv(DATA / "expanded_dish_env_costs_v2.csv")
print(f"  New E scores: {len(new_env)} dishes", flush=True)

# ── Build unified E table ────────────────────────────────────────
print("\n── Building unified E table ──", flush=True)

# Existing: extract E columns
exist_e = existing[["dish_id", "E_carbon", "E_water", "E_energy"]].copy()
exist_e["cuisine"] = existing.get("cuisine", "")
exist_e["cook_method"] = existing.get("cook_method", "")
exist_e["source"] = existing.get("source", "original")

# New: extract E columns
new_e = new_env[["dish_id", "E_carbon", "E_water", "E_energy",
                  "primary_cuisine", "cook_method"]].copy()
new_e.rename(columns={"primary_cuisine": "cuisine"}, inplace=True)
new_e["source"] = "worldcuisines_llm"

# Combine
all_e = pd.concat([exist_e, new_e], ignore_index=True)
all_e = all_e.drop_duplicates("dish_id", keep="first")
print(f"  Combined E: {len(all_e)} dishes", flush=True)

# ── Normalize E on combined range ────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.01, 1.0))
for col in ["E_carbon", "E_water", "E_energy"]:
    all_e[col + "_norm"] = scaler.fit_transform(all_e[[col]])

all_e["E_composite"] = (all_e["E_carbon_norm"] + all_e["E_water_norm"] +
                         all_e["E_energy_norm"]) / 3
print(f"  E_composite range: [{all_e['E_composite'].min():.4f}, "
      f"{all_e['E_composite'].max():.4f}]", flush=True)

# ── Merge H + E → DEI ───────────────────────────────────────────
print("\n── Computing DEI ──", flush=True)

all_e = all_e.set_index("dish_id")
common_ids = h_df.index.intersection(all_e.index)
print(f"  Dishes with both H and E: {len(common_ids)}", flush=True)

dei = pd.DataFrame(index=common_ids)
dei["H_mean"] = h_df.loc[common_ids, "H_pairwise"]
dei["BT_strength"] = h_df.loc[common_ids, "BT_strength"]
dei["E_composite"] = all_e.loc[common_ids, "E_composite"]
dei["E_carbon"] = all_e.loc[common_ids, "E_carbon"]
dei["E_water"] = all_e.loc[common_ids, "E_water"]
dei["E_energy"] = all_e.loc[common_ids, "E_energy"]
dei["cuisine"] = all_e.loc[common_ids, "cuisine"]
dei["cook_method"] = all_e.loc[common_ids, "cook_method"]
dei["source"] = all_e.loc[common_ids, "source"]

# Log transforms
dei["log_H"] = np.log(dei["H_mean"].clip(lower=1.0))
dei["log_E"] = np.log(dei["E_composite"].clip(lower=1e-6))
dei["log_DEI"] = dei["log_H"] - dei["log_E"]

# Z-score DEI
dei["Z_H"] = (dei["log_H"] - dei["log_H"].mean()) / dei["log_H"].std()
dei["Z_E"] = (dei["log_E"] - dei["log_E"].mean()) / dei["log_E"].std()
dei["DEI_z"] = dei["Z_H"] - dei["Z_E"]

# Tier assignment (quintiles)
dei["rank_DEI"] = dei["log_DEI"].rank(ascending=False).astype(int)
dei["rank_invE"] = dei["E_composite"].rank(ascending=True).astype(int)
dei["rank_shift"] = dei["rank_invE"] - dei["rank_DEI"]

quintile_labels = ["A", "B", "C", "D", "E"]
dei["DEI_tier"] = pd.qcut(dei["log_DEI"], 5, labels=quintile_labels[::-1])

# Pareto frontier
def find_pareto(df):
    """Find Pareto-optimal dishes (high H, low E)."""
    pareto = []
    sorted_df = df.sort_values("log_H", ascending=False)
    min_e = float("inf")
    for idx, row in sorted_df.iterrows():
        if row["log_E"] < min_e:
            pareto.append(idx)
            min_e = row["log_E"]
    return pareto

pareto_ids = find_pareto(dei)
dei["is_pareto"] = dei.index.isin(pareto_ids)

print(f"\n  DEI computed for {len(dei)} dishes", flush=True)
print(f"  log_DEI range: [{dei['log_DEI'].min():.2f}, {dei['log_DEI'].max():.2f}]",
      flush=True)
print(f"  Pareto-optimal: {dei['is_pareto'].sum()} dishes", flush=True)

# ── Variance decomposition ───────────────────────────────────────
print("\n── Variance decomposition (Shapley) ──", flush=True)

var_logH = dei["log_H"].var()
var_logE = dei["log_E"].var()
cov_HE = np.cov(dei["log_H"], dei["log_E"])[0, 1]
var_logDEI = dei["log_DEI"].var()

# Shapley values for Var(logDEI) = Var(logH) + Var(logE) - 2Cov(logH,logE)
shapley_H = var_logH - cov_HE
shapley_E = var_logE - cov_HE
total_shapley = shapley_H + shapley_E
pct_H = shapley_H / total_shapley * 100
pct_E = shapley_E / total_shapley * 100

print(f"  Var(log H) = {var_logH:.4f}", flush=True)
print(f"  Var(log E) = {var_logE:.4f}", flush=True)
print(f"  Cov(log H, log E) = {cov_HE:.4f}", flush=True)
print(f"  Var(log DEI) = {var_logDEI:.4f}", flush=True)
print(f"  Shapley H contribution: {pct_H:.1f}%", flush=True)
print(f"  Shapley E contribution: {pct_E:.1f}%", flush=True)

# Correlation H vs E
rho_HE, p_HE = stats.spearmanr(dei["log_H"], dei["log_E"])
r_HE = np.corrcoef(dei["log_H"], dei["log_E"])[0, 1]
print(f"  r(log H, log E) = {r_HE:.3f}", flush=True)
print(f"  ρ(log H, log E) = {rho_HE:.3f} (p={p_HE:.2e})", flush=True)

# ρ(log DEI, 1/E) — how much DEI is just 1/E ranking
rho_invE, _ = stats.spearmanr(dei["log_DEI"], 1 / dei["E_composite"])
print(f"  ρ(log DEI, 1/E) = {rho_invE:.3f}", flush=True)

# ── Save ─────────────────────────────────────────────────────────
dei.index.name = "dish_id"
dei.to_csv(DATA / "combined_dish_DEI_v2.csv")
print(f"\n  Saved: data/combined_dish_DEI_v2.csv ({len(dei)} dishes)", flush=True)

# ── Summary by cuisine ──────────────────────────────────────────
print("\n── Cuisine summary ──", flush=True)
cuisine_stats = dei.groupby("cuisine").agg(
    n=("log_DEI", "count"),
    mean_H=("H_mean", "mean"),
    mean_E=("E_composite", "mean"),
    mean_DEI=("log_DEI", "mean"),
).sort_values("mean_DEI", ascending=False)

print(f"  {'Cuisine':25s} {'N':>4s} {'H':>6s} {'E':>8s} {'logDEI':>8s}", flush=True)
print(f"  {'-'*55}", flush=True)
for cuisine, row in cuisine_stats.head(20).iterrows():
    print(f"  {cuisine:25s} {int(row['n']):4d} {row['mean_H']:6.2f} "
          f"{row['mean_E']:8.4f} {row['mean_DEI']:8.2f}", flush=True)

# ── Top/Bottom dishes ────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print("TOP 20 DISHES by log(DEI)", flush=True)
for i, (idx, row) in enumerate(dei.nlargest(20, "log_DEI").iterrows(), 1):
    src = "★" if row["source"] == "worldcuisines_llm" else " "
    p = "P" if row["is_pareto"] else " "
    print(f"  {i:2d}. {src}{p} {idx:<30s} logDEI={row['log_DEI']:.2f} "
          f"H={row['H_mean']:.2f} E={row['E_composite']:.4f} ({row['cuisine']})",
          flush=True)

print(f"\nBOTTOM 20 DISHES by log(DEI)", flush=True)
for i, (idx, row) in enumerate(dei.nsmallest(20, "log_DEI").iterrows(), 1):
    src = "★" if row["source"] == "worldcuisines_llm" else " "
    print(f"  {i:2d}. {src}  {idx:<30s} logDEI={row['log_DEI']:.2f} "
          f"H={row['H_mean']:.2f} E={row['E_composite']:.4f} ({row['cuisine']})",
          flush=True)

print(f"\nPARETO-OPTIMAL DISHES ({dei['is_pareto'].sum()}):", flush=True)
for idx, row in dei[dei["is_pareto"]].sort_values("log_DEI", ascending=False).iterrows():
    src = "★" if row["source"] == "worldcuisines_llm" else " "
    print(f"  {src} {idx:<30s} logDEI={row['log_DEI']:.2f} "
          f"H={row['H_mean']:.2f} E={row['E_composite']:.4f} ({row['cuisine']})",
          flush=True)

# ── Plot ─────────────────────────────────────────────────────────
print("\n── Plotting ──", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.1)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. H vs E scatter
ax = axes[0, 0]
colors = dei["log_DEI"]
sc = ax.scatter(dei["log_E"], dei["log_H"], c=colors, cmap="RdYlGn",
                alpha=0.4, s=15, edgecolors="white", linewidth=0.2)
# Mark Pareto
pareto_df = dei[dei["is_pareto"]]
ax.scatter(pareto_df["log_E"], pareto_df["log_H"], c="red", s=60,
           marker="D", edgecolors="black", linewidth=0.5, zorder=5)
for idx, row in pareto_df.iterrows():
    ax.annotate(idx.replace("_", " "), (row["log_E"], row["log_H"]),
                fontsize=5, ha="left", va="bottom")
ax.set_xlabel("log(E)")
ax.set_ylabel("log(H)")
ax.set_title(f"H vs E ({len(dei)} dishes)")
plt.colorbar(sc, ax=ax, label="log(DEI)")

# 2. DEI distribution
ax = axes[0, 1]
ax.hist(dei["log_DEI"], bins=50, alpha=0.7, color="#2196F3", edgecolor="white")
ax.axvline(dei["log_DEI"].mean(), color="red", linestyle="--", label=f"Mean={dei['log_DEI'].mean():.2f}")
ax.set_xlabel("log(DEI)")
ax.set_ylabel("Count")
ax.set_title("DEI Distribution")
ax.legend()

# 3. Cuisine comparison (top 15 by n)
ax = axes[1, 0]
top_cuisines = dei["cuisine"].value_counts().head(15).index.tolist()
cuisine_data = dei[dei["cuisine"].isin(top_cuisines)]
order = cuisine_data.groupby("cuisine")["log_DEI"].median().sort_values(ascending=False).index
sns.boxplot(data=cuisine_data, x="log_DEI", y="cuisine", order=order,
            ax=ax, palette="viridis", showfliers=False)
ax.set_xlabel("log(DEI)")
ax.set_title("DEI by Cuisine (top 15)")

# 4. Variance decomposition pie
ax = axes[1, 1]
ax.pie([pct_H, pct_E], labels=[f"H: {pct_H:.1f}%", f"E: {pct_E:.1f}%"],
       colors=["#FF9800", "#2196F3"], autopct="%1.1f%%", startangle=90)
ax.set_title(f"Variance Decomposition of log(DEI)\n(n={len(dei)})")

plt.suptitle(f"DEI Analysis: {len(dei)} Dishes, {cuisine_stats.shape[0]} Cuisines",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES / "dei_v2_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: results/figures/dei_v2_overview.png", flush=True)

print(f"\n{'='*70}", flush=True)
print("DONE", flush=True)
print(f"{'='*70}", flush=True)
