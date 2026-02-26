"""
18_dei_with_pairwise_h.py — Recompute DEI using Pairwise H
==========================================================
Compares variance decomposition, Pareto frontier, and rankings
between BERT-H and Pairwise-H versions of DEI.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, FIGURES_DIR, TABLES_DIR

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Load data ──────────────────────────────────────────────────────

pw = pd.read_csv(DATA_DIR / "dish_h_pairwise.csv", index_col="dish_id")
env = pd.read_csv(DATA_DIR / "dish_environmental_costs.csv", index_col="dish_id")
bert_h = pd.read_csv(DATA_DIR / "dish_hedonic_scores.csv", index_col="dish_id")

# Merge
df = pw[["H_pairwise", "H_bert"]].join(env[["E_composite", "E_carbon", "E_water",
    "E_energy", "n_ingredients", "total_grams"]], how="inner")
df = df.join(bert_h[["cuisine", "cook_method"]], how="left")

print(f"Dishes with all data: {len(df)}")

# ── Compute DEI for both H measures ────────────────────────────────

for prefix, h_col in [("bert", "H_bert"), ("pw", "H_pairwise")]:
    H = df[h_col]
    E = df["E_composite"]
    df[f"log_H_{prefix}"] = np.log(H)
    df[f"log_E"] = np.log(E)
    df[f"log_DEI_{prefix}"] = np.log(H) - np.log(E)
    df[f"Z_H_{prefix}"] = (H - H.mean()) / H.std()
    df[f"Z_E"] = (E - E.mean()) / E.std()
    df[f"DEI_z_{prefix}"] = df[f"Z_H_{prefix}"] - df["Z_E"]

# ── Variance decomposition comparison ──────────────────────────────

print("\n" + "=" * 70)
print("VARIANCE DECOMPOSITION COMPARISON")
print("=" * 70)

results = {}
for prefix, h_col, label in [("bert", "H_bert", "BERT absolute scoring"),
                               ("pw", "H_pairwise", "Pairwise LLM ranking")]:
    log_h = df[f"log_H_{prefix}"]
    log_e = df["log_E"]
    log_dei = df[f"log_DEI_{prefix}"]

    var_h = log_h.var()
    var_e = log_e.var()
    cov_he = np.cov(log_h, log_e)[0, 1]
    var_dei = var_h + var_e - 2 * cov_he

    share_h = (var_h - cov_he) / var_dei * 100
    share_e = (var_e - cov_he) / var_dei * 100

    H = df[h_col]
    cv_h = H.std() / H.mean() * 100

    print(f"\n── {label} ──")
    print(f"  H range:     [{H.min():.2f}, {H.max():.2f}], CV = {cv_h:.1f}%")
    print(f"  Var(log H):  {var_h:.6f}")
    print(f"  Var(log E):  {var_e:.6f}")
    print(f"  Cov(log H, log E): {cov_he:.6f}")
    print(f"  Var(log DEI): {var_dei:.6f}")
    print(f"  H contribution: {share_h:.1f}%")
    print(f"  E contribution: {share_e:.1f}%")

    results[prefix] = {
        "label": label, "cv_h": cv_h,
        "var_log_h": var_h, "var_log_e": var_e,
        "cov": cov_he, "var_log_dei": var_dei,
        "share_h": share_h, "share_e": share_e,
    }

print(f"\n── IMPROVEMENT ──")
print(f"  H CV:           {results['bert']['cv_h']:.1f}% → {results['pw']['cv_h']:.1f}% "
      f"({results['pw']['cv_h']/results['bert']['cv_h']:.1f}× wider)")
print(f"  H contribution: {results['bert']['share_h']:.1f}% → {results['pw']['share_h']:.1f}%")
print(f"  E contribution: {results['bert']['share_e']:.1f}% → {results['pw']['share_e']:.1f}%")

# ── Rank comparison ─────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("DEI RANK COMPARISON (BERT vs Pairwise)")
print("=" * 70)

df["rank_DEI_bert"] = df["log_DEI_bert"].rank(ascending=False)
df["rank_DEI_pw"] = df["log_DEI_pw"].rank(ascending=False)
df["rank_shift"] = df["rank_DEI_bert"] - df["rank_DEI_pw"]

rho, p = stats.spearmanr(df["log_DEI_bert"], df["log_DEI_pw"])
print(f"\n  Spearman ρ(log DEI_bert, log DEI_pw) = {rho:.3f} (p={p:.2e})")

# H drives rank changes — show correlation of H rank shift to DEI rank shift
h_rank_bert = df["H_bert"].rank(ascending=False)
h_rank_pw = df["H_pairwise"].rank(ascending=False)
h_rank_shift = h_rank_bert - h_rank_pw
rho_shift, _ = stats.spearmanr(h_rank_shift, df["rank_shift"])
print(f"  Spearman ρ(H rank shift, DEI rank shift) = {rho_shift:.3f}")

# Top / Bottom 10 comparison
for label, col, asc in [("Top 10", "log_DEI_pw", False), ("Bottom 10", "log_DEI_pw", True)]:
    print(f"\n  {label} by Pairwise DEI:")
    subset = df.nlargest(10, col) if not asc else df.nsmallest(10, col)
    for i, (idx, row) in enumerate(subset.iterrows(), 1):
        print(f"    {i:2d}. {idx:<25s}  log_DEI_pw={row['log_DEI_pw']:.3f}  "
              f"log_DEI_bert={row['log_DEI_bert']:.3f}  "
              f"H_pw={row['H_pairwise']:.2f}  H_bert={row['H_bert']:.2f}  "
              f"E={row['E_composite']:.4f}")

# ── Pareto frontier with pairwise H ─────────────────────────────────

print(f"\n{'=' * 70}")
print("PARETO FRONTIER (Pairwise H)")
print("=" * 70)

points = df[["E_composite", "H_pairwise"]].values
n = len(points)
is_pareto = np.ones(n, dtype=bool)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
            if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                is_pareto[i] = False
                break

df["is_pareto_pw"] = is_pareto
pareto_dishes = df[df["is_pareto_pw"]].sort_values("E_composite")
print(f"  Pareto-optimal dishes: {len(pareto_dishes)}")
for idx, row in pareto_dishes.iterrows():
    print(f"    {idx:<25s}  H_pw={row['H_pairwise']:.2f}  E={row['E_composite']:.4f}")

# ── Visualization ────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Variance decomposition comparison (bar chart)
ax = axes[0, 0]
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [results["bert"]["share_h"], results["pw"]["share_h"]],
       width, label="H contribution", color="#2196F3")
ax.bar(x + width/2, [results["bert"]["share_e"], results["pw"]["share_e"]],
       width, label="E contribution", color="#FF5722")
ax.set_xticks(x)
ax.set_xticklabels(["BERT\n(absolute)", "Pairwise\n(Bradley-Terry)"])
ax.set_ylabel("% of Var(log DEI)")
ax.set_title("Variance Decomposition: BERT vs Pairwise H")
ax.legend()
for i, (sh, se) in enumerate([(results["bert"]["share_h"], results["bert"]["share_e"]),
                               (results["pw"]["share_h"], results["pw"]["share_e"])]):
    ax.text(i - width/2, sh + 1, f"{sh:.1f}%", ha="center", fontsize=9)
    ax.text(i + width/2, se + 1, f"{se:.1f}%", ha="center", fontsize=9)

# 2. H distribution comparison
ax = axes[0, 1]
ax.hist(df["H_bert"], bins=25, alpha=0.6, label=f"BERT (CV={results['bert']['cv_h']:.1f}%)",
        color="#2196F3", density=True)
ax.hist(df["H_pairwise"], bins=25, alpha=0.6, label=f"Pairwise (CV={results['pw']['cv_h']:.1f}%)",
        color="#FF9800", density=True)
ax.set_xlabel("Hedonic Score (H)")
ax.set_ylabel("Density")
ax.set_title("H Score Distribution")
ax.legend()

# 3. log(DEI) distribution comparison
ax = axes[0, 2]
ax.hist(df["log_DEI_bert"], bins=25, alpha=0.6, label="BERT DEI", color="#2196F3", density=True)
ax.hist(df["log_DEI_pw"], bins=25, alpha=0.6, label="Pairwise DEI", color="#FF9800", density=True)
ax.set_xlabel("log(DEI)")
ax.set_ylabel("Density")
ax.set_title("log(DEI) Distribution")
ax.legend()

# 4. DEI rank comparison
ax = axes[1, 0]
ax.scatter(df["rank_DEI_bert"], df["rank_DEI_pw"], alpha=0.5, s=30, color="steelblue",
           edgecolors="white", linewidth=0.3)
ax.plot([0, 160], [0, 160], "k--", alpha=0.4)
ax.set_xlabel("DEI Rank (BERT H)")
ax.set_ylabel("DEI Rank (Pairwise H)")
ax.set_title(f"DEI Rank Comparison (ρ={rho:.3f})")

# 5. H vs E scatter with Pareto (pairwise)
ax = axes[1, 1]
cuisines = df["cuisine"].dropna().unique()
palette = sns.color_palette("husl", len(cuisines))
for cuisine, color in zip(cuisines, palette):
    mask = df["cuisine"] == cuisine
    ax.scatter(df.loc[mask, "E_composite"], df.loc[mask, "H_pairwise"],
               alpha=0.6, s=40, color=color, label=cuisine, edgecolors="white", linewidth=0.3)
# Pareto frontier line
front = pareto_dishes.sort_values("E_composite")
ax.plot(front["E_composite"], front["H_pairwise"], "k-", linewidth=2, zorder=5)
ax.scatter(front["E_composite"], front["H_pairwise"],
           color="gold", s=100, zorder=6, edgecolors="black", linewidth=1.5)
for idx, row in front.iterrows():
    name = idx.replace("_", " ").title()
    ax.annotate(name, (row["E_composite"], row["H_pairwise"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7, fontweight="bold")
ax.set_xlabel("Environmental Cost (E)")
ax.set_ylabel("Hedonic Score (H, Pairwise)")
ax.set_title("H (Pairwise) vs E with Pareto Frontier")

# 6. log(H) vs log(E) showing spread improvement
ax = axes[1, 2]
ax.scatter(df["log_E"], df["log_H_bert"], alpha=0.5, s=30, color="#2196F3",
           label=f"BERT (Var={results['bert']['var_log_h']:.4f})", edgecolors="white", linewidth=0.3)
ax.scatter(df["log_E"], df["log_H_pw"], alpha=0.5, s=30, color="#FF9800",
           label=f"Pairwise (Var={results['pw']['var_log_h']:.4f})", edgecolors="white", linewidth=0.3)
ax.set_xlabel("log(E)")
ax.set_ylabel("log(H)")
ax.set_title("log(H) vs log(E): BERT vs Pairwise")
ax.legend()

plt.tight_layout()
path = FIGURES_DIR / "dei_bert_vs_pairwise.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {path}")

# ── Save updated DEI data ───────────────────────────────────────────

out_cols = ["H_pairwise", "H_bert", "E_composite", "E_carbon", "E_water", "E_energy",
            "cuisine", "cook_method", "n_ingredients", "total_grams",
            "log_H_pw", "log_E", "log_DEI_pw", "Z_H_pw", "Z_E", "DEI_z_pw",
            "log_H_bert", "log_DEI_bert", "Z_H_bert", "DEI_z_bert",
            "rank_DEI_pw", "rank_DEI_bert", "rank_shift", "is_pareto_pw"]
df[[c for c in out_cols if c in df.columns]].to_csv(DATA_DIR / "dish_DEI_pairwise.csv")
print(f"Saved: {DATA_DIR / 'dish_DEI_pairwise.csv'}")

# ── Summary table ────────────────────────────────────────────────────

summary = pd.DataFrame([
    {"Measure": "BERT (absolute)", "H_CV_pct": results["bert"]["cv_h"],
     "Var_logH": results["bert"]["var_log_h"], "Var_logE": results["bert"]["var_log_e"],
     "H_contribution_pct": results["bert"]["share_h"], "E_contribution_pct": results["bert"]["share_e"]},
    {"Measure": "Pairwise (Bradley-Terry)", "H_CV_pct": results["pw"]["cv_h"],
     "Var_logH": results["pw"]["var_log_h"], "Var_logE": results["pw"]["var_log_e"],
     "H_contribution_pct": results["pw"]["share_h"], "E_contribution_pct": results["pw"]["share_e"]},
])
summary.to_csv(TABLES_DIR / "variance_decomposition_comparison.csv", index=False)
print(f"Saved: {TABLES_DIR / 'variance_decomposition_comparison.csv'}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
