"""
05_dei_computation.py - DEI Calculation and Core Analysis
=========================================================
Combines hedonic scores (H) and environmental costs (E) to compute DEI.

Key methodological features:
  - log(DEI) = log(H) - log(E)  as primary specification
    (avoids ratio domination when Var(E) >> Var(H))
  - Z-score standardised DEI_z = Z(H) - Z(E)  as robustness check
  - Variance decomposition: Var contribution of H vs E to DEI
  - OLS on log(DEI) with cuisine + method controls

Core analyses:
  1. DEI distribution (overall and by cuisine)
  2. Pareto frontier identification
  3. "Waste space" quantification
  4. Regression: determinants of DEI (log-scale)
  5. Sensitivity analysis across weighting schemes
  6. H vs E variance contribution analysis

Outputs:
  - data/dish_DEI_scores.csv
  - results/figures/dei_*.png
  - results/tables/dei_*.csv
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
from config import (
    DATA_DIR, FIGURES_DIR, TABLES_DIR,
    E_WEIGHT_SCHEMES, DEI_TIERS, MIN_REVIEWS_PER_DISH,
)

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_dei(hedonic: pd.DataFrame, env_cost: pd.DataFrame) -> pd.DataFrame:
    """Merge hedonic and environmental data, compute DEI in multiple forms.

    Computes:
      - DEI       = H / E              (raw ratio, kept for backward compat)
      - log_DEI   = log(H) - log(E)    (PRIMARY specification)
      - DEI_z     = Z(H) - Z(E)        (standardised difference)
    """
    # Merge on dish_id
    merged = hedonic.merge(env_cost, left_index=True, right_index=True,
                           how="inner", suffixes=("_h", "_e"))
    print(f"  Merged dishes (H + E available): {len(merged)}")

    H = merged["H_mean"]
    E = merged["E_composite"]

    # ── 1. Raw ratio (backward-compatible) ──
    merged["DEI"] = H / E

    # ── 2. Log-transformed DEI (PRIMARY) ──
    # log(DEI) = log(H) − log(E)
    merged["log_H"] = np.log(H)
    merged["log_E"] = np.log(E)
    merged["log_DEI"] = merged["log_H"] - merged["log_E"]

    # ── 3. Z-score standardised DEI ──
    merged["Z_H"] = (H - H.mean()) / H.std()
    merged["Z_E"] = (E - E.mean()) / E.std()
    merged["DEI_z"] = merged["Z_H"] - merged["Z_E"]

    # ── Weighting scheme variants (on log scale) ──
    for scheme_name in E_WEIGHT_SCHEMES:
        e_col = f"E_composite_{scheme_name}"
        if e_col in merged.columns:
            merged[f"DEI_{scheme_name}"] = H / merged[e_col]
            merged[f"log_DEI_{scheme_name}"] = np.log(H) - np.log(merged[e_col])

    # ── DEI tier (quintile on log_DEI) ──
    merged["DEI_tier"] = pd.qcut(
        merged["log_DEI"], q=5, labels=DEI_TIERS, duplicates="drop"
    )

    return merged


def variance_decomposition(df: pd.DataFrame):
    """Decompose DEI variance into H and E contributions.

    For log(DEI) = log(H) − log(E):
        Var(log DEI) = Var(log H) + Var(log E) − 2·Cov(log H, log E)

    Reports the fraction explained by each component.
    """
    print("\n── Variance Decomposition ──")

    log_h = df["log_H"]
    log_e = df["log_E"]
    log_dei = df["log_DEI"]

    var_log_h = log_h.var()
    var_log_e = log_e.var()
    cov_he = np.cov(log_h, log_e)[0, 1]
    var_log_dei = log_dei.var()

    # Theoretical: Var(log DEI) = Var(log H) + Var(log E) - 2*Cov
    var_theoretical = var_log_h + var_log_e - 2 * cov_he

    # Marginal contributions (Shapley-like decomposition)
    # Share_H = [Var(log H) - Cov] / Var(log DEI)
    # Share_E = [Var(log E) - Cov] / Var(log DEI)
    share_h = (var_log_h - cov_he) / var_theoretical * 100
    share_e = (var_log_e - cov_he) / var_theoretical * 100

    print(f"  Var(log H)   = {var_log_h:.6f}")
    print(f"  Var(log E)   = {var_log_e:.6f}")
    print(f"  Cov(log H, log E) = {cov_he:.6f}")
    print(f"  Var(log DEI) = {var_log_dei:.6f}  (theoretical: {var_theoretical:.6f})")
    print(f"")
    print(f"  H contribution:  {share_h:.1f}%")
    print(f"  E contribution:  {share_e:.1f}%")
    print(f"  Cor(log H, log E): {np.corrcoef(log_h, log_e)[0,1]:.3f}")

    # Also report raw-scale stats for context
    print(f"\n  Raw-scale summary:")
    print(f"    H  range: [{df['H_mean'].min():.2f}, {df['H_mean'].max():.2f}], "
          f"CV = {df['H_mean'].std()/df['H_mean'].mean()*100:.1f}%")
    print(f"    E  range: [{df['E_composite'].min():.4f}, {df['E_composite'].max():.4f}], "
          f"CV = {df['E_composite'].std()/df['E_composite'].mean()*100:.1f}%")
    print(f"    log(H) range: [{log_h.min():.3f}, {log_h.max():.3f}], "
          f"CV = {log_h.std()/log_h.mean()*100:.1f}%")
    print(f"    log(E) range: [{log_e.min():.3f}, {log_e.max():.3f}], "
          f"CV = {log_e.std()/abs(log_e.mean())*100:.1f}%")

    result = {
        "var_log_H": var_log_h, "var_log_E": var_log_e,
        "cov_log_HE": cov_he, "var_log_DEI": var_log_dei,
        "H_pct_contribution": share_h, "E_pct_contribution": share_e,
        "H_CV_pct": df['H_mean'].std()/df['H_mean'].mean()*100,
        "E_CV_pct": df['E_composite'].std()/df['E_composite'].mean()*100,
    }
    pd.DataFrame([result]).to_csv(TABLES_DIR / "variance_decomposition.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'variance_decomposition.csv'}")

    return result


def pareto_frontier(df: pd.DataFrame, h_col="H_mean", e_col="E_composite"):
    """Identify Pareto-optimal dishes (max H for given E, or min E for given H).

    A dish is Pareto-optimal if no other dish is both tastier AND more
    eco-friendly.
    """
    points = df[[e_col, h_col]].values
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if: j has lower E AND higher H
            if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
                if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                    is_pareto[i] = False
                    break

    return is_pareto


def waste_space_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify "waste space" - how much E could be reduced without losing H.

    For each dish, find the best benchmark: a dish with H >= 95% of this dish's H
    but lower E. The difference is the "reducible E".
    """
    results = []
    for idx, row in df.iterrows():
        h_threshold = row["H_mean"] * 0.95
        # Find dishes with similar or better H but lower E
        benchmarks = df[
            (df["H_mean"] >= h_threshold) &
            (df["E_composite"] < row["E_composite"])
        ]
        if len(benchmarks) > 0:
            best = benchmarks.loc[benchmarks["E_composite"].idxmin()]
            results.append({
                "dish_id": idx,
                "E_current": row["E_composite"],
                "E_best_benchmark": best["E_composite"],
                "E_reducible": row["E_composite"] - best["E_composite"],
                "E_reduction_pct": (row["E_composite"] - best["E_composite"]) / row["E_composite"] * 100,
                "benchmark_dish": best.name,
                "benchmark_H": best["H_mean"],
                "has_efficient_alternative": True,
            })
        else:
            results.append({
                "dish_id": idx,
                "E_current": row["E_composite"],
                "E_best_benchmark": np.nan,
                "E_reducible": 0,
                "E_reduction_pct": 0,
                "benchmark_dish": "",
                "benchmark_H": np.nan,
                "has_efficient_alternative": False,
            })
    return pd.DataFrame(results).set_index("dish_id")


# ── Visualization ────────────────────────────────────────────────────

def plot_h_vs_e_scatter(df: pd.DataFrame):
    """H vs E scatter with Pareto frontier and DEI contours."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # DEI contour lines
    e_range = np.linspace(df["E_composite"].min(), df["E_composite"].max(), 100)
    for dei_val in [5, 10, 20, 40, 80]:
        h_line = dei_val * e_range
        valid = h_line <= 10
        ax.plot(e_range[valid], h_line[valid], '--', alpha=0.3, color='gray',
                linewidth=0.8)
        # Label
        idx = np.argmin(np.abs(h_line - 8))
        if valid[idx]:
            ax.text(e_range[idx], h_line[idx], f'DEI={dei_val}',
                    fontsize=7, color='gray', alpha=0.6)

    # Get cuisine info
    cuisine_col = "cuisine" if "cuisine" in df.columns else "cuisine_h"
    if cuisine_col not in df.columns:
        cuisine_col = None

    if cuisine_col:
        cuisines = df[cuisine_col].unique()
        palette = sns.color_palette("husl", len(cuisines))
        for cuisine, color in zip(cuisines, palette):
            mask = df[cuisine_col] == cuisine
            ax.scatter(df.loc[mask, "E_composite"], df.loc[mask, "H_mean"],
                       label=cuisine, alpha=0.7, s=50, color=color, edgecolors='white',
                       linewidth=0.5)
    else:
        ax.scatter(df["E_composite"], df["H_mean"], alpha=0.7, s=50)

    # Pareto frontier
    is_front = pareto_frontier(df)
    front_dishes = df[is_front].sort_values("E_composite")
    ax.plot(front_dishes["E_composite"], front_dishes["H_mean"],
            'k-', linewidth=2, label="Pareto frontier", zorder=5)
    ax.scatter(front_dishes["E_composite"], front_dishes["H_mean"],
               color='gold', s=100, zorder=6, edgecolors='black', linewidth=1.5,
               label="Pareto-optimal dishes")

    # Label frontier dishes
    for idx, row in front_dishes.iterrows():
        name = idx.replace("_", " ").title()
        ax.annotate(name, (row["E_composite"], row["H_mean"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, fontweight='bold')

    ax.set_xlabel("Environmental Cost (E, composite, normalized)", fontsize=12)
    ax.set_ylabel("Hedonic Score (H, 1-10 scale)", fontsize=12)
    ax.set_title("Hedonic Value vs Environmental Cost\n"
                 "with DEI Contours and Pareto Frontier", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "dei_h_vs_e_scatter.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dei_by_cuisine(df: pd.DataFrame):
    """Violin plot of log(DEI) by cuisine."""
    cuisine_col = "cuisine" if "cuisine" in df.columns else "cuisine_h"
    if cuisine_col not in df.columns:
        print("  Warning: no cuisine column found, skipping cuisine plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby(cuisine_col)["log_DEI"].median().sort_values(ascending=False).index
    sns.violinplot(data=df.reset_index(), x=cuisine_col, y="log_DEI",
                   order=order, ax=ax, inner="box", cut=0)
    ax.set_xlabel("Cuisine")
    ax.set_ylabel("log(DEI) = log(H) − log(E)")
    ax.set_title("log(DEI) Distribution by Cuisine")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    path = FIGURES_DIR / "dei_by_cuisine_violin.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dei_distribution(df: pd.DataFrame):
    """DEI histogram + KDE (log-scale primary, raw as inset)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # log(DEI) distribution (PRIMARY)
    axes[0].hist(df["log_DEI"], bins=20, edgecolor='white', alpha=0.8, density=True)
    df["log_DEI"].plot.kde(ax=axes[0], color='red', linewidth=2)
    axes[0].set_xlabel("log(DEI) = log(H) − log(E)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("log(DEI) Distribution (Primary)")
    # Shapiro-Wilk test for normality
    if len(df) <= 5000:
        sw_stat, sw_p = stats.shapiro(df["log_DEI"])
        axes[0].text(0.05, 0.95, f"Shapiro-Wilk p={sw_p:.3f}",
                     transform=axes[0].transAxes, va='top', fontsize=9)

    # Z-score DEI distribution
    axes[1].hist(df["DEI_z"], bins=20, edgecolor='white', alpha=0.8, density=True,
                 color='steelblue')
    df["DEI_z"].plot.kde(ax=axes[1], color='darkred', linewidth=2)
    axes[1].set_xlabel("DEI_z = Z(H) − Z(E)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Standardised DEI Distribution")

    # DEI tier counts (based on log_DEI quintiles)
    tier_counts = df["DEI_tier"].value_counts().reindex(DEI_TIERS)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    axes[2].bar(tier_counts.index, tier_counts.values, color=colors,
                edgecolor='white')
    axes[2].set_xlabel("DEI Tier (based on log DEI)")
    axes[2].set_ylabel("Number of Dishes")
    axes[2].set_title("DEI Tier Distribution")

    plt.tight_layout()
    path = FIGURES_DIR / "dei_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sensitivity(df: pd.DataFrame):
    """Sensitivity of DEI rankings to weighting schemes (log-scale)."""
    schemes = list(E_WEIGHT_SCHEMES.keys())
    dei_cols = [f"log_DEI_{s}" for s in schemes if f"log_DEI_{s}" in df.columns]

    if len(dei_cols) < 2:
        print("  Not enough weighting schemes for sensitivity plot")
        return

    # Rank correlation matrix
    ranks = pd.DataFrame({col: df[col].rank() for col in dei_cols})
    corr = ranks.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="YlGn", vmin=0.9, vmax=1.0,
                ax=ax, square=True)
    ax.set_title("Spearman Rank Correlation of log(DEI)\nAcross Weighting Schemes")
    plt.tight_layout()

    path = FIGURES_DIR / "dei_sensitivity_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_variance_decomposition(df: pd.DataFrame, var_result: dict):
    """Visualize H vs E contribution to DEI variance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: pie chart of variance contribution
    labels = ['log(H)', 'log(E)']
    sizes = [var_result['H_pct_contribution'], var_result['E_pct_contribution']]
    colors_pie = ['#2196F3', '#FF5722']
    axes[0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
    axes[0].set_title("Variance Contribution to log(DEI)")

    # Right: scatter log(H) vs log(E) showing the spread
    axes[1].scatter(df["log_E"], df["log_H"], alpha=0.6, s=40, c='steelblue',
                    edgecolors='white', linewidth=0.5)
    axes[1].set_xlabel("log(E)")
    axes[1].set_ylabel("log(H)")
    r = np.corrcoef(df["log_H"], df["log_E"])[0, 1]
    axes[1].set_title(f"log(H) vs log(E)   (r = {r:.3f})")
    # Add y=x-c contour lines for equal log(DEI)
    x_range = np.array([df["log_E"].min(), df["log_E"].max()])
    for c in np.linspace(df["log_DEI"].min(), df["log_DEI"].max(), 5):
        axes[1].plot(x_range, x_range + c, '--', alpha=0.3, color='gray', linewidth=0.8)

    plt.tight_layout()
    path = FIGURES_DIR / "dei_variance_decomposition.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_waste_space(df: pd.DataFrame, waste: pd.DataFrame):
    """Visualize "waste space" - dishes that could be replaced."""
    waste_with_alt = waste[waste["has_efficient_alternative"]].copy()
    if len(waste_with_alt) == 0:
        print("  No dishes with efficient alternatives found")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # All dishes
    ax.scatter(df["E_composite"], df["H_mean"], alpha=0.4, s=40, color='gray',
               label='All dishes')

    # Dishes with reducible E
    for idx, row in waste_with_alt.iterrows():
        if idx in df.index and row["benchmark_dish"] in df.index:
            dish = df.loc[idx]
            bench = df.loc[row["benchmark_dish"]]
            ax.annotate('', xy=(bench["E_composite"], bench["H_mean"]),
                         xytext=(dish["E_composite"], dish["H_mean"]),
                         arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

    # Pareto frontier
    is_front = pareto_frontier(df)
    front = df[is_front].sort_values("E_composite")
    ax.plot(front["E_composite"], front["H_mean"], 'g-', linewidth=2.5,
            label="Pareto frontier")

    ax.set_xlabel("Environmental Cost (E)")
    ax.set_ylabel("Hedonic Score (H)")
    ax.set_title(f'"Waste Space" Visualization\n'
                 f'{len(waste_with_alt)}/{len(waste)} dishes have '
                 f'more efficient alternatives')
    ax.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "dei_waste_space.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DEI Project - DEI Computation and Analysis")
    print("=" * 60)

    # 1. Load data
    hedonic_path = DATA_DIR / "dish_hedonic_scores.csv"
    env_path = DATA_DIR / "dish_environmental_costs.csv"

    if not hedonic_path.exists():
        print(f"\n  {hedonic_path} not found.")
        print("  Creating synthetic hedonic scores from Yelp star ratings as placeholder...")
        # Use average star rating as proxy hedonic score (scaled to 1-10)
        mentions = pd.read_parquet(DATA_DIR / "dish_mentions.parquet")
        dish_stars = mentions.groupby("dish_id").agg(
            H_mean=("stars", "mean"),
            H_median=("stars", "median"),
            H_std=("stars", "std"),
            H_n=("stars", "count"),
        )
        # Scale 1-5 stars to 1-10 hedonic
        dish_stars["H_mean"] = (dish_stars["H_mean"] - 1) * 2.25 + 1
        dish_stars["H_median"] = (dish_stars["H_median"] - 1) * 2.25 + 1
        dish_stars["H_ci95"] = 1.96 * dish_stars["H_std"] * 2.25 / np.sqrt(dish_stars["H_n"])

        # Add cuisine info
        dish_info = mentions[["dish_id", "cuisine", "cook_method"]].drop_duplicates("dish_id")
        dish_stars = dish_stars.merge(dish_info, left_index=True, right_on="dish_id",
                                      how="left").set_index("dish_id")
        dish_stars = dish_stars[dish_stars["H_n"] >= MIN_REVIEWS_PER_DISH]
        dish_stars.to_csv(hedonic_path)
        print(f"  Saved proxy hedonic scores: {hedonic_path}")

    hedonic = pd.read_csv(hedonic_path, index_col="dish_id")
    env_cost = pd.read_csv(env_path, index_col="dish_id")

    print(f"\n  Hedonic scores: {len(hedonic)} dishes")
    print(f"  Environmental costs: {len(env_cost)} dishes")

    # 2. Compute DEI
    print("\n-- Computing DEI --")
    dei_df = compute_dei(hedonic, env_cost)
    dei_df.to_csv(DATA_DIR / "dish_DEI_scores.csv")
    print(f"  Saved: {DATA_DIR / 'dish_DEI_scores.csv'}")

    # 2b. Variance decomposition
    var_result = variance_decomposition(dei_df)

    # 3. Print full DEI ranking (sorted by log_DEI)
    print(f"\n  DEI Ranking (sorted by log(DEI)):")
    print(f"  {'Rank':<5s} {'Dish':<25s} {'H':<6s} {'E':<8s} {'logDEI':<8s} {'DEI_z':<7s} "
          f"{'Tier':<5s} {'Cuisine':<15s} {'Method':<12s}")
    print(f"  {'-'*94}")

    cuisine_col = "cuisine" if "cuisine" in dei_df.columns else "cuisine_h"
    method_col = "cook_method" if "cook_method" in dei_df.columns else "cook_method_h"

    for rank, (idx, row) in enumerate(dei_df.sort_values("log_DEI", ascending=False).iterrows(), 1):
        cuisine = row.get(cuisine_col, "?")
        method = row.get(method_col, "?")
        print(f"  {rank:<5d} {idx:<25s} {row['H_mean']:<6.2f} "
              f"{row['E_composite']:<8.3f} {row['log_DEI']:<8.3f} {row['DEI_z']:<7.2f} "
              f"{row['DEI_tier']:<5s} {cuisine:<15s} {method:<12s}")

    # 4. Pareto frontier
    print(f"\n-- Pareto Frontier Analysis --")
    is_front = pareto_frontier(dei_df)
    front_dishes = dei_df[is_front]
    print(f"  Pareto-optimal dishes: {len(front_dishes)} / {len(dei_df)}")
    for idx in front_dishes.sort_values("E_composite").index:
        row = front_dishes.loc[idx]
        print(f"    {idx:<25s} H={row['H_mean']:.2f}, E={row['E_composite']:.3f}")

    # 5. Waste space analysis
    print(f"\n-- Waste Space Analysis --")
    waste = waste_space_analysis(dei_df)
    n_with_alt = waste["has_efficient_alternative"].sum()
    print(f"  Dishes with more efficient alternatives: "
          f"{n_with_alt} / {len(waste)} ({n_with_alt/len(waste)*100:.1f}%)")
    if n_with_alt > 0:
        reducible = waste[waste["has_efficient_alternative"]]
        print(f"  Mean E reduction possible: {reducible['E_reduction_pct'].mean():.1f}%")
        print(f"  Max E reduction possible: {reducible['E_reduction_pct'].max():.1f}%")
        print(f"\n  Top 10 dishes with most 'waste':")
        for idx, row in reducible.nlargest(10, "E_reduction_pct").iterrows():
            print(f"    {idx:<25s} could reduce E by {row['E_reduction_pct']:.1f}% "
                  f"(switch to {row['benchmark_dish']})")
    waste.to_csv(TABLES_DIR / "waste_space_analysis.csv")

    # 6. Cuisine-level summary
    print(f"\n-- Cuisine-Level Summary --")
    if cuisine_col in dei_df.columns:
        cuisine_summary = dei_df.groupby(cuisine_col).agg(
            n_dishes=("log_DEI", "count"),
            mean_log_DEI=("log_DEI", "mean"),
            median_log_DEI=("log_DEI", "median"),
            std_log_DEI=("log_DEI", "std"),
            mean_DEI_z=("DEI_z", "mean"),
            mean_H=("H_mean", "mean"),
            mean_E=("E_composite", "mean"),
        ).sort_values("mean_log_DEI", ascending=False)
        # Add pareto percentage separately
        dei_df["is_pareto"] = is_front
        cuisine_summary["pct_pareto"] = dei_df.groupby(cuisine_col)["is_pareto"].mean() * 100
        print(cuisine_summary.round(3).to_string())
        cuisine_summary.to_csv(TABLES_DIR / "dei_cuisine_summary.csv")

    # 7. Sensitivity analysis (log-scale)
    print(f"\n-- Sensitivity: log(DEI) Stability Across Weighting Schemes --")
    schemes = list(E_WEIGHT_SCHEMES.keys())
    for i in range(len(schemes)):
        for j in range(i+1, len(schemes)):
            col_i = f"log_DEI_{schemes[i]}"
            col_j = f"log_DEI_{schemes[j]}"
            if col_i in dei_df.columns and col_j in dei_df.columns:
                r, p = stats.spearmanr(dei_df[col_i].rank(), dei_df[col_j].rank())
                print(f"  {schemes[i]:15s} vs {schemes[j]:15s}: "
                      f"Spearman rho = {r:.4f} (p={p:.2e})")

    # 7b. Rank correlation between log_DEI and DEI_z
    rho_methods, _ = stats.spearmanr(dei_df["log_DEI"].rank(), dei_df["DEI_z"].rank())
    print(f"\n  log(DEI) vs DEI_z rank correlation: rho = {rho_methods:.4f}")

    # 8. Regression: what predicts log(DEI)?
    print(f"\n-- OLS Regression: Determinants of log(DEI) --")
    try:
        import statsmodels.api as sm

        reg_data = dei_df.copy()
        # Create dummy variables
        if cuisine_col in reg_data.columns:
            cuisine_dummies = pd.get_dummies(reg_data[cuisine_col], prefix="cuisine",
                                              drop_first=True)
            reg_data = pd.concat([reg_data, cuisine_dummies], axis=1)

        if method_col in reg_data.columns:
            method_dummies = pd.get_dummies(reg_data[method_col], prefix="method",
                                             drop_first=True)
            reg_data = pd.concat([reg_data, method_dummies], axis=1)

        # Features: recipe characteristics + controls
        feature_cols = ["n_ingredients", "total_grams"]
        # Add log(E) components if available
        for ec in ["E_carbon", "E_water", "E_energy"]:
            if ec in reg_data.columns:
                reg_data[f"log_{ec}"] = np.log(reg_data[ec].clip(lower=1e-6))
                feature_cols.append(f"log_{ec}")
        feature_cols += [c for c in reg_data.columns if c.startswith("cuisine_")]
        feature_cols += [c for c in reg_data.columns if c.startswith("method_")]
        feature_cols = [c for c in feature_cols if c in reg_data.columns]

        if feature_cols:
            X = reg_data[feature_cols].astype(float)
            X = sm.add_constant(X)
            y = reg_data["log_DEI"]

            model = sm.OLS(y, X).fit()
            print(model.summary2().tables[1].to_string())
            print(f"\n  R-squared: {model.rsquared:.4f}")
            print(f"  Adj R-squared: {model.rsquared_adj:.4f}")

            # Save regression results
            with open(TABLES_DIR / "dei_regression.txt", "w", encoding="utf-8") as f:
                f.write("OLS Regression: log(DEI) = log(H) - log(E)\n")
                f.write("=" * 60 + "\n\n")
                f.write(str(model.summary()))

            # 8b. Decompose log(DEI) into H component and E component regressions
            print(f"\n-- Separate H and E Regressions --")
            # log(H) regression
            y_h = reg_data["log_H"]
            model_h = sm.OLS(y_h, X).fit()
            print(f"  log(H) regression R²: {model_h.rsquared:.4f}")

            # log(E) regression
            y_e = reg_data["log_E"]
            model_e = sm.OLS(y_e, X).fit()
            print(f"  log(E) regression R²: {model_e.rsquared:.4f}")

            with open(TABLES_DIR / "dei_regression.txt", "a", encoding="utf-8") as f:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("Separate log(H) Regression\n")
                f.write(str(model_h.summary()))
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("Separate log(E) Regression\n")
                f.write(str(model_e.summary()))

    except ImportError:
        print("  statsmodels not installed, skipping regression")

    # 9. Generate all plots
    print(f"\n-- Generating Figures --")
    plot_dei_distribution(dei_df)
    plot_h_vs_e_scatter(dei_df)
    plot_dei_by_cuisine(dei_df)
    plot_sensitivity(dei_df)
    plot_waste_space(dei_df, waste)
    plot_variance_decomposition(dei_df, var_result)

    # 10. Top/Bottom tables for paper
    print(f"\n-- Tables for Paper --")
    # Overall top/bottom 10
    print(f"\n  Top 10 dishes by log(DEI):")
    for rank, (idx, row) in enumerate(dei_df.nlargest(10, "log_DEI").iterrows(), 1):
        print(f"    {rank}. {idx:<25s} log(DEI)={row['log_DEI']:.3f}  "
              f"H={row['H_mean']:.2f}  E={row['E_composite']:.4f}")

    print(f"\n  Bottom 10 dishes by log(DEI):")
    for rank, (idx, row) in enumerate(dei_df.nsmallest(10, "log_DEI").iterrows(), 1):
        print(f"    {rank}. {idx:<25s} log(DEI)={row['log_DEI']:.3f}  "
              f"H={row['H_mean']:.2f}  E={row['E_composite']:.4f}")

    if cuisine_col in dei_df.columns:
        for cuisine in sorted(dei_df[cuisine_col].unique()):
            subset = dei_df[dei_df[cuisine_col] == cuisine].sort_values("log_DEI", ascending=False)
            if len(subset) >= 3:
                print(f"\n  {cuisine} - Top 5 log(DEI):")
                for idx, row in subset.head(5).iterrows():
                    print(f"    {idx:<25s} log(DEI)={row['log_DEI']:.3f}")

    print("\n" + "=" * 60)
    print("DEI computation and analysis complete.")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Tables saved to: {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
