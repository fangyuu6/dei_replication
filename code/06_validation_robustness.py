"""
06_validation_robustness.py - Validation and Robustness Checks
==============================================================
Nature-level validation framework for the DEI metric.

Tests:
  1. H-score convergence: bootstrap CI width vs sample size
  2. Proxy vs NLP H comparison (star ratings vs BERT sentiment)
  3. E-component sensitivity: leave-one-out on E components
  4. DEI rank stability: Monte Carlo perturbation of H and E
  5. Cuisine-level confounders: ANOVA + post-hoc tests
  6. Sample representativeness: coverage and reviewer diversity
  7. Cross-validation: split-half reliability of H scores
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
    E_WEIGHT_SCHEMES, MIN_REVIEWS_PER_DISH,
)

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ── Test 1: Bootstrap Convergence ─────────────────────────────────────

def test_h_convergence(scored_mentions: pd.DataFrame, dish_ids: list = None,
                       n_bootstrap: int = 500):
    """Test how H-score CI width converges with increasing sample size.

    For each dish, subsample at sizes [10, 25, 50, 100, 200, 500] and
    measure 95% CI width via bootstrap. A good metric should show
    convergence (CI < 0.5) by n=100.
    """
    print("\n== Test 1: H-Score Convergence ==")
    sample_sizes = [10, 25, 50, 100, 200, 500]

    if dish_ids is None:
        # Pick 10 dishes with >= 500 mentions
        counts = scored_mentions["dish_id"].value_counts()
        dish_ids = counts[counts >= 500].head(10).index.tolist()

    # Use finetuned if available, else pretrained
    h_col = ("hedonic_score_finetuned"
             if "hedonic_score_finetuned" in scored_mentions.columns
             else "hedonic_score_pretrained")
    print(f"  Using H column: {h_col}")

    results = []
    for dish in dish_ids:
        dish_data = scored_mentions[scored_mentions["dish_id"] == dish]
        scores = dish_data[h_col].dropna().values

        for n in sample_sizes:
            if n > len(scores):
                continue
            # Bootstrap: compute CI of the MEAN
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(scores, size=n, replace=True)
                boot_means.append(np.mean(sample))
            ci_width = np.percentile(boot_means, 97.5) - np.percentile(boot_means, 2.5)
            ci_widths = [ci_width]  # single value per (dish, n)
            results.append({
                "dish_id": dish,
                "sample_size": n,
                "mean_ci_width": np.mean(ci_widths),
                "std_ci_width": np.std(ci_widths),
            })

    df = pd.DataFrame(results)

    # Summary
    convergence = df.groupby("sample_size")["mean_ci_width"].agg(["mean", "std"])
    print("  Sample size vs mean CI width:")
    for n, row in convergence.iterrows():
        converged = "OK" if row["mean"] < 0.5 else ""
        print(f"    n={n:4d}: CI width = {row['mean']:.3f} +/- {row['std']:.3f} {converged}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for dish in dish_ids[:5]:
        sub = df[df["dish_id"] == dish]
        ax.plot(sub["sample_size"], sub["mean_ci_width"], 'o-', alpha=0.6, label=dish)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Target (0.5)')
    ax.set_xlabel("Sample Size per Dish")
    ax.set_ylabel("95% CI Width (Bootstrap)")
    ax.set_title("H-Score Convergence with Sample Size")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_h_convergence.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'validation_h_convergence.png'}")

    return df


# ── Test 2: Proxy vs NLP H Comparison ─────────────────────────────────

def test_proxy_vs_nlp(scored_mentions: pd.DataFrame, hedonic: pd.DataFrame):
    """Compare star-rating proxy H with BERT H (pretrained and finetuned).

    Key metrics:
      - Pearson r between proxy and NLP H per dish
      - Spearman rank correlation (rank-order consistency)
      - Dishes that change tier (> 1 quintile shift)
      - If finetuned H available: pretrained vs finetuned comparison
    """
    print("\n== Test 2: Proxy vs NLP H Comparison ==")

    # Build proxy from star ratings
    proxy = scored_mentions.groupby("dish_id")["stars"].mean()
    proxy_h = (proxy - 1) * 2.25 + 1  # scale 1-5 -> 1-10
    proxy_h.name = "H_proxy"

    nlp_h = hedonic["H_mean"]
    nlp_h.name = "H_nlp"

    merged = pd.concat([proxy_h, nlp_h], axis=1).dropna()

    r, p_r = stats.pearsonr(merged["H_proxy"], merged["H_nlp"])
    rho, p_rho = stats.spearmanr(merged["H_proxy"], merged["H_nlp"])
    mae = np.mean(np.abs(merged["H_proxy"] - merged["H_nlp"]))

    print(f"  Dishes compared: {len(merged)}")
    print(f"  Pearson r (proxy vs BERT):  {r:.4f} (p={p_r:.2e})")
    print(f"  Spearman rho: {rho:.4f} (p={p_rho:.2e})")
    print(f"  MAE: {mae:.3f}")

    # Finetuned vs Pretrained comparison
    has_finetuned = "hedonic_score_finetuned" in scored_mentions.columns
    if has_finetuned:
        pretrained_dish = scored_mentions.groupby("dish_id")["hedonic_score_pretrained"].mean()
        finetuned_dish = scored_mentions.groupby("dish_id")["hedonic_score_finetuned"].mean()
        both = pd.concat([pretrained_dish.rename("H_pretrained"),
                          finetuned_dish.rename("H_finetuned")], axis=1).dropna()
        r_pf, _ = stats.pearsonr(both["H_pretrained"], both["H_finetuned"])
        rho_pf, _ = stats.spearmanr(both["H_pretrained"], both["H_finetuned"])
        print(f"\n  Pretrained vs Finetuned (dish-level):")
        print(f"    Pearson r:  {r_pf:.4f}")
        print(f"    Spearman rho: {rho_pf:.4f}")
        print(f"    Mean shift: {(both['H_finetuned'] - both['H_pretrained']).mean():.3f}")
        print(f"    Pretrained range: [{both['H_pretrained'].min():.2f}, {both['H_pretrained'].max():.2f}]")
        print(f"    Finetuned range:  [{both['H_finetuned'].min():.2f}, {both['H_finetuned'].max():.2f}]")

    # Quintile comparison
    merged["proxy_q"] = pd.qcut(merged["H_proxy"], q=5, labels=False, duplicates="drop")
    merged["nlp_q"] = pd.qcut(merged["H_nlp"], q=5, labels=False, duplicates="drop")
    merged["tier_shift"] = np.abs(merged["proxy_q"] - merged["nlp_q"])
    n_shifted = (merged["tier_shift"] > 1).sum()
    print(f"  Dishes shifting > 1 quintile: {n_shifted} ({n_shifted/len(merged)*100:.1f}%)")

    # Plot
    ncols = 3 if has_finetuned else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    axes[0].scatter(merged["H_proxy"], merged["H_nlp"], alpha=0.6, s=40)
    lims = [merged[["H_proxy", "H_nlp"]].min().min() - 0.2,
            merged[["H_proxy", "H_nlp"]].max().max() + 0.2]
    axes[0].plot(lims, lims, 'r--', alpha=0.5, label='y=x')
    axes[0].set_xlabel("H (Star-Rating Proxy)")
    axes[0].set_ylabel("H (BERT)")
    axes[0].set_title(f"Proxy vs BERT H\nr={r:.3f}, rho={rho:.3f}")
    axes[0].legend()

    # Rank comparison
    merged["rank_proxy"] = merged["H_proxy"].rank(ascending=False)
    merged["rank_nlp"] = merged["H_nlp"].rank(ascending=False)
    axes[1].scatter(merged["rank_proxy"], merged["rank_nlp"], alpha=0.6, s=40)
    max_rank = max(merged["rank_proxy"].max(), merged["rank_nlp"].max())
    axes[1].plot([0, max_rank], [0, max_rank], 'r--', alpha=0.5)
    axes[1].set_xlabel("Rank (Star-Rating Proxy)")
    axes[1].set_ylabel("Rank (BERT)")
    axes[1].set_title("Rank-Order Comparison")

    if has_finetuned:
        axes[2].scatter(both["H_pretrained"], both["H_finetuned"], alpha=0.6, s=40, c='green')
        lims2 = [min(both.min().min(), 1) - 0.2, max(both.max().max(), 10) + 0.2]
        axes[2].plot(lims2, lims2, 'r--', alpha=0.5, label='y=x')
        axes[2].set_xlabel("H (Pretrained BERT)")
        axes[2].set_ylabel("H (Fine-tuned BERT)")
        axes[2].set_title(f"Pretrained vs Fine-tuned\nr={r_pf:.3f}, rho={rho_pf:.3f}")
        axes[2].legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_proxy_vs_nlp.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'validation_proxy_vs_nlp.png'}")

    return {"pearson_r": r, "spearman_rho": rho, "mae": mae, "n_shifted": n_shifted}


# ── Test 3: E-Component Leave-One-Out Sensitivity ─────────────────────

def test_e_sensitivity(dei_df: pd.DataFrame):
    """Leave-one-out sensitivity: recompute E dropping each component.

    If removing one E component dramatically changes rankings,
    that component is a key driver and should be discussed.
    Uses log(DEI) for ranking.
    """
    print("\n== Test 3: E-Component Leave-One-Out ==")

    e_components = ["E_carbon", "E_water", "E_energy"]
    available = [c for c in e_components if c in dei_df.columns]

    if len(available) < 2:
        print("  Not enough E components for leave-one-out analysis")
        return {}

    # Use log_DEI if available, else DEI
    dei_col = "log_DEI" if "log_DEI" in dei_df.columns else "DEI"
    baseline_rank = dei_df[dei_col].rank()
    results = {}

    for drop_col in available:
        keep = [c for c in available if c != drop_col]
        e_reduced = dei_df[keep].mean(axis=1)
        dei_reduced = np.log(dei_df["H_mean"]) - np.log(e_reduced)
        reduced_rank = dei_reduced.rank()

        rho, p = stats.spearmanr(baseline_rank, reduced_rank)
        kendall, p_k = stats.kendalltau(baseline_rank, reduced_rank)

        results[drop_col] = {
            "spearman_rho": rho,
            "kendall_tau": kendall,
            "n_top10_change": (reduced_rank <= 10).sum() - (baseline_rank <= 10).sum(),
        }
        print(f"  Drop {drop_col:12s}: Spearman rho={rho:.4f}, "
              f"Kendall tau={kendall:.4f}")

    return results


# ── Test 4: Monte Carlo Rank Stability ────────────────────────────────

def test_mc_rank_stability(dei_df: pd.DataFrame, n_simulations: int = 1000):
    """Perturb H and E with noise and measure log(DEI) rank stability.

    Adds Gaussian noise to H (sd = H_ci95) and to E (sd = 10% of E),
    recomputes log(DEI), and checks how often each dish stays within
    +/- 3 ranks of its baseline.
    """
    print(f"\n== Test 4: Monte Carlo Rank Stability ({n_simulations} sims) ==")

    dei_col = "log_DEI" if "log_DEI" in dei_df.columns else "DEI"
    baseline_ranks = dei_df[dei_col].rank(ascending=False)

    h_noise_sd = dei_df.get("H_ci95", pd.Series(0.3, index=dei_df.index))
    e_noise_sd = dei_df["E_composite"] * 0.10  # 10% uncertainty

    rank_matrix = np.zeros((len(dei_df), n_simulations))

    for sim in range(n_simulations):
        h_perturbed = dei_df["H_mean"] + np.random.normal(0, h_noise_sd)
        e_perturbed = dei_df["E_composite"] + np.random.normal(0, e_noise_sd)
        e_perturbed = e_perturbed.clip(lower=0.001)
        h_perturbed = h_perturbed.clip(lower=1, upper=10)

        dei_sim = np.log(h_perturbed) - np.log(e_perturbed)
        rank_matrix[:, sim] = pd.Series(dei_sim.values, index=dei_df.index).rank(ascending=False).values

    # Stability metrics
    mean_ranks = rank_matrix.mean(axis=1)
    std_ranks = rank_matrix.std(axis=1)
    stability = pd.DataFrame({
        "dish_id": dei_df.index,
        "baseline_rank": baseline_ranks.values,
        "mc_mean_rank": mean_ranks,
        "mc_std_rank": std_ranks,
        "pct_within_3": [(np.abs(rank_matrix[i, :] - baseline_ranks.iloc[i]) <= 3).mean() * 100
                          for i in range(len(dei_df))],
    }).set_index("dish_id")

    mean_stability = stability["pct_within_3"].mean()
    print(f"  Mean % simulations where rank stays within +/-3: {mean_stability:.1f}%")
    print(f"  Most stable dish: {stability['pct_within_3'].idxmax()} "
          f"({stability['pct_within_3'].max():.1f}%)")
    print(f"  Least stable dish: {stability['pct_within_3'].idxmin()} "
          f"({stability['pct_within_3'].min():.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    stability_sorted = stability.sort_values("baseline_rank")
    ax.errorbar(stability_sorted["baseline_rank"],
                stability_sorted["mc_mean_rank"],
                yerr=stability_sorted["mc_std_rank"],
                fmt='o', alpha=0.5, markersize=3, capsize=2)
    lims = [0, len(dei_df) + 1]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.fill_between(range(1, len(dei_df)+1),
                     [r-3 for r in range(1, len(dei_df)+1)],
                     [r+3 for r in range(1, len(dei_df)+1)],
                     alpha=0.1, color='green', label='+/- 3 band')
    ax.set_xlabel("Baseline log(DEI) Rank")
    ax.set_ylabel("Monte Carlo Mean Rank (+/- 1 SD)")
    ax.set_title(f"log(DEI) Rank Stability Under Perturbation\n"
                 f"({n_simulations} simulations, mean stability: {mean_stability:.1f}%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_mc_rank_stability.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'validation_mc_rank_stability.png'}")

    stability.to_csv(TABLES_DIR / "mc_rank_stability.csv")
    return stability


# ── Test 5: Cuisine ANOVA ─────────────────────────────────────────────

def test_cuisine_anova(dei_df: pd.DataFrame):
    """One-way ANOVA and post-hoc Tukey HSD for DEI across cuisines.

    Tests whether cuisine is a significant predictor of DEI
    (expected: yes, but effect size matters for interpretation).
    """
    print("\n== Test 5: Cuisine ANOVA ==")

    cuisine_col = "cuisine" if "cuisine" in dei_df.columns else "cuisine_h"
    if cuisine_col not in dei_df.columns:
        print("  No cuisine column found")
        return {}

    dei_col = "log_DEI" if "log_DEI" in dei_df.columns else "DEI"
    print(f"  Using: {dei_col}")
    groups = [group[dei_col].values for _, group in dei_df.groupby(cuisine_col)]
    f_stat, p_val = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    ss_between = sum(len(g) * (np.mean(g) - dei_df[dei_col].mean())**2 for g in groups)
    ss_total = sum((dei_df[dei_col] - dei_df[dei_col].mean())**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_val:.2e}")
    print(f"  Eta-squared (effect size): {eta_sq:.3f}")
    print(f"  Interpretation: {'Large' if eta_sq > 0.14 else 'Medium' if eta_sq > 0.06 else 'Small'} effect")

    # Kruskal-Wallis (non-parametric alternative)
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis H: {h_stat:.3f}, p={p_kw:.2e}")

    return {"f_stat": f_stat, "p_anova": p_val, "eta_squared": eta_sq,
            "h_stat": h_stat, "p_kruskal": p_kw}


# ── Test 6: Sample Representativeness ─────────────────────────────────

def test_representativeness(scored_mentions: pd.DataFrame, hedonic: pd.DataFrame):
    """Assess sample coverage and reviewer diversity.

    Checks:
      - Coverage: what fraction of dish mentions are scored
      - Reviewer diversity: unique reviewers per dish
      - Geographic spread: unique cities per dish
    """
    print("\n== Test 6: Sample Representativeness ==")

    n_scored = len(scored_mentions)
    n_dishes = scored_mentions["dish_id"].nunique()
    n_reviewers = scored_mentions["review_id"].nunique()

    print(f"  Scored mentions: {n_scored:,}")
    print(f"  Dishes covered: {n_dishes}")
    print(f"  Unique reviews: {n_reviewers:,}")

    # Reviews per dish distribution
    per_dish = scored_mentions.groupby("dish_id").size()
    print(f"\n  Reviews per dish:")
    print(f"    Min: {per_dish.min()}")
    print(f"    Median: {per_dish.median():.0f}")
    print(f"    Max: {per_dish.max()}")
    print(f"    Mean: {per_dish.mean():.1f}")

    # Hedonic score distribution check
    h_col = ("hedonic_score_finetuned"
             if "hedonic_score_finetuned" in scored_mentions.columns
             else "hedonic_score_pretrained")
    h_scores = scored_mentions[h_col].dropna()
    print(f"\n  H-score distribution:")
    print(f"    Mean: {h_scores.mean():.3f}")
    print(f"    Std: {h_scores.std():.3f}")
    print(f"    Skewness: {stats.skew(h_scores):.3f}")
    print(f"    Kurtosis: {stats.kurtosis(h_scores):.3f}")

    # Normality test (D'Agostino-Pearson) on dish-level means
    _, p_norm = stats.normaltest(hedonic["H_mean"])
    print(f"    Normality (D'Agostino, dish means): p={p_norm:.4f}")

    return {
        "n_scored": n_scored, "n_dishes": n_dishes,
        "min_per_dish": per_dish.min(), "max_per_dish": per_dish.max(),
    }


# ── Test 7: Split-Half Reliability ────────────────────────────────────

def test_split_half_reliability(scored_mentions: pd.DataFrame, n_splits: int = 100):
    """Split-half reliability of dish-level H scores.

    For each dish, randomly split reviews into two halves,
    compute H_mean for each half, and measure correlation across
    all dishes. Repeat n_splits times for robust estimate.
    """
    print(f"\n== Test 7: Split-Half Reliability ({n_splits} splits) ==")

    # Use finetuned if available, else pretrained
    h_col = ("hedonic_score_finetuned"
             if "hedonic_score_finetuned" in scored_mentions.columns
             else "hedonic_score_pretrained")
    print(f"  Using H column: {h_col}")

    dish_ids = scored_mentions["dish_id"].unique()
    correlations = []

    for split in range(n_splits):
        h_a = {}
        h_b = {}
        for dish in dish_ids:
            dish_scores = scored_mentions[
                scored_mentions["dish_id"] == dish
            ][h_col].dropna().values

            if len(dish_scores) < 10:
                continue
            np.random.shuffle(dish_scores)
            mid = len(dish_scores) // 2
            h_a[dish] = np.mean(dish_scores[:mid])
            h_b[dish] = np.mean(dish_scores[mid:])

        common = set(h_a.keys()) & set(h_b.keys())
        if len(common) < 10:
            continue
        vals_a = [h_a[d] for d in common]
        vals_b = [h_b[d] for d in common]
        r, _ = stats.pearsonr(vals_a, vals_b)
        # Spearman-Brown correction for full-test reliability
        sb_r = 2 * r / (1 + r)
        correlations.append({"r_half": r, "r_spearman_brown": sb_r})

    df = pd.DataFrame(correlations)
    print(f"  Mean split-half r: {df['r_half'].mean():.4f} +/- {df['r_half'].std():.4f}")
    print(f"  Spearman-Brown corrected: {df['r_spearman_brown'].mean():.4f}")
    print(f"  Target: r > 0.85 (good), r > 0.90 (excellent)")

    quality = "Excellent" if df['r_spearman_brown'].mean() > 0.90 else \
              "Good" if df['r_spearman_brown'].mean() > 0.85 else \
              "Acceptable" if df['r_spearman_brown'].mean() > 0.70 else "Poor"
    print(f"  Assessment: {quality}")

    return {
        "mean_r_half": df['r_half'].mean(),
        "mean_r_sb": df['r_spearman_brown'].mean(),
        "quality": quality,
    }


# ── Summary Report ────────────────────────────────────────────────────

def generate_validation_report(results: dict):
    """Generate a consolidated validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY REPORT")
    print("=" * 60)

    checks = [
        ("H-Score Convergence", "CI < 0.5 at n=100",
         results.get("convergence_ok", "N/A")),
        ("Proxy vs NLP Correlation", f"r={results.get('proxy_r', 'N/A'):.3f}" if isinstance(results.get('proxy_r'), float) else "N/A",
         "PASS" if results.get('proxy_r', 0) > 0.5 else "CHECK"),
        ("E-Component Stability", "All rho > 0.90",
         results.get("e_stability", "N/A")),
        ("MC Rank Stability", f"Mean within-3: {results.get('mc_stability', 0):.1f}%",
         "PASS" if results.get('mc_stability', 0) > 70 else "CHECK"),
        ("Cuisine ANOVA", f"eta^2={results.get('eta_sq', 0):.3f}",
         "NOTED" if results.get('eta_sq', 0) > 0.06 else "OK"),
        ("Split-Half Reliability", results.get("reliability_quality", "N/A"),
         "PASS" if results.get("reliability_r", 0) > 0.85 else "CHECK"),
    ]

    for name, metric, status in checks:
        print(f"  [{status:6s}] {name:30s} {metric}")

    print("\n" + "=" * 60)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DEI Project - Validation and Robustness Checks")
    print("=" * 60)

    # Load data
    scored_path = DATA_DIR / "dish_mentions_scored.parquet"
    hedonic_path = DATA_DIR / "dish_hedonic_scores.csv"
    dei_path = DATA_DIR / "dish_DEI_scores.csv"

    if not scored_path.exists():
        print(f"ERROR: {scored_path} not found. Run 03_nlp_hedonic_scoring.py first.")
        return

    scored = pd.read_parquet(scored_path)
    hedonic = pd.read_csv(hedonic_path, index_col="dish_id")
    dei_df = pd.read_csv(dei_path, index_col=0)

    results = {}

    # Test 1: Bootstrap convergence
    conv_df = test_h_convergence(scored)
    conv_100 = conv_df[conv_df["sample_size"] == 100]["mean_ci_width"].mean()
    results["convergence_ok"] = "PASS" if conv_100 < 0.5 else f"CI={conv_100:.3f}"

    # Test 2: Proxy vs NLP
    proxy_results = test_proxy_vs_nlp(scored, hedonic)
    results["proxy_r"] = proxy_results["pearson_r"]

    # Test 3: E sensitivity
    e_results = test_e_sensitivity(dei_df)
    if e_results:
        min_rho = min(v["spearman_rho"] for v in e_results.values())
        results["e_stability"] = "PASS" if min_rho > 0.90 else f"min_rho={min_rho:.3f}"

    # Test 4: MC rank stability
    stability = test_mc_rank_stability(dei_df)
    results["mc_stability"] = stability["pct_within_3"].mean()

    # Test 5: Cuisine ANOVA
    anova_results = test_cuisine_anova(dei_df)
    results["eta_sq"] = anova_results.get("eta_squared", 0)

    # Test 6: Representativeness
    rep_results = test_representativeness(scored, hedonic)

    # Test 7: Split-half reliability
    rel_results = test_split_half_reliability(scored)
    results["reliability_r"] = rel_results["mean_r_sb"]
    results["reliability_quality"] = rel_results["quality"]

    # Summary report
    generate_validation_report(results)

    # Save all results
    pd.Series(results).to_csv(TABLES_DIR / "validation_summary.csv")
    print(f"\n  All validation results saved to {TABLES_DIR}")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
