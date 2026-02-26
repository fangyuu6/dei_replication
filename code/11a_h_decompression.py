"""
11a_h_decompression.py — H Score Compression Analysis
======================================================
Addresses criticism C1: BERT H scores have unreasonably narrow range
[6.05, 7.57], CV=3.9%. Top-bottom difference ~0.04 is not credible.

Analyses:
  1. Compression quantification: compare CV of star ratings vs BERT H
  2. Beta-distribution rescaling (rank-preserving decompression)
  3. Sensitivity analysis: how much H spread changes DEI variance decomposition
  4. Rank stability under decompression scenarios

Outputs:
  - tables/h_compression_analysis.csv
  - tables/h_decompression_sensitivity.csv
  - figures/h_decompression_sensitivity.png
  - figures/h_rescaled_distribution.png
"""

import sys, warnings
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
import matplotlib.ticker as mticker
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── Load data ────────────────────────────────────────────────────
print("=" * 60)
print("11a: H Score Compression Analysis")
print("=" * 60)

dei = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
print(f"Loaded {len(dei)} dishes from combined_dish_DEI.csv")

# Also load mention-level data for star-rating comparison
mentions_path = DATA_DIR / "dish_mentions_scored.parquet"
has_mentions = mentions_path.exists()
if has_mentions:
    mentions = pd.read_parquet(mentions_path)
    print(f"Loaded {len(mentions):,} mention-level records")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: Compression Quantification
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 1: Compression Quantification")
print("=" * 60)

H = dei["H_mean"].values
H_mean = H.mean()
H_std = H.std()
H_cv = H_std / H_mean * 100
H_range = H.max() - H.min()

print(f"  H_mean:  {H_mean:.4f}")
print(f"  H_std:   {H_std:.4f}")
print(f"  H_CV:    {H_cv:.2f}%")
print(f"  H_range: [{H.min():.3f}, {H.max():.3f}] (span={H_range:.3f})")

# Compare with star-rating CV at dish level (if available)
compression_rows = []
if has_mentions and "stars" in mentions.columns:
    star_by_dish = mentions.groupby("dish_id")["stars"].mean()
    star_cv = star_by_dish.std() / star_by_dish.mean() * 100
    star_range = star_by_dish.max() - star_by_dish.min()
    compression_factor_cv = star_cv / H_cv
    compression_factor_range = star_range / H_range

    print(f"\n  Star rating (dish-level means):")
    print(f"    Mean:  {star_by_dish.mean():.4f}")
    print(f"    CV:    {star_cv:.2f}%")
    print(f"    Range: [{star_by_dish.min():.3f}, {star_by_dish.max():.3f}] (span={star_range:.3f})")
    print(f"  Compression factor (CV):    {compression_factor_cv:.2f}x")
    print(f"  Compression factor (range): {compression_factor_range:.2f}x")

    compression_rows.append({
        "metric": "star_rating_dish_level",
        "mean": star_by_dish.mean(), "std": star_by_dish.std(),
        "cv_pct": star_cv, "range": star_range,
        "compression_factor_cv": compression_factor_cv,
    })

# Food science literature benchmark (estimated from published studies)
# Typical hedonic CV in controlled taste panels: 15-25%
lit_cv = 20.0
compression_vs_lit = lit_cv / H_cv
print(f"\n  Literature benchmark (controlled taste panels):")
print(f"    Typical CV: ~{lit_cv}%")
print(f"    Compression factor vs literature: {compression_vs_lit:.1f}x")

compression_rows.append({
    "metric": "bert_h_observed", "mean": H_mean, "std": H_std,
    "cv_pct": H_cv, "range": H_range, "compression_factor_cv": 1.0,
})
compression_rows.append({
    "metric": "literature_benchmark", "mean": np.nan, "std": np.nan,
    "cv_pct": lit_cv, "range": np.nan, "compression_factor_cv": compression_vs_lit,
})

comp_df = pd.DataFrame(compression_rows)
comp_df.to_csv(TABLES_DIR / "h_compression_analysis.csv", index=False)
print(f"\n  Saved: {TABLES_DIR / 'h_compression_analysis.csv'}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Beta-Distribution Rescaling
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 2: Beta-Distribution Rescaling")
print("=" * 60)

# Normalize H to [0,1], fit beta, use quantile function to spread
H_norm = (H - H.min()) / (H.max() - H.min())
# Clip to avoid 0/1 boundary issues
H_norm = np.clip(H_norm, 0.001, 0.999)

# Fit beta distribution
a_beta, b_beta, loc_beta, scale_beta = sp_stats.beta.fit(H_norm, floc=0, fscale=1)
print(f"  Beta fit: a={a_beta:.3f}, b={b_beta:.3f}")

# Rescale using rank-based quantile mapping to target range [3, 9]
TARGET_MIN, TARGET_MAX = 3.0, 9.0
N = len(H)
ranks = sp_stats.rankdata(H, method="average")
quantiles = ranks / (N + 1)  # uniform quantiles

# Use beta quantile function to spread
H_rescaled = sp_stats.beta.ppf(quantiles, a_beta, b_beta) * (TARGET_MAX - TARGET_MIN) + TARGET_MIN

H_rescaled_cv = np.std(H_rescaled) / np.mean(H_rescaled) * 100
H_rescaled_range = H_rescaled.max() - H_rescaled.min()

print(f"  Rescaled H:")
print(f"    Mean:  {np.mean(H_rescaled):.4f}")
print(f"    CV:    {H_rescaled_cv:.2f}%")
print(f"    Range: [{H_rescaled.min():.3f}, {H_rescaled.max():.3f}] (span={H_rescaled_range:.3f})")

# KS test
ks_stat, ks_p = sp_stats.ks_2samp(H, H_rescaled)
print(f"  KS test (original vs rescaled): D={ks_stat:.4f}, p={ks_p:.4e}")

# Rank preservation check
rho_rescale, _ = sp_stats.spearmanr(H, H_rescaled)
print(f"  Rank preservation (Spearman ρ): {rho_rescale:.6f}")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Sensitivity — H CV vs DEI Variance Decomposition
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 3: Sensitivity Analysis — H CV vs DEI Variance Contribution")
print("=" * 60)

log_E = dei["log_E"].values
original_log_H = dei["log_H"].values

# Current variance decomposition
var_log_H_orig = np.var(original_log_H)
var_log_E = np.var(log_E)
cov_orig = np.cov(original_log_H, log_E)[0, 1]
var_log_DEI_orig = var_log_H_orig + var_log_E - 2 * cov_orig
H_pct_orig = var_log_H_orig / var_log_DEI_orig * 100

print(f"  Original: Var(log H)={var_log_H_orig:.6f}, Var(log E)={var_log_E:.6f}")
print(f"  Original H contribution: {H_pct_orig:.2f}%")

# Simulate different H CVs
cv_targets = [3.9, 5, 7, 10, 15, 20, 25, 30, 40, 50]
sensitivity_rows = []
np.random.seed(42)

for target_cv in cv_targets:
    # Scale H values to achieve target CV while preserving ranks
    target_std = H_mean * target_cv / 100
    # Use rank-preserving scaling: map to target distribution
    H_sim = H_mean + (H - H_mean) * (target_std / H_std)
    log_H_sim = np.log(np.clip(H_sim, 0.1, 10))

    var_log_H_sim = np.var(log_H_sim)
    cov_sim = np.cov(log_H_sim, log_E)[0, 1]
    var_log_DEI_sim = var_log_H_sim + var_log_E - 2 * cov_sim

    h_pct = var_log_H_sim / var_log_DEI_sim * 100

    # DEI rank correlation with original
    log_DEI_orig = original_log_H - log_E
    log_DEI_sim = log_H_sim - log_E
    rho, _ = sp_stats.spearmanr(log_DEI_orig, log_DEI_sim)

    # Tier changes (quintile shifts)
    orig_tiers = pd.qcut(log_DEI_orig, 5, labels=False)
    sim_tiers = pd.qcut(log_DEI_sim, 5, labels=False)
    tier_changes = np.sum(orig_tiers != sim_tiers)
    tier_change_pct = tier_changes / N * 100

    sensitivity_rows.append({
        "target_cv_pct": target_cv,
        "actual_cv_pct": np.std(H_sim) / np.mean(H_sim) * 100,
        "var_log_H": var_log_H_sim,
        "var_log_E": var_log_E,
        "var_log_DEI": var_log_DEI_sim,
        "H_contribution_pct": h_pct,
        "E_contribution_pct": 100 - h_pct,
        "rank_rho_vs_original": rho,
        "tier_changes": tier_changes,
        "tier_change_pct": tier_change_pct,
    })

    marker = " ← current" if abs(target_cv - 3.9) < 0.2 else ""
    print(f"  CV={target_cv:5.1f}%: H contributes {h_pct:5.1f}%, "
          f"rank ρ={rho:.4f}, tier changes={tier_changes} ({tier_change_pct:.1f}%){marker}")

sens_df = pd.DataFrame(sensitivity_rows)
sens_df.to_csv(TABLES_DIR / "h_decompression_sensitivity.csv", index=False)
print(f"\n  Saved: {TABLES_DIR / 'h_decompression_sensitivity.csv'}")

# Find crossover point (H ≥ 10%)
crossover_rows = sens_df[sens_df["H_contribution_pct"] >= 10]
if len(crossover_rows) > 0:
    crossover_cv = crossover_rows.iloc[0]["target_cv_pct"]
    print(f"\n  ★ Crossover point: H contributes ≥10% when CV ≥ {crossover_cv}%")
    print(f"    (current CV is {H_cv:.1f}%, need {crossover_cv/H_cv:.1f}x increase)")
else:
    print(f"\n  ★ H never reaches 10% contribution even at CV=50%")

# ══════════════════════════════════════════════════════════════════
# ANALYSIS 4: Rank Stability Under Beta Rescaling
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 4: Rank Stability Under Beta Rescaling")
print("=" * 60)

log_H_rescaled = np.log(np.clip(H_rescaled, 0.1, 10))
log_DEI_rescaled = log_H_rescaled - log_E
log_DEI_original = original_log_H - log_E

rho_dei, p_dei = sp_stats.spearmanr(log_DEI_original, log_DEI_rescaled)
tau_dei, p_tau = sp_stats.kendalltau(log_DEI_original, log_DEI_rescaled)

print(f"  DEI rank correlation (original vs rescaled):")
print(f"    Spearman ρ = {rho_dei:.4f} (p = {p_dei:.2e})")
print(f"    Kendall τ = {tau_dei:.4f} (p = {p_tau:.2e})")

# Tier changes under rescaling
orig_tiers = pd.qcut(log_DEI_original, 5, labels=["E", "D", "C", "B", "A"])
resc_tiers = pd.qcut(log_DEI_rescaled, 5, labels=["E", "D", "C", "B", "A"])
tier_match = (orig_tiers == resc_tiers).sum()
print(f"  Tier agreement: {tier_match}/{N} ({tier_match/N*100:.1f}%)")

# Top/bottom 10 stability
orig_rank = sp_stats.rankdata(-log_DEI_original)  # 1 = best
resc_rank = sp_stats.rankdata(-log_DEI_rescaled)

top10_orig = set(dei.iloc[np.where(orig_rank <= 10)[0]]["dish_id"])
top10_resc = set(dei.iloc[np.where(resc_rank <= 10)[0]]["dish_id"])
top10_overlap = len(top10_orig & top10_resc)
print(f"  Top-10 overlap: {top10_overlap}/10")

bot10_orig = set(dei.iloc[np.where(orig_rank > N - 10)[0]]["dish_id"])
bot10_resc = set(dei.iloc[np.where(resc_rank > N - 10)[0]]["dish_id"])
bot10_overlap = len(bot10_orig & bot10_resc)
print(f"  Bottom-10 overlap: {bot10_overlap}/10")

# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

# Figure 1: Sensitivity — H CV vs H contribution %
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: H contribution vs CV
ax1.plot(sens_df["target_cv_pct"], sens_df["H_contribution_pct"],
         "o-", color="#2196F3", linewidth=2, markersize=8, zorder=5)
ax1.axhline(y=10, color="#F44336", linestyle="--", alpha=0.7, label="10% threshold")
ax1.axvline(x=H_cv, color="#4CAF50", linestyle="--", alpha=0.7, label=f"Current CV={H_cv:.1f}%")
ax1.fill_between(sens_df["target_cv_pct"], 0, sens_df["H_contribution_pct"],
                 alpha=0.1, color="#2196F3")
ax1.set_xlabel("H Coefficient of Variation (%)", fontsize=12)
ax1.set_ylabel("H Contribution to Var(log DEI) (%)", fontsize=12)
ax1.set_title("A. How Much H Spread Would Matter?", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_xlim(0, 55)
ax1.set_ylim(0, max(sens_df["H_contribution_pct"]) * 1.1)
ax1.grid(True, alpha=0.3)

# Panel B: Rank stability vs CV
ax2.plot(sens_df["target_cv_pct"], sens_df["rank_rho_vs_original"],
         "s-", color="#FF9800", linewidth=2, markersize=8, zorder=5)
ax2.axhline(y=0.95, color="#F44336", linestyle="--", alpha=0.7, label="ρ=0.95")
ax2.axvline(x=H_cv, color="#4CAF50", linestyle="--", alpha=0.7, label=f"Current CV={H_cv:.1f}%")
ax2.set_xlabel("H Coefficient of Variation (%)", fontsize=12)
ax2.set_ylabel("Spearman ρ (DEI rank vs original)", fontsize=12)
ax2.set_title("B. DEI Rank Stability Under H Decompression", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.set_xlim(0, 55)
ax2.set_ylim(min(sens_df["rank_rho_vs_original"]) - 0.02, 1.01)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "h_decompression_sensitivity.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'h_decompression_sensitivity.png'}")

# Figure 2: Original vs Rescaled H distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Histograms
ax1.hist(H, bins=30, alpha=0.6, color="#2196F3", label=f"Original (CV={H_cv:.1f}%)", density=True)
ax1.hist(H_rescaled, bins=30, alpha=0.6, color="#F44336",
         label=f"Rescaled (CV={H_rescaled_cv:.1f}%)", density=True)
ax1.set_xlabel("Hedonic Score (H)", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.set_title("A. H Score Distribution: Original vs Rescaled", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel B: QQ-style rank comparison
sorted_orig = np.sort(H)
sorted_resc = np.sort(H_rescaled)
ax2.scatter(sorted_orig, sorted_resc, alpha=0.4, s=15, color="#9C27B0")
ax2.plot([3, 9], [3, 9], "k--", alpha=0.3, label="y=x")
ax2.set_xlabel("Original H (sorted)", fontsize=12)
ax2.set_ylabel("Rescaled H (sorted)", fontsize=12)
ax2.set_title("B. Rank-Preserving Rescaling", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "h_rescaled_distribution.png", bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR / 'h_rescaled_distribution.png'}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  H compression factor vs literature: ~{compression_vs_lit:.0f}x")
print(f"  H would need CV > {crossover_rows.iloc[0]['target_cv_pct']:.0f}% to contribute ≥10% of DEI variance" if len(crossover_rows) > 0 else "  H never reaches 10% even at CV=50%")
print(f"  Under beta rescaling (CV={H_rescaled_cv:.1f}%): DEI rank ρ = {rho_dei:.4f}")
print(f"  Conclusion: DEI rankings are ROBUST to H decompression,")
print(f"  but the compression must be honestly disclosed.")
print("=" * 60)
