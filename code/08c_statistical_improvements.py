"""
08c_statistical_improvements.py — P2-6: Statistical Presentation Improvements
===============================================================================
Upgrades all key statistics to publication standard.

Improvements:
  1. Bootstrap 95% CIs for split-half reliability
  2. HC3 robust standard errors for OLS regression
  3. Monte Carlo median rank shift + P90 rank shift
  4. Bootstrap CIs for cross-platform correlations
  5. Comprehensive summary table

Outputs:
  - tables/statistical_improvements.csv
  - tables/ols_hc3_regression.csv
  - tables/bootstrap_reliability.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, FIGURES_DIR, TABLES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_BOOT = 5000
print(f"Bootstrap iterations: {N_BOOT:,}")

# ── Load ─────────────────────────────────────────────────────────
dei = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv")
mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
H = "hedonic_score_finetuned"

print(f"Loaded {len(dei)} dishes, {len(mentions):,} mentions")

# ══════════════════════════════════════════════════════════════════
# 1. Bootstrap split-half reliability with 95% CI
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. Bootstrap split-half reliability")
print("=" * 60)

def split_half_reliability(mentions_df, h_col, n_boot=N_BOOT):
    """Bootstrap split-half reliability with 95% CI."""
    dish_ids = mentions_df["dish_id"].unique()
    boot_rhos = []

    for b in range(n_boot):
        # For each dish, randomly split mentions in half
        h1, h2 = {}, {}
        for dish in dish_ids:
            scores = mentions_df.loc[mentions_df["dish_id"] == dish, h_col].values
            np.random.shuffle(scores)
            mid = len(scores) // 2
            if mid < 3:
                continue
            h1[dish] = scores[:mid].mean()
            h2[dish] = scores[mid:].mean()

        common = set(h1.keys()) & set(h2.keys())
        if len(common) < 20:
            continue
        vals1 = [h1[d] for d in common]
        vals2 = [h2[d] for d in common]
        rho, _ = stats.spearmanr(vals1, vals2)
        boot_rhos.append(rho)

        if (b + 1) % 1000 == 0:
            print(f"    Bootstrap {b+1}/{n_boot}...")

    boot_rhos = np.array(boot_rhos)
    return {
        "mean": boot_rhos.mean(),
        "median": np.median(boot_rhos),
        "ci_lo": np.percentile(boot_rhos, 2.5),
        "ci_hi": np.percentile(boot_rhos, 97.5),
        "std": boot_rhos.std(),
        "n_boot": len(boot_rhos),
    }

reliability = split_half_reliability(mentions, H)
print(f"  Split-half reliability: {reliability['mean']:.4f}")
print(f"  95% CI: [{reliability['ci_lo']:.4f}, {reliability['ci_hi']:.4f}]")
print(f"  → Spearman-Brown corrected: {2*reliability['mean']/(1+reliability['mean']):.4f}")

sb_corrected = 2 * reliability['mean'] / (1 + reliability['mean'])
sb_lo = 2 * reliability['ci_lo'] / (1 + reliability['ci_lo'])
sb_hi = 2 * reliability['ci_hi'] / (1 + reliability['ci_hi'])
print(f"  → SB corrected 95% CI: [{sb_lo:.4f}, {sb_hi:.4f}]")

# ══════════════════════════════════════════════════════════════════
# 2. OLS with HC3 robust standard errors
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. OLS regression with HC3 robust standard errors")
print("=" * 60)

# log(DEI) ~ log(E_carbon_norm) + log(E_water_norm) + log(E_energy_norm)
valid = dei.dropna(subset=["E_carbon_norm", "E_water_norm", "E_energy_norm"]).copy()
# Clip to avoid log(0)
for col in ["E_carbon_norm", "E_water_norm", "E_energy_norm"]:
    valid[col] = valid[col].clip(lower=1e-6)

y = valid["log_DEI"].values
X = np.column_stack([
    np.ones(len(valid)),
    np.log(valid["E_carbon_norm"].values),
    np.log(valid["E_water_norm"].values),
    np.log(valid["E_energy_norm"].values),
])

# OLS
from numpy.linalg import lstsq, inv
beta, _, _, _ = lstsq(X, y, rcond=None)
residuals = y - X @ beta
n, k = X.shape

# R²
ss_res = (residuals ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

# HC3 robust standard errors
# HC3: u_i^2 / (1 - h_ii)^2
H_hat = X @ inv(X.T @ X) @ X.T
h_diag = np.diag(H_hat)
u_hc3 = residuals / (1 - h_diag)
meat = X.T @ np.diag(u_hc3 ** 2) @ X
bread = inv(X.T @ X)
V_hc3 = bread @ meat @ bread
se_hc3 = np.sqrt(np.diag(V_hc3))

# Regular OLS SE for comparison
mse = ss_res / (n - k)
se_ols = np.sqrt(np.diag(mse * inv(X.T @ X)))

# t-stats and p-values (HC3)
t_hc3 = beta / se_hc3
p_hc3 = 2 * (1 - stats.t.cdf(np.abs(t_hc3), df=n - k))

# 95% CI
ci_lo = beta - 1.96 * se_hc3
ci_hi = beta + 1.96 * se_hc3

feature_names = ["intercept", "log(E_carbon_norm)", "log(E_water_norm)", "log(E_energy_norm)"]
print(f"\n  R² = {r2:.6f}, Adj R² = {r2_adj:.6f}, n = {n}")
print(f"\n  {'Feature':25s} {'Coef':>8s} {'SE_OLS':>8s} {'SE_HC3':>8s} {'t_HC3':>8s} {'p':>10s} {'95% CI':>20s}")
print(f"  {'-'*90}")

ols_results = []
for name, b, se_o, se_h, t, p, lo, hi in zip(
    feature_names, beta, se_ols, se_hc3, t_hc3, p_hc3, ci_lo, ci_hi):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:25s} {b:+8.4f} {se_o:8.4f} {se_h:8.4f} {t:8.2f} {p:10.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")
    ols_results.append({
        "feature": name, "coef": b, "se_ols": se_o, "se_hc3": se_h,
        "t_hc3": t, "p_hc3": p, "ci95_lo": lo, "ci95_hi": hi,
    })

# ══════════════════════════════════════════════════════════════════
# 3. Monte Carlo rank shift statistics (from 07e results)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Monte Carlo rank shift summary")
print("=" * 60)

rank_intervals = pd.read_csv(TABLES_DIR / "dei_rank_intervals.csv")

median_shift = rank_intervals["rank_90CI_width"].median()
mean_shift = rank_intervals["rank_90CI_width"].mean()
p90_shift = rank_intervals["rank_90CI_width"].quantile(0.9)

print(f"  Median rank 90% CI width: {median_shift:.0f} positions")
print(f"  Mean rank 90% CI width: {mean_shift:.0f} positions")
print(f"  P90 rank 90% CI width: {p90_shift:.0f} positions")

# Median rank shift from point estimate
rank_intervals["median_shift"] = (rank_intervals["rank_mc_median"] - rank_intervals["rank_original"]).abs()
print(f"  Median |rank shift| from point estimate: {rank_intervals['median_shift'].median():.1f}")
print(f"  P90 |rank shift|: {rank_intervals['median_shift'].quantile(0.9):.1f}")

# ══════════════════════════════════════════════════════════════════
# 4. Bootstrap CIs for cross-platform correlations
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Bootstrap CIs for cross-platform H correlations")
print("=" * 60)

cross_h = pd.read_csv(DATA_DIR / "cross_platform_h_scores.csv")
platform_cols = [c for c in cross_h.columns if c.startswith("H_") and c != "H_yelp"]

boot_corr_results = []
for col in platform_cols:
    valid_mask = cross_h[["H_yelp", col]].dropna()
    if len(valid_mask) < 10:
        continue

    x = valid_mask["H_yelp"].values
    y_vals = valid_mask[col].values

    # Point estimate
    rho, p_rho = stats.spearmanr(x, y_vals)

    # Bootstrap
    boot_rhos = []
    for _ in range(N_BOOT):
        idx = np.random.randint(0, len(x), len(x))
        r_boot, _ = stats.spearmanr(x[idx], y_vals[idx])
        if not np.isnan(r_boot):
            boot_rhos.append(r_boot)

    boot_rhos = np.array(boot_rhos)
    ci_lo = np.percentile(boot_rhos, 2.5)
    ci_hi = np.percentile(boot_rhos, 97.5)

    platform_name = col.replace("H_", "")
    print(f"  Yelp vs {platform_name:15s}: ρ = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (n={len(valid_mask)})")

    boot_corr_results.append({
        "platform": platform_name,
        "spearman_rho": rho,
        "p_value": p_rho,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n": len(valid_mask),
    })

# ══════════════════════════════════════════════════════════════════
# 5. Bootstrap CIs for key dish-level statistics
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Bootstrap CIs for key statistics")
print("=" * 60)

# Bootstrap CI for variance decomposition ratio (Var_E / Var_H in log space)
boot_var_ratios = []
for _ in range(N_BOOT):
    idx = np.random.randint(0, len(dei), len(dei))
    sample = dei.iloc[idx]
    var_h = sample["log_H"].var()
    var_e = sample["log_E"].var()
    if var_h > 0:
        boot_var_ratios.append(var_e / var_h)

boot_var_ratios = np.array(boot_var_ratios)
print(f"  Var(log_E)/Var(log_H) ratio:")
print(f"    Point estimate: {dei['log_E'].var() / dei['log_H'].var():.1f}")
print(f"    95% CI: [{np.percentile(boot_var_ratios, 2.5):.1f}, {np.percentile(boot_var_ratios, 97.5):.1f}]")

# Bootstrap CI for R²
boot_r2s = []
for _ in range(N_BOOT):
    idx = np.random.randint(0, len(valid), len(valid))
    y_b = y[idx]
    X_b = X[idx]
    beta_b, _, _, _ = lstsq(X_b, y_b, rcond=None)
    pred_b = X_b @ beta_b
    ss_res_b = ((y_b - pred_b) ** 2).sum()
    ss_tot_b = ((y_b - y_b.mean()) ** 2).sum()
    if ss_tot_b > 0:
        boot_r2s.append(1 - ss_res_b / ss_tot_b)

boot_r2s = np.array(boot_r2s)
print(f"\n  OLS R²:")
print(f"    Point estimate: {r2:.6f}")
print(f"    95% CI: [{np.percentile(boot_r2s, 2.5):.6f}, {np.percentile(boot_r2s, 97.5):.6f}]")

# ══════════════════════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════════════════════
ols_df = pd.DataFrame(ols_results)
ols_df.to_csv(TABLES_DIR / "ols_hc3_regression.csv", index=False)

boot_rel_df = pd.DataFrame([{
    "statistic": "split_half_reliability",
    "point_estimate": reliability["mean"],
    "ci95_lo": reliability["ci_lo"],
    "ci95_hi": reliability["ci_hi"],
    "spearman_brown": sb_corrected,
    "sb_ci_lo": sb_lo,
    "sb_ci_hi": sb_hi,
}])
boot_rel_df.to_csv(TABLES_DIR / "bootstrap_reliability.csv", index=False)

boot_corr_df = pd.DataFrame(boot_corr_results)
boot_corr_df.to_csv(TABLES_DIR / "cross_platform_bootstrap_ci.csv", index=False)

# Comprehensive summary
summary_rows = [
    {"metric": "Split-half reliability (Spearman)", "value": f"{reliability['mean']:.3f}",
     "CI95": f"[{reliability['ci_lo']:.3f}, {reliability['ci_hi']:.3f}]"},
    {"metric": "Split-half (Spearman-Brown corrected)", "value": f"{sb_corrected:.3f}",
     "CI95": f"[{sb_lo:.3f}, {sb_hi:.3f}]"},
    {"metric": "OLS R² (log_DEI ~ log_E components)", "value": f"{r2:.4f}",
     "CI95": f"[{np.percentile(boot_r2s, 2.5):.4f}, {np.percentile(boot_r2s, 97.5):.4f}]"},
    {"metric": "MC rank 90% CI width (median)", "value": f"{median_shift:.0f}",
     "CI95": "—"},
    {"metric": "MC rank 90% CI width (P90)", "value": f"{p90_shift:.0f}",
     "CI95": "—"},
    {"metric": "CV(E)/CV(H) ratio", "value": f"{dei['E_composite'].std()/dei['E_composite'].mean() / (dei['H_mean'].std()/dei['H_mean'].mean()):.1f}x",
     "CI95": "—"},
]

for res in boot_corr_results:
    summary_rows.append({
        "metric": f"Yelp vs {res['platform']} (Spearman ρ)",
        "value": f"{res['spearman_rho']:.3f}",
        "CI95": f"[{res['ci95_lo']:.3f}, {res['ci95_hi']:.3f}]",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLES_DIR / "statistical_improvements.csv", index=False)

print(f"\n  Saved: {TABLES_DIR / 'ols_hc3_regression.csv'}")
print(f"  Saved: {TABLES_DIR / 'bootstrap_reliability.csv'}")
print(f"  Saved: {TABLES_DIR / 'cross_platform_bootstrap_ci.csv'}")
print(f"  Saved: {TABLES_DIR / 'statistical_improvements.csv'}")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: Statistical Improvements")
print("=" * 60)
print(summary_df.to_string(index=False))
print("\nDone!")
