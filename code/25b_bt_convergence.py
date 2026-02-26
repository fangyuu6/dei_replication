#!/usr/bin/env python3
"""
25b_bt_convergence.py — BT Convergence Curve + Split-Half Consistency
=====================================================================
Shows that pairwise H scores stabilize with increasing comparison data,
addressing the critique that variance decomposition is an arithmetic artifact.

Pipeline:
  1. Load 62,343 pairwise comparisons
  2. Subsample at 10%, 20%, ..., 100%
  3. Fit BT at each level → CV, rank correlation with full BT
  4. Split-half consistency at full data
  5. Plot convergence curve

Input:  data/pairwise_wins_v2.csv, data/combined_dish_DEI_v2.csv
Output: results/tables/bt_convergence_curve.csv
        results/figures/bt_convergence.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import choix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 20  # number of random splits for confidence intervals


def fit_bt(comparisons, n_items, alpha=0.01):
    """Fit BT and return 1-10 scaled scores."""
    params = choix.ilsr_pairwise(n_items, comparisons, alpha=alpha)
    p_min, p_max = params.min(), params.max()
    if p_max > p_min:
        scores = 1 + 9 * (params - p_min) / (p_max - p_min)
    else:
        scores = np.full(n_items, 5.5)
    return scores, params


def main():
    print("=" * 60, flush=True)
    print("25b — BT Convergence Curve", flush=True)
    print("=" * 60, flush=True)

    # Load data
    wins = pd.read_csv(DATA_DIR / "pairwise_wins_v2.csv")
    dei = pd.read_csv(DATA_DIR / "combined_dish_DEI_v2.csv")
    print(f"  Pairwise wins: {len(wins):,}", flush=True)
    print(f"  Dishes in DEI: {len(dei):,}", flush=True)

    # Build dish-id mapping
    all_dishes = sorted(set(wins["winner"].unique()) | set(wins["loser"].unique()))
    id2idx = {d: i for i, d in enumerate(all_dishes)}
    n_items = len(all_dishes)
    print(f"  Unique dishes in comparisons: {n_items}", flush=True)

    # Convert to comparison tuples
    comparisons = []
    for _, row in wins.iterrows():
        w = id2idx.get(row["winner"])
        l = id2idx.get(row["loser"])
        if w is not None and l is not None:
            comparisons.append((w, l))
    comparisons = np.array(comparisons)
    print(f"  Valid comparisons: {len(comparisons):,}", flush=True)

    # ── Full BT fit (reference) ──
    full_scores, full_params = fit_bt(comparisons.tolist(), n_items)
    full_cv = np.std(full_scores) / np.mean(full_scores) * 100
    print(f"\n  Full BT: CV={full_cv:.2f}%, range=[{full_scores.min():.2f}, {full_scores.max():.2f}]",
          flush=True)

    # ── Convergence curve ──
    fractions = np.arange(0.05, 1.05, 0.05)  # 5% to 100%
    rng = np.random.default_rng(SEED)

    records = []
    for frac in fractions:
        n_sample = int(len(comparisons) * frac)
        cvs = []
        rhos = []

        for split_i in range(N_SPLITS):
            idx = rng.choice(len(comparisons), size=n_sample, replace=False)
            sub = comparisons[idx].tolist()
            try:
                scores, _ = fit_bt(sub, n_items)
                cv = np.std(scores) / np.mean(scores) * 100
                rho, _ = stats.spearmanr(scores, full_scores)
                cvs.append(cv)
                rhos.append(rho)
            except Exception:
                pass

        if cvs:
            records.append({
                "fraction": frac,
                "n_comparisons": n_sample,
                "cv_mean": np.mean(cvs),
                "cv_std": np.std(cvs),
                "cv_q25": np.percentile(cvs, 25),
                "cv_q75": np.percentile(cvs, 75),
                "rho_mean": np.mean(rhos),
                "rho_std": np.std(rhos),
                "rho_q25": np.percentile(rhos, 25),
                "rho_q75": np.percentile(rhos, 75),
            })
            print(f"  {frac*100:5.0f}% ({n_sample:6,d} comps): "
                  f"CV={np.mean(cvs):.2f}±{np.std(cvs):.2f}%, "
                  f"ρ={np.mean(rhos):.4f}±{np.std(rhos):.4f}",
                  flush=True)

    conv_df = pd.DataFrame(records)

    # ── Split-half consistency ──
    print(f"\n  Split-half consistency ({N_SPLITS} random splits):", flush=True)
    split_rhos = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(comparisons))
        half = len(comparisons) // 2
        half_a = comparisons[perm[:half]].tolist()
        half_b = comparisons[perm[half:2*half]].tolist()
        try:
            scores_a, _ = fit_bt(half_a, n_items)
            scores_b, _ = fit_bt(half_b, n_items)
            rho, _ = stats.spearmanr(scores_a, scores_b)
            split_rhos.append(rho)
        except Exception:
            pass

    if split_rhos:
        print(f"    Mean split-half ρ: {np.mean(split_rhos):.4f} "
              f"± {np.std(split_rhos):.4f}", flush=True)
        print(f"    Range: [{min(split_rhos):.4f}, {max(split_rhos):.4f}]", flush=True)

    # ── Variance decomposition at each level ──
    # Merge with E to check H contribution stability
    dei_idx = dei.set_index("dish_id")
    dish_has_e = [d for d in all_dishes if d in dei_idx.index]

    print(f"\n  Variance decomposition stability:", flush=True)
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        n_sample = int(len(comparisons) * frac)
        h_contribs = []
        for _ in range(min(N_SPLITS, 10)):
            if frac < 1.0:
                idx = rng.choice(len(comparisons), size=n_sample, replace=False)
                sub = comparisons[idx].tolist()
            else:
                sub = comparisons.tolist()
            try:
                scores, _ = fit_bt(sub, n_items)
                score_map = {d: scores[i] for i, d in enumerate(all_dishes)}
                h_vals = np.array([np.log(score_map[d]) for d in dish_has_e if d in score_map])
                e_vals = np.array([dei_idx.loc[d, "log_E"] for d in dish_has_e if d in score_map])
                var_h = np.var(h_vals)
                var_e = np.var(e_vals)
                h_contribs.append(var_h / (var_h + var_e) * 100)
            except Exception:
                pass
        if h_contribs:
            print(f"    {frac*100:5.0f}%: H contrib = {np.mean(h_contribs):.2f}% "
                  f"± {np.std(h_contribs):.2f}%", flush=True)

    # ── Save ──
    conv_df.to_csv(TABLES_DIR / "bt_convergence_curve.csv", index=False)
    print(f"\n  Saved: {TABLES_DIR / 'bt_convergence_curve.csv'}", flush=True)

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: CV convergence
    ax = axes[0]
    ax.plot(conv_df["fraction"] * 100, conv_df["cv_mean"], "o-", color="#2077B4", lw=2, ms=4)
    ax.fill_between(conv_df["fraction"] * 100, conv_df["cv_q25"], conv_df["cv_q75"],
                     alpha=0.2, color="#2077B4")
    ax.axhline(full_cv, color="red", ls="--", lw=1, alpha=0.7, label=f"Full CV={full_cv:.1f}%")
    ax.set_xlabel("% of comparisons used", fontsize=11)
    ax.set_ylabel("CV (%)", fontsize=11)
    ax.set_title("BT Score CV Convergence", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Rank stability
    ax = axes[1]
    ax.plot(conv_df["fraction"] * 100, conv_df["rho_mean"], "o-", color="#E87D2F", lw=2, ms=4)
    ax.fill_between(conv_df["fraction"] * 100, conv_df["rho_q25"], conv_df["rho_q75"],
                     alpha=0.2, color="#E87D2F")
    ax.set_xlabel("% of comparisons used", fontsize=11)
    ax.set_ylabel("Spearman ρ (vs full BT)", fontsize=11)
    ax.set_title("Rank Stability vs Full Data", fontsize=12)
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)

    # Panel 3: Split-half distribution
    ax = axes[2]
    if split_rhos:
        ax.hist(split_rhos, bins=15, alpha=0.7, color="#8EBA42", edgecolor="white")
        ax.axvline(np.mean(split_rhos), color="red", ls="--", lw=1.5,
                   label=f"Mean={np.mean(split_rhos):.3f}")
        ax.set_xlabel("Split-half Spearman ρ", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Split-Half Consistency", fontsize=12)
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("BT Convergence Analysis (25b)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bt_convergence.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {FIGURES_DIR / 'bt_convergence.png'}", flush=True)
    plt.close()

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
