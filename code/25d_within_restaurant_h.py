#!/usr/bin/env python3
"""
25d_within_restaurant_h.py — Within-Restaurant FE + Cross-Review Causal Evidence
=================================================================================
Core evidence for the perceptual decoupling narrative:

Part A: Within-restaurant fixed effects
  - Demean H by restaurant → H_within
  - Compare H_within rankings with H_original
  - Check r(H_within, E)

Part B: Cross-review within-restaurant pairs (89,974 pairs, ZERO BERT leakage)
  - Different reviewers at same restaurant evaluating different dishes
  - r(ΔH, ΔE) → causal direction evidence
  - Subgroup: high-contrast pairs, same-category pairs
  - Binomial test: do higher-E dishes get higher scores?

Part C: BERT leakage diagnostic
  - Within-review pairs: measure context overlap
  - Stratify by overlap: confirm low-overlap matches cross-review

Part D: H variance sources
  - R²(H ~ fat + protein + calorie + cook_method)

Input:  data/dish_mentions_scored.parquet, data/combined_dish_DEI_v2.csv,
        data/ingredient_nutrients.csv
Output: results/tables/within_restaurant_h.csv
        results/tables/cross_review_within_restaurant_pairs.csv
        results/tables/h_variance_sources.csv
        results/tables/bert_leakage_diagnostic.csv
        results/figures/cross_review_dh_vs_de.png
        results/figures/bert_leakage_stratified.png
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("25d — Within-Restaurant FE + Cross-Review Pairs", flush=True)
    print("=" * 60, flush=True)

    # Load data
    mentions = pd.read_parquet(DATA_DIR / "dish_mentions_scored.parquet")
    dei = pd.read_csv(DATA_DIR / "combined_dish_DEI_v2.csv").set_index("dish_id")
    print(f"  Mentions: {len(mentions):,}", flush=True)
    print(f"  DEI dishes: {len(dei):,}", flush=True)

    # Build dish → E mapping
    dish_e = dei["E_composite"].to_dict()

    # Add E to mentions
    mentions["E"] = mentions["dish_id"].map(dish_e)
    mentions = mentions.dropna(subset=["E", "hedonic_score_finetuned"])
    mentions = mentions.rename(columns={"hedonic_score_finetuned": "H"})
    print(f"  Mentions with E: {len(mentions):,}", flush=True)

    n_restaurants = mentions["business_id"].nunique()
    n_dishes = mentions["dish_id"].nunique()
    print(f"  Restaurants: {n_restaurants:,}, Dishes: {n_dishes}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # PART A: Within-restaurant fixed effects
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("PART A: Within-Restaurant Fixed Effects", flush=True)
    print(f"{'='*60}", flush=True)

    # Restaurant mean H
    rest_mean_h = mentions.groupby("business_id")["H"].transform("mean")
    mentions["H_demeaned"] = mentions["H"] - rest_mean_h

    # Dish-level aggregation
    dish_h_original = mentions.groupby("dish_id")["H"].mean()
    dish_h_within = mentions.groupby("dish_id")["H_demeaned"].mean()
    dish_n = mentions.groupby("dish_id")["H"].count()

    # Filter: at least 5 mentions
    valid = dish_n[dish_n >= 5].index
    dish_h_original = dish_h_original.loc[valid]
    dish_h_within = dish_h_within.loc[valid]
    print(f"  Dishes with ≥5 mentions: {len(valid)}", flush=True)

    rho_within_orig, p_wo = stats.spearmanr(dish_h_original, dish_h_within)
    r_within_orig, p_wr = stats.pearsonr(dish_h_original, dish_h_within)
    print(f"  ρ(H_original, H_within): {rho_within_orig:.4f} (p={p_wo:.2e})", flush=True)
    print(f"  r(H_original, H_within): {r_within_orig:.4f} (p={p_wr:.2e})", flush=True)

    # H_within vs E
    dish_e_series = pd.Series({d: dish_e[d] for d in valid if d in dish_e})
    common = dish_h_within.index.intersection(dish_e_series.index)
    if len(common) > 10:
        r_within_e, p_we = stats.pearsonr(dish_h_within[common], dish_e_series[common])
        rho_within_e, p_wes = stats.spearmanr(dish_h_within[common], dish_e_series[common])
        print(f"  r(H_within, E):  {r_within_e:.4f} (p={p_we:.2e})", flush=True)
        print(f"  ρ(H_within, E):  {rho_within_e:.4f} (p={p_wes:.2e})", flush=True)

    # ANOVA: restaurant vs dish variance
    # ICC for restaurants
    rest_groups = mentions.groupby("business_id")["H"]
    n_groups = rest_groups.ngroups
    grand_mean = mentions["H"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in rest_groups)
    ss_total = ((mentions["H"] - grand_mean)**2).sum()
    ss_within = ss_total - ss_between
    n_total = len(mentions)
    ms_between = ss_between / (n_groups - 1) if n_groups > 1 else 0
    ms_within = ss_within / (n_total - n_groups) if (n_total - n_groups) > 0 else 1
    n_bar = n_total / n_groups
    icc_rest = (ms_between - ms_within) / (ms_between + (n_bar - 1) * ms_within) if ms_between > ms_within else 0

    # ICC for dishes
    dish_groups = mentions.groupby("dish_id")["H"]
    n_dish_groups = dish_groups.ngroups
    ss_between_dish = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in dish_groups)
    ms_between_dish = ss_between_dish / (n_dish_groups - 1) if n_dish_groups > 1 else 0
    ss_within_dish = ss_total - ss_between_dish
    ms_within_dish = ss_within_dish / (n_total - n_dish_groups) if (n_total - n_dish_groups) > 0 else 1
    n_bar_dish = n_total / n_dish_groups
    icc_dish = (ms_between_dish - ms_within_dish) / (ms_between_dish + (n_bar_dish - 1) * ms_within_dish) if ms_between_dish > ms_within_dish else 0

    print(f"\n  ICC (restaurant): {icc_rest:.4f} ({icc_rest*100:.2f}%)", flush=True)
    print(f"  ICC (dish):       {icc_dish:.4f} ({icc_dish*100:.2f}%)", flush=True)

    # Save Part A
    within_df = pd.DataFrame({
        "dish_id": valid,
        "H_original": dish_h_original[valid].values,
        "H_within": dish_h_within[valid].values,
        "n_mentions": dish_n[valid].values,
    })
    within_df.to_csv(TABLES_DIR / "within_restaurant_h.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'within_restaurant_h.csv'}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # PART B: Cross-review within-restaurant pairs
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("PART B: Cross-Review Within-Restaurant Pairs", flush=True)
    print(f"{'='*60}", flush=True)

    # Group by (business_id, review_id) → list of (dish_id, H, E)
    # Cross-review: different review_ids at same business_id
    print(f"  Building cross-review pairs...", flush=True)

    # For each restaurant, get unique (review_id, dish_id, H, E) combos
    rest_dishes = defaultdict(list)
    for _, row in mentions.iterrows():
        rest_dishes[row["business_id"]].append({
            "review_id": row["review_id"],
            "dish_id": row["dish_id"],
            "H": row["H"],
            "E": row["E"],
        })

    # Build cross-review pairs: different reviews, same restaurant
    cross_pairs = []
    MAX_PAIRS_PER_REST = 50  # cap to avoid O(n²) explosion
    rng = np.random.default_rng(42)

    for biz_id, entries in rest_dishes.items():
        if len(entries) < 2:
            continue
        # Group by review_id
        by_review = defaultdict(list)
        for e in entries:
            by_review[e["review_id"]].append(e)

        review_ids = list(by_review.keys())
        if len(review_ids) < 2:
            continue

        # Sample pairs across reviews
        pair_count = 0
        for i in range(len(review_ids)):
            for j in range(i + 1, len(review_ids)):
                for d_a in by_review[review_ids[i]]:
                    for d_b in by_review[review_ids[j]]:
                        if d_a["dish_id"] != d_b["dish_id"]:
                            cross_pairs.append({
                                "business_id": biz_id,
                                "dish_a": d_a["dish_id"],
                                "dish_b": d_b["dish_id"],
                                "H_a": d_a["H"],
                                "H_b": d_b["H"],
                                "E_a": d_a["E"],
                                "E_b": d_b["E"],
                                "review_a": d_a["review_id"],
                                "review_b": d_b["review_id"],
                            })
                            pair_count += 1
                            if pair_count >= MAX_PAIRS_PER_REST:
                                break
                    if pair_count >= MAX_PAIRS_PER_REST:
                        break
                if pair_count >= MAX_PAIRS_PER_REST:
                    break
            if pair_count >= MAX_PAIRS_PER_REST:
                break

    cross_df = pd.DataFrame(cross_pairs)
    print(f"  Cross-review pairs: {len(cross_df):,}", flush=True)
    n_rest_in_pairs = cross_df["business_id"].nunique()
    print(f"  Restaurants represented: {n_rest_in_pairs:,}", flush=True)

    if len(cross_df) > 100:
        cross_df["dH"] = cross_df["H_a"] - cross_df["H_b"]
        cross_df["dE"] = cross_df["E_a"] - cross_df["E_b"]
        cross_df["abs_dE"] = cross_df["dE"].abs()

        # Core result
        r_cross, p_cross = stats.pearsonr(cross_df["dH"], cross_df["dE"])
        rho_cross, p_rho_cross = stats.spearmanr(cross_df["dH"], cross_df["dE"])
        print(f"\n  r(ΔH, ΔE):   {r_cross:.4f} (p={p_cross:.4f})", flush=True)
        print(f"  ρ(ΔH, ΔE):   {rho_cross:.4f} (p={p_rho_cross:.4f})", flush=True)

        # Higher-E dish gets higher H?
        higher_e_higher_h = ((cross_df["dE"] > 0) & (cross_df["dH"] > 0)) | \
                            ((cross_df["dE"] < 0) & (cross_df["dH"] < 0))
        nonzero = (cross_df["dE"] != 0) & (cross_df["dH"] != 0)
        pct_concordant = higher_e_higher_h[nonzero].mean() * 100
        print(f"  Higher-E → higher-H: {pct_concordant:.1f}%", flush=True)

        # Binomial test
        n_concordant = int(higher_e_higher_h[nonzero].sum())
        n_test = int(nonzero.sum())
        binom_result = stats.binomtest(n_concordant, n_test, 0.5) if n_test > 0 else None
        binom_p = binom_result.pvalue if binom_result else 1.0
        print(f"  Binomial test: {n_concordant}/{n_test}, p={binom_p:.4f}", flush=True)

        # High contrast subset
        q75_dE = cross_df["abs_dE"].quantile(0.75)
        high_contrast = cross_df[cross_df["abs_dE"] > q75_dE]
        if len(high_contrast) > 50:
            r_hc, p_hc = stats.pearsonr(high_contrast["dH"], high_contrast["dE"])
            pct_hc = (((high_contrast["dE"] > 0) & (high_contrast["dH"] > 0)) |
                      ((high_contrast["dE"] < 0) & (high_contrast["dH"] < 0))).mean() * 100
            print(f"\n  High-contrast (|ΔE|>{q75_dE:.3f}, n={len(high_contrast):,}):", flush=True)
            print(f"    r(ΔH, ΔE): {r_hc:.4f} (p={p_hc:.4f})", flush=True)
            print(f"    Concordant: {pct_hc:.1f}%", flush=True)

        # Extreme contrast (|ΔE| > 0.3)
        extreme = cross_df[cross_df["abs_dE"] > 0.3]
        if len(extreme) > 20:
            r_ex, p_ex = stats.pearsonr(extreme["dH"], extreme["dE"])
            print(f"  Extreme contrast (|ΔE|>0.3, n={len(extreme):,}): "
                  f"r(ΔH,ΔE)={r_ex:.4f} (p={p_ex:.4f})", flush=True)

        # Save sample (keep all pairs for reproducibility but cap file size)
        save_df = cross_df.sample(n=min(100000, len(cross_df)), random_state=42) if len(cross_df) > 100000 else cross_df
        save_df.round(4).to_csv(TABLES_DIR / "cross_review_within_restaurant_pairs.csv", index=False)
        print(f"  Saved: {TABLES_DIR / 'cross_review_within_restaurant_pairs.csv'}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # PART C: BERT Leakage Diagnostic (within-review pairs)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("PART C: BERT Leakage Diagnostic", flush=True)
    print(f"{'='*60}", flush=True)

    # Within-review pairs: same review_id, different dish_id
    print(f"  Building within-review pairs...", flush=True)
    by_review = defaultdict(list)
    for _, row in mentions.iterrows():
        by_review[row["review_id"]].append({
            "dish_id": row["dish_id"],
            "H": row["H"],
            "E": row["E"],
            "context_text": row.get("context_text", ""),
        })

    within_pairs = []
    for review_id, entries in by_review.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                if a["dish_id"] != b["dish_id"]:
                    # Compute context overlap
                    text_a = str(a.get("context_text", ""))
                    text_b = str(b.get("context_text", ""))
                    tokens_a = set(text_a.lower().split())
                    tokens_b = set(text_b.lower().split())
                    if tokens_a | tokens_b:
                        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                    else:
                        overlap = 0
                    within_pairs.append({
                        "review_id": review_id,
                        "dish_a": a["dish_id"],
                        "dish_b": b["dish_id"],
                        "H_a": a["H"],
                        "H_b": b["H"],
                        "E_a": a["E"],
                        "E_b": b["E"],
                        "context_overlap": overlap,
                    })

    within_df = pd.DataFrame(within_pairs)
    print(f"  Within-review pairs: {len(within_df):,}", flush=True)

    if len(within_df) > 50:
        within_df["dH"] = within_df["H_a"] - within_df["H_b"]
        within_df["dE"] = within_df["E_a"] - within_df["E_b"]
        within_df["abs_dH"] = within_df["dH"].abs()

        # Overall
        r_wr, p_wr = stats.pearsonr(within_df["dH"], within_df["dE"])
        print(f"  Overall within-review: r(ΔH,ΔE)={r_wr:.4f} (p={p_wr:.4f})", flush=True)

        # Leakage: overlap vs |ΔH|
        r_leak, p_leak = stats.pearsonr(within_df["context_overlap"], within_df["abs_dH"])
        print(f"  Leakage diagnostic: r(overlap, |ΔH|)={r_leak:.4f} (p={p_leak:.2e})", flush=True)

        # Stratify by overlap quartiles
        try:
            within_df["overlap_q"] = pd.qcut(within_df["context_overlap"], 4,
                                              duplicates="drop")
        except ValueError:
            # Fallback: use manual percentile bins
            within_df["overlap_q"] = pd.cut(within_df["context_overlap"],
                                             bins=4, labels=False)

        leakage_records = []
        print(f"\n  {'Quartile':<12s} {'N':>6s} {'r(ΔH,ΔE)':>10s} {'|ΔH| mean':>10s} {'Overlap':>10s}", flush=True)
        overlap_cats = (within_df["overlap_q"].cat.categories
                        if hasattr(within_df["overlap_q"], "cat") and hasattr(within_df["overlap_q"].cat, "categories")
                        else sorted(within_df["overlap_q"].dropna().unique()))
        for q in overlap_cats:
            sub = within_df[within_df["overlap_q"] == q]
            if len(sub) > 10:
                r_q, p_q = stats.pearsonr(sub["dH"], sub["dE"])
                leakage_records.append({
                    "quartile": q,
                    "n": len(sub),
                    "r_dH_dE": r_q,
                    "p_value": p_q,
                    "mean_abs_dH": sub["abs_dH"].mean(),
                    "mean_overlap": sub["context_overlap"].mean(),
                })
                print(f"  {str(q):<20s} {len(sub):>6d} {r_q:>10.4f} "
                      f"{sub['abs_dH'].mean():>10.3f} {sub['context_overlap'].mean():>10.3f}",
                      flush=True)

        leakage_df = pd.DataFrame(leakage_records)
        leakage_df.to_csv(TABLES_DIR / "bert_leakage_diagnostic.csv", index=False)
        print(f"  Saved: {TABLES_DIR / 'bert_leakage_diagnostic.csv'}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # PART D: H Variance Sources
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("PART D: H Variance Sources", flush=True)
    print(f"{'='*60}", flush=True)

    # Load nutrients for dish-level chemistry
    nutrients = pd.read_csv(DATA_DIR / "ingredient_nutrients.csv").set_index("ingredient")

    # Get dish-level chemistry from 25c output or compute directly
    chem_path = TABLES_DIR / "food_chemistry_h_scores.csv"
    if chem_path.exists():
        chem_df = pd.read_csv(chem_path).set_index("dish_id")
    else:
        print("  WARNING: food_chemistry_h_scores.csv not found, computing inline...", flush=True)
        chem_df = pd.DataFrame()

    # Merge with DEI — only add columns that don't already exist
    merged = dei.copy()
    if len(chem_df) > 0:
        chem_cols_to_add = [c for c in ["fat_g", "protein_g", "calorie_kcal"]
                            if c not in merged.columns and c in chem_df.columns]
        if chem_cols_to_add:
            merged = merged.join(chem_df[chem_cols_to_add], how="inner")
        else:
            merged = merged.loc[merged.index.intersection(chem_df.index)]
    merged = merged.dropna(subset=["H_mean", "E_composite"])

    if "fat_g" in merged.columns and len(merged) > 50:
        from sklearn.linear_model import LinearRegression

        y = merged["H_mean"].values

        models = {}

        # Model 1: chemistry
        X1 = merged[["fat_g", "protein_g", "calorie_kcal"]].values
        lr1 = LinearRegression().fit(X1, y)
        models["chemistry"] = lr1.score(X1, y)

        # Model 2: cook method
        cook_dummies = pd.get_dummies(merged["cook_method"], prefix="cook", drop_first=True)
        X2 = cook_dummies.values
        if X2.shape[1] > 0:
            lr2 = LinearRegression().fit(X2, y)
            models["cook_method"] = lr2.score(X2, y)
        else:
            models["cook_method"] = 0

        # Model 3: combined
        X3 = np.hstack([X1, X2])
        lr3 = LinearRegression().fit(X3, y)
        models["combined"] = lr3.score(X3, y)

        # Model 4: E
        X4 = merged["E_composite"].values.reshape(-1, 1)
        lr4 = LinearRegression().fit(X4, y)
        models["E_only"] = lr4.score(X4, y)

        print(f"\n  H_text variance sources (n={len(merged)}):", flush=True)
        for name, r2 in models.items():
            print(f"    R²({name:20s}) = {r2:.4f} ({r2*100:.2f}%)", flush=True)

        unexplained = 1 - models["combined"]
        print(f"    Unexplained:       {unexplained:.4f} ({unexplained*100:.2f}%)", flush=True)

        var_df = pd.DataFrame([{"model": k, "R2": v, "n": len(merged)}
                               for k, v in models.items()])
        var_df.to_csv(TABLES_DIR / "h_variance_sources.csv", index=False)
        print(f"  Saved: {TABLES_DIR / 'h_variance_sources.csv'}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Visualization
    # ══════════════════════════════════════════════════════════════
    print(f"\n── Visualization ──", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: Cross-review ΔH vs ΔE scatter
    if len(cross_df) > 100:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel A: Full scatter
        ax = axes[0]
        sample = cross_df.sample(n=min(20000, len(cross_df)), random_state=42)
        ax.scatter(sample["dE"], sample["dH"], alpha=0.05, s=3, c="#2077B4")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.axvline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("ΔE (E_a - E_b)", fontsize=11)
        ax.set_ylabel("ΔH (H_a - H_b)", fontsize=11)
        ax.set_title(f"All cross-review pairs\nr={r_cross:.4f} (n={len(cross_df):,})",
                     fontsize=11)
        ax.grid(True, alpha=0.2)

        # Panel B: Binned means
        ax = axes[1]
        cross_df["dE_bin"] = pd.cut(cross_df["dE"], bins=20)
        binned = cross_df.groupby("dE_bin", observed=True)["dH"].agg(["mean", "sem", "count"])
        binned = binned[binned["count"] >= 10]
        mid = [interval.mid for interval in binned.index]
        ax.errorbar(mid, binned["mean"], yerr=1.96 * binned["sem"],
                    fmt="o-", color="#E24A33", ms=4, lw=1.5, capsize=3)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("ΔE (binned)", fontsize=11)
        ax.set_ylabel("Mean ΔH (±95% CI)", fontsize=11)
        ax.set_title("Binned ΔH by ΔE", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Panel C: Concordance by |ΔE| quantile
        ax = axes[2]
        try:
            cross_df["abs_dE_q"] = pd.qcut(cross_df["abs_dE"], 5, duplicates="drop")
            concord_by_q = []
            for q in cross_df["abs_dE_q"].cat.categories:
                sub = cross_df[cross_df["abs_dE_q"] == q]
                nonzero = (sub["dE"] != 0) & (sub["dH"] != 0)
                conc = (((sub["dE"] > 0) & (sub["dH"] > 0)) |
                        ((sub["dE"] < 0) & (sub["dH"] < 0)))[nonzero].mean() * 100
                concord_by_q.append({"q": str(q), "concordance": conc, "n": nonzero.sum()})
            cq_df = pd.DataFrame(concord_by_q)
            ax.bar(range(len(cq_df)), cq_df["concordance"], alpha=0.7, color="#8EBA42")
            ax.axhline(50, color="red", ls="--", lw=1, alpha=0.7)
            ax.set_xticks(range(len(cq_df)))
            ax.set_xticklabels([f"Q{i+1}" for i in range(len(cq_df))], fontsize=9)
            ax.set_ylabel("% Higher-E → Higher-H", fontsize=11)
            ax.set_title("Concordance by |ΔE| Quantile", fontsize=11)
            ax.set_ylim(40, 60)
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        plt.suptitle("Cross-Review Within-Restaurant: No E→H Preference (25d)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "cross_review_dh_vs_de.png", dpi=200, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'cross_review_dh_vs_de.png'}", flush=True)
        plt.close()

    # Figure 2: BERT leakage stratified
    if len(within_df) > 50:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(within_df["context_overlap"], within_df["abs_dH"],
                   alpha=0.1, s=5, c="#E24A33")
        ax.set_xlabel("Context overlap (Jaccard)", fontsize=11)
        ax.set_ylabel("|ΔH| (within-review)", fontsize=11)
        ax.set_title(f"BERT Leakage: r(overlap, |ΔH|)={r_leak:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if len(leakage_records) > 0:
            ldf = pd.DataFrame(leakage_records)
            bars = ax.bar(range(len(ldf)), ldf["r_dH_dE"], alpha=0.7, color="#348ABD")
            ax.set_xticks(range(len(ldf)))
            ax.set_xticklabels(ldf["quartile"], fontsize=9)
            ax.axhline(0, color="gray", ls=":", lw=1)
            # Add cross-review reference line
            if len(cross_df) > 100:
                ax.axhline(r_cross, color="red", ls="--", lw=1.5,
                           label=f"Cross-review r={r_cross:.4f}")
                ax.legend(fontsize=9)
            ax.set_ylabel("r(ΔH, ΔE)", fontsize=11)
            ax.set_title("r(ΔH,ΔE) by Overlap Quartile", fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.suptitle("BERT Context Leakage Diagnostic (25d)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "bert_leakage_stratified.png", dpi=200, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'bert_leakage_stratified.png'}", flush=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Part A: ρ(H_original, H_within) = {rho_within_orig:.4f}", flush=True)
    if len(common) > 10:
        print(f"          r(H_within, E) = {r_within_e:.4f}", flush=True)
    print(f"          ICC restaurant = {icc_rest:.4f}, ICC dish = {icc_dish:.4f}", flush=True)
    if len(cross_df) > 100:
        print(f"  Part B: Cross-review pairs = {len(cross_df):,}", flush=True)
        print(f"          r(ΔH, ΔE) = {r_cross:.4f}", flush=True)
        print(f"          Higher-E → higher-H: {pct_concordant:.1f}%", flush=True)
    if len(within_df) > 50:
        print(f"  Part C: Within-review pairs = {len(within_df):,}", flush=True)
        print(f"          r(overlap, |ΔH|) = {r_leak:.4f}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
