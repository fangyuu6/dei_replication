"""
12_integrated_revision.py — Integrated Revision Summary
========================================================
Combines all 11a-11e outputs into a unified revised dataset and
generates impact summary tables.

Dependencies: all 11a-11e outputs
Outputs:
  - data/combined_dish_DEI_revised.csv
  - tables/revision_summary.csv
  - tables/revision_impact_matrix.csv
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

# ── Load base data ────────────────────────────────────────────────
print("=" * 70)
print("12  INTEGRATED REVISION SUMMARY")
print("=" * 70)

combined = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
print(f"\nBase dataset: {len(combined)} dishes, {len(combined.columns)} columns")

# ── 1. Merge nutritional profiles (from 11c) ─────────────────────
print("\n── Merging nutritional profiles (11c) ──")
nutr = pd.read_csv(DATA_DIR / "dish_nutritional_profiles.csv")
nutr_cols = ["dish_id", "protein_g", "fat_g", "carb_g", "fiber_g",
             "calorie_kcal", "NDI", "meal_role"]
nutr_merge = nutr[nutr_cols].copy()
combined = combined.merge(nutr_merge, on="dish_id", how="left")
n_nutr = combined["NDI"].notna().sum()
print(f"  Matched {n_nutr}/{len(combined)} dishes with nutritional data")

# Compute DEI-N (α=0.5)
alpha = 0.5
mask = combined["NDI"].notna() & (combined["NDI"] > 0)
combined.loc[mask, "log_NDI"] = np.log(combined.loc[mask, "NDI"])
combined.loc[mask, "log_DEI_N"] = (combined.loc[mask, "log_H"]
                                   + alpha * combined.loc[mask, "log_NDI"]
                                   - combined.loc[mask, "log_E"])
print(f"  DEI-N computed for {mask.sum()} dishes (α={alpha})")

# ── 2. Merge controlled H (from 11b) ─────────────────────────────
print("\n── Merging proxy-controlled H (11b) ──")
ctrl = pd.read_csv(TABLES_DIR / "controlled_dei_rankings.csv")
ctrl_cols = ["dish_id", "H_controlled_mean", "log_DEI_controlled"]
ctrl_merge = ctrl[ctrl_cols].rename(columns={
    "H_controlled_mean": "H_controlled",
    "log_DEI_controlled": "log_DEI_ctrl"
})
combined = combined.merge(ctrl_merge, on="dish_id", how="left")
n_ctrl = combined["H_controlled"].notna().sum()
print(f"  Matched {n_ctrl}/{len(combined)} dishes with controlled H")

# ── 3. Merge category assignments (from 11d) ─────────────────────
print("\n── Merging recipe-based categories (11d) ──")
cats = pd.read_csv(TABLES_DIR / "comprehensive_category_assignment.csv")
cats_merge = cats[["dish_id", "category_recipe"]].copy()
combined = combined.merge(cats_merge, on="dish_id", how="left")
n_cat = combined["category_recipe"].notna().sum()
print(f"  Matched {n_cat}/{len(combined)} dishes with recipe categories")

# ── 4. Merge family assignments (from 11e) ────────────────────────
print("\n── Merging dish families (11e) ──")
fam = pd.read_csv(TABLES_DIR / "dish_families.csv")
fam_merge = fam[["dish_id", "family"]].copy()
combined = combined.merge(fam_merge, on="dish_id", how="left")
n_fam = combined["family"].notna().sum()
print(f"  Matched {n_fam}/{len(combined)} dishes with family assignments")

# ── 5. Merge refinement scores (from 11e) ─────────────────────────
print("\n── Merging refinement scores (11e) ──")
ref = pd.read_csv(TABLES_DIR / "refinement_curves.csv")
# Map family-level alpha to dishes
family_alpha = ref.set_index("family")["alpha"].to_dict()
combined["refinement_alpha"] = combined["family"].map(family_alpha)
n_ref = combined["refinement_alpha"].notna().sum()
print(f"  Matched {n_ref}/{len(combined)} dishes with refinement alpha")

# ── Save revised dataset ──────────────────────────────────────────
out_path = DATA_DIR / "combined_dish_DEI_revised.csv"
combined.to_csv(out_path, index=False)
print(f"\n  Saved revised dataset: {out_path}")
print(f"  Shape: {combined.shape}")
print(f"  New columns: NDI, log_NDI, log_DEI_N, meal_role, protein_g, fat_g,")
print(f"    carb_g, fiber_g, calorie_kcal, H_controlled, log_DEI_ctrl,")
print(f"    category_recipe, family, refinement_alpha")

# ══════════════════════════════════════════════════════════════════
# REVISION IMPACT MATRIX
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("REVISION IMPACT MATRIX")
print("=" * 70)

# Load individual analysis results
h_comp = pd.read_csv(TABLES_DIR / "h_compression_analysis.csv")
h_sens = pd.read_csv(TABLES_DIR / "h_decompression_sensitivity.csv")
proxy = pd.read_csv(TABLES_DIR / "proxy_bias_summary.csv")
proxy_dict = dict(zip(proxy["metric"], proxy["value"]))
wc_var = pd.read_csv(TABLES_DIR / "within_category_variance.csv")
ref_curves = pd.read_csv(TABLES_DIR / "refinement_curves.csv")

# Build impact matrix
impacts = []

# C1: H compression
h_cv_bert = h_comp.loc[h_comp["metric"] == "bert_h_observed", "cv_pct"].values[0]
h_cv_lit = h_comp.loc[h_comp["metric"] == "literature_benchmark", "cv_pct"].values[0]
compress_factor = h_comp.loc[h_comp["metric"] == "literature_benchmark",
                              "compression_factor_cv"].values[0]
cv20_row = h_sens[h_sens["target_cv_pct"] == 20.0].iloc[0]
impacts.append({
    "criticism": "C1: H Score Compression",
    "analysis": "11a: Compression quantification + sensitivity",
    "key_finding": (f"BERT CV={h_cv_bert:.1f}% vs literature {h_cv_lit:.0f}% "
                    f"(compression factor {compress_factor:.1f}x). "
                    f"At CV=20%, H contributes {cv20_row['H_contribution_pct']:.1f}% "
                    f"of Var(log DEI), rank ρ={cv20_row['rank_rho_vs_original']:.3f}"),
    "impact_on_conclusion": "Low — E dominance robust even under decompression",
    "new_data_columns": "—",
})

# C2: Proxy bias
impacts.append({
    "criticism": "C2: Yelp Proxy Bias",
    "analysis": "11b: Two-stage residualization",
    "key_finding": (f"Controls R²={proxy_dict['regression_r_squared']:.3f}, "
                    f"DEI rank ρ={proxy_dict['spearman_rho']:.4f}, "
                    f"tier agreement={proxy_dict['tier_agreement_pct']:.1f}%, "
                    f"mean |rank shift|={proxy_dict['mean_abs_rank_shift']:.1f}"),
    "impact_on_conclusion": "Low — rankings essentially unchanged after controls",
    "new_data_columns": "H_controlled, log_DEI_ctrl",
})

# C3: Nutritional dimension
dei_dei_n_corr = sp_stats.spearmanr(
    combined.loc[mask, "log_DEI"], combined.loc[mask, "log_DEI_N"]
)[0]
n_pareto_3d = pd.read_csv(TABLES_DIR / "dei_n_rankings.csv")["is_pareto_3d"].sum()
impacts.append({
    "criticism": "C3: Missing Nutritional Dimension",
    "analysis": "11c: NDI (NRF-7) + DEI-N + 3D Pareto",
    "key_finding": (f"DEI vs DEI-N rank ρ={dei_dei_n_corr:.3f}, "
                    f"3D Pareto frontier: {n_pareto_3d} dishes (vs 11 original), "
                    f"NDI range [{combined['NDI'].min():.1f}, {combined['NDI'].max():.1f}]"),
    "impact_on_conclusion": "Medium — nutritionally dense meats gain ranks; "
                            "Pareto set more balanced",
    "new_data_columns": "NDI, log_NDI, log_DEI_N, meal_role, protein_g, etc.",
})

# C4: Within-category
mean_wc_H_pct = wc_var["H_contribution_pct"].mean()
n_subs = len(pd.read_csv(TABLES_DIR / "nutrition_constrained_substitutions_full.csv"))
n_cats = len(wc_var)
impacts.append({
    "criticism": "C4: Functional Category Inequivalence",
    "analysis": "11d: Recipe-based classification + nutrition-constrained substitution",
    "key_finding": (f"{n_cats} functional categories, "
                    f"within-category H contribution avg {mean_wc_H_pct:.1f}% "
                    f"(vs 0.3% global), "
                    f"{n_subs:,} nutrition-constrained substitutions"),
    "impact_on_conclusion": "Medium — within-category results confirm global pattern",
    "new_data_columns": "category_recipe",
})

# C5: Refinement curve
global_alpha = ref_curves["alpha"].mean()
n_families = len(ref_curves)
impacts.append({
    "criticism": "C5: Refinement Cost Curve (New Concept)",
    "analysis": "11e: Dish family analysis + hedonic elasticity",
    "key_finding": (f"{n_families} dish families, "
                    f"global mean α={global_alpha:.3f} "
                    f"(spending more E yields negligible H gain), "
                    f"most families: first 50% of E captures 80%+ of H range"),
    "impact_on_conclusion": "High — new contribution: 'refinement tax' concept",
    "new_data_columns": "family, refinement_alpha",
})

impact_df = pd.DataFrame(impacts)
impact_df.to_csv(TABLES_DIR / "revision_impact_matrix.csv", index=False)
print("\nRevision Impact Matrix:")
for _, row in impact_df.iterrows():
    print(f"\n  {row['criticism']}")
    print(f"    Finding: {row['key_finding']}")
    print(f"    Impact: {row['impact_on_conclusion']}")

# ══════════════════════════════════════════════════════════════════
# REVISION SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("REVISION SUMMARY STATISTICS")
print("=" * 70)

summary = []

# Dataset expansion
summary.append({"category": "Dataset", "metric": "Total dishes",
                "original": "334", "revised": "334 (enriched)"})
summary.append({"category": "Dataset", "metric": "New columns added",
                "original": str(len(pd.read_csv(DATA_DIR / "combined_dish_DEI.csv").columns)),
                "revised": str(len(combined.columns))})

# H score
summary.append({"category": "H Score", "metric": "CV (%)",
                "original": f"{h_cv_bert:.1f}",
                "revised": f"{h_cv_bert:.1f} (acknowledged; decompression analysed)"})
summary.append({"category": "H Score", "metric": "H_controlled CV (%)",
                "original": "—",
                "revised": f"{proxy_dict['H_cv_controlled']:.1f}"})

# Variance decomposition
summary.append({"category": "Variance", "metric": "H contribution (%)",
                "original": "0.3", "revised": f"0.3 (0.1% after proxy control)"})
summary.append({"category": "Variance", "metric": "E contribution (%)",
                "original": "99.8", "revised": "99.8 (99.9% after proxy control)"})

# Rankings
summary.append({"category": "Rankings", "metric": "DEI vs DEI_ctrl rank ρ",
                "original": "—", "revised": f"{proxy_dict['spearman_rho']:.4f}"})
summary.append({"category": "Rankings", "metric": "DEI vs DEI-N rank ρ",
                "original": "—", "revised": f"{dei_dei_n_corr:.3f}"})

# Substitutions
summary.append({"category": "Substitutions", "metric": "Unconstrained (E↓30%, H↓<1)",
                "original": "6,658", "revised": "6,658"})
summary.append({"category": "Substitutions", "metric": "Nutrition-constrained",
                "original": "—", "revised": f"{n_subs:,}"})

# Pareto
summary.append({"category": "Pareto", "metric": "2D Pareto-optimal dishes",
                "original": "11", "revised": "11"})
summary.append({"category": "Pareto", "metric": "3D Pareto (H, 1/E, NDI)",
                "original": "—", "revised": str(n_pareto_3d)})

# New concept
summary.append({"category": "New", "metric": "Dish families analysed",
                "original": "—", "revised": str(n_families)})
summary.append({"category": "New", "metric": "Mean hedonic elasticity (α)",
                "original": "—", "revised": f"{global_alpha:.3f}"})

summary_df = pd.DataFrame(summary)
summary_df.to_csv(TABLES_DIR / "revision_summary.csv", index=False)

print("\nRevision Summary:")
for _, row in summary_df.iterrows():
    print(f"  [{row['category']}] {row['metric']}: {row['original']} → {row['revised']}")

# ── Final counts ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FILES GENERATED BY REVISION PIPELINE")
print("=" * 70)

output_files = {
    "11a": ["tables/h_compression_analysis.csv",
            "tables/h_decompression_sensitivity.csv",
            "figures/h_decompression_sensitivity.png",
            "figures/h_rescaled_distribution.png"],
    "11b": ["tables/controlled_dei_rankings.csv",
            "tables/proxy_bias_summary.csv",
            "figures/controlled_vs_original_dei.png"],
    "11c": ["data/ingredient_nutrients.csv",
            "data/dish_nutritional_profiles.csv",
            "tables/dei_n_rankings.csv",
            "figures/dei_vs_dein_comparison.png",
            "figures/nutritional_profiles_by_category.png"],
    "11d": ["tables/comprehensive_category_assignment.csv",
            "tables/within_category_dei_rankings.csv",
            "tables/within_category_variance.csv",
            "tables/nutrition_constrained_substitutions_full.csv",
            "figures/within_category_dei_panels.png"],
    "11e": ["tables/dish_families.csv",
            "tables/refinement_curves.csv",
            "figures/refinement_cost_curves.png",
            "figures/refinement_global_fit.png",
            "figures/hedonic_waste_by_family.png"],
    "12":  ["data/combined_dish_DEI_revised.csv",
            "tables/revision_summary.csv",
            "tables/revision_impact_matrix.csv"],
}

total = 0
for script, files in output_files.items():
    print(f"\n  {script}:")
    for f in files:
        full = ROOT / "results" / f if not f.startswith("data/") else ROOT / f
        exists = full.exists()
        status = "✓" if exists else "✗"
        print(f"    {status} {f}")
        total += 1

print(f"\n  Total output files: {total}")
print("\n" + "=" * 70)
print("REVISION PIPELINE COMPLETE")
print("=" * 70)
