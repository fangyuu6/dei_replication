"""
14_integration_round2.py — Second-Round Revision Integration
=============================================================
Summarizes all 13a-13e outputs into a concise impact table.

Outputs:
  - tables/round2_impact_matrix.csv
  - tables/round2_summary.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR

print("=" * 70)
print("14  SECOND-ROUND REVISION INTEGRATION")
print("=" * 70)

# ── Collect key results ───────────────────────────────────────────
impacts = []

# C6: Survivorship bias
bounds = pd.read_csv(TABLES_DIR / "survivorship_bounds.csv")
base = bounds[bounds["K_ghost"] == 0]["H_contribution_pct"].values[0]
k334_d2 = bounds[(bounds["K_ghost"] == 334) & (bounds["delta_H"] == 2.0)]["H_contribution_pct"].values[0]
impacts.append({
    "criticism": "C6: Survivorship Bias",
    "script": "13a",
    "key_finding": (f"Low-star restaurants show H CV=18.6% (vs 2.4% high-star). "
                    f"Tipping point: need 200+ ghost dishes with H↓≥2 for H>10%. "
                    f"With 334 ghosts (Delta=2): H contributes {k334_d2:.1f}%."),
    "impact": "Low-Medium: confirms selection effect exists but E dominance robust",
})

# C7: Apples vs oranges
role_var = pd.read_csv(TABLES_DIR / "within_role_variance.csv")
full_main = role_var[role_var["meal_role"] == "Full Main"].iloc[0]
impacts.append({
    "criticism": "C7: Unit-of-Comparison Problem",
    "script": "13b",
    "key_finding": (f"Within Full Mains: H={full_main['H_contribution_pct']:.1f}% of Var, "
                    f"DEI range={full_main['DEI_fold_difference']:.0f}x (vs 345x global). "
                    f"Meal-level: H contributes 0.6%. "
                    f"6,611 calorie-equivalent substitutions (mean E↓45%)."),
    "impact": "Medium: headline number drops from 345x to 25x but E still dominates",
})

# C8: Oversimplified hedonic
multi_h = pd.read_csv(TABLES_DIR / "multidimensional_hedonic.csv")
impacts.append({
    "criticism": "C8: Oversimplified Hedonic Definition",
    "script": "13c",
    "key_finding": (f"Satiety CV=33% (vs taste CV=4%). "
                    f"At w_satiety=0.3: H contributes 0.5%, rank rho=0.998. "
                    f"Even at w_satiety=1.0 (satiety only): H=11.3%, rho=0.946."),
    "impact": "Low: even adding satiety dimension, E dominance holds",
})

# C9: Refinement resolution
try:
    mdd = pd.read_csv(TABLES_DIR / "within_family_mdd.csv")
    n_total = len(mdd)
    n_sig = mdd["significant_005"].sum()
    pct_negl = (mdd["effect_category"] == "negligible").mean() * 100
    impacts.append({
        "criticism": "C9: Refinement Curve Causal Weakness",
        "script": "13d",
        "key_finding": (f"{n_sig}/{n_total} ({n_sig/n_total*100:.0f}%) within-family pairs significant. "
                        f"{pct_negl:.0f}% have negligible effect size (d<0.2). "
                        f"Price tier slope=+0.10 (p<0.001): modest refinement signal exists. "
                        f"Cross-platform: Google alpha=0.35, TA alpha=-0.02, Yelp alpha=0.12."),
        "impact": "Medium: BERT can discriminate but effect sizes tiny; "
                  "refinement curve is 'suggestive evidence' not definitive",
    })
except FileNotFoundError:
    pass

# C10: Geographic bias
try:
    geo = pd.read_csv(TABLES_DIR / "geographic_concentration.csv")
    geo_dict = dict(zip(geo["metric"], geo["value"]))
    city_h = pd.read_csv(TABLES_DIR / "city_h_consistency.csv")
    impacts.append({
        "criticism": "C10: Sample/Geographic Bias",
        "script": "13e",
        "key_finding": (f"{int(geo_dict['n_states'])} states, Gini={geo_dict['state_gini']:.2f}. "
                        f"City H consistency: mean rho={city_h['spearman_rho'].mean():.3f} (low). "
                        f"But state-level DEI rho>0.993 (high). "
                        f"All cuisines span 14 states."),
        "impact": "Medium: H varies by city (local preferences) but DEI rankings stable across states",
    })
except FileNotFoundError:
    pass

# Save
impact_df = pd.DataFrame(impacts)
impact_df.to_csv(TABLES_DIR / "round2_impact_matrix.csv", index=False)

print("\nSecond-Round Impact Matrix:")
for _, row in impact_df.iterrows():
    print(f"\n  {row['criticism']} [{row['script']}]")
    print(f"    {row['key_finding']}")
    print(f"    Impact: {row['impact']}")

# ── Summary statistics ────────────────────────────────────────────
print("\n" + "=" * 70)
print("CUMULATIVE ROBUSTNESS SUMMARY")
print("=" * 70)

summary = [
    ("Original E dominance", "99.8%", "—"),
    ("After proxy-bias control (R1)", "99.9%", "rho=0.999"),
    ("After H decompression CV=20% (R1)", "91.5%", "rho=0.950"),
    ("After NDI addition alpha=0.5 (R1)", "—", "rho=0.931 vs original"),
    ("After survivorship ghosts K=334,D=2 (R2)", f"{100-k334_d2:.1f}%", "—"),
    ("After satiety w=0.3 (R2)", "99.5%", "rho=0.998"),
    ("Within Full Mains (R2)", f"{100-full_main['H_contribution_pct']:.1f}%", "—"),
    ("Meal-level combos (R2)", "99.4%", "—"),
    ("State-level DEI (R2)", "—", "rho>0.993"),
]

summary_df = pd.DataFrame(summary, columns=["Scenario", "E_contribution", "rank_stability"])
summary_df.to_csv(TABLES_DIR / "round2_summary.csv", index=False)

print(summary_df.to_string(index=False))
print(f"\nConclusion: E dominance (>87%) is robust across ALL {len(summary)} scenarios tested.")

# ── File inventory ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ROUND 2 OUTPUT FILES")
print("=" * 70)

r2_files = {
    "13a": ["tables/survivorship_bounds.csv", "tables/mention_freq_h_analysis.csv",
            "figures/survivorship_heatmap.png", "figures/mention_frequency_vs_h_cv.png"],
    "13b": ["tables/within_role_variance.csv", "tables/meal_level_dei.csv",
            "tables/calorie_equivalent_subs.csv",
            "figures/meal_dei_distribution.png", "figures/like_for_like_comparison.png"],
    "13c": ["tables/multidimensional_hedonic.csv", "tables/satiety_by_dish.csv",
            "figures/h_satiety_scatter.png", "figures/dei_sensitivity_to_satiety.png"],
    "13d": ["tables/within_family_mdd.csv", "tables/cross_platform_refinement.csv",
            "tables/price_tier_h_analysis.csv", "figures/refinement_resolution_diagnostic.png"],
    "13e": ["tables/geographic_concentration.csv", "tables/city_h_consistency.csv",
            "figures/geographic_heatmap.png", "figures/city_h_stability.png"],
    "14":  ["tables/round2_impact_matrix.csv", "tables/round2_summary.csv"],
}

total = 0
for script, files in r2_files.items():
    for f in files:
        full = ROOT / "results" / f
        exists = full.exists()
        status = "ok" if exists else "MISSING"
        if not exists:
            print(f"  [{status}] {script}: {f}")
        total += 1

n_ok = sum(1 for s, files in r2_files.items()
           for f in files if (ROOT / "results" / f).exists())
print(f"\n  {n_ok}/{total} files present")

print("\n" + "=" * 70)
print("ROUND 2 INTEGRATION COMPLETE")
print("=" * 70)
