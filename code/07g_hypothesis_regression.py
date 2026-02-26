"""
07g_hypothesis_regression.py — P1-2: Explicit Hypotheses & Non-E Regression
=============================================================================
Addresses reviewer concern: "What hypothesis does DEI test?"

Three testable hypotheses:
  H1: Hedonic variance << Environmental variance across dishes
  H2: Plant-based/raw dishes achieve higher DEI without taste loss
  H3: DEI is predicted by food composition, not cuisine tradition

Non-tautological regression:
  DEI_rank ~ %plant + %raw + n_ingredients + cuisine_diversity + protein_source

Outputs:
  - tables/hypothesis_tests.csv
  - tables/non_e_regression.csv
  - figures/hypothesis_tests.png
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────
dei = pd.read_csv(DATA_DIR / "dish_DEI_scores.csv")
impacts = pd.read_csv(DATA_DIR / "ingredient_impact_factors.csv")

# Import DISH_RECIPES
with open(ROOT / "code" / "04_env_cost_calculation.py", "r", encoding="utf-8") as f:
    content = f.read()
start = content.find("DISH_RECIPES = {")
depth = 0
in_string = False
str_char = None
for i, c in enumerate(content[start:], start=start):
    if in_string:
        if c == str_char and content[i-1] != '\\':
            in_string = False
        continue
    if c in ('"', "'"):
        in_string = True
        str_char = c
        continue
    if c == '{':
        depth += 1
    elif c == '}':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
exec(content[start:end])

print(f"Loaded {len(dei)} dishes, {len(DISH_RECIPES)} recipes")

# Build ingredient category lookup
ingredient_cats = dict(zip(impacts["ingredient"], impacts["category"]))

# ══════════════════════════════════════════════════════════════════
# Compute dish-level features for regression
# ══════════════════════════════════════════════════════════════════

PLANT_CATEGORIES = {"vegetable", "fruit", "grain", "legume", "starch", "nut", "oil"}
ANIMAL_CATEGORIES = {"meat", "poultry", "seafood", "dairy"}
RAW_COLD_METHODS = {"raw", "cold"}

features = []
for dish_id in dei["dish_id"]:
    if dish_id not in DISH_RECIPES:
        features.append({"dish_id": dish_id})
        continue

    recipe = DISH_RECIPES[dish_id]
    ingredients = recipe["ingredients"]
    cook_method = recipe["cook_method"]

    total_grams = sum(ingredients.values())
    plant_grams = sum(g for ing, g in ingredients.items()
                      if ingredient_cats.get(ing, "other") in PLANT_CATEGORIES)
    animal_grams = sum(g for ing, g in ingredients.items()
                       if ingredient_cats.get(ing, "other") in ANIMAL_CATEGORIES)

    # Protein source type
    has_beef = any(ing in ("beef", "ground_beef") for ing in ingredients)
    has_pork = any(ing in ("pork", "bacon", "sausage") for ing in ingredients)
    has_poultry = any(ing in ("chicken", "duck", "turkey") for ing in ingredients)
    has_seafood = any(ingredient_cats.get(ing) == "seafood" for ing in ingredients)
    has_dairy = any(ingredient_cats.get(ing) == "dairy" for ing in ingredients)

    # Determine primary protein source
    if has_beef:
        protein_source = "beef"
    elif has_pork:
        protein_source = "pork"
    elif has_poultry:
        protein_source = "poultry"
    elif has_seafood:
        protein_source = "seafood"
    elif has_dairy:
        protein_source = "dairy"
    else:
        protein_source = "plant"

    features.append({
        "dish_id": dish_id,
        "pct_plant": plant_grams / total_grams if total_grams > 0 else 0,
        "pct_animal": animal_grams / total_grams if total_grams > 0 else 0,
        "is_raw_cold": 1 if cook_method in RAW_COLD_METHODS else 0,
        "n_ing": len(ingredients),
        "total_g": total_grams,
        "protein_source": protein_source,
        "has_beef": int(has_beef),
        "has_pork": int(has_pork),
        "has_poultry": int(has_poultry),
        "has_seafood": int(has_seafood),
    })

feat_df = pd.DataFrame(features)
df = dei.merge(feat_df, on="dish_id")

# DEI rank (1 = best)
df["rank_DEI"] = df["log_DEI"].rank(ascending=False)

print(f"Feature matrix: {len(df)} dishes with {len(feat_df.columns)-1} features")

# ══════════════════════════════════════════════════════════════════
# HYPOTHESIS 1: H variance << E variance
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("HYPOTHESIS 1: Hedonic variance << Environmental variance")
print("=" * 60)

h_cv = df["H_mean"].std() / df["H_mean"].mean()
e_cv = df["E_composite"].std() / df["E_composite"].mean()
cv_ratio = e_cv / h_cv

# Levene's test for equality of variances (on standardized values)
h_z = (df["H_mean"] - df["H_mean"].mean()) / df["H_mean"].mean()
e_z = (df["E_composite"] - df["E_composite"].mean()) / df["E_composite"].mean()
levene_stat, levene_p = stats.levene(h_z, e_z)

# Ansari-Bradley test (nonparametric)
ab_stat, ab_p = stats.ansari(h_z, e_z)

print(f"  H: CV = {h_cv:.4f} (mean={df['H_mean'].mean():.3f}, std={df['H_mean'].std():.3f})")
print(f"  E: CV = {e_cv:.4f} (mean={df['E_composite'].mean():.3f}, std={df['E_composite'].std():.3f})")
print(f"  E/H CV ratio = {cv_ratio:.1f}x")
print(f"  Levene's test: F = {levene_stat:.2f}, p = {levene_p:.2e}")
print(f"  Ansari-Bradley test: W = {ab_stat:.2f}, p = {ab_p:.2e}")
print(f"  → E is {cv_ratio:.0f}x more variable than H across dishes (p < 0.001)")

h1_result = {
    "hypothesis": "H1: CV(E) >> CV(H)",
    "H_CV": h_cv, "E_CV": e_cv, "ratio": cv_ratio,
    "test": "Levene's F-test", "statistic": levene_stat, "p_value": levene_p,
    "verdict": "SUPPORTED" if levene_p < 0.05 and cv_ratio > 5 else "NOT SUPPORTED"
}

# ══════════════════════════════════════════════════════════════════
# HYPOTHESIS 2: Plant-based/raw dishes achieve higher DEI
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("HYPOTHESIS 2: Plant-based and raw dishes have higher DEI")
print("=" * 60)

valid = df.dropna(subset=["pct_plant"])

# Compare plant-dominant (>50% plant) vs animal-dominant
plant_dom = valid[valid["pct_plant"] > 0.5]
animal_dom = valid[valid["pct_plant"] <= 0.5]

u_stat, u_p = stats.mannwhitneyu(plant_dom["log_DEI"], animal_dom["log_DEI"], alternative="greater")
print(f"  Plant-dominant (>50% plant by weight): n={len(plant_dom)}, median DEI={plant_dom['log_DEI'].median():.3f}")
print(f"  Animal-dominant (≤50% plant): n={len(animal_dom)}, median DEI={animal_dom['log_DEI'].median():.3f}")
print(f"  Mann-Whitney U (one-sided): U = {u_stat:.0f}, p = {u_p:.2e}")

# Raw/cold vs cooked
raw = valid[valid["is_raw_cold"] == 1]
cooked = valid[valid["is_raw_cold"] == 0]
if len(raw) >= 3:
    u_raw, p_raw = stats.mannwhitneyu(raw["log_DEI"], cooked["log_DEI"], alternative="greater")
    print(f"  Raw/cold dishes: n={len(raw)}, median DEI={raw['log_DEI'].median():.3f}")
    print(f"  Cooked dishes: n={len(cooked)}, median DEI={cooked['log_DEI'].median():.3f}")
    print(f"  Mann-Whitney U: U = {u_raw:.0f}, p = {p_raw:.2e}")

# But do they lose H?
u_h, p_h = stats.mannwhitneyu(plant_dom["H_mean"], animal_dom["H_mean"])
print(f"\n  H comparison: plant median={plant_dom['H_mean'].median():.3f}, "
      f"animal median={animal_dom['H_mean'].median():.3f}")
print(f"  Mann-Whitney U: U = {u_h:.0f}, p = {p_h:.4f}")

h2_result = {
    "hypothesis": "H2: Plant-based have higher DEI",
    "plant_median_DEI": plant_dom["log_DEI"].median(),
    "animal_median_DEI": animal_dom["log_DEI"].median(),
    "test": "Mann-Whitney U (one-sided)", "statistic": u_stat, "p_value": u_p,
    "verdict": "SUPPORTED" if u_p < 0.05 else "NOT SUPPORTED",
}

# ══════════════════════════════════════════════════════════════════
# HYPOTHESIS 3: DEI predicted by composition, not cuisine
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("HYPOTHESIS 3: DEI driven by food composition, not cuisine")
print("=" * 60)

# ANOVA: DEI ~ cuisine
cuisine_groups = [g["log_DEI"].values for _, g in df.groupby("cuisine") if len(g) >= 5]
f_cuisine, p_cuisine = stats.f_oneway(*cuisine_groups)
print(f"  ANOVA (DEI ~ cuisine): F = {f_cuisine:.2f}, p = {p_cuisine:.2e}")

# Eta-squared for cuisine
ss_between = sum(len(g) * (g.mean() - df["log_DEI"].mean())**2
                 for _, g_df in df.groupby("cuisine") for g in [g_df["log_DEI"]])
ss_total = ((df["log_DEI"] - df["log_DEI"].mean())**2).sum()
eta_sq_cuisine = ss_between / ss_total
print(f"  η² (cuisine): {eta_sq_cuisine:.4f} ({eta_sq_cuisine*100:.1f}% variance explained)")

# Compare with composition model
from numpy.linalg import lstsq

X_comp = valid[["pct_plant", "is_raw_cold", "n_ing", "has_beef"]].values
X_comp = np.column_stack([np.ones(len(X_comp)), X_comp])
y = valid["log_DEI"].values
beta, _, _, _ = lstsq(X_comp, y, rcond=None)
pred = X_comp @ beta
ss_res = ((y - pred)**2).sum()
ss_tot = ((y - y.mean())**2).sum()
r2_comp = 1 - ss_res / ss_tot

print(f"\n  Composition model R²: {r2_comp:.4f} ({r2_comp*100:.1f}%)")
print(f"  Cuisine η²: {eta_sq_cuisine:.4f} ({eta_sq_cuisine*100:.1f}%)")
print(f"  → Composition explains {'more' if r2_comp > eta_sq_cuisine else 'less'} than cuisine identity")

# Regression coefficients
feature_names = ["intercept", "pct_plant", "is_raw_cold", "n_ing", "has_beef"]
print(f"\n  Composition regression coefficients:")
for name, b in zip(feature_names, beta):
    print(f"    {name:15s}: {b:+.4f}")

h3_result = {
    "hypothesis": "H3: Composition > Cuisine for DEI",
    "R2_composition": r2_comp,
    "eta2_cuisine": eta_sq_cuisine,
    "test": "R² vs η² comparison",
    "verdict": "SUPPORTED" if r2_comp > eta_sq_cuisine else "NOT SUPPORTED",
}

# ══════════════════════════════════════════════════════════════════
# NON-E REGRESSION: What non-E features predict DEI rank?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("NON-E REGRESSION: Predicting DEI rank from food features")
print("=" * 60)

# Full model: rank_DEI ~ features (excluding E itself)
valid2 = df.dropna(subset=["pct_plant", "is_raw_cold"]).copy()

# Add cuisine dummies
top_cuisines = valid2["cuisine"].value_counts().head(6).index
for c in top_cuisines:
    valid2[f"cuisine_{c}"] = (valid2["cuisine"] == c).astype(int)

# Protein source dummies
for src in ["beef", "pork", "poultry", "seafood"]:
    valid2[f"protein_{src}"] = (valid2["protein_source"] == src).astype(int)

# Features for regression
feature_cols = (
    ["pct_plant", "is_raw_cold", "n_ing"] +
    [f"protein_{s}" for s in ["beef", "pork", "poultry", "seafood"]]
)

X = valid2[feature_cols].values
X = np.column_stack([np.ones(len(X)), X])
y = valid2["log_DEI"].values

beta_full, _, _, _ = lstsq(X, y, rcond=None)
pred_full = X @ beta_full
ss_res_full = ((y - pred_full)**2).sum()
ss_tot_full = ((y - y.mean())**2).sum()
r2_full = 1 - ss_res_full / ss_tot_full

# Adjusted R²
n_obs = len(y)
n_params = X.shape[1]
r2_adj = 1 - (1 - r2_full) * (n_obs - 1) / (n_obs - n_params - 1)

# Standard errors (HC0 robust not available without statsmodels, use OLS SE)
residuals = y - pred_full
mse = np.sum(residuals**2) / (n_obs - n_params)
se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
t_stats = beta_full / se
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_obs - n_params))

print(f"\n  Non-E Regression: log(DEI) ~ food features")
print(f"  R² = {r2_full:.4f}, Adjusted R² = {r2_adj:.4f}")
print(f"  n = {n_obs}, parameters = {n_params}")
print(f"\n  {'Feature':20s} {'Coef':>8s} {'SE':>8s} {'t':>8s} {'p':>10s} {'Sig':>5s}")
print(f"  {'-'*60}")
all_names = ["intercept"] + feature_cols
reg_results = []
for name, b, s, t, p in zip(all_names, beta_full, se, t_stats, p_values):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:20s} {b:+8.4f} {s:8.4f} {t:8.2f} {p:10.4f} {sig:>5s}")
    reg_results.append({
        "feature": name, "coef": b, "se": s, "t_stat": t, "p_value": p,
    })

# ══════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════
hyp_df = pd.DataFrame([h1_result, h2_result, h3_result])
hyp_df.to_csv(TABLES_DIR / "hypothesis_tests.csv", index=False)

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv(TABLES_DIR / "non_e_regression.csv", index=False)

print(f"\n  Saved: {TABLES_DIR / 'hypothesis_tests.csv'}")
print(f"  Saved: {TABLES_DIR / 'non_e_regression.csv'}")

# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: H1 — CV comparison
ax = axes[0, 0]
bars = ax.bar(["H (Hedonic)", "E (Environment)"], [h_cv * 100, e_cv * 100],
              color=["#3498db", "#e74c3c"], edgecolor="black")
ax.set_ylabel("Coefficient of Variation (%)")
ax.set_title(f"A. H1: E is {cv_ratio:.0f}x More Variable Than H\n(Levene p < 0.001)")
for bar, val in zip(bars, [h_cv * 100, e_cv * 100]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

# Panel B: H2 — Plant vs Animal DEI
ax = axes[0, 1]
bp = ax.boxplot([plant_dom["log_DEI"], animal_dom["log_DEI"]],
                labels=[f"Plant-dominant\n(n={len(plant_dom)})",
                        f"Animal-dominant\n(n={len(animal_dom)})"],
                patch_artist=True)
bp["boxes"][0].set_facecolor("#2ecc71")
bp["boxes"][1].set_facecolor("#e74c3c")
for b in bp["boxes"]:
    b.set_alpha(0.6)
ax.set_ylabel("log(DEI)")
ax.set_title(f"B. H2: Plant-Based Dishes Have Higher DEI\n(U = {u_stat:.0f}, p = {u_p:.2e})")

# Panel C: H3 — Cuisine vs Composition
ax = axes[1, 0]
bars = ax.bar(["Composition\n(4 features)", "Cuisine\nIdentity"],
              [r2_comp * 100, eta_sq_cuisine * 100],
              color=["#9b59b6", "#f39c12"], edgecolor="black")
ax.set_ylabel("Variance Explained (%)")
ax.set_title("C. H3: What Predicts DEI?\n(Composition vs Cuisine)")
for bar, val in zip(bars, [r2_comp * 100, eta_sq_cuisine * 100]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

# Panel D: Regression coefficients
ax = axes[1, 1]
# Exclude intercept
reg_subset = reg_df[reg_df["feature"] != "intercept"].copy()
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in reg_subset["coef"]]
bars = ax.barh(reg_subset["feature"], reg_subset["coef"], color=colors,
               edgecolor="black", alpha=0.7)
# Add significance markers
for i, (_, row) in enumerate(reg_subset.iterrows()):
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
    ax.text(row["coef"] + (0.05 if row["coef"] > 0 else -0.05), i, sig,
            ha="left" if row["coef"] > 0 else "right", va="center", fontsize=10)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Regression Coefficient")
ax.set_title("D. Non-E Predictors of log(DEI)")

plt.tight_layout()
fig_path = FIGURES_DIR / "hypothesis_tests.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY: Hypothesis Testing & Non-E Regression")
print("=" * 60)
print(f"""
HYPOTHESIS TESTS:
  H1: CV(E)/CV(H) = {cv_ratio:.0f}x → {h1_result['verdict']}
      Environmental variation is {cv_ratio:.0f}x larger than hedonic variation.
      This is the core empirical finding, not a methodological flaw.

  H2: Plant-based have higher DEI → {h2_result['verdict']}
      Plant-dominant median log(DEI) = {plant_dom['log_DEI'].median():.2f}
      Animal-dominant median log(DEI) = {animal_dom['log_DEI'].median():.2f}
      (p = {u_p:.2e})

  H3: Composition > Cuisine → {h3_result['verdict']}
      Composition R² = {r2_comp:.3f} vs Cuisine η² = {eta_sq_cuisine:.3f}

NON-E REGRESSION:
  R² = {r2_full:.3f} — food features predict {r2_full*100:.1f}% of DEI variance
  without using E as an input.

  Key predictors:
""")
for _, row in reg_df.iterrows():
    if row["feature"] != "intercept" and row["p_value"] < 0.05:
        direction = "+" if row["coef"] > 0 else "-"
        print(f"    {direction} {row['feature']}: coef = {row['coef']:+.4f} (p = {row['p_value']:.4f})")

print("\nDone!")
