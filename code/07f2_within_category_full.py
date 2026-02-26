"""
07f2_within_category_full.py — Within-Category Substitution on ALL 334 dishes
=============================================================================
Extends 07f to cover both original (158) and expanded (176) dishes.
Outputs correct substitution counts for paper.
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Load combined 334-dish dataset ──────────────────────────────
dei = pd.read_csv(DATA_DIR / "combined_dish_DEI.csv")
print(f"Loaded {len(dei)} dishes ({dei['source'].value_counts().to_dict()})")

# ══════════════════════════════════════════════════════════════════
# FULL functional category mapping for ALL 334 dishes
# ══════════════════════════════════════════════════════════════════

FUNCTIONAL_CATEGORIES = {
    "Protein Main": [
        # Original
        "steak", "hamburger", "fried_chicken", "buffalo_wings", "pulled_pork",
        "ribs", "brisket", "korean_fried_chicken", "teriyaki", "katsu",
        "general_tso", "orange_chicken", "kung_pao_chicken", "sweet_and_sour",
        "peking_duck", "bulgogi", "samgyeopsal", "char_siu", "kebab",
        "gyro", "shawarma", "tandoori", "butter_chicken", "vindaloo",
        "korma", "lamb_chops", "osso_buco", "beef_bourguignon", "coq_au_vin",
        "fish_and_chips", "tempura", "lobster_roll", "crab_cake", "fish_taco",
        "ceviche", "sashimi", "falafel",
        # Expanded
        "ackee_saltfish", "adobo", "aji_de_gallina", "al_pastor", "anticucho",
        "asado", "bacalhau", "beef_tartare", "biltong", "birria",
        "bossam", "bratwurst", "carnitas", "churrasco", "conch_fritters",
        "croquetas", "currywurst", "dakgalbi", "doner_kebab", "doro_wot",
        "duck_confit", "escargot", "escovitch_fish",
        "fesenjan", "gambas_al_ajillo", "ghormeh_sabzi", "gyudon",
        "iskender_kebab", "jerk_chicken", "jokbal",
        "karaage", "kare_kare", "katsu_curry", "keftedes",
        "khao_man_gai", "kibbeh", "kitfo", "kofte", "koobideh",
        "lomo_saltado", "malai_kofta", "meatloaf",
        "oxtail_stew", "pernil", "picanha", "piri_piri",
        "pot_roast", "rendang",
        "salt_pepper_squid", "sardines_grilled", "satay", "satay_indonesian",
        "schnitzel", "shashlik", "sisig", "souvlaki",
        "stroganoff", "suya", "tagine", "tandoori_chicken", "tikka_masala",
        "yakitori",
        # Previously uncategorized
        "aloo_gobi", "bun_cha", "eggplant_parm", "eggs_benedict",
        "fish_tacos", "halloumi", "paneer_tikka",
        "quiche", "raclette", "saganaki", "tamagoyaki",
    ],
    "Noodle/Rice Dish": [
        # Original
        "fried_rice", "lo_mein", "dan_dan_noodles", "chow_mein", "chow_fun",
        "pad_thai", "pad_see_ew", "japchae", "bibimbap", "donburi",
        "ramen", "pho", "bun_bo_hue", "laksa", "khao_soi",
        "pasta_bolognese", "pasta_carbonara", "lasagna", "mac_and_cheese",
        "risotto", "paella", "biryani",
        # Expanded
        "arroz_con_pollo", "boat_noodles", "char_kway_teow", "com_tam",
        "hu_tieu", "jjajangmyeon", "jollof_rice", "lagman",
        "mee_goreng", "nasi_goreng", "nasi_lemak", "nasi_padang",
        "okonomiyaki", "pastitsio", "penne_arrabbiata",
        "plov", "ravioli", "gnocchi",
        "rice_and_peas", "tonkotsu_ramen", "udon", "wonton_noodle",
        # Previously uncategorized
        "mapo_tofu", "tteokbokki", "spaetzle", "polenta", "couscous",
        "tahdig", "dal",
    ],
    "Wrapped/Stuffed": [
        # Original
        "dumplings", "xiao_long_bao", "spring_rolls", "sushi", "maki_roll",
        "taco", "burrito", "quesadilla", "enchilada", "chile_relleno",
        "empanada", "samosa", "banh_mi", "gyoza", "wonton",
        "crepe", "fajita", "nachos",
        # Expanded
        "arepa", "banh_xeo", "borek", "char_siu_bao",
        "cubano_sandwich", "dolma", "galette",
        "gozleme", "kimbap", "lahmacun", "lumpia",
        "manti_turkish", "manty", "onigiri", "pastelitos",
        "pide", "pierogi", "pupusa", "sfeeha", "sopes",
        "tamale", "warak_enab",
        # Sandwich-type
        "chicken_sandwich", "club_sandwich", "grilled_cheese", "hot_dog",
        # Previously uncategorized
        "arancini", "chole_bhature", "coxinha", "dim_sum", "dim_sum_items",
        "fresh_spring_roll", "mofongo", "pajeon",
        "spanakopita", "takoyaki", "tiropita", "tostones",
    ],
    "Soup/Stew": [
        # Original
        "miso_soup", "hot_and_sour_soup", "tom_yum", "tom_kha",
        "french_onion_soup", "clam_chowder", "pozole", "gumbo",
        "massaman_curry", "green_curry", "red_curry", "kimchi_jjigae",
        # Expanded
        "bak_kut_teh", "borscht", "bouillabaisse", "budae_jjigae",
        "bun_rieu", "caldo_verde", "callaloo", "chana_masala", "chili",
        "congee", "egusi_soup", "feijoada", "fondue",
        "minestrone", "mole", "moussaka", "mujaddara",
        "paneer_butter", "sancocho", "shakshuka", "sinigang",
        "sundubu_jjigae", "wonton_soup",
        # Previously uncategorized
        "mafe", "ful_medames", "palak_paneer", "soup", "poutine",
    ],
    "Salad/Cold/Side": [
        # Original
        "coleslaw", "tabbouleh", "fattoush", "greek_salad", "caesar_salad",
        "caprese", "bruschetta", "edamame", "kimchi", "pickled_vegetables",
        "baba_ganoush", "hummus", "raita", "papaya_salad", "larb",
        "elote", "gazpacho",
        # Expanded
        "causa", "chicken_salad", "chimichurri", "cobb_salad",
        "french_fries", "fufu", "gado_gado", "guacamole",
        "harissa", "horiatiki", "labneh", "muhammara",
        "nicoise_salad", "patatas_bravas", "prosciutto",
        "rojak", "som_tam", "tzatziki",
        "pav_bhaji", "ratatouille",
        # Previously uncategorized
        "plantain", "tapas",
    ],
    "Bread/Pastry": [
        # Original
        "naan", "pita", "focaccia", "croissant", "pancake",
        "waffle", "french_toast",
        # Expanded
        "blini", "chilaquiles", "cornbread", "dosa",
        "idli", "injera", "manakish", "pao_de_queijo",
        "pretzel", "scallion_pancake", "simit",
        "uttapam", "tortilla_espanola",
        # Previously uncategorized
        "pizza", "pizza_margherita", "roti_caribbean",
    ],
    "Dessert": [
        # Original
        "baklava", "churro", "churros_spanish", "brownie", "cheesecake",
        "tiramisu", "creme_brulee", "souffle", "panna_cotta", "gelato",
        "mochi", "mango_sticky_rice",
        # Expanded
        "cannoli", "churros_mexican", "dan_tat", "eclair",
        "galaktoboureko", "gulab_jamun", "ice_cream", "jalebi",
        "kunefe", "loukoumades", "macarons",
        "pasteis_de_nata", "profiterole",
        # Previously uncategorized
        "halo_halo",
    ],
    "Beverage": [
        # Original
        "thai_iced_tea", "ca_phe_sua_da", "horchata", "lassi", "chai",
        "matcha",
        # Expanded
        "coquito",
    ],
}

# ── Assign categories ───────────────────────────────────────────
dish_to_cat = {}
for cat, dishes in FUNCTIONAL_CATEGORIES.items():
    for d in dishes:
        dish_to_cat[d] = cat

dei["functional_category"] = dei["dish_id"].map(dish_to_cat)

categorized = dei.dropna(subset=["functional_category"])
uncategorized = dei[dei["functional_category"].isna()]

print(f"\nCategorized: {len(categorized)}/{len(dei)} dishes")
if len(uncategorized) > 0:
    print(f"UNCATEGORIZED ({len(uncategorized)}): {uncategorized['dish_id'].tolist()}")

print(f"\nFunctional categories:")
for cat in sorted(dei["functional_category"].dropna().unique()):
    n = (dei["functional_category"] == cat).sum()
    print(f"  {cat:20s}: {n:3d} dishes")

# ══════════════════════════════════════════════════════════════════
# Pareto front helper
# ══════════════════════════════════════════════════════════════════
def find_pareto(df, h_col="H_mean", e_col="E_composite"):
    pareto = []
    sorted_df = df.sort_values(e_col)
    max_h = -np.inf
    for _, row in sorted_df.iterrows():
        if row[h_col] > max_h:
            pareto.append(row["dish_id"])
            max_h = row[h_col]
    return pareto

# ══════════════════════════════════════════════════════════════════
# Substitution analysis
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUBSTITUTION ANALYSIS (all 334 dishes)")
print("=" * 60)

substitutions = []
for cat, group in categorized.groupby("functional_category"):
    if len(group) < 3:
        continue
    for _, row_from in group.iterrows():
        for _, row_to in group.iterrows():
            if row_from["dish_id"] == row_to["dish_id"]:
                continue
            e_ratio = row_from["E_composite"] / row_to["E_composite"] if row_to["E_composite"] > 0 else np.nan
            h_change = row_to["H_mean"] - row_from["H_mean"]
            e_pct_reduction = (1 - row_to["E_composite"] / row_from["E_composite"]) * 100

            # Constraints: E drops >30%, H loss <1.0
            if e_ratio > 1.3 and h_change > -1.0:
                substitutions.append({
                    "category": cat,
                    "from_dish": row_from["dish_id"],
                    "to_dish": row_to["dish_id"],
                    "from_E": row_from["E_composite"],
                    "to_E": row_to["E_composite"],
                    "E_reduction_pct": e_pct_reduction,
                    "E_ratio": e_ratio,
                    "from_H": row_from["H_mean"],
                    "to_H": row_to["H_mean"],
                    "H_change": h_change,
                    "from_cuisine": row_from["cuisine"],
                    "to_cuisine": row_to["cuisine"],
                })

sub_df = pd.DataFrame(substitutions)
sub_df["sub_score"] = sub_df["E_ratio"] * (1 + sub_df["H_change"])

print(f"\n  Total viable substitutions: {len(sub_df)}")
print(f"\n  Per category:")
cat_counts = Counter(s["category"] for s in substitutions)
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    n_dishes = (categorized["functional_category"] == cat).sum()
    print(f"    {cat:20s}: {cnt:5d} swaps  ({n_dishes} dishes)")

# Best substitution per source dish
best_subs = sub_df.sort_values("sub_score", ascending=False).groupby("from_dish").first().reset_index()

print(f"\n  Top 10 best substitutions:")
for _, row in best_subs.nlargest(10, "sub_score").iterrows():
    print(f"    [{row['category']:15s}] {row['from_dish']:20s} → {row['to_dish']:20s}  "
          f"E: -{row['E_reduction_pct']:.0f}%  H: {row['H_change']:+.2f}")

# ── Save ────────────────────────────────────────────────────────
sub_df.to_csv(TABLES_DIR / "substitution_network_full.csv", index=False)
print(f"\n  Saved: {TABLES_DIR / 'substitution_network_full.csv'}")

# ══════════════════════════════════════════════════════════════════
# Regenerate substitution network figure
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating substitution network figure...")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 10))

top_subs = best_subs.nlargest(30, "sub_score")

all_dishes_in_subs = set(top_subs["from_dish"]) | set(top_subs["to_dish"])
pos = {}
for d in all_dishes_in_subs:
    row = dei[dei["dish_id"] == d]
    if len(row) > 0:
        pos[d] = (row["E_composite"].iloc[0], row["H_mean"].iloc[0])

for _, sub in top_subs.iterrows():
    if sub["from_dish"] in pos and sub["to_dish"] in pos:
        x1, y1 = pos[sub["from_dish"]]
        x2, y2 = pos[sub["to_dish"]]
        alpha = min(0.8, sub["E_ratio"] / 10)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="steelblue",
                                    alpha=alpha, lw=1.5))

for d, (x, y) in pos.items():
    is_source = d in set(top_subs["from_dish"])
    is_target = d in set(top_subs["to_dish"])
    if is_target and not is_source:
        color = "#2ecc71"
        s = 80
    elif is_source and not is_target:
        color = "#e74c3c"
        s = 60
    else:
        color = "#f1c40f"
        s = 60
    ax.scatter(x, y, c=color, s=s, edgecolors="k", linewidths=0.5, zorder=5)
    ax.annotate(d.replace("_", " "), (x, y), fontsize=6, ha="left",
                xytext=(4, 4), textcoords="offset points")

ax.set_xlabel("E (Environmental Cost)", fontsize=12)
ax.set_ylabel("H (Hedonic Score)", fontsize=12)
ax.set_title("Substitution Network (334 dishes)\n"
             "Arrows: high-E → low-E within same category, <1 pt hedonic loss",
             fontsize=13)
if dei["E_composite"].max() / dei["E_composite"].min() > 10:
    ax.set_xscale("log")

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="Source (high E)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8, label="Target (low E)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#f1c40f", markersize=8, label="Both"),
    Line2D([0], [0], color="steelblue", lw=2, label="Substitution arrow"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

fig_path = FIGURES_DIR / "substitution_network.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"  Saved: {fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════
# Also check specific claims in paper
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("VERIFYING PAPER CLAIMS")
print("=" * 60)

# brisket -> ceviche
bc = sub_df[(sub_df["from_dish"] == "brisket") & (sub_df["to_dish"] == "ceviche")]
if len(bc) > 0:
    row = bc.iloc[0]
    print(f"  brisket → ceviche: E reduction {row['E_reduction_pct']:.0f}%, H change {row['H_change']:+.2f}")
else:
    print("  brisket → ceviche: NOT FOUND (different categories?)")

# baba_ganoush -> rojak
br = sub_df[(sub_df["from_dish"] == "baba_ganoush") & (sub_df["to_dish"] == "rojak")]
if len(br) > 0:
    row = br.iloc[0]
    print(f"  baba_ganoush → rojak: E reduction {row['E_reduction_pct']:.0f}%, H change {row['H_change']:+.2f}")
else:
    print("  baba_ganoush → rojak: NOT FOUND")

# oxtail_stew -> ceviche
oc = sub_df[(sub_df["from_dish"] == "oxtail_stew") & (sub_df["to_dish"] == "ceviche")]
if len(oc) > 0:
    row = oc.iloc[0]
    print(f"  oxtail_stew → ceviche: E reduction {row['E_reduction_pct']:.0f}%, H change {row['H_change']:+.2f}")
else:
    print("  oxtail_stew → ceviche: NOT FOUND (different categories?)")

# Number of categories used
cats_used = categorized["functional_category"].nunique()
print(f"\n  Categories used: {cats_used}")

print("\n" + "=" * 60)
print(f"FINAL: {len(sub_df)} viable substitutions across {len(categorized)} categorized dishes")
print("=" * 60)
