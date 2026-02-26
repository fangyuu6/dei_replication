"""
07c_cross_validate_h.py
Cross-platform hedonic score validation

Matches 158 DEI dishes to external data sources and computes
cross-platform H correlation matrix to validate that Yelp-derived
H scores reflect genuine dish-level hedonic quality, not platform artifacts.

External sources:
  1. Food.com (recipes.parquet + reviews.parquet) — recipe ratings
  2. Google Local Reviews — NLP sentiment from restaurant reviews
  3. TripAdvisor (6 cities) — NLP sentiment from restaurant reviews
  4. CROCUFID — lab-based cross-cultural desire-to-eat ratings
  5. Food-Pics Extended — lab-based palatability ratings
  6. ASAP/Dianping — Chinese restaurant review sentiment (dish_taste)
"""

import sys, re, json, warnings
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXT = ROOT / "raw" / "external"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load DEI dishes with Yelp H scores ────────────────────────────
dei = pd.read_csv(DATA / "dish_DEI_scores.csv")
dei = dei.set_index("dish_id")
dish_names = list(dei.index)
print(f"Loaded {len(dish_names)} DEI dishes")

# ── Dish name matching utilities ──────────────────────────────────
# Build a mapping from various name forms to canonical dish_id
DISH_ALIASES = {}
for d in dish_names:
    canonical = d.lower().replace("_", " ")
    DISH_ALIASES[canonical] = d
    DISH_ALIASES[d] = d
    # Without underscores
    DISH_ALIASES[d.replace("_", " ")] = d
    # Singular/plural
    if canonical.endswith("s") and len(canonical) > 4:
        DISH_ALIASES[canonical[:-1]] = d
    if not canonical.endswith("s"):
        DISH_ALIASES[canonical + "s"] = d

# Additional manual aliases for common variations
EXTRA_ALIASES = {
    "pad thai": "pad_thai", "pad see ew": "pad_see_ew",
    "tom yum": "tom_yum", "tom kha": "tom_kha",
    "pho bo": "pho", "pho ga": "pho",
    "bahn mi": "banh_mi", "banh mi sandwich": "banh_mi",
    "macaroni and cheese": "mac_and_cheese", "mac & cheese": "mac_and_cheese",
    "mac n cheese": "mac_and_cheese",
    "grilled cheese sandwich": "grilled_cheese",
    "fried chicken sandwich": "chicken_sandwich",
    "general tso chicken": "general_tso", "general tso's chicken": "general_tso",
    "general tso's": "general_tso",
    "kung pao": "kung_pao_chicken", "kung pao chicken": "kung_pao_chicken",
    "orange chicken": "orange_chicken",
    "sweet and sour chicken": "sweet_and_sour", "sweet & sour": "sweet_and_sour",
    "chicken tikka masala": "tikka_masala", "tikka masala": "tikka_masala",
    "butter chicken": "butter_chicken",
    "tandoori": "tandoori_chicken", "tandoori chicken": "tandoori_chicken",
    "palak paneer": "palak_paneer", "saag paneer": "palak_paneer",
    "chana masala": "chana_masala", "chickpea curry": "chana_masala",
    "paneer tikka": "paneer_tikka",
    "dal": "dal", "daal": "dal", "dhal": "dal", "lentil soup": "dal",
    "dosa": "dosa", "masala dosa": "dosa",
    "naan bread": "naan", "naan": "naan",
    "samosa": "samosa", "samosas": "samosa",
    "biryani": "biryani", "chicken biryani": "biryani", "lamb biryani": "biryani",
    "korma": "korma", "chicken korma": "korma",
    "vindaloo": "vindaloo", "chicken vindaloo": "vindaloo",
    "raita": "raita", "cucumber raita": "raita",
    "hamburger": "hamburger", "burger": "hamburger", "cheeseburger": "hamburger",
    "steak": "steak", "ribeye": "steak", "filet mignon": "steak",
    "ribs": "ribs", "baby back ribs": "ribs", "spare ribs": "ribs",
    "bbq ribs": "ribs", "barbecue ribs": "ribs",
    "brisket": "brisket", "beef brisket": "brisket", "smoked brisket": "brisket",
    "pulled pork": "pulled_pork", "pulled pork sandwich": "pulled_pork",
    "fried chicken": "fried_chicken",
    "buffalo wings": "buffalo_wings", "chicken wings": "buffalo_wings",
    "wings": "buffalo_wings",
    "lobster roll": "lobster_roll",
    "fish and chips": "fish_and_chips", "fish & chips": "fish_and_chips",
    "french fries": "french_fries", "fries": "french_fries",
    "clam chowder": "clam_chowder", "new england clam chowder": "clam_chowder",
    "french onion soup": "french_onion_soup", "onion soup": "french_onion_soup",
    "caesar salad": "caesar_salad", "caesar": "caesar_salad",
    "chicken salad": "chicken_salad",
    "coleslaw": "coleslaw", "cole slaw": "coleslaw",
    "corn on the cob": "elote", "elote": "elote", "mexican corn": "elote",
    "nachos": "nachos",
    "guacamole": "guacamole", "guac": "guacamole",
    "burrito": "burrito", "burritos": "burrito",
    "taco": "taco", "tacos": "taco",
    "quesadilla": "quesadilla", "quesadillas": "quesadilla",
    "enchilada": "enchilada", "enchiladas": "enchilada",
    "fajita": "fajita", "fajitas": "fajita",
    "tamale": "tamale", "tamales": "tamale",
    "chile relleno": "chile_relleno", "chiles rellenos": "chile_relleno",
    "churro": "churro", "churros": "churro",
    "pozole": "pozole", "posole": "pozole",
    "pizza": "pizza", "pepperoni pizza": "pizza",
    "margherita pizza": "pizza_margherita", "margherita": "pizza_margherita",
    "lasagna": "lasagna", "lasagne": "lasagna",
    "pasta bolognese": "pasta_bolognese", "bolognese": "pasta_bolognese",
    "spaghetti bolognese": "pasta_bolognese",
    "carbonara": "pasta_carbonara", "spaghetti carbonara": "pasta_carbonara",
    "pasta carbonara": "pasta_carbonara",
    "penne arrabbiata": "penne_arrabbiata", "arrabbiata": "penne_arrabbiata",
    "gnocchi": "gnocchi",
    "ravioli": "ravioli",
    "risotto": "risotto",
    "tiramisu": "tiramisu",
    "gelato": "gelato",
    "panna cotta": "panna_cotta", "pannacotta": "panna_cotta",
    "bruschetta": "bruschetta",
    "caprese": "caprese", "caprese salad": "caprese",
    "minestrone": "minestrone", "minestrone soup": "minestrone",
    "osso buco": "osso_buco", "ossobuco": "osso_buco",
    "sushi": "sushi", "sushi roll": "sushi",
    "sashimi": "sashimi",
    "ramen": "ramen",
    "udon": "udon", "udon noodles": "udon",
    "tempura": "tempura",
    "teriyaki": "teriyaki", "teriyaki chicken": "teriyaki",
    "miso soup": "miso_soup", "miso": "miso_soup",
    "edamame": "edamame",
    "takoyaki": "takoyaki",
    "onigiri": "onigiri", "rice ball": "onigiri",
    "katsu": "katsu", "tonkatsu": "katsu", "chicken katsu": "katsu",
    "donburi": "donburi", "don": "donburi",
    "okonomiyaki": "okonomiyaki",
    "yakitori": "yakitori",
    "kimchi": "kimchi",
    "bibimbap": "bibimbap",
    "bulgogi": "bulgogi",
    "japchae": "japchae",
    "kimbap": "kimbap", "gimbap": "kimbap",
    "tteokbokki": "tteokbokki", "tteok-bokki": "tteokbokki",
    "korean fried chicken": "korean_fried_chicken",
    "kimchi jjigae": "kimchi_jjigae", "kimchi stew": "kimchi_jjigae",
    "sundubu jjigae": "sundubu_jjigae", "sundubu": "sundubu_jjigae",
    "samgyeopsal": "samgyeopsal",
    "green curry": "green_curry", "thai green curry": "green_curry",
    "red curry": "red_curry", "thai red curry": "red_curry",
    "massaman curry": "massaman_curry", "massaman": "massaman_curry",
    "papaya salad": "papaya_salad", "som tam": "papaya_salad",
    "thai iced tea": "thai_iced_tea",
    "mango sticky rice": "mango_sticky_rice",
    "satay": "satay", "chicken satay": "satay",
    "larb": "larb", "laab": "larb",
    "spring roll": "spring_rolls", "spring rolls": "spring_rolls",
    "egg roll": "spring_rolls",
    "fresh spring roll": "fresh_spring_roll",
    "summer roll": "fresh_spring_roll",
    "pho": "pho",
    "bun bo hue": "bun_bo_hue",
    "bun cha": "bun_cha",
    "com tam": "com_tam", "broken rice": "com_tam",
    "ca phe sua da": "ca_phe_sua_da", "vietnamese coffee": "ca_phe_sua_da",
    "vietnamese iced coffee": "ca_phe_sua_da",
    "peking duck": "peking_duck", "beijing duck": "peking_duck",
    "dim sum": "dim_sum",
    "dumplings": "dumplings", "pot stickers": "dumplings",
    "potstickers": "dumplings", "gyoza": "dumplings",
    "wonton soup": "wonton_soup", "wonton": "wonton_soup",
    "xiao long bao": "xiao_long_bao", "soup dumplings": "xiao_long_bao",
    "xlb": "xiao_long_bao",
    "hot and sour soup": "hot_and_sour_soup", "hot & sour soup": "hot_and_sour_soup",
    "mapo tofu": "mapo_tofu",
    "kung pao chicken": "kung_pao_chicken",
    "dan dan noodles": "dan_dan_noodles",
    "chow mein": "chow_mein",
    "chow fun": "chow_fun",
    "lo mein": "lo_mein",
    "fried rice": "fried_rice",
    "scallion pancake": "scallion_pancake",
    "char siu": "char_siu", "bbq pork": "char_siu",
    "congee": "congee", "rice porridge": "congee",
    "coq au vin": "coq_au_vin",
    "escargot": "escargot", "escargots": "escargot",
    "crepe": "crepe", "crepes": "crepe",
    "croissant": "croissant",
    "quiche": "quiche",
    "souffle": "souffle", "soufflé": "souffle",
    "ratatouille": "ratatouille",
    "creme brulee": "creme_brulee", "crème brûlée": "creme_brulee",
    "paella": "paella",
    "patatas bravas": "patatas_bravas",
    "gazpacho": "gazpacho",
    "tapas": "tapas",
    "spanakopita": "spanakopita",
    "falafel": "falafel",
    "hummus": "hummus",
    "gyro": "gyro", "gyros": "gyro",
    "shawarma": "shawarma",
    "kebab": "kebab", "kebabs": "kebab", "kabob": "kebab",
    "baba ganoush": "baba_ganoush", "baba ghanoush": "baba_ganoush",
    "tabbouleh": "tabbouleh", "tabouleh": "tabbouleh",
    "dolma": "dolma", "dolmas": "dolma", "stuffed grape leaves": "dolma",
    "moussaka": "moussaka",
    "pita": "pita", "pita bread": "pita",
    "baklava": "baklava",
    "pancake": "pancake", "pancakes": "pancake",
    "waffle": "waffle", "waffles": "waffle",
    "brownie": "brownie", "brownies": "brownie",
    "cheesecake": "cheesecake",
    "cornbread": "cornbread",
    "grilled cheese": "grilled_cheese",
    "club sandwich": "club_sandwich",
    "ice cream": "ice_cream",
    "beef bourguignon": "beef_bourguignon",
}
DISH_ALIASES.update(EXTRA_ALIASES)


def match_dish(text):
    """Try to match text to a canonical dish_id."""
    text_lower = text.lower().strip()
    if text_lower in DISH_ALIASES:
        return DISH_ALIASES[text_lower]
    # Remove common prefixes/suffixes
    for prefix in ["homemade ", "easy ", "best ", "simple ", "classic ",
                    "authentic ", "traditional ", "my "]:
        if text_lower.startswith(prefix):
            rest = text_lower[len(prefix):]
            if rest in DISH_ALIASES:
                return DISH_ALIASES[rest]
    for suffix in [" recipe", " recipes", " dish", " bowl", " plate"]:
        if text_lower.endswith(suffix):
            rest = text_lower[:-len(suffix)]
            if rest in DISH_ALIASES:
                return DISH_ALIASES[rest]
    return None


# Build fast lookup: group aliases by dish_id, keep only unique substrings
_DISH_TO_ALIASES = defaultdict(list)
for alias, did in sorted(DISH_ALIASES.items(), key=lambda x: -len(x[0])):
    if len(alias) >= 4:
        _DISH_TO_ALIASES[did].append(alias)

# Flat list sorted longest-first for scanning
_ALIAS_SCAN = sorted(
    [(a, DISH_ALIASES[a]) for a in DISH_ALIASES if len(a) >= 4],
    key=lambda x: -len(x[0])
)

def search_dish_in_text(text):
    """Search for dish mentions using pure string matching (no regex, fast)."""
    text_lower = text.lower()
    found = set()
    for alias, dish_id in _ALIAS_SCAN:
        if dish_id in found:
            continue
        if alias not in text_lower:
            continue
        # Simple word-boundary check without regex
        idx = text_lower.find(alias)
        while idx != -1:
            before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
            end = idx + len(alias)
            after_ok = (end >= len(text_lower) or not text_lower[end].isalnum())
            if before_ok and after_ok:
                found.add(dish_id)
                break
            idx = text_lower.find(alias, idx + 1)
    return found


# ══════════════════════════════════════════════════════════════════
# SOURCE 1: Food.com — recipe-level ratings
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SOURCE 1: Food.com (522K recipes, 1.4M reviews)")
print("=" * 60)

recipes = pd.read_parquet(EXT / "recipes.parquet")
reviews_fc = pd.read_parquet(EXT / "reviews.parquet")

# Match recipe names to DEI dishes
recipes["dish_id"] = recipes["Name"].apply(lambda x: match_dish(str(x)))
matched_recipes = recipes.dropna(subset=["dish_id"])
print(f"  Matched recipes: {len(matched_recipes)} / {len(recipes)}")
print(f"  Unique dishes matched: {matched_recipes['dish_id'].nunique()}")

# Get reviews for matched recipes
matched_reviews = reviews_fc.merge(
    matched_recipes[["RecipeId", "dish_id"]],
    on="RecipeId", how="inner"
)
print(f"  Reviews for matched dishes: {len(matched_reviews)}")

# Compute dish-level H from Food.com ratings (1-5 scale → 1-10)
h_foodcom = (
    matched_reviews.groupby("dish_id")["Rating"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "H_foodcom_raw", "count": "n_foodcom"})
)
h_foodcom["H_foodcom"] = h_foodcom["H_foodcom_raw"] * 2  # scale to 1-10
h_foodcom = h_foodcom[h_foodcom["n_foodcom"] >= 5]  # min 5 reviews
print(f"  Dishes with ≥5 reviews: {len(h_foodcom)}")
if len(h_foodcom) > 0:
    print(f"  H_foodcom range: [{h_foodcom['H_foodcom'].min():.2f}, {h_foodcom['H_foodcom'].max():.2f}]")

# ══════════════════════════════════════════════════════════════════
# SOURCE 2: CROCUFID — lab-based cross-cultural desire-to-eat
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SOURCE 2: CROCUFID (564 food images, cross-cultural)")
print("=" * 60)

croc = pd.read_excel(
    EXT / "crocufid" / "All_Foodpictures_information.xlsx",
    sheet_name=0
)
# Match food descriptions to DEI dishes
croc["dish_id"] = croc["Description"].apply(lambda x: match_dish(str(x)))
croc_matched = croc.dropna(subset=["dish_id"])
print(f"  Matched items: {len(croc_matched)} / {len(croc)}")
print(f"  Unique dishes: {croc_matched['dish_id'].nunique()}")

# Use Average_Desire_ALL (0-100 VAS) → scale to 1-10
desire_cols = [c for c in croc.columns if "Desire" in c and "Average" in c]
print(f"  Desire columns: {desire_cols}")

if len(croc_matched) > 0:
    h_crocufid = (
        croc_matched.groupby("dish_id")["Average_Desire_ALL"]
        .mean()
        .to_frame("H_crocufid_raw")
    )
    h_crocufid["H_crocufid"] = h_crocufid["H_crocufid_raw"] / 10  # 0-100 → 0-10
    h_crocufid["n_crocufid"] = croc_matched.groupby("dish_id").size()
    print(f"  H_crocufid range: [{h_crocufid['H_crocufid'].min():.2f}, {h_crocufid['H_crocufid'].max():.2f}]")

# ══════════════════════════════════════════════════════════════════
# SOURCE 3: Food-Pics Extended — lab-based palatability ratings
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SOURCE 3: Food-Pics Extended (896 food images)")
print("=" * 60)

fp_file = EXT / "foodpics" / "food.pics.extended.database (Stand 2022).xlsx"
fp = pd.read_excel(fp_file, sheet_name="data", header=1)

# Match food descriptions
fp["dish_id"] = fp["Item_description_english"].apply(lambda x: match_dish(str(x)))
fp_matched = fp.dropna(subset=["dish_id"])
print(f"  Matched items: {len(fp_matched)} / {len(fp)}")
print(f"  Unique dishes: {fp_matched['dish_id'].nunique()}")

# Use palatability columns (0-100 VAS average across groups)
pal_cols = [c for c in fp.columns if "Palatability" in c]
if len(fp_matched) > 0:
    fp_matched = fp_matched.copy()
    fp_matched["avg_palatability"] = fp_matched[pal_cols].mean(axis=1)
    h_foodpics = (
        fp_matched.groupby("dish_id")["avg_palatability"]
        .mean()
        .to_frame("H_foodpics_raw")
    )
    h_foodpics["H_foodpics"] = h_foodpics["H_foodpics_raw"] / 10  # 0-100 → 0-10
    h_foodpics["n_foodpics"] = fp_matched.groupby("dish_id").size()
    print(f"  H_foodpics range: [{h_foodpics['H_foodpics'].min():.2f}, {h_foodpics['H_foodpics'].max():.2f}]")

# ══════════════════════════════════════════════════════════════════
# SOURCE 4: Google Local Reviews — streaming (low memory)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SOURCE 4: Google Local Reviews (streaming)")
print("=" * 60)

google_scores = defaultdict(list)
google_file = EXT / "google_local" / "image_review_all.json"
n_scanned = 0

with open(google_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            review = json.loads(line)
        except (json.JSONDecodeError, Exception):
            continue
        n_scanned += 1
        text = review.get("review_text", "")
        rating = review.get("rating")
        if not text or rating is None:
            continue
        for d in search_dish_in_text(text):
            google_scores[d].append(float(rating))
        if n_scanned % 300000 == 0:
            print(f"  {n_scanned:,} reviews, {sum(len(v) for v in google_scores.values()):,} mentions...")

n_mentions = sum(len(v) for v in google_scores.values())
print(f"  Total: {n_scanned:,} reviews, {n_mentions:,} dish mentions, {len(google_scores)} dishes")

h_google = pd.DataFrame([
    {"dish_id": d, "H_google_raw": np.mean(scores), "n_google": len(scores)}
    for d, scores in google_scores.items()
    if len(scores) >= 3
]).set_index("dish_id")
h_google["H_google"] = h_google["H_google_raw"] * 2  # 1-5 → 1-10
if len(h_google) > 0:
    print(f"  Dishes with ≥3 mentions: {len(h_google)}")
    print(f"  H_google range: [{h_google['H_google'].min():.2f}, {h_google['H_google'].max():.2f}]")

# ══════════════════════════════════════════════════════════════════
# SOURCE 5: TripAdvisor — chunked streaming (low memory)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SOURCE 5: TripAdvisor (6 cities)")
print("=" * 60)

ta_scores = defaultdict(list)
ta_dir = EXT / "tripadvisor"
ta_files = sorted(ta_dir.glob("*_reviews.csv"))

for f in ta_files:
    city = f.stem.replace("_reviews", "")
    n_city = 0
    for chunk in pd.read_csv(f, chunksize=20000,
                              usecols=["review_full", "rating_review"],
                              on_bad_lines="skip"):
        for text, rating in zip(chunk["review_full"], chunk["rating_review"]):
            if not isinstance(text, str) or pd.isna(rating):
                continue
            for d in search_dish_in_text(text):
                ta_scores[d].append(float(rating))
        n_city += len(chunk)
    print(f"  {city}: {n_city:,} reviews")

print(f"  Total dish mentions: {sum(len(v) for v in ta_scores.values()):,}")
print(f"  Unique dishes found: {len(ta_scores)}")

h_tripadvisor = pd.DataFrame([
    {"dish_id": d, "H_ta_raw": np.mean(scores), "n_ta": len(scores)}
    for d, scores in ta_scores.items()
    if len(scores) >= 3
]).set_index("dish_id")
h_tripadvisor["H_tripadvisor"] = h_tripadvisor["H_ta_raw"] * 2  # 1-5 → 1-10
if len(h_tripadvisor) > 0:
    print(f"  Dishes with ≥3 mentions: {len(h_tripadvisor)}")
    print(f"  H_tripadvisor range: [{h_tripadvisor['H_tripadvisor'].min():.2f}, {h_tripadvisor['H_tripadvisor'].max():.2f}]")

# ══════════════════════════════════════════════════════════════════
# MERGE ALL SOURCES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CROSS-PLATFORM MERGE")
print("=" * 60)

# Start with Yelp H
merged = dei[["H_mean"]].rename(columns={"H_mean": "H_yelp"}).copy()

# Join each source
sources = {
    "H_foodcom": h_foodcom[["H_foodcom"]] if "h_foodcom" in dir() and len(h_foodcom) > 0 else None,
    "H_crocufid": h_crocufid[["H_crocufid"]] if "h_crocufid" in dir() and len(h_crocufid) > 0 else None,
    "H_foodpics": h_foodpics[["H_foodpics"]] if "h_foodpics" in dir() and len(h_foodpics) > 0 else None,
    "H_google": h_google[["H_google"]] if "h_google" in dir() and len(h_google) > 0 else None,
    "H_tripadvisor": h_tripadvisor[["H_tripadvisor"]] if "h_tripadvisor" in dir() and len(h_tripadvisor) > 0 else None,
}

for name, df_src in sources.items():
    if df_src is not None:
        merged = merged.join(df_src, how="left")

print(f"\nMerged shape: {merged.shape}")
print(f"\nCoverage per source:")
h_cols = [c for c in merged.columns if c.startswith("H_")]
for col in h_cols:
    n_valid = merged[col].notna().sum()
    print(f"  {col}: {n_valid}/158 dishes ({n_valid/158*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# CROSS-PLATFORM CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CROSS-PLATFORM CORRELATION MATRIX")
print("=" * 60)

# Spearman rank correlation (more robust for cross-scale comparisons)
corr_results = []
for i, col1 in enumerate(h_cols):
    for col2 in h_cols[i + 1:]:
        valid = merged[[col1, col2]].dropna()
        n = len(valid)
        if n >= 5:
            rho, p = stats.spearmanr(valid[col1], valid[col2])
            r_pearson, p_p = stats.pearsonr(valid[col1], valid[col2])
            corr_results.append({
                "Source_1": col1, "Source_2": col2,
                "N_dishes": n,
                "Spearman_rho": rho, "Spearman_p": p,
                "Pearson_r": r_pearson, "Pearson_p": p_p,
            })
            print(f"  {col1} vs {col2}: rho={rho:.3f} (p={p:.4f}), r={r_pearson:.3f}, N={n}")

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(TABLES / "cross_platform_h_correlations.csv", index=False)
print(f"\nSaved: {TABLES / 'cross_platform_h_correlations.csv'}")

# Full correlation matrix (Spearman)
if len(h_cols) >= 2:
    # Pairwise correlation matrix
    corr_matrix = merged[h_cols].corr(method="spearman")
    corr_matrix.to_csv(TABLES / "cross_platform_h_corr_matrix.csv")
    print(f"Saved: {TABLES / 'cross_platform_h_corr_matrix.csv'}")

# ══════════════════════════════════════════════════════════════════
# SAVE MERGED DATA
# ══════════════════════════════════════════════════════════════════
# Add sample sizes
for name in ["foodcom", "crocufid", "foodpics", "google", "ta"]:
    n_col = f"n_{name}"
    h_col = f"H_{name}" if name != "ta" else "H_tripadvisor"
    src_df = {
        "foodcom": h_foodcom if "h_foodcom" in dir() else None,
        "crocufid": h_crocufid if "h_crocufid" in dir() else None,
        "foodpics": h_foodpics if "h_foodpics" in dir() else None,
        "google": h_google if "h_google" in dir() else None,
        "ta": h_tripadvisor if "h_tripadvisor" in dir() else None,
    }[name]
    if src_df is not None and n_col in src_df.columns:
        merged = merged.join(src_df[[n_col]], how="left")

merged.to_csv(DATA / "cross_platform_h_scores.csv")
print(f"\nSaved: {DATA / 'cross_platform_h_scores.csv'}")

# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Cross-Platform Hedonic Score Validation", fontsize=14, fontweight="bold")

    ext_sources = [c for c in h_cols if c != "H_yelp"]
    for idx, ext_col in enumerate(ext_sources[:6]):
        ax = axes[idx // 3, idx % 3]
        valid = merged[["H_yelp", ext_col]].dropna()
        if len(valid) < 3:
            ax.set_visible(False)
            continue
        ax.scatter(valid["H_yelp"], valid[ext_col], alpha=0.6, s=30)
        # Fit line
        slope, intercept = np.polyfit(valid["H_yelp"], valid[ext_col], 1)
        x_range = np.linspace(valid["H_yelp"].min(), valid["H_yelp"].max(), 50)
        ax.plot(x_range, slope * x_range + intercept, "r--", alpha=0.7)
        rho, p = stats.spearmanr(valid["H_yelp"], valid[ext_col])
        ax.set_xlabel("H (Yelp, finetuned BERT)")
        ax.set_ylabel(f"H ({ext_col.replace('H_', '')})")
        ax.set_title(f"ρ={rho:.3f}, N={len(valid)}")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(ext_sources), 6):
        axes[idx // 3, idx % 3].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES / "cross_platform_h_validation.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {FIGURES / 'cross_platform_h_validation.png'}")
    plt.close()

    # Heatmap of correlation matrix
    if len(h_cols) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(h_cols)))
        ax.set_yticks(range(len(h_cols)))
        labels = [c.replace("H_", "").replace("tripadvisor", "TripAdvisor")
                   .replace("foodcom", "Food.com").replace("foodpics", "Food-Pics")
                   .replace("crocufid", "CROCUFID").replace("google", "Google")
                   .replace("yelp", "Yelp") for c in h_cols]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(h_cols)):
            for j in range(len(h_cols)):
                ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="Spearman ρ")
        ax.set_title("Cross-Platform H Score Correlation Matrix")
        plt.tight_layout()
        plt.savefig(FIGURES / "cross_platform_h_heatmap.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {FIGURES / 'cross_platform_h_heatmap.png'}")
        plt.close()

except ImportError:
    print("matplotlib not available, skipping plots")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total external sources processed: {len(ext_sources)}")
print(f"\nKey result — Yelp H correlations with external sources:")
for _, row in corr_df[corr_df["Source_1"] == "H_yelp"].iterrows():
    sig = "***" if row["Spearman_p"] < 0.001 else "**" if row["Spearman_p"] < 0.01 else "*" if row["Spearman_p"] < 0.05 else "n.s."
    print(f"  vs {row['Source_2']}: ρ = {row['Spearman_rho']:.3f} {sig} (N={row['N_dishes']})")

# Overall average
yelp_corrs = corr_df[corr_df["Source_1"] == "H_yelp"]["Spearman_rho"]
if len(yelp_corrs) > 0:
    print(f"\n  Average ρ(Yelp, external): {yelp_corrs.mean():.3f}")
    print(f"  Range: [{yelp_corrs.min():.3f}, {yelp_corrs.max():.3f}]")

print("\nDone!")
