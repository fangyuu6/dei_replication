"""
24_foodcom_validation.py
========================
Cross-context validation: Restaurant (Yelp) vs Home Cooking (Food.com)

Validates that hedonic (H) scores capture food-intrinsic quality rather
than restaurant-experience confounders, by independently applying the
same BERT + Bradley-Terry pipeline to Food.com home-cooking reviews.

Pipeline:
  Phase 1: Match Food.com recipes to DEI dish_ids
  Phase 2: Extract & filter reviews for matched dishes
  Phase 3: BERT scoring + domain-shift calibration diagnostic
  Phase 4: Dish-level aggregation
  Phase 5: BT pairwise ranking (exhaustive or anchor-bridging)
  Phase 6: Statistical comparison (4-layer, bootstrap CIs)
  Phase 7: Visualization + rank-mover analysis

Input:
  raw/external/recipes.parquet   (~522K recipes)
  raw/external/reviews.parquet   (~1.4M reviews)
  models/hedonic_bert_finetuned/ (finetuned BERT)
  data/combined_dish_DEI_v2.csv  (2,563 dishes with Yelp H)
  data/dish_h_pairwise_v2.csv    (Yelp BT H scores)
Output:
  data/foodcom_recipe_dish_map.csv
  data/foodcom_mentions_bert.parquet
  data/foodcom_dish_h_bert.csv
  data/foodcom_pairwise_wins.csv
  data/foodcom_dish_h_pairwise.csv
  results/tables/foodcom_cross_context_validation.csv
  results/figures/foodcom_cross_context_validation.png
  results/figures/foodcom_h_distribution.png
  results/figures/foodcom_rank_movers.png
"""

import sys, os, re, json, time, itertools, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
MODEL_DIR = ROOT / "models" / "hedonic_bert_finetuned"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── API config ────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"

BATCH_SIZE = 10       # pairs per API call
MAX_WORKERS = 50      # concurrent API calls
REVIEW_SAMPLES = 5    # representative reviews per dish
MAX_PER_DISH = 1000   # max reviews per dish for BERT scoring
MIN_WORD_COUNT = 20   # minimum review word count

# ── Load DEI dishes ──────────────────────────────────────────────
dei = pd.read_csv(DATA / "combined_dish_DEI_v2.csv")
if "dish_id" in dei.columns:
    dei = dei.set_index("dish_id")
DISH_IDS = set(dei.index.tolist())
print(f"Loaded {len(DISH_IDS)} DEI dishes")

# Also load Yelp BT H scores for comparison
yelp_bt = pd.read_csv(DATA / "dish_h_pairwise_v2.csv")
if "dish_id" in yelp_bt.columns:
    yelp_bt = yelp_bt.set_index("dish_id")
print(f"Loaded Yelp BT H for {len(yelp_bt)} dishes")

# ── Dish name matching (from 07c) ────────────────────────────────
DISH_ALIASES = {}
for d in DISH_IDS:
    canonical = d.lower().replace("_", " ")
    DISH_ALIASES[canonical] = d
    DISH_ALIASES[d] = d
    DISH_ALIASES[d.replace("_", " ")] = d
    if canonical.endswith("s") and len(canonical) > 4:
        DISH_ALIASES[canonical[:-1]] = d
    if not canonical.endswith("s"):
        DISH_ALIASES[canonical + "s"] = d

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
    "gnocchi": "gnocchi", "ravioli": "ravioli", "risotto": "risotto",
    "tiramisu": "tiramisu", "gelato": "gelato",
    "panna cotta": "panna_cotta", "pannacotta": "panna_cotta",
    "bruschetta": "bruschetta",
    "caprese": "caprese", "caprese salad": "caprese",
    "minestrone": "minestrone", "minestrone soup": "minestrone",
    "osso buco": "osso_buco", "ossobuco": "osso_buco",
    "sushi": "sushi", "sushi roll": "sushi",
    "sashimi": "sashimi", "ramen": "ramen",
    "udon": "udon", "udon noodles": "udon",
    "tempura": "tempura",
    "teriyaki": "teriyaki", "teriyaki chicken": "teriyaki",
    "miso soup": "miso_soup", "miso": "miso_soup",
    "edamame": "edamame", "takoyaki": "takoyaki",
    "onigiri": "onigiri", "rice ball": "onigiri",
    "katsu": "katsu", "tonkatsu": "katsu", "chicken katsu": "katsu",
    "donburi": "donburi", "don": "donburi",
    "okonomiyaki": "okonomiyaki", "yakitori": "yakitori",
    "kimchi": "kimchi", "bibimbap": "bibimbap",
    "bulgogi": "bulgogi", "japchae": "japchae",
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
    "fresh spring roll": "fresh_spring_roll", "summer roll": "fresh_spring_roll",
    "pho": "pho", "bun bo hue": "bun_bo_hue", "bun cha": "bun_cha",
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
    "dan dan noodles": "dan_dan_noodles",
    "chow mein": "chow_mein", "chow fun": "chow_fun",
    "lo mein": "lo_mein", "fried rice": "fried_rice",
    "scallion pancake": "scallion_pancake",
    "char siu": "char_siu", "bbq pork": "char_siu",
    "congee": "congee", "rice porridge": "congee",
    "coq au vin": "coq_au_vin",
    "escargot": "escargot", "escargots": "escargot",
    "crepe": "crepe", "crepes": "crepe",
    "croissant": "croissant", "quiche": "quiche",
    "souffle": "souffle", "soufflé": "souffle",
    "ratatouille": "ratatouille",
    "creme brulee": "creme_brulee", "crème brûlée": "creme_brulee",
    "paella": "paella", "patatas bravas": "patatas_bravas",
    "gazpacho": "gazpacho", "tapas": "tapas",
    "spanakopita": "spanakopita",
    "falafel": "falafel", "hummus": "hummus",
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
    "cheesecake": "cheesecake", "cornbread": "cornbread",
    "grilled cheese": "grilled_cheese",
    "club sandwich": "club_sandwich", "ice cream": "ice_cream",
    "beef bourguignon": "beef_bourguignon",
    "ceviche": "ceviche", "fattoush": "fattoush",
    "rojak": "rojak", "som tam": "som_tam",
}
DISH_ALIASES.update(EXTRA_ALIASES)


def match_dish(text):
    """Try to match text to a canonical dish_id."""
    text_lower = text.lower().strip()
    if text_lower in DISH_ALIASES:
        return DISH_ALIASES[text_lower]
    for prefix in ["homemade ", "easy ", "best ", "simple ", "classic ",
                    "authentic ", "traditional ", "my ", "copycat ",
                    "the best ", "world's best ", "mom's ", "grandma's "]:
        if text_lower.startswith(prefix):
            rest = text_lower[len(prefix):]
            if rest in DISH_ALIASES:
                return DISH_ALIASES[rest]
    for suffix in [" recipe", " recipes", " dish", " bowl", " plate",
                   " i", " ii", " iii", " style"]:
        if text_lower.endswith(suffix):
            rest = text_lower[:-len(suffix)]
            if rest in DISH_ALIASES:
                return DISH_ALIASES[rest]
    return None


def match_dish_fuzzy(text):
    """Extended fuzzy matching: token-set overlap."""
    text_lower = text.lower().strip()
    text_lower = re.sub(r'[^a-z0-9\s]', '', text_lower)
    text_tokens = set(text_lower.split())
    if len(text_tokens) < 2:
        return None
    best_match = None
    best_overlap = 0
    for d in DISH_IDS:
        d_tokens = set(d.lower().replace("_", " ").split())
        if len(d_tokens) < 2:
            continue
        # All dish tokens must appear in recipe name
        if d_tokens.issubset(text_tokens):
            overlap = len(d_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = d
    return best_match


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Recipe matching
# ══════════════════════════════════════════════════════════════════
def phase1_match_recipes():
    """Match Food.com recipes to DEI dish_ids."""
    out_path = DATA / "foodcom_recipe_dish_map.csv"
    if out_path.exists():
        print(f"  [skip] {out_path} exists, loading...")
        return pd.read_csv(out_path)

    print("\n" + "=" * 60)
    print("PHASE 1: Match Food.com recipes to DEI dishes")
    print("=" * 60)

    recipes = pd.read_parquet(EXT / "recipes.parquet")
    print(f"  Loaded {len(recipes):,} Food.com recipes")

    # Pass 1: exact + prefix/suffix stripping
    recipes["dish_id"] = recipes["Name"].apply(lambda x: match_dish(str(x)))
    n_pass1 = recipes["dish_id"].notna().sum()
    print(f"  Pass 1 (exact): {n_pass1:,} recipes matched")

    # Pass 2: fuzzy token-set matching for unmatched
    mask_unmatched = recipes["dish_id"].isna()
    recipes.loc[mask_unmatched, "dish_id"] = (
        recipes.loc[mask_unmatched, "Name"]
        .apply(lambda x: match_dish_fuzzy(str(x)))
    )
    n_pass2 = recipes["dish_id"].notna().sum() - n_pass1
    print(f"  Pass 2 (fuzzy): {n_pass2:,} additional matches")

    matched = recipes.dropna(subset=["dish_id"]).copy()
    matched["match_method"] = "exact"
    matched.loc[matched.index.isin(
        recipes[mask_unmatched & recipes["dish_id"].notna()].index
    ), "match_method"] = "fuzzy"

    print(f"  Total matched recipes: {len(matched):,}")
    print(f"  Unique DEI dishes matched: {matched['dish_id'].nunique()}")

    # Save mapping
    result = matched[["RecipeId", "Name", "dish_id", "match_method"]].copy()
    result.columns = ["RecipeId", "recipe_name", "dish_id", "match_method"]
    result.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return result


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Review extraction & filtering
# ══════════════════════════════════════════════════════════════════
def phase2_extract_reviews(recipe_map):
    """Extract and filter Food.com reviews for matched dishes."""
    out_path = DATA / "foodcom_mentions_bert.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        if "H_bert" in df.columns and df["H_bert"].notna().all():
            print(f"  [skip] {out_path} exists with H_bert, loading...")
            return df
        print(f"  [skip] {out_path} exists (no H_bert yet), loading...")
        return df

    print("\n" + "=" * 60)
    print("PHASE 2: Extract & filter Food.com reviews")
    print("=" * 60)

    reviews = pd.read_parquet(EXT / "reviews.parquet")
    print(f"  Loaded {len(reviews):,} Food.com reviews")

    # Merge with recipe-dish mapping
    merged = reviews.merge(
        recipe_map[["RecipeId", "dish_id"]],
        on="RecipeId", how="inner"
    )
    print(f"  Reviews for matched dishes: {len(merged):,}")

    # Filter: minimum word count
    merged["word_count"] = merged["Review"].astype(str).str.split().str.len()
    merged = merged[merged["word_count"] >= MIN_WORD_COUNT].copy()
    print(f"  After word count filter (≥{MIN_WORD_COUNT}): {len(merged):,}")

    # Sample: max per dish
    before = len(merged)
    merged = (
        merged.groupby("dish_id", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), MAX_PER_DISH), random_state=42))
        .reset_index(drop=True)
    )
    print(f"  After sampling (max {MAX_PER_DISH}/dish): {before:,} → {len(merged):,}")

    # Rename for consistency
    result = merged[["dish_id", "RecipeId", "Review", "Rating", "word_count"]].copy()
    result.columns = ["dish_id", "RecipeId", "context_text", "star_rating", "word_count"]
    result["platform"] = "foodcom"

    result.to_parquet(out_path, index=False)
    print(f"  Unique dishes: {result['dish_id'].nunique()}")
    print(f"  Saved: {out_path}")

    return result


# ══════════════════════════════════════════════════════════════════
# PHASE 3: BERT scoring + domain-shift calibration
# ══════════════════════════════════════════════════════════════════
def load_bert_model():
    """Load finetuned BERT model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"  Loading finetuned BERT from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")
    return tokenizer, model, device


def score_batch(texts, tokenizer, model, device, batch_size=128):
    """Score texts with finetuned BERT. Returns numpy array."""
    import torch
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:512] for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            preds = model(**enc).logits.squeeze(-1).cpu().numpy()
        if preds.ndim == 0:
            all_preds.append(float(preds))
        else:
            all_preds.extend(preds.tolist())
    return np.clip(np.array(all_preds), 1.0, 10.0)


def phase3_bert_scoring(mentions_df):
    """Score Food.com reviews with finetuned BERT + calibration diagnostic."""
    if "H_bert" in mentions_df.columns and mentions_df["H_bert"].notna().all():
        print("  [skip] H_bert already computed")
        return mentions_df

    print("\n" + "=" * 60)
    print("PHASE 3: BERT scoring + domain-shift calibration")
    print("=" * 60)

    tokenizer, model, device = load_bert_model()
    texts = mentions_df["context_text"].astype(str).tolist()

    # Score in mega-batches with progress
    mega = 10000
    all_scores = []
    for start in range(0, len(texts), mega):
        end = min(start + mega, len(texts))
        batch_scores = score_batch(texts[start:end], tokenizer, model, device)
        all_scores.extend(batch_scores.tolist())
        pct = end / len(texts) * 100
        print(f"    {end:,}/{len(texts):,} ({pct:.0f}%) — "
              f"batch mean H={np.mean(batch_scores):.2f}", flush=True)

    mentions_df = mentions_df.copy()
    mentions_df["H_bert"] = all_scores

    # Domain-shift calibration diagnostic:
    # Fit linear calibration using Food.com star ratings
    valid_stars = mentions_df.dropna(subset=["star_rating"])
    if len(valid_stars) > 100:
        star_h = valid_stars["star_rating"] * 2  # 1-5 → 2-10
        bert_h = valid_stars["H_bert"]
        slope, intercept, _, _, _ = stats.linregress(bert_h, star_h)
        mentions_df["H_bert_calibrated"] = np.clip(
            slope * mentions_df["H_bert"] + intercept, 1.0, 10.0
        )
        print(f"\n  Domain-shift calibration: H_cal = {slope:.3f} * H_bert + {intercept:.3f}")
        print(f"  Raw BERT mean={mentions_df['H_bert'].mean():.3f}, "
              f"Calibrated mean={mentions_df['H_bert_calibrated'].mean():.3f}")
    else:
        mentions_df["H_bert_calibrated"] = mentions_df["H_bert"]
        print("  [warn] Not enough star ratings for calibration")

    # Save
    out_path = DATA / "foodcom_mentions_bert.parquet"
    mentions_df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    return mentions_df


# ══════════════════════════════════════════════════════════════════
# PHASE 4: Dish-level aggregation
# ══════════════════════════════════════════════════════════════════
def phase4_aggregate(mentions_df):
    """Aggregate mention-level BERT scores to dish-level H."""
    print("\n" + "=" * 60)
    print("PHASE 4: Dish-level aggregation")
    print("=" * 60)

    agg = (
        mentions_df.groupby("dish_id")["H_bert"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "H_foodcom_bert",
                         "std": "H_foodcom_bert_std",
                         "count": "n_foodcom_bert"})
    )
    agg["H_foodcom_bert_ci95"] = (
        1.96 * agg["H_foodcom_bert_std"] / np.sqrt(agg["n_foodcom_bert"])
    )

    # Also aggregate calibrated scores
    if "H_bert_calibrated" in mentions_df.columns:
        cal_agg = (
            mentions_df.groupby("dish_id")["H_bert_calibrated"]
            .mean()
            .rename("H_foodcom_bert_cal")
        )
        agg = agg.join(cal_agg)

    # Also aggregate star ratings
    star_agg = (
        mentions_df.dropna(subset=["star_rating"])
        .groupby("dish_id")["star_rating"]
        .mean() * 2  # 1-5 → 2-10
    ).rename("H_foodcom_star")
    agg = agg.join(star_agg)

    # Filter: min 5 reviews
    agg = agg[agg["n_foodcom_bert"] >= 5]

    out_path = DATA / "foodcom_dish_h_bert.csv"
    agg.to_csv(out_path)
    print(f"  Dishes with ≥5 reviews: {len(agg)}")
    print(f"  H_foodcom_bert: mean={agg['H_foodcom_bert'].mean():.3f}, "
          f"std={agg['H_foodcom_bert'].std():.3f}, "
          f"CV={agg['H_foodcom_bert'].std()/agg['H_foodcom_bert'].mean()*100:.1f}%")
    print(f"  Saved: {out_path}")

    return agg


# ══════════════════════════════════════════════════════════════════
# PHASE 5: BT pairwise ranking
# ══════════════════════════════════════════════════════════════════
def select_representative_reviews(scored_df, n=REVIEW_SAMPLES):
    """Pick n reviews per dish, stratified by BERT score percentile."""
    profiles = {}
    percentiles = np.linspace(0.1, 0.9, n)

    for dish_id, group in scored_df.groupby("dish_id"):
        s = group["H_bert"].dropna()
        if len(s) < n:
            idx = s.index
        else:
            targets = np.quantile(s.values, percentiles)
            idx = []
            used = set()
            for t in targets:
                dists = (s - t).abs()
                for i in dists.sort_values().index:
                    if i not in used:
                        idx.append(i)
                        used.add(i)
                        break
            idx = idx[:n]

        excerpts = []
        for i in idx:
            text = group.loc[i, "context_text"]
            if isinstance(text, str) and len(text) > 20:
                excerpts.append(text[:200].strip())
        if excerpts:
            profiles[dish_id] = excerpts

    return profiles


def make_batch_prompt(pairs_with_reviews):
    """Create a prompt comparing multiple pairs at once."""
    lines = [
        "You are a food expert. For each numbered pair below, read the home cooking "
        "review excerpts and decide which dish sounds MORE DELICIOUS based on the "
        "taste experience described. "
        "Consider flavor, texture, freshness, and overall enjoyment — NOT healthiness or price. "
        "Answer with ONLY the dish letter (A or B) for each pair, one per line. "
        "Format: 1:A  2:B  3:A  etc.\n"
    ]
    for i, (dish_a, dish_b, revs_a, revs_b) in enumerate(pairs_with_reviews, 1):
        a_name = dish_a.replace("_", " ").title()
        b_name = dish_b.replace("_", " ").title()
        a_text = " | ".join(revs_a[:3])
        b_text = " | ".join(revs_b[:3])
        lines.append(
            f"Pair {i}:\n"
            f"  Dish A ({a_name}): {a_text}\n"
            f"  Dish B ({b_name}): {b_text}\n"
        )
    return "\n".join(lines)


def parse_batch_response(text, n_pairs):
    """Parse '1:A  2:B  3:A' format responses."""
    results = {}
    text = text.strip().upper()
    matches = re.findall(r'(\d+)\s*[:\-\.]\s*([AB])', text)
    if matches:
        for num_str, choice in matches:
            results[int(num_str)] = choice
    else:
        choices = re.findall(r'[AB]', text)
        for i, c in enumerate(choices[:n_pairs], 1):
            results[i] = c
    return results


def call_api(prompt, max_retries=3):
    """Call OpenRouter API."""
    import urllib.request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def process_batch(batch_idx, pairs_batch, profiles):
    """Process one batch of pairwise comparisons."""
    pairs_with_reviews = []
    for dish_a, dish_b in pairs_batch:
        revs_a = profiles.get(dish_a, ["(no reviews)"])
        revs_b = profiles.get(dish_b, ["(no reviews)"])
        pairs_with_reviews.append((dish_a, dish_b, revs_a, revs_b))

    prompt = make_batch_prompt(pairs_with_reviews)
    response = call_api(prompt)

    results = []
    if response:
        parsed = parse_batch_response(response, len(pairs_batch))
        for i, (dish_a, dish_b) in enumerate(pairs_batch, 1):
            choice = parsed.get(i)
            if choice == "A":
                results.append({"winner": dish_a, "loser": dish_b})
            elif choice == "B":
                results.append({"winner": dish_b, "loser": dish_a})
    return batch_idx, results


def fit_bradley_terry(wins_df, dish_ids):
    """Fit Bradley-Terry model using choix library."""
    import choix

    id2idx = {d: i for i, d in enumerate(dish_ids)}
    n = len(dish_ids)
    comparisons = []
    for _, row in wins_df.iterrows():
        w = id2idx.get(row["winner"])
        l = id2idx.get(row["loser"])
        if w is not None and l is not None:
            comparisons.append((w, l))

    if not comparisons:
        raise ValueError("No valid comparisons found")

    print(f"  Fitting BT on {len(comparisons)} comparisons, {n} items", flush=True)
    params = choix.ilsr_pairwise(n, comparisons, alpha=0.01)

    p_min, p_max = params.min(), params.max()
    if p_max > p_min:
        h_pairwise = 1 + 9 * (params - p_min) / (p_max - p_min)
    else:
        h_pairwise = np.full(n, 5.5)

    return pd.DataFrame({
        "dish_id": dish_ids,
        "H_pairwise_home": h_pairwise,
        "BT_strength": params,
    }).set_index("dish_id")


def phase5_pairwise(mentions_df, dish_h_bert):
    """Run BT pairwise ranking on Food.com reviews."""
    wins_path = DATA / "foodcom_pairwise_wins.csv"
    bt_path = DATA / "foodcom_dish_h_pairwise.csv"

    if bt_path.exists():
        print(f"  [skip] {bt_path} exists, loading...")
        return pd.read_csv(bt_path, index_col="dish_id")

    if not API_KEY:
        print("  [skip] No OPENROUTER_API_KEY set, skipping pairwise phase")
        return None

    print("\n" + "=" * 60)
    print("PHASE 5: BT pairwise ranking on Food.com reviews")
    print("=" * 60)

    # Only use dishes with enough reviews for meaningful profiles
    dish_ids = sorted(dish_h_bert.index.tolist())
    print(f"  Dishes for pairwise: {len(dish_ids)}")

    # Select representative reviews
    profiles = select_representative_reviews(mentions_df, n=REVIEW_SAMPLES)
    dish_ids = [d for d in dish_ids if d in profiles]
    print(f"  Dishes with profiles: {len(dish_ids)}")

    # Generate pairs
    n_dishes = len(dish_ids)
    if n_dishes <= 200:
        # Exhaustive
        all_pairs = list(itertools.combinations(dish_ids, 2))
        print(f"  Exhaustive: C({n_dishes},2) = {len(all_pairs)} pairs")
    else:
        # Anchor-bridging
        sorted_dishes = dish_h_bert.loc[dish_ids].sort_values("H_foodcom_bert")
        n_anchors = 20
        anchor_idx = np.linspace(0, len(sorted_dishes) - 1, n_anchors, dtype=int)
        anchors = sorted_dishes.index[anchor_idx].tolist()
        all_pairs = []
        for d in dish_ids:
            for a in anchors:
                if d != a:
                    all_pairs.append((d, a) if d < a else (a, d))
        all_pairs = list(set(all_pairs))
        print(f"  Anchor-bridging: {n_dishes} dishes × {n_anchors} anchors "
              f"= {len(all_pairs)} pairs")

    # Batch API calls
    np.random.seed(42)
    np.random.shuffle(all_pairs)
    batches = [all_pairs[i:i+BATCH_SIZE] for i in range(0, len(all_pairs), BATCH_SIZE)]
    print(f"  API batches: {len(batches)} (BATCH_SIZE={BATCH_SIZE}, "
          f"MAX_WORKERS={MAX_WORKERS})")

    all_wins = []
    checkpoint_path = DATA / "foodcom_pairwise_checkpoint.json"

    # Load checkpoint if exists
    start_batch = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        all_wins = ckpt.get("wins", [])
        start_batch = ckpt.get("next_batch", 0)
        print(f"  Resuming from batch {start_batch} ({len(all_wins)} wins so far)")

    remaining_batches = list(enumerate(batches[start_batch:], start=start_batch))
    done = 0
    total = len(remaining_batches)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for batch_idx, batch in remaining_batches:
            fut = executor.submit(process_batch, batch_idx, batch, profiles)
            futures[fut] = batch_idx

        for fut in as_completed(futures):
            batch_idx, results = fut.result()
            all_wins.extend(results)
            done += 1

            if done % 100 == 0 or done == total:
                # Checkpoint
                with open(checkpoint_path, "w") as f:
                    json.dump({"wins": all_wins, "next_batch": batch_idx + 1}, f)
                print(f"    {done}/{total} batches done, "
                      f"{len(all_wins)} wins", flush=True)

    # Save wins
    wins_df = pd.DataFrame(all_wins)
    wins_df.to_csv(wins_path, index=False)
    print(f"  Saved {len(wins_df)} wins → {wins_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Fit Bradley-Terry
    bt_scores = fit_bradley_terry(wins_df, dish_ids)
    bt_scores.to_csv(bt_path)
    print(f"  Saved BT scores → {bt_path}")

    return bt_scores


# ══════════════════════════════════════════════════════════════════
# PHASE 6: Statistical comparison
# ══════════════════════════════════════════════════════════════════
def bootstrap_spearman(x, y, n_boot=2000, seed=42):
    """Compute Spearman rho with bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    rho, p = stats.spearmanr(x, y)
    rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(r)
    ci_lo, ci_hi = np.percentile(rhos, [2.5, 97.5])
    return rho, p, ci_lo, ci_hi


def phase6_compare(dish_h_foodcom, bt_scores_foodcom):
    """Statistical comparison: Food.com H vs Yelp H."""
    print("\n" + "=" * 60)
    print("PHASE 6: Statistical comparison")
    print("=" * 60)

    # Load Yelp scores
    yelp_bert = pd.read_csv(DATA / "cross_platform_h_bert.csv", index_col="dish_id")
    yelp_h_col = "H_yelp"

    corr_rows = []

    # --- Comparison 1: Food.com BERT H vs Yelp BERT H ---
    merged = dish_h_foodcom[["H_foodcom_bert"]].join(
        yelp_bert[[yelp_h_col]], how="inner"
    ).dropna()
    if len(merged) >= 5:
        rho, p, ci_lo, ci_hi = bootstrap_spearman(
            merged["H_foodcom_bert"].values, merged[yelp_h_col].values
        )
        r_p, _ = stats.pearsonr(merged["H_foodcom_bert"], merged[yelp_h_col])
        print(f"\n  Food.com BERT H vs Yelp BERT H:")
        print(f"    N={len(merged)}, ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], r={r_p:.3f}")
        corr_rows.append({
            "Comparison": "Food.com BERT H vs Yelp BERT H",
            "Type": "cross_context_NLP",
            "N_dishes": len(merged), "Spearman_rho": rho,
            "CI95_lo": ci_lo, "CI95_hi": ci_hi,
            "Spearman_p": p, "Pearson_r": r_p,
        })

    # --- Comparison 1b: Calibrated BERT ---
    if "H_foodcom_bert_cal" in dish_h_foodcom.columns:
        merged_cal = dish_h_foodcom[["H_foodcom_bert_cal"]].join(
            yelp_bert[[yelp_h_col]], how="inner"
        ).dropna()
        if len(merged_cal) >= 5:
            rho_cal, p_cal, ci_lo_cal, ci_hi_cal = bootstrap_spearman(
                merged_cal["H_foodcom_bert_cal"].values,
                merged_cal[yelp_h_col].values
            )
            delta_rho = rho_cal - rho
            print(f"\n  Calibrated BERT H vs Yelp BERT H:")
            print(f"    N={len(merged_cal)}, ρ={rho_cal:.3f} [{ci_lo_cal:.3f}, {ci_hi_cal:.3f}]")
            print(f"    Δρ (calibration effect) = {delta_rho:+.3f}")
            if abs(delta_rho) < 0.05:
                print(f"    → Domain shift negligible (Δρ < 0.05)")
            elif abs(delta_rho) > 0.10:
                print(f"    → Domain shift significant (Δρ > 0.10)")
            else:
                print(f"    → Moderate domain shift")
            corr_rows.append({
                "Comparison": "Food.com Calibrated BERT H vs Yelp BERT H",
                "Type": "cross_context_NLP_calibrated",
                "N_dishes": len(merged_cal), "Spearman_rho": rho_cal,
                "CI95_lo": ci_lo_cal, "CI95_hi": ci_hi_cal,
                "Spearman_p": p_cal, "Pearson_r": np.nan,
                "Delta_rho": delta_rho,
            })

    # --- Comparison 2: Food.com BT H vs Yelp BT H ---
    if bt_scores_foodcom is not None:
        merged_bt = bt_scores_foodcom[["H_pairwise_home"]].join(
            yelp_bt[["H_pairwise"]], how="inner"
        ).dropna()
        if len(merged_bt) >= 5:
            rho_bt, p_bt, ci_lo_bt, ci_hi_bt = bootstrap_spearman(
                merged_bt["H_pairwise_home"].values,
                merged_bt["H_pairwise"].values
            )
            r_bt, _ = stats.pearsonr(
                merged_bt["H_pairwise_home"], merged_bt["H_pairwise"]
            )
            print(f"\n  Food.com BT H vs Yelp BT H:")
            print(f"    N={len(merged_bt)}, ρ={rho_bt:.3f} [{ci_lo_bt:.3f}, {ci_hi_bt:.3f}], "
                  f"r={r_bt:.3f}")
            corr_rows.append({
                "Comparison": "Food.com BT H vs Yelp BT H",
                "Type": "cross_context_BT",
                "N_dishes": len(merged_bt), "Spearman_rho": rho_bt,
                "CI95_lo": ci_lo_bt, "CI95_hi": ci_hi_bt,
                "Spearman_p": p_bt, "Pearson_r": r_bt,
            })

    # --- Comparison 3: Food.com star vs Yelp BERT H ---
    if "H_foodcom_star" in dish_h_foodcom.columns:
        merged_star = dish_h_foodcom[["H_foodcom_star"]].join(
            yelp_bert[[yelp_h_col]], how="inner"
        ).dropna()
        if len(merged_star) >= 5:
            rho_s, p_s, ci_lo_s, ci_hi_s = bootstrap_spearman(
                merged_star["H_foodcom_star"].values,
                merged_star[yelp_h_col].values
            )
            print(f"\n  Food.com star vs Yelp BERT H:")
            print(f"    N={len(merged_star)}, ρ={rho_s:.3f} [{ci_lo_s:.3f}, {ci_hi_s:.3f}]")
            corr_rows.append({
                "Comparison": "Food.com star rating vs Yelp BERT H",
                "Type": "cross_method",
                "N_dishes": len(merged_star), "Spearman_rho": rho_s,
                "CI95_lo": ci_lo_s, "CI95_hi": ci_hi_s,
                "Spearman_p": p_s, "Pearson_r": np.nan,
            })

    # --- Comparison 4: Food.com BERT H vs Food.com star ---
    if "H_foodcom_star" in dish_h_foodcom.columns:
        merged_int = dish_h_foodcom[["H_foodcom_bert", "H_foodcom_star"]].dropna()
        if len(merged_int) >= 5:
            rho_i, p_i, ci_lo_i, ci_hi_i = bootstrap_spearman(
                merged_int["H_foodcom_bert"].values,
                merged_int["H_foodcom_star"].values
            )
            print(f"\n  Food.com BERT H vs Food.com star:")
            print(f"    N={len(merged_int)}, ρ={rho_i:.3f} [{ci_lo_i:.3f}, {ci_hi_i:.3f}]")
            corr_rows.append({
                "Comparison": "Food.com BERT H vs Food.com star rating",
                "Type": "internal_method",
                "N_dishes": len(merged_int), "Spearman_rho": rho_i,
                "CI95_lo": ci_lo_i, "CI95_hi": ci_hi_i,
                "Spearman_p": p_i, "Pearson_r": np.nan,
            })

    # Save
    corr_df = pd.DataFrame(corr_rows)
    out_path = TABLES / "foodcom_cross_context_validation.csv"
    corr_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return corr_df


# ══════════════════════════════════════════════════════════════════
# PHASE 7: Visualization + rank-mover analysis
# ══════════════════════════════════════════════════════════════════
def phase7_visualize(dish_h_foodcom, bt_scores_foodcom):
    """Generate figures and rank-mover analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("PHASE 7: Visualization + rank-mover analysis")
    print("=" * 60)

    yelp_bert = pd.read_csv(DATA / "cross_platform_h_bert.csv", index_col="dish_id")

    # ── Figure 1: Cross-context scatter (2 panels) ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: BERT-BERT
    merged = dish_h_foodcom[["H_foodcom_bert"]].join(
        yelp_bert[["H_yelp"]], how="inner"
    ).dropna()
    if len(merged) >= 5:
        ax = axes[0]
        ax.scatter(merged["H_yelp"], merged["H_foodcom_bert"],
                   alpha=0.5, s=30, c="#2196F3")
        rho, p = stats.spearmanr(merged["H_yelp"], merged["H_foodcom_bert"])
        # Fit line
        z = np.polyfit(merged["H_yelp"], merged["H_foodcom_bert"], 1)
        xline = np.linspace(merged["H_yelp"].min(), merged["H_yelp"].max(), 100)
        ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.7)
        ax.set_xlabel("H (Yelp BERT)", fontsize=11)
        ax.set_ylabel("H (Food.com BERT)", fontsize=11)
        ax.set_title(f"BERT H: Restaurant vs Home\n"
                     f"ρ = {rho:.3f}, N = {len(merged)}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Panel B: BT-BT
    if bt_scores_foodcom is not None:
        merged_bt = bt_scores_foodcom[["H_pairwise_home"]].join(
            yelp_bt[["H_pairwise"]], how="inner"
        ).dropna()
        if len(merged_bt) >= 5:
            ax = axes[1]
            ax.scatter(merged_bt["H_pairwise"], merged_bt["H_pairwise_home"],
                       alpha=0.5, s=30, c="#FF9800")
            rho_bt, _ = stats.spearmanr(
                merged_bt["H_pairwise"], merged_bt["H_pairwise_home"]
            )
            z = np.polyfit(merged_bt["H_pairwise"], merged_bt["H_pairwise_home"], 1)
            xline = np.linspace(merged_bt["H_pairwise"].min(),
                                merged_bt["H_pairwise"].max(), 100)
            ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.7)
            ax.set_xlabel("H (Yelp BT)", fontsize=11)
            ax.set_ylabel("H (Food.com BT)", fontsize=11)
            ax.set_title(f"Bradley-Terry H: Restaurant vs Home\n"
                         f"ρ = {rho_bt:.3f}, N = {len(merged_bt)}", fontsize=12)
            ax.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "BT pairwise\nnot available\n(no API key)",
                     ha="center", va="center", fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title("Bradley-Terry H")

    plt.tight_layout()
    fig_path = FIGURES / "foodcom_cross_context_validation.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Figure 2: H distribution comparison ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    merged_all = dish_h_foodcom[["H_foodcom_bert"]].join(
        yelp_bert[["H_yelp"]], how="inner"
    ).dropna()
    if len(merged_all) >= 5:
        ax.hist(merged_all["H_yelp"], bins=30, alpha=0.5, label="Yelp (restaurant)",
                color="#2196F3", density=True)
        ax.hist(merged_all["H_foodcom_bert"], bins=30, alpha=0.5,
                label="Food.com (home)", color="#FF9800", density=True)
        cv_yelp = merged_all["H_yelp"].std() / merged_all["H_yelp"].mean() * 100
        cv_fc = (merged_all["H_foodcom_bert"].std() /
                 merged_all["H_foodcom_bert"].mean() * 100)
        ax.set_xlabel("BERT Hedonic Score (H)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"H distribution: Yelp CV={cv_yelp:.1f}% vs "
                     f"Food.com CV={cv_fc:.1f}%", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig_path = FIGURES / "foodcom_h_distribution.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Figure 3: Rank movers ─────────────────────────────────────
    # Use BERT H for rank comparison (available for more dishes)
    merged_rank = dish_h_foodcom[["H_foodcom_bert"]].join(
        yelp_bert[["H_yelp"]], how="inner"
    ).dropna()
    if len(merged_rank) >= 10:
        merged_rank["rank_yelp"] = merged_rank["H_yelp"].rank(ascending=False)
        merged_rank["rank_foodcom"] = merged_rank["H_foodcom_bert"].rank(ascending=False)
        merged_rank["rank_shift"] = merged_rank["rank_foodcom"] - merged_rank["rank_yelp"]
        merged_rank["abs_shift"] = merged_rank["rank_shift"].abs()

        # Top 10 movers
        top_movers = merged_rank.nlargest(10, "abs_shift").sort_values("rank_shift")

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4CAF50" if s > 0 else "#F44336" for s in top_movers["rank_shift"]]
        bars = ax.barh(
            [d.replace("_", " ").title() for d in top_movers.index],
            top_movers["rank_shift"],
            color=colors, alpha=0.8
        )
        ax.set_xlabel("Rank Shift (Food.com − Yelp)", fontsize=11)
        ax.set_title("Top 10 Context-Sensitive Dishes\n"
                     "(+) = rated higher at home, (−) = rated higher in restaurants",
                     fontsize=12)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        fig_path = FIGURES / "foodcom_rank_movers.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

        # Print rank movers summary
        mean_abs_shift = merged_rank["abs_shift"].mean()
        context_pct = (merged_rank["abs_shift"] > len(merged_rank) * 0.1).mean() * 100
        print(f"\n  Rank-mover analysis:")
        print(f"    Mean absolute rank shift: {mean_abs_shift:.1f}")
        print(f"    Dishes with >10% rank shift: {context_pct:.1f}%")
        print(f"    Top movers:")
        for d in top_movers.index:
            shift = top_movers.loc[d, "rank_shift"]
            direction = "↑ home" if shift > 0 else "↓ home"
            print(f"      {d:30s}: {shift:+.0f} ({direction})")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Food.com Cross-Context Validation")
    print("Restaurant (Yelp) vs Home Cooking (Food.com)")
    print("=" * 60)

    # Phase 1
    recipe_map = phase1_match_recipes()

    # Phase 2
    mentions = phase2_extract_reviews(recipe_map)

    # Phase 3
    mentions = phase3_bert_scoring(mentions)

    # Phase 4
    dish_h = phase4_aggregate(mentions)

    # Phase 5
    bt_scores = phase5_pairwise(mentions, dish_h)

    # Phase 6
    corr_df = phase6_compare(dish_h, bt_scores)

    # Phase 7
    phase7_visualize(dish_h, bt_scores)

    print("\n" + "=" * 60)
    print("DONE. Key results:")
    print("=" * 60)
    for _, row in corr_df.iterrows():
        print(f"  {row['Comparison']}: ρ={row['Spearman_rho']:.3f} "
              f"[{row.get('CI95_lo', np.nan):.3f}, {row.get('CI95_hi', np.nan):.3f}] "
              f"(N={row['N_dishes']})")


if __name__ == "__main__":
    main()
