"""
02_extract_dishes.py - From restaurant reviews to dish-level data
=================================================================
Pipeline:
  1. Load restaurant reviews (from 01_explore_yelp.py output)
  2. Filter reviews: length >= 20 words, English
  3. Build dish dictionary from common dish names
  4. Match dish names in review text
  5. Extract dish-level context sentences
  6. Output: dish_mentions table (dish_name, review_id, context_text)

This is the RULE-BASED baseline (Plan A in the research plan).
It will be superseded by LLM-based extraction in 03_nlp_extraction.py,
but provides the initial dish dictionary and coverage estimates.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    TABLES_DIR,
    MIN_REVIEW_WORDS,
    CUISINE_KEYWORDS,
    RESTAURANT_REVIEWS_PARQUET,
    RESTAURANTS_PARQUET,
)


# ── Dish Dictionary ──────────────────────────────────────────────────
# Comprehensive list of common dish names for keyword matching.
# Organized by cuisine for traceability.
# This will be expanded with recipe database data in later phases.

DISH_DICTIONARY = {
    # ── American ──
    "hamburger": {"cuisine": "American", "cook_method": "grill",
                  "aliases": ["burger", "cheeseburger", "hamburger"]},
    "french_fries": {"cuisine": "American", "cook_method": "deep_fry",
                     "aliases": ["fries", "french fries", "curly fries", "waffle fries"]},
    "steak": {"cuisine": "American", "cook_method": "grill",
              "aliases": ["steak", "ribeye", "filet mignon", "ny strip",
                          "new york strip", "sirloin", "t-bone"]},
    "buffalo_wings": {"cuisine": "American", "cook_method": "deep_fry",
                      "aliases": ["wings", "buffalo wings", "chicken wings", "hot wings"]},
    "mac_and_cheese": {"cuisine": "American", "cook_method": "bake",
                       "aliases": ["mac and cheese", "mac & cheese", "mac n cheese",
                                   "macaroni and cheese"]},
    "pulled_pork": {"cuisine": "American", "cook_method": "slow_cook",
                    "aliases": ["pulled pork", "pulled pork sandwich"]},
    "coleslaw": {"cuisine": "American", "cook_method": "raw",
                 "aliases": ["coleslaw", "cole slaw"]},
    "cornbread": {"cuisine": "American", "cook_method": "bake",
                  "aliases": ["cornbread", "corn bread"]},
    "ribs": {"cuisine": "American", "cook_method": "smoke",
             "aliases": ["ribs", "baby back ribs", "spare ribs", "bbq ribs"]},
    "brisket": {"cuisine": "American", "cook_method": "smoke",
                "aliases": ["brisket", "smoked brisket"]},
    "lobster_roll": {"cuisine": "American", "cook_method": "boil",
                     "aliases": ["lobster roll"]},
    "clam_chowder": {"cuisine": "American", "cook_method": "simmer",
                     "aliases": ["clam chowder", "new england clam chowder"]},
    "pancake": {"cuisine": "American", "cook_method": "pan_fry",
                "aliases": ["pancake", "pancakes", "flapjacks"]},
    "waffle": {"cuisine": "American", "cook_method": "bake",
               "aliases": ["waffle", "waffles", "belgian waffle"]},

    # ── Chinese ──
    "kung_pao_chicken": {"cuisine": "Chinese", "cook_method": "stir_fry",
                         "aliases": ["kung pao chicken", "kung pao", "gong bao chicken"]},
    "fried_rice": {"cuisine": "Chinese", "cook_method": "stir_fry",
                   "aliases": ["fried rice", "egg fried rice", "chicken fried rice",
                               "shrimp fried rice", "pork fried rice",
                               "vegetable fried rice", "yang chow fried rice"]},
    "lo_mein": {"cuisine": "Chinese", "cook_method": "stir_fry",
                "aliases": ["lo mein", "chicken lo mein", "beef lo mein",
                            "shrimp lo mein", "vegetable lo mein"]},
    "chow_mein": {"cuisine": "Chinese", "cook_method": "stir_fry",
                  "aliases": ["chow mein", "chicken chow mein"]},
    "dumplings": {"cuisine": "Chinese", "cook_method": "steam",
                  "aliases": ["dumplings", "dumpling", "pot stickers",
                              "potstickers", "jiaozi", "gyoza"]},
    "spring_rolls": {"cuisine": "Chinese", "cook_method": "deep_fry",
                     "aliases": ["spring rolls", "spring roll", "egg rolls",
                                 "egg roll"]},
    "wonton_soup": {"cuisine": "Chinese", "cook_method": "boil",
                    "aliases": ["wonton soup", "wonton", "won ton"]},
    "general_tso": {"cuisine": "Chinese", "cook_method": "deep_fry",
                    "aliases": ["general tso", "general tso's chicken",
                                "general tso chicken", "general tsao"]},
    "orange_chicken": {"cuisine": "Chinese", "cook_method": "deep_fry",
                       "aliases": ["orange chicken"]},
    "hot_and_sour_soup": {"cuisine": "Chinese", "cook_method": "boil",
                          "aliases": ["hot and sour soup", "hot & sour soup"]},
    "mapo_tofu": {"cuisine": "Chinese", "cook_method": "stir_fry",
                  "aliases": ["mapo tofu", "ma po tofu"]},
    "peking_duck": {"cuisine": "Chinese", "cook_method": "roast",
                    "aliases": ["peking duck", "beijing duck"]},
    "dim_sum": {"cuisine": "Chinese", "cook_method": "steam",
                "aliases": ["dim sum", "dimsum"]},
    "char_siu": {"cuisine": "Chinese", "cook_method": "roast",
                 "aliases": ["char siu", "bbq pork", "chinese bbq pork"]},
    "congee": {"cuisine": "Chinese", "cook_method": "boil",
               "aliases": ["congee", "jook", "rice porridge"]},
    "chow_fun": {"cuisine": "Chinese", "cook_method": "stir_fry",
                 "aliases": ["chow fun", "ho fun", "beef chow fun"]},
    "dan_dan_noodles": {"cuisine": "Chinese", "cook_method": "boil",
                        "aliases": ["dan dan noodles", "dan dan mian",
                                    "dandan noodles"]},
    "sweet_and_sour": {"cuisine": "Chinese", "cook_method": "deep_fry",
                       "aliases": ["sweet and sour chicken", "sweet and sour pork",
                                   "sweet & sour"]},
    "scallion_pancake": {"cuisine": "Chinese", "cook_method": "pan_fry",
                         "aliases": ["scallion pancake", "green onion pancake",
                                     "cong you bing"]},
    "xiao_long_bao": {"cuisine": "Chinese", "cook_method": "steam",
                      "aliases": ["xiao long bao", "xlb", "soup dumplings",
                                  "soup dumpling", "xiaolongbao"]},

    # ── Japanese ──
    "sushi": {"cuisine": "Japanese", "cook_method": "raw",
              "aliases": ["sushi", "nigiri", "maki", "sushi roll"]},
    "sashimi": {"cuisine": "Japanese", "cook_method": "raw",
                "aliases": ["sashimi"]},
    "ramen": {"cuisine": "Japanese", "cook_method": "boil",
              "aliases": ["ramen", "tonkotsu ramen", "shoyu ramen",
                          "miso ramen", "shio ramen"]},
    "tempura": {"cuisine": "Japanese", "cook_method": "deep_fry",
                "aliases": ["tempura", "shrimp tempura", "vegetable tempura"]},
    "teriyaki": {"cuisine": "Japanese", "cook_method": "grill",
                 "aliases": ["teriyaki", "teriyaki chicken", "teriyaki salmon"]},
    "udon": {"cuisine": "Japanese", "cook_method": "boil",
             "aliases": ["udon", "udon noodles"]},
    "katsu": {"cuisine": "Japanese", "cook_method": "deep_fry",
              "aliases": ["katsu", "tonkatsu", "chicken katsu", "katsu curry"]},
    "edamame": {"cuisine": "Japanese", "cook_method": "boil",
                "aliases": ["edamame"]},
    "miso_soup": {"cuisine": "Japanese", "cook_method": "boil",
                  "aliases": ["miso soup", "miso"]},
    "takoyaki": {"cuisine": "Japanese", "cook_method": "pan_fry",
                 "aliases": ["takoyaki", "octopus balls"]},
    "okonomiyaki": {"cuisine": "Japanese", "cook_method": "pan_fry",
                    "aliases": ["okonomiyaki", "japanese pancake"]},
    "yakitori": {"cuisine": "Japanese", "cook_method": "grill",
                 "aliases": ["yakitori", "chicken skewer"]},
    "onigiri": {"cuisine": "Japanese", "cook_method": "raw",
                "aliases": ["onigiri", "rice ball"]},
    "donburi": {"cuisine": "Japanese", "cook_method": "simmer",
                "aliases": ["donburi", "don", "gyudon", "katsudon",
                            "oyakodon", "tendon"]},

    # ── Italian ──
    "pizza_margherita": {"cuisine": "Italian", "cook_method": "bake",
                         "aliases": ["margherita pizza", "margherita"]},
    "pizza": {"cuisine": "Italian", "cook_method": "bake",
              "aliases": ["pizza", "pepperoni pizza", "cheese pizza"]},
    "pasta_carbonara": {"cuisine": "Italian", "cook_method": "boil",
                        "aliases": ["carbonara", "pasta carbonara",
                                    "spaghetti carbonara"]},
    "lasagna": {"cuisine": "Italian", "cook_method": "bake",
                "aliases": ["lasagna", "lasagne"]},
    "risotto": {"cuisine": "Italian", "cook_method": "simmer",
                "aliases": ["risotto", "mushroom risotto", "seafood risotto"]},
    "tiramisu": {"cuisine": "Italian", "cook_method": "cold",
                 "aliases": ["tiramisu"]},
    "bruschetta": {"cuisine": "Italian", "cook_method": "bake",
                   "aliases": ["bruschetta"]},
    "ravioli": {"cuisine": "Italian", "cook_method": "boil",
                "aliases": ["ravioli", "cheese ravioli"]},
    "gnocchi": {"cuisine": "Italian", "cook_method": "boil",
                "aliases": ["gnocchi"]},
    "penne_arrabbiata": {"cuisine": "Italian", "cook_method": "boil",
                         "aliases": ["arrabbiata", "penne arrabbiata"]},
    "pasta_bolognese": {"cuisine": "Italian", "cook_method": "simmer",
                        "aliases": ["bolognese", "spaghetti bolognese",
                                    "pasta bolognese", "ragu"]},
    "caprese": {"cuisine": "Italian", "cook_method": "raw",
                "aliases": ["caprese", "caprese salad"]},
    "minestrone": {"cuisine": "Italian", "cook_method": "simmer",
                   "aliases": ["minestrone", "minestrone soup"]},
    "osso_buco": {"cuisine": "Italian", "cook_method": "braise",
                  "aliases": ["osso buco", "ossobuco"]},
    "panna_cotta": {"cuisine": "Italian", "cook_method": "cold",
                    "aliases": ["panna cotta"]},
    "gelato": {"cuisine": "Italian", "cook_method": "cold",
               "aliases": ["gelato"]},

    # ── Mexican ──
    "taco": {"cuisine": "Mexican", "cook_method": "grill",
             "aliases": ["taco", "tacos", "fish taco", "fish tacos",
                         "street taco", "street tacos", "carnitas taco",
                         "al pastor taco"]},
    "burrito": {"cuisine": "Mexican", "cook_method": "grill",
                "aliases": ["burrito", "burritos", "breakfast burrito"]},
    "quesadilla": {"cuisine": "Mexican", "cook_method": "pan_fry",
                   "aliases": ["quesadilla", "quesadillas"]},
    "guacamole": {"cuisine": "Mexican", "cook_method": "raw",
                  "aliases": ["guacamole", "guac"]},
    "enchilada": {"cuisine": "Mexican", "cook_method": "bake",
                  "aliases": ["enchilada", "enchiladas"]},
    "nachos": {"cuisine": "Mexican", "cook_method": "bake",
               "aliases": ["nachos"]},
    "churro": {"cuisine": "Mexican", "cook_method": "deep_fry",
               "aliases": ["churro", "churros"]},
    "tamale": {"cuisine": "Mexican", "cook_method": "steam",
               "aliases": ["tamale", "tamales"]},
    "fajita": {"cuisine": "Mexican", "cook_method": "grill",
               "aliases": ["fajita", "fajitas", "chicken fajita", "steak fajita"]},
    "elote": {"cuisine": "Mexican", "cook_method": "grill",
              "aliases": ["elote", "mexican corn", "street corn"]},
    "chile_relleno": {"cuisine": "Mexican", "cook_method": "deep_fry",
                      "aliases": ["chile relleno", "chiles rellenos"]},
    "pozole": {"cuisine": "Mexican", "cook_method": "simmer",
               "aliases": ["pozole", "posole"]},
    "ceviche": {"cuisine": "Mexican", "cook_method": "raw",
                "aliases": ["ceviche"]},

    # ── Thai ──
    "pad_thai": {"cuisine": "Thai", "cook_method": "stir_fry",
                 "aliases": ["pad thai", "phad thai"]},
    "green_curry": {"cuisine": "Thai", "cook_method": "simmer",
                    "aliases": ["green curry", "thai green curry"]},
    "red_curry": {"cuisine": "Thai", "cook_method": "simmer",
                  "aliases": ["red curry", "thai red curry"]},
    "massaman_curry": {"cuisine": "Thai", "cook_method": "simmer",
                       "aliases": ["massaman curry", "massaman"]},
    "tom_yum": {"cuisine": "Thai", "cook_method": "boil",
                "aliases": ["tom yum", "tom yum soup", "tom yam"]},
    "tom_kha": {"cuisine": "Thai", "cook_method": "boil",
                "aliases": ["tom kha", "tom kha gai", "coconut soup"]},
    "papaya_salad": {"cuisine": "Thai", "cook_method": "raw",
                     "aliases": ["papaya salad", "som tam", "som tum"]},
    "satay": {"cuisine": "Thai", "cook_method": "grill",
              "aliases": ["satay", "chicken satay", "pork satay"]},
    "pad_see_ew": {"cuisine": "Thai", "cook_method": "stir_fry",
                   "aliases": ["pad see ew", "pad si ew"]},
    "mango_sticky_rice": {"cuisine": "Thai", "cook_method": "steam",
                          "aliases": ["mango sticky rice", "mango rice"]},
    "thai_iced_tea": {"cuisine": "Thai", "cook_method": "cold",
                      "aliases": ["thai iced tea", "thai tea"]},
    "larb": {"cuisine": "Thai", "cook_method": "raw",
             "aliases": ["larb", "laab", "thai larb"]},

    # ── Indian ──
    "tikka_masala": {"cuisine": "Indian", "cook_method": "simmer",
                     "aliases": ["tikka masala", "chicken tikka masala"]},
    "butter_chicken": {"cuisine": "Indian", "cook_method": "simmer",
                       "aliases": ["butter chicken", "murgh makhani"]},
    "biryani": {"cuisine": "Indian", "cook_method": "bake",
                "aliases": ["biryani", "chicken biryani", "lamb biryani",
                            "vegetable biryani"]},
    "naan": {"cuisine": "Indian", "cook_method": "bake",
             "aliases": ["naan", "garlic naan", "cheese naan",
                         "plain naan", "naan bread"]},
    "samosa": {"cuisine": "Indian", "cook_method": "deep_fry",
               "aliases": ["samosa", "samosas"]},
    "tandoori_chicken": {"cuisine": "Indian", "cook_method": "roast",
                         "aliases": ["tandoori chicken", "tandoori"]},
    "palak_paneer": {"cuisine": "Indian", "cook_method": "simmer",
                     "aliases": ["palak paneer", "saag paneer"]},
    "dal": {"cuisine": "Indian", "cook_method": "simmer",
            "aliases": ["dal", "daal", "dhal", "lentil dal"]},
    "vindaloo": {"cuisine": "Indian", "cook_method": "simmer",
                 "aliases": ["vindaloo", "chicken vindaloo", "lamb vindaloo"]},
    "korma": {"cuisine": "Indian", "cook_method": "simmer",
              "aliases": ["korma", "chicken korma", "lamb korma"]},
    "dosa": {"cuisine": "Indian", "cook_method": "pan_fry",
             "aliases": ["dosa", "masala dosa"]},
    "chana_masala": {"cuisine": "Indian", "cook_method": "simmer",
                     "aliases": ["chana masala", "chole"]},
    "paneer_tikka": {"cuisine": "Indian", "cook_method": "grill",
                     "aliases": ["paneer tikka"]},
    "raita": {"cuisine": "Indian", "cook_method": "raw",
              "aliases": ["raita"]},

    # ── Korean ──
    "bibimbap": {"cuisine": "Korean", "cook_method": "stir_fry",
                 "aliases": ["bibimbap", "dolsot bibimbap"]},
    "bulgogi": {"cuisine": "Korean", "cook_method": "grill",
                "aliases": ["bulgogi", "beef bulgogi"]},
    "kimchi": {"cuisine": "Korean", "cook_method": "raw",
               "aliases": ["kimchi"]},
    "japchae": {"cuisine": "Korean", "cook_method": "stir_fry",
                "aliases": ["japchae", "japchae noodles"]},
    "tteokbokki": {"cuisine": "Korean", "cook_method": "simmer",
                   "aliases": ["tteokbokki", "topokki", "rice cake"]},
    "korean_fried_chicken": {"cuisine": "Korean", "cook_method": "deep_fry",
                             "aliases": ["korean fried chicken", "kfc",
                                         "yangnyeom chicken"]},
    "kimchi_jjigae": {"cuisine": "Korean", "cook_method": "boil",
                      "aliases": ["kimchi jjigae", "kimchi stew"]},
    "samgyeopsal": {"cuisine": "Korean", "cook_method": "grill",
                    "aliases": ["samgyeopsal", "pork belly", "korean bbq"]},
    "kimbap": {"cuisine": "Korean", "cook_method": "raw",
               "aliases": ["kimbap", "gimbap"]},
    "sundubu_jjigae": {"cuisine": "Korean", "cook_method": "boil",
                       "aliases": ["sundubu", "sundubu jjigae", "soft tofu stew",
                                   "soon tofu"]},

    # ── Vietnamese ──
    "pho": {"cuisine": "Vietnamese", "cook_method": "boil",
            "aliases": ["pho", "pho bo", "pho ga", "beef pho", "chicken pho"]},
    "banh_mi": {"cuisine": "Vietnamese", "cook_method": "bake",
                "aliases": ["banh mi", "bahn mi", "vietnamese sandwich"]},
    "bun_bo_hue": {"cuisine": "Vietnamese", "cook_method": "boil",
                   "aliases": ["bun bo hue"]},
    "fresh_spring_roll": {"cuisine": "Vietnamese", "cook_method": "raw",
                          "aliases": ["fresh spring roll", "summer roll",
                                      "goi cuon", "fresh roll"]},
    "bun_cha": {"cuisine": "Vietnamese", "cook_method": "grill",
                "aliases": ["bun cha"]},
    "com_tam": {"cuisine": "Vietnamese", "cook_method": "grill",
                "aliases": ["com tam", "broken rice"]},
    "ca_phe_sua_da": {"cuisine": "Vietnamese", "cook_method": "cold",
                      "aliases": ["vietnamese coffee", "ca phe sua da",
                                  "vietnamese iced coffee"]},

    # ── Mediterranean / Middle Eastern ──
    "hummus": {"cuisine": "Mediterranean", "cook_method": "raw",
               "aliases": ["hummus", "humous", "houmous"]},
    "falafel": {"cuisine": "Mediterranean", "cook_method": "deep_fry",
                "aliases": ["falafel"]},
    "shawarma": {"cuisine": "Mediterranean", "cook_method": "roast",
                 "aliases": ["shawarma", "chicken shawarma", "beef shawarma"]},
    "kebab": {"cuisine": "Mediterranean", "cook_method": "grill",
              "aliases": ["kebab", "kebabs", "shish kebab", "doner kebab"]},
    "pita": {"cuisine": "Mediterranean", "cook_method": "bake",
             "aliases": ["pita", "pita bread"]},
    "baklava": {"cuisine": "Mediterranean", "cook_method": "bake",
                "aliases": ["baklava"]},
    "gyro": {"cuisine": "Mediterranean", "cook_method": "roast",
             "aliases": ["gyro", "gyros"]},
    "tabbouleh": {"cuisine": "Mediterranean", "cook_method": "raw",
                  "aliases": ["tabbouleh", "tabouleh", "tabouli"]},
    "baba_ganoush": {"cuisine": "Mediterranean", "cook_method": "grill",
                     "aliases": ["baba ganoush", "baba ghanoush"]},
    "moussaka": {"cuisine": "Mediterranean", "cook_method": "bake",
                 "aliases": ["moussaka"]},
    "spanakopita": {"cuisine": "Mediterranean", "cook_method": "bake",
                    "aliases": ["spanakopita", "spinach pie"]},
    "dolma": {"cuisine": "Mediterranean", "cook_method": "simmer",
              "aliases": ["dolma", "dolmades", "stuffed grape leaves"]},

    # ── French ──
    "croissant": {"cuisine": "French", "cook_method": "bake",
                  "aliases": ["croissant"]},
    "quiche": {"cuisine": "French", "cook_method": "bake",
               "aliases": ["quiche", "quiche lorraine"]},
    "crepe": {"cuisine": "French", "cook_method": "pan_fry",
              "aliases": ["crepe", "crepes"]},
    "escargot": {"cuisine": "French", "cook_method": "bake",
                 "aliases": ["escargot", "escargots", "snails"]},
    "coq_au_vin": {"cuisine": "French", "cook_method": "braise",
                   "aliases": ["coq au vin"]},
    "french_onion_soup": {"cuisine": "French", "cook_method": "simmer",
                          "aliases": ["french onion soup", "onion soup"]},
    "creme_brulee": {"cuisine": "French", "cook_method": "bake",
                     "aliases": ["creme brulee"]},
    "ratatouille": {"cuisine": "French", "cook_method": "simmer",
                    "aliases": ["ratatouille"]},
    "beef_bourguignon": {"cuisine": "French", "cook_method": "braise",
                         "aliases": ["beef bourguignon", "boeuf bourguignon"]},
    "souffle": {"cuisine": "French", "cook_method": "bake",
                "aliases": ["souffle"]},

    # ── Spanish ──
    "paella": {"cuisine": "Spanish", "cook_method": "simmer",
               "aliases": ["paella", "seafood paella"]},
    "tapas": {"cuisine": "Spanish", "cook_method": "stir_fry",
              "aliases": ["tapas"]},
    "gazpacho": {"cuisine": "Spanish", "cook_method": "cold",
                 "aliases": ["gazpacho"]},
    "patatas_bravas": {"cuisine": "Spanish", "cook_method": "deep_fry",
                       "aliases": ["patatas bravas"]},
    "churros_spanish": {"cuisine": "Spanish", "cook_method": "deep_fry",
                        "aliases": ["churros con chocolate"]},

    # ── General / Cross-cuisine ──
    "cheesecake": {"cuisine": "Other", "cook_method": "bake",
                   "aliases": ["cheesecake"]},
    "caesar_salad": {"cuisine": "Other", "cook_method": "raw",
                     "aliases": ["caesar salad", "caesar"]},
    "fish_and_chips": {"cuisine": "Other", "cook_method": "deep_fry",
                       "aliases": ["fish and chips", "fish & chips",
                                   "fish n chips"]},
    "grilled_cheese": {"cuisine": "American", "cook_method": "pan_fry",
                       "aliases": ["grilled cheese", "grilled cheese sandwich"]},
    "club_sandwich": {"cuisine": "American", "cook_method": "raw",
                      "aliases": ["club sandwich"]},
    "chicken_sandwich": {"cuisine": "American", "cook_method": "deep_fry",
                         "aliases": ["chicken sandwich", "fried chicken sandwich"]},
    "fried_chicken": {"cuisine": "American", "cook_method": "deep_fry",
                      "aliases": ["fried chicken"]},
    "chicken_salad": {"cuisine": "Other", "cook_method": "raw",
                      "aliases": ["chicken salad"]},
    "soup": {"cuisine": "Other", "cook_method": "simmer",
             "aliases": ["soup"]},
    "brownie": {"cuisine": "Other", "cook_method": "bake",
                "aliases": ["brownie", "brownies"]},
    "ice_cream": {"cuisine": "Other", "cook_method": "cold",
                  "aliases": ["ice cream"]},
}


def build_alias_index(dish_dict: dict) -> dict:
    """Build a reverse index: alias (lowercased) -> dish_id."""
    index = {}
    for dish_id, info in dish_dict.items():
        for alias in info["aliases"]:
            index[alias.lower()] = dish_id
    return index


def extract_context_sentences(text: str, dish_alias: str, window: int = 2) -> str:
    """Extract sentences around the dish mention.

    Args:
        text: Full review text
        dish_alias: The matched dish alias
        window: Number of sentences before/after to include

    Returns:
        Context string with surrounding sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    target_indices = []
    alias_lower = dish_alias.lower()

    for i, sent in enumerate(sentences):
        if alias_lower in sent.lower():
            target_indices.append(i)

    if not target_indices:
        return text[:500]  # fallback

    # Expand window around each mention
    include = set()
    for idx in target_indices:
        for j in range(max(0, idx - window), min(len(sentences), idx + window + 1)):
            include.add(j)

    context = " ".join(sentences[i] for i in sorted(include))
    return context[:1000]  # cap at 1000 chars


def match_dishes_in_reviews(
    reviews: pd.DataFrame,
    alias_index: dict,
    dish_dict: dict,
) -> pd.DataFrame:
    """Match dish names in review texts using keyword matching.

    Strategy:
      - Sort aliases by length (longest first) to prefer specific matches
      - Case-insensitive matching
      - Extract context sentences around each mention

    Returns:
        DataFrame with columns: review_id, dish_id, dish_alias_matched,
        cuisine, cook_method, context_text, stars, business_id
    """
    # Sort aliases longest-first for greedy matching
    sorted_aliases = sorted(alias_index.keys(), key=len, reverse=True)

    # Pre-compile patterns for efficiency
    patterns = {}
    for alias in sorted_aliases:
        # Word-boundary matching to avoid partial matches
        # e.g., "rice" shouldn't match "price"
        pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
        patterns[alias] = pattern

    records = []
    reviews_with_match = 0

    for _, row in tqdm(reviews.iterrows(), total=len(reviews),
                       desc="Matching dishes"):
        text = row["text"]
        text_lower = text.lower()
        matched_dishes = set()  # avoid duplicate dish_ids per review

        for alias in sorted_aliases:
            if alias in text_lower:
                # Verify with word boundary
                if patterns[alias].search(text):
                    dish_id = alias_index[alias]
                    if dish_id not in matched_dishes:
                        matched_dishes.add(dish_id)
                        info = dish_dict[dish_id]
                        context = extract_context_sentences(text, alias)
                        records.append({
                            "review_id": row["review_id"],
                            "business_id": row["business_id"],
                            "dish_id": dish_id,
                            "dish_alias_matched": alias,
                            "cuisine": info["cuisine"],
                            "cook_method": info["cook_method"],
                            "context_text": context,
                            "stars": row["stars"],
                        })

        if matched_dishes:
            reviews_with_match += 1

    print(f"\n  Reviews with at least one dish match: "
          f"{reviews_with_match:,} / {len(reviews):,} "
          f"({reviews_with_match/len(reviews)*100:.1f}%)")

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("DEI Project - Dish Extraction from Reviews")
    print("=" * 60)

    # 1. Load data
    print("\nLoading restaurant reviews...")
    reviews = pd.read_parquet(RESTAURANT_REVIEWS_PARQUET)
    print(f"  Total reviews: {len(reviews):,}")

    # 2. Filter: minimum word count
    reviews["word_count"] = reviews["text"].str.split().str.len()
    reviews = reviews[reviews["word_count"] >= MIN_REVIEW_WORDS].copy()
    print(f"  After word count filter (>={MIN_REVIEW_WORDS}): {len(reviews):,}")

    # 3. Build alias index
    alias_index = build_alias_index(DISH_DICTIONARY)
    print(f"\n  Dish dictionary: {len(DISH_DICTIONARY)} dishes, "
          f"{len(alias_index)} aliases")

    # 4. Match dishes
    print("\nMatching dishes in reviews...")
    dish_mentions = match_dishes_in_reviews(reviews, alias_index, DISH_DICTIONARY)

    if len(dish_mentions) == 0:
        print("  WARNING: No dish mentions found!")
        return

    # 5. Statistics
    print(f"\n  Total dish mentions: {len(dish_mentions):,}")
    print(f"  Unique dishes mentioned: {dish_mentions['dish_id'].nunique()}")

    dish_counts = dish_mentions["dish_id"].value_counts()
    print(f"\n  Top 30 most mentioned dishes:")
    for dish, count in dish_counts.head(30).items():
        cuisine = DISH_DICTIONARY[dish]["cuisine"]
        print(f"    {dish:30s} {cuisine:15s} {count:6d}")

    cuisine_counts = dish_mentions["cuisine"].value_counts()
    print(f"\n  Mentions by cuisine:")
    for cuisine, count in cuisine_counts.items():
        print(f"    {cuisine:20s} {count:6d}")

    cook_method_counts = dish_mentions["cook_method"].value_counts()
    print(f"\n  Mentions by cooking method:")
    for method, count in cook_method_counts.items():
        print(f"    {method:20s} {count:6d}")

    # 6. Filter dishes with enough mentions
    sufficient = dish_counts[dish_counts >= 10]
    print(f"\n  Dishes with >= 10 mentions: {len(sufficient)}")

    # 7. Save
    dish_mentions.to_parquet(DATA_DIR / "dish_mentions.parquet", index=False)
    print(f"\n  Saved: {DATA_DIR / 'dish_mentions.parquet'}")

    # Save dish dictionary as reference
    dish_dict_records = []
    for dish_id, info in DISH_DICTIONARY.items():
        dish_dict_records.append({
            "dish_id": dish_id,
            "cuisine": info["cuisine"],
            "cook_method": info["cook_method"],
            "aliases": "|".join(info["aliases"]),
            "mention_count": dish_counts.get(dish_id, 0),
        })
    dish_dict_df = pd.DataFrame(dish_dict_records)
    dish_dict_df = dish_dict_df.sort_values("mention_count", ascending=False)
    dish_dict_df.to_csv(DATA_DIR / "dish_dictionary.csv", index=False)
    print(f"  Saved: {DATA_DIR / 'dish_dictionary.csv'}")

    # Save summary stats
    summary = {
        "total_reviews_processed": len(reviews),
        "total_dish_mentions": len(dish_mentions),
        "unique_dishes_found": dish_mentions["dish_id"].nunique(),
        "dishes_with_10plus_mentions": len(sufficient),
        "pct_reviews_with_dish": (
            dish_mentions["review_id"].nunique() / len(reviews) * 100
        ),
    }
    pd.DataFrame([summary]).T.to_csv(TABLES_DIR / "dish_extraction_summary.csv",
                                     header=["value"])
    print(f"  Saved: {TABLES_DIR / 'dish_extraction_summary.csv'}")

    print("\n" + "=" * 60)
    print("Dish extraction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
