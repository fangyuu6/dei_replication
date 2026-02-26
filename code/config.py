"""
Project configuration — Deliciousness Efficiency Index (DEI)
All paths, constants, and parameters centralised here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
YELP_DIR = RAW_DIR / "yelp"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Yelp file paths ───────────────────────────────────────────────────
YELP_BUSINESS = YELP_DIR / "yelp_academic_dataset_business.json"
YELP_REVIEW = YELP_DIR / "yelp_academic_dataset_review.json"
YELP_USER = YELP_DIR / "yelp_academic_dataset_user.json"
YELP_TIP = YELP_DIR / "yelp_academic_dataset_tip.json"
YELP_CHECKIN = YELP_DIR / "yelp_academic_dataset_checkin.json"

# ── Intermediate data paths ───────────────────────────────────────────
RESTAURANTS_PARQUET = DATA_DIR / "restaurants.parquet"
RESTAURANT_REVIEWS_PARQUET = DATA_DIR / "restaurant_reviews.parquet"
DISH_REVIEWS_PARQUET = DATA_DIR / "dish_reviews.parquet"
DISH_HEDONIC_CSV = DATA_DIR / "dish_hedonic_scores.csv"
DISH_ENV_CSV = DATA_DIR / "dish_environmental_costs.csv"
DISH_DEI_CSV = DATA_DIR / "dish_DEI_scores.csv"
INGREDIENT_IMPACT_CSV = DATA_DIR / "ingredient_impact_factors.csv"
COOKING_ENERGY_CSV = DATA_DIR / "cooking_method_energy.csv"
DISH_RECIPES_PARQUET = DATA_DIR / "dish_recipes.parquet"

# ── Cuisine classification ────────────────────────────────────────────
CUISINE_KEYWORDS = {
    "Chinese": ["Chinese", "Dim Sum", "Szechuan", "Cantonese", "Hunan",
                "Shanghainese", "Taiwanese", "Hot Pot"],
    "Japanese": ["Japanese", "Sushi", "Ramen", "Izakaya", "Udon",
                 "Tempura", "Yakitori", "Tonkatsu"],
    "Italian": ["Italian", "Pizza", "Pasta", "Gelato", "Trattoria"],
    "Mexican": ["Mexican", "Tacos", "Burrito", "Tex-Mex", "Taqueria"],
    "Indian": ["Indian", "Curry", "Tandoori", "Biryani", "Dosa",
               "Pakistani", "Nepali"],
    "French": ["French", "Bistro", "Brasserie", "Crêpes"],
    "American": ["American", "Burger", "BBQ", "Steakhouse",
                 "Southern", "Soul Food", "Cajun"],
    "Thai": ["Thai", "Pad Thai", "Tom Yum"],
    "Korean": ["Korean", "Korean BBQ", "Bibimbap"],
    "Mediterranean": ["Mediterranean", "Greek", "Turkish", "Lebanese",
                      "Middle Eastern", "Falafel"],
    "Vietnamese": ["Vietnamese", "Pho", "Banh Mi"],
    "Spanish": ["Spanish", "Tapas", "Paella"],
}

# ── NLP parameters ────────────────────────────────────────────────────
MIN_REVIEW_WORDS = 20          # Minimum word count for review inclusion
MIN_REVIEWS_PER_DISH = 10      # Minimum reviews to include a dish
HEDONIC_SCALE = (1, 10)        # Hedonic score range
NLP_BATCH_SIZE = 32            # Batch size for model inference

# ── Environmental parameters ──────────────────────────────────────────
# Global average grid emission factor (kg CO2 / kWh)
# Source: IEA 2023 — used for cooking energy → CO2 conversion
GRID_EMISSION_FACTOR = 0.475

COOKING_ENERGY_KWH = {
    "raw":       0.0,
    "cold":      0.0,
    "steam":     0.3,
    "boil":      0.4,
    "simmer":    0.4,
    "stir_fry":  0.5,
    "saute":     0.5,
    "pan_fry":   0.5,
    "grill":     0.6,
    "bake":      0.8,
    "roast":     0.8,
    "deep_fry":  1.0,
    "braise":    1.2,
    "slow_cook": 1.2,
    "smoke":     1.5,
}

# ── DEI parameters ────────────────────────────────────────────────────
# Weighting schemes for sensitivity analysis
E_WEIGHT_SCHEMES = {
    "equal":        {"carbon": 1/3, "water": 1/3, "energy": 1/3},
    "carbon_first":  {"carbon": 0.50, "water": 0.25, "energy": 0.25},
    "water_first":   {"carbon": 0.25, "water": 0.50, "energy": 0.25},
    "energy_first":  {"carbon": 0.25, "water": 0.25, "energy": 0.50},
}

DEI_TIERS = ["E", "D", "C", "B", "A"]   # quintile labels, low → high
