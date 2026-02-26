"""
01_explore_yelp.py — Yelp Open Dataset 探索性分析
=================================================
目标：
  1. 加载 business.json, review.json
  2. 筛选餐饮类商家
  3. 按菜系分类
  4. 统计评论中菜品提及的可行性
  5. 输出基础统计数据，指导后续管线设计

产出：
  - data/restaurants.parquet        筛选后的餐饮商家
  - results/tables/yelp_summary.csv 汇总统计
  - results/figures/yelp_eda_*.png  EDA 图表
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ── Setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    YELP_BUSINESS,
    YELP_REVIEW,
    CUISINE_KEYWORDS,
    DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MIN_REVIEW_WORDS,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── 1. Load business data ────────────────────────────────────────────
def load_businesses() -> pd.DataFrame:
    """Load Yelp business JSON (line-delimited)."""
    print(f"Loading business data from {YELP_BUSINESS} ...")
    records = []
    with open(YELP_BUSINESS, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="businesses"):
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"  Total businesses: {len(df):,}")
    return df


def filter_restaurants(biz: pd.DataFrame) -> pd.DataFrame:
    """Filter to food/restaurant businesses."""
    mask = biz["categories"].fillna("").str.contains(
        r"Restaurant|Food|Café|Bakery|Bar|Pub|Diner|Bistro|Brasserie|"
        r"Pizzeria|Steakhouse|Sushi|Ramen|Taco|Burger|BBQ|Grill|"
        r"Coffee|Tea|Dessert|Ice Cream|Juice|Smoothie",
        case=False,
        regex=True,
    )
    rest = biz[mask].copy()
    print(f"  Restaurants / food businesses: {len(rest):,}")
    return rest


def assign_cuisine(categories: str) -> str:
    """Assign primary cuisine based on category string."""
    if pd.isna(categories):
        return "Other"
    cats_upper = categories.upper()
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in cats_upper:
                return cuisine
    return "Other"


# ── 2. Load review data (streaming, memory-efficient) ─────────────────
def load_restaurant_reviews(restaurant_ids: set) -> pd.DataFrame:
    """Load reviews belonging to restaurant business IDs.

    Uses streaming to avoid loading all 7M reviews into memory at once.
    """
    print(f"\nLoading reviews from {YELP_REVIEW} ...")
    print(f"  Filtering to {len(restaurant_ids):,} restaurant IDs")

    records = []
    total = 0
    with open(YELP_REVIEW, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="reviews"):
            total += 1
            obj = json.loads(line)
            if obj["business_id"] in restaurant_ids:
                records.append({
                    "review_id": obj["review_id"],
                    "business_id": obj["business_id"],
                    "user_id": obj["user_id"],
                    "stars": obj["stars"],
                    "date": obj["date"],
                    "text": obj["text"],
                    "useful": obj.get("useful", 0),
                    "funny": obj.get("funny", 0),
                    "cool": obj.get("cool", 0),
                })

    df = pd.DataFrame(records)
    print(f"  Total reviews scanned: {total:,}")
    print(f"  Restaurant reviews kept: {len(df):,}")
    return df


# ── 3. Analysis functions ────────────────────────────────────────────
def cuisine_distribution(restaurants: pd.DataFrame) -> pd.DataFrame:
    """Count restaurants and reviews per cuisine."""
    stats = (
        restaurants.groupby("cuisine")
        .agg(
            n_restaurants=("business_id", "count"),
            avg_stars=("stars", "mean"),
            total_reviews=("review_count", "sum"),
        )
        .sort_values("n_restaurants", ascending=False)
    )
    return stats


def review_length_stats(reviews: pd.DataFrame) -> dict:
    """Compute review length statistics."""
    reviews = reviews.copy()
    reviews["word_count"] = reviews["text"].str.split().str.len()
    return {
        "total_reviews": len(reviews),
        "mean_words": reviews["word_count"].mean(),
        "median_words": reviews["word_count"].median(),
        "pct_above_threshold": (
            (reviews["word_count"] >= MIN_REVIEW_WORDS).mean() * 100
        ),
        "stars_distribution": reviews["stars"].value_counts().sort_index().to_dict(),
    }


def estimate_dish_mentions(reviews: pd.DataFrame, n_sample: int = 5000) -> dict:
    """Estimate how many reviews mention specific dish names.

    Uses a lightweight heuristic: looks for common dish-name patterns
    in a random sample of reviews.
    """
    # Common dish names for quick estimation
    common_dishes = [
        # American
        "burger", "fries", "steak", "wings", "mac and cheese", "salad",
        "sandwich", "ribs", "pulled pork", "coleslaw",
        # Chinese
        "kung pao", "fried rice", "lo mein", "chow mein", "dumplings",
        "spring rolls", "egg rolls", "wonton", "general tso",
        "orange chicken", "hot and sour soup", "mapo tofu",
        # Italian
        "pizza", "pasta", "lasagna", "risotto", "tiramisu", "bruschetta",
        "ravioli", "gnocchi", "carbonara", "margherita",
        # Japanese
        "sushi", "ramen", "sashimi", "tempura", "miso soup", "gyoza",
        "udon", "edamame", "teriyaki", "katsu",
        # Mexican
        "taco", "burrito", "quesadilla", "guacamole", "enchilada",
        "nachos", "churro", "tamale", "fajita",
        # Thai
        "pad thai", "green curry", "red curry", "tom yum", "thai iced tea",
        "papaya salad", "satay", "massaman",
        # Indian
        "naan", "tikka masala", "biryani", "samosa", "tandoori",
        "butter chicken", "palak paneer", "dal", "vindaloo",
        # Korean
        "bibimbap", "kimchi", "bulgogi", "japchae", "tteokbokki",
        # Mediterranean
        "hummus", "falafel", "shawarma", "kebab", "pita", "baklava",
        "gyro", "tabouleh",
        # Vietnamese
        "pho", "banh mi", "spring roll",
        # French
        "croissant", "quiche", "crepe", "soufflé", "escargot",
        # General
        "soup", "cheesecake", "pancake", "waffle", "omelet",
    ]

    sample = reviews.sample(n=min(n_sample, len(reviews)), random_state=42)

    dish_counts = Counter()
    reviews_with_dish = 0

    for text in sample["text"].str.lower():
        found_any = False
        for dish in common_dishes:
            if dish in text:
                dish_counts[dish] += 1
                found_any = True
        if found_any:
            reviews_with_dish += 1

    return {
        "sample_size": len(sample),
        "reviews_with_dish_mention": reviews_with_dish,
        "pct_with_dish": reviews_with_dish / len(sample) * 100,
        "top_dishes": dish_counts.most_common(30),
        "estimated_total_with_dish": int(
            reviews_with_dish / len(sample) * len(reviews)
        ),
    }


# ── 4. Visualization ─────────────────────────────────────────────────
def plot_cuisine_distribution(cuisine_stats: pd.DataFrame):
    """Bar chart of restaurants per cuisine."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top = cuisine_stats.head(15)

    axes[0].barh(top.index[::-1], top["n_restaurants"][::-1])
    axes[0].set_xlabel("Number of Restaurants")
    axes[0].set_title("Restaurant Count by Cuisine")

    axes[1].barh(top.index[::-1], top["total_reviews"][::-1])
    axes[1].set_xlabel("Total Reviews")
    axes[1].set_title("Review Count by Cuisine")

    plt.tight_layout()
    path = FIGURES_DIR / "yelp_eda_cuisine_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_star_distribution(reviews: pd.DataFrame):
    """Star rating distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    reviews["stars"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Review Star Distribution (Restaurant Reviews)")
    plt.tight_layout()
    path = FIGURES_DIR / "yelp_eda_star_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_review_length(reviews: pd.DataFrame):
    """Review word-count distribution."""
    wc = reviews["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(wc.clip(upper=500), bins=100, edgecolor="white", alpha=0.8)
    ax.axvline(MIN_REVIEW_WORDS, color="red", linestyle="--",
               label=f"Threshold = {MIN_REVIEW_WORDS} words")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Review Length Distribution")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "yelp_eda_review_length.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DEI Project — Yelp Open Dataset Exploration")
    print("=" * 60)

    # 1. Business data
    biz = load_businesses()
    restaurants = filter_restaurants(biz)
    restaurants["cuisine"] = restaurants["categories"].apply(assign_cuisine)

    # Save filtered restaurants
    restaurants.to_parquet(DATA_DIR / "restaurants.parquet", index=False)
    print(f"\n  Saved: {DATA_DIR / 'restaurants.parquet'}")

    # 2. Cuisine stats
    print("\n── Cuisine Distribution ──")
    c_stats = cuisine_distribution(restaurants)
    print(c_stats.to_string())
    c_stats.to_csv(TABLES_DIR / "cuisine_distribution.csv")

    # 3. Load reviews
    rest_ids = set(restaurants["business_id"])
    reviews = load_restaurant_reviews(rest_ids)

    # Save as parquet for fast subsequent access
    reviews.to_parquet(DATA_DIR / "restaurant_reviews.parquet", index=False)
    print(f"  Saved: {DATA_DIR / 'restaurant_reviews.parquet'}")

    # 4. Review stats
    print("\n── Review Statistics ──")
    r_stats = review_length_stats(reviews)
    for k, v in r_stats.items():
        if k != "stars_distribution":
            print(f"  {k}: {v}")
    print(f"  Stars distribution: {r_stats['stars_distribution']}")

    # 5. Dish mention estimation
    print("\n── Dish Mention Estimation ──")
    d_stats = estimate_dish_mentions(reviews)
    print(f"  Sample size: {d_stats['sample_size']:,}")
    print(f"  Reviews with dish mention: {d_stats['reviews_with_dish_mention']:,} "
          f"({d_stats['pct_with_dish']:.1f}%)")
    print(f"  Estimated total reviews with dish mentions: "
          f"{d_stats['estimated_total_with_dish']:,}")
    print(f"\n  Top 20 most mentioned dishes:")
    for dish, count in d_stats["top_dishes"][:20]:
        pct = count / d_stats["sample_size"] * 100
        print(f"    {dish:25s}  {count:5d}  ({pct:.1f}%)")

    # 6. Summary table
    summary = {
        "total_businesses": len(biz),
        "restaurant_businesses": len(restaurants),
        "cuisines_identified": len(c_stats[c_stats.index != "Other"]),
        "total_restaurant_reviews": len(reviews),
        "mean_review_words": round(r_stats["mean_words"], 1),
        "pct_reviews_above_threshold": round(r_stats["pct_above_threshold"], 1),
        "pct_reviews_with_dish_mention": round(d_stats["pct_with_dish"], 1),
        "estimated_usable_reviews": d_stats["estimated_total_with_dish"],
    }
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ["value"]
    summary_df.to_csv(TABLES_DIR / "yelp_summary.csv")
    print(f"\n  Saved summary: {TABLES_DIR / 'yelp_summary.csv'}")

    # 7. Plots
    print("\n── Generating Plots ──")
    plot_cuisine_distribution(c_stats)
    plot_star_distribution(reviews)
    plot_review_length(reviews)

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
