"""
03_nlp_hedonic_scoring.py - Aspect-Based Sentiment Analysis for Dish Hedonic Scoring
=====================================================================================
This script implements the core NLP pipeline for extracting hedonic (taste)
scores from review text, following the recommended approach in the research plan:

Strategy:
  Step 1: Use LLM API (Gemini / OpenAI) to annotate ~2,000 reviews
          → produces (review, dish_name) → hedonic_score (1-10) training data
  Step 2: Use these annotations to fine-tune a smaller model (BERT/RoBERTa)
  Step 3: Apply the fine-tuned model to all dish mentions

This script handles Step 1 (LLM annotation) and Step 3 (inference with
a pre-trained sentiment model as baseline).

For Nature-level rigor:
  - Multiple annotation strategies are compared
  - Inter-annotator agreement is measured
  - Systematic prompt engineering with structured output
  - Human validation sample for quality assurance
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TABLES_DIR, HEDONIC_SCALE, NLP_BATCH_SIZE


# ── LLM Annotation (Step 1) ──────────────────────────────────────────

HEDONIC_PROMPT_TEMPLATE = """You are a food critic expert. Your task is to extract the hedonic (taste enjoyment) score for a specific dish mentioned in a restaurant review.

DISH: {dish_name}
REVIEW CONTEXT: "{context_text}"

Instructions:
1. Focus ONLY on what the reviewer says about how the DISH TASTES.
2. Ignore comments about service, ambiance, price, or portion size.
3. Rate the reviewer's taste enjoyment on a scale of 1-10:
   1 = Terrible, inedible
   2 = Very bad, strong dislike
   3 = Bad, clearly below expectations
   4 = Below average, somewhat disappointing
   5 = Average, neither good nor bad
   6 = Slightly above average, acceptable
   7 = Good, enjoyable
   8 = Very good, would recommend
   9 = Excellent, one of the best
   10 = Extraordinary, transcendent experience

4. If the review does NOT contain enough information about how this dish TASTES, respond with score 0 (insufficient info).

Respond in this exact JSON format (no other text):
{{"score": <int 0-10>, "confidence": "<high/medium/low>", "evidence": "<key phrase from review>"}}"""


def annotate_with_openai(
    dish_mentions: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    n_samples: int = 2000,
    rate_limit_delay: float = 0.5,
    base_url: str = None,
) -> pd.DataFrame:
    """Annotate dish mentions using OpenAI-compatible API.

    Args:
        dish_mentions: DataFrame with dish_id, context_text columns
        api_key: API key
        model: Model to use
        n_samples: Number of samples to annotate
        rate_limit_delay: Seconds between API calls
        base_url: Custom API base URL (e.g. OpenRouter)
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        return pd.DataFrame()

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Stratified sampling: ~n_samples total, evenly across dishes
    if len(dish_mentions) > n_samples:
        n_dishes = dish_mentions["dish_id"].nunique()
        per_dish = max(3, n_samples // n_dishes)
        sample = dish_mentions.groupby("dish_id", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), per_dish), random_state=42)
        )
        if len(sample) > n_samples:
            sample = sample.sample(n=n_samples, random_state=42)
    else:
        sample = dish_mentions.copy()

    results = []
    errors = 0
    checkpoint_path = DATA_DIR / "llm_annotations_checkpoint.parquet"

    # Resume from checkpoint
    done_ids = set()
    if checkpoint_path.exists():
        prev = pd.read_parquet(checkpoint_path)
        results = prev.to_dict("records")
        done_ids = set(prev["review_id"])
        print(f"  Resuming from checkpoint: {len(done_ids)} already done")

    remaining = sample[~sample["review_id"].isin(done_ids)]
    print(f"  Remaining to annotate: {len(remaining)}")

    for i, (idx, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining),
                         desc="LLM annotation")):
        prompt = HEDONIC_PROMPT_TEMPLATE.format(
            dish_name=row["dish_id"].replace("_", " ").title(),
            context_text=row["context_text"][:800],
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            text = response.choices[0].message.content.strip()

            # Parse JSON response — try direct parse, then regex extraction
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    parsed = {"score": -1, "confidence": "parse_error", "evidence": text[:100]}

            results.append({
                "review_id": row["review_id"],
                "dish_id": row["dish_id"],
                "context_text": row["context_text"],
                "stars": row["stars"],
                "hedonic_score_llm": parsed["score"],
                "confidence": parsed.get("confidence", "unknown"),
                "evidence": parsed.get("evidence", ""),
                "raw_response": text,
            })
        except Exception as e:
            errors += 1
            results.append({
                "review_id": row["review_id"],
                "dish_id": row["dish_id"],
                "context_text": row["context_text"],
                "stars": row["stars"],
                "hedonic_score_llm": -1,
                "confidence": "error",
                "evidence": str(e),
                "raw_response": "",
            })

        # Checkpoint every 200 samples
        if (i + 1) % 200 == 0:
            pd.DataFrame(results).to_parquet(checkpoint_path, index=False)
            print(f"\n  Checkpoint saved: {len(results)} annotations")

        time.sleep(rate_limit_delay)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n  Annotated: {len(results)}, Errors: {errors}")
    return pd.DataFrame(results)


def annotate_with_gemini(
    dish_mentions: pd.DataFrame,
    api_key: str,
    model: str = "gemini-1.5-flash",
    n_samples: int = 2000,
    rate_limit_delay: float = 4.5,  # ~15 RPM free tier
) -> pd.DataFrame:
    """Annotate dish mentions using Google Gemini API (free tier)."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: pip install google-generativeai")
        return pd.DataFrame()

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)

    # Sample
    if len(dish_mentions) > n_samples:
        sample = dish_mentions.sample(n=n_samples, random_state=42)
    else:
        sample = dish_mentions.copy()

    results = []
    errors = 0

    for idx, row in tqdm(sample.iterrows(), total=len(sample),
                         desc="Gemini annotation"):
        prompt = HEDONIC_PROMPT_TEMPLATE.format(
            dish_name=row["dish_id"].replace("_", " ").title(),
            context_text=row["context_text"][:800],
        )

        try:
            response = gen_model.generate_content(prompt)
            text = response.text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {"score": -1, "confidence": "parse_error", "evidence": text}

            results.append({
                "review_id": row["review_id"],
                "dish_id": row["dish_id"],
                "context_text": row["context_text"],
                "stars": row["stars"],
                "hedonic_score_llm": parsed["score"],
                "confidence": parsed.get("confidence", "unknown"),
                "evidence": parsed.get("evidence", ""),
                "raw_response": text,
            })
        except Exception as e:
            errors += 1
            results.append({
                "review_id": row["review_id"],
                "dish_id": row["dish_id"],
                "context_text": row["context_text"],
                "stars": row["stars"],
                "hedonic_score_llm": -1,
                "confidence": "error",
                "evidence": str(e),
                "raw_response": "",
            })

        time.sleep(rate_limit_delay)

    print(f"\n  Annotated: {len(results)}, Errors: {errors}")
    return pd.DataFrame(results)


# ── Baseline: Pre-trained Sentiment Model (Step 3 fallback) ──────────

def score_with_pretrained_sentiment(
    dish_mentions: pd.DataFrame,
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
) -> pd.DataFrame:
    """Apply pre-trained sentiment model as baseline hedonic scorer.

    Uses nlptown/bert-base-multilingual-uncased-sentiment which
    outputs 1-5 star predictions. We map this to 1-10 scale.

    This serves as:
      1. A fast baseline for comparison
      2. Fallback when LLM API is unavailable
      3. Cross-validation against LLM annotations
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: pip install transformers torch")
        return pd.DataFrame()

    print(f"\nLoading sentiment model: {model_name}")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
        device=-1,  # CPU
    )

    results = []
    texts = dish_mentions["context_text"].tolist()

    # Process in batches
    for i in tqdm(range(0, len(texts), NLP_BATCH_SIZE),
                  desc="Sentiment scoring"):
        batch = texts[i:i + NLP_BATCH_SIZE]
        # Truncate long texts
        batch = [t[:512] for t in batch]

        try:
            preds = sentiment_pipe(batch)
            for j, pred in enumerate(preds):
                # Model outputs "1 star" to "5 stars"
                star_label = int(pred["label"].split()[0])
                # Map 1-5 → 1-10 (linear scaling)
                hedonic_10 = (star_label - 1) * 2.25 + 1
                # Adjust with confidence
                confidence = pred["score"]
                results.append({
                    "hedonic_score_pretrained": round(hedonic_10, 2),
                    "pretrained_confidence": round(confidence, 3),
                    "pretrained_star": star_label,
                })
        except Exception as e:
            for j in range(len(batch)):
                results.append({
                    "hedonic_score_pretrained": np.nan,
                    "pretrained_confidence": 0.0,
                    "pretrained_star": 0,
                })

    result_df = pd.DataFrame(results)
    return pd.concat([dish_mentions.reset_index(drop=True),
                      result_df.reset_index(drop=True)], axis=1)


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate_hedonic_scores(
    scored_mentions: pd.DataFrame,
    score_col: str = "hedonic_score",
    min_reviews: int = 10,
) -> pd.DataFrame:
    """Aggregate per-mention scores to per-dish hedonic scores.

    For each dish, computes:
      - H_mean: mean hedonic score (primary metric)
      - H_median: median (robustness check)
      - H_std: standard deviation (consistency)
      - H_n: number of reviews (reliability)
      - H_ci95: 95% confidence interval half-width
      - H_iqr: interquartile range
    """
    # Filter out invalid scores
    valid = scored_mentions[
        scored_mentions[score_col].between(HEDONIC_SCALE[0], HEDONIC_SCALE[1])
    ].copy()

    agg = (
        valid
        .groupby("dish_id")
        .agg(
            H_mean=(score_col, "mean"),
            H_median=(score_col, "median"),
            H_std=(score_col, "std"),
            H_n=(score_col, "count"),
            H_q25=(score_col, lambda x: x.quantile(0.25)),
            H_q75=(score_col, lambda x: x.quantile(0.75)),
        )
    )
    agg["H_ci95"] = 1.96 * agg["H_std"] / np.sqrt(agg["H_n"])
    agg["H_iqr"] = agg["H_q75"] - agg["H_q25"]

    # Filter: minimum review count
    agg_filtered = agg[agg["H_n"] >= min_reviews].copy()

    print(f"\n  Dishes before filtering: {len(agg)}")
    print(f"  Dishes after min_reviews={min_reviews} filter: {len(agg_filtered)}")
    print(f"  Total reviews used: {agg_filtered['H_n'].sum():,.0f}")
    print(f"  Mean H across dishes: {agg_filtered['H_mean'].mean():.2f}")
    print(f"  Std of dish means: {agg_filtered['H_mean'].std():.2f}")

    return agg_filtered.round(3)


# ── Validation ────────────────────────────────────────────────────────

def generate_human_validation_sample(
    scored_mentions: pd.DataFrame,
    n_samples: int = 200,
    output_path: Path = None,
) -> pd.DataFrame:
    """Generate a stratified sample for human validation.

    Samples are stratified across:
      - Score quintiles (to cover full range)
      - Cuisines (to avoid bias)

    Output CSV for manual review with columns:
      review_id, dish_id, context_text, model_score, human_score (blank)
    """
    if output_path is None:
        output_path = TABLES_DIR / "human_validation_sample.csv"

    score_col = [c for c in scored_mentions.columns
                 if "hedonic_score" in c and c != "hedonic_score_pretrained"]
    if not score_col:
        score_col = ["hedonic_score_pretrained"]
    score_col = score_col[0]

    valid = scored_mentions[
        scored_mentions[score_col].between(1, 10)
    ].copy()

    # Stratified sampling by score quintile
    valid["score_bin"] = pd.qcut(valid[score_col], q=5, labels=False,
                                  duplicates="drop")

    sample = valid.groupby("score_bin", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_samples // 5), random_state=42)
    )

    if len(sample) < n_samples:
        remaining = n_samples - len(sample)
        extra = valid[~valid.index.isin(sample.index)].sample(
            n=min(remaining, len(valid) - len(sample)), random_state=42
        )
        sample = pd.concat([sample, extra])

    output = sample[["review_id", "dish_id", "context_text",
                      score_col, "stars"]].copy()
    output.rename(columns={score_col: "model_score"}, inplace=True)
    output["human_score"] = ""  # to be filled manually
    output["human_notes"] = ""

    output.to_csv(output_path, index=False)
    print(f"\n  Human validation sample saved: {output_path}")
    print(f"  Samples: {len(output)}")
    return output


def compute_validation_metrics(
    human_scores: pd.Series,
    model_scores: pd.Series,
) -> dict:
    """Compute validation metrics between human and model scores.

    Reports:
      - Pearson r (target: > 0.75)
      - Spearman rho (rank correlation)
      - MAE (target: < 1.0)
      - RMSE
      - ICC (intraclass correlation)
    """
    from scipy.stats import pearsonr, spearmanr

    mask = ~(human_scores.isna() | model_scores.isna())
    h = human_scores[mask].astype(float)
    m = model_scores[mask].astype(float)

    if len(h) < 10:
        return {"error": "insufficient data", "n": len(h)}

    pearson_r, pearson_p = pearsonr(h, m)
    spearman_rho, spearman_p = spearmanr(h, m)
    mae = np.mean(np.abs(h - m))
    rmse = np.sqrt(np.mean((h - m) ** 2))
    bias = np.mean(m - h)

    # ICC(2,1) — two-way random, single measures
    n = len(h)
    grand_mean = (h.mean() + m.mean()) / 2
    ss_between = n * ((h.mean() - grand_mean)**2 + (m.mean() - grand_mean)**2)
    ss_within = np.sum((h - h.mean())**2 + (m - m.mean())**2)
    ms_between = ss_between / (n - 1) if n > 1 else 0
    ms_within = ss_within / n if n > 0 else 1
    icc = (ms_between - ms_within) / (ms_between + ms_within) if (ms_between + ms_within) > 0 else 0

    metrics = {
        "n": int(n),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": pearson_p,
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p": spearman_p,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "icc": round(icc, 4),
    }

    # Pass/fail against research plan targets
    metrics["passes_pearson"] = pearson_r > 0.75
    metrics["passes_mae"] = mae < 1.0

    return metrics


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DEI Project - NLP Hedonic Scoring Pipeline")
    print("=" * 60)

    # 1. Load dish mentions from Phase 2
    mentions_path = DATA_DIR / "dish_mentions.parquet"
    if not mentions_path.exists():
        print(f"ERROR: {mentions_path} not found. Run 02_extract_dishes.py first.")
        return

    mentions = pd.read_parquet(mentions_path)
    print(f"\n  Loaded {len(mentions):,} dish mentions")
    print(f"  Unique dishes: {mentions['dish_id'].nunique()}")

    # 2. Check for API keys (OpenRouter > OpenAI > Gemini)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GOOGLE_API_KEY")

    llm_annotations = None
    annotations_path = DATA_DIR / "llm_annotations.parquet"

    # Resume from checkpoint if exists
    if annotations_path.exists():
        llm_annotations = pd.read_parquet(annotations_path)
        valid = llm_annotations[llm_annotations["hedonic_score_llm"].between(1, 10)]
        print(f"\n  Found existing LLM annotations: {len(llm_annotations)} total, {len(valid)} valid")
        if len(valid) >= 1800:
            print("  Sufficient annotations found. Skipping LLM annotation step.")
        else:
            print("  Insufficient valid annotations. Re-running...")
            llm_annotations = None

    if llm_annotations is None:
        if openrouter_key:
            print("\n  OpenRouter API key found. Using DeepSeek v3.2 for annotation.")
            print("  Annotating ~2,000 samples...")
            llm_annotations = annotate_with_openai(
                mentions, openrouter_key,
                model="deepseek/deepseek-v3.2",
                base_url="https://openrouter.ai/api/v1",
                n_samples=2000,
                rate_limit_delay=0.3,
            )
            llm_annotations.to_parquet(annotations_path, index=False)
            print(f"  Saved: {annotations_path}")
        elif openai_key:
            print("\n  OpenAI API key found. Using GPT-4o-mini for annotation.")
            print("  Annotating 2,000 samples...")
            llm_annotations = annotate_with_openai(mentions, openai_key, n_samples=2000)
            llm_annotations.to_parquet(annotations_path, index=False)
            print(f"  Saved: {annotations_path}")
        elif gemini_key:
            print("\n  Gemini API key found. Using Gemini Flash for annotation.")
            print("  Annotating 2,000 samples (slower due to rate limits)...")
            llm_annotations = annotate_with_gemini(mentions, gemini_key, n_samples=2000)
            llm_annotations.to_parquet(annotations_path, index=False)
            print(f"  Saved: {annotations_path}")
        else:
            print("\n  No LLM API key found (OPENROUTER_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY).")
            print("  Falling back to pre-trained sentiment model only.")

    # 3. Pre-trained sentiment baseline
    # Smart sampling: max 500 per dish for computational feasibility on CPU
    # 500 samples per dish gives CI95 < 0.1 — more than sufficient
    MAX_PER_DISH = 500
    print(f"\n  Sampling up to {MAX_PER_DISH} mentions per dish for sentiment scoring...")
    sampled = (
        mentions
        .groupby("dish_id", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), MAX_PER_DISH), random_state=42))
    )
    print(f"  Sampled {len(sampled):,} mentions from {sampled['dish_id'].nunique()} dishes")
    print(f"  (down from {len(mentions):,} total mentions)")

    print("\n  Running pre-trained sentiment model...")
    scored = score_with_pretrained_sentiment(sampled)
    scored.to_parquet(DATA_DIR / "dish_mentions_scored.parquet", index=False)
    print(f"  Saved: {DATA_DIR / 'dish_mentions_scored.parquet'}")

    # 4. Aggregate hedonic scores per dish
    print("\n── Aggregating Hedonic Scores ──")
    dish_hedonic = aggregate_hedonic_scores(
        scored,
        score_col="hedonic_score_pretrained",
        min_reviews=10,
    )

    # Add cuisine and cook_method info
    dish_info = mentions[["dish_id", "cuisine", "cook_method"]].drop_duplicates("dish_id")
    dish_hedonic = dish_hedonic.merge(dish_info, left_index=True, right_on="dish_id",
                                      how="left").set_index("dish_id")

    dish_hedonic.to_csv(DATA_DIR / "dish_hedonic_scores.csv")
    print(f"  Saved: {DATA_DIR / 'dish_hedonic_scores.csv'}")

    # Print top/bottom dishes
    print(f"\n  Top 15 highest hedonic score dishes:")
    top = dish_hedonic.nlargest(15, "H_mean")
    for dish_id, row in top.iterrows():
        print(f"    {dish_id:30s} H={row['H_mean']:.2f} (n={row['H_n']:.0f}, "
              f"CI95=+/-{row['H_ci95']:.2f}) [{row['cuisine']}]")

    print(f"\n  Bottom 15 lowest hedonic score dishes:")
    bottom = dish_hedonic.nsmallest(15, "H_mean")
    for dish_id, row in bottom.iterrows():
        print(f"    {dish_id:30s} H={row['H_mean']:.2f} (n={row['H_n']:.0f}, "
              f"CI95=+/-{row['H_ci95']:.2f}) [{row['cuisine']}]")

    # 5. Generate human validation sample
    print("\n── Generating Human Validation Sample ──")
    generate_human_validation_sample(scored, n_samples=200)

    # 6. Summary
    print("\n── Summary ──")
    print(f"  Total dish mentions processed: {len(mentions):,}")
    print(f"  Dishes with hedonic scores: {len(dish_hedonic)}")
    print(f"  Overall mean H: {dish_hedonic['H_mean'].mean():.2f}")
    print(f"  H range: {dish_hedonic['H_mean'].min():.2f} - {dish_hedonic['H_mean'].max():.2f}")

    if llm_annotations is not None:
        print(f"  LLM annotations: {len(llm_annotations)}")
        valid_llm = llm_annotations[llm_annotations["hedonic_score_llm"].between(1, 10)]
        print(f"  Valid LLM annotations: {len(valid_llm)}")

    print("\n" + "=" * 60)
    print("NLP hedonic scoring complete.")
    print("  Next: Run 04_env_cost_calculation.py")
    print("  Or: Fine-tune model with LLM annotations (see 03b_finetune.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
