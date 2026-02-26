"""
03c_apply_finetuned.py - Apply Fine-tuned BERT to All Mentions
==============================================================
Loads the fine-tuned hedonic BERT model and rescores all 76,927
sampled mentions, then re-aggregates dish-level H scores.

Input:  models/hedonic_bert_finetuned/ (from 03b)
        data/dish_mentions_scored.parquet (original scored mentions)
Output: data/dish_mentions_scored.parquet (updated with finetuned scores)
        data/dish_hedonic_scores.csv (updated dish-level H)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TABLES_DIR, HEDONIC_SCALE

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hedonic_bert_finetuned"


def score_with_finetuned(texts: list, batch_size: int = 32) -> np.ndarray:
    """Score texts using the fine-tuned BERT model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Loading fine-tuned model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch = texts[i:i + batch_size]
        batch = [t[:512] for t in batch]
        encodings = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)
            preds = outputs.logits.squeeze(-1).cpu().numpy()

        all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])

    # Clip to valid range
    return np.clip(np.array(all_preds), 1.0, 10.0)


def main():
    print("=" * 60)
    print("DEI Project - Apply Fine-tuned BERT to All Mentions")
    print("=" * 60)

    if not MODEL_DIR.exists():
        print(f"ERROR: Model not found at {MODEL_DIR}")
        print("Run 03b_finetune_bert.py first.")
        return

    # 1. Load scored mentions
    scored_path = DATA_DIR / "dish_mentions_scored.parquet"
    scored = pd.read_parquet(scored_path)
    print(f"\nLoaded {len(scored):,} scored mentions")
    print(f"Unique dishes: {scored['dish_id'].nunique()}")

    # 2. Score with fine-tuned model
    print("\n── Scoring with fine-tuned BERT ──")
    texts = scored["context_text"].tolist()
    finetuned_scores = score_with_finetuned(texts)

    scored["hedonic_score_finetuned"] = finetuned_scores

    # 3. Compare with pretrained scores
    print("\n── Comparison: Pretrained vs Fine-tuned ──")
    pretrained = scored["hedonic_score_pretrained"].values
    finetuned = scored["hedonic_score_finetuned"].values

    mask = ~(np.isnan(pretrained) | np.isnan(finetuned))
    r, _ = pearsonr(pretrained[mask], finetuned[mask])
    rho, _ = spearmanr(pretrained[mask], finetuned[mask])
    diff = np.mean(finetuned[mask] - pretrained[mask])
    print(f"  Pearson r:    {r:.3f}")
    print(f"  Spearman rho: {rho:.3f}")
    print(f"  Mean diff (finetuned - pretrained): {diff:.3f}")
    print(f"  Pretrained mean: {np.nanmean(pretrained):.3f}")
    print(f"  Finetuned mean:  {np.nanmean(finetuned):.3f}")

    # 4. Save updated scored mentions
    scored.to_parquet(scored_path, index=False)
    print(f"\nUpdated: {scored_path}")

    # 5. Re-aggregate dish-level H scores using finetuned scores
    print("\n── Re-aggregating Dish Hedonic Scores ──")
    valid = scored[scored["hedonic_score_finetuned"].between(
        HEDONIC_SCALE[0], HEDONIC_SCALE[1]
    )].copy()

    agg = (
        valid.groupby("dish_id")
        .agg(
            H_mean=("hedonic_score_finetuned", "mean"),
            H_median=("hedonic_score_finetuned", "median"),
            H_std=("hedonic_score_finetuned", "std"),
            H_n=("hedonic_score_finetuned", "count"),
            H_q25=("hedonic_score_finetuned", lambda x: x.quantile(0.25)),
            H_q75=("hedonic_score_finetuned", lambda x: x.quantile(0.75)),
        )
    )
    agg["H_ci95"] = 1.96 * agg["H_std"] / np.sqrt(agg["H_n"])
    agg["H_iqr"] = agg["H_q75"] - agg["H_q25"]

    # Filter: min 10 reviews
    agg = agg[agg["H_n"] >= 10].copy()

    # Add cuisine and cook_method
    dish_info = scored[["dish_id", "cuisine", "cook_method"]].drop_duplicates("dish_id")
    agg = agg.merge(dish_info, left_index=True, right_on="dish_id", how="left").set_index("dish_id")

    # Save — backup old scores first
    hedonic_path = DATA_DIR / "dish_hedonic_scores.csv"
    old_hedonic = pd.read_csv(hedonic_path, index_col="dish_id") if hedonic_path.exists() else None

    agg.round(3).to_csv(hedonic_path)
    print(f"Saved: {hedonic_path}")
    print(f"Dishes: {len(agg)}")

    # 6. Compare dish-level rankings
    if old_hedonic is not None:
        print("\n── Dish-level Ranking Comparison ──")
        merged = agg[["H_mean"]].merge(
            old_hedonic[["H_mean"]].rename(columns={"H_mean": "H_mean_old"}),
            left_index=True, right_index=True, how="inner"
        )
        dish_r, _ = pearsonr(merged["H_mean"], merged["H_mean_old"])
        dish_rho, _ = spearmanr(merged["H_mean"], merged["H_mean_old"])
        print(f"  Pearson r (dish means):    {dish_r:.3f}")
        print(f"  Spearman rho (rankings):   {dish_rho:.3f}")

        # Biggest movers
        merged["rank_new"] = merged["H_mean"].rank(ascending=False)
        merged["rank_old"] = merged["H_mean_old"].rank(ascending=False)
        merged["rank_change"] = merged["rank_old"] - merged["rank_new"]
        movers = merged.reindex(merged["rank_change"].abs().nlargest(10).index)
        print(f"\n  Top 10 biggest rank changes:")
        for dish, row in movers.iterrows():
            direction = "↑" if row["rank_change"] > 0 else "↓"
            print(f"    {dish:30s} {direction}{abs(row['rank_change']):.0f} "
                  f"(H: {row['H_mean_old']:.2f} → {row['H_mean']:.2f})")

    # Print top/bottom
    print(f"\n  Top 10 dishes (fine-tuned H):")
    for dish, row in agg.nlargest(10, "H_mean").iterrows():
        print(f"    {dish:30s} H={row['H_mean']:.2f}")

    print(f"\n  Bottom 10 dishes (fine-tuned H):")
    for dish, row in agg.nsmallest(10, "H_mean").iterrows():
        print(f"    {dish:30s} H={row['H_mean']:.2f}")

    print("\n" + "=" * 60)
    print("Done. Next: rerun 05_dei_computation.py and 06_validation_robustness.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
