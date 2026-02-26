"""
03b_finetune_bert.py - Fine-tune BERT for Hedonic Scoring
=========================================================
Takes LLM annotations from Step 1 and fine-tunes a BERT model
for regression (predicting hedonic score 1-10 from review text).

Input:  data/llm_annotations.parquet (from 03_nlp_hedonic_scoring.py)
Output: models/hedonic_bert_finetuned/ (saved model + tokenizer)
        results/tables/finetune_metrics.csv (validation metrics)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hedonic_bert_finetuned"


def load_and_prepare_data(annotations_path: Path):
    """Load LLM annotations and prepare train/val splits."""
    df = pd.read_parquet(annotations_path)
    print(f"Loaded {len(df)} annotations")

    # Filter: keep only valid scores 1-10
    valid = df[df["hedonic_score_llm"].between(1, 10)].copy()
    print(f"Valid annotations (score 1-10): {len(valid)}")
    print(f"Dropped: {len(df) - len(valid)} (score=0 insufficient info, score=-1 errors)")

    # Score distribution
    print(f"\nScore distribution:")
    print(valid["hedonic_score_llm"].value_counts().sort_index())
    print(f"Mean: {valid['hedonic_score_llm'].mean():.2f}, Std: {valid['hedonic_score_llm'].std():.2f}")

    # Stratified train/val split by dish_id (80/20)
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(valid, groups=valid["dish_id"]))

    train_df = valid.iloc[train_idx].reset_index(drop=True)
    val_df = valid.iloc[val_idx].reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} samples, {train_df['dish_id'].nunique()} dishes")
    print(f"Val:   {len(val_df)} samples, {val_df['dish_id'].nunique()} dishes")

    return train_df, val_df


class HedonicDataset:
    """PyTorch dataset for hedonic score regression."""

    def __init__(self, texts, scores, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.labels = scores

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def train_model(train_df, val_df):
    """Fine-tune BERT for hedonic score regression."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )

    model_name = "bert-base-uncased"
    print(f"\nLoading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Regression: num_labels=1
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )

    # Prepare datasets
    train_texts = train_df["context_text"].str[:512].tolist()
    train_scores = train_df["hedonic_score_llm"].tolist()
    val_texts = val_df["context_text"].str[:512].tolist()
    val_scores = val_df["hedonic_score_llm"].tolist()

    train_dataset = HedonicDataset(train_texts, train_scores, tokenizer)
    val_dataset = HedonicDataset(val_texts, val_scores, tokenizer)

    # Training arguments
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.flatten()
        labels = eval_pred.label_ids.flatten()
        mae = np.mean(np.abs(preds - labels))
        rmse = np.sqrt(np.mean((preds - labels) ** 2))
        r, _ = pearsonr(preds, labels)
        rho, _ = spearmanr(preds, labels)
        return {"mae": mae, "rmse": rmse, "pearson_r": r, "spearman_rho": rho}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    # Save best model
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nModel saved to: {MODEL_DIR}")

    # Final evaluation
    results = trainer.evaluate()
    print(f"\n=== Validation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return trainer, results


def evaluate_and_report(trainer, val_df):
    """Generate detailed evaluation report."""
    import torch

    # Get predictions
    val_texts = val_df["context_text"].str[:512].tolist()
    val_scores = val_df["hedonic_score_llm"].values

    preds_output = trainer.predict(
        HedonicDataset(val_texts, val_scores.tolist(),
                       trainer.processing_class, max_length=256)
    )
    preds = preds_output.predictions.flatten()

    # Clip predictions to valid range
    preds = np.clip(preds, 1, 10)

    # Metrics
    mae = np.mean(np.abs(preds - val_scores))
    rmse = np.sqrt(np.mean((preds - val_scores) ** 2))
    pearson_r, pearson_p = pearsonr(preds, val_scores)
    spearman_rho, spearman_p = spearmanr(preds, val_scores)
    bias = np.mean(preds - val_scores)

    metrics = {
        "n_train": len(trainer.train_dataset),
        "n_val": len(val_df),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": pearson_p,
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p": spearman_p,
        "bias": round(bias, 4),
        "passes_mae": mae < 1.0,
        "passes_pearson": pearson_r > 0.75,
    }

    # Save metrics
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "finetune_metrics.csv", index=False)
    print(f"\nMetrics saved to: {TABLES_DIR / 'finetune_metrics.csv'}")

    # Print report
    print(f"\n{'=' * 50}")
    print(f"Fine-tuning Evaluation Report")
    print(f"{'=' * 50}")
    print(f"  Training samples: {metrics['n_train']}")
    print(f"  Validation samples: {metrics['n_val']}")
    print(f"  MAE:  {mae:.3f}  {'PASS' if mae < 1.0 else 'FAIL'} (target < 1.0)")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Pearson r:  {pearson_r:.3f}  {'PASS' if pearson_r > 0.75 else 'FAIL'} (target > 0.75)")
    print(f"  Spearman rho: {spearman_rho:.3f}")
    print(f"  Bias: {bias:.3f}")

    # Per-dish analysis on validation set
    val_df = val_df.copy()
    val_df["pred"] = preds
    dish_comparison = (
        val_df.groupby("dish_id")
        .agg(
            true_mean=("hedonic_score_llm", "mean"),
            pred_mean=("pred", "mean"),
            n=("pred", "count"),
        )
    )
    dish_comparison["abs_error"] = np.abs(dish_comparison["true_mean"] - dish_comparison["pred_mean"])
    dish_r, _ = pearsonr(dish_comparison["true_mean"], dish_comparison["pred_mean"])
    print(f"\n  Dish-level Pearson r: {dish_r:.3f}")
    print(f"  Dish-level MAE: {dish_comparison['abs_error'].mean():.3f}")

    return metrics


def main():
    print("=" * 60)
    print("DEI Project - Fine-tune BERT for Hedonic Scoring")
    print("=" * 60)

    annotations_path = DATA_DIR / "llm_annotations.parquet"
    if not annotations_path.exists():
        print(f"ERROR: {annotations_path} not found.")
        print("Run 03_nlp_hedonic_scoring.py with an LLM API key first.")
        return

    # 1. Load and prepare data
    print("\n── Loading Data ──")
    train_df, val_df = load_and_prepare_data(annotations_path)

    # 2. Train model
    print("\n── Training ──")
    trainer, results = train_model(train_df, val_df)

    # 3. Evaluate
    print("\n── Evaluation ──")
    metrics = evaluate_and_report(trainer, val_df)

    print("\n" + "=" * 60)
    print("Fine-tuning complete.")
    print(f"  Model saved to: {MODEL_DIR}")
    print(f"  Next: Run 03c_apply_finetuned.py to rescore all mentions")
    print("=" * 60)


if __name__ == "__main__":
    main()
