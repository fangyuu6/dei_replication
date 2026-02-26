"""
08d_lodocv.py — P2-3: Leave-one-dish-out Cross-Validation (Grouped K-Fold)
===========================================================================
Evaluates whether the finetuned BERT model generalizes to unseen dishes.

The concern: if a dish's mentions appear in both train and test, the model
might memorize dish-specific cues rather than learning general hedonic patterns.

Approach:
  - Grouped 10-fold CV (dishes as groups, ~16 dishes per fold)
  - Each fold: train on ~141 dishes, predict on ~16 held-out dishes
  - Compute mention-level and dish-level metrics per fold
  - Final dish-level r is computed on ALL dishes (each predicted out-of-fold)

This is more rigorous than the existing GroupShuffleSplit 80/20 split,
which already groups by dish (r=0.844 on ~31 held-out dishes).

Outputs:
  - results/tables/lodocv_results.csv
  - results/tables/lodocv_dish_predictions.csv
  - results/figures/lodocv_calibration.png
"""

import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Settings ─────────────────────────────────────────────────────
N_FOLDS = 10
MAX_LENGTH = 256
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5

# ── Load data ────────────────────────────────────────────────────
df = pd.read_parquet(DATA_DIR / "llm_annotations.parquet")
valid = df[df["hedonic_score_llm"].between(1, 10)].copy().reset_index(drop=True)
print(f"Valid annotations: {len(valid)}, dishes: {valid['dish_id'].nunique()}")

# ── Grouped K-Fold split ─────────────────────────────────────────
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=N_FOLDS)
groups = valid["dish_id"].values

print(f"\nGrouped {N_FOLDS}-fold CV (each fold holds out ~{valid['dish_id'].nunique()//N_FOLDS} dishes)")

# ── Import torch and transformers ────────────────────────────────
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

class HedonicDataset:
    def __init__(self, texts, scores, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.labels = scores

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

# ── Run K-Fold CV ────────────────────────────────────────────────
all_predictions = []  # collect (dish_id, true_score, pred_score) for all folds
fold_metrics = []

total_start = time.time()

for fold_i, (train_idx, test_idx) in enumerate(gkf.split(valid, groups=groups)):
    fold_start = time.time()
    train_df = valid.iloc[train_idx]
    test_df = valid.iloc[test_idx]

    n_train_dishes = train_df["dish_id"].nunique()
    n_test_dishes = test_df["dish_id"].nunique()
    test_dishes = set(test_df["dish_id"].unique())
    train_dishes = set(train_df["dish_id"].unique())
    overlap = train_dishes & test_dishes

    print(f"\n{'='*60}")
    print(f"Fold {fold_i+1}/{N_FOLDS}: train={len(train_df)} ({n_train_dishes} dishes), "
          f"test={len(test_df)} ({n_test_dishes} dishes), overlap={len(overlap)}")
    print(f"{'='*60}")

    assert len(overlap) == 0, "Dish leakage detected!"

    # Load fresh model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )

    # Prepare datasets
    train_texts = train_df["context_text"].str[:512].tolist()
    train_scores = train_df["hedonic_score_llm"].tolist()
    test_texts = test_df["context_text"].str[:512].tolist()
    test_scores = test_df["hedonic_score_llm"].tolist()

    train_dataset = HedonicDataset(train_texts, train_scores, tokenizer)
    test_dataset = HedonicDataset(test_texts, test_scores, tokenizer)

    # Training
    output_dir = str(ROOT / "models" / "lodocv_temp" / f"fold_{fold_i}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=True,
    )

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.flatten()
        labels = eval_pred.label_ids.flatten()
        mae = np.mean(np.abs(preds - labels))
        r, _ = pearsonr(preds, labels)
        return {"mae": mae, "pearson_r": r}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Predict on held-out fold
    preds_output = trainer.predict(test_dataset)
    preds = np.clip(preds_output.predictions.flatten(), 1, 10)

    # Mention-level metrics
    mae = np.mean(np.abs(preds - np.array(test_scores)))
    rmse = np.sqrt(np.mean((preds - np.array(test_scores))**2))
    r_mention, _ = pearsonr(preds, test_scores)

    # Dish-level aggregation
    test_df = test_df.copy()
    test_df["pred"] = preds
    dish_agg = test_df.groupby("dish_id").agg(
        true_mean=("hedonic_score_llm", "mean"),
        pred_mean=("pred", "mean"),
        n=("pred", "count"),
    )

    if len(dish_agg) >= 5:
        r_dish, _ = pearsonr(dish_agg["true_mean"], dish_agg["pred_mean"])
        rho_dish, _ = spearmanr(dish_agg["true_mean"], dish_agg["pred_mean"])
    else:
        r_dish = np.nan
        rho_dish = np.nan

    elapsed = time.time() - fold_start
    print(f"  Mention: MAE={mae:.3f}, RMSE={rmse:.3f}, r={r_mention:.3f}")
    print(f"  Dish:    r={r_dish:.3f}, ρ={rho_dish:.3f} (n={len(dish_agg)} dishes)")
    print(f"  Time: {elapsed:.0f}s")

    fold_metrics.append({
        "fold": fold_i + 1,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_dishes": n_train_dishes,
        "n_test_dishes": n_test_dishes,
        "mention_mae": mae,
        "mention_rmse": rmse,
        "mention_r": r_mention,
        "dish_r": r_dish,
        "dish_rho": rho_dish,
        "time_sec": elapsed,
    })

    # Collect dish-level predictions
    for dish_id, row in dish_agg.iterrows():
        all_predictions.append({
            "dish_id": dish_id,
            "true_mean": row["true_mean"],
            "pred_mean": row["pred_mean"],
            "n_mentions": row["n"],
            "fold": fold_i + 1,
        })

    # Clean up to free memory
    del model, trainer, train_dataset, test_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"Total CV time: {total_elapsed/60:.1f} minutes")

# ── Aggregate results ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"CROSS-VALIDATION SUMMARY")
print(f"{'='*60}")

fold_df = pd.DataFrame(fold_metrics)
print(f"\nPer-fold metrics:")
print(fold_df[["fold","n_test_dishes","mention_mae","mention_r","dish_r","dish_rho"]].to_string(index=False))

print(f"\nMean across folds:")
print(f"  Mention MAE:  {fold_df['mention_mae'].mean():.3f} ± {fold_df['mention_mae'].std():.3f}")
print(f"  Mention RMSE: {fold_df['mention_rmse'].mean():.3f} ± {fold_df['mention_rmse'].std():.3f}")
print(f"  Mention r:    {fold_df['mention_r'].mean():.3f} ± {fold_df['mention_r'].std():.3f}")
print(f"  Dish r:       {fold_df['dish_r'].mean():.3f} ± {fold_df['dish_r'].std():.3f}")
print(f"  Dish ρ:       {fold_df['dish_rho'].mean():.3f} ± {fold_df['dish_rho'].std():.3f}")

# Overall dish-level correlation (all dishes, each predicted out-of-fold)
pred_df = pd.DataFrame(all_predictions)
r_overall, p_r = pearsonr(pred_df["true_mean"], pred_df["pred_mean"])
rho_overall, p_rho = spearmanr(pred_df["true_mean"], pred_df["pred_mean"])
mae_overall = np.mean(np.abs(pred_df["true_mean"] - pred_df["pred_mean"]))

print(f"\nOverall dish-level (all {len(pred_df)} dishes, out-of-fold):")
print(f"  Pearson r  = {r_overall:.3f} (p = {p_r:.2e})")
print(f"  Spearman ρ = {rho_overall:.3f} (p = {p_rho:.2e})")
print(f"  MAE        = {mae_overall:.3f}")

# Compare with original GroupShuffleSplit result
print(f"\nComparison:")
print(f"  Original GroupShuffleSplit (80/20): dish r = 0.844 (~31 test dishes)")
print(f"  Grouped 10-fold CV:                dish r = {r_overall:.3f} ({len(pred_df)} test dishes)")
print(f"  → {'Consistent' if abs(r_overall - 0.844) < 0.15 else 'Significant change'}")

# ── Save results ─────────────────────────────────────────────────
TABLES_DIR.mkdir(parents=True, exist_ok=True)
fold_df.to_csv(TABLES_DIR / "lodocv_results.csv", index=False)
pred_df.to_csv(TABLES_DIR / "lodocv_dish_predictions.csv", index=False)
print(f"\n  Saved: {TABLES_DIR / 'lodocv_results.csv'}")
print(f"  Saved: {TABLES_DIR / 'lodocv_dish_predictions.csv'}")

# ── Calibration plot ─────────────────────────────────────────────
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: dish-level true vs predicted
axes[0].scatter(pred_df["true_mean"], pred_df["pred_mean"], alpha=0.6, s=30,
                c="steelblue", edgecolors="white", linewidth=0.3)
lims = [pred_df[["true_mean","pred_mean"]].min().min() - 0.2,
        pred_df[["true_mean","pred_mean"]].max().max() + 0.2]
axes[0].plot(lims, lims, "k--", alpha=0.5, linewidth=1)
axes[0].set_xlabel("True H (LLM annotation mean)", fontsize=12)
axes[0].set_ylabel("Predicted H (BERT out-of-fold)", fontsize=12)
axes[0].set_title(f"Dish-level Calibration (LODOCV)\n"
                   f"r={r_overall:.3f}, ρ={rho_overall:.3f}, MAE={mae_overall:.3f}",
                   fontsize=13)
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)

# Right: per-fold dish r distribution
axes[1].bar(fold_df["fold"], fold_df["dish_r"], color="steelblue",
            edgecolor="white", alpha=0.8)
axes[1].axhline(r_overall, color="red", linestyle="--", linewidth=1.5,
                label=f"Overall r = {r_overall:.3f}")
axes[1].axhline(0.844, color="green", linestyle=":", linewidth=1.5,
                label="Original split r = 0.844")
axes[1].set_xlabel("Fold", fontsize=12)
axes[1].set_ylabel("Dish-level Pearson r", fontsize=12)
axes[1].set_title("Per-fold Dish-level r", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, 1)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "lodocv_calibration.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIGURES_DIR / 'lodocv_calibration.png'}")

# ── Clean up temp model dirs ─────────────────────────────────────
import shutil
temp_dir = ROOT / "models" / "lodocv_temp"
if temp_dir.exists():
    shutil.rmtree(temp_dir)
    print(f"  Cleaned up: {temp_dir}")

print(f"\nDone! Total time: {total_elapsed/60:.1f} minutes")
