"""
19_update_h_to_pairwise.py — Replace BERT H with Pairwise H in hedonic scores
===============================================================================
Updates dish_hedonic_scores.csv so H_mean uses Bradley-Terry pairwise scores.
Keeps BERT-level statistics (H_std, H_n, etc.) for mention-level analyses.
Adds H_bert column for reference.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("c:/project_EF/data")

# Load pairwise H
pw = pd.read_csv(DATA_DIR / "dish_h_pairwise.csv", index_col="dish_id")
print(f"Pairwise H: {len(pw)} dishes, range [{pw['H_pairwise'].min():.2f}, {pw['H_pairwise'].max():.2f}]")

# Load original BERT hedonic scores
bert = pd.read_csv(DATA_DIR / "dish_hedonic_scores_bert.csv", index_col="dish_id")
print(f"BERT H:     {len(bert)} dishes, range [{bert['H_mean'].min():.2f}, {bert['H_mean'].max():.2f}]")

# Create updated hedonic scores
updated = bert.copy()

# Replace H_mean with pairwise H
updated["H_bert"] = updated["H_mean"]  # keep BERT as reference
updated["H_mean"] = pw.loc[updated.index, "H_pairwise"]

# Update H_median, H_q25, H_q75, H_ci95, H_iqr — set to NaN since pairwise H
# is a dish-level point estimate (no mention-level distribution)
# Keep H_std and H_n from BERT for mention-level analyses
# The key change is H_mean → pairwise

print(f"\nUpdated H:  {len(updated)} dishes, range [{updated['H_mean'].min():.2f}, {updated['H_mean'].max():.2f}]")
print(f"  CV: {updated['H_mean'].std()/updated['H_mean'].mean()*100:.1f}%")

# Save
updated.to_csv(DATA_DIR / "dish_hedonic_scores.csv")
print(f"\nSaved: {DATA_DIR / 'dish_hedonic_scores.csv'}")
print("H_mean is now pairwise Bradley-Terry scores")
print("H_bert column preserves original BERT scores")
