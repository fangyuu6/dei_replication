"""
20_pairwise_expansion.py — Extend pairwise ranking to 337 dishes
================================================================
Uses anchor-bridging: 158 original dishes have full pairwise data;
179 new dishes are each compared against ~30 anchor dishes spanning
the full H range. All wins (original + new) are then jointly fitted
with a Bradley-Terry model to place all 337 dishes on one scale.

Strategy:
  - Select 30 anchor dishes from the original 158 (evenly spaced by H)
  - For each new dish, compare against all 30 anchors → 179×30 = 5,370 pairs
  - Batch 10/call → 537 API calls (8 workers → ~2 min)
  - Merge with 12,403 original wins → global Bradley-Terry fit

Input:
  data/pairwise_wins.csv         (12,403 original wins)
  data/dish_mentions_scored.parquet (original review text)
  data/expanded_dish_mentions.parquet (expanded review text)
  data/dish_h_pairwise.csv       (original pairwise H for anchor selection)

Output:
  data/pairwise_wins_all.csv     (original + expansion wins)
  data/dish_h_pairwise_all.csv   (337 dishes, unified BT scale)
  results/figures/pairwise_expansion.png
"""

import sys, os, json, time, random, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"

N_ANCHORS = 30          # anchor dishes from original 158
BATCH_SIZE = 10         # pairs per API call
MAX_WORKERS = 8         # concurrent API calls
REVIEW_SAMPLES = 5      # representative reviews per dish


# ══════════════════════════════════════════════════════════════════
# STEP 1: Select representative reviews
# ══════════════════════════════════════════════════════════════════
def select_reviews_by_bert(scored_df, n=REVIEW_SAMPLES):
    """Pick n reviews per dish, stratified by BERT score percentile."""
    profiles = {}
    percentiles = np.linspace(0.1, 0.9, n)

    for dish_id, group in scored_df.groupby("dish_id"):
        s = group["hedonic_score_finetuned"].dropna()
        if len(s) < n:
            idx = s.index.tolist()
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

        excerpts = []
        for i in idx[:n]:
            text = group.loc[i, "context_text"]
            if isinstance(text, str) and len(text) > 20:
                excerpts.append(text[:200].strip())
        if excerpts:
            profiles[dish_id] = excerpts

    return profiles


def select_reviews_by_stars(mentions_df, n=REVIEW_SAMPLES):
    """Pick n reviews per dish, stratified by star rating (for expanded dishes)."""
    profiles = {}

    for dish_id, group in mentions_df.groupby("dish_id"):
        if len(group) < n:
            sample = group
        else:
            # Stratify by star rating — pick reviews at different quality levels
            stars_sorted = group.sort_values("stars")
            idx = np.linspace(0, len(stars_sorted) - 1, n, dtype=int)
            sample = stars_sorted.iloc[idx]

        excerpts = []
        for _, row in sample.iterrows():
            text = row["context_text"]
            if isinstance(text, str) and len(text) > 20:
                excerpts.append(text[:200].strip())
        if excerpts:
            profiles[dish_id] = excerpts

    return profiles


# ══════════════════════════════════════════════════════════════════
# STEP 2: Anchor selection
# ══════════════════════════════════════════════════════════════════
def select_anchors(pairwise_df, n_anchors=N_ANCHORS):
    """Select n anchor dishes evenly spaced across the H range."""
    sorted_dishes = pairwise_df.sort_values("H_pairwise")
    n = len(sorted_dishes)
    idx = np.linspace(0, n - 1, n_anchors, dtype=int)
    anchors = sorted_dishes.iloc[idx].index.tolist()
    print(f"  Selected {len(anchors)} anchors spanning H=[{sorted_dishes.iloc[idx[0]]['H_pairwise']:.2f}, "
          f"{sorted_dishes.iloc[idx[-1]]['H_pairwise']:.2f}]", flush=True)
    return anchors


# ══════════════════════════════════════════════════════════════════
# STEP 3: API calling (reused from script 17)
# ══════════════════════════════════════════════════════════════════
def make_batch_prompt(pairs_with_reviews):
    lines = [
        "You are a food expert. For each numbered pair below, read the restaurant review excerpts "
        "and decide which dish sounds MORE DELICIOUS based on the dining experience described. "
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
    import re
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


# ══════════════════════════════════════════════════════════════════
# STEP 4: Bradley-Terry fitting
# ══════════════════════════════════════════════════════════════════
def fit_bradley_terry(wins_df, dish_ids):
    import choix

    id2idx = {d: i for i, d in enumerate(dish_ids)}
    n = len(dish_ids)

    comparisons = []
    for _, row in wins_df.iterrows():
        w = id2idx.get(row["winner"])
        l = id2idx.get(row["loser"])
        if w is not None and l is not None:
            comparisons.append((w, l))

    print(f"  Fitting Bradley-Terry on {len(comparisons)} comparisons, {n} items",
          flush=True)

    params = choix.ilsr_pairwise(n, comparisons, alpha=0.01)

    p_min, p_max = params.min(), params.max()
    if p_max > p_min:
        h_pairwise = 1 + 9 * (params - p_min) / (p_max - p_min)
    else:
        h_pairwise = np.full(n, 5.5)

    result = pd.DataFrame({
        "dish_id": dish_ids,
        "H_pairwise": h_pairwise,
        "BT_strength": params,
    }).set_index("dish_id")

    return result


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70, flush=True)
    print("20 — Pairwise Expansion: 158 → 337 dishes via anchor bridging", flush=True)
    print("=" * 70, flush=True)

    if not API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable", flush=True)
        return

    # ── Load data ──────────────────────────────────────────────────
    print("\n── Loading data ──", flush=True)

    # Original scored mentions (have BERT finetuned scores)
    orig_scored = pd.read_parquet(DATA / "dish_mentions_scored.parquet")
    print(f"  Original mentions: {len(orig_scored):,} ({orig_scored.dish_id.nunique()} dishes)",
          flush=True)

    # Expanded mentions (no BERT scores, but have star ratings)
    exp_mentions = pd.read_parquet(DATA / "expanded_dish_mentions.parquet")
    print(f"  Expanded mentions: {len(exp_mentions):,} ({exp_mentions.dish_id.nunique()} dishes)",
          flush=True)

    # Original pairwise results (for anchor selection)
    pw_orig = pd.read_csv(DATA / "dish_h_pairwise.csv", index_col="dish_id")
    print(f"  Original pairwise H: {len(pw_orig)} dishes", flush=True)

    # ── Build review profiles ──────────────────────────────────────
    print("\n── Building review profiles ──", flush=True)

    # Original dishes: stratified by BERT score
    profiles_orig = select_reviews_by_bert(orig_scored)
    print(f"  Original profiles: {len(profiles_orig)} dishes", flush=True)

    # Expanded dishes: stratified by star rating
    profiles_exp = select_reviews_by_stars(exp_mentions)
    print(f"  Expanded profiles: {len(profiles_exp)} dishes", flush=True)

    # Merge all profiles
    profiles = {**profiles_orig, **profiles_exp}
    all_dishes = sorted(profiles.keys())
    new_dishes = sorted(set(profiles_exp.keys()) - set(profiles_orig.keys()))
    print(f"  Total profiles: {len(profiles)} ({len(new_dishes)} new)", flush=True)

    # ── Select anchors ─────────────────────────────────────────────
    print("\n── Selecting anchor dishes ──", flush=True)
    anchors = select_anchors(pw_orig)

    # ── Generate anchor pairs for new dishes ───────────────────────
    expansion_pairs = []
    for new_dish in new_dishes:
        for anchor in anchors:
            expansion_pairs.append((new_dish, anchor))

    print(f"  Expansion pairs: {len(expansion_pairs)} "
          f"({len(new_dishes)} new × {len(anchors)} anchors)", flush=True)

    # ── Check for cached expansion wins ────────────────────────────
    wins_exp_path = DATA / "pairwise_wins_expansion.csv"
    if wins_exp_path.exists():
        print(f"\n  [cache] Loading existing expansion wins", flush=True)
        wins_exp = pd.read_csv(wins_exp_path)
        print(f"  Loaded {len(wins_exp)} expansion wins", flush=True)

        # Find remaining pairs
        done_pairs = set()
        for _, r in wins_exp.iterrows():
            pair = tuple(sorted([r["winner"], r["loser"]]))
            done_pairs.add(pair)

        remaining = [(a, b) for a, b in expansion_pairs
                     if tuple(sorted([a, b])) not in done_pairs]
        print(f"  {len(done_pairs)} done, {len(remaining)} remaining", flush=True)
    else:
        wins_exp = pd.DataFrame(columns=["winner", "loser"])
        remaining = expansion_pairs
        print(f"  No cache, running all {len(remaining)} pairs", flush=True)

    # ── Run pairwise comparisons ───────────────────────────────────
    if remaining:
        print(f"\n── Running {len(remaining)} pairwise comparisons ──", flush=True)

        random.seed(42)
        random.shuffle(remaining)

        batches = [remaining[i:i+BATCH_SIZE]
                   for i in range(0, len(remaining), BATCH_SIZE)]
        print(f"  {len(batches)} API calls, {MAX_WORKERS} workers", flush=True)

        all_exp_wins = list(wins_exp.to_dict("records"))
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for batch_idx, batch in enumerate(batches):
                f = executor.submit(process_batch, batch_idx, batch, profiles)
                futures[f] = batch_idx

            for f in as_completed(futures):
                batch_idx, results = f.result()
                all_exp_wins.extend(results)
                completed += 1
                if not results:
                    failed += 1
                if completed % 50 == 0 or completed == len(batches):
                    pct = completed / len(batches) * 100
                    print(f"    {completed}/{len(batches)} ({pct:.0f}%) — "
                          f"{len(all_exp_wins)} wins, {failed} failed",
                          flush=True)

                # Periodic save
                if completed % 100 == 0:
                    pd.DataFrame(all_exp_wins).to_csv(wins_exp_path, index=False)

        wins_exp = pd.DataFrame(all_exp_wins)
        wins_exp.to_csv(wins_exp_path, index=False)
        print(f"\n  Saved {len(wins_exp)} expansion wins", flush=True)

    # ── Merge original + expansion wins ────────────────────────────
    print("\n── Merging wins ──", flush=True)
    wins_orig = pd.read_csv(DATA / "pairwise_wins.csv")
    print(f"  Original wins: {len(wins_orig)}", flush=True)
    print(f"  Expansion wins: {len(wins_exp)}", flush=True)

    wins_all = pd.concat([wins_orig, wins_exp], ignore_index=True)
    wins_all.to_csv(DATA / "pairwise_wins_all.csv", index=False)
    print(f"  Total wins: {len(wins_all)}", flush=True)

    # ── Global Bradley-Terry fit ───────────────────────────────────
    print("\n── Fitting global Bradley-Terry model ──", flush=True)
    bt_all = fit_bradley_terry(wins_all, all_dishes)

    # Add BERT H for comparison (original dishes only)
    bert_h_orig = pd.read_csv(DATA / "dish_hedonic_scores_bert.csv", index_col="dish_id")
    exp_h = pd.read_csv(DATA / "expanded_dish_hedonic.csv", index_col="dish_id")
    bert_h_all = pd.concat([bert_h_orig[["H_mean"]], exp_h[["H_mean"]]])
    bert_h_all.columns = ["H_bert"]

    bt_all = bt_all.join(bert_h_all, how="left")

    # Mark source
    bt_all["source"] = "expanded"
    bt_all.loc[bt_all.index.isin(pw_orig.index), "source"] = "original"

    # Stats
    h = bt_all["H_pairwise"]
    print(f"\n  All dishes: {len(bt_all)}", flush=True)
    print(f"  H range: [{h.min():.2f}, {h.max():.2f}]", flush=True)
    print(f"  H mean:  {h.mean():.2f}, std: {h.std():.2f}, CV: {h.std()/h.mean()*100:.1f}%",
          flush=True)

    # Original dishes: compare old vs new BT scores
    orig_mask = bt_all["source"] == "original"
    h_orig_new = bt_all.loc[orig_mask, "H_pairwise"]
    h_orig_old = pw_orig.loc[h_orig_new.index, "H_pairwise"]
    rho_stability, _ = stats.spearmanr(h_orig_old, h_orig_new)
    print(f"\n  Stability of original 158 dishes:", flush=True)
    print(f"    Spearman ρ(old BT, new BT): {rho_stability:.4f}", flush=True)
    print(f"    Mean absolute shift: {(h_orig_new - h_orig_old).abs().mean():.3f}", flush=True)

    # Correlation with BERT H
    valid = bt_all.dropna(subset=["H_bert"])
    rho_bert, p_bert = stats.spearmanr(valid["H_pairwise"], valid["H_bert"])
    print(f"\n  Pairwise vs BERT correlation (all {len(valid)} dishes):", flush=True)
    print(f"    Spearman ρ: {rho_bert:.3f} (p={p_bert:.2e})", flush=True)

    # Save
    bt_all.round(4).to_csv(DATA / "dish_h_pairwise_all.csv")
    print(f"\n  Saved: {DATA / 'dish_h_pairwise_all.csv'}", flush=True)

    # ── Visualization ──────────────────────────────────────────────
    print("\n── Plotting ──", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. H distribution: original vs expanded
    ax = axes[0]
    ax.hist(bt_all.loc[orig_mask, "H_pairwise"], bins=20, alpha=0.6,
            label=f"Original (n={orig_mask.sum()})", color="#2196F3", density=True)
    ax.hist(bt_all.loc[~orig_mask, "H_pairwise"], bins=20, alpha=0.6,
            label=f"Expanded (n={(~orig_mask).sum()})", color="#FF9800", density=True)
    ax.set_xlabel("H (Pairwise Bradley-Terry)")
    ax.set_ylabel("Density")
    ax.set_title("H Distribution by Source")
    ax.legend()

    # 2. Stability of original dish rankings
    ax = axes[1]
    ax.scatter(h_orig_old, h_orig_new, alpha=0.5, s=30, color="steelblue",
               edgecolors="white", linewidth=0.3)
    ax.plot([1, 10], [1, 10], "k--", alpha=0.4)
    ax.set_xlabel("H (Original 158-dish BT)")
    ax.set_ylabel("H (Global 337-dish BT)")
    ax.set_title(f"BT Stability (ρ={rho_stability:.3f})")

    # 3. Pairwise vs BERT for all dishes
    ax = axes[2]
    colors = ["#2196F3" if s == "original" else "#FF9800" for s in valid["source"]]
    ax.scatter(valid["H_bert"], valid["H_pairwise"], alpha=0.5, s=30,
               c=colors, edgecolors="white", linewidth=0.3)
    ax.set_xlabel("H (BERT Absolute)")
    ax.set_ylabel("H (Pairwise Bradley-Terry)")
    ax.set_title(f"BERT vs Pairwise (ρ={rho_bert:.3f})")

    plt.suptitle(f"Pairwise Expansion: 158 → {len(bt_all)} Dishes",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "pairwise_expansion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES / 'pairwise_expansion.png'}", flush=True)

    # ── Top/Bottom dishes ──────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("TOP 15 DISHES (Pairwise H, 337-dish scale)", flush=True)
    for i, (idx, row) in enumerate(bt_all.nlargest(15, "H_pairwise").iterrows(), 1):
        src = "★" if row["source"] == "expanded" else " "
        print(f"  {i:2d}. {src} {idx:<30s} H_pw={row['H_pairwise']:.2f}  "
              f"H_bert={row.get('H_bert', float('nan')):.2f}", flush=True)

    print(f"\nBOTTOM 15 DISHES", flush=True)
    for i, (idx, row) in enumerate(bt_all.nsmallest(15, "H_pairwise").iterrows(), 1):
        src = "★" if row["source"] == "expanded" else " "
        print(f"  {i:2d}. {src} {idx:<30s} H_pw={row['H_pairwise']:.2f}  "
              f"H_bert={row.get('H_bert', float('nan')):.2f}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
