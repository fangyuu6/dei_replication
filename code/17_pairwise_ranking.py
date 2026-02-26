"""
17_pairwise_ranking.py
======================
Pairwise LLM ranking of dishes → Bradley-Terry model → new H scores.

Bypasses ceiling effects in absolute rating by using forced-choice
comparisons: "Which dish sounds more delicious based on these reviews?"

Pipeline:
  1. Select representative review excerpts per dish (stratified by BERT score)
  2. Generate all C(158,2) = 12,403 pairwise comparisons
  3. Batch-query DeepSeek v3.2 via OpenRouter (10 pairs per call)
  4. Fit Bradley-Terry model → latent deliciousness scores
  5. Compare with BERT H (CV, rank correlation)

Input:  data/dish_mentions_scored.parquet
Output: data/pairwise_wins.csv
        data/dish_h_pairwise.csv
        results/figures/pairwise_vs_bert_h.png
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
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"

N_REPS_PER_PAIR = 1        # comparisons per pair
BATCH_SIZE = 10             # pairs per API call
MAX_WORKERS = 8             # concurrent API calls
REVIEW_SAMPLES = 5          # representative reviews per dish


# ══════════════════════════════════════════════════════════════════
# STEP 1: Select representative reviews per dish
# ══════════════════════════════════════════════════════════════════
def select_representative_reviews(scored_df, n=REVIEW_SAMPLES):
    """Pick n reviews per dish, stratified by BERT score percentile."""
    profiles = {}
    percentiles = np.linspace(0.1, 0.9, n)

    for dish_id, group in scored_df.groupby("dish_id"):
        s = group["hedonic_score_finetuned"].dropna()
        if len(s) < n:
            # Not enough reviews, take all
            idx = s.index
        else:
            # Stratified sampling at fixed percentiles
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
                # Truncate to ~150 chars for conciseness
                excerpts.append(text[:200].strip())
        profiles[dish_id] = excerpts

    return profiles


# ══════════════════════════════════════════════════════════════════
# STEP 2: Generate pairwise prompts
# ══════════════════════════════════════════════════════════════════
def make_batch_prompt(pairs_with_reviews):
    """Create a prompt comparing multiple pairs at once."""
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
        a_text = " | ".join(revs_a[:3])  # Use top 3 for brevity
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

    # Try structured format first: "1:A", "2:B", etc.
    import re
    matches = re.findall(r'(\d+)\s*[:\-\.]\s*([AB])', text)
    if matches:
        for num_str, choice in matches:
            results[int(num_str)] = choice
    else:
        # Fallback: just look for A/B in sequence
        choices = re.findall(r'[AB]', text)
        for i, c in enumerate(choices[:n_pairs], 1):
            results[i] = c

    return results


# ══════════════════════════════════════════════════════════════════
# STEP 3: API calling
# ══════════════════════════════════════════════════════════════════
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
            # else: skip unparseable

    return batch_idx, results


# ══════════════════════════════════════════════════════════════════
# STEP 4: Bradley-Terry fitting
# ══════════════════════════════════════════════════════════════════
def fit_bradley_terry(wins_df, dish_ids):
    """Fit Bradley-Terry model using choix library."""
    import choix

    # Map dish_id to integer index
    id2idx = {d: i for i, d in enumerate(dish_ids)}
    n = len(dish_ids)

    # Convert wins to (winner_idx, loser_idx) pairs
    comparisons = []
    for _, row in wins_df.iterrows():
        w = id2idx.get(row["winner"])
        l = id2idx.get(row["loser"])
        if w is not None and l is not None:
            comparisons.append((w, l))

    if not comparisons:
        raise ValueError("No valid comparisons found")

    print(f"  Fitting Bradley-Terry on {len(comparisons)} comparisons, {n} items",
          flush=True)

    # Fit using iterative Luce Spectral Ranking (fast, stable)
    params = choix.ilsr_pairwise(n, comparisons, alpha=0.01)

    # Convert to 1-10 scale for comparability
    # params are log-strength values; normalize to [1, 10]
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
# STEP 5: Analysis and plotting
# ══════════════════════════════════════════════════════════════════
def analyze_and_plot(bt_scores, bert_h):
    """Compare pairwise H with BERT H."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = bt_scores.join(bert_h[["H_mean"]].rename(columns={"H_mean": "H_bert"}))
    merged = merged.dropna()

    h_pw = merged["H_pairwise"]
    h_bt = merged["H_bert"]

    rho, p_rho = stats.spearmanr(h_pw, h_bt)
    r, p_r = stats.pearsonr(h_pw, h_bt)

    cv_pw = h_pw.std() / h_pw.mean() * 100
    cv_bt = h_bt.std() / h_bt.mean() * 100

    print(f"\n{'='*60}", flush=True)
    print("PAIRWISE vs BERT COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  N dishes: {len(merged)}", flush=True)
    print(f"  BERT H:     mean={h_bt.mean():.3f}, std={h_bt.std():.3f}, "
          f"CV={cv_bt:.2f}%, range=[{h_bt.min():.2f}, {h_bt.max():.2f}]",
          flush=True)
    print(f"  Pairwise H: mean={h_pw.mean():.3f}, std={h_pw.std():.3f}, "
          f"CV={cv_pw:.2f}%, range=[{h_pw.min():.2f}, {h_pw.max():.2f}]",
          flush=True)
    print(f"  CV improvement: {cv_bt:.1f}% → {cv_pw:.1f}% "
          f"({cv_pw/cv_bt:.1f}× {'wider' if cv_pw > cv_bt else 'narrower'})",
          flush=True)
    print(f"  Spearman ρ: {rho:.3f} (p={p_rho:.2e})", flush=True)
    print(f"  Pearson r:  {r:.3f} (p={p_r:.2e})", flush=True)

    # Top/Bottom comparison
    print(f"\n  Top 10 (Pairwise H):", flush=True)
    for dish, row in merged.nlargest(10, "H_pairwise").iterrows():
        print(f"    {dish:30s} Pairwise={row['H_pairwise']:.2f}  BERT={row['H_bert']:.2f}",
              flush=True)
    print(f"\n  Bottom 10 (Pairwise H):", flush=True)
    for dish, row in merged.nsmallest(10, "H_pairwise").iterrows():
        print(f"    {dish:30s} Pairwise={row['H_pairwise']:.2f}  BERT={row['H_bert']:.2f}",
              flush=True)

    # Biggest rank changes
    merged["rank_pw"] = merged["H_pairwise"].rank(ascending=False)
    merged["rank_bt"] = merged["H_bert"].rank(ascending=False)
    merged["rank_change"] = merged["rank_bt"] - merged["rank_pw"]
    movers = merged.reindex(merged["rank_change"].abs().nlargest(10).index)
    print(f"\n  Biggest rank movers:", flush=True)
    for dish, row in movers.iterrows():
        d = "↑" if row["rank_change"] > 0 else "↓"
        print(f"    {dish:30s} {d}{abs(row['rank_change']):.0f} "
              f"(BERT #{row['rank_bt']:.0f} → Pairwise #{row['rank_pw']:.0f})",
              flush=True)

    # Save
    merged.round(4).to_csv(DATA / "dish_h_pairwise.csv")
    print(f"\nSaved: {DATA / 'dish_h_pairwise.csv'}", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Scatter BERT vs Pairwise
    ax = axes[0]
    ax.scatter(h_bt, h_pw, alpha=0.5, s=25, c="#2077B4")
    sl, ic = np.polyfit(h_bt, h_pw, 1)
    xr = np.linspace(h_bt.min(), h_bt.max(), 50)
    ax.plot(xr, sl * xr + ic, "r--", lw=1.5, alpha=0.7)
    ax.set_xlabel("H (BERT, absolute scoring)", fontsize=11)
    ax.set_ylabel("H (Pairwise, Bradley-Terry)", fontsize=11)
    ax.set_title(f"ρ = {rho:.3f}, r = {r:.3f}", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 2: Distribution comparison
    ax = axes[1]
    ax.hist(h_bt, bins=25, alpha=0.5, label=f"BERT (CV={cv_bt:.1f}%)",
            color="#2077B4", density=True)
    ax.hist(h_pw, bins=25, alpha=0.5, label=f"Pairwise (CV={cv_pw:.1f}%)",
            color="#E87D2F", density=True)
    ax.set_xlabel("H score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Score Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Rank comparison
    ax = axes[2]
    ax.scatter(merged["rank_bt"], merged["rank_pw"], alpha=0.5, s=25, c="#2077B4")
    ax.plot([1, len(merged)], [1, len(merged)], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Rank (BERT)", fontsize=11)
    ax.set_ylabel("Rank (Pairwise)", fontsize=11)
    ax.set_title("Rank Comparison", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle("BERT Absolute Scoring vs Pairwise LLM Ranking",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "pairwise_vs_bert_h.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {FIGURES / 'pairwise_vs_bert_h.png'}", flush=True)
    plt.close()

    return merged


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60, flush=True)
    print("17 — Pairwise LLM Ranking → Bradley-Terry H", flush=True)
    print("=" * 60, flush=True)

    if not API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable", flush=True)
        return

    # Load scored mentions
    scored = pd.read_parquet(DATA / "dish_mentions_scored.parquet")
    print(f"Loaded {len(scored):,} scored mentions", flush=True)

    # Step 1: Representative reviews
    print("\n── Step 1: Selecting representative reviews ──", flush=True)
    profiles = select_representative_reviews(scored)
    dish_ids = sorted(profiles.keys())
    print(f"  {len(dish_ids)} dishes with review profiles", flush=True)

    # Check for cached wins
    wins_path = DATA / "pairwise_wins.csv"
    if wins_path.exists():
        print(f"\n  [cache] Loading existing wins from {wins_path}", flush=True)
        wins_df = pd.read_csv(wins_path)
        print(f"  Loaded {len(wins_df)} existing wins", flush=True)

        # Check coverage
        done_pairs = set()
        for _, r in wins_df.iterrows():
            pair = tuple(sorted([r["winner"], r["loser"]]))
            done_pairs.add(pair)

        all_pairs = list(itertools.combinations(dish_ids, 2))
        remaining = [(a, b) for a, b in all_pairs if tuple(sorted([a, b])) not in done_pairs]
        print(f"  {len(done_pairs)} pairs done, {len(remaining)} remaining", flush=True)
    else:
        wins_df = pd.DataFrame(columns=["winner", "loser"])
        all_pairs = list(itertools.combinations(dish_ids, 2))
        remaining = all_pairs
        print(f"  {len(all_pairs)} total pairs to compare", flush=True)

    # Step 2: Run pairwise comparisons
    if remaining:
        print(f"\n── Step 2: Running {len(remaining)} pairwise comparisons ──",
              flush=True)
        print(f"  Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}", flush=True)

        random.seed(42)
        random.shuffle(remaining)

        # Split into batches
        batches = []
        for i in range(0, len(remaining), BATCH_SIZE):
            batches.append(remaining[i:i + BATCH_SIZE])

        print(f"  {len(batches)} API calls needed", flush=True)

        all_wins = list(wins_df.to_dict("records"))
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for batch_idx, batch in enumerate(batches):
                f = executor.submit(process_batch, batch_idx, batch, profiles)
                futures[f] = batch_idx

            for f in as_completed(futures):
                batch_idx, results = f.result()
                all_wins.extend(results)
                completed += 1
                if not results:
                    failed += 1
                if completed % 50 == 0 or completed == len(batches):
                    pct = completed / len(batches) * 100
                    print(f"    {completed}/{len(batches)} ({pct:.0f}%) — "
                          f"{len(all_wins)} wins, {failed} failed batches",
                          flush=True)

                # Periodic save
                if completed % 200 == 0:
                    pd.DataFrame(all_wins).to_csv(wins_path, index=False)

        wins_df = pd.DataFrame(all_wins)
        wins_df.to_csv(wins_path, index=False)
        print(f"\n  Saved {len(wins_df)} wins to {wins_path}", flush=True)

    # Step 3: Fit Bradley-Terry
    print(f"\n── Step 3: Fitting Bradley-Terry model ──", flush=True)
    bt_scores = fit_bradley_terry(wins_df, dish_ids)

    # Step 4: Compare with BERT H
    print(f"\n── Step 4: Analysis ──", flush=True)
    dei = pd.read_csv(DATA / "dish_DEI_scores.csv").set_index("dish_id")
    analyze_and_plot(bt_scores, dei)

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
