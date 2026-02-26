#!/usr/bin/env python3
"""
22d_pairwise_expansion_v2.py — Extend pairwise ranking to ~2,500+ dishes
========================================================================
Uses anchor-bridging: 334 dishes with existing H scores serve as anchors.
Each new dish is compared against 20 anchor dishes spanning the full H range.
All wins are then jointly fitted with a Bradley-Terry model.

For dishes without Yelp reviews, we use dish name + cuisine + description.

Strategy:
  - Select 20 anchors from existing 334 (evenly spaced by H)
  - Each new dish compared against 20 anchors → ~2,229×20 = ~44,580 pairs
  - Batch 10/call → ~4,458 API calls
  - 50 concurrent workers → ~15 min

Input:
  data/all_dishes_master.csv        (2,563 dishes, 334 with H)
  data/worldcuisines_matched.csv    (new dish descriptions)
  data/pairwise_wins_all.csv        (existing 17,763 wins)
  data/dish_h_pairwise_all.csv      (existing 337 H scores)

Output:
  data/pairwise_wins_v2.csv         (all wins including expansion)
  data/dish_h_pairwise_v2.csv       (all dishes, unified BT scale)
"""

import sys, os, json, time, random, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"

N_ANCHORS = 20          # anchor dishes from existing set
BATCH_SIZE = 10         # pairs per API call
MAX_WORKERS = 50        # concurrent API calls


# ══════════════════════════════════════════════════════════════════
# Build dish profiles for comparison
# ══════════════════════════════════════════════════════════════════
def build_profiles(master_df, meta_df):
    """Build text profiles for all dishes."""
    profiles = {}

    # For WorldCuisines dishes: use name + cuisine + description
    meta_map = {}
    for _, row in meta_df.iterrows():
        meta_map[row["dish_id"]] = row

    for _, row in master_df.iterrows():
        did = row["dish_id"]
        name = str(row.get("name", did)).replace("_", " ").title()
        cuisine = str(row.get("primary_cuisine", "")).strip()

        if did in meta_map:
            meta = meta_map[did]
            desc = str(meta.get("description", ""))[:300]
            profile = f"{name} ({cuisine} cuisine): {desc}"
        else:
            # Existing dish without WorldCuisines entry
            profile = f"{name} ({cuisine} cuisine)"

        profiles[did] = profile

    return profiles


# ══════════════════════════════════════════════════════════════════
# Anchor selection
# ══════════════════════════════════════════════════════════════════
def select_anchors(master_df, n_anchors=N_ANCHORS):
    """Select anchors from dishes with H scores, evenly spaced."""
    with_h = master_df.dropna(subset=["H_mean"]).sort_values("H_mean")
    n = len(with_h)
    idx = np.linspace(0, n - 1, n_anchors, dtype=int)
    anchors = with_h.iloc[idx]["dish_id"].tolist()
    h_vals = with_h.iloc[idx]["H_mean"].values
    print(f"  Selected {len(anchors)} anchors spanning H=[{h_vals[0]:.2f}, {h_vals[-1]:.2f}]",
          flush=True)
    return anchors


# ══════════════════════════════════════════════════════════════════
# API calling
# ══════════════════════════════════════════════════════════════════
def make_batch_prompt(pairs_with_profiles):
    lines = [
        "You are a food expert. For each numbered pair, decide which dish sounds "
        "MORE DELICIOUS based on the description and your culinary knowledge. "
        "Consider flavor complexity, texture, freshness, and overall enjoyment. "
        "Answer with ONLY the dish letter (A or B) for each pair, one per line. "
        "Format: 1:A  2:B  3:A  etc.\n"
    ]
    for i, (dish_a, dish_b, prof_a, prof_b) in enumerate(pairs_with_profiles, 1):
        lines.append(
            f"Pair {i}:\n"
            f"  A: {prof_a}\n"
            f"  B: {prof_b}\n"
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
    import requests
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.1, "max_tokens": 200},
                timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


# Thread-safe writer
write_lock = threading.Lock()
counter = {"done": 0, "wins": 0, "failed": 0}


def process_batch(batch_idx, pairs_batch, profiles, fout):
    pairs_with_profiles = []
    for dish_a, dish_b in pairs_batch:
        prof_a = profiles.get(dish_a, dish_a)
        prof_b = profiles.get(dish_b, dish_b)
        # Randomize order to avoid position bias
        if random.random() < 0.5:
            pairs_with_profiles.append((dish_a, dish_b, prof_a, prof_b))
        else:
            pairs_with_profiles.append((dish_b, dish_a, prof_b, prof_a))

    prompt = make_batch_prompt(pairs_with_profiles)
    response = call_api(prompt)

    results = []
    if response:
        parsed = parse_batch_response(response, len(pairs_batch))
        for i, (dish_a, dish_b, _, _) in enumerate(pairs_with_profiles, 1):
            choice = parsed.get(i)
            if choice == "A":
                results.append({"winner": dish_a, "loser": dish_b})
            elif choice == "B":
                results.append({"winner": dish_b, "loser": dish_a})

    with write_lock:
        for r in results:
            fout.write(f"{r['winner']},{r['loser']}\n")
        fout.flush()
        counter["done"] += 1
        counter["wins"] += len(results)
        if not results:
            counter["failed"] += 1

    return batch_idx, results


# ══════════════════════════════════════════════════════════════════
# Bradley-Terry fitting
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
    print("22d — Pairwise Expansion: 334 → 2,500+ dishes via anchor bridging",
          flush=True)
    print("=" * 70, flush=True)

    if not API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable", flush=True)
        return

    # ── Load data ────────────────────────────────────────────────
    print("\n── Loading data ──", flush=True)

    master = pd.read_csv(DATA / "all_dishes_master.csv")
    meta = pd.read_csv(DATA / "worldcuisines_matched.csv")
    print(f"  Master list: {len(master)} dishes", flush=True)
    print(f"  With H: {master['H_mean'].notna().sum()}", flush=True)
    print(f"  Need H: {master['H_mean'].isna().sum()}", flush=True)

    # ── Build profiles ───────────────────────────────────────────
    print("\n── Building profiles ──", flush=True)
    profiles = build_profiles(master, meta)
    print(f"  Profiles: {len(profiles)} dishes", flush=True)

    # ── Select anchors ───────────────────────────────────────────
    print("\n── Selecting anchors ──", flush=True)
    anchors = select_anchors(master, N_ANCHORS)

    # ── Generate pairs ───────────────────────────────────────────
    new_dishes = master[master["H_mean"].isna()]["dish_id"].tolist()
    print(f"\n  New dishes: {len(new_dishes)}", flush=True)

    all_pairs = []
    for new_dish in new_dishes:
        if new_dish not in profiles:
            continue
        for anchor in anchors:
            all_pairs.append((new_dish, anchor))

    print(f"  Total pairs: {len(all_pairs)}", flush=True)

    # ── Check checkpoint ─────────────────────────────────────────
    ckpt_path = DATA / "pairwise_wins_expansion_v2.csv"
    done_pairs = set()
    if ckpt_path.exists() and ckpt_path.stat().st_size > 10:
        existing_wins = pd.read_csv(ckpt_path)
        for _, r in existing_wins.iterrows():
            pair = tuple(sorted([r["winner"], r["loser"]]))
            done_pairs.add(pair)
        print(f"  Checkpoint: {len(done_pairs)} pairs done", flush=True)

    remaining = [(a, b) for a, b in all_pairs
                 if tuple(sorted([a, b])) not in done_pairs]
    print(f"  Remaining: {len(remaining)} pairs", flush=True)

    if not remaining:
        print("  All comparisons done!", flush=True)
    else:
        # ── Run comparisons ──────────────────────────────────────
        random.seed(42)
        random.shuffle(remaining)

        batches = [remaining[i:i+BATCH_SIZE]
                   for i in range(0, len(remaining), BATCH_SIZE)]
        print(f"\n── Running {len(batches)} API calls with {MAX_WORKERS} workers ──",
              flush=True)

        t0 = time.time()
        with open(ckpt_path, "a", encoding="utf-8") as fout:
            # Write header if new file
            if not done_pairs:
                fout.write("winner,loser\n")
                fout.flush()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(process_batch, i, batch, profiles, fout): i
                           for i, batch in enumerate(batches)}

                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"  Worker error: {e}", flush=True)

                    n = counter["done"]
                    if n % 200 == 0 and n > 0:
                        elapsed = time.time() - t0
                        rate = n / elapsed
                        eta = (len(batches) - n) / rate / 60
                        print(f"  [{n}/{len(batches)}] wins={counter['wins']} "
                              f"failed={counter['failed']} {rate:.1f}/s "
                              f"ETA={eta:.1f}min", flush=True)

        elapsed = time.time() - t0
        print(f"\n  Done! {counter['wins']} wins from {counter['done']} batches "
              f"in {elapsed/60:.1f}min", flush=True)

    # ── Merge all wins ───────────────────────────────────────────
    print("\n── Merging wins ──", flush=True)

    # Original wins
    wins_orig = pd.read_csv(DATA / "pairwise_wins_all.csv")
    print(f"  Original+prev expansion wins: {len(wins_orig)}", flush=True)

    # New expansion wins
    wins_new = pd.read_csv(ckpt_path)
    print(f"  New expansion wins: {len(wins_new)}", flush=True)

    wins_all = pd.concat([wins_orig, wins_new], ignore_index=True)
    wins_all.to_csv(DATA / "pairwise_wins_v2.csv", index=False)
    print(f"  Total wins: {len(wins_all)}", flush=True)

    # ── Bradley-Terry fit ────────────────────────────────────────
    print("\n── Fitting global Bradley-Terry model ──", flush=True)

    # All dishes that appear in any comparison
    all_dish_ids = sorted(set(wins_all["winner"]) | set(wins_all["loser"]))
    print(f"  Dishes in comparisons: {len(all_dish_ids)}", flush=True)

    bt = fit_bradley_terry(wins_all, all_dish_ids)

    # Stats
    h = bt["H_pairwise"]
    print(f"\n  H range: [{h.min():.2f}, {h.max():.2f}]", flush=True)
    print(f"  H mean:  {h.mean():.2f}, std: {h.std():.2f}, CV: {h.std()/h.mean()*100:.1f}%",
          flush=True)

    bt.round(4).to_csv(DATA / "dish_h_pairwise_v2.csv")
    print(f"\n  Saved: data/dish_h_pairwise_v2.csv ({len(bt)} dishes)", flush=True)

    # ── Stability check ──────────────────────────────────────────
    from scipy import stats as sp_stats

    # Compare existing dishes' H before vs after expansion
    existing_h = pd.read_csv(DATA / "dish_h_pairwise_all.csv", index_col="dish_id")
    common = bt.index.intersection(existing_h.index)
    if len(common) > 10:
        rho, _ = sp_stats.spearmanr(
            existing_h.loc[common, "H_pairwise"],
            bt.loc[common, "H_pairwise"]
        )
        print(f"\n  Stability check ({len(common)} existing dishes):", flush=True)
        print(f"    Spearman ρ(old H, new H): {rho:.4f}", flush=True)

    # ── Top/Bottom ───────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("TOP 20 DISHES by H_pairwise", flush=True)
    for i, (idx, row) in enumerate(bt.nlargest(20, "H_pairwise").iterrows(), 1):
        is_new = idx not in existing_h.index
        tag = "★" if is_new else " "
        print(f"  {i:2d}. {tag} {idx:<35s} H={row['H_pairwise']:.2f}",
              flush=True)

    print(f"\nBOTTOM 20 DISHES by H_pairwise", flush=True)
    for i, (idx, row) in enumerate(bt.nsmallest(20, "H_pairwise").iterrows(), 1):
        is_new = idx not in existing_h.index
        tag = "★" if is_new else " "
        print(f"  {i:2d}. {tag} {idx:<35s} H={row['H_pairwise']:.2f}",
              flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
