#!/usr/bin/env python3
"""
22b_llm_recipes.py — Generate standardised recipes via LLM (concurrent, robust)
"""
import pandas as pd, json, os, sys, time, re, requests, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ENV_INGREDIENTS = pd.read_csv(DATA / "ingredient_impact_factors.csv")["ingredient"].tolist()
ENV_INGREDIENTS_STR = ", ".join(sorted(ENV_INGREDIENTS))

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"
WORKERS = 100

SYSTEM_PROMPT = f"""You are a culinary expert. Given a dish name, cuisine, and description,
produce a standardised recipe as JSON.

Map all ingredients to the closest match from this list:
{ENV_INGREDIENTS_STR}

If no close match, use the closest substitute (e.g. "ghee"→"butter", "paneer"→"cheese", "naan"→"wheat_flour").

Output ONLY valid JSON:
{{"ingredients": [{{"name": "ingredient_name", "grams": 150}}, ...],
  "cook_method": "raw|boil|steam|fry|deep_fry|bake|grill|roast|braise|saute|smoke|ferment|no_cook",
  "total_grams": 350, "calories_approx": 450, "protein_g_approx": 25}}

Rules: Use ONLY names from the list. 3-12 ingredients. Realistic single-serving grams."""


def call_llm(dish):
    user_msg = (f"Dish: {dish['name']}\nCuisine: {dish['primary_cuisine']}\n"
                f"Category: {dish['primary_coarse']}\n"
                f"Description: {str(dish.get('description',''))[:300]}\n\n"
                f"Generate the recipe JSON.")
    for attempt in range(3):
        try:
            resp = requests.post(API_URL,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}],
                    "temperature": 0.1, "max_tokens": 500},
                timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                recipe = json.loads(m.group())
                recipe["dish_id"] = dish["dish_id"]
                recipe["status"] = "ok"
                return recipe
            return {"dish_id": dish["dish_id"], "status": "parse_error",
                    "raw": content[:200]}
        except Exception as e:
            if attempt == 2:
                return {"dish_id": dish["dish_id"], "status": "error",
                        "error": str(e)[:100]}
            time.sleep(1)
    return {"dish_id": dish["dish_id"], "status": "error", "error": "max_retries"}


# Thread-safe writer
write_lock = threading.Lock()
counter = {"ok": 0, "err": 0, "total": 0}


def process_and_write(dish, f):
    result = call_llm(dish)
    with write_lock:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        if result["status"] == "ok":
            counter["ok"] += 1
        else:
            counter["err"] += 1
        counter["total"] += 1
    return result


def main():
    if not API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY"); sys.exit(1)

    df = pd.read_csv(DATA / "worldcuisines_matched.csv")
    print(f"Total dishes: {len(df)}", flush=True)

    ckpt = DATA / "llm_recipes_v2.jsonl"
    done = set()
    if ckpt.exists() and ckpt.stat().st_size > 0:
        with open(ckpt) as f:
            for line in f:
                try: done.add(json.loads(line)["dish_id"])
                except: pass
        print(f"Checkpoint: {len(done)} done", flush=True)

    remaining = df[~df["dish_id"].isin(done)].to_dict("records")
    print(f"Remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("All done!"); compile_results(); return

    t0 = time.time()
    with open(ckpt, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(process_and_write, dish, f): dish
                       for dish in remaining}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  Worker error: {e}", flush=True)
                n = counter["total"]
                if n % 100 == 0 and n > 0:
                    elapsed = time.time() - t0
                    rate = n / elapsed
                    eta = (len(remaining) - n) / rate / 60
                    print(f"  [{n}/{len(remaining)}] ok={counter['ok']} "
                          f"err={counter['err']} {rate:.1f}/s "
                          f"ETA={eta:.1f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! ok={counter['ok']} err={counter['err']} "
          f"in {elapsed/60:.1f}min", flush=True)
    compile_results()


def compile_results():
    ckpt = DATA / "llm_recipes_v2.jsonl"
    env_set = {e.lower() for e in ENV_INGREDIENTS}
    recipes = []
    with open(ckpt) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("status") != "ok" or "ingredients" not in rec:
                    continue
                valid = [{"name": i["name"].lower().strip(),
                          "grams": i.get("grams", 0)}
                         for i in rec["ingredients"]
                         if i.get("name","").lower().strip() in env_set
                         and i.get("grams", 0) > 0]
                if valid:
                    recipes.append({
                        "dish_id": rec["dish_id"],
                        "ingredients_json": json.dumps(valid),
                        "n_ingredients": len(valid),
                        "cook_method": rec.get("cook_method", "unknown"),
                        "total_grams": rec.get("total_grams",
                                               sum(v["grams"] for v in valid)),
                        "calories_approx": rec.get("calories_approx", 0),
                        "protein_g_approx": rec.get("protein_g_approx", 0),
                    })
            except:
                pass

    out = pd.DataFrame(recipes).drop_duplicates("dish_id", keep="last")
    out.to_csv(DATA / "llm_generated_recipes.csv", index=False)
    print(f"\nCompiled {len(out)} recipes → data/llm_generated_recipes.csv",
          flush=True)
    print(f"  Mean ings: {out['n_ingredients'].mean():.1f}, "
          f"Mean grams: {out['total_grams'].mean():.0f}", flush=True)


if __name__ == "__main__":
    main()
