"""
16_cross_platform_bert_scoring.py
=================================
Apply finetuned BERT to Google Local & TripAdvisor review text,
producing true NLP-based H scores (not raw star ratings).

Pipeline:
  1. Stream reviews → find dish mentions → extract ±2 sentence context
  2. Batch-score all contexts with finetuned BERT
  3. Aggregate to dish-level H (mean, n, CI)
  4. Compute cross-platform correlations (BERT-BERT)
  5. Generate validation figures (convergent + discriminant)

Input:
  raw/external/google_local/image_review_all.json
  raw/external/tripadvisor/*_reviews.csv
  models/hedonic_bert_finetuned/
Output:
  data/google_mentions_bert.parquet
  data/tripadvisor_mentions_bert.parquet
  data/cross_platform_h_bert.csv
  results/tables/cross_platform_bert_correlations.csv
  results/figures/cross_platform_convergent.png
  results/figures/cross_platform_discriminant.png
"""

import sys, json, re, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXT = ROOT / "raw" / "external"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODEL_DIR = ROOT / "models" / "hedonic_bert_finetuned"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load DEI dishes ──────────────────────────────────────────────
dei = pd.read_csv(DATA / "dish_DEI_scores.csv")
DISH_IDS = set(dei["dish_id"].tolist())

# ── Build dish alias map (reuse from 07c) ────────────────────────
sys.path.insert(0, str(ROOT / "code"))

DISH_ALIASES = {}
for d in DISH_IDS:
    canonical = d.lower().replace("_", " ")
    DISH_ALIASES[canonical] = d
    DISH_ALIASES[d] = d
    DISH_ALIASES[d.replace("_", " ")] = d
    if canonical.endswith("s") and len(canonical) > 4:
        DISH_ALIASES[canonical[:-1]] = d
    if not canonical.endswith("s"):
        DISH_ALIASES[canonical + "s"] = d

# Manual aliases (same as 07c)
EXTRA = {
    "pad thai": "pad_thai", "pad see ew": "pad_see_ew",
    "tom yum": "tom_yum", "tom kha": "tom_kha",
    "pho bo": "pho", "pho ga": "pho",
    "bahn mi": "banh_mi", "banh mi sandwich": "banh_mi",
    "macaroni and cheese": "mac_and_cheese", "mac & cheese": "mac_and_cheese",
    "mac n cheese": "mac_and_cheese",
    "grilled cheese sandwich": "grilled_cheese",
    "general tso chicken": "general_tso", "general tso's chicken": "general_tso",
    "general tso's": "general_tso",
    "kung pao": "kung_pao_chicken", "kung pao chicken": "kung_pao_chicken",
    "chicken tikka masala": "tikka_masala", "tikka masala": "tikka_masala",
    "butter chicken": "butter_chicken",
    "tandoori": "tandoori_chicken", "tandoori chicken": "tandoori_chicken",
    "palak paneer": "palak_paneer", "saag paneer": "palak_paneer",
    "chana masala": "chana_masala", "chickpea curry": "chana_masala",
    "dal": "dal", "daal": "dal", "dhal": "dal",
    "dosa": "dosa", "masala dosa": "dosa",
    "naan bread": "naan", "naan": "naan",
    "samosa": "samosa", "samosas": "samosa",
    "biryani": "biryani", "chicken biryani": "biryani", "lamb biryani": "biryani",
    "korma": "korma", "chicken korma": "korma",
    "vindaloo": "vindaloo", "chicken vindaloo": "vindaloo",
    "raita": "raita", "cucumber raita": "raita",
    "hamburger": "hamburger", "burger": "hamburger", "cheeseburger": "hamburger",
    "steak": "steak", "ribeye": "steak", "filet mignon": "steak",
    "ribs": "ribs", "baby back ribs": "ribs", "spare ribs": "ribs",
    "bbq ribs": "ribs", "barbecue ribs": "ribs",
    "brisket": "brisket", "beef brisket": "brisket", "smoked brisket": "brisket",
    "pulled pork": "pulled_pork", "pulled pork sandwich": "pulled_pork",
    "fried chicken": "fried_chicken",
    "buffalo wings": "buffalo_wings", "chicken wings": "buffalo_wings",
    "wings": "buffalo_wings",
    "lobster roll": "lobster_roll",
    "fish and chips": "fish_and_chips", "fish & chips": "fish_and_chips",
    "french fries": "french_fries", "fries": "french_fries",
    "clam chowder": "clam_chowder", "new england clam chowder": "clam_chowder",
    "french onion soup": "french_onion_soup", "onion soup": "french_onion_soup",
    "caesar salad": "caesar_salad", "caesar": "caesar_salad",
    "coleslaw": "coleslaw", "cole slaw": "coleslaw",
    "nachos": "nachos",
    "guacamole": "guacamole", "guac": "guacamole",
    "burrito": "burrito", "burritos": "burrito",
    "taco": "taco", "tacos": "taco",
    "quesadilla": "quesadilla", "quesadillas": "quesadilla",
    "enchilada": "enchilada", "enchiladas": "enchilada",
    "fajita": "fajita", "fajitas": "fajita",
    "tamale": "tamale", "tamales": "tamale",
    "churro": "churro", "churros": "churro",
    "pozole": "pozole", "posole": "pozole",
    "pizza": "pizza", "pepperoni pizza": "pizza",
    "margherita pizza": "pizza_margherita", "margherita": "pizza_margherita",
    "lasagna": "lasagna", "lasagne": "lasagna",
    "pasta bolognese": "pasta_bolognese", "bolognese": "pasta_bolognese",
    "spaghetti bolognese": "pasta_bolognese",
    "carbonara": "pasta_carbonara", "spaghetti carbonara": "pasta_carbonara",
    "pasta carbonara": "pasta_carbonara",
    "gnocchi": "gnocchi", "ravioli": "ravioli", "risotto": "risotto",
    "tiramisu": "tiramisu", "gelato": "gelato",
    "panna cotta": "panna_cotta", "pannacotta": "panna_cotta",
    "bruschetta": "bruschetta",
    "caprese": "caprese", "caprese salad": "caprese",
    "minestrone": "minestrone",
    "osso buco": "osso_buco", "ossobuco": "osso_buco",
    "sushi": "sushi", "sashimi": "sashimi", "ramen": "ramen",
    "udon": "udon", "tempura": "tempura",
    "teriyaki": "teriyaki", "teriyaki chicken": "teriyaki",
    "miso soup": "miso_soup", "miso": "miso_soup",
    "edamame": "edamame", "takoyaki": "takoyaki",
    "onigiri": "onigiri", "rice ball": "onigiri",
    "katsu": "katsu", "tonkatsu": "katsu", "chicken katsu": "katsu",
    "donburi": "donburi", "okonomiyaki": "okonomiyaki", "yakitori": "yakitori",
    "kimchi": "kimchi", "bibimbap": "bibimbap", "bulgogi": "bulgogi",
    "japchae": "japchae", "kimbap": "kimbap", "gimbap": "kimbap",
    "tteokbokki": "tteokbokki",
    "korean fried chicken": "korean_fried_chicken",
    "kimchi jjigae": "kimchi_jjigae", "kimchi stew": "kimchi_jjigae",
    "sundubu jjigae": "sundubu_jjigae", "sundubu": "sundubu_jjigae",
    "green curry": "green_curry", "thai green curry": "green_curry",
    "red curry": "red_curry", "thai red curry": "red_curry",
    "massaman curry": "massaman_curry", "massaman": "massaman_curry",
    "papaya salad": "papaya_salad", "som tam": "papaya_salad",
    "thai iced tea": "thai_iced_tea",
    "mango sticky rice": "mango_sticky_rice",
    "satay": "satay", "chicken satay": "satay",
    "larb": "larb", "laab": "larb",
    "spring roll": "spring_rolls", "spring rolls": "spring_rolls",
    "egg roll": "spring_rolls",
    "fresh spring roll": "fresh_spring_roll", "summer roll": "fresh_spring_roll",
    "pho": "pho", "bun bo hue": "bun_bo_hue", "bun cha": "bun_cha",
    "com tam": "com_tam", "broken rice": "com_tam",
    "ca phe sua da": "ca_phe_sua_da", "vietnamese coffee": "ca_phe_sua_da",
    "vietnamese iced coffee": "ca_phe_sua_da",
    "peking duck": "peking_duck", "beijing duck": "peking_duck",
    "dim sum": "dim_sum",
    "dumplings": "dumplings", "pot stickers": "dumplings",
    "potstickers": "dumplings", "gyoza": "dumplings",
    "wonton soup": "wonton_soup", "wonton": "wonton_soup",
    "xiao long bao": "xiao_long_bao", "soup dumplings": "xiao_long_bao",
    "xlb": "xiao_long_bao",
    "hot and sour soup": "hot_and_sour_soup",
    "hot & sour soup": "hot_and_sour_soup",
    "mapo tofu": "mapo_tofu",
    "dan dan noodles": "dan_dan_noodles",
    "chow mein": "chow_mein", "chow fun": "chow_fun",
    "lo mein": "lo_mein", "fried rice": "fried_rice",
    "scallion pancake": "scallion_pancake",
    "char siu": "char_siu", "bbq pork": "char_siu",
    "congee": "congee", "rice porridge": "congee",
    "coq au vin": "coq_au_vin",
    "escargot": "escargot", "escargots": "escargot",
    "crepe": "crepe", "crepes": "crepe",
    "croissant": "croissant", "quiche": "quiche",
    "souffle": "souffle", "ratatouille": "ratatouille",
    "creme brulee": "creme_brulee",
    "paella": "paella", "patatas bravas": "patatas_bravas",
    "gazpacho": "gazpacho", "tapas": "tapas",
    "spanakopita": "spanakopita", "falafel": "falafel",
    "hummus": "hummus", "gyro": "gyro", "gyros": "gyro",
    "shawarma": "shawarma",
    "kebab": "kebab", "kebabs": "kebab", "kabob": "kebab",
    "baba ganoush": "baba_ganoush", "baba ghanoush": "baba_ganoush",
    "tabbouleh": "tabbouleh", "tabouleh": "tabbouleh",
    "dolma": "dolma", "dolmas": "dolma", "stuffed grape leaves": "dolma",
    "moussaka": "moussaka", "pita": "pita", "pita bread": "pita",
    "baklava": "baklava",
    "pancake": "pancake", "pancakes": "pancake",
    "waffle": "waffle", "waffles": "waffle",
    "brownie": "brownie", "brownies": "brownie",
    "cheesecake": "cheesecake", "cornbread": "cornbread",
    "grilled cheese": "grilled_cheese", "club sandwich": "club_sandwich",
    "ice cream": "ice_cream", "beef bourguignon": "beef_bourguignon",
}
DISH_ALIASES.update(EXTRA)

# Sorted longest-first for greedy scanning
_ALIAS_SCAN = sorted(
    [(a, DISH_ALIASES[a]) for a in DISH_ALIASES if len(a) >= 4],
    key=lambda x: -len(x[0])
)

# ── Sentence splitting + context extraction ──────────────────────
_SENT_RE = re.compile(r'(?<=[.!?])\s+')

def extract_context(text, alias, window=2):
    """Extract ±window sentences around the dish mention."""
    sents = _SENT_RE.split(text)
    if len(sents) <= 1:
        return text[:512]
    alias_lower = alias.lower()
    for i, s in enumerate(sents):
        if alias_lower in s.lower():
            lo = max(0, i - window)
            hi = min(len(sents), i + window + 1)
            return " ".join(sents[lo:hi])[:512]
    return text[:512]


def find_dishes_with_context(text):
    """Find dish mentions and extract context for each."""
    text_lower = text.lower()
    results = []
    found_ids = set()
    for alias, dish_id in _ALIAS_SCAN:
        if dish_id in found_ids:
            continue
        idx = text_lower.find(alias)
        while idx != -1:
            before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
            end = idx + len(alias)
            after_ok = (end >= len(text_lower) or not text_lower[end].isalnum())
            if before_ok and after_ok:
                ctx = extract_context(text, alias)
                results.append((dish_id, ctx))
                found_ids.add(dish_id)
                break
            idx = text_lower.find(alias, idx + 1)
    return results


# ── BERT scoring ─────────────────────────────────────────────────
def load_bert_model():
    """Load finetuned BERT model once."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Loading finetuned BERT from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")
    return tokenizer, model, device


def score_batch(texts, tokenizer, model, device, batch_size=128):
    """Score a list of texts with finetuned BERT. Returns numpy array."""
    import torch
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:512] for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            preds = model(**enc).logits.squeeze(-1).cpu().numpy()
        if preds.ndim == 0:
            all_preds.append(float(preds))
        else:
            all_preds.extend(preds.tolist())
    return np.clip(np.array(all_preds), 1.0, 10.0)


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Extract dish mentions with context
# ══════════════════════════════════════════════════════════════════
def extract_google_mentions():
    """Scan Google Local reviews, extract dish mention contexts."""
    out_path = DATA / "google_mentions_bert.parquet"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists, loading...")
        return pd.read_parquet(out_path)

    print("\n" + "=" * 60)
    print("PHASE 1a: Extract Google Local dish mentions")
    print("=" * 60)

    google_file = EXT / "google_local" / "image_review_all.json"
    mentions = []
    n_scanned = 0

    with open(google_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except (json.JSONDecodeError, Exception):
                continue
            n_scanned += 1
            text = review.get("review_text", "")
            rating = review.get("rating")
            if not text or not isinstance(text, str) or len(text) < 20:
                continue

            hits = find_dishes_with_context(text)
            for dish_id, ctx in hits:
                mentions.append({
                    "dish_id": dish_id,
                    "context_text": ctx,
                    "star_rating": float(rating) if rating else np.nan,
                    "platform": "google",
                })

            if n_scanned % 500000 == 0:
                print(f"  {n_scanned:,} reviews → {len(mentions):,} mentions")

    print(f"  Total: {n_scanned:,} reviews → {len(mentions):,} mentions")
    print(f"  Unique dishes: {len(set(m['dish_id'] for m in mentions))}")

    df = pd.DataFrame(mentions)
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    return df


def extract_tripadvisor_mentions():
    """Scan TripAdvisor reviews, extract dish mention contexts."""
    out_path = DATA / "tripadvisor_mentions_bert.parquet"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists, loading...")
        return pd.read_parquet(out_path)

    print("\n" + "=" * 60)
    print("PHASE 1b: Extract TripAdvisor dish mentions")
    print("=" * 60)

    ta_dir = EXT / "tripadvisor"
    ta_files = sorted(ta_dir.glob("*_reviews.csv"))
    mentions = []

    for f in ta_files:
        city = f.stem.replace("_reviews", "")
        n_city = 0
        for chunk in pd.read_csv(f, chunksize=20000,
                                  usecols=["review_full", "rating_review"],
                                  on_bad_lines="skip"):
            for text, rating in zip(chunk["review_full"], chunk["rating_review"]):
                if not isinstance(text, str) or len(text) < 20:
                    continue
                hits = find_dishes_with_context(text)
                for dish_id, ctx in hits:
                    mentions.append({
                        "dish_id": dish_id,
                        "context_text": ctx,
                        "star_rating": float(rating) if not pd.isna(rating) else np.nan,
                        "platform": "tripadvisor",
                        "city": city,
                    })
            n_city += len(chunk)
        print(f"  {city}: {n_city:,} reviews → {len(mentions):,} mentions cumulative")

    print(f"  Total mentions: {len(mentions):,}")
    print(f"  Unique dishes: {len(set(m['dish_id'] for m in mentions))}")

    df = pd.DataFrame(mentions)
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════
# PHASE 2: BERT scoring
# ══════════════════════════════════════════════════════════════════
MAX_PER_DISH = 1000  # Sample cap per dish — keeps total manageable on GPU

def sample_mentions(df, max_per_dish=MAX_PER_DISH):
    """Stratified sample: up to max_per_dish mentions per dish."""
    before = len(df)
    df = df.groupby("dish_id", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), max_per_dish), random_state=42)
    ).reset_index(drop=True)
    print(f"  Sampled {before:,} → {len(df):,} mentions "
          f"(max {max_per_dish}/dish)", flush=True)
    return df


def bert_score_mentions(df, platform_name, tokenizer, model, device):
    """Score mention contexts with finetuned BERT (GPU-accelerated)."""
    col = "H_bert"
    if col in df.columns and df[col].notna().all():
        print(f"  [{platform_name}] Already scored, skipping...", flush=True)
        return df

    print(f"\n  [{platform_name}] Scoring {len(df):,} mentions on {device}...",
          flush=True)
    texts = df["context_text"].tolist()

    # Process in mega-batches to show progress
    mega = 10000
    all_scores = []
    for start in range(0, len(texts), mega):
        end = min(start + mega, len(texts))
        batch_scores = score_batch(texts[start:end], tokenizer, model, device)
        all_scores.extend(batch_scores.tolist())
        pct = end / len(texts) * 100
        print(f"    {end:,}/{len(texts):,} ({pct:.0f}%) — "
              f"batch mean H={np.mean(batch_scores):.2f}", flush=True)

    df = df.copy()
    df[col] = all_scores
    return df


# ══════════════════════════════════════════════════════════════════
# PHASE 3: Aggregation + correlation + figures
# ══════════════════════════════════════════════════════════════════
def aggregate_dish_h(df, platform):
    """Aggregate mention-level BERT scores to dish-level H."""
    agg = (
        df.groupby("dish_id")["H_bert"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": f"H_{platform}_bert",
                         "std": f"H_{platform}_std",
                         "count": f"n_{platform}_bert"})
    )
    agg[f"H_{platform}_ci95"] = 1.96 * agg[f"H_{platform}_std"] / np.sqrt(agg[f"n_{platform}_bert"])
    # Also compute star-rating H for comparison
    if "star_rating" in df.columns:
        star_agg = (
            df.dropna(subset=["star_rating"])
            .groupby("dish_id")["star_rating"]
            .mean() * 2  # 1-5 → 1-10
        ).rename(f"H_{platform}_star")
        agg = agg.join(star_agg)
    return agg[agg[f"n_{platform}_bert"] >= 5]  # min 5 mentions


def compute_correlations_and_plot(yelp_h, google_agg, ta_agg):
    """Compute all correlations and generate figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Also load old Food.com/CroCoFiD/Food-Pics data for discriminant
    old_cp = pd.read_csv(DATA / "cross_platform_h_scores.csv", index_col="dish_id")

    # ── Merge ────────────────────────────────────────────────────
    merged = yelp_h[["H_mean"]].rename(columns={"H_mean": "H_yelp"}).copy()
    if google_agg is not None:
        merged = merged.join(google_agg)
    if ta_agg is not None:
        merged = merged.join(ta_agg)
    # Add old non-BERT sources for discriminant comparison
    for col in ["H_foodcom", "H_crocufid", "H_foodpics"]:
        if col in old_cp.columns:
            merged = merged.join(old_cp[[col]])

    merged.to_csv(DATA / "cross_platform_h_bert.csv")
    print(f"\nSaved: {DATA / 'cross_platform_h_bert.csv'}")

    # ── Correlation table ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CROSS-PLATFORM CORRELATIONS (BERT-scored)")
    print("=" * 60)

    corr_rows = []
    yelp_col = "H_yelp"

    pairs = []
    # Convergent: BERT-BERT
    for col, label in [
        ("H_google_bert", "Google (BERT)"),
        ("H_tripadvisor_bert", "TripAdvisor (BERT)"),
    ]:
        if col in merged.columns:
            pairs.append((col, label, "convergent"))
    # Also BERT vs star for comparison
    for col, label in [
        ("H_google_star", "Google (star×2)"),
        ("H_tripadvisor_star", "TripAdvisor (star×2)"),
    ]:
        if col in merged.columns:
            pairs.append((col, label, "method_comparison"))
    # Discriminant
    for col, label in [
        ("H_foodcom", "Food.com (recipe rating)"),
        ("H_foodpics", "Food-Pics (visual)"),
        ("H_crocufid", "CroCoFiD (desire-to-eat)"),
    ]:
        if col in merged.columns:
            pairs.append((col, label, "discriminant"))

    for col, label, vtype in pairs:
        valid = merged[["H_yelp", col]].dropna()
        n = len(valid)
        if n < 5:
            continue
        rho, p = stats.spearmanr(valid["H_yelp"], valid[col])
        r_p, p_p = stats.pearsonr(valid["H_yelp"], valid[col])
        # Bootstrap CI
        rhos = []
        for _ in range(2000):
            idx = np.random.choice(n, n, replace=True)
            rhos.append(stats.spearmanr(valid.iloc[idx, 0], valid.iloc[idx, 1])[0])
        ci_lo, ci_hi = np.percentile(rhos, [2.5, 97.5])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label:30s}: ρ={rho:.3f} {sig} [{ci_lo:.3f}, {ci_hi:.3f}] (N={n})")
        corr_rows.append({
            "Comparison": label,
            "Type": vtype,
            "N_dishes": n,
            "Spearman_rho": round(rho, 3),
            "Spearman_p": p,
            "CI95_lo": round(ci_lo, 3),
            "CI95_hi": round(ci_hi, 3),
            "Pearson_r": round(r_p, 3),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(TABLES / "cross_platform_bert_correlations.csv", index=False)
    print(f"\nSaved: {TABLES / 'cross_platform_bert_correlations.csv'}")

    # ── Figure 1: Convergent validity (BERT-BERT) ────────────────
    conv_cols = [c for c, _, t in pairs if t == "convergent" and c in merged.columns]
    if conv_cols:
        fig, axes = plt.subplots(1, len(conv_cols), figsize=(6 * len(conv_cols), 5))
        if len(conv_cols) == 1:
            axes = [axes]
        fig.suptitle("Convergent Validity: BERT H Scores Across Platforms",
                     fontsize=13, fontweight="bold")

        for ax, col in zip(axes, conv_cols):
            valid = merged[["H_yelp", col]].dropna()
            rho, p = stats.spearmanr(valid["H_yelp"], valid[col])
            ax.scatter(valid["H_yelp"], valid[col], alpha=0.5, s=25, c="#2077B4")
            # Fit line
            sl, ic = np.polyfit(valid["H_yelp"], valid[col], 1)
            xr = np.linspace(valid["H_yelp"].min(), valid["H_yelp"].max(), 50)
            ax.plot(xr, sl * xr + ic, "r--", alpha=0.7, lw=1.5)
            platform_label = col.replace("H_", "").replace("_bert", "").title()
            ax.set_xlabel("H (Yelp, finetuned BERT)", fontsize=11)
            ax.set_ylabel(f"H ({platform_label}, finetuned BERT)", fontsize=11)
            ax.set_title(f"ρ = {rho:.3f}, N = {len(valid)}", fontsize=12)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES / "cross_platform_convergent.png",
                    dpi=200, bbox_inches="tight")
        print(f"Saved: {FIGURES / 'cross_platform_convergent.png'}")
        plt.close()

    # ── Figure 2: Discriminant validity ──────────────────────────
    disc_cols = [c for c, _, t in pairs if t == "discriminant" and c in merged.columns]
    if disc_cols:
        fig, axes = plt.subplots(1, len(disc_cols), figsize=(5 * len(disc_cols), 4.5))
        if len(disc_cols) == 1:
            axes = [axes]
        fig.suptitle("Discriminant Validity: Different Hedonic Constructs",
                     fontsize=13, fontweight="bold")

        for ax, col in zip(axes, disc_cols):
            valid = merged[["H_yelp", col]].dropna()
            if len(valid) < 3:
                ax.set_visible(False)
                continue
            rho, p = stats.spearmanr(valid["H_yelp"], valid[col])
            sig = "" if p < 0.05 else " (n.s.)"
            ax.scatter(valid["H_yelp"], valid[col], alpha=0.5, s=25, c="#999999")
            sl, ic = np.polyfit(valid["H_yelp"], valid[col], 1)
            xr = np.linspace(valid["H_yelp"].min(), valid["H_yelp"].max(), 50)
            ax.plot(xr, sl * xr + ic, "--", color="#999", alpha=0.7, lw=1.5)
            label = col.replace("H_", "").replace("_", " ").title()
            ax.set_xlabel("H (Yelp, finetuned BERT)", fontsize=10)
            ax.set_ylabel(f"H ({label})", fontsize=10)
            ax.set_title(f"ρ = {rho:.3f}{sig}, N = {len(valid)}", fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES / "cross_platform_discriminant.png",
                    dpi=200, bbox_inches="tight")
        print(f"Saved: {FIGURES / 'cross_platform_discriminant.png'}")
        plt.close()

    # ── Figure 3: BERT vs Star comparison ────────────────────────
    method_cols = [(c, l) for c, l, t in pairs if t == "method_comparison" and c in merged.columns]
    bert_cols = [(c, l) for c, l, t in pairs if t == "convergent" and c in merged.columns]
    if method_cols and bert_cols:
        n_panels = len(bert_cols) + len(method_cols)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
        if n_panels == 1:
            axes = [axes]
        fig.suptitle("BERT NLP vs Raw Star Rating: Method Comparison",
                     fontsize=13, fontweight="bold")
        all_items = bert_cols + method_cols
        for ax, (col, label) in zip(axes, all_items):
            valid = merged[["H_yelp", col]].dropna()
            if len(valid) < 3:
                ax.set_visible(False)
                continue
            rho, _ = stats.spearmanr(valid["H_yelp"], valid[col])
            color = "#2077B4" if "BERT" in label else "#E87D2F"
            ax.scatter(valid["H_yelp"], valid[col], alpha=0.5, s=25, c=color)
            sl, ic = np.polyfit(valid["H_yelp"], valid[col], 1)
            xr = np.linspace(valid["H_yelp"].min(), valid["H_yelp"].max(), 50)
            ax.plot(xr, sl * xr + ic, "--", color=color, alpha=0.7, lw=1.5)
            ax.set_xlabel("H (Yelp BERT)", fontsize=10)
            ax.set_ylabel(f"H ({label.split('(')[0].strip()})", fontsize=10)
            ax.set_title(f"{label}\nρ = {rho:.3f}, N = {len(valid)}", fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES / "cross_platform_bert_vs_star.png",
                    dpi=200, bbox_inches="tight")
        print(f"Saved: {FIGURES / 'cross_platform_bert_vs_star.png'}")
        plt.close()

    # ── Summary stats ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA VOLUME SUMMARY")
    print("=" * 60)
    yelp_n = dei["H_mean"].notna().sum()  # approximate
    # Get actual Yelp mention count
    yelp_mentions_path = DATA / "dish_mentions_scored.parquet"
    if yelp_mentions_path.exists():
        yelp_n_mentions = len(pd.read_parquet(yelp_mentions_path, columns=["dish_id"]))
    else:
        yelp_n_mentions = 76927

    google_n = google_agg[f"n_google_bert"].sum() if google_agg is not None else 0
    ta_n = ta_agg[f"n_tripadvisor_bert"].sum() if ta_agg is not None else 0
    total = yelp_n_mentions + google_n + ta_n

    print(f"  Yelp (BERT):         {yelp_n_mentions:>10,} mentions")
    print(f"  Google Local (BERT): {int(google_n):>10,} mentions")
    print(f"  TripAdvisor (BERT):  {int(ta_n):>10,} mentions")
    print(f"  ─────────────────────────────────")
    print(f"  Total BERT-scored:   {int(total):>10,} mentions")
    print(f"  Multiplier vs Yelp-only: {total / yelp_n_mentions:.1f}×")

    return corr_df


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("16 — Cross-Platform BERT Scoring")
    print("    Google Local + TripAdvisor → finetuned BERT H")
    print("=" * 60)

    # Phase 1: Extract mentions
    google_df = extract_google_mentions()
    ta_df = extract_tripadvisor_mentions()

    # Phase 1.5: Sample to keep GPU runtime reasonable
    print("\n── Stratified sampling ──", flush=True)
    google_df = sample_mentions(google_df)
    ta_df = sample_mentions(ta_df)

    # Phase 2: BERT scoring (GPU)
    tokenizer, model, device = load_bert_model()

    google_df = bert_score_mentions(google_df, "Google", tokenizer, model, device)
    google_df.to_parquet(DATA / "google_mentions_bert_scored.parquet", index=False)

    ta_df = bert_score_mentions(ta_df, "TripAdvisor", tokenizer, model, device)
    ta_df.to_parquet(DATA / "tripadvisor_mentions_bert_scored.parquet", index=False)

    # Free GPU memory
    del model, tokenizer
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except: pass

    # Phase 3: Aggregate + correlate + plot
    print("\n" + "=" * 60)
    print("PHASE 3: Aggregation")
    print("=" * 60)

    google_agg = aggregate_dish_h(google_df, "google")
    print(f"  Google: {len(google_agg)} dishes with ≥5 BERT mentions")

    ta_agg = aggregate_dish_h(ta_df, "tripadvisor")
    print(f"  TripAdvisor: {len(ta_agg)} dishes with ≥5 BERT mentions")

    yelp_h = dei.set_index("dish_id") if "dish_id" in dei.columns else dei
    corr_df = compute_correlations_and_plot(yelp_h, google_agg, ta_agg)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
