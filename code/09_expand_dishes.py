"""
09_expand_dishes.py — Expand dish coverage beyond 158
======================================================
Scans Yelp reviews for additional dishes from underrepresented cuisines.
Only adds dishes with sufficient review mentions (≥10).

Strategy:
  1. Define candidate dishes with aliases (200+ candidates)
  2. Scan Yelp restaurant reviews for mentions
  3. Report coverage stats and select viable dishes
  4. Generate new dish entries with recipes & environmental data

Output:
  - data/expanded_dish_candidates.csv (all candidates with mention counts)
  - data/expanded_dish_mentions.parquet (review-level mentions for new dishes)
"""

import sys, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import DATA_DIR, YELP_REVIEW, RESTAURANTS_PARQUET

# ══════════════════════════════════════════════════════════════════
# CANDIDATE DISHES — organized by cuisine region
# Each entry: dish_id → list of search aliases
# ══════════════════════════════════════════════════════════════════

CANDIDATE_DISHES = {
    # ── AFRICAN ──
    "jollof_rice":       ["jollof rice", "jollof"],
    "injera":            ["injera"],
    "doro_wot":          ["doro wot", "doro wat", "doro wet"],
    "kitfo":             ["kitfo", "kifto"],
    "tagine":            ["tagine", "tajine"],
    "couscous":          ["couscous"],
    "harissa":           ["harissa"],
    "shakshuka":         ["shakshuka", "shakshouka"],
    "fufu":              ["fufu", "foo foo"],
    "suya":              ["suya"],
    "mafe":              ["mafe", "maafe", "mafé"],
    "egusi_soup":        ["egusi soup", "egusi"],
    "bunny_chow":        ["bunny chow"],
    "bobotie":           ["bobotie"],
    "plantain":          ["fried plantain", "plantains", "sweet plantain"],
    "piri_piri":         ["piri piri", "peri peri", "piri-piri", "peri-peri"],
    "thieboudienne":     ["thieboudienne", "thieb", "ceebu jen"],
    "biltong":           ["biltong"],
    "chakalaka":         ["chakalaka"],
    "ful_medames":       ["ful medames", "ful", "foul medames", "foul mudammas"],

    # ── CARIBBEAN ──
    "jerk_chicken":      ["jerk chicken", "jerk pork", "jerk"],
    "rice_and_peas":     ["rice and peas", "rice & peas"],
    "mofongo":           ["mofongo"],
    "arroz_con_pollo":   ["arroz con pollo"],
    "oxtail_stew":       ["oxtail stew", "oxtail", "braised oxtail"],
    "roti_caribbean":    ["roti wrap", "curry roti", "doubles"],
    "callaloo":          ["callaloo"],
    "conch_fritters":    ["conch fritters", "conch"],
    "escovitch_fish":    ["escovitch", "escoveitch"],
    "ackee_saltfish":    ["ackee and saltfish", "ackee & saltfish", "ackee"],
    "cubano_sandwich":   ["cubano", "cuban sandwich", "cubano sandwich"],
    "pernil":            ["pernil"],
    "tostones":          ["tostones"],
    "coquito":           ["coquito"],
    "pastelitos":        ["pastelitos", "pastelito"],
    "pupusa":            ["pupusa", "pupusas"],

    # ── SOUTH AMERICAN ──
    "arepa":             ["arepa", "arepas"],
    "empanada_argentina":["empanada argentina", "empanadas argentinas"],
    "lomo_saltado":      ["lomo saltado"],
    "anticucho":         ["anticucho", "anticuchos"],
    "causa":             ["causa", "causa limeña"],
    "churrasco":         ["churrasco"],
    "picanha":           ["picanha"],
    "feijoada":          ["feijoada"],
    "pao_de_queijo":     ["pao de queijo", "pão de queijo", "cheese bread"],
    "coxinha":           ["coxinha"],
    "chimichurri":       ["chimichurri"],
    "asado":             ["asado"],
    "bandeja_paisa":     ["bandeja paisa"],
    "sancocho":          ["sancocho"],
    "aji_de_gallina":    ["aji de gallina"],
    "pastel_brasileiro": ["pastel de carne", "pastel de queijo"],

    # ── CENTRAL ASIAN / PERSIAN ──
    "plov":              ["plov", "pilaf", "pilau", "pulao"],
    "lagman":            ["lagman"],
    "manty":             ["manty", "manti", "mandu"],
    "shashlik":          ["shashlik", "shashlyk"],
    "ghormeh_sabzi":     ["ghormeh sabzi"],
    "tahdig":            ["tahdig"],
    "fesenjan":          ["fesenjan", "fesenjān", "fesenjon"],
    "zereshk_polo":      ["zereshk polo"],
    "koobideh":          ["koobideh", "kubideh", "koobide"],
    "ash_reshteh":       ["ash reshteh"],
    "kashk_bademjan":    ["kashk bademjan", "kashk-e bademjan"],
    "joojeh_kabab":      ["joojeh kabab", "jujeh kabab"],

    # ── TURKISH ──
    "doner_kebab":       ["doner kebab", "doner", "döner"],
    "lahmacun":          ["lahmacun"],
    "borek":             ["borek", "börek", "burek"],
    "iskender_kebab":    ["iskender", "iskender kebab"],
    "pide":              ["pide", "turkish pide"],
    "kofte":             ["kofte", "köfte", "kofta"],
    "manti_turkish":     ["turkish manti", "manti"],
    "gozleme":           ["gozleme", "gözleme"],
    "simit":             ["simit"],
    "kisir":             ["kisir", "kısır"],
    "kunefe":            ["kunefe", "künefe", "kunafa", "kanafeh"],

    # ── GREEK ──
    "souvlaki":          ["souvlaki"],
    "saganaki":          ["saganaki"],
    "pastitsio":         ["pastitsio"],
    "tzatziki":          ["tzatziki"],
    "horiatiki":         ["horiatiki", "greek salad"],
    "loukoumades":       ["loukoumades", "loukoumada"],
    "galaktoboureko":    ["galaktoboureko"],
    "tiropita":          ["tiropita"],
    "keftedes":          ["keftedes", "keftedakia"],

    # ── LEBANESE / LEVANTINE ──
    "kibbeh":            ["kibbeh", "kibbe", "kebbeh"],
    "fattoush":          ["fattoush"],
    "manakish":          ["manakish", "manakeesh", "manoushe"],
    "muhammara":         ["muhammara"],
    "mujaddara":         ["mujaddara", "mujadara", "mudardara"],
    "warak_enab":        ["warak enab", "stuffed grape leaves", "dolmades"],
    "labneh":            ["labneh", "labne", "labaneh"],
    "sfeeha":            ["sfeeha", "sfiha"],
    "halloumi":          ["halloumi", "haloumi"],

    # ── SOUTHEAST ASIAN (BEYOND THAI/VIET) ──
    "nasi_goreng":       ["nasi goreng"],
    "rendang":           ["rendang"],
    "satay_indonesian":  ["satay", "sate"],
    "lumpia":            ["lumpia"],
    "adobo":             ["adobo", "chicken adobo", "pork adobo"],
    "sinigang":          ["sinigang"],
    "sisig":             ["sisig"],
    "kare_kare":         ["kare kare", "kare-kare"],
    "halo_halo":         ["halo halo", "halo-halo"],
    "nasi_lemak":        ["nasi lemak"],
    "char_kway_teow":    ["char kway teow", "char kuey teow"],
    "laksa":             ["laksa"],
    "rojak":             ["rojak", "rujak"],
    "bak_kut_teh":       ["bak kut teh"],
    "mee_goreng":        ["mee goreng", "mi goreng"],
    "nasi_padang":       ["nasi padang"],
    "gado_gado":         ["gado gado", "gado-gado"],
    "banh_xeo":          ["banh xeo", "bánh xèo"],
    "bun_rieu":          ["bun rieu"],
    "pho_cuon":          ["pho cuon"],
    "hu_tieu":           ["hu tieu", "hủ tiếu"],

    # ── PORTUGUESE / SPANISH EXPANSION ──
    "bacalhau":          ["bacalhau", "bacalao"],
    "pasteis_de_nata":   ["pastel de nata", "pasteis de nata", "portuguese tart"],
    "francesinha":       ["francesinha"],
    "caldo_verde":       ["caldo verde"],
    "sardines_grilled":  ["grilled sardines", "sardine"],
    "croquetas":         ["croquetas", "croquettes"],
    "gambas_al_ajillo":  ["gambas al ajillo", "garlic shrimp"],
    "tortilla_espanola": ["tortilla espanola", "spanish omelet", "spanish omelette", "spanish tortilla"],

    # ── MISSING COMMON DISHES (existing cuisines) ──
    "hot_dog":           ["hot dog", "hotdog"],
    "meatloaf":          ["meatloaf", "meat loaf"],
    "chili":             ["chili con carne", "beef chili"],
    "pot_roast":         ["pot roast"],
    "cobb_salad":        ["cobb salad"],
    "eggs_benedict":     ["eggs benedict", "egg benedict"],
    "fish_tacos":        ["fish taco", "fish tacos"],
    "carnitas":          ["carnitas"],
    "churros_mexican":   ["churros"],
    "mole":              ["mole", "mole poblano", "mole negro"],
    "chilaquiles":       ["chilaquiles"],
    "al_pastor":         ["al pastor", "tacos al pastor"],
    "sopes":             ["sopes"],
    "birria":            ["birria", "birria tacos"],
    "arancini":          ["arancini"],
    "polenta":           ["polenta"],
    "eggplant_parm":     ["eggplant parmesan", "eggplant parm", "melanzane"],
    "focaccia":          ["focaccia"],
    "prosciutto":        ["prosciutto"],
    "cannoli":           ["cannoli", "cannolo"],
    "gyudon":            ["gyudon", "beef bowl"],
    "tonkotsu_ramen":    ["tonkotsu ramen", "tonkotsu"],
    "karaage":           ["karaage", "kara-age", "japanese fried chicken"],
    "katsu_curry":       ["katsu curry"],
    "tamagoyaki":        ["tamagoyaki"],
    "okra_soup":         ["okra soup"],
    "jjajangmyeon":      ["jjajangmyeon", "jajangmyeon", "black bean noodles"],
    "dakgalbi":          ["dakgalbi", "dak galbi"],
    "bossam":            ["bossam"],
    "pajeon":            ["pajeon", "pa jun", "korean pancake"],
    "budae_jjigae":      ["budae jjigae", "army stew"],
    "jokbal":            ["jokbal"],
    "som_tam":           ["som tam", "som tum"],
    "khao_man_gai":      ["khao man gai", "chicken rice"],
    "boat_noodles":      ["boat noodles"],
    "massaman":          ["massaman"],
    "gaeng_keow_wan":    ["gaeng keow wan"],
    "idli":              ["idli", "idly"],
    "uttapam":           ["uttapam", "uttappam"],
    "pav_bhaji":         ["pav bhaji"],
    "chole_bhature":     ["chole bhature", "chole bhatura"],
    "aloo_gobi":         ["aloo gobi", "aloo gobhi"],
    "paneer_butter":     ["paneer butter masala", "paneer makhani"],
    "malai_kofta":       ["malai kofta"],
    "gulab_jamun":       ["gulab jamun"],
    "jalebi":            ["jalebi"],
    "poutine":           ["poutine"],
    "pierogi":           ["pierogi", "pierogy", "perogies"],
    "borscht":           ["borscht", "borsch"],
    "blini":             ["blini", "blinis"],
    "stroganoff":        ["stroganoff", "beef stroganoff"],
    "schnitzel":         ["schnitzel", "wiener schnitzel"],
    "bratwurst":         ["bratwurst", "brat"],
    "pretzel":           ["pretzel", "soft pretzel"],
    "spaetzle":          ["spaetzle", "spätzle"],
    "currywurst":        ["currywurst"],
    "dim_sum_items":     ["har gow", "siu mai", "char siu bao", "cheung fun"],
    "dan_tat":           ["egg tart", "dan tat", "dan ta"],
    "char_siu_bao":      ["char siu bao", "bbq pork bun", "pork bun"],
    "wonton_noodle":     ["wonton noodle", "wonton noodles"],
    "clay_pot_rice":     ["clay pot rice"],
    "roast_goose":       ["roast goose"],
    "salt_pepper_squid": ["salt and pepper squid", "salt pepper squid", "salt & pepper squid"],
    "macarons":          ["macaron", "macarons"],
    "eclair":            ["eclair", "éclair"],
    "profiterole":       ["profiterole", "profiteroles"],
    "beef_tartare":      ["beef tartare", "steak tartare"],
    "duck_confit":       ["duck confit", "confit de canard"],
    "bouillabaisse":     ["bouillabaisse"],
    "nicoise_salad":     ["nicoise salad", "niçoise"],
    "galette":           ["galette"],
    "fondue":            ["fondue", "cheese fondue"],
    "raclette":          ["raclette"],
}

print(f"Candidate dishes: {len(CANDIDATE_DISHES)}")

# ══════════════════════════════════════════════════════════════════
# Build fast lookup (same approach as 07c)
# ══════════════════════════════════════════════════════════════════

# Flatten aliases
ALIAS_MAP = {}
for dish_id, aliases in CANDIDATE_DISHES.items():
    for alias in aliases:
        ALIAS_MAP[alias.lower()] = dish_id

# Sort longest-first for scanning
_ALIAS_SCAN = sorted(
    [(a, ALIAS_MAP[a]) for a in ALIAS_MAP if len(a) >= 3],
    key=lambda x: -len(x[0])
)

def search_dish_in_text(text):
    """Search for dish mentions using pure string matching."""
    text_lower = text.lower()
    found = set()
    for alias, dish_id in _ALIAS_SCAN:
        if dish_id in found:
            continue
        idx = text_lower.find(alias)
        while idx != -1:
            before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
            end = idx + len(alias)
            after_ok = (end >= len(text_lower) or not text_lower[end].isalnum())
            if before_ok and after_ok:
                found.add(dish_id)
                break
            idx = text_lower.find(alias, idx + 1)
    return found

# ══════════════════════════════════════════════════════════════════
# Scan Yelp reviews
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Scanning Yelp reviews for new dish mentions...")
print("=" * 60)

# Load restaurant business_ids (to filter to restaurant reviews only)
restaurants = pd.read_parquet(RESTAURANTS_PARQUET, columns=["business_id"])
restaurant_ids = set(restaurants["business_id"])

dish_mentions = defaultdict(list)  # dish_id → list of (review_id, business_id, stars, text_snippet)
dish_counts = defaultdict(int)
n_scanned = 0

with open(YELP_REVIEW, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            review = json.loads(line)
        except:
            continue

        biz_id = review.get("business_id", "")
        if biz_id not in restaurant_ids:
            continue

        n_scanned += 1
        text = review.get("text", "")
        if not text or len(text) < 20:
            continue

        found = search_dish_in_text(text)
        for dish_id in found:
            dish_counts[dish_id] += 1
            # Store first 500 mentions per dish for later scoring
            if len(dish_mentions[dish_id]) < 500:
                # Find the sentence containing the dish mention
                dish_mentions[dish_id].append({
                    "review_id": review.get("review_id", ""),
                    "business_id": biz_id,
                    "stars": review.get("stars", 0),
                    "text": text[:500],  # truncate for memory
                })

        if n_scanned % 500000 == 0:
            n_found = sum(1 for v in dish_counts.values() if v >= 10)
            print(f"  {n_scanned:,} reviews scanned, {len(dish_counts)} dishes found, {n_found} with ≥10 mentions")

print(f"\n  Total: {n_scanned:,} restaurant reviews scanned")
print(f"  Candidate dishes found: {len(dish_counts)}")

# ══════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Results: Dish mention counts")
print("=" * 60)

# Assign cuisine labels
DISH_CUISINES = {}
cuisine_groups = {
    "African": ["jollof_rice", "injera", "doro_wot", "kitfo", "tagine", "couscous",
                "harissa", "shakshuka", "fufu", "suya", "mafe", "egusi_soup",
                "bunny_chow", "bobotie", "plantain", "piri_piri", "thieboudienne",
                "biltong", "chakalaka", "ful_medames"],
    "Caribbean": ["jerk_chicken", "rice_and_peas", "mofongo", "arroz_con_pollo",
                  "oxtail_stew", "roti_caribbean", "callaloo", "conch_fritters",
                  "escovitch_fish", "ackee_saltfish", "cubano_sandwich", "pernil",
                  "tostones", "coquito", "pastelitos", "pupusa"],
    "South American": ["arepa", "empanada_argentina", "lomo_saltado", "anticucho",
                       "causa", "churrasco", "picanha", "feijoada", "pao_de_queijo",
                       "coxinha", "chimichurri", "asado", "bandeja_paisa", "sancocho",
                       "aji_de_gallina", "pastel_brasileiro"],
    "Central Asian/Persian": ["plov", "lagman", "manty", "shashlik", "ghormeh_sabzi",
                              "tahdig", "fesenjan", "zereshk_polo", "koobideh",
                              "ash_reshteh", "kashk_bademjan", "joojeh_kabab"],
    "Turkish": ["doner_kebab", "lahmacun", "borek", "iskender_kebab", "pide",
                "kofte", "manti_turkish", "gozleme", "simit", "kisir", "kunefe"],
    "Greek": ["souvlaki", "saganaki", "pastitsio", "tzatziki", "horiatiki",
              "loukoumades", "galaktoboureko", "tiropita", "keftedes"],
    "Lebanese": ["kibbeh", "fattoush", "manakish", "muhammara", "mujaddara",
                 "warak_enab", "labneh", "sfeeha", "halloumi"],
    "Filipino": ["lumpia", "adobo", "sinigang", "sisig", "kare_kare", "halo_halo"],
    "Indonesian/Malay": ["nasi_goreng", "rendang", "satay_indonesian", "nasi_lemak",
                         "char_kway_teow", "laksa", "rojak", "bak_kut_teh",
                         "mee_goreng", "nasi_padang", "gado_gado"],
    "Vietnamese+": ["banh_xeo", "bun_rieu", "pho_cuon", "hu_tieu"],
    "Portuguese": ["bacalhau", "pasteis_de_nata", "francesinha", "caldo_verde", "sardines_grilled"],
    "Spanish+": ["croquetas", "gambas_al_ajillo", "tortilla_espanola"],
    "Mexican+": ["fish_tacos", "carnitas", "churros_mexican", "mole", "chilaquiles",
                 "al_pastor", "sopes", "birria"],
    "American+": ["hot_dog", "meatloaf", "chili", "pot_roast", "cobb_salad",
                  "eggs_benedict", "poutine"],
    "Japanese+": ["gyudon", "tonkotsu_ramen", "karaage", "katsu_curry", "tamagoyaki"],
    "Korean+": ["jjajangmyeon", "dakgalbi", "bossam", "pajeon", "budae_jjigae", "jokbal"],
    "Thai+": ["som_tam", "khao_man_gai", "boat_noodles"],
    "Indian+": ["idli", "uttapam", "pav_bhaji", "chole_bhature", "aloo_gobi",
                "paneer_butter", "malai_kofta", "gulab_jamun", "jalebi"],
    "Chinese+": ["dim_sum_items", "dan_tat", "char_siu_bao", "wonton_noodle",
                 "clay_pot_rice", "roast_goose", "salt_pepper_squid"],
    "French+": ["macarons", "eclair", "profiterole", "beef_tartare", "duck_confit",
                "bouillabaisse", "nicoise_salad", "galette"],
    "European": ["pierogi", "borscht", "blini", "stroganoff", "schnitzel",
                 "bratwurst", "pretzel", "spaetzle", "currywurst", "fondue", "raclette"],
}

for cuisine, dishes in cuisine_groups.items():
    for d in dishes:
        DISH_CUISINES[d] = cuisine

# Build results table
results = []
for dish_id, count in sorted(dish_counts.items(), key=lambda x: -x[1]):
    results.append({
        "dish_id": dish_id,
        "mention_count": count,
        "cuisine_group": DISH_CUISINES.get(dish_id, "Other"),
        "viable": count >= 10,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_DIR / "expanded_dish_candidates.csv", index=False)

# Summary
viable = results_df[results_df["viable"]]
print(f"\n  Viable new dishes (≥10 mentions): {len(viable)}")
print(f"  Total new mentions: {viable['mention_count'].sum():,}")

print(f"\n  By cuisine group:")
for cuisine in viable.groupby("cuisine_group")["dish_id"].count().sort_values(ascending=False).index:
    group = viable[viable["cuisine_group"] == cuisine]
    dishes = group["dish_id"].tolist()
    total = group["mention_count"].sum()
    print(f"    {cuisine:25s}: {len(dishes):2d} dishes, {total:>6,} mentions")
    for _, row in group.sort_values("mention_count", ascending=False).head(5).iterrows():
        print(f"      {row['dish_id']:25s}: {row['mention_count']:>5,}")

# Non-viable
non_viable = results_df[~results_df["viable"]]
print(f"\n  Non-viable (<10 mentions): {len(non_viable)}")
if len(non_viable) > 0:
    for _, row in non_viable.sort_values("mention_count", ascending=False).head(20).iterrows():
        print(f"    {row['dish_id']:25s}: {row['mention_count']:>3} mentions ({row['cuisine_group']})")

# Not found at all
not_found = set(CANDIDATE_DISHES.keys()) - set(dish_counts.keys())
print(f"\n  Not found in Yelp at all: {len(not_found)}")
if not_found:
    for d in sorted(not_found):
        print(f"    {d} ({DISH_CUISINES.get(d, 'Other')})")

# ══════════════════════════════════════════════════════════════════
# Save mention data for viable dishes
# ══════════════════════════════════════════════════════════════════
viable_ids = set(viable["dish_id"])
mention_rows = []
for dish_id in viable_ids:
    for m in dish_mentions[dish_id]:
        mention_rows.append({
            "dish_id": dish_id,
            "review_id": m["review_id"],
            "business_id": m["business_id"],
            "stars": m["stars"],
            "context_text": m["text"],
        })

if mention_rows:
    mention_df = pd.DataFrame(mention_rows)
    mention_df.to_parquet(DATA_DIR / "expanded_dish_mentions.parquet", index=False)
    print(f"\n  Saved: {DATA_DIR / 'expanded_dish_mentions.parquet'} ({len(mention_df):,} mentions)")

print(f"\n  Saved: {DATA_DIR / 'expanded_dish_candidates.csv'}")

print(f"\n{'='*60}")
print(f"SUMMARY: {len(viable)} new dishes viable for expansion")
print(f"Total coverage: 158 existing + {len(viable)} new = {158 + len(viable)} dishes")
print(f"{'='*60}")
print("Done!")
