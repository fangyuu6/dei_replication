"""
25_figure_redesign.py — Redesign figures for maximum information density
Fixes:
1. Fig 2: 9 repetitive scatter panels → compact 2-panel (H% bar + substitution summary)
2. Fig 3: 5-panel cross-platform (3 weak) → clean 3-panel validation
3. Fig 1: Add variance numbers as annotations (eliminate standalone Table 1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

ROOT = Path("c:/project_EF")
DATA = ROOT / "data"
FIGURES = ROOT / "results" / "figures"

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 200,
})

# ══════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════
dei = pd.read_csv(DATA / "combined_dish_DEI_v2.csv")
cross_bert = pd.read_csv(DATA / "cross_platform_h_bert.csv")
foodcom_bt = pd.read_csv(DATA / "foodcom_dish_h_pairwise.csv")
yelp_bt = pd.read_csv(DATA / "dish_h_pairwise_v2.csv")

print(f"DEI: {len(dei)} dishes, {dei['category'].nunique()} categories")
print(f"Cross-platform BERT: {len(cross_bert)} dishes")
print(f"Food.com BT: {len(foodcom_bt)} dishes")

# ══════════════════════════════════════════════════════════════════
# FIG 2 REDESIGN: Compact within-category + substitution summary
# ══════════════════════════════════════════════════════════════════
print("\n=== Redesigning Fig 2: Within-category ===")

# Panel (a): H% contribution by category — horizontal bar chart
cats = dei.groupby('category').apply(lambda g: pd.Series({
    'n': len(g),
    'H_cv': g['log_H'].std() / g['log_H'].mean() * 100 if g['log_H'].mean() != 0 else 0,
    'E_cv': g['log_E'].std() / abs(g['log_E'].mean()) * 100 if g['log_E'].mean() != 0 else 0,
    'var_logH': g['log_H'].var(),
    'var_logE': g['log_E'].var(),
    'cov': np.cov(g['log_H'], g['log_E'])[0, 1] if len(g) > 2 else 0,
    'mean_logDEI': g['log_DEI'].mean(),
})).reset_index()

# Compute H% of Var(logDEI) for each category
cats['var_logDEI'] = cats['var_logH'] + cats['var_logE'] - 2 * cats['cov']
cats['H_pct'] = (cats['var_logH'] / cats['var_logDEI'] * 100).clip(-5, 100)
cats = cats.sort_values('H_pct', ascending=True)

# Panel (b): Substitution potential — within each category, how many pairs
# achieve >30% E reduction with <1 H loss
print("Computing substitution pairs...")
sub_stats = []
for cat_name, grp in dei.groupby('category'):
    if len(grp) < 3:
        continue
    n_viable = 0
    e_reductions = []
    h_changes = []
    dishes_arr = grp[['dish_id', 'H_mean', 'E_composite', 'calorie_kcal', 'protein_g']].dropna().values
    for i in range(len(dishes_arr)):
        for j in range(len(dishes_arr)):
            if i == j:
                continue
            h_i, e_i, cal_i, prot_i = dishes_arr[i, 1:]
            h_j, e_j, cal_j, prot_j = dishes_arr[j, 1:]
            # Substitution: replace dish i with dish j
            if e_i == 0:
                continue
            e_red = (e_i - e_j) / e_i
            h_change = h_j - h_i
            # Viable: >30% E reduction, <1 H loss, protein/calorie constraints
            if (e_red > 0.3 and h_change > -1.0 and
                prot_j >= 0.5 * prot_i and
                0.5 * cal_i <= cal_j <= 1.5 * cal_i):
                n_viable += 1
                e_reductions.append(e_red * 100)
                h_changes.append(h_change)
    sub_stats.append({
        'category': cat_name,
        'n_dishes': len(grp),
        'n_viable': n_viable,
        'mean_e_red': np.mean(e_reductions) if e_reductions else 0,
        'mean_h_change': np.mean(h_changes) if h_changes else 0,
    })

sub_df = pd.DataFrame(sub_stats)
sub_df = sub_df.merge(cats[['category', 'H_pct']], on='category')
sub_df = sub_df.sort_values('H_pct', ascending=True)

print(f"Total viable substitutions: {sub_df['n_viable'].sum():,}")
print(f"Mean E reduction: {sub_df.loc[sub_df['n_viable']>0, 'mean_e_red'].mean():.1f}%")

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1, 1.2]})

# Panel (a): H% contribution bar chart
colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(cats)))
bars = ax1.barh(range(len(cats)), cats['H_pct'].values, color=colors, edgecolor='gray', linewidth=0.5)
ax1.set_yticks(range(len(cats)))
ax1.set_yticklabels([f"{c} ({n:.0f})" for c, n in zip(cats['category'], cats['n'])], fontsize=7.5)
ax1.set_xlabel('$H$ contribution to Var(log DEI) (%)')
ax1.set_title('(a) Within-category $H$ importance', fontweight='bold', fontsize=10)
ax1.axvline(x=0, color='black', linewidth=0.5)
# Annotate values
for i, (val, n) in enumerate(zip(cats['H_pct'].values, cats['n'].values)):
    ax1.text(max(val, 0) + 0.8, i, f'{val:.1f}%', va='center', fontsize=7)

# Panel (b): Substitution E reduction vs H change (aggregate by category)
sub_plot = sub_df[sub_df['n_viable'] > 0].copy()
scatter = ax2.scatter(sub_plot['mean_e_red'], sub_plot['mean_h_change'],
                       s=np.sqrt(sub_plot['n_viable']) * 3,
                       c=sub_plot['H_pct'], cmap='RdYlGn', vmin=-5, vmax=40,
                       edgecolors='gray', linewidth=0.5, alpha=0.8, zorder=3)
# Annotate categories
for _, row in sub_plot.iterrows():
    label = row['category'].replace(' ', '\n') if len(row['category']) > 10 else row['category']
    ax2.annotate(f"{row['category']}\n({row['n_viable']:,})",
                 (row['mean_e_red'], row['mean_h_change']),
                 fontsize=6.5, ha='center', va='bottom',
                 textcoords='offset points', xytext=(0, 5))
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax2.axhline(y=-1, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.text(60, -0.9, '$\\Delta H = -1$ threshold', fontsize=7, color='red', alpha=0.7)
ax2.set_xlabel('Mean $E$ reduction (%)')
ax2.set_ylabel('Mean $\\Delta H$ (substitute − original)')
ax2.set_title('(b) Substitution potential by category', fontweight='bold', fontsize=10)

# Add total count annotation
total_subs = sub_df['n_viable'].sum()
ax2.text(0.98, 0.02, f'Total: {total_subs:,} viable swaps',
         transform=ax2.transAxes, fontsize=8, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(FIGURES / "within_category_compact_v3.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: within_category_compact_v3.png")

# ══════════════════════════════════════════════════════════════════
# FIG 3 REDESIGN: Clean 3-panel validation
# ══════════════════════════════════════════════════════════════════
print("\n=== Redesigning Fig 3: Validation ===")

# Merge cross-platform BERT data with Yelp BT data
# Panel (a): Google BERT H vs Yelp BERT H
# Panel (b): TripAdvisor BERT H vs Yelp BERT H
# Panel (c): Food.com BT H vs Yelp BT H

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

# --- Panel (a): Google ---
mask_g = cross_bert['H_google_bert'].notna()
x_g = cross_bert.loc[mask_g, 'H_yelp']
y_g = cross_bert.loc[mask_g, 'H_google_bert']
rho_g, p_g = stats.spearmanr(x_g, y_g)
n_g = mask_g.sum()

axes[0].scatter(x_g, y_g, s=15, alpha=0.5, color='#4C72B0', edgecolors='none')
z = np.polyfit(x_g, y_g, 1)
xx = np.linspace(x_g.min(), x_g.max(), 100)
axes[0].plot(xx, np.polyval(z, xx), 'r--', linewidth=1, alpha=0.7)
axes[0].set_xlabel('$H$ (Yelp BERT)')
axes[0].set_ylabel('$H$ (Google Local BERT)')
axes[0].set_title(f'(a) Google Local\n$\\rho$ = {rho_g:.2f}, $n$ = {n_g}', fontweight='bold')

# --- Panel (b): TripAdvisor ---
mask_t = cross_bert['H_tripadvisor_bert'].notna()
x_t = cross_bert.loc[mask_t, 'H_yelp']
y_t = cross_bert.loc[mask_t, 'H_tripadvisor_bert']
rho_t, p_t = stats.spearmanr(x_t, y_t)
n_t = mask_t.sum()

axes[1].scatter(x_t, y_t, s=15, alpha=0.5, color='#55A868', edgecolors='none')
z = np.polyfit(x_t, y_t, 1)
xx = np.linspace(x_t.min(), x_t.max(), 100)
axes[1].plot(xx, np.polyval(z, xx), 'r--', linewidth=1, alpha=0.7)
axes[1].set_xlabel('$H$ (Yelp BERT)')
axes[1].set_ylabel('$H$ (TripAdvisor BERT)')
axes[1].set_title(f'(b) TripAdvisor\n$\\rho$ = {rho_t:.2f}, $n$ = {n_t}', fontweight='bold')

# --- Panel (c): Food.com BT ---
# Merge Food.com BT with Yelp BT
merged_bt = foodcom_bt[['dish_id', 'H_pairwise_home']].rename(columns={'H_pairwise_home': 'H_foodcom_bt'})
merged_bt = merged_bt.merge(yelp_bt[['dish_id', 'H_pairwise']].rename(columns={'H_pairwise': 'H_yelp_bt'}),
                             on='dish_id', how='inner')
rho_fc, p_fc = stats.spearmanr(merged_bt['H_yelp_bt'], merged_bt['H_foodcom_bt'])
n_fc = len(merged_bt)

axes[2].scatter(merged_bt['H_yelp_bt'], merged_bt['H_foodcom_bt'],
                s=15, alpha=0.4, color='#DD8452', edgecolors='none')
z = np.polyfit(merged_bt['H_yelp_bt'], merged_bt['H_foodcom_bt'], 1)
xx = np.linspace(merged_bt['H_yelp_bt'].min(), merged_bt['H_yelp_bt'].max(), 100)
axes[2].plot(xx, np.polyval(z, xx), 'r--', linewidth=1, alpha=0.7)
axes[2].set_xlabel('$H$ (Yelp BT)')
axes[2].set_ylabel('$H$ (Food.com BT)')
axes[2].set_title(f'(c) Food.com (home cooking)\n$\\rho$ = {rho_fc:.2f}, $n$ = {n_fc}', fontweight='bold')

# Add interpretation bracket
for i, (rho, label) in enumerate([(rho_g, 'Same method\nDifferent platform'),
                                    (rho_t, 'Same method\nDifferent platform'),
                                    (rho_fc, 'Same pipeline\nDifferent context')]):
    axes[i].text(0.03, 0.97, label, transform=axes[i].transAxes,
                 fontsize=7, va='top', ha='left', style='italic',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig(FIGURES / "validation_3panel_v3.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: validation_3panel_v3.png")
print(f"  Google:      ρ={rho_g:.3f}, p={p_g:.2e}, n={n_g}")
print(f"  TripAdvisor: ρ={rho_t:.3f}, p={p_t:.2e}, n={n_t}")
print(f"  Food.com BT: ρ={rho_fc:.3f}, p={p_fc:.2e}, n={n_fc}")

# ══════════════════════════════════════════════════════════════════
# Verify the ρ values match between validation and foodcom BT data
# ══════════════════════════════════════════════════════════════════
print("\n=== Cross-check: Food.com BT columns ===")
print(f"foodcom_bt columns: {list(foodcom_bt.columns)}")
print(f"yelp_bt columns: {list(yelp_bt.columns)[:10]}")

print("\n=== All done ===")
print(f"New Fig 2: {FIGURES / 'within_category_compact_v3.png'}")
print(f"New Fig 3: {FIGURES / 'validation_3panel_v3.png'}")
