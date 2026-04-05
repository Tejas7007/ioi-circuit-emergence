#!/usr/bin/env python3
"""Generate plots for dev interp results."""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# Load Part 1 results
with open('results/part1_component_emergence.json') as f:
    part1 = json.load(f)

# Load Part 2 results
with open('results/part2_pile_vs_synthetic.json') as f:
    part2 = json.load(f)

steps1 = [r['step'] for r in part1['results']]
ld1 = [r['performance']['mean_ld'] for r in part1['results']]
acc1 = [r['performance']['accuracy'] * 100 for r in part1['results']]
n_nm = [r['n_name_movers'] for r in part1['results']]
n_ioi = [r['n_total_ioi'] for r in part1['results']]
n_sp = [r['n_subject_promoters'] for r in part1['results']]

# ============================================================
# FIGURE 1: Component Emergence + Performance (Part 1)
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 5.5))

color_acc = '#2563eb'
color_nm = '#dc2626'
color_ioi = '#f97316'

ax1.set_xlabel('Training Step')
ax1.set_ylabel('IOI Accuracy (%)', color=color_acc)
ax1.plot(steps1, acc1, 'o-', color=color_acc, linewidth=2, markersize=5, label='IOI Accuracy', zorder=3)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylim(25, 105)
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_xscale('symlog', linthresh=100)

ax2 = ax1.twinx()
ax2.set_ylabel('Number of Heads', color=color_nm)
ax2.plot(steps1, n_nm, 's--', color=color_nm, linewidth=1.5, markersize=4, alpha=0.8, label='Name-Movers')
ax2.plot(steps1, n_ioi, '^--', color=color_ioi, linewidth=1.5, markersize=4, alpha=0.8, label='Total IOI Heads')
ax2.set_ylim(0, 80)
ax2.tick_params(axis='y', labelcolor=color_nm)

# Annotations
ax1.annotate('NMs emerge\nbut model\ngets worse', xy=(1000, 40.7), xytext=(200, 70),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

ax1.annotate('L8H9 takes over\nmodel flips positive', xy=(3000, 61.3), xytext=(8000, 45),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='gray'))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

plt.title('IOI Circuit Emergence in Pythia-160M\nComponents emerge at step 1000 but hurt performance until step 3000', fontsize=12)
plt.tight_layout()
plt.savefig('figures/part1_component_emergence.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/part1_component_emergence.pdf', bbox_inches='tight')
print('Saved figures/part1_component_emergence.png')
plt.close()

# ============================================================
# FIGURE 2: Pile vs Synthetic (Part 2)
# ============================================================
steps2 = [r['step'] for r in part2['results']]
syn_acc = [r['synthetic']['accuracy'] * 100 for r in part2['results']]
pile_acc = [r['pile']['accuracy'] * 100 for r in part2['results']]
syn_ld = [r['synthetic']['mean_ld'] for r in part2['results']]
pile_ld = [r['pile']['mean_ld'] for r in part2['results']]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 2]})

# Top: Accuracy
ax_top.plot(steps2, syn_acc, 'o-', color='#2563eb', linewidth=2, markersize=5, label='Synthetic (Wang et al.)')
ax_top.plot(steps2, pile_acc, 's-', color='#dc2626', linewidth=2, markersize=5, label='Pile (n=384)')
ax_top.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_top.axhline(y=70, color='gray', linestyle=':', alpha=0.4, linewidth=1)
ax_top.set_ylabel('IOI Accuracy (%)')
ax_top.set_ylim(25, 105)
ax_top.set_xscale('symlog', linthresh=100)
ax_top.legend(fontsize=10, loc='center right')
ax_top.set_title('Synthetic vs Natural IOI Performance Across Training\nPythia-160M evaluated on Wang et al. templates vs 500 validated Pile examples', fontsize=12)

# Shade the "Pile ahead" region
for i in range(len(steps2)-1):
    if pile_acc[i] > syn_acc[i]:
        ax_top.axvspan(steps2[i], steps2[i+1], alpha=0.1, color='red')

ax_top.annotate('Pile ahead\n(+15pp)', xy=(2000, 49.7), xytext=(100, 65),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray'),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='gray'))

ax_top.annotate('Synthetic\nreaches 70%', xy=(4000, 84), xytext=(10000, 50),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray'),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#dbeafe', edgecolor='gray'))

ax_top.annotate('Pile peaks\nat 76%', xy=(66000, 76), xytext=(100000, 55),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray'),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='gray'))

# Bottom: Logit Difference
ax_bot.plot(steps2, syn_ld, 'o-', color='#2563eb', linewidth=2, markersize=5, label='Synthetic LD')
ax_bot.plot(steps2, pile_ld, 's-', color='#dc2626', linewidth=2, markersize=5, label='Pile LD')
ax_bot.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_bot.set_ylabel('Logit Difference')
ax_bot.set_xlabel('Training Step')
ax_bot.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/part2_pile_vs_synthetic.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/part2_pile_vs_synthetic.pdf', bbox_inches='tight')
print('Saved figures/part2_pile_vs_synthetic.png')
plt.close()

# ============================================================
# FIGURE 3: Component counts over training
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(steps1, n_nm, 'o-', color='#2563eb', linewidth=2, markersize=5, label='Name-Movers')
ax.plot(steps1, n_ioi, 's-', color='#f97316', linewidth=2, markersize=5, label='Total IOI Heads')
ax.plot(steps1, n_sp, '^-', color='#dc2626', linewidth=2, markersize=5, label='Subject-Promoters')

ax.set_xlabel('Training Step')
ax.set_ylabel('Number of Heads')
ax.set_xscale('symlog', linthresh=100)
ax.legend(fontsize=10)
ax.set_title('Functional Component Counts Across Training\nPythia-160M, tau=0.02', fontsize=12)
ax.set_ylim(0, 80)

plt.tight_layout()
plt.savefig('figures/part3_component_counts.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/part3_component_counts.pdf', bbox_inches='tight')
print('Saved figures/part3_component_counts.png')
plt.close()

# ============================================================
# FIGURE 4: Gap between Pile and Synthetic
# ============================================================
gap = [p - s for p, s in zip(pile_acc, syn_acc)]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#dc2626' if g > 0 else '#2563eb' for g in gap]
ax.bar(range(len(steps2)), gap, color=colors, alpha=0.7)
ax.set_xticks(range(len(steps2)))
ax.set_xticklabels([str(s) for s in steps2], rotation=45, ha='right', fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('Pile Accuracy - Synthetic Accuracy (pp)')
ax.set_title('Accuracy Gap: Pile vs Synthetic Across Training\nRed = Pile ahead, Blue = Synthetic ahead', fontsize=12)

plt.tight_layout()
plt.savefig('figures/part4_accuracy_gap.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/part4_accuracy_gap.pdf', bbox_inches='tight')
print('Saved figures/part4_accuracy_gap.png')
plt.close()

print('\nAll figures saved to figures/')
