"""
Generate Figures 11-12: L8H9 attention distribution + multi-hop pipeline.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

with open('results/mega_experiments.json') as f:
    mega = json.load(f)

# ============================================================
# FIGURE 11: Attention distribution (not just mean)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

data = mega['distribution_and_multihop']['attention_distribution']
raw = data['raw_values']  # first 100 examples

# Left: histogram of per-example S2 attention
ax1.hist(raw, bins=25, color='#e41a1c', alpha=0.75, edgecolor='black', linewidth=0.5)
ax1.axvline(data['l8h9_s2_mean'], color='darkred', linestyle='--', linewidth=2, label='Mean = %.2f' % data['l8h9_s2_mean'])
ax1.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='50% threshold')
ax1.set_xlabel('Attention to S2 (from final token)')
ax1.set_ylabel('Number of examples')
ax1.set_title('Per-example distribution (n=%d)' % len(raw))
ax1.legend(fontsize=10)
ax1.set_xlim(0, 1.0)
ax1.grid(True, alpha=0.2, axis='y')

# Right: percentile breakdown
percentiles = ['p10', 'p25', 'p50', 'p75', 'p90']
values = [data['l8h9_s2_%s' % p] for p in percentiles]
ax2.bar(percentiles, values, color='#e41a1c', alpha=0.75, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Attention to S2')
ax2.set_title('Percentiles — tight, high distribution')
ax2.set_ylim(0, 1.0)
ax2.axhline(0.8, color='gray', linestyle=':', alpha=0.7, linewidth=1)
ax2.grid(True, alpha=0.2, axis='y')
for i, v in enumerate(values):
    ax2.text(i, v + 0.02, '%.2f' % v, ha='center', fontsize=10, fontweight='bold')

# Add stats annotation
stat_text = ("Min: %.2f\nMax: %.2f\n%% >0.8: %.0f%%\n%% <0.5: %.0f%%" %
             (data['l8h9_s2_min'], data['l8h9_s2_max'],
              data['pct_above_80']*100, data['pct_below_50']*100))
ax2.text(0.98, 0.02, stat_text, transform=ax2.transAxes, fontsize=9,
         ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Figure 11: L8H9 Attention to S2 — Robust Across Individual Prompts (not bimodal)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig11_l8h9_distribution.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig11_l8h9_distribution.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 11")

# ============================================================
# FIGURE 12: Multi-hop pipeline — heads that write TO S2
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

multihop = mega['distribution_and_multihop']['multihop_duptoken_candidates']

# Sort
multihop_sorted = sorted(multihop, key=lambda x: -x['s2_to_s1_attn'])[:12]

heads = [h['head'] for h in multihop_sorted]
scores = [h['s2_to_s1_attn'] for h in multihop_sorted]
# Color by strength
colors = []
for s in scores:
    if s > 0.7: colors.append('#b30000')
    elif s > 0.4: colors.append('#e34a33')
    elif s > 0.2: colors.append('#fc8d59')
    else: colors.append('#fdcc8a')

bars = ax.barh(range(len(heads)), scores, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(heads)))
ax.set_yticklabels(heads)
ax.set_xlabel('Attention from S2 position to S1 position')
ax.set_title('Figure 12: Duplicate-Token Heads in Pythia-160M — Multi-Hop Pipeline Feeds L8H9',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.grid(True, alpha=0.2, axis='x')
ax.set_xlim(0, 1.0)

for i, s in enumerate(scores):
    ax.text(s + 0.01, i, '%.3f' % s, va='center', fontsize=9)

# Annotation
ax.annotate('L1H8 attends 91% from S2 → S1\nWrites "this is a repeat" at S2\nL8H9 later reads this processed info',
            xy=(0.91, 0), xytext=(0.42, 3.5),
            fontsize=9, ha='left',
            bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

plt.tight_layout()
plt.savefig('figures/fig12_multihop_pipeline.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig12_multihop_pipeline.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 12")
