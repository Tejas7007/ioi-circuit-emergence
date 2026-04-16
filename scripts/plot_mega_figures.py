"""
Generate NEW figures from mega experiments for the paper.
Figures 7-10: L8H9 attention dev, output projection, dominant head cross-scale, top-1 predictions.
Also regenerates figures 1-6 from the old plot script.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

C160 = '#1f77b4'
C410 = '#ff7f0e'
C1B = '#2ca02c'

# Load mega results
with open('results/mega_experiments.json') as f:
    mega = json.load(f)


# ============================================================
# FIGURE 7: L8H9 Attention Development
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

steps = ['step_1000', 'step_3000', 'step_143000']
step_labels = ['Step 1000\n(dip, 41% acc)', 'Step 3000\n(recovery, 61% acc)', 'Step 143000\n(final, 100% acc)']

attn_io = []
attn_s1 = []
attn_s2 = []
attn_prev = []

for s in steps:
    r = mega['exp_e_attention'][s]['L8H9']
    attn_io.append(r['attn_to_IO'])
    attn_s1.append(r['attn_to_S1'])
    attn_s2.append(r['attn_to_S2'])
    attn_prev.append(r['attn_to_prev'])

x = np.arange(len(steps))
width = 0.2

ax.bar(x - 1.5*width, attn_io, width, label='to IO', color='#4daf4a', alpha=0.85)
ax.bar(x - 0.5*width, attn_s1, width, label='to S1', color='#377eb8', alpha=0.85)
ax.bar(x + 0.5*width, attn_s2, width, label='to S2', color='#e41a1c', alpha=0.85)
ax.bar(x + 1.5*width, attn_prev, width, label='to prev', color='#984ea3', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(step_labels, fontsize=10)
ax.set_ylabel('Attention weight (from final token)')
ax.set_title('Figure 7: L8H9 Attention Development — Locks Onto S2 by Step 3000', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, 1.0)

# Annotations
ax.annotate('Uniform', xy=(0, 0.12), ha='center', fontsize=9, style='italic')
ax.annotate('S2=86.9%', xy=(1, 0.92), ha='center', fontsize=9, fontweight='bold', color='#e41a1c')
ax.annotate('S2=92.2%', xy=(2, 0.95), ha='center', fontsize=9, fontweight='bold', color='#e41a1c')

plt.tight_layout()
plt.savefig('figures/fig7_l8h9_attention_development.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig7_l8h9_attention_development.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 7: L8H9 attention development")


# ============================================================
# FIGURE 8: L8H9 Output Projection Over Training
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

proj_steps = [1000, 3000, 143000]
io_projs = [mega['exp_a_output_projection']['step_%d' % s]['L8H9']['mean_io_projection'] for s in proj_steps]
s_projs = [mega['exp_a_output_projection']['step_%d' % s]['L8H9']['mean_s_projection'] for s in proj_steps]
diffs = [mega['exp_a_output_projection']['step_%d' % s]['L8H9']['io_minus_s'] for s in proj_steps]

x = np.arange(len(proj_steps))
width = 0.35

ax1.bar(x - width/2, io_projs, width, label='IO projection', color='#4daf4a', alpha=0.85)
ax1.bar(x + width/2, s_projs, width, label='S projection', color='#e41a1c', alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(['Step 1000', 'Step 3000', 'Step 143000'])
ax1.set_ylabel('Mean projection onto unembedding direction')
ax1.set_title('L8H9: What it writes into residual stream')
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.legend()
ax1.grid(True, alpha=0.2, axis='y')

# Add values
for i, (io, s) in enumerate(zip(io_projs, s_projs)):
    ax1.text(i - width/2, io, '%.3f' % io, ha='center', va='bottom' if io >= 0 else 'top', fontsize=9)
    ax1.text(i + width/2, s, '%.3f' % s, ha='center', va='bottom' if s >= 0 else 'top', fontsize=9)

# Right panel: IO - S difference
ax2.bar(x, diffs, 0.5, color='#377eb8', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(['Step 1000', 'Step 3000', 'Step 143000'])
ax2.set_ylabel('IO projection − S projection')
ax2.set_title('Net effect: S-suppression strengthens over training')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.grid(True, alpha=0.2, axis='y')

for i, d in enumerate(diffs):
    ax2.text(i, d, '%.3f' % d, ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.suptitle('Figure 8: L8H9 Mechanism — Suppresses S Rather Than Copies IO', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig8_l8h9_output_projection.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig8_l8h9_output_projection.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 8: L8H9 output projection")


# ============================================================
# FIGURE 9: Dominant Head Attention Across Scales
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))

# Data: dominant head attention at step 143000
dominant_data = [
    ('160M\nL8H9', 0.029, 0.031, 0.922, C160),
    ('410M\nL4H6', 0.003, 0.005, 0.007, C410),
    ('1B\nL11H0', 0.055, 0.041, 0.605, C1B),
]

x = np.arange(3)
width = 0.25

io_vals = [d[1] for d in dominant_data]
s1_vals = [d[2] for d in dominant_data]
s2_vals = [d[3] for d in dominant_data]

ax.bar(x - width, io_vals, width, label='to IO', color='#4daf4a', alpha=0.85)
ax.bar(x, s1_vals, width, label='to S1', color='#377eb8', alpha=0.85)
ax.bar(x + width, s2_vals, width, label='to S2', color='#e41a1c', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in dominant_data])
ax.set_ylabel('Attention weight (from final token)')
ax.set_title('Figure 9: Dominant Head Attention at Step 143k — Mechanism Varies by Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, 1.0)

# Annotations
for i, d in enumerate(dominant_data):
    model, io, s1, s2, color = d
    if s2 > 0.5:
        ax.annotate('S2=%.1f%%' % (s2*100), xy=(i+width, s2), ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='#e41a1c')
    else:
        ax.annotate('attends\nelsewhere', xy=(i, 0.1), ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('figures/fig9_dominant_head_scales.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig9_dominant_head_scales.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 9: dominant head cross-scale")


# ============================================================
# FIGURE 10: Top-1 Predictions at Step 1000
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Top-1 breakdown at step 1000
probe3 = mega['final_probes']['probe3_top_predictions']
categories = ['IO', 'S', 'Other\n("the", "him")']
pcts = [probe3['io_top1_pct']*100, probe3['s_top1_pct']*100, probe3['other_top1_pct']*100]
colors = ['#4daf4a', '#e41a1c', '#984ea3']

bars = ax1.bar(categories, pcts, color=colors, alpha=0.85)
ax1.set_ylabel('% of examples where this is top-1 prediction')
ax1.set_title('Pythia-160M at Step 1000: What is top-1?')
ax1.set_ylim(0, 105)
for bar, pct in zip(bars, pcts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             '%.1f%%' % pct, ha='center', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.2, axis='y')

# Right: IO and S median ranks across models
models = ['160M', '410M', '1B']
io_ranks = [mega['exp_d_logit_distribution']['pythia_%s' % m.replace('M', 'm').replace('B', '000m')]['median_io_rank'] for m in models]
s_ranks = [mega['exp_d_logit_distribution']['pythia_%s' % m.replace('M', 'm').replace('B', '000m')]['median_s_rank'] for m in models]

x = np.arange(len(models))
width = 0.35
ax2.bar(x - width/2, io_ranks, width, label='IO median rank', color='#4daf4a', alpha=0.85)
ax2.bar(x + width/2, s_ranks, width, label='S median rank', color='#e41a1c', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel('Median rank (out of 50,257 tokens)')
ax2.set_title('Both names are deep in the ranking')
ax2.legend()
ax2.grid(True, alpha=0.2, axis='y')
for i, (io, s) in enumerate(zip(io_ranks, s_ranks)):
    ax2.text(i - width/2, io + 3, '%.0f' % io, ha='center', fontsize=9)
    ax2.text(i + width/2, s + 3, '%.0f' % s, ha='center', fontsize=9)

fig.suptitle('Figure 10: The Dip Reframed — Model Predicts "the", Not Names', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig10_top_predictions.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig10_top_predictions.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 10: top predictions reframing")

print("\nAll new figures saved. Combine with figures 1-6 from scripts/plot_all_figures.py")
