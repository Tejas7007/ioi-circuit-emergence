"""
Generate all figures for IOI Circuit Emergence paper.
Reads from results/ directory, saves to figures/ directory.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# Color scheme
C160 = '#1f77b4'   # blue
C410 = '#ff7f0e'   # orange  
C1B  = '#2ca02c'   # green
CSYN = '#d62728'   # red
CPILE = '#9467bd'  # purple

# Load all data
with open('results/part1_component_emergence.json') as f:
    p1_160m = json.load(f)
with open('results/dev_interp_EleutherAI_pythia-410m-deduped.json') as f:
    p1_410m = json.load(f)
with open('results/dev_interp_EleutherAI_pythia-1b-deduped.json') as f:
    p1_1b = json.load(f)

with open('results/part2_pile_vs_synthetic.json') as f:
    p2_160m = json.load(f)
with open('results/dev_interp_grokking_EleutherAI_pythia-410m-deduped.json') as f:
    p2_410m = json.load(f)
with open('results/dev_interp_grokking_EleutherAI_pythia-1b-deduped.json') as f:
    p2_1b = json.load(f)

with open('results/induction_emergence.json') as f:
    induction = json.load(f)

with open('results/ablation_cross_scale.json') as f:
    ablation = json.load(f)

with open('results/quick_experiments.json') as f:
    quick = json.load(f)

def get_steps_and_accs(data):
    steps = [r['step'] for r in data['results']]
    accs = [r['performance']['accuracy'] * 100 for r in data['results']]
    return steps, accs

def get_steps_and_nms(data):
    steps = [r['step'] for r in data['results']]
    nms = [r['n_name_movers'] for r in data['results']]
    return steps, nms

def safe_log_steps(steps):
    return [max(s, 0.5) for s in steps]

# ============================================================
# FIGURE 1: IOI Accuracy Across Training (3 models)
# Two panels: log scale full + linear zoom 0-10000
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for data, color, label in [
    (p1_160m, C160, 'Pythia-160M'),
    (p1_410m, C410, 'Pythia-410M'),
    (p1_1b, C1B, 'Pythia-1B'),
]:
    steps, accs = get_steps_and_accs(data)
    
    # Left: log scale full training
    ax1.plot(safe_log_steps(steps), accs, 'o-', color=color, label=label, 
             markersize=4, linewidth=1.5)
    
    # Right: linear zoom 0-10000
    mask = [i for i, s in enumerate(steps) if s <= 10000]
    ax2.plot([steps[i] for i in mask], [accs[i] for i in mask], 'o-', 
             color=color, label=label, markersize=5, linewidth=1.5)

ax1.set_xscale('log')
ax1.set_xlabel('Training Step (log scale)')
ax1.set_ylabel('IOI Accuracy (%)')
ax1.set_title('Full Training')
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax1.legend(fontsize=9)
ax1.set_ylim(25, 105)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Training Step')
ax2.set_ylabel('IOI Accuracy (%)')
ax2.set_title('Transition Zone (0–10,000)')
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.axvspan(800, 2500, alpha=0.1, color='red', label='Performance dip')
ax2.legend(fontsize=9)
ax2.set_ylim(25, 105)
ax2.grid(True, alpha=0.3)

fig.suptitle('Figure 1: IOI Accuracy During Training — "Worse Before Better"', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig1_accuracy_across_training.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig1_accuracy_across_training.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 1: accuracy across training")

# ============================================================
# FIGURE 2: Pile vs Synthetic (3 models, 3 subplots)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, p2_data, title in [
    (axes[0], p2_160m, 'Pythia-160M'),
    (axes[1], p2_410m, 'Pythia-410M'),
    (axes[2], p2_1b, 'Pythia-1B'),
]:
    steps = [r['step'] for r in p2_data['results']]
    syn_acc = [r['synthetic']['accuracy'] * 100 for r in p2_data['results']]
    pile_acc = [r['pile']['accuracy'] * 100 for r in p2_data['results']]
    
    ax.plot(safe_log_steps(steps), syn_acc, 'o-', color=CSYN, label='Synthetic', 
            markersize=4, linewidth=1.5)
    ax.plot(safe_log_steps(steps), pile_acc, 's-', color=CPILE, label='Pile (natural)', 
            markersize=4, linewidth=1.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Step (log scale)')
    ax.set_ylabel('IOI Accuracy (%)')
    ax.set_title(title)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(800, 2500, alpha=0.08, color='red')
    ax.legend(fontsize=9)
    ax.set_ylim(25, 105)
    ax.grid(True, alpha=0.3)

fig.suptitle('Figure 2: Pile vs Synthetic IOI Performance — Natural IOI Leads During Dip', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig2_pile_vs_synthetic.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig2_pile_vs_synthetic.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 2: pile vs synthetic")

# ============================================================
# FIGURE 3: Name-Mover Counts + Top Head Labels
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, data, title in [
    (axes[0], p1_160m, 'Pythia-160M'),
    (axes[1], p1_410m, 'Pythia-410M'),
    (axes[2], p1_1b, 'Pythia-1B'),
]:
    steps, nms = get_steps_and_nms(data)
    steps_acc, accs = get_steps_and_accs(data)
    
    ax2 = ax.twinx()
    
    bars = ax.bar(range(len(steps)), nms, color='steelblue', alpha=0.6, label='Name-movers')
    ax2.plot(range(len(steps)), accs, 'o-', color='darkred', linewidth=2, markersize=4, label='Accuracy')
    
    # Label top head at key steps
    for i, r in enumerate(data['results']):
        if r['step'] in [1000, 3000, 66000, 143000] and r.get('top_ioi_head'):
            ax2.annotate(r['top_ioi_head'], (i, accs[i]), fontsize=7,
                        ha='center', va='bottom', color='darkred', fontweight='bold')
    
    ax.set_xticks(range(0, len(steps), 3))
    ax.set_xticklabels([str(steps[i]) for i in range(0, len(steps), 3)], rotation=45, fontsize=7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('# Name-Mover Heads', color='steelblue')
    ax2.set_ylabel('IOI Accuracy (%)', color='darkred')
    ax.set_title(title)
    ax2.set_ylim(25, 105)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

fig.suptitle('Figure 3: Name-Mover Component Count vs Task Performance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig3_components_vs_accuracy.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig3_components_vs_accuracy.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 3: components vs accuracy")

# ============================================================
# FIGURE 4: Induction Head Emergence (monotonic) vs IOI (non-monotonic)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Induction head max score across training
for model_key, color, label in [
    ('EleutherAI/pythia-160m-deduped', C160, '160M'),
    ('EleutherAI/pythia-410m-deduped', C410, '410M'),
    ('EleutherAI/pythia-1b-deduped', C1B, '1B'),
]:
    data = induction[model_key]
    steps = [r['step'] for r in data]
    scores = [r['max_induction_score'] for r in data]
    ax1.plot(safe_log_steps(steps), scores, 'o-', color=color, label=label, 
             markersize=5, linewidth=1.5)

ax1.set_xscale('log')
ax1.set_xlabel('Training Step (log scale)')
ax1.set_ylabel('Max Induction Score')
ax1.set_title('Induction Heads: Monotonic Emergence')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.0)

# Right: IOI accuracy (non-monotonic) for comparison
for data, color, label in [
    (p1_160m, C160, '160M'),
    (p1_410m, C410, '410M'),
    (p1_1b, C1B, '1B'),
]:
    steps, accs = get_steps_and_accs(data)
    ax2.plot(safe_log_steps(steps), [a/100 for a in accs], 'o-', color=color, 
             label=label, markersize=5, linewidth=1.5)

ax2.set_xscale('log')
ax2.set_xlabel('Training Step (log scale)')
ax2.set_ylabel('IOI Accuracy (fraction)')
ax2.set_title('IOI Circuit: Non-Monotonic Emergence')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

fig.suptitle('Figure 4: Different Circuits, Different Developmental Dynamics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig4_induction_vs_ioi.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig4_induction_vs_ioi.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 4: induction vs IOI")

# ============================================================
# FIGURE 5: Ablation Results Across Scales
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# 160M ablation data (from earlier experiments, hardcoded)
abl_160m = {
    'step1000': {'baseline': 0.407, 'L0H5': -0.013, 'L0H6': -0.020, 'L0H10': -0.023, 'L1H3': -0.007, 'L1H7': -0.023, 'ALL': -0.050},
    'step3000': {'baseline': 0.613, 'L0H5': +0.027, 'L0H6': 0.000, 'L0H10': +0.117, 'L1H3': -0.003, 'L1H7': -0.017, 'L8H9': -0.167},
}

# 160M
ax = axes[0]
ax.set_title('Pythia-160M', fontsize=11)
heads_1k = ['L0H5', 'L0H6', 'L0H10', 'L1H3', 'L1H7', 'ALL']
diffs_1k = [abl_160m['step1000'][h] for h in heads_1k]
heads_3k = ['L0H5', 'L0H6', 'L0H10', 'L1H3', 'L1H7', 'L8H9']
diffs_3k = [abl_160m['step3000'][h] for h in heads_3k]

x = np.arange(len(heads_1k))
width = 0.35
bars1 = ax.bar(x - width/2, [d*100 for d in diffs_1k], width, label='Step 1000', color='#4292c6', alpha=0.8)
bars2 = ax.bar(x + width/2, [d*100 for d in diffs_3k], width, label='Step 3000', color='#ef6548', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(heads_1k, rotation=45, fontsize=8)
ax.set_ylabel('Accuracy change (pp)')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# 410M
ax = axes[1]
ax.set_title('Pythia-410M', fontsize=11)
abl_410m_1k = ablation.get('EleutherAI/pythia-410m-deduped_step1000', {})
abl_410m_3k = ablation.get('EleutherAI/pythia-410m-deduped_step3000', {})

heads = ['L5H11']
d1k = [abl_410m_1k.get('ablations', {}).get('L5H11', {}).get('diff', 0)]
d3k_heads = ['L5H11', 'L5H2\n(dominant)']
d3k = [
    abl_410m_3k.get('ablations', {}).get('L5H11', {}).get('diff', 0),
    abl_410m_3k.get('ablations', {}).get('L5H2_dominant', {}).get('diff', 0),
]

x = np.arange(2)
ax.bar([0 - 0.175], [d1k[0]*100], 0.35, label='Step 1000', color='#4292c6', alpha=0.8)
ax.bar([0 + 0.175, 1], [d3k[0]*100, d3k[1]*100], 0.35, label='Step 3000', color='#ef6548', alpha=0.8)
ax.set_xticks([0, 1])
ax.set_xticklabels(['L5H11', 'L5H2\n(dominant)'], fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# 1B
ax = axes[2]
ax.set_title('Pythia-1B', fontsize=11)
abl_1b_1k = ablation.get('EleutherAI/pythia-1b-deduped_step1000', {})
abl_1b_3k = ablation.get('EleutherAI/pythia-1b-deduped_step3000', {})

heads_1b = ['L0H0', 'L0H4', 'L0H5', 'L0H6', 'L1H3', 'ALL']
d1k_1b = []
d3k_1b = []
for h in ['L0H0', 'L0H4', 'L0H5', 'L0H6', 'L1H3']:
    d1k_1b.append(abl_1b_1k.get('ablations', {}).get(h, {}).get('diff', 0))
    d3k_1b.append(abl_1b_3k.get('ablations', {}).get(h, {}).get('diff', 0))
d1k_1b.append(abl_1b_1k.get('ablations', {}).get('ALL_early', {}).get('diff', 0))
# For step 3000, use ALL_early and add L8H7 dominant
d3k_1b.append(abl_1b_3k.get('ablations', {}).get('L8H7_dominant', {}).get('diff', 0))
heads_1b_3k = ['L0H0', 'L0H4', 'L0H5', 'L0H6', 'L1H3', 'L8H7\n(dom)']

x = np.arange(len(heads_1b))
width = 0.35
ax.bar(x - width/2, [d*100 for d in d1k_1b], width, label='Step 1000', color='#4292c6', alpha=0.8)
ax.bar(x + width/2, [d*100 for d in d3k_1b], width, label='Step 3000', color='#ef6548', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(heads_1b_3k, rotation=45, fontsize=7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Figure 5: Ablation of Early Name-Movers — Helpful Then Harmful', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig5_ablation_results.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig5_ablation_results.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 5: ablation results")

# ============================================================
# FIGURE 6: Head Succession Timeline
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4))

# Data from our results
models_succession = {
    '160M': [
        ('L0H0', 1000, 1000, '#aec7e8'),
        ('L5H8', 2000, 2000, '#c7c7c7'),
        ('L8H9', 3000, 143000, C160),
    ],
    '410M': [
        ('L5H11', 1000, 1000, '#aec7e8'),
        ('L11H3', 2000, 2000, '#c7c7c7'),
        ('L5H2', 3000, 5000, '#ffbb78'),
        ('L12H12', 6000, 33000, C410),
        ('L4H6', 66000, 143000, '#d62728'),
    ],
    '1B': [
        ('L0H5', 1000, 1000, '#aec7e8'),
        ('L2H0', 2000, 2000, '#c7c7c7'),
        ('L8H7', 3000, 33000, C1B),
        ('L11H0', 66000, 143000, '#d62728'),
    ],
}

y_positions = {'160M': 2, '410M': 1, '1B': 0}
for model_name, heads in models_succession.items():
    y = y_positions[model_name]
    for head_name, start, end, color in heads:
        width = max(end - start, 500)
        ax.barh(y, width, left=start, height=0.5, color=color, edgecolor='white', linewidth=0.5)
        mid = start + width / 2
        if width > 3000:
            ax.text(mid, y, head_name, ha='center', va='center', fontsize=8, fontweight='bold')
        else:
            ax.text(mid, y + 0.3, head_name, ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['1B', '410M', '160M'])
ax.set_xlabel('Training Step')
ax.set_xscale('log')
ax.set_xlim(500, 200000)
ax.set_title('Figure 6: Dominant Head Succession Across Training', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('figures/fig6_head_succession.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/fig6_head_succession.pdf', bbox_inches='tight')
plt.close()
print("Saved Figure 6: head succession")

print("\nAll figures saved to figures/")
print("Files:", sorted(os.listdir('figures')))
