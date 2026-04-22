"""
Generate ALL figures for IOI Circuit Emergence paper.
Run on the pod with: python3 scripts/generate_all_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("figures", exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,

})

COLORS = {
    '160m': '#2196F3',
    '410m': '#FF9800',
    '1b': '#4CAF50',
    'stanford_alias': '#E91E63',
    'stanford_battlestar': '#9C27B0',
    'chance': '#999999',
}

# ============================================================
# LOAD ALL DATA
# ============================================================
def load(path):
    with open(path) as f:
        return json.load(f)

part1 = load('results/part1_component_emergence.json')
part2 = load('results/part2_pile_vs_synthetic.json')
mega = load('results/mega_experiments.json')
cole = load('results/cole_followups.json')
stanford = load('results/stanford_gpt2_ioi.json')
polypythias = load('results/polypythias_ioi.json')
final3 = load('results/final_three.json')
polish = load('results/polish_experiments.json')
two_more = load('results/two_more.json')

# Also load 410M and 1B
dev410 = load('results/dev_interp_EleutherAI_pythia-410m-deduped.json')
dev1b = load('results/dev_interp_EleutherAI_pythia-1b-deduped.json')


# ============================================================
# FIG 1: Universal Dip — Pythia (3 scales) + Stanford (2 seeds)
# ============================================================
def fig1_universal_dip():
    fig, ax = plt.subplots(figsize=(10, 5))

    # Pythia 160M
    steps_160 = [r['step'] for r in part1['results']]
    accs_160 = [r['performance']['accuracy'] * 100 for r in part1['results']]
    ax.plot(steps_160, accs_160, 'o-', color=COLORS['160m'], label='Pythia-160M', markersize=3, linewidth=1.5)

    # Pythia 410M
    steps_410 = [r['step'] for r in dev410['results']]
    accs_410 = [r['performance']['accuracy'] * 100 for r in dev410['results']]
    ax.plot(steps_410, accs_410, 's-', color=COLORS['410m'], label='Pythia-410M', markersize=3, linewidth=1.5)

    # Pythia 1B
    steps_1b = [r['step'] for r in dev1b['results']]
    accs_1b = [r['performance']['accuracy'] * 100 for r in dev1b['results']]
    ax.plot(steps_1b, accs_1b, '^-', color=COLORS['1b'], label='Pythia-1B', markersize=3, linewidth=1.5)

    # Stanford alias
    stan_steps = sorted(stanford['part1_sweep'].keys(), key=lambda x: int(x.split('_')[1]))
    sx = [int(s.split('_')[1]) for s in stan_steps]
    sy = [stanford['part1_sweep'][s]['accuracy'] * 100 for s in stan_steps]
    ax.plot(sx, sy, 'D-', color=COLORS['stanford_alias'], label='Stanford GPT-2 (alias)', markersize=3, linewidth=1.5)

    # Stanford battlestar
    bat_steps = sorted(stanford['part3_second_seed'].keys(), key=lambda x: int(x.split('_')[1]))
    bx = [int(s.split('_')[1]) for s in bat_steps]
    by = [stanford['part3_second_seed'][s]['accuracy'] * 100 for s in bat_steps]
    ax.plot(bx, by, 'v-', color=COLORS['stanford_battlestar'], label='Stanford GPT-2 (battlestar)', markersize=3, linewidth=1.5)

    ax.axhline(y=50, color=COLORS['chance'], linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('IOI Accuracy (%)')
    ax.set_title('Universal Below-Chance Dip Across Model Families')
    ax.set_xscale('log')
    ax.set_xlim(100, 500000)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.savefig('figures/fig1_universal_dip.png')
    plt.savefig('figures/fig1_universal_dip.pdf')
    plt.close()
    print("  Fig 1: Universal dip saved")


# ============================================================
# FIG 2: PolyPythias — 9 variants all dip
# ============================================================
def fig2_polypythias():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    checkpoints = [0, 512, 1000, 2000, 3000, 5000, 8000, 10000, 33000, 143000]

    groups = [
        ("Different Seeds", ['seed1', 'seed3', 'seed5'], ['#1976D2', '#388E3C', '#F57C00']),
        ("Data Order Only", ['data-seed1', 'data-seed2', 'data-seed3'], ['#7B1FA2', '#C2185B', '#00796B']),
        ("Weight Init Only", ['weight-seed1', 'weight-seed2', 'weight-seed3'], ['#5D4037', '#455A64', '#BF360C']),
    ]

    for idx, (title, labels, colors) in enumerate(groups):
        ax = axes[idx]
        for label, color in zip(labels, colors):
            if label in polypythias and 'checkpoints' in polypythias[label]:
                accs = []
                steps = []
                for step in checkpoints:
                    sk = 'step_%d' % step
                    if sk in polypythias[label]['checkpoints']:
                        steps.append(step)
                        accs.append(polypythias[label]['checkpoints'][sk]['accuracy'] * 100)
                ax.plot(steps, accs, 'o-', color=color, label=label, markersize=4, linewidth=1.5)

        ax.axhline(y=50, color=COLORS['chance'], linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Step')
        if idx == 0:
            ax.set_ylabel('IOI Accuracy (%)')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_xlim(300, 200000)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('IOI Dip Across 9 PolyPythias-160M Variants', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig2_polypythias.png')
    plt.savefig('figures/fig2_polypythias.pdf')
    plt.close()
    print("  Fig 2: PolyPythias saved")


# ============================================================
# FIG 3: High-Resolution Stanford Phase Transition
# ============================================================
def fig3_highres_transition():
    fig, ax = plt.subplots(figsize=(10, 5))

    # High-res from polish (61 points)
    hr = polish['exp3_phase_transition']
    hr_steps = sorted(hr.keys(), key=lambda x: int(x.split('_')[1]))
    hx = [int(s.split('_')[1]) for s in hr_steps]
    hy = [hr[s]['accuracy'] * 100 for s in hr_steps]
    ax.plot(hx, hy, 'o-', color=COLORS['stanford_alias'], markersize=3, linewidth=1, alpha=0.8)

    # Add verified volatile points
    volatile = two_more.get('volatile_retest', {})
    for sk, v in volatile.items():
        step = int(sk.split('_')[1])
        acc = v['accuracy_n600'] * 100
        se = v['std_err'] * 100
        ax.errorbar(step, acc, yerr=se*2, fmt='s', color='red', markersize=6, capsize=3, zorder=5)

    ax.axhline(y=50, color=COLORS['chance'], linestyle='--', alpha=0.5, label='Chance')

    # Annotate phases
    ax.axvspan(500, 1450, alpha=0.08, color='blue', label='Descent')
    ax.axvspan(1450, 2500, alpha=0.08, color='red', label='Noisy bottom')
    ax.axvspan(2500, 5000, alpha=0.08, color='green', label='Noisy recovery')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('IOI Accuracy (%)')
    ax.set_title('High-Resolution IOI Transition in Stanford GPT-2 Small (61 checkpoints)')
    ax.set_xlim(400, 5200)
    ax.set_ylim(0, 60)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.savefig('figures/fig3_highres_transition.png')
    plt.savefig('figures/fig3_highres_transition.pdf')
    plt.close()
    print("  Fig 3: High-res transition saved")


# ============================================================
# FIG 4: Rank/Probability Progression
# ============================================================
def fig4_rank_progression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    steps_order = ['step_1000', 'step_2000', 'step_3000', 'step_5000', 'step_8000', 'step_143000']
    step_labels = [1000, 2000, 3000, 5000, 8000, 143000]

    io_ranks = [cole['rank_progression'][s]['median_io_rank'] for s in steps_order]
    s_ranks = [cole['rank_progression'][s]['median_s_rank'] for s in steps_order]
    io_probs = [cole['rank_progression'][s]['mean_io_prob'] * 100 for s in steps_order]
    s_probs = [cole['rank_progression'][s]['mean_s_prob'] * 100 for s in steps_order]

    # Ranks
    ax1.plot(step_labels, io_ranks, 'o-', color='#2196F3', label='IO rank', linewidth=2, markersize=6)
    ax1.plot(step_labels, s_ranks, 's-', color='#F44336', label='S rank', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Median Rank (lower = better)')
    ax1.set_title('Name Token Rank During Training')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Probabilities
    ax2.plot(step_labels, io_probs, 'o-', color='#2196F3', label='IO probability', linewidth=2, markersize=6)
    ax2.plot(step_labels, s_probs, 's-', color='#F44336', label='S probability', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Mean Probability (%)')
    ax2.set_title('Name Token Probability During Training')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Pythia-160M: When Names Enter Consideration', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig4_rank_progression.png')
    plt.savefig('figures/fig4_rank_progression.pdf')
    plt.close()
    print("  Fig 4: Rank progression saved")


# ============================================================
# FIG 5: L8H9 Attention Phase Transition
# ============================================================
def fig5_head_trajectories():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    traj = final3['head_trajectories']
    steps_sorted = sorted(traj.keys(), key=lambda x: int(x.split('_')[1]))
    steps = [int(s.split('_')[1]) for s in steps_sorted]

    # L8H9 attention to different positions
    l8h9_s2 = [traj[s].get('L8H9', {}).get('attn_S2', 0) for s in steps_sorted]
    l8h9_io = [traj[s].get('L8H9', {}).get('attn_IO', 0) for s in steps_sorted]
    l8h9_self = [traj[s].get('L8H9', {}).get('attn_self', 0) for s in steps_sorted]
    accs = [traj[s].get('accuracy', 0) * 100 for s in steps_sorted]

    ax1.plot(steps, l8h9_s2, 'o-', color='#F44336', label='→ S2', linewidth=2, markersize=4)
    ax1.plot(steps, l8h9_io, 's-', color='#2196F3', label='→ IO', linewidth=2, markersize=4)
    ax1.plot(steps, l8h9_self, '^-', color='#999999', label='→ self', linewidth=1, markersize=3)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('L8H9 Attention Development')
    ax1.set_xscale('log')
    ax1.set_xlim(1, 200000)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate phase transition
    ax1.annotate('Phase transition\n(step 2000→3000)',
                xy=(2500, 0.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    # Multiple heads
    heads = ['L0H10', 'L8H9', 'L1H8']
    head_colors = ['#FF9800', '#F44336', '#4CAF50']
    for head, color in zip(heads, head_colors):
        vals = [traj[s].get(head, {}).get('attn_S2', 0) for s in steps_sorted]
        ax2.plot(steps, vals, 'o-', color=color, label=head + ' → S2', linewidth=1.5, markersize=3)

    # Add accuracy on twin axis
    ax2b = ax2.twinx()
    ax2b.plot(steps, accs, '--', color='grey', alpha=0.5, label='Accuracy')
    ax2b.set_ylabel('IOI Accuracy (%)', color='grey')
    ax2b.tick_params(axis='y', labelcolor='grey')

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Attention to S2 Position')
    ax2.set_title('Head Trajectories vs Accuracy')
    ax2.set_xscale('log')
    ax2.set_xlim(1, 200000)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig5_head_trajectories.png')
    plt.savefig('figures/fig5_head_trajectories.pdf')
    plt.close()
    print("  Fig 5: Head trajectories saved")


# ============================================================
# FIG 6: S-Suppression vs IO-Copying Mechanism
# ============================================================
def fig6_mechanism_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pythia 160M
    pythia_heads = {
        'L8H9\n(S-inhib)': {'attn_IO': 0.023, 'attn_S2': 0.925, 'proj_diff': 5.743},
        'L8H1\n(Name mover)': {'attn_IO': 0.724, 'attn_S2': 0.036, 'proj_diff': 0.526},
        'L9H1\n(Negative NM)': {'attn_IO': 0.836, 'attn_S2': 0.164, 'proj_diff': -1.020},
    }

    x = range(len(pythia_heads))
    names = list(pythia_heads.keys())
    proj_diffs = [pythia_heads[n]['proj_diff'] for n in names]
    colors_p = ['#F44336', '#2196F3', '#FF9800']

    bars1 = ax1.bar(x, proj_diffs, color=colors_p, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel('Output Projection (IO - S)')
    ax1.set_title('Pythia-160M: S-suppression dominates (10:1)')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, proj_diffs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                '%.1f' % val, ha='center', fontsize=10, fontweight='bold')

    # Stanford GPT-2
    stanford_heads = two_more.get('stanford_projections', {})
    stan_names = ['L10H10\n(S-inhib)', 'L10H4\n(Name mover)', 'L11H11\n(Negative NM)']
    stan_keys = ['L10H10', 'L10H4', 'L11H11']
    stan_diffs = [stanford_heads.get(k, {}).get('proj_diff', 0) for k in stan_keys]
    colors_s = ['#F44336', '#2196F3', '#FF9800']

    bars2 = ax2.bar(range(len(stan_names)), stan_diffs, color=colors_s, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(stan_names)))
    ax2.set_xticklabels(stan_names, fontsize=9)
    ax2.set_ylabel('Output Projection (IO - S)')
    ax2.set_title('Stanford GPT-2: S-suppression dominates (2.4:1)')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, stan_diffs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                '%.1f' % val, ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('S-Suppression vs IO-Copying: Cross-Family Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig6_mechanism_comparison.png')
    plt.savefig('figures/fig6_mechanism_comparison.pdf')
    plt.close()
    print("  Fig 6: Mechanism comparison saved")


# ============================================================
# FIG 7: Sensitivity Analysis
# ============================================================
def fig7_sensitivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sens = final3['sensitivity']
    taus = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    tau_labels = ['0.005', '0.01', '0.02', '0.05', '0.1', '0.2']

    # Left: counts at step 143000
    step = 'step_143000'
    if step in sens:
        nms = [sens[step]['thresholds']['tau_%.3f' % t]['name_movers'] for t in taus]
        negs = [sens[step]['thresholds']['tau_%.3f' % t]['negative_nm'] for t in taus]
        totals = [sens[step]['thresholds']['tau_%.3f' % t]['total_classified'] for t in taus]

        x = range(len(taus))
        ax1.bar([i-0.15 for i in x], nms, 0.3, color='#2196F3', label='Name Movers')
        ax1.bar([i+0.15 for i in x], negs, 0.3, color='#F44336', label='Negative NM')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tau_labels)
        ax1.set_xlabel('Threshold (τ)')
        ax1.set_ylabel('Number of Heads')
        ax1.set_title('Step 143000 (100% accuracy)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

    # Right: % classified across steps
    key_steps = ['step_512', 'step_1000', 'step_2000', 'step_3000', 'step_5000', 'step_143000']
    step_labels_s = ['512', '1K', '2K', '3K', '5K', '143K']
    colors_tau = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']

    for i, tau in enumerate([0.02, 0.1, 0.2]):
        pcts = []
        for s in key_steps:
            if s in sens:
                pcts.append(sens[s]['thresholds']['tau_%.3f' % tau]['pct_classified'] * 100)
            else:
                pcts.append(0)
        ax2.plot(range(len(key_steps)), pcts, 'o-', label='τ=%.2f' % tau, linewidth=2, markersize=5)

    ax2.set_xticks(range(len(key_steps)))
    ax2.set_xticklabels(step_labels_s)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('% Heads Classified')
    ax2.set_title('Classification Rate Across Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Sensitivity of Ablation-Based Head Classification to τ Threshold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig7_sensitivity.png')
    plt.savefig('figures/fig7_sensitivity.pdf')
    plt.close()
    print("  Fig 7: Sensitivity saved")


# ============================================================
# FIG 8: Pile vs Synthetic
# ============================================================
def fig8_pile_vs_synthetic():
    fig, ax = plt.subplots(figsize=(10, 5))

    synth_steps = [r['step'] for r in part2['results']]
    synth_accs = []
    pile_accs = []
    pile_steps = []
    for r in part2['results']:
        if r['synthetic']:
            synth_accs.append(r['synthetic']['accuracy'] * 100)
        else:
            synth_accs.append(None)
        if r['pile'] and r['pile']['accuracy'] is not None:
            pile_accs.append(r['pile']['accuracy'] * 100)
            pile_steps.append(r['step'])

    # Filter Nones for synthetic
    valid_synth = [(s, a) for s, a in zip(synth_steps, synth_accs) if a is not None]
    if valid_synth:
        sx, sy = zip(*valid_synth)
        ax.plot(sx, sy, 'o-', color=COLORS['160m'], label='Synthetic', markersize=3, linewidth=1.5)

    if pile_steps:
        ax.plot(pile_steps, pile_accs, 's-', color='#F44336', label='Pile (natural)', markersize=3, linewidth=1.5)

    ax.axhline(y=50, color=COLORS['chance'], linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('IOI Accuracy (%)')
    ax.set_title('Pythia-160M: Synthetic vs Natural (Pile) IOI Performance')
    ax.set_xscale('log')
    ax.set_xlim(1, 200000)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig('figures/fig8_pile_vs_synthetic.png')
    plt.savefig('figures/fig8_pile_vs_synthetic.pdf')
    plt.close()
    print("  Fig 8: Pile vs synthetic saved")


# ============================================================
# FIG 9: Wang et al. Classification Over Training
# ============================================================
def fig9_wang_classification():
    fig, ax = plt.subplots(figsize=(10, 5))

    wang = final3['wang_classification']
    steps_sorted = sorted(wang.keys(), key=lambda x: int(x.split('_')[1]))
    steps = [int(s.split('_')[1]) for s in steps_sorted]

    roles = ['name_mover', 's_inhibition', 'duplicate_token', 'previous_token']
    role_labels = ['Name Mover', 'S-Inhibition', 'Duplicate Token', 'Previous Token']
    role_colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    for role, label, color in zip(roles, role_labels, role_colors):
        counts = [wang[s]['counts'].get(role, 0) for s in steps_sorted]
        ax.plot(steps, counts, 'o-', color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Number of Heads')
    ax.set_title('Wang et al. Attention-Based Head Classification Over Training')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig('figures/fig9_wang_classification.png')
    plt.savefig('figures/fig9_wang_classification.pdf')
    plt.close()
    print("  Fig 9: Wang classification saved")


# ============================================================
# FIG 10: Recovery Instability (Volatile Steps)
# ============================================================
def fig10_recovery_instability():
    fig, ax = plt.subplots(figsize=(10, 5))

    # High-res data
    hr = polish['exp3_phase_transition']
    hr_steps = sorted(hr.keys(), key=lambda x: int(x.split('_')[1]))
    hx = [int(s.split('_')[1]) for s in hr_steps]
    hy = [hr[s]['accuracy'] * 100 for s in hr_steps]
    ax.plot(hx, hy, 'o-', color=COLORS['stanford_alias'], markersize=4, linewidth=1, alpha=0.7, label='n=300')

    # Verified volatile points with error bars
    volatile = two_more.get('volatile_retest', {})
    vx, vy, verr = [], [], []
    for sk in sorted(volatile.keys(), key=lambda x: int(x.split('_')[1])):
        v = volatile[sk]
        vx.append(int(sk.split('_')[1]))
        vy.append(v['accuracy_n600'] * 100)
        verr.append(v['std_err'] * 200)  # 2x SE for 95% CI

    ax.errorbar(vx, vy, yerr=verr, fmt='s', color='red', markersize=7, capsize=4,
                linewidth=2, label='n=600 (verified)', zorder=5)

    ax.axhline(y=50, color=COLORS['chance'], linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('IOI Accuracy (%)')
    ax.set_title('Circuit Formation Instability: Repeated Formation and Collapse')
    ax.set_xlim(1400, 4600)
    ax.set_ylim(0, 60)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate key oscillations
    ax.annotate('Circuit forms\n(50%)', xy=(4100, 50), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'),
                xytext=(4300, 55), color='red')
    ax.annotate('Circuit collapses\n(24%)', xy=(4300, 24), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='blue'),
                xytext=(4500, 10), color='blue')

    plt.savefig('figures/fig10_recovery_instability.png')
    plt.savefig('figures/fig10_recovery_instability.pdf')
    plt.close()
    print("  Fig 10: Recovery instability saved")


# ============================================================
# FIG 11: Cross-Scale Pile Ablation
# ============================================================
def fig11_pile_ablation():
    fig, ax = plt.subplots(figsize=(8, 5))

    pile_data = mega['exp_c_pile_ablation']
    models = ['pythia_160m', 'pythia_410m', 'pythia_1000m']
    model_labels = ['160M\n(L8H9)', '410M\n(L4H6)', '1B\n(L11H0)']
    baseline = [pile_data[m]['pile_baseline_acc'] * 100 for m in models]
    ablated = [pile_data[m]['pile_ablated_acc'] * 100 for m in models]
    diffs = [pile_data[m]['pile_ablation_diff'] * 100 for m in models]

    x = range(len(models))
    bars1 = ax.bar([i-0.15 for i in x], baseline, 0.3, color='#2196F3', label='Baseline')
    bars2 = ax.bar([i+0.15 for i in x], ablated, 0.3, color='#F44336', label='Ablated')

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel('Pile IOI Accuracy (%)')
    ax.set_title('Effect of Dominant Head Ablation on Natural (Pile) IOI')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, d in enumerate(diffs):
        ax.text(i, max(baseline[i], ablated[i]) + 2, '%.1fpp' % d,
                ha='center', fontsize=10, fontweight='bold', color='red')

    plt.savefig('figures/fig11_pile_ablation.png')
    plt.savefig('figures/fig11_pile_ablation.pdf')
    plt.close()
    print("  Fig 11: Pile ablation saved")


# ============================================================
# FIG 12: Cross-Family Mechanism Summary
# ============================================================
def fig12_mechanism_summary():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a comparison table as a figure
    data = [
        ['', 'Pythia-160M', 'Stanford GPT-2', 'Match?'],
        ['Dominant head', 'L8H9', 'L10H10', '—'],
        ['Wang classification', 'S-inhibition', 'S-inhibition', '✓'],
        ['Attention to S2', '92.5%', '59.3%', '✓'],
        ['S projection', '-6.22', '-1.94', '✓'],
        ['IO-S proj diff', '+5.74', '+1.89', '✓'],
        ['Actual name mover', 'L8H1', 'L10H4', '—'],
        ['NM proj diff', '+0.53', '+0.80', '✓'],
        ['S-supp / IO-copy', '10.8:1', '2.4:1', '~'],
        ['Negative NM', 'L9H1', 'L11H11', '✓'],
        ['NegNM proj diff', '-1.02', '-1.15', '✓'],
    ]

    ax.axis('off')
    table = ax.table(cellText=data[1:], colLabels=data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color header
    for j in range(4):
        table[(0, j)].set_facecolor('#E3F2FD')
        table[(0, j)].set_text_props(fontweight='bold')

    # Color match column
    for i in range(1, len(data)):
        cell = table[(i, 3)]
        if data[i][3] == '✓':
            cell.set_facecolor('#E8F5E9')
        elif data[i][3] == '~':
            cell.set_facecolor('#FFF9C4')

    ax.set_title('Cross-Family Mechanism Comparison', fontsize=14, pad=20)

    plt.savefig('figures/fig12_mechanism_summary.png')
    plt.savefig('figures/fig12_mechanism_summary.pdf')
    plt.close()
    print("  Fig 12: Mechanism summary saved")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Generating all figures...")
    print()

    fig1_universal_dip()
    fig2_polypythias()
    fig3_highres_transition()
    fig4_rank_progression()
    fig5_head_trajectories()
    fig6_mechanism_comparison()
    fig7_sensitivity()
    fig8_pile_vs_synthetic()
    fig9_wang_classification()
    fig10_recovery_instability()
    fig11_pile_ablation()
    fig12_mechanism_summary()

    print()
    print("All 12 figures saved to figures/")
    print("  PNG (for viewing) + PDF (for paper)")

if __name__ == "__main__":
    main()
