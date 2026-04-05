# IOI Circuit Emergence

Studying when and how the Indirect Object Identification (IOI) circuit emerges during Pythia language model training.

**Author:** Tejas Dahiya, UW-Madison
**Advisor:** Cole Blondin

## Research Questions

1. When does the model become good at IOI on the synthetic IOI dataset? Is it contemporaneous with the emergence of identifiable name-movers and other components, or was the model able to do the task before the circuit emerged?
2. When does the model become good at IOI on its own pretraining data? Is it before the model can do IOI on the synthetic IOI dataset?

## Key Results

### Part 1: Component Emergence (Pythia-160M, 24 checkpoints)

At each training checkpoint, we measure IOI performance and classify every attention head into functional roles (name-movers, other IOI, subject-promoters, copy-suppression) using delta_ioi and delta_anti metrics.

| Step | Logit Diff | Accuracy | Name-Movers | Total IOI | Top Head |
|------|-----------|----------|-------------|-----------|----------|
| 0 | +0.027 | 52.3% | 0 | 0 | L0H2 |
| 512 | -0.042 | 47.3% | 0 | 0 | L3H1 |
| 1000 | -0.307 | 40.7% | 8 | 12 | L0H0 |
| 2000 | -0.589 | 35.0% | 30 | 50 | L5H8 |
| 3000 | +0.595 | 61.3% | 31 | 56 | L8H9 |
| 4000 | +1.679 | 84.3% | 27 | 47 | L8H9 |
| 8000 | +2.826 | 98.3% | 27 | 56 | L8H9 |
| 10000 | +3.745 | 100.0% | 24 | 50 | L8H9 |
| 66000 | +4.206 | 99.3% | 36 | 67 | L8H9 |
| 143000 | +4.132 | 100.0% | 25 | 58 | L8H9 |

**Finding: "Worse before better."** Components emerge at step 1000 (8 name-movers) but the model gets WORSE at IOI (accuracy drops from 52% to 41%). By step 2000, 30 name-movers exist but accuracy is at its lowest (35%). At step 3000, L8H9 appears and reorganizes the circuit, flipping performance positive. L8H9 remains the dominant head for the rest of training.

### Part 2: Pile vs Synthetic IOI (288 natural examples from The Pile)

We compare IOI performance on the synthetic Wang et al. dataset vs 288 naturally occurring IOI examples from Pythia's actual training data (The Pile, deduplicated). Every Pile example is validated to contain both IO and S names in the prompt. 174 examples have single-token IO names and are used for evaluation.

| Step | Synthetic Acc | Pile Acc (n=174) | Gap |
|------|-------------|-----------------|-----|
| 0 | 52.3% | 45.4% | -6.9pp |
| 1000 | 40.7% | **50.6%** | **+9.9pp** |
| 2000 | 35.0% | **51.1%** | **+16.1pp** |
| 3000 | 61.3% | 56.9% | -4.4pp |
| 4000 | 84.3% | 63.2% | -21.1pp |
| 33000 | 98.7% | 71.3% | -27.4pp |
| 66000 | 99.3% | 74.1% | -25.2pp |
| 143000 | 100.0% | 66.7% | -33.3pp |

**Findings:**
- **Steps 1000-2000: Pile leads synthetic.** While the circuit is forming and hurting synthetic performance (35%), the model maintains ~51% on natural text.
- **Synthetic reaches 70% at step 4000. Pile reaches 70% at step 33000.** The circuit specializes for synthetic templates first, with an 8x delay before helping natural IOI.
- **Pile peaks at 74% then declines to 67%.** Late-training degradation on natural IOI.

## Datasets

### Synthetic IOI
Wang et al. 2022 dataset: 136 names, 30 templates (15 ABBA + 15 BABA), 300 prompts evaluated per checkpoint.

### Natural IOI (288 examples)
Extracted from EleutherAI/the_pile_deduplicated (Pythia's actual training data). Each example has a transfer verb, both IO and S names in the prompt, and the S name repeated. Scanned 13.9M texts. Blacklisted crypto (Alice/Bob), property records, and historical titles. Filtered out bAbI synthetic QA examples (42% of initial finds). 288 clean natural language examples, 174 with single-token IO names used for evaluation.

## Repo Structure

- **scripts/dev_interp_checkpoints.py** - Part 1: component emergence at each checkpoint
- **scripts/dev_interp_pile_vs_synthetic.py** - Part 2: Pile vs synthetic comparison
- **scripts/parse_pile_ioi.py** - Pile IOI example extraction
- **scripts/plot_results.py** - Figure generation
- **data/pile_ioi_natural.json** - 288 validated natural IOI examples
- **data/example_prompts.txt** - Side-by-side synthetic vs natural prompt samples
- **results/part1_component_emergence.json** - Part 1 results
- **results/part2_pile_vs_synthetic.json** - Part 2 results
- **figures/** - All plots (png + pdf)

## Component Classification (tau = 0.02)
- **Name-mover:** delta_ioi < -tau AND delta_anti > tau
- **Other IOI:** delta_ioi < -tau AND delta_anti <= tau
- **Subject-promoter:** delta_ioi > tau AND delta_anti < -tau
- **Copy-suppression:** delta_anti > tau

## References
- Wang et al. 2022. "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small"
- Biderman et al. 2023. "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"
- Power et al. 2022. "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
- Nanda et al. 2023. "Progress measures for grokking via mechanistic interpretability"
