import torch, json, os, shutil
from transformer_lens import HookedTransformer
try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except:
    from src.circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

templates = ALL_TEMPLATES[:15]
ppt, seed = 20, 42

def clear_cache():
    cache_dir = '/workspace/.hf_home/hub'
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith('models--'):
                shutil.rmtree(os.path.join(cache_dir, d), ignore_errors=True)

###############################################################
# EXPERIMENT 1: Ablate L8H9 at step 143000
###############################################################
print("=" * 60)
print("  EXP 1: Ablate dominant head at final checkpoint")
print("=" * 60)

clear_cache()
model = HookedTransformer.from_pretrained('EleutherAI/pythia-160m-deduped',
    center_writing_weights=True, center_unembed=True, fold_ln=True,
    device='cuda', checkpoint_value=143000)

# Baseline
all_correct = []
for tmpl in templates:
    ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl], symmetric=True, seed=seed)
    tokens = model.to_tokens(ds.prompts).cuda()
    io_ids = torch.tensor(ds.io_token_ids, device='cuda')
    s_ids = torch.tensor(ds.s_token_ids, device='cuda')
    logits = model(tokens)
    last = logits[:, -1, :]
    ld = last[torch.arange(len(io_ids)), io_ids] - last[torch.arange(len(s_ids)), s_ids]
    all_correct.append((ld > 0).float().detach().cpu())
base_acc = torch.cat(all_correct).mean().item()
print("  Baseline (step 143000): acc=%.3f" % base_acc)

# Ablate L8H9
def hook_l8h9(value, hook):
    value[:, :, 9, :] = 0.0
    return value

all_correct = []
for tmpl in templates:
    ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl], symmetric=True, seed=seed)
    tokens = model.to_tokens(ds.prompts).cuda()
    io_ids = torch.tensor(ds.io_token_ids, device='cuda')
    s_ids = torch.tensor(ds.s_token_ids, device='cuda')
    logits = model.run_with_hooks(tokens, fwd_hooks=[('blocks.8.attn.hook_z', hook_l8h9)])
    last = logits[:, -1, :]
    ld = last[torch.arange(len(io_ids)), io_ids] - last[torch.arange(len(s_ids)), s_ids]
    all_correct.append((ld > 0).float().detach().cpu())
abl_acc = torch.cat(all_correct).mean().item()
diff = abl_acc - base_acc
print("  Ablate L8H9: acc=%.3f (%+.3f)" % (abl_acc, diff))
print("  Compare: at step 3000 ablation was -0.167")
print()

del model
torch.cuda.empty_cache()

###############################################################
# EXPERIMENT 2: Strip prefix tokens from Pile prompts
###############################################################
print("=" * 60)
print("  EXP 2: Pile prefix robustness test")
print("=" * 60)

clear_cache()
model = HookedTransformer.from_pretrained('EleutherAI/pythia-160m-deduped',
    center_writing_weights=True, center_unembed=True, fold_ln=True,
    device='cuda', checkpoint_value=143000)

with open('data/pile_ioi_natural.json') as f:
    pile_data = json.load(f)

# Filter to single-token IO names
valid = []
for e in pile_data:
    toks = model.to_tokens(" " + e['io_name'])
    if toks.shape[1] == 2:  # BOS + one token
        valid.append(e)

print("  Valid Pile examples: %d" % len(valid))

def eval_pile(model, examples, strip_before_io=False):
    correct = 0
    total = 0
    for e in examples:
        prompt = e['prompt']
        if strip_before_io:
            # Find IO name position and strip everything before it
            io_pos = prompt.find(e['io_name'])
            if io_pos > 0:
                prompt = prompt[io_pos:]
        
        tokens = model.to_tokens(prompt).cuda()
        io_tok = model.to_tokens(" " + e['io_name'])[0, 1].item()
        s_tok = model.to_tokens(" " + e['s_name'])[0, 1].item()
        
        logits = model(tokens)
        last = logits[0, -1, :]
        
        if last[io_tok] > last[s_tok]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0, total

# Full prompts (original)
acc_full, n = eval_pile(model, valid, strip_before_io=False)
print("  Full prompt:     acc=%.3f (n=%d)" % (acc_full, n))

# Stripped prompts (start at IO name)
acc_strip, n = eval_pile(model, valid, strip_before_io=True)
print("  Stripped prefix:  acc=%.3f (n=%d)" % (acc_strip, n))

diff = acc_strip - acc_full
print("  Difference: %+.3f" % diff)
if abs(diff) < 0.03:
    print("  --> Prefix doesn't matter much (robust)")
else:
    print("  --> Prefix matters (fragile)")
print()

###############################################################
# EXPERIMENT 3: General next-token prediction baseline
###############################################################
print("=" * 60)
print("  EXP 3: General next-token prediction on Pile")
print("=" * 60)

# For each Pile example, measure accuracy on predicting ALL tokens
# (not just the IO token) as a baseline
import numpy as np

all_accs = []
all_io_ranks = []

for e in valid[:100]:  # first 100 for speed
    tokens = model.to_tokens(e['prompt']).cuda()
    if tokens.shape[1] < 3:
        continue
    
    logits = model(tokens)
    
    # General next-token accuracy (top-1)
    preds = logits[0, :-1, :].argmax(dim=-1)  # predicted next tokens
    targets = tokens[0, 1:]  # actual next tokens
    correct = (preds == targets).float().mean().item()
    all_accs.append(correct)
    
    # Where does IO token rank at the final position?
    io_tok = model.to_tokens(" " + e['io_name'])[0, 1].item()
    last_logits = logits[0, -1, :]
    sorted_indices = last_logits.argsort(descending=True)
    rank = (sorted_indices == io_tok).nonzero(as_tuple=True)[0].item()
    all_io_ranks.append(rank)

mean_acc = np.mean(all_accs)
mean_rank = np.mean(all_io_ranks)
median_rank = np.median(all_io_ranks)

print("  General next-token accuracy: %.3f" % mean_acc)
print("  IO token mean rank: %.1f" % mean_rank)
print("  IO token median rank: %.1f" % median_rank)
print("  IOI accuracy (for reference): %.3f" % acc_full)
print()
print("  Interpretation: general accuracy = %.1f%%, IOI accuracy = %.1f%%" % (mean_acc * 100, acc_full * 100))
print("  The model is %s at IOI than general next-token prediction" % 
      ("better" if acc_full > mean_acc else "worse"))

del model
torch.cuda.empty_cache()

###############################################################
# SAVE ALL RESULTS
###############################################################
results = {
    "exp1_ablate_final": {
        "model": "160m",
        "step": 143000,
        "baseline_acc": base_acc,
        "ablate_L8H9_acc": abl_acc,
        "ablate_L8H9_diff": diff,
        "compare_step3000_diff": -0.167
    },
    "exp2_prefix_robustness": {
        "model": "160m",
        "step": 143000,
        "n_examples": n,
        "full_prompt_acc": acc_full,
        "stripped_prefix_acc": acc_strip,
        "difference": acc_strip - acc_full
    },
    "exp3_general_baseline": {
        "model": "160m",
        "step": 143000,
        "n_examples": len(all_accs),
        "general_next_token_acc": mean_acc,
        "io_mean_rank": mean_rank,
        "io_median_rank": median_rank,
        "ioi_accuracy": acc_full
    }
}

with open('results/quick_experiments.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nAll results saved to results/quick_experiments.json")
print("DONE")
