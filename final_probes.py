import torch, json, os, shutil
import numpy as np
from transformer_lens import HookedTransformer
try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except:
    from src.circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

def clear_cache():
    cache_dir = '/workspace/.hf_home/hub'
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith('models--'):
                shutil.rmtree(os.path.join(cache_dir, d), ignore_errors=True)

TEMPLATES = ALL_TEMPLATES[:15]
PPT, SEED = 20, 42
results = {}

###############################################################
# PROBE 1: Full attention distribution of 410M L4H6
###############################################################
print("=" * 60)
print("  PROBE 1: Where does 410M L4H6 attend?")
print("=" * 60)

clear_cache()
model = HookedTransformer.from_pretrained('EleutherAI/pythia-410m-deduped',
    center_writing_weights=True, center_unembed=True, fold_ln=True,
    device='cuda', checkpoint_value=143000)

# Collect full attention pattern for L4H6
attn_by_position = {}  # relative position name -> attention weight
all_attn_maps = []

for tmpl in TEMPLATES[:10]:
    ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                    symmetric=True, seed=SEED)
    tokens = model.to_tokens(ds.prompts).cuda()
    io_ids = torch.tensor(ds.io_token_ids, device='cuda')
    s_ids = torch.tensor(ds.s_token_ids, device='cuda')

    _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    attn = cache["blocks.4.attn.hook_pattern"]
    final_pos = tokens.shape[1] - 1

    for i in range(tokens.shape[0]):
        io_tok = io_ids[i].item()
        s_tok = s_ids[i].item()

        # Find positions
        io_pos = -1
        s1_pos = -1
        s2_pos = -1
        s_count = 0
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == io_tok and io_pos == -1:
                io_pos = j
            if tokens[i, j].item() == s_tok:
                s_count += 1
                if s_count == 1: s1_pos = j
                elif s_count == 2: s2_pos = j

        full_attn = attn[i, 6, final_pos, :].detach().cpu().numpy()

        # Categorize attention
        positions = {
            "to_IO": full_attn[io_pos] if io_pos > 0 else 0,
            "to_S1": full_attn[s1_pos] if s1_pos > 0 else 0,
            "to_S2": full_attn[s2_pos] if s2_pos > 0 else 0,
            "to_BOS": full_attn[0],
            "to_prev": full_attn[final_pos - 1] if final_pos > 0 else 0,
            "to_final": full_attn[final_pos],
        }
        # Everything else
        accounted = sum(positions.values())
        positions["to_other"] = float(full_attn.sum()) - accounted

        for k, v in positions.items():
            if k not in attn_by_position:
                attn_by_position[k] = []
            attn_by_position[k].append(float(v))

    del cache
    torch.cuda.empty_cache()

probe1_result = {}
for k in attn_by_position:
    probe1_result[k] = round(float(np.mean(attn_by_position[k])), 4)
    print("  %s: %.4f" % (k, probe1_result[k]))

# Also check: what are the top 5 attended positions on average?
print("\n  Top attended position types:")
sorted_attn = sorted(probe1_result.items(), key=lambda x: -x[1])
for k, v in sorted_attn:
    print("    %s: %.4f (%.1f%%)" % (k, v, v * 100))

results["probe1_410m_l4h6_attention"] = probe1_result

del model
torch.cuda.empty_cache()

###############################################################
# PROBE 2: L8H9 behavior on Pile vs Synthetic at step 1000
###############################################################
print("\n" + "=" * 60)
print("  PROBE 2: L8H9 on Pile vs Synthetic at step 1000")
print("=" * 60)

clear_cache()
model = HookedTransformer.from_pretrained('EleutherAI/pythia-160m-deduped',
    center_writing_weights=True, center_unembed=True, fold_ln=True,
    device='cuda', checkpoint_value=1000)

# Synthetic attention
syn_attn_io = []
syn_attn_s1 = []
syn_attn_s2 = []

for tmpl in TEMPLATES[:10]:
    ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                    symmetric=True, seed=SEED)
    tokens = model.to_tokens(ds.prompts).cuda()
    io_ids = torch.tensor(ds.io_token_ids, device='cuda')
    s_ids = torch.tensor(ds.s_token_ids, device='cuda')

    _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    attn = cache["blocks.8.attn.hook_pattern"]
    final_pos = tokens.shape[1] - 1

    for i in range(tokens.shape[0]):
        io_tok = io_ids[i].item()
        s_tok = s_ids[i].item()
        io_pos = -1
        s1_pos = -1
        s2_pos = -1
        s_count = 0
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == io_tok and io_pos == -1:
                io_pos = j
            if tokens[i, j].item() == s_tok:
                s_count += 1
                if s_count == 1: s1_pos = j
                elif s_count == 2: s2_pos = j
        if io_pos > 0:
            syn_attn_io.append(attn[i, 9, final_pos, io_pos].item())
        if s1_pos > 0:
            syn_attn_s1.append(attn[i, 9, final_pos, s1_pos].item())
        if s2_pos > 0:
            syn_attn_s2.append(attn[i, 9, final_pos, s2_pos].item())

    del cache
    torch.cuda.empty_cache()

print("  SYNTHETIC step 1000:")
print("    L8H9: IO=%.4f S1=%.4f S2=%.4f" % (
    np.mean(syn_attn_io), np.mean(syn_attn_s1), np.mean(syn_attn_s2)))

# Pile attention
with open('data/pile_ioi_natural.json') as f:
    pile_data = json.load(f)

pile_attn_io = []
pile_attn_s1 = []
pile_attn_s2 = []
pile_count = 0

for e in pile_data[:50]:
    tokens = model.to_tokens(e['prompt']).cuda()
    if tokens.shape[1] < 3:
        continue

    io_toks = model.to_tokens(" " + e['io_name'])
    s_toks = model.to_tokens(" " + e['s_name'])
    if io_toks.shape[1] != 2 or s_toks.shape[1] != 2:
        continue

    io_tok = io_toks[0, 1].item()
    s_tok = s_toks[0, 1].item()

    try:
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        attn = cache["blocks.8.attn.hook_pattern"]
        final_pos = tokens.shape[1] - 1

        io_pos = -1
        s1_pos = -1
        s2_pos = -1
        s_count = 0
        for j in range(1, tokens.shape[1]):
            if tokens[0, j].item() == io_tok and io_pos == -1:
                io_pos = j
            if tokens[0, j].item() == s_tok:
                s_count += 1
                if s_count == 1: s1_pos = j
                elif s_count == 2: s2_pos = j

        if io_pos > 0:
            pile_attn_io.append(attn[0, 9, final_pos, io_pos].item())
        if s1_pos > 0:
            pile_attn_s1.append(attn[0, 9, final_pos, s1_pos].item())
        if s2_pos > 0:
            pile_attn_s2.append(attn[0, 9, final_pos, s2_pos].item())
        pile_count += 1

        del cache
        torch.cuda.empty_cache()
    except Exception as ex:
        continue

print("  PILE step 1000 (n=%d):" % pile_count)
if pile_attn_io:
    print("    L8H9: IO=%.4f S1=%.4f S2=%.4f" % (
        np.mean(pile_attn_io),
        np.mean(pile_attn_s1) if pile_attn_s1 else 0,
        np.mean(pile_attn_s2) if pile_attn_s2 else 0))
else:
    print("    No valid examples")

results["probe2_l8h9_pile_vs_syn"] = {
    "synthetic": {
        "attn_IO": round(float(np.mean(syn_attn_io)), 4),
        "attn_S1": round(float(np.mean(syn_attn_s1)), 4),
        "attn_S2": round(float(np.mean(syn_attn_s2)), 4),
    },
    "pile": {
        "attn_IO": round(float(np.mean(pile_attn_io)), 4) if pile_attn_io else None,
        "attn_S1": round(float(np.mean(pile_attn_s1)), 4) if pile_attn_s1 else None,
        "attn_S2": round(float(np.mean(pile_attn_s2)), 4) if pile_attn_s2 else None,
        "n_examples": pile_count,
    }
}

del model
torch.cuda.empty_cache()

###############################################################
# PROBE 3: Top predictions at step 1000
###############################################################
print("\n" + "=" * 60)
print("  PROBE 3: What does the model predict at step 1000?")
print("=" * 60)

clear_cache()
model = HookedTransformer.from_pretrained('EleutherAI/pythia-160m-deduped',
    center_writing_weights=True, center_unembed=True, fold_ln=True,
    device='cuda', checkpoint_value=1000)

io_ranks = []
s_ranks = []
io_in_top5 = 0
s_in_top5 = 0
io_is_top1 = 0
s_is_top1 = 0
total = 0

# Check if top prediction is IO, S, or neither
top_is_io = 0
top_is_s = 0
top_is_other = 0
top_other_tokens = []

for tmpl in TEMPLATES:
    ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                    symmetric=True, seed=SEED)
    tokens = model.to_tokens(ds.prompts).cuda()
    io_ids = torch.tensor(ds.io_token_ids, device='cuda')
    s_ids = torch.tensor(ds.s_token_ids, device='cuda')

    logits = model(tokens)
    last = logits[:, -1, :]

    for i in range(len(io_ids)):
        total += 1
        sorted_idx = last[i].argsort(descending=True)

        io_rank = (sorted_idx == io_ids[i]).nonzero(as_tuple=True)[0].item()
        s_rank = (sorted_idx == s_ids[i]).nonzero(as_tuple=True)[0].item()

        io_ranks.append(io_rank)
        s_ranks.append(s_rank)

        if io_rank < 5: io_in_top5 += 1
        if s_rank < 5: s_in_top5 += 1
        if io_rank == 0: io_is_top1 += 1
        if s_rank == 0: s_is_top1 += 1

        top_tok = sorted_idx[0].item()
        if top_tok == io_ids[i].item():
            top_is_io += 1
        elif top_tok == s_ids[i].item():
            top_is_s += 1
        else:
            top_is_other += 1
            decoded = model.tokenizer.decode([top_tok]).strip()
            top_other_tokens.append(decoded)

print("  Total examples: %d" % total)
print("  IO is top-1: %.1f%%" % (io_is_top1 / total * 100))
print("  S is top-1: %.1f%%" % (s_is_top1 / total * 100))
print("  Neither is top-1: %.1f%%" % (top_is_other / total * 100))
print("  IO in top-5: %.1f%%" % (io_in_top5 / total * 100))
print("  S in top-5: %.1f%%" % (s_in_top5 / total * 100))
print("  Median IO rank: %.0f" % np.median(io_ranks))
print("  Median S rank: %.0f" % np.median(s_ranks))

# Most common "other" top predictions
from collections import Counter
other_counts = Counter(top_other_tokens).most_common(10)
print("  Most common non-IO/non-S top predictions:")
for tok, count in other_counts:
    print("    '%s': %d times" % (tok, count))

results["probe3_top_predictions"] = {
    "total": total,
    "io_top1_pct": round(io_is_top1 / total, 4),
    "s_top1_pct": round(s_is_top1 / total, 4),
    "other_top1_pct": round(top_is_other / total, 4),
    "io_top5_pct": round(io_in_top5 / total, 4),
    "s_top5_pct": round(s_in_top5 / total, 4),
    "median_io_rank": float(np.median(io_ranks)),
    "median_s_rank": float(np.median(s_ranks)),
    "top_other_predictions": [{"token": t, "count": c} for t, c in other_counts],
}

del model
torch.cuda.empty_cache()

###############################################################
# SAVE
###############################################################
mega_path = "results/mega_experiments.json"
with open(mega_path) as f:
    mega = json.load(f)
mega["final_probes"] = results
with open(mega_path, 'w') as f:
    json.dump(mega, f, indent=2)
print("\nAll probes saved. DONE.")
