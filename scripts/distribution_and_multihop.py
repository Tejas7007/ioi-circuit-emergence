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

#############################################################
# PART 1: Distribution of L8H9 attention (not just mean)
#############################################################
print("=" * 60)
print("  PART 1: L8H9 Attention Distribution (160M, step 143000)")
print("=" * 60)

clear_cache()
model = HookedTransformer.from_pretrained('EleutherAI/pythia-160m-deduped',
    center_writing_weights=True, center_unembed=True, fold_ln=True,
    device='cuda', checkpoint_value=143000)

all_attn_to_s2 = []
all_attn_to_io = []

for tmpl in TEMPLATES:
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
        s2_pos = -1
        s_count = 0
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == io_tok and io_pos == -1:
                io_pos = j
            if tokens[i, j].item() == s_tok:
                s_count += 1
                if s_count == 2: s2_pos = j
        if s2_pos > 0:
            all_attn_to_s2.append(attn[i, 9, final_pos, s2_pos].item())
        if io_pos > 0:
            all_attn_to_io.append(attn[i, 9, final_pos, io_pos].item())

    del cache
    torch.cuda.empty_cache()

s2_arr = np.array(all_attn_to_s2)
io_arr = np.array(all_attn_to_io)

print("  L8H9 attention to S2:")
print("    Mean: %.4f" % s2_arr.mean())
print("    Std: %.4f" % s2_arr.std())
print("    Min: %.4f, Max: %.4f" % (s2_arr.min(), s2_arr.max()))
print("    Percentiles: p10=%.3f, p25=%.3f, p50=%.3f, p75=%.3f, p90=%.3f" % (
    np.percentile(s2_arr, 10), np.percentile(s2_arr, 25),
    np.percentile(s2_arr, 50), np.percentile(s2_arr, 75),
    np.percentile(s2_arr, 90)))
print("    %% with >80%% attention to S2: %.1f%%" % ((s2_arr > 0.8).mean() * 100))
print("    %% with <50%% attention to S2: %.1f%%" % ((s2_arr < 0.5).mean() * 100))

results["attention_distribution"] = {
    "n_examples": len(s2_arr),
    "l8h9_s2_mean": float(s2_arr.mean()),
    "l8h9_s2_std": float(s2_arr.std()),
    "l8h9_s2_min": float(s2_arr.min()),
    "l8h9_s2_max": float(s2_arr.max()),
    "l8h9_s2_p10": float(np.percentile(s2_arr, 10)),
    "l8h9_s2_p25": float(np.percentile(s2_arr, 25)),
    "l8h9_s2_p50": float(np.percentile(s2_arr, 50)),
    "l8h9_s2_p75": float(np.percentile(s2_arr, 75)),
    "l8h9_s2_p90": float(np.percentile(s2_arr, 90)),
    "pct_above_80": float((s2_arr > 0.8).mean()),
    "pct_below_50": float((s2_arr < 0.5).mean()),
    "raw_values": s2_arr.tolist()[:100],
}

#############################################################
# PART 2: Multi-hop - what writes TO S2 position?
#############################################################
print("\n" + "=" * 60)
print("  PART 2: Multi-hop — what attends TO S2 from earlier?")
print("=" * 60)

# For each earlier layer head, check its attention FROM S2 back to earlier positions
# If a head strongly attends from S2 to S1, it's a duplicate-token head
# That tells us what info has been placed at S2 before L8H9 reads it

# Check all heads in layers 0-7 (before L8H9)
duptoken_heads = []  # heads that attend S2 -> S1

for layer in range(8):  # layers 0-7
    for head in range(model.cfg.n_heads):
        s2_to_s1_scores = []

        for tmpl in TEMPLATES[:10]:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                            symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).cuda()
            io_ids = torch.tensor(ds.io_token_ids, device='cuda')
            s_ids = torch.tensor(ds.s_token_ids, device='cuda')

            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
            attn = cache["blocks.%d.attn.hook_pattern" % layer]

            for i in range(tokens.shape[0]):
                s_tok = s_ids[i].item()
                s1_pos = -1
                s2_pos = -1
                s_count = 0
                for j in range(1, tokens.shape[1]):
                    if tokens[i, j].item() == s_tok:
                        s_count += 1
                        if s_count == 1: s1_pos = j
                        elif s_count == 2: s2_pos = j
                if s1_pos > 0 and s2_pos > 0:
                    # Attention from S2 position back to S1 position
                    s2_to_s1_scores.append(attn[i, head, s2_pos, s1_pos].item())

            del cache
            torch.cuda.empty_cache()

        if s2_to_s1_scores:
            mean_score = float(np.mean(s2_to_s1_scores))
            if mean_score > 0.1:
                duptoken_heads.append({
                    "head": "L%dH%d" % (layer, head),
                    "s2_to_s1_attn": round(mean_score, 4),
                })

# Sort by score
duptoken_heads.sort(key=lambda x: -x["s2_to_s1_attn"])

print("  Heads with high S2 -> S1 attention (duplicate-token candidates):")
for h in duptoken_heads[:10]:
    print("    %s: %.4f" % (h["head"], h["s2_to_s1_attn"]))

if not duptoken_heads:
    print("    No heads found with S2->S1 attention > 0.1")

results["multihop_duptoken_candidates"] = duptoken_heads

# Save
mega_path = "results/mega_experiments.json"
with open(mega_path) as f:
    mega = json.load(f)
mega["distribution_and_multihop"] = results
with open(mega_path, 'w') as f:
    json.dump(mega, f, indent=2)

print("\nDONE. Saved to results/mega_experiments.json")

del model
torch.cuda.empty_cache()
