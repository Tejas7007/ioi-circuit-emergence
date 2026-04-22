import os
os.environ["HF_TOKEN"] = ""
import torch, json, numpy as np, sys
sys.path.insert(0, '/workspace/MLP-Paper-Cole/src')
from transformer_lens import HookedTransformer
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = "cuda"
TEMPLATES = ALL_TEMPLATES[:15]
results = {}

###############################################################
# 1. Volatile steps with n=600
###############################################################
print("=" * 60)
print("  TEST 1: Volatile steps with n=600 (double examples)")
print("=" * 60)

volatile_steps = [1600, 1650, 3300, 3700, 3800, 4100, 4300]
results["volatile_retest"] = {}

for step in volatile_steps:
    print("\n  Step %d:" % step)
    try:
        model = HookedTransformer.from_pretrained(
            "stanford-crfm/alias-gpt2-small-x21",
            device=DEVICE, revision="checkpoint-%d" % step)
    except:
        print("    SKIP")
        continue

    all_lds = []
    # Use ALL 15 templates x 40 prompts = 600 examples
    for tmpl in TEMPLATES:
        ds = IOIDataset(model=model, n_prompts=40, templates=[tmpl],
                        symmetric=True, seed=42)
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
        logits = model(tokens)
        last = logits[:, -1, :].float()
        for i in range(len(io_ids)):
            all_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())

    lds = np.array(all_lds)
    acc = float((lds > 0).mean())
    results["volatile_retest"]["step_%d" % step] = {
        "accuracy_n600": round(acc, 4),
        "n": len(lds),
        "std_err": round(float(np.sqrt(acc * (1-acc) / len(lds))), 4),
    }
    print("    n=%d, acc=%.3f (SE=%.3f)" % (len(lds), acc,
        np.sqrt(acc * (1-acc) / len(lds))))

    del model
    torch.cuda.empty_cache()

###############################################################
# 2. Stanford L11H11 (actual name mover) projection
###############################################################
print("\n" + "=" * 60)
print("  TEST 2: Stanford L11H11 + L10H10 Output Projection")
print("=" * 60)

model = HookedTransformer.from_pretrained(
    "stanford-crfm/alias-gpt2-small-x21",
    device=DEVICE, revision="checkpoint-400000")

W_U = model.W_U
W_O = model.W_O

heads = [(11, 11), (10, 10), (10, 4), (10, 7)]
results["stanford_projections"] = {}

for layer, head in heads:
    attn_io, attn_s2 = [], []
    io_projs, s_projs = [], []

    for tmpl in TEMPLATES[:10]:
        ds = IOIDataset(model=model, n_prompts=20, templates=[tmpl],
                        symmetric=True, seed=42)
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        attn = cache["blocks.%d.attn.hook_pattern" % layer]
        final_pos = tokens.shape[1] - 1

        for i in range(tokens.shape[0]):
            io_tok = io_ids[i].item()
            s_tok = s_ids[i].item()
            io_pos, s2_pos = -1, -1
            s_count = 0
            for j in range(1, tokens.shape[1]):
                if tokens[i, j].item() == io_tok and io_pos == -1:
                    io_pos = j
                if tokens[i, j].item() == s_tok:
                    s_count += 1
                    if s_count == 2: s2_pos = j
            if io_pos > 0:
                attn_io.append(attn[i, head, final_pos, io_pos].item())
            if s2_pos > 0:
                attn_s2.append(attn[i, head, final_pos, s2_pos].item())

        z = cache["blocks.%d.attn.hook_z" % layer][:, -1, head, :]
        head_out = z @ W_O[layer, head]
        for i in range(len(io_ids)):
            io_projs.append(torch.dot(head_out[i], W_U[:, io_ids[i].item()]).item())
            s_projs.append(torch.dot(head_out[i], W_U[:, s_ids[i].item()]).item())

        del cache
        torch.cuda.empty_cache()

    head_name = "L%dH%d" % (layer, head)
    mean_io = float(np.mean(io_projs))
    mean_s = float(np.mean(s_projs))
    diff = mean_io - mean_s

    if float(np.mean(attn_io)) > float(np.mean(attn_s2)) and diff > 0:
        role = "IO-copier (name mover)"
    elif float(np.mean(attn_s2)) > float(np.mean(attn_io)) and diff > 0:
        role = "S-suppressor"
    else:
        role = "other"

    results["stanford_projections"][head_name] = {
        "attn_IO": round(float(np.mean(attn_io)), 4),
        "attn_S2": round(float(np.mean(attn_s2)), 4),
        "proj_IO": round(mean_io, 4),
        "proj_S": round(mean_s, 4),
        "proj_diff": round(diff, 4),
        "role": role,
    }
    print("  %s: attn IO=%.3f S2=%.3f | proj IO=%.3f S=%.3f diff=%.3f | %s" % (
        head_name, np.mean(attn_io), np.mean(attn_s2),
        mean_io, mean_s, diff, role))

# Compute ratio
l10h10_diff = results["stanford_projections"]["L10H10"]["proj_diff"]
l11h11_diff = results["stanford_projections"]["L11H11"]["proj_diff"]
if l11h11_diff != 0:
    ratio = abs(l10h10_diff / l11h11_diff)
    print("\n  S-suppression / IO-copying ratio: %.1f:1" % ratio)
    results["stanford_projections"]["suppression_to_copying_ratio"] = round(ratio, 2)

del model
torch.cuda.empty_cache()

with open("results/two_more.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE. Saved to results/two_more.json")
