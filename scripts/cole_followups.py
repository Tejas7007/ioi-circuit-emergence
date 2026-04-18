import torch, json, os
import numpy as np
from transformer_lens import HookedTransformer
import sys
sys.path.insert(0, os.path.expanduser('~/MLP-Paper-Cole/src'))
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = 'mps'
TEMPLATES = ALL_TEMPLATES[:15]
PPT, SEED = 20, 42
results = {}

print("=" * 60)
print("  Q1: IO/S ranks + probabilities across training")
print("=" * 60)

results["rank_progression"] = {}

for step in [1000, 2000, 3000, 5000, 8000, 143000]:
    print("\n--- Step %d ---" % step)
    model = HookedTransformer.from_pretrained('EleutherAI/pythia-160m-deduped',
        center_writing_weights=True, center_unembed=True, fold_ln=True,
        device=DEVICE, checkpoint_value=step)

    io_ranks = []
    s_ranks = []
    io_probs = []
    s_probs = []
    top1_is_io = 0
    top1_is_s = 0
    top1_other_tokens = []
    total = 0

    for tmpl in TEMPLATES:
        ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                        symmetric=True, seed=SEED)
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

        logits = model(tokens)
        last = logits[:, -1, :].float()
        probs = torch.softmax(last, dim=-1)

        for i in range(len(io_ids)):
            total += 1
            sorted_idx = last[i].argsort(descending=True)

            io_rank = (sorted_idx == io_ids[i]).nonzero(as_tuple=True)[0].item()
            s_rank = (sorted_idx == s_ids[i]).nonzero(as_tuple=True)[0].item()
            io_ranks.append(io_rank)
            s_ranks.append(s_rank)

            io_prob = probs[i, io_ids[i]].item()
            s_prob = probs[i, s_ids[i]].item()
            io_probs.append(io_prob)
            s_probs.append(s_prob)

            top_tok = sorted_idx[0].item()
            if top_tok == io_ids[i].item():
                top1_is_io += 1
            elif top_tok == s_ids[i].item():
                top1_is_s += 1
            else:
                decoded = model.tokenizer.decode([top_tok]).strip()
                top1_other_tokens.append(decoded)

    from collections import Counter
    top_others = Counter(top1_other_tokens).most_common(5)

    step_result = {
        "median_io_rank": float(np.median(io_ranks)),
        "median_s_rank": float(np.median(s_ranks)),
        "mean_io_prob": round(float(np.mean(io_probs)), 6),
        "mean_s_prob": round(float(np.mean(s_probs)), 6),
        "max_io_prob": round(float(np.max(io_probs)), 6),
        "pct_io_top1": round(top1_is_io / total, 4),
        "pct_s_top1": round(top1_is_s / total, 4),
        "pct_other_top1": round(1 - top1_is_io/total - top1_is_s/total, 4),
        "top_other": [{"token": t, "count": c} for t, c in top_others],
    }
    results["rank_progression"]["step_%d" % step] = step_result

    print("  IO: median_rank=%d, mean_prob=%.4f%%" % (
        np.median(io_ranks), np.mean(io_probs) * 100))
    print("  S:  median_rank=%d, mean_prob=%.4f%%" % (
        np.median(s_ranks), np.mean(s_probs) * 100))
    print("  Top-1: IO=%.1f%%, S=%.1f%%, other=%.1f%%" % (
        top1_is_io/total*100, top1_is_s/total*100,
        (1-top1_is_io/total-top1_is_s/total)*100))
    if top_others:
        print("  Top other: %s" % str(top_others[:3]))

    del model
    torch.mps.empty_cache()

print("\n" + "=" * 60)
print("  Q2: Self-attention analysis in 410M")
print("=" * 60)

model = HookedTransformer.from_pretrained('EleutherAI/pythia-410m-deduped',
    center_writing_weights=True, center_unembed=True, fold_ln=True,
    device=DEVICE, checkpoint_value=143000)

self_attn_scores = []
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

for tmpl in TEMPLATES[:5]:
    ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                    symmetric=True, seed=SEED)
    tokens = model.to_tokens(ds.prompts).to(DEVICE)
    final_pos = tokens.shape[1] - 1

    _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

    for layer in range(n_layers):
        attn = cache["blocks.%d.attn.hook_pattern" % layer]
        for head in range(n_heads):
            self_attn = attn[:, head, final_pos, final_pos].mean().item()

            idx = layer * n_heads + head
            if len(self_attn_scores) <= idx:
                self_attn_scores.append({
                    "head": "L%dH%d" % (layer, head),
                    "self_attn": [],
                })
            self_attn_scores[idx]["self_attn"].append(self_attn)

    del cache
    torch.mps.empty_cache()

for entry in self_attn_scores:
    entry["mean_self_attn"] = round(float(np.mean(entry["self_attn"])), 4)
    del entry["self_attn"]

self_attn_sorted = sorted(self_attn_scores, key=lambda x: -x["mean_self_attn"])

print("  Top 15 heads by self-attention (final -> final):")
for h in self_attn_sorted[:15]:
    marker = " <-- DOMINANT" if h["head"] == "L4H6" else ""
    print("    %s: %.4f%s" % (h["head"], h["mean_self_attn"], marker))

results["self_attention_410m"] = {
    "top_15": self_attn_sorted[:15],
    "l4h6_rank": next(i for i, h in enumerate(self_attn_sorted) if h["head"] == "L4H6") + 1,
}

print("\n  L4H6 rank among all heads by self-attention: %d / %d" % (
    results["self_attention_410m"]["l4h6_rank"], len(self_attn_sorted)))

del model
torch.mps.empty_cache()

with open("results/cole_followups.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/cole_followups.json. DONE.")
