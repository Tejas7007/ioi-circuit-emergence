"""
PolyPythias Mechanism Analysis
Find dominant head + mechanism for seeds 1, 3, 5 at step 143000.
Tests whether different seeds produce different circuits (n=5 total with original + retrained).
"""
import os
os.environ["HF_TOKEN"] = ""
import torch, json, time, numpy as np, sys
sys.path.insert(0, '/workspace/MLP-Paper-Cole/src')
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = "cuda"
MODEL_BASE = "EleutherAI/pythia-160m-deduped"
TEMPLATES = ALL_TEMPLATES[:15]
PPT, SEED = 20, 42
RESULTS_FILE = "results/polypythias_mechanism.json"

SEEDS_TO_TEST = [
    ("EleutherAI/pythia-160m-seed1", "seed1"),
    ("EleutherAI/pythia-160m-seed3", "seed3"),
    ("EleutherAI/pythia-160m-seed5", "seed5"),
]

def get_positions(tokens, io_ids, s_ids):
    positions = []
    for i in range(tokens.shape[0]):
        io_tok, s_tok = io_ids[i].item(), s_ids[i].item()
        io_pos, s1_pos, s2_pos = -1, -1, -1
        s_count = 0
        for j in range(1, tokens.shape[1]):
            if tokens[i,j].item() == io_tok and io_pos == -1: io_pos = j
            if tokens[i,j].item() == s_tok:
                s_count += 1
                if s_count == 1: s1_pos = j
                elif s_count == 2: s2_pos = j
        positions.append((io_pos, s1_pos, s2_pos))
    return positions

def analyze_seed(model_name, label):
    print("\n" + "=" * 60)
    print("  %s (%s) at step 143000" % (label, model_name))
    print("=" * 60)

    # Load via HF then into TransformerLens
    print("  Loading...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, revision="step143000")
    model = HookedTransformer.from_pretrained(
        MODEL_BASE, hf_model=hf_model, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf_model

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U
    W_O = model.W_O

    # Get template data
    template_data = []
    base_lds = []
    for tmpl in TEMPLATES[:10]:
        ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl], symmetric=True, seed=SEED)
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
        logits = model(tokens)
        last = logits[:, -1, :]
        for i in range(len(io_ids)):
            base_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())
        template_data.append((tokens, io_ids, s_ids))

    base_mean_ld = float(np.mean(base_lds))
    base_acc = float((np.array(base_lds) > 0).mean())
    print("  Baseline: acc=%.1f%%, mean_LD=%.4f" % (base_acc * 100, base_mean_ld))

    # Ablate each head
    print("  Ablating all %d heads..." % (n_layers * n_heads))
    head_deltas = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            def hook_fn(value, hook, h=head):
                value[:, :, h, :] = 0.0
                return value
            hook = ("blocks.%d.attn.hook_z" % layer, hook_fn)
            abl_lds = []
            for tokens, io_ids, s_ids in template_data:
                logits = model.run_with_hooks(tokens, fwd_hooks=[hook])
                last = logits[:, -1, :]
                for i in range(len(io_ids)):
                    abl_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())
            delta = float(np.mean(abl_lds)) - base_mean_ld
            head_deltas["L%dH%d" % (layer, head)] = round(delta, 4)
        if (layer + 1) % 4 == 0:
            print("    Layer %d/%d" % (layer + 1, n_layers))

    sorted_heads = sorted(head_deltas.items(), key=lambda x: x[1])
    top_head = sorted_heads[0][0]
    top_layer = int(top_head.split("H")[0][1:])
    top_head_idx = int(top_head.split("H")[1])

    print("  Top 5 heads:")
    for h, d in sorted_heads[:5]:
        print("    %s: %+.4f" % (h, d))

    # Attention patterns of top head
    attn_io, attn_s1, attn_s2, attn_self = [], [], [], []
    io_projs, s_projs = [], []

    for tokens, io_ids, s_ids in template_data:
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        attn = cache["blocks.%d.attn.hook_pattern" % top_layer]
        final_pos = tokens.shape[1] - 1
        positions = get_positions(tokens, io_ids, s_ids)

        for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
            if io_pos > 0: attn_io.append(attn[i, top_head_idx, final_pos, io_pos].item())
            if s1_pos > 0: attn_s1.append(attn[i, top_head_idx, final_pos, s1_pos].item())
            if s2_pos > 0: attn_s2.append(attn[i, top_head_idx, final_pos, s2_pos].item())
            attn_self.append(attn[i, top_head_idx, final_pos, final_pos].item())

        # Output projection
        z = cache["blocks.%d.attn.hook_z" % top_layer][:, -1, top_head_idx, :]
        head_out = z @ W_O[top_layer, top_head_idx]
        for i in range(len(io_ids)):
            io_dir = W_U[:, io_ids[i].item()]
            s_dir = W_U[:, s_ids[i].item()]
            io_projs.append(torch.dot(head_out[i], io_dir).item())
            s_projs.append(torch.dot(head_out[i], s_dir).item())

        del cache
        torch.cuda.empty_cache()

    mean_io_proj = float(np.mean(io_projs))
    mean_s_proj = float(np.mean(s_projs))
    proj_diff = mean_io_proj - mean_s_proj

    # Classify mechanism
    mean_attn_s2 = float(np.mean(attn_s2))
    mean_attn_io = float(np.mean(attn_io))
    mean_attn_self = float(np.mean(attn_self))

    if mean_attn_s2 > 0.3 and proj_diff > 0.1:
        mechanism = "Direct S2-suppression"
    elif mean_attn_io > 0.3 and proj_diff > 0.1:
        mechanism = "IO-copying (name mover)"
    elif mean_attn_self > 0.3 and abs(proj_diff) < 0.05:
        mechanism = "Indirect relay (self-attending)"
    elif abs(proj_diff) < 0.05:
        mechanism = "Indirect relay (low projection)"
    else:
        mechanism = "Other"

    result = {
        "baseline_accuracy": round(base_acc, 4),
        "baseline_mean_ld": round(base_mean_ld, 4),
        "top_head": top_head,
        "top_head_delta": sorted_heads[0][1],
        "top_head_layer_depth": round(top_layer / n_layers, 2),
        "top5_heads": [{"head": h, "delta": d} for h, d in sorted_heads[:5]],
        "attention": {
            "to_IO": round(mean_attn_io, 4),
            "to_S1": round(float(np.mean(attn_s1)), 4),
            "to_S2": round(mean_attn_s2, 4),
            "to_self": round(mean_attn_self, 4),
        },
        "projection": {
            "IO": round(mean_io_proj, 4),
            "S": round(mean_s_proj, 4),
            "diff": round(proj_diff, 4),
        },
        "mechanism": mechanism,
    }

    print("\n  %s: attn IO=%.3f S2=%.3f self=%.3f | proj diff=%.3f | %s" % (
        top_head, mean_attn_io, mean_attn_s2, mean_attn_self, proj_diff, mechanism))

    del model
    torch.cuda.empty_cache()
    return result


def main():
    print("=" * 60)
    print("  POLYPYTHIAS MECHANISM ANALYSIS")
    print("  Testing seeds 1, 3, 5 at step 143000")
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()

    for model_name, label in SEEDS_TO_TEST:
        if label in results:
            print("  %s already done, skip" % label)
            continue
        try:
            results[label] = analyze_seed(model_name, label)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print("  FAILED: %s" % str(e))

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY: Circuit Degeneracy Across Seeds")
    print("=" * 60)
    print("  %-12s  %-8s  %-6s  %-8s  %-8s  %-8s  %s" % (
        "Seed", "Head", "Depth", "S2 attn", "Proj", "Acc", "Mechanism"))
    print("  " + "-" * 75)

    # Add known results for comparison
    known = {
        "original(1234)": {"top_head": "L8H9", "depth": "67%", "s2": "92.5%", "proj": "+5.74", "acc": "100%", "mech": "Direct S2-suppression"},
        "retrained(42)": {"top_head": "L2H6", "depth": "17%", "s2": "0.3%", "proj": "~0", "acc": "87.7%", "mech": "Indirect relay"},
    }
    for label, k in known.items():
        print("  %-12s  %-8s  %-6s  %-8s  %-8s  %-8s  %s" % (
            label, k["top_head"], k["depth"], k["s2"], k["proj"], k["acc"], k["mech"]))

    for label in ["seed1", "seed3", "seed5"]:
        if label in results:
            r = results[label]
            print("  %-12s  %-8s  %-6s  %-8s  %-8s  %-8s  %s" % (
                label, r["top_head"],
                "%.0f%%" % (r["top_head_layer_depth"] * 100),
                "%.1f%%" % (r["attention"]["to_S2"] * 100),
                "%+.2f" % r["projection"]["diff"],
                "%.1f%%" % (r["baseline_accuracy"] * 100),
                r["mechanism"]))

    elapsed = time.time() - t0
    print("\n  Time: %.1f hours" % (elapsed / 3600))
    print("  DONE.")


if __name__ == "__main__":
    main()
