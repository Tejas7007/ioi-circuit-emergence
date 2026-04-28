"""
IOI Analysis on Retrained Pythia-160M (seed=42)
================================================
Runs on all 103 checkpoints. For each:
  - IOI accuracy + logit diff
  - IO/S rank and probability
  - Top-1 prediction

For key checkpoints (every 200 in dip zone + every 1000 after):
  - Find top head by ablation
  - Attention pattern of key heads
  - Output projection of dominant head

Saves incrementally to results/retrain_ioi_analysis.json
"""

import os
os.environ["HF_TOKEN"] = ""

import torch, json, time, traceback
import numpy as np
from collections import Counter
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
from transformer_lens import HookedTransformer
import sys

# Try importing IOIDataset
for path in ['/workspace/MLP-Paper-Cole/src', '/workspace/ioi-circuit-emergence/src',
             os.path.expanduser('~/MLP-Paper-Cole/src')]:
    sys.path.insert(0, path)
try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
    print("IOIDataset loaded")
except ImportError:
    print("ERROR: Cannot import IOIDataset")
    print("Run: cd /workspace && git clone https://github.com/Tejas7007/MLP-Paper-Cole.git")
    raise

DEVICE = "cuda"
CHECKPOINT_DIR = "/workspace/pythia-160m-retrain/checkpoints"
RESULTS_FILE = "/workspace/pythia-160m-retrain/ioi_analysis.json"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"

TEMPLATES = ALL_TEMPLATES[:15]
PPT = 20
SEED = 42

# Get all checkpoint steps
all_steps = sorted([int(d.split('_')[1]) for d in os.listdir(CHECKPOINT_DIR) if d.startswith('step_')])
print("Found %d checkpoints: %s ... %s" % (len(all_steps), all_steps[:5], all_steps[-3:]))

# Deep analysis at these steps
DEEP_STEPS = set(list(range(0, 3001, 200)) + list(range(3000, 10001, 1000)))

def empty_cache():
    torch.cuda.empty_cache()

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_retrained_model(step):
    """Load retrained checkpoint into TransformerLens."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "step_%d" % step)
    hf_model = GPTNeoXForCausalLM.from_pretrained(ckpt_path)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        hf_model=hf_model,
        device=DEVICE,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
    )
    del hf_model
    return model

def get_positions(tokens, io_ids, s_ids):
    positions = []
    for i in range(tokens.shape[0]):
        io_tok = io_ids[i].item()
        s_tok = s_ids[i].item()
        io_pos, s1_pos, s2_pos = -1, -1, -1
        s_count = 0
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == io_tok and io_pos == -1:
                io_pos = j
            if tokens[i, j].item() == s_tok:
                s_count += 1
                if s_count == 1: s1_pos = j
                elif s_count == 2: s2_pos = j
        positions.append((io_pos, s1_pos, s2_pos))
    return positions


def run_fast_analysis(model, step):
    """IOI accuracy, logit diff, ranks, probabilities, top-1."""
    all_lds = []
    io_ranks, s_ranks = [], []
    io_probs, s_probs = [], []
    top1_io, top1_s = 0, 0
    top1_other = []
    total = 0

    for tmpl in TEMPLATES:
        try:
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
                ld = last[i, io_ids[i]].item() - last[i, s_ids[i]].item()
                all_lds.append(ld)

                sorted_idx = last[i].argsort(descending=True)
                io_ranks.append((sorted_idx == io_ids[i]).nonzero(as_tuple=True)[0].item())
                s_ranks.append((sorted_idx == s_ids[i]).nonzero(as_tuple=True)[0].item())
                io_probs.append(probs[i, io_ids[i]].item())
                s_probs.append(probs[i, s_ids[i]].item())

                top_tok = sorted_idx[0].item()
                if top_tok == io_ids[i].item():
                    top1_io += 1
                elif top_tok == s_ids[i].item():
                    top1_s += 1
                else:
                    top1_other.append(model.tokenizer.decode([top_tok]).strip())
        except Exception as e:
            continue

    if total == 0:
        return None

    lds = np.array(all_lds)
    top_others = Counter(top1_other).most_common(3)

    return {
        "accuracy": round(float((lds > 0).mean()), 4),
        "mean_ld": round(float(lds.mean()), 4),
        "std_ld": round(float(lds.std()), 4),
        "median_io_rank": float(np.median(io_ranks)),
        "median_s_rank": float(np.median(s_ranks)),
        "mean_io_prob": round(float(np.mean(io_probs)), 6),
        "mean_s_prob": round(float(np.mean(s_probs)), 6),
        "pct_io_top1": round(top1_io / total, 4),
        "pct_s_top1": round(top1_s / total, 4),
        "top_other": [{"token": t, "count": c} for t, c in top_others],
        "n_examples": total,
    }


def run_deep_analysis(model, step):
    """Ablate all heads, find top head, get attention patterns."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Get baseline
    template_data = []
    base_lds = []
    for tmpl in TEMPLATES[:10]:
        try:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                            symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).to(DEVICE)
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
            logits = model(tokens)
            last = logits[:, -1, :]
            for i in range(len(io_ids)):
                base_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())
            template_data.append((tokens, io_ids, s_ids))
        except:
            continue

    if not template_data:
        return None

    base_mean_ld = float(np.mean(base_lds))

    # Ablate each head
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

    sorted_heads = sorted(head_deltas.items(), key=lambda x: x[1])
    top_head = sorted_heads[0][0]
    top_layer = int(top_head.split("H")[0][1:])
    top_head_idx = int(top_head.split("H")[1])

    # Get attention patterns of top head + key known heads
    heads_to_check = set()
    heads_to_check.add((top_layer, top_head_idx))
    # Always check layers 0, 1, 8 heads that might be interesting
    for l, h in [(0, 10), (1, 8), (8, 9)]:
        if l < n_layers and h < n_heads:
            heads_to_check.add((l, h))

    attention_results = {}
    for layer, head in heads_to_check:
        attn_io, attn_s1, attn_s2, attn_self = [], [], [], []
        for tokens, io_ids, s_ids in template_data:
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
            attn = cache["blocks.%d.attn.hook_pattern" % layer]
            final_pos = tokens.shape[1] - 1
            positions = get_positions(tokens, io_ids, s_ids)
            for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                if io_pos > 0:
                    attn_io.append(attn[i, head, final_pos, io_pos].item())
                if s1_pos > 0:
                    attn_s1.append(attn[i, head, final_pos, s1_pos].item())
                if s2_pos > 0:
                    attn_s2.append(attn[i, head, final_pos, s2_pos].item())
                attn_self.append(attn[i, head, final_pos, final_pos].item())
            del cache
            empty_cache()

        head_name = "L%dH%d" % (layer, head)
        attention_results[head_name] = {
            "attn_IO": round(float(np.mean(attn_io)), 4) if attn_io else 0,
            "attn_S1": round(float(np.mean(attn_s1)), 4) if attn_s1 else 0,
            "attn_S2": round(float(np.mean(attn_s2)), 4) if attn_s2 else 0,
            "attn_self": round(float(np.mean(attn_self)), 4) if attn_self else 0,
        }

    return {
        "base_mean_ld": round(base_mean_ld, 4),
        "top_head": top_head,
        "top_head_delta": sorted_heads[0][1],
        "top5_heads": [{"head": h, "delta": d} for h, d in sorted_heads[:5]],
        "attention": attention_results,
    }


def main():
    print("=" * 60)
    print("  IOI ANALYSIS ON RETRAINED PYTHIA-160M")
    print("  Checkpoints: %d" % len(all_steps))
    print("  Deep analysis at: %d steps" % len([s for s in all_steps if s in DEEP_STEPS]))
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {"model": "retrained-pythia-160m-seed42", "checkpoints": {}}

    t0 = time.time()

    for step in all_steps:
        step_key = "step_%d" % step

        if step_key in results["checkpoints"] and "accuracy" in results["checkpoints"][step_key]:
            # Check if deep analysis needed and missing
            if step in DEEP_STEPS and "deep" not in results["checkpoints"][step_key]:
                pass  # Will do deep analysis below
            else:
                print("  Step %d done, skip" % step)
                continue

        print("\n--- Step %d ---" % step)
        try:
            model = load_retrained_model(step)
        except Exception as e:
            print("  FAILED to load: %s" % str(e)[:100])
            continue

        # Fast analysis (always)
        if step_key not in results["checkpoints"] or "accuracy" not in results["checkpoints"][step_key]:
            fast = run_fast_analysis(model, step)
            if fast:
                results["checkpoints"][step_key] = fast
                print("  acc=%.3f, LD=%.4f, IO_rank=%d, IO_prob=%.4f%%" % (
                    fast["accuracy"], fast["mean_ld"],
                    fast["median_io_rank"], fast["mean_io_prob"] * 100))
            else:
                print("  No valid examples")
                del model
                empty_cache()
                continue

        # Deep analysis (selected steps)
        if step in DEEP_STEPS:
            print("  Running deep analysis...")
            try:
                deep = run_deep_analysis(model, step)
                if deep:
                    results["checkpoints"][step_key]["deep"] = deep
                    print("  Top head: %s (delta=%.4f)" % (deep["top_head"], deep["top_head_delta"]))
                    for h, a in deep["attention"].items():
                        print("    %s: IO=%.3f S1=%.3f S2=%.3f" % (
                            h, a["attn_IO"], a["attn_S1"], a["attn_S2"]))
            except Exception as e:
                print("  Deep analysis failed: %s" % str(e)[:80])

        save_results(results)
        del model
        empty_cache()

    # Print summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("  Time: %.1f hours" % (elapsed / 3600))
    print("=" * 60)

    print("\n  Step  |  Acc   |  LD     | IO_rank | Top Head")
    print("  " + "-" * 55)
    for step in all_steps:
        sk = "step_%d" % step
        if sk in results["checkpoints"]:
            r = results["checkpoints"][sk]
            top = r.get("deep", {}).get("top_head", "")
            print("  %5d | %5.1f%% | %+7.4f | %7d | %s" % (
                step, r["accuracy"] * 100, r["mean_ld"],
                r["median_io_rank"], top))

    save_results(results)
    print("\nSaved to %s" % RESULTS_FILE)


if __name__ == "__main__":
    main()
