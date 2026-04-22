"""
Three polish experiments:
  1. Verify L9H1 (actual name mover) has IO-copying mechanism
  2. Stanford GPT-2 L10H10 Wang et al. classification
  3. High-resolution phase transition curve (Stanford 609 checkpoints)
"""
import os
os.environ["HF_TOKEN"] = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""

import torch, json, time, shutil, traceback
import numpy as np
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
import sys
sys.path.insert(0, '/workspace/MLP-Paper-Cole/src')
sys.path.insert(0, os.path.expanduser('~/MLP-Paper-Cole/src'))
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TEMPLATES = ALL_TEMPLATES[:15]
PPT, SEED = 20, 42
RESULTS_FILE = "results/polish_experiments.json"

def empty_cache():
    if DEVICE == "cuda": torch.cuda.empty_cache()
    elif DEVICE == "mps": torch.mps.empty_cache()

def clear_cache():
    for d in ['/workspace/.hf_home/hub', os.path.expanduser('~/.cache/huggingface/hub')]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith('models--'):
                    shutil.rmtree(os.path.join(d, f), ignore_errors=True)

def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_pythia(step):
    hf_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m-deduped", revision="step%d" % step)
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-160m-deduped", hf_model=hf_model, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
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


# ============================================================
# EXPERIMENT 1: L9H1 mechanism verification
# ============================================================
def run_exp1(results):
    print("\n" + "=" * 60)
    print("  EXP 1: L9H1 Mechanism (the actual name mover?)")
    print("=" * 60)

    results["exp1_l9h1"] = {}

    for step in [3000, 143000]:
        print("\n  --- Step %d ---" % step)
        clear_cache()
        model = load_pythia(step)

        # Heads to compare: L9H1 (top NM by Wang), L8H9 (top by ablation), L8H1
        heads_to_check = [(9, 1), (8, 9), (8, 1)]

        step_results = {}
        W_U = model.W_U
        W_O = model.W_O

        for layer, head in heads_to_check:
            attn_io, attn_s1, attn_s2 = [], [], []
            io_projs, s_projs = [], []

            for tmpl in TEMPLATES[:10]:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                attn = cache["blocks.%d.attn.hook_pattern" % layer]
                final_pos = tokens.shape[1] - 1
                positions = get_positions(tokens, io_ids, s_ids)

                # Attention
                for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                    if io_pos > 0:
                        attn_io.append(attn[i, head, final_pos, io_pos].item())
                    if s1_pos > 0:
                        attn_s1.append(attn[i, head, final_pos, s1_pos].item())
                    if s2_pos > 0:
                        attn_s2.append(attn[i, head, final_pos, s2_pos].item())

                # Output projection
                z = cache["blocks.%d.attn.hook_z" % layer][:, -1, head, :]
                head_out = z @ W_O[layer, head]
                for i in range(len(io_ids)):
                    io_dir = W_U[:, io_ids[i].item()]
                    s_dir = W_U[:, s_ids[i].item()]
                    io_projs.append(torch.dot(head_out[i], io_dir).item())
                    s_projs.append(torch.dot(head_out[i], s_dir).item())

                del cache
                empty_cache()

            head_name = "L%dH%d" % (layer, head)
            step_results[head_name] = {
                "attn_IO": round(float(np.mean(attn_io)), 4),
                "attn_S1": round(float(np.mean(attn_s1)), 4),
                "attn_S2": round(float(np.mean(attn_s2)), 4),
                "proj_IO": round(float(np.mean(io_projs)), 4),
                "proj_S": round(float(np.mean(s_projs)), 4),
                "proj_diff": round(float(np.mean(io_projs)) - float(np.mean(s_projs)), 4),
                "role": "",
            }

            r = step_results[head_name]
            # Classify
            if r["attn_IO"] > r["attn_S2"] and r["proj_diff"] > 0:
                r["role"] = "IO-copier (name mover)"
            elif r["attn_S2"] > r["attn_IO"] and r["proj_diff"] > 0:
                r["role"] = "S-suppressor (S-inhibition)"
            else:
                r["role"] = "unclear"

            print("    %s: attn IO=%.3f S2=%.3f | proj IO=%.3f S=%.3f diff=%.3f | %s" % (
                head_name, r["attn_IO"], r["attn_S2"],
                r["proj_IO"], r["proj_S"], r["proj_diff"], r["role"]))

        results["exp1_l9h1"]["step_%d" % step] = step_results
        save_results(results)

        del model
        empty_cache()

    print("\nEXP 1 COMPLETE")


# ============================================================
# EXPERIMENT 2: Stanford GPT-2 L10H10 classification
# ============================================================
def run_exp2(results):
    print("\n" + "=" * 60)
    print("  EXP 2: Stanford GPT-2 L10H10 Wang Classification")
    print("=" * 60)

    results["exp2_stanford_classification"] = {}

    model = HookedTransformer.from_pretrained(
        "stanford-crfm/alias-gpt2-small-x21",
        device=DEVICE, revision="checkpoint-400000")

    # Check top heads by attention pattern
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.W_U
    W_O = model.W_O

    # Specifically check L10H10 and find top NM/S-inh heads
    heads_to_detail = [(10, 10)]  # dominant head by ablation

    # Also scan all heads for top Wang-classified NM and S-inh
    all_end_to_io = {}
    all_end_to_s2 = {}
    all_s2_to_s1 = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            attn_io, attn_s2, s2s1 = [], [], []

            for tmpl in TEMPLATES[:8]:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                attn = cache["blocks.%d.attn.hook_pattern" % layer]
                final_pos = tokens.shape[1] - 1
                positions = get_positions(tokens, io_ids, s_ids)

                for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                    if io_pos > 0:
                        attn_io.append(attn[i, head, final_pos, io_pos].item())
                    if s2_pos > 0:
                        attn_s2.append(attn[i, head, final_pos, s2_pos].item())
                    if s2_pos > 0 and s1_pos > 0:
                        s2s1.append(attn[i, head, s2_pos, s1_pos].item())

                del cache
                empty_cache()

            head_name = "L%dH%d" % (layer, head)
            all_end_to_io[head_name] = float(np.mean(attn_io)) if attn_io else 0
            all_end_to_s2[head_name] = float(np.mean(attn_s2)) if attn_s2 else 0
            all_s2_to_s1[head_name] = float(np.mean(s2s1)) if s2s1 else 0

        if (layer + 1) % 4 == 0:
            print("    Layer %d/%d" % (layer + 1, n_layers))

    # Top heads by each metric
    top_nm = sorted(all_end_to_io.items(), key=lambda x: -x[1])[:5]
    top_si = sorted(all_end_to_s2.items(), key=lambda x: -x[1])[:5]
    top_dt = sorted(all_s2_to_s1.items(), key=lambda x: -x[1])[:5]

    print("\n  Top Wang-style Name Movers (END->IO):")
    for h, s in top_nm:
        print("    %s: %.4f" % (h, s))

    print("  Top S-Inhibition (END->S2):")
    for h, s in top_si:
        print("    %s: %.4f" % (h, s))

    print("  Top Duplicate Token (S2->S1):")
    for h, s in top_dt:
        print("    %s: %.4f" % (h, s))

    # L10H10 specifically
    print("\n  L10H10: END->IO=%.4f, END->S2=%.4f, S2->S1=%.4f" % (
        all_end_to_io.get("L10H10", 0),
        all_end_to_s2.get("L10H10", 0),
        all_s2_to_s1.get("L10H10", 0)))

    l10h10_role = "S-inhibition" if all_end_to_s2.get("L10H10", 0) > all_end_to_io.get("L10H10", 0) else "Name mover"
    print("  L10H10 Wang classification: %s" % l10h10_role)

    # Also get L10H10 output projection
    io_projs, s_projs = [], []
    for tmpl in TEMPLATES[:10]:
        ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                        symmetric=True, seed=SEED)
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        z = cache["blocks.10.attn.hook_z"][:, -1, 10, :]
        head_out = z @ W_O[10, 10]
        for i in range(len(io_ids)):
            io_dir = W_U[:, io_ids[i].item()]
            s_dir = W_U[:, s_ids[i].item()]
            io_projs.append(torch.dot(head_out[i], io_dir).item())
            s_projs.append(torch.dot(head_out[i], s_dir).item())

        del cache
        empty_cache()

    print("  L10H10 projection: IO=%.4f, S=%.4f, diff=%.4f" % (
        np.mean(io_projs), np.mean(s_projs),
        np.mean(io_projs) - np.mean(s_projs)))

    results["exp2_stanford_classification"] = {
        "l10h10": {
            "end_to_IO": round(all_end_to_io.get("L10H10", 0), 4),
            "end_to_S2": round(all_end_to_s2.get("L10H10", 0), 4),
            "s2_to_s1": round(all_s2_to_s1.get("L10H10", 0), 4),
            "proj_IO": round(float(np.mean(io_projs)), 4),
            "proj_S": round(float(np.mean(s_projs)), 4),
            "proj_diff": round(float(np.mean(io_projs)) - float(np.mean(s_projs)), 4),
            "wang_role": l10h10_role,
        },
        "top5_name_movers": [{"head": h, "score": round(s, 4)} for h, s in top_nm],
        "top5_s_inhibition": [{"head": h, "score": round(s, 4)} for h, s in top_si],
        "top5_duplicate_token": [{"head": h, "score": round(s, 4)} for h, s in top_dt],
    }
    save_results(results)

    del model
    empty_cache()
    print("\nEXP 2 COMPLETE")


# ============================================================
# EXPERIMENT 3: High-resolution phase transition (Stanford)
# ============================================================
def run_exp3(results):
    print("\n" + "=" * 60)
    print("  EXP 3: High-Resolution Phase Transition (Stanford)")
    print("=" * 60)

    # Dense checkpoints in the transition zone
    transition_steps = list(range(500, 5050, 50))  # 500 to 5000 every 50 steps = 90 points

    if "exp3_phase_transition" not in results:
        results["exp3_phase_transition"] = {}

    for step in transition_steps:
        step_key = "step_%d" % step
        if step_key in results["exp3_phase_transition"]:
            continue

        try:
            model = HookedTransformer.from_pretrained(
                "stanford-crfm/alias-gpt2-small-x21",
                device=DEVICE, revision="checkpoint-%d" % step)
        except Exception as e:
            print("    Step %d: FAILED (%s)" % (step, str(e)[:60]))
            continue

        all_lds = []
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
                for i in range(len(io_ids)):
                    total += 1
                    ld = last[i, io_ids[i]].item() - last[i, s_ids[i]].item()
                    all_lds.append(ld)
            except:
                continue

        if total == 0:
            del model
            empty_cache()
            continue

        lds = np.array(all_lds)
        accuracy = float((lds > 0).mean())

        results["exp3_phase_transition"][step_key] = {
            "accuracy": round(accuracy, 4),
            "mean_ld": round(float(lds.mean()), 4),
            "pct_s_preferred": round(float((lds < 0).mean()), 4),
        }

        if step % 500 == 0:
            save_results(results)
            print("    Step %d: acc=%.3f, LD=%.4f" % (step, accuracy, lds.mean()))

        del model
        empty_cache()

    save_results(results)
    print("\nEXP 3 COMPLETE")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  POLISH EXPERIMENTS")
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()

    try:
        run_exp1(results)
    except Exception as e:
        print("EXP 1 FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_exp2(results)
    except Exception as e:
        print("EXP 2 FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_exp3(results)
    except Exception as e:
        print("EXP 3 FAILED: %s" % str(e))
        traceback.print_exc()

    elapsed = time.time() - t0
    save_results(results)
    print("\n  ALL DONE. Time: %.0fs (%.1f hours)" % (elapsed, elapsed / 3600))

if __name__ == "__main__":
    main()
