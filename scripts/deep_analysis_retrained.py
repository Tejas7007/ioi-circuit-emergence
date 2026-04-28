"""
Deep Analysis of Retrained Pythia-160M (seed=42)
=================================================
Experiment 1: L2H6 output projection — does it write zero like 410M's L4H6?
Experiment 2: Track L1H9, L6H7, L2H6, L8H9 attention across ALL 103 checkpoints
Experiment 3: Path patching L2H6 — which downstream heads depend on it?
Experiment 4: L6H7 output projection — it attended 64% to S2 at step 1800, what did it write?

Saves incrementally to /workspace/pythia-160m-retrain/deep_analysis.json
"""

import os
os.environ["HF_TOKEN"] = ""

import torch, json, time, traceback
import numpy as np
from transformers import GPTNeoXForCausalLM
from transformer_lens import HookedTransformer
import sys

for path in ['/workspace/MLP-Paper-Cole/src', os.path.expanduser('~/MLP-Paper-Cole/src')]:
    sys.path.insert(0, path)
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = "cuda"
CHECKPOINT_DIR = "/workspace/pythia-160m-retrain/checkpoints"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
RESULTS_FILE = "/workspace/pythia-160m-retrain/deep_analysis.json"
TEMPLATES = ALL_TEMPLATES[:15]
PPT = 20
SEED = 42

all_steps = sorted([int(d.split('_')[1]) for d in os.listdir(CHECKPOINT_DIR) if d.startswith('step_')])
print("Found %d checkpoints" % len(all_steps))

def empty_cache():
    torch.cuda.empty_cache()

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_model(step):
    ckpt_path = os.path.join(CHECKPOINT_DIR, "step_%d" % step)
    hf_model = GPTNeoXForCausalLM.from_pretrained(ckpt_path)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, hf_model=hf_model, device=DEVICE,
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

def get_template_data(model, n_templates=10):
    """Get cached template data for reuse across experiments."""
    template_data = []
    for tmpl in TEMPLATES[:n_templates]:
        try:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                            symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).to(DEVICE)
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
            template_data.append((tokens, io_ids, s_ids))
        except:
            continue
    return template_data


# ============================================================
# EXPERIMENT 1: Output projections of key heads
# ============================================================
def run_exp1(results):
    print("\n" + "=" * 60)
    print("  EXP 1: Output Projections of Key Heads")
    print("  Heads: L2H6, L6H7, L1H9, L8H9, L1H8")
    print("  Steps: key checkpoints where each head was dominant")
    print("=" * 60)

    if "exp1_projections" not in results:
        results["exp1_projections"] = {}

    # Check projections at steps where each head was dominant + final
    proj_steps = [200, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
                  2200, 2400, 2600, 2800, 3000, 4000, 5000, 6000, 8000, 10000]
    heads_to_check = [(2, 6), (6, 7), (1, 9), (8, 9), (1, 8), (1, 2)]

    for step in proj_steps:
        step_key = "step_%d" % step
        if step_key in results["exp1_projections"]:
            print("  Step %d done, skip" % step)
            continue

        if step not in all_steps:
            continue

        print("\n  --- Step %d ---" % step)
        try:
            model = load_model(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        W_U = model.W_U
        W_O = model.W_O
        template_data = get_template_data(model)

        step_results = {}

        for layer, head in heads_to_check:
            attn_io, attn_s1, attn_s2, attn_self = [], [], [], []
            io_projs, s_projs = [], []

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
            mean_io = float(np.mean(io_projs)) if io_projs else 0
            mean_s = float(np.mean(s_projs)) if s_projs else 0

            step_results[head_name] = {
                "attn_IO": round(float(np.mean(attn_io)), 4) if attn_io else 0,
                "attn_S1": round(float(np.mean(attn_s1)), 4) if attn_s1 else 0,
                "attn_S2": round(float(np.mean(attn_s2)), 4) if attn_s2 else 0,
                "attn_self": round(float(np.mean(attn_self)), 4) if attn_self else 0,
                "proj_IO": round(mean_io, 4),
                "proj_S": round(mean_s, 4),
                "proj_diff": round(mean_io - mean_s, 4),
            }

            print("    %s: attn IO=%.3f S2=%.3f self=%.3f | proj IO=%.3f S=%.3f diff=%.3f" % (
                head_name,
                step_results[head_name]["attn_IO"],
                step_results[head_name]["attn_S2"],
                step_results[head_name]["attn_self"],
                mean_io, mean_s, mean_io - mean_s))

        results["exp1_projections"][step_key] = step_results
        save_results(results)

        del model
        empty_cache()

    print("\nEXP 1 COMPLETE")


# ============================================================
# EXPERIMENT 2: Head trajectories across ALL 103 checkpoints
# ============================================================
def run_exp2(results):
    print("\n" + "=" * 60)
    print("  EXP 2: Head Trajectories Across All 103 Checkpoints")
    print("  Tracking: L1H2, L1H9, L6H7, L2H6, L8H9, L1H8, L0H10")
    print("=" * 60)

    if "exp2_trajectories" not in results:
        results["exp2_trajectories"] = {}

    track_heads = [(1, 2), (1, 9), (6, 7), (2, 6), (8, 9), (1, 8), (0, 10)]

    for step in all_steps:
        step_key = "step_%d" % step
        if step_key in results["exp2_trajectories"]:
            continue

        print("  Step %d..." % step, end=" ", flush=True)
        try:
            model = load_model(step)
        except Exception as e:
            print("FAILED")
            continue

        step_data = {}

        # Also get accuracy
        all_lds = []

        for layer, head in track_heads:
            attn_io, attn_s1, attn_s2, attn_self = [], [], [], []
            s2_to_s1 = []

            for tmpl in TEMPLATES[:8]:
                try:
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
                        if s1_pos > 0:
                            attn_s1.append(attn[i, head, final_pos, s1_pos].item())
                        if s2_pos > 0:
                            attn_s2.append(attn[i, head, final_pos, s2_pos].item())
                        attn_self.append(attn[i, head, final_pos, final_pos].item())
                        # Also track duplicate-token behavior (S2 -> S1)
                        if s2_pos > 0 and s1_pos > 0:
                            s2_to_s1.append(attn[i, head, s2_pos, s1_pos].item())

                    # Collect LDs once
                    if layer == track_heads[0][0] and head == track_heads[0][1]:
                        logits = model(tokens)
                        last = logits[:, -1, :]
                        for i in range(len(io_ids)):
                            all_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())

                    del cache
                    empty_cache()
                except:
                    continue

            head_name = "L%dH%d" % (layer, head)
            step_data[head_name] = {
                "end_to_IO": round(float(np.mean(attn_io)), 4) if attn_io else 0,
                "end_to_S1": round(float(np.mean(attn_s1)), 4) if attn_s1 else 0,
                "end_to_S2": round(float(np.mean(attn_s2)), 4) if attn_s2 else 0,
                "end_to_self": round(float(np.mean(attn_self)), 4) if attn_self else 0,
                "s2_to_s1": round(float(np.mean(s2_to_s1)), 4) if s2_to_s1 else 0,
            }

        lds = np.array(all_lds) if all_lds else np.array([0])
        step_data["accuracy"] = round(float((lds > 0).mean()), 4)

        results["exp2_trajectories"][step_key] = step_data
        save_results(results)

        # Print key heads
        l2h6 = step_data.get("L2H6", {})
        l6h7 = step_data.get("L6H7", {})
        l8h9 = step_data.get("L8H9", {})
        print("acc=%.1f%% L2H6_S2=%.3f L6H7_S2=%.3f L8H9_S2=%.3f" % (
            step_data["accuracy"] * 100,
            l2h6.get("end_to_S2", 0),
            l6h7.get("end_to_S2", 0),
            l8h9.get("end_to_S2", 0)))

        del model
        empty_cache()

    print("\nEXP 2 COMPLETE")


# ============================================================
# EXPERIMENT 3: Path Patching L2H6
# ============================================================
def run_exp3(results):
    print("\n" + "=" * 60)
    print("  EXP 3: Path Patching L2H6")
    print("  Question: Which downstream heads depend on L2H6?")
    print("=" * 60)

    if "exp3_path_patching" not in results:
        results["exp3_path_patching"] = {}

    # Test at steps where L2H6 is dominant
    patch_steps = [3000, 5000, 8000, 10000]

    for step in patch_steps:
        step_key = "step_%d" % step
        if step_key in results["exp3_path_patching"]:
            print("  Step %d done, skip" % step)
            continue

        if step not in all_steps:
            continue

        print("\n  --- Step %d ---" % step)
        try:
            model = load_model(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        template_data = get_template_data(model, n_templates=8)

        # Step 1: Get clean L2H6 output and corrupted L2H6 output
        # "Corrupted" = L2H6 zeroed out
        # For each downstream head, measure: does its attention pattern change
        # when L2H6 is zeroed?

        # Get clean attention patterns for all downstream heads (layers 3-11)
        clean_attns = {}  # head_name -> list of attention to IO, S2

        for tokens, io_ids, s_ids in template_data[:5]:
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
            positions = get_positions(tokens, io_ids, s_ids)
            final_pos = tokens.shape[1] - 1

            for layer in range(3, n_layers):  # Downstream of L2
                for head in range(n_heads):
                    head_name = "L%dH%d" % (layer, head)
                    if head_name not in clean_attns:
                        clean_attns[head_name] = {"to_IO": [], "to_S2": [], "to_self": []}

                    attn = cache["blocks.%d.attn.hook_pattern" % layer]
                    for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                        if io_pos > 0:
                            clean_attns[head_name]["to_IO"].append(
                                attn[i, head, final_pos, io_pos].item())
                        if s2_pos > 0:
                            clean_attns[head_name]["to_S2"].append(
                                attn[i, head, final_pos, s2_pos].item())
                        clean_attns[head_name]["to_self"].append(
                            attn[i, head, final_pos, final_pos].item())

            del cache
            empty_cache()

        # Get corrupted attention patterns (L2H6 zeroed)
        def zero_l2h6(value, hook):
            value[:, :, 6, :] = 0.0
            return value

        corrupted_attns = {}

        for tokens, io_ids, s_ids in template_data[:5]:
            _, cache = model.run_with_cache(
                tokens, remove_batch_dim=False,
                fwd_hooks=[("blocks.2.attn.hook_z", zero_l2h6)])
            positions = get_positions(tokens, io_ids, s_ids)
            final_pos = tokens.shape[1] - 1

            for layer in range(3, n_layers):
                for head in range(n_heads):
                    head_name = "L%dH%d" % (layer, head)
                    if head_name not in corrupted_attns:
                        corrupted_attns[head_name] = {"to_IO": [], "to_S2": [], "to_self": []}

                    attn = cache["blocks.%d.attn.hook_pattern" % layer]
                    for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                        if io_pos > 0:
                            corrupted_attns[head_name]["to_IO"].append(
                                attn[i, head, final_pos, io_pos].item())
                        if s2_pos > 0:
                            corrupted_attns[head_name]["to_S2"].append(
                                attn[i, head, final_pos, s2_pos].item())
                        corrupted_attns[head_name]["to_self"].append(
                            attn[i, head, final_pos, final_pos].item())

            del cache
            empty_cache()

        # Compute changes
        head_changes = {}
        for head_name in clean_attns:
            if head_name in corrupted_attns:
                clean_io = np.mean(clean_attns[head_name]["to_IO"]) if clean_attns[head_name]["to_IO"] else 0
                corr_io = np.mean(corrupted_attns[head_name]["to_IO"]) if corrupted_attns[head_name]["to_IO"] else 0
                clean_s2 = np.mean(clean_attns[head_name]["to_S2"]) if clean_attns[head_name]["to_S2"] else 0
                corr_s2 = np.mean(corrupted_attns[head_name]["to_S2"]) if corrupted_attns[head_name]["to_S2"] else 0

                delta_io = corr_io - clean_io
                delta_s2 = corr_s2 - clean_s2
                total_change = abs(delta_io) + abs(delta_s2)

                head_changes[head_name] = {
                    "clean_IO": round(float(clean_io), 4),
                    "corrupted_IO": round(float(corr_io), 4),
                    "delta_IO": round(float(delta_io), 4),
                    "clean_S2": round(float(clean_s2), 4),
                    "corrupted_S2": round(float(corr_s2), 4),
                    "delta_S2": round(float(delta_s2), 4),
                    "total_change": round(float(total_change), 4),
                }

        # Top affected heads
        sorted_changes = sorted(head_changes.items(), key=lambda x: -x[1]["total_change"])
        top10 = sorted_changes[:10]

        print("    Top 10 heads affected by L2H6 ablation:")
        for h, c in top10:
            print("      %s: dIO=%+.4f dS2=%+.4f total=%.4f" % (
                h, c["delta_IO"], c["delta_S2"], c["total_change"]))

        results["exp3_path_patching"][step_key] = {
            "top10_affected": [{"head": h, **c} for h, c in top10],
            "all_changes": head_changes,
        }
        save_results(results)

        del model
        empty_cache()

    print("\nEXP 3 COMPLETE")


# ============================================================
# EXPERIMENT 4: Wang et al. classification at key steps
# ============================================================
def run_exp4(results):
    print("\n" + "=" * 60)
    print("  EXP 4: Wang et al. Classification (Retrained Model)")
    print("=" * 60)

    if "exp4_wang_classification" not in results:
        results["exp4_wang_classification"] = {}

    class_steps = [1000, 1400, 1800, 2600, 3000, 5000, 10000]

    for step in class_steps:
        step_key = "step_%d" % step
        if step_key in results["exp4_wang_classification"]:
            print("  Step %d done, skip" % step)
            continue

        if step not in all_steps:
            continue

        print("\n  --- Step %d ---" % step)
        try:
            model = load_model(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        head_metrics = {}

        for layer in range(n_layers):
            for head in range(n_heads):
                end_to_io, end_to_s2, s2_to_s1 = [], [], []
                prev_token_scores = []

                for tmpl in TEMPLATES[:8]:
                    try:
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
                                end_to_io.append(attn[i, head, final_pos, io_pos].item())
                            if s2_pos > 0:
                                end_to_s2.append(attn[i, head, final_pos, s2_pos].item())
                            if s2_pos > 0 and s1_pos > 0:
                                s2_to_s1.append(attn[i, head, s2_pos, s1_pos].item())
                            pt_sum = 0.0
                            pt_count = 0
                            for p in range(1, min(final_pos, attn.shape[2])):
                                if p < attn.shape[3]:
                                    pt_sum += attn[i, head, p, p-1].item()
                                    pt_count += 1
                            if pt_count > 0:
                                prev_token_scores.append(pt_sum / pt_count)

                        del cache
                        empty_cache()
                    except:
                        continue

                head_name = "L%dH%d" % (layer, head)
                metrics = {
                    "end_to_IO": round(float(np.mean(end_to_io)), 4) if end_to_io else 0,
                    "end_to_S2": round(float(np.mean(end_to_s2)), 4) if end_to_s2 else 0,
                    "s2_to_s1": round(float(np.mean(s2_to_s1)), 4) if s2_to_s1 else 0,
                    "prev_token": round(float(np.mean(prev_token_scores)), 4) if prev_token_scores else 0,
                }

                roles = []
                if metrics["end_to_IO"] > 0.1: roles.append("name_mover")
                if metrics["end_to_S2"] > 0.1: roles.append("s_inhibition")
                if metrics["s2_to_s1"] > 0.2: roles.append("duplicate_token")
                if metrics["prev_token"] > 0.3: roles.append("previous_token")
                metrics["roles"] = roles
                head_metrics[head_name] = metrics

            if (layer + 1) % 4 == 0:
                print("    Layer %d/%d" % (layer + 1, n_layers))

        counts = {}
        top_heads = {}
        for role in ["name_mover", "s_inhibition", "duplicate_token", "previous_token"]:
            matching = [(h, m) for h, m in head_metrics.items() if role in m["roles"]]
            counts[role] = len(matching)
            score_key = {"name_mover": "end_to_IO", "s_inhibition": "end_to_S2",
                        "duplicate_token": "s2_to_s1", "previous_token": "prev_token"}[role]
            matching.sort(key=lambda x: -x[1][score_key])
            top_heads[role] = [{"head": h, "score": round(m[score_key], 4)} for h, m in matching[:5]]

        print("    Counts: NM=%d S-inh=%d DupTok=%d PrevTok=%d" % (
            counts.get("name_mover", 0), counts.get("s_inhibition", 0),
            counts.get("duplicate_token", 0), counts.get("previous_token", 0)))
        for role in top_heads:
            if top_heads[role]:
                print("    Top %s: %s (%.4f)" % (role, top_heads[role][0]["head"], top_heads[role][0]["score"]))

        results["exp4_wang_classification"][step_key] = {
            "counts": counts,
            "top_heads": top_heads,
            "all_heads": head_metrics,
        }
        save_results(results)

        del model
        empty_cache()

    print("\nEXP 4 COMPLETE")


# ============================================================
# EXPERIMENT 5: Compare original vs retrained at step 10000
# ============================================================
def run_exp5(results):
    print("\n" + "=" * 60)
    print("  EXP 5: Original vs Retrained Comparison at Step 10000")
    print("=" * 60)

    if "exp5_comparison" in results:
        print("  Already done, skip")
        return

    # Load retrained
    print("  Loading retrained (seed=42)...")
    retrained = load_model(10000)

    # Load original
    print("  Loading original (seed=1234)...")
    from transformers import AutoModelForCausalLM
    hf_orig = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="step10000")
    original = HookedTransformer.from_pretrained(
        MODEL_NAME, hf_model=hf_orig, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf_orig

    comparison = {}

    for label, model in [("original_seed1234", original), ("retrained_seed42", retrained)]:
        print("\n  --- %s ---" % label)

        # IOI accuracy
        all_lds = []
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
                    all_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())
            except:
                continue

        lds = np.array(all_lds)
        accuracy = float((lds > 0).mean())

        # Find top head by ablation
        template_data = get_template_data(model, n_templates=10)
        base_lds = []
        for tokens, io_ids, s_ids in template_data:
            logits = model(tokens)
            last = logits[:, -1, :]
            for i in range(len(io_ids)):
                base_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())
        base_mean = float(np.mean(base_lds))

        head_deltas = {}
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
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
                delta = float(np.mean(abl_lds)) - base_mean
                head_deltas["L%dH%d" % (layer, head)] = round(delta, 4)

        sorted_heads = sorted(head_deltas.items(), key=lambda x: x[1])

        comparison[label] = {
            "accuracy": round(accuracy, 4),
            "mean_ld": round(float(lds.mean()), 4),
            "top5_heads": [{"head": h, "delta": d} for h, d in sorted_heads[:5]],
        }
        print("    Acc: %.1f%%, Top head: %s (delta=%.4f)" % (
            accuracy * 100, sorted_heads[0][0], sorted_heads[0][1]))

    results["exp5_comparison"] = comparison
    save_results(results)

    del original, retrained
    empty_cache()
    print("\nEXP 5 COMPLETE")


# ============================================================
# EXPERIMENT 6: Linear Probes for IO Identity
# ============================================================
def run_exp6(results):
    """Train linear probes on residual stream at each layer to predict IO token.
    Tests whether IO information is represented at different layers depending on seed."""
    print("\n" + "=" * 60)
    print("  EXP 6: Linear Probes for IO Identity")
    print("  Question: At which layer can we decode IO from the residual stream?")
    print("=" * 60)

    if "exp6_linear_probes" not in results:
        results["exp6_linear_probes"] = {}

    probe_steps = [1000, 2000, 3000, 5000, 10000]

    for step in probe_steps:
        step_key = "step_%d" % step
        if step_key in results["exp6_linear_probes"]:
            print("  Step %d done, skip" % step)
            continue

        if step not in all_steps:
            continue

        print("\n  --- Step %d ---" % step)
        try:
            model = load_model(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model

        # Collect residual stream activations and IO labels
        all_resids = {l: [] for l in range(n_layers + 1)}  # +1 for post-final-layer
        all_io_ids = []

        for tmpl in TEMPLATES[:10]:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)

                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                final_pos = tokens.shape[1] - 1

                # Get residual stream at final position after each layer
                for layer in range(n_layers):
                    resid = cache["blocks.%d.hook_resid_post" % layer][:, final_pos, :]
                    all_resids[layer].append(resid.detach().cpu().float())

                # Also get embedding layer (layer 0 input)
                resid_pre = cache["blocks.0.hook_resid_pre"][:, final_pos, :]
                all_resids[n_layers].append(resid_pre.detach().cpu().float())

                all_io_ids.extend(io_ids.cpu().tolist())

                del cache
                empty_cache()
            except:
                continue

        if not all_io_ids:
            del model
            empty_cache()
            continue

        # Concatenate
        for l in all_resids:
            if all_resids[l]:
                all_resids[l] = torch.cat(all_resids[l], dim=0)

        io_labels = torch.tensor(all_io_ids)
        n_samples = len(io_labels)

        # Get unique IO tokens for classification
        unique_ios = torch.unique(io_labels)
        n_classes = len(unique_ios)
        # Map to class indices
        label_map = {v.item(): i for i, v in enumerate(unique_ios)}
        class_labels = torch.tensor([label_map[v.item()] for v in io_labels])

        # Train/test split (80/20)
        n_train = int(0.8 * n_samples)
        perm = torch.randperm(n_samples)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        layer_accuracies = {}

        for layer in range(n_layers):
            if not isinstance(all_resids[layer], torch.Tensor):
                continue

            X = all_resids[layer]
            y = class_labels

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Simple linear probe: logistic regression via SGD
            probe = torch.nn.Linear(d_model, n_classes)
            optimizer_probe = torch.optim.Adam(probe.parameters(), lr=1e-3)
            loss_fn = torch.nn.CrossEntropyLoss()

            # Train for 200 steps
            probe.train()
            for ep in range(200):
                optimizer_probe.zero_grad()
                logits = probe(X_train)
                loss = loss_fn(logits, y_train)
                loss.backward()
                optimizer_probe.step()

            # Test accuracy
            probe.eval()
            with torch.no_grad():
                test_logits = probe(X_test)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == y_test).float().mean().item()

            layer_accuracies["layer_%d" % layer] = round(test_acc, 4)

        # Find best layer
        if layer_accuracies:
            best_layer = max(layer_accuracies, key=layer_accuracies.get)
            print("    Probe accuracies:")
            for l in sorted(layer_accuracies.keys(), key=lambda x: int(x.split('_')[1])):
                marker = " <-- BEST" if l == best_layer else ""
                print("      %s: %.1f%%%s" % (l, layer_accuracies[l] * 100, marker))

        results["exp6_linear_probes"][step_key] = {
            "n_samples": n_samples,
            "n_classes": n_classes,
            "layer_accuracies": layer_accuracies,
        }
        save_results(results)

        del model
        empty_cache()

    print("\nEXP 6 COMPLETE")


# ============================================================
# EXPERIMENT 7: Causal Tracing / Activation Patching
# ============================================================
def run_exp7(results):
    """Patch activations from clean run into corrupted run at each layer/position.
    Maps complete information flow for IOI circuit."""
    print("\n" + "=" * 60)
    print("  EXP 7: Causal Tracing (Activation Patching)")
    print("  Question: Where does IOI information flow through the network?")
    print("=" * 60)

    if "exp7_causal_tracing" not in results:
        results["exp7_causal_tracing"] = {}

    trace_steps = [3000, 5000, 10000]

    for step in trace_steps:
        step_key = "step_%d" % step
        if step_key in results["exp7_causal_tracing"]:
            print("  Step %d done, skip" % step)
            continue

        if step not in all_steps:
            continue

        print("\n  --- Step %d ---" % step)
        try:
            model = load_model(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        n_layers = model.cfg.n_layers

        # Strategy: for each prompt, create a corrupted version where S is replaced
        # with a different name. Then patch clean activations into corrupted run
        # one layer at a time and measure how much IOI performance recovers.

        # Collect clean and corrupted logit diffs
        clean_lds = []
        corrupted_lds = []
        patch_results_by_layer = {l: [] for l in range(n_layers)}
        patch_results_by_layer_pos = {}  # (layer, pos_type) -> []

        for tmpl in TEMPLATES[:8]:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

                # Clean run with cache
                clean_logits, clean_cache = model.run_with_cache(tokens, remove_batch_dim=False)
                clean_last = clean_logits[:, -1, :].float()

                for i in range(len(io_ids)):
                    clean_lds.append(clean_last[i, io_ids[i]].item() - clean_last[i, s_ids[i]].item())

                # Create corrupted tokens: swap S name with a random different name
                corrupted_tokens = tokens.clone()
                positions = get_positions(tokens, io_ids, s_ids)

                for i in range(tokens.shape[0]):
                    io_pos, s1_pos, s2_pos = positions[i]
                    s_tok = s_ids[i].item()
                    # Replace S with IO (simple corruption that breaks the task)
                    io_tok = io_ids[i].item()
                    if s1_pos > 0:
                        corrupted_tokens[i, s1_pos] = io_tok
                    if s2_pos > 0:
                        corrupted_tokens[i, s2_pos] = io_tok

                # Corrupted run
                corrupted_logits = model(corrupted_tokens)
                corrupted_last = corrupted_logits[:, -1, :].float()
                for i in range(len(io_ids)):
                    corrupted_lds.append(corrupted_last[i, io_ids[i]].item() - corrupted_last[i, s_ids[i]].item())

                # Patch each layer: run corrupted but restore clean resid at one layer
                final_pos = tokens.shape[1] - 1

                for layer in range(n_layers):
                    # Patch at ALL positions for this layer
                    def patch_hook_all(value, hook, clean_val=clean_cache["blocks.%d.hook_resid_post" % layer]):
                        return clean_val

                    patched_logits = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[("blocks.%d.hook_resid_post" % layer, patch_hook_all)])
                    patched_last = patched_logits[:, -1, :].float()

                    for i in range(len(io_ids)):
                        patched_ld = patched_last[i, io_ids[i]].item() - patched_last[i, s_ids[i]].item()
                        patch_results_by_layer[layer].append(patched_ld)

                    # Patch at specific positions only
                    for pos_name, pos_idx_fn in [
                        ("S1", lambda p: p[1]),
                        ("S2", lambda p: p[2]),
                        ("IO", lambda p: p[0]),
                        ("END", lambda p: final_pos),
                    ]:
                        key = (layer, pos_name)
                        if key not in patch_results_by_layer_pos:
                            patch_results_by_layer_pos[key] = []

                        def patch_hook_pos(value, hook,
                                          clean_val=clean_cache["blocks.%d.hook_resid_post" % layer],
                                          pos_fn=pos_idx_fn, positions_local=positions):
                            for i in range(value.shape[0]):
                                pos = pos_fn(positions_local[i])
                                if pos is not None and pos > 0 and pos < value.shape[1]:
                                    value[i, pos, :] = clean_val[i, pos, :]
                            return value

                        patched_logits = model.run_with_hooks(
                            corrupted_tokens,
                            fwd_hooks=[("blocks.%d.hook_resid_post" % layer, patch_hook_pos)])
                        patched_last = patched_logits[:, -1, :].float()

                        for i in range(len(io_ids)):
                            patched_ld = patched_last[i, io_ids[i]].item() - patched_last[i, s_ids[i]].item()
                            patch_results_by_layer_pos[key].append(patched_ld)

                del clean_cache
                empty_cache()

            except Exception as e:
                print("    Template error: %s" % str(e)[:60])
                continue

        if not clean_lds:
            del model
            empty_cache()
            continue

        clean_mean = float(np.mean(clean_lds))
        corrupted_mean = float(np.mean(corrupted_lds))
        total_effect = clean_mean - corrupted_mean

        # Compute recovery fraction per layer (all positions)
        layer_recovery = {}
        for layer in range(n_layers):
            if patch_results_by_layer[layer]:
                patched_mean = float(np.mean(patch_results_by_layer[layer]))
                recovery = (patched_mean - corrupted_mean) / total_effect if total_effect != 0 else 0
                layer_recovery["layer_%d" % layer] = {
                    "patched_mean_ld": round(patched_mean, 4),
                    "recovery_fraction": round(recovery, 4),
                }

        # Compute recovery per (layer, position)
        position_recovery = {}
        for (layer, pos_name), lds_list in patch_results_by_layer_pos.items():
            if lds_list:
                patched_mean = float(np.mean(lds_list))
                recovery = (patched_mean - corrupted_mean) / total_effect if total_effect != 0 else 0
                key = "layer_%d_%s" % (layer, pos_name)
                position_recovery[key] = {
                    "recovery_fraction": round(recovery, 4),
                }

        # Print summary
        print("    Clean LD: %.4f, Corrupted LD: %.4f, Total effect: %.4f" % (
            clean_mean, corrupted_mean, total_effect))
        print("    Layer recovery (all positions):")
        for l in sorted(layer_recovery.keys(), key=lambda x: int(x.split('_')[1])):
            r = layer_recovery[l]
            bar = "#" * int(max(0, r["recovery_fraction"]) * 30)
            print("      %s: %.1f%% %s" % (l, r["recovery_fraction"] * 100, bar))

        # Find top position-specific patches
        sorted_pos = sorted(position_recovery.items(), key=lambda x: -x[1]["recovery_fraction"])
        print("    Top 10 (layer, position) patches:")
        for key, val in sorted_pos[:10]:
            print("      %s: %.1f%%" % (key, val["recovery_fraction"] * 100))

        results["exp7_causal_tracing"][step_key] = {
            "clean_mean_ld": round(clean_mean, 4),
            "corrupted_mean_ld": round(corrupted_mean, 4),
            "total_effect": round(total_effect, 4),
            "layer_recovery": layer_recovery,
            "position_recovery": position_recovery,
        }
        save_results(results)

        del model
        empty_cache()

    print("\nEXP 7 COMPLETE")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  DEEP ANALYSIS - RETRAINED PYTHIA-160M")
    print("  7 Experiments")
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()

    for i, fn in enumerate([run_exp1, run_exp2, run_exp3, run_exp4, run_exp5, run_exp6, run_exp7], 1):
        try:
            fn(results)
        except Exception as e:
            print("EXP %d FAILED: %s" % (i, str(e)))
            traceback.print_exc()

    elapsed = time.time() - t0
    save_results(results)
    print("\n" + "=" * 60)
    print("  ALL DONE. Time: %.1f hours" % (elapsed / 3600))
    print("  Results: %s" % RESULTS_FILE)
    print("=" * 60)

if __name__ == "__main__":
    main()
