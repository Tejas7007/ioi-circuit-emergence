import os
os.environ["HF_TOKEN"] = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
"""
Final Experiments:
  1. Individual head trajectories across ALL 154 Pythia-160M checkpoints
     (attention patterns only - fast, no ablation)
  2. Sensitivity analysis on tau threshold (steps 1000, 3000, 143000)
  3. Wang et al. style attention-based head classification (steps 1000, 3000, 143000)

Saves to results/final_three.json
"""

import torch, json, os, time, shutil, traceback
import numpy as np
from transformer_lens import HookedTransformer
import sys
sys.path.insert(0, '/workspace/MLP-Paper-Cole/src')
sys.path.insert(0, os.path.expanduser('~/MLP-Paper-Cole/src'))
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TEMPLATES = ALL_TEMPLATES[:15]
PPT, SEED = 20, 42
RESULTS_FILE = "results/final_three.json"
MODEL = "EleutherAI/pythia-160m-deduped"

ALL_PYTHIA_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
    21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000,
    31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000,
    41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000,
    51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000,
    61000, 62000, 63000, 64000, 65000, 66000, 67000, 68000, 69000, 70000,
    71000, 72000, 73000, 74000, 75000, 76000, 77000, 78000, 79000, 80000,
    81000, 82000, 83000, 84000, 85000, 86000, 87000, 88000, 89000, 90000,
    91000, 92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000, 100000,
    101000, 102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000, 110000,
    111000, 112000, 113000, 114000, 115000, 116000, 117000, 118000, 119000, 120000,
    121000, 122000, 123000, 124000, 125000, 126000, 127000, 128000, 129000, 130000,
    131000, 132000, 133000, 134000, 135000, 136000, 137000, 138000, 139000, 140000,
    141000, 142000, 143000]

# Only track attention at a subset for speed (still dense in the interesting zone)
TRAJECTORY_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000,
    16000, 33000, 66000, 100000, 143000]

# Key heads to track
TRACK_HEADS = [(0, 5), (0, 6), (0, 10), (1, 4), (1, 8), (8, 9)]

# Steps for deep analysis (ablation + classification)
DEEP_STEPS = [512, 1000, 2000, 3000, 5000, 143000]

from transformers import AutoModelForCausalLM

def load_pythia(step):
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision="step%d" % step,
    )
    model = HookedTransformer.from_pretrained(
        MODEL, hf_model=hf_model, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True,
    )
    del hf_model
    return model

def clear_cache():
    for d in ['/workspace/.hf_home/hub', os.path.expanduser('~/.cache/huggingface/hub')]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith('models--'):
                    shutil.rmtree(os.path.join(d, f), ignore_errors=True)

def empty_cache():
    if DEVICE == "cuda": torch.cuda.empty_cache()
    elif DEVICE == "mps": torch.mps.empty_cache()

def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def get_positions(tokens, io_ids, s_ids):
    """Find IO, S1, S2 positions for each example."""
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
# PART 1: Head Trajectories (attention patterns across training)
# ============================================================
def run_part1(results):
    print("\n" + "=" * 60)
    print("  PART 1: Head Trajectories (%d steps)" % len(TRAJECTORY_STEPS))
    print("=" * 60)

    if "head_trajectories" not in results:
        results["head_trajectories"] = {}

    for step in TRAJECTORY_STEPS:
        step_key = "step_%d" % step
        if step_key in results["head_trajectories"]:
            print("  Step %d done, skip" % step)
            continue

        print("\n  --- Step %d ---" % step)
        try:
            clear_cache()
            model = load_pythia(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        step_data = {}

        # Also compute IOI accuracy
        all_lds = []

        for layer, head in TRACK_HEADS:
            attn_io, attn_s1, attn_s2, attn_self = [], [], [], []

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

                for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                    if io_pos > 0:
                        attn_io.append(attn[i, head, final_pos, io_pos].item())
                    if s1_pos > 0:
                        attn_s1.append(attn[i, head, final_pos, s1_pos].item())
                    if s2_pos > 0:
                        attn_s2.append(attn[i, head, final_pos, s2_pos].item())
                    attn_self.append(attn[i, head, final_pos, final_pos].item())

                # Collect LDs (only once per step, not per head)
                if layer == TRACK_HEADS[0][0] and head == TRACK_HEADS[0][1]:
                    logits = model(tokens)
                    last = logits[:, -1, :]
                    for i in range(len(io_ids)):
                        ld = last[i, io_ids[i]].item() - last[i, s_ids[i]].item()
                        all_lds.append(ld)

                del cache
                empty_cache()

            head_name = "L%dH%d" % (layer, head)
            step_data[head_name] = {
                "attn_IO": round(float(np.mean(attn_io)), 4) if attn_io else 0,
                "attn_S1": round(float(np.mean(attn_s1)), 4) if attn_s1 else 0,
                "attn_S2": round(float(np.mean(attn_s2)), 4) if attn_s2 else 0,
                "attn_self": round(float(np.mean(attn_self)), 4) if attn_self else 0,
            }

        lds = np.array(all_lds) if all_lds else np.array([0])
        step_data["accuracy"] = round(float((lds > 0).mean()), 4)
        step_data["mean_ld"] = round(float(lds.mean()), 4)

        results["head_trajectories"][step_key] = step_data
        save_results(results)

        # Print key heads
        for h in ["L0H10", "L8H9", "L1H8"]:
            if h in step_data:
                r = step_data[h]
                print("    %s: IO=%.3f S1=%.3f S2=%.3f self=%.3f" % (
                    h, r["attn_IO"], r["attn_S1"], r["attn_S2"], r["attn_self"]))
        print("    acc=%.3f" % step_data["accuracy"])

        del model
        empty_cache()

    print("\nPART 1 COMPLETE")


# ============================================================
# PART 2: Sensitivity Analysis on tau
# ============================================================
def run_part2(results):
    print("\n" + "=" * 60)
    print("  PART 2: Sensitivity Analysis")
    print("=" * 60)

    if "sensitivity" not in results:
        results["sensitivity"] = {}

    TAUS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    for step in DEEP_STEPS:
        step_key = "step_%d" % step
        if step_key in results["sensitivity"]:
            print("  Step %d done, skip" % step)
            continue

        print("\n  --- Step %d ---" % step)
        try:
            clear_cache()
            model = load_pythia(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        # Get baseline
        template_data = []
        base_lds = []
        for tmpl in TEMPLATES[:10]:
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

        base_mean_ld = float(np.mean(base_lds))

        # Ablate each head, store raw delta_ioi
        all_deltas = {}
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
                abl_mean_ld = float(np.mean(abl_lds))
                delta = abl_mean_ld - base_mean_ld
                head_name = "L%dH%d" % (layer, head)
                all_deltas[head_name] = round(delta, 6)

            if (layer + 1) % 4 == 0:
                print("    Layer %d/%d" % (layer + 1, n_layers))

        # Count at each threshold
        tau_results = {}
        for tau in TAUS:
            n_nm = sum(1 for d in all_deltas.values() if d < -tau)
            n_neg = sum(1 for d in all_deltas.values() if d > tau)
            tau_results["tau_%.3f" % tau] = {
                "name_movers": n_nm,
                "negative_nm": n_neg,
                "total_classified": n_nm + n_neg,
                "pct_classified": round((n_nm + n_neg) / len(all_deltas), 4),
            }
            print("    tau=%.3f: NM=%d, NegNM=%d, total=%d/%d (%.0f%%)" % (
                tau, n_nm, n_neg, n_nm + n_neg, len(all_deltas),
                (n_nm + n_neg) / len(all_deltas) * 100))

        # Store top/bottom heads
        sorted_deltas = sorted(all_deltas.items(), key=lambda x: x[1])
        top5_nm = [{"head": h, "delta": d} for h, d in sorted_deltas[:5]]
        top5_neg = [{"head": h, "delta": d} for h, d in sorted_deltas[-5:]]

        results["sensitivity"][step_key] = {
            "base_mean_ld": round(base_mean_ld, 4),
            "thresholds": tau_results,
            "top5_name_movers": top5_nm,
            "top5_negative_nm": top5_neg,
            "all_deltas": all_deltas,
        }
        save_results(results)

        del model
        empty_cache()

    print("\nPART 2 COMPLETE")


# ============================================================
# PART 3: Wang et al. Style Attention-Based Classification
# ============================================================
def run_part3(results):
    print("\n" + "=" * 60)
    print("  PART 3: Wang et al. Attention-Based Classification")
    print("=" * 60)

    if "wang_classification" not in results:
        results["wang_classification"] = {}

    for step in DEEP_STEPS:
        step_key = "step_%d" % step
        if step_key in results["wang_classification"]:
            print("  Step %d done, skip" % step)
            continue

        print("\n  --- Step %d ---" % step)
        try:
            clear_cache()
            model = load_pythia(step)
        except Exception as e:
            print("    FAILED: %s" % str(e)[:80])
            continue

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        # Collect attention patterns for ALL heads
        # For each head, measure:
        #   - attn END -> IO (name mover signal)
        #   - attn END -> S2 (S-inhibition signal)
        #   - attn S2 -> S1 (duplicate token signal)
        #   - attn pos -> pos-1 average (previous token signal)

        head_metrics = {}

        for layer in range(n_layers):
            for head in range(n_heads):
                end_to_io, end_to_s1, end_to_s2 = [], [], []
                s2_to_s1 = []
                prev_token_scores = []

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

                    for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                        # END -> name positions
                        if io_pos > 0:
                            end_to_io.append(attn[i, head, final_pos, io_pos].item())
                        if s1_pos > 0:
                            end_to_s1.append(attn[i, head, final_pos, s1_pos].item())
                        if s2_pos > 0:
                            end_to_s2.append(attn[i, head, final_pos, s2_pos].item())
                        # S2 -> S1
                        if s2_pos > 0 and s1_pos > 0:
                            s2_to_s1.append(attn[i, head, s2_pos, s1_pos].item())
                        # Previous token (average)
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

                head_name = "L%dH%d" % (layer, head)
                metrics = {
                    "end_to_IO": round(float(np.mean(end_to_io)), 4) if end_to_io else 0,
                    "end_to_S1": round(float(np.mean(end_to_s1)), 4) if end_to_s1 else 0,
                    "end_to_S2": round(float(np.mean(end_to_s2)), 4) if end_to_s2 else 0,
                    "s2_to_s1": round(float(np.mean(s2_to_s1)), 4) if s2_to_s1 else 0,
                    "prev_token": round(float(np.mean(prev_token_scores)), 4) if prev_token_scores else 0,
                }

                # Wang et al. style classification
                roles = []
                if metrics["end_to_IO"] > 0.1:
                    roles.append("name_mover")
                if metrics["end_to_S2"] > 0.1:
                    roles.append("s_inhibition")
                if metrics["s2_to_s1"] > 0.2:
                    roles.append("duplicate_token")
                if metrics["prev_token"] > 0.3:
                    roles.append("previous_token")

                metrics["roles"] = roles
                head_metrics[head_name] = metrics

            if (layer + 1) % 4 == 0:
                print("    Layer %d/%d" % (layer + 1, n_layers))

        # Count by role
        counts = {}
        top_heads = {}
        for role in ["name_mover", "s_inhibition", "duplicate_token", "previous_token"]:
            matching = [(h, m) for h, m in head_metrics.items() if role in m["roles"]]
            counts[role] = len(matching)
            if role == "name_mover":
                matching.sort(key=lambda x: -x[1]["end_to_IO"])
            elif role == "s_inhibition":
                matching.sort(key=lambda x: -x[1]["end_to_S2"])
            elif role == "duplicate_token":
                matching.sort(key=lambda x: -x[1]["s2_to_s1"])
            elif role == "previous_token":
                matching.sort(key=lambda x: -x[1]["prev_token"])
            top_heads[role] = [{"head": h, "score": round(m[{
                "name_mover": "end_to_IO",
                "s_inhibition": "end_to_S2",
                "duplicate_token": "s2_to_s1",
                "previous_token": "prev_token",
            }[role]], 4)} for h, m in matching[:5]]

        print("    Wang et al. counts: NM=%d, S-inh=%d, DupTok=%d, PrevTok=%d" % (
            counts.get("name_mover", 0), counts.get("s_inhibition", 0),
            counts.get("duplicate_token", 0), counts.get("previous_token", 0)))

        results["wang_classification"][step_key] = {
            "counts": counts,
            "top_heads": top_heads,
            "all_heads": head_metrics,
        }
        save_results(results)

        del model
        empty_cache()

    print("\nPART 3 COMPLETE")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  FINAL THREE EXPERIMENTS")
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()

    try:
        run_part1(results)
    except Exception as e:
        print("PART 1 FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_part2(results)
    except Exception as e:
        print("PART 2 FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_part3(results)
    except Exception as e:
        print("PART 3 FAILED: %s" % str(e))
        traceback.print_exc()

    elapsed = time.time() - t0
    save_results(results)
    print("\n" + "=" * 60)
    print("  ALL DONE. Time: %.0fs (%.1f hours)" % (elapsed, elapsed / 3600))
    print("=" * 60)

if __name__ == "__main__":
    main()
