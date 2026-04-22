"""
PolyPythias IOI Analysis
========================
Tests whether the IOI dip appears across different Pythia-160M
training variants with different seeds and data orderings.

Models tested:
  - pythia-160m-seed{1,3,5}: different full random seeds
  - pythia-160m-data-seed{1,2,3}: only data ordering changed, weight init fixed
  - pythia-160m-weight-seed{1,2,3}: only weight init changed, data order fixed

Saves to results/polypythias_ioi.json
"""

import torch
import json
import os
import time
import traceback
import shutil
import numpy as np

from transformer_lens import HookedTransformer

try:
    import sys
    sys.path.insert(0, os.path.expanduser('~/MLP-Paper-Cole/src'))
    sys.path.insert(0, '/workspace/MLP-Paper-Cole/src')
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except ImportError:
    print("ERROR: Cannot import IOIDataset")
    raise

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MODELS = [
    # Different full seeds
    ("EleutherAI/pythia-160m-seed1", "seed1"),
    ("EleutherAI/pythia-160m-seed3", "seed3"),
    ("EleutherAI/pythia-160m-seed5", "seed5"),
    # Only data ordering changed
    ("EleutherAI/pythia-160m-data-seed1", "data-seed1"),
    ("EleutherAI/pythia-160m-data-seed2", "data-seed2"),
    ("EleutherAI/pythia-160m-data-seed3", "data-seed3"),
    # Only weight init changed
    ("EleutherAI/pythia-160m-weight-seed1", "weight-seed1"),
    ("EleutherAI/pythia-160m-weight-seed2", "weight-seed2"),
    ("EleutherAI/pythia-160m-weight-seed3", "weight-seed3"),
]

CHECKPOINTS = [0, 512, 1000, 2000, 3000, 5000, 8000, 10000, 33000, 143000]

TEMPLATES = ALL_TEMPLATES[:15]
PPT = 20
SEED = 42

RESULTS_FILE = "results/polypythias_ioi.json"


def clear_cache():
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(cache_dir, "hub")
    if os.path.exists(hub_dir):
        for d in os.listdir(hub_dir):
            if d.startswith("models--"):
                shutil.rmtree(os.path.join(hub_dir, d), ignore_errors=True)


def empty_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    print("=" * 60)
    print("  POLYPYTHIAS IOI ANALYSIS")
    print("  %d models x %d checkpoints = %d runs" % (
        len(MODELS), len(CHECKPOINTS), len(MODELS) * len(CHECKPOINTS)))
    print("  Device: %s" % DEVICE)
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print("  Loaded existing results")
    else:
        results = {}

    t0 = time.time()

    for model_name, label in MODELS:
        if label not in results:
            results[label] = {"model": model_name, "checkpoints": {}}

        print("\n" + "=" * 40)
        print("  %s (%s)" % (label, model_name))
        print("=" * 40)

        for step in CHECKPOINTS:
            step_key = "step_%d" % step

            if step_key in results[label]["checkpoints"]:
                print("  Step %d already done, skipping" % step)
                continue

            print("\n  --- Step %d ---" % step)

            try:
                clear_cache()
                model = HookedTransformer.from_pretrained(
                    model_name,
                    center_writing_weights=True,
                    center_unembed=True,
                    fold_ln=True,
                    device=DEVICE,
                    checkpoint_value=step,
                )
            except Exception as e:
                print("    FAILED to load: %s" % str(e)[:100])
                continue

            all_lds = []
            io_ranks = []
            s_ranks = []
            io_probs = []
            s_probs = []
            top1_is_io = 0
            top1_is_s = 0
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
                        io_rank = (sorted_idx == io_ids[i]).nonzero(as_tuple=True)[0].item()
                        s_rank = (sorted_idx == s_ids[i]).nonzero(as_tuple=True)[0].item()
                        io_ranks.append(io_rank)
                        s_ranks.append(s_rank)

                        io_probs.append(probs[i, io_ids[i]].item())
                        s_probs.append(probs[i, s_ids[i]].item())

                        top_tok = sorted_idx[0].item()
                        if top_tok == io_ids[i].item():
                            top1_is_io += 1
                        elif top_tok == s_ids[i].item():
                            top1_is_s += 1
                        else:
                            decoded = model.tokenizer.decode([top_tok]).strip()
                            top1_other.append(decoded)

                except Exception as e:
                    continue

            if total == 0:
                del model
                empty_cache()
                continue

            lds = np.array(all_lds)
            accuracy = float((lds > 0).mean())

            from collections import Counter
            top_others = Counter(top1_other).most_common(3)

            step_result = {
                "accuracy": round(accuracy, 4),
                "mean_ld": round(float(lds.mean()), 4),
                "pct_s_preferred": round(float((lds < 0).mean()), 4),
                "median_io_rank": float(np.median(io_ranks)),
                "median_s_rank": float(np.median(s_ranks)),
                "mean_io_prob": round(float(np.mean(io_probs)), 6),
                "mean_s_prob": round(float(np.mean(s_probs)), 6),
                "pct_io_top1": round(top1_is_io / total, 4),
                "n_examples": total,
                "top_other": [{"token": t, "count": c} for t, c in top_others],
            }

            results[label]["checkpoints"][step_key] = step_result
            save_results(results)

            print("    Acc=%.3f, LD=%.4f, pct_S=%.1f%%, IO_rank=%d, IO_prob=%.4f%%" % (
                accuracy, lds.mean(), (lds < 0).mean() * 100,
                np.median(io_ranks), np.mean(io_probs) * 100))

            del model
            empty_cache()

    # Print summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("%15s" % "Step", end="")
    for _, label in MODELS:
        print("  %10s" % label, end="")
    print()

    for step in CHECKPOINTS:
        step_key = "step_%d" % step
        print("%15s" % step_key, end="")
        for _, label in MODELS:
            if label in results and step_key in results[label].get("checkpoints", {}):
                acc = results[label]["checkpoints"][step_key]["accuracy"]
                print("  %9.1f%%" % (acc * 100), end="")
            else:
                print("  %10s" % "---", end="")
        print()

    elapsed = time.time() - t0
    results["total_time_seconds"] = round(elapsed, 1)
    save_results(results)

    print("\n  Total time: %.0f seconds (%.1f hours)" % (elapsed, elapsed / 3600))
    print("  Results: %s" % RESULTS_FILE)
    print("  DONE.")


if __name__ == "__main__":
    main()
