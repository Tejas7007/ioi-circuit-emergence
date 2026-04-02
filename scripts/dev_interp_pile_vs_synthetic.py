#!/usr/bin/env python3
import argparse, json, time, os, torch, numpy as np
try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except ImportError:
    from src.circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
from transformer_lens import HookedTransformer

CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1000, 2000, 3000, 4000, 5000, 6000, 8000,
               10000, 16000, 33000, 66000, 100000, 143000]

def eval_synthetic_ioi(model, templates, ppt, seed, device):
    all_lds, all_correct = [], []
    for tmpl in templates[:15]:
        ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl], symmetric=True, seed=seed)
        tokens = model.to_tokens(ds.prompts)
        if device == "cuda": tokens = tokens.cuda()
        io_ids = torch.tensor(ds.io_token_ids, device=tokens.device)
        s_ids = torch.tensor(ds.s_token_ids, device=tokens.device)
        logits = model(tokens)
        last = logits[:, -1, :]
        ld = last[torch.arange(len(io_ids)), io_ids] - last[torch.arange(len(s_ids)), s_ids]
        all_lds.append(ld.detach().cpu())
        all_correct.append((ld > 0).float().detach().cpu())
    all_lds = torch.cat(all_lds)
    all_correct = torch.cat(all_correct)
    return {"mean_ld": all_lds.mean().item(), "accuracy": all_correct.mean().item(), "n": len(all_lds)}

def eval_pile_ioi(model, pile_prompts, device):
    if not pile_prompts: return {"mean_ld": 0, "accuracy": 0, "n": 0, "n_skipped": 0}
    all_lds, all_correct = [], []
    n_skipped = 0
    for p in pile_prompts:
        prompt, io_name, s_name = p["prompt"], p["io_name"], p["s_name"]
        try:
            io_tokens = model.to_tokens(io_name, prepend_bos=False)[0]
            s_tokens = model.to_tokens(s_name, prepend_bos=False)[0]
            if len(io_tokens) != 1 or len(s_tokens) != 1:
                n_skipped += 1
                continue
            io_id, s_id = io_tokens[0].item(), s_tokens[0].item()
        except:
            n_skipped += 1
            continue
        tokens = model.to_tokens(prompt)
        if device == "cuda": tokens = tokens.cuda()
        try:
            logits = model(tokens)
            last_logits = logits[0, -1, :]
            ld = last_logits[io_id].item() - last_logits[s_id].item()
            all_lds.append(ld)
            all_correct.append(1.0 if ld > 0 else 0.0)
        except:
            n_skipped += 1
    if not all_lds: return {"mean_ld": 0, "accuracy": 0, "n": 0, "n_skipped": n_skipped}
    return {"mean_ld": np.mean(all_lds), "accuracy": np.mean(all_correct), "n": len(all_lds), "n_skipped": n_skipped}

def main(args):
    print(f"\n{'='*60}")
    print(f"  Dev Interp: Pile vs Synthetic IOI")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")
    with open(args.pile_prompts) as f:
        pile_prompts = json.load(f)
    print(f"  Loaded {len(pile_prompts)} Pile IOI prompts")
    results = []
    for step in CHECKPOINTS:
        print(f"\n--- Step {step} ---")
        try:
            model = HookedTransformer.from_pretrained(args.model, center_writing_weights=True,
                center_unembed=True, fold_ln=True, device=args.device, checkpoint_value=step)
        except Exception as e:
            print(f"  FAILED ({e})")
            continue
        synth = eval_synthetic_ioi(model, ALL_TEMPLATES, args.ppt, args.seed, args.device)
        pile = eval_pile_ioi(model, pile_prompts, args.device)
        result = {"step": step, "synthetic": synth, "pile": pile}
        results.append(result)
        print(f"  Synthetic: LD={synth['mean_ld']:+.4f}, acc={synth['accuracy']:.3f} (n={synth['n']})")
        print(f"  Pile:      LD={pile['mean_ld']:+.4f}, acc={pile['accuracy']:.3f} (n={pile['n']})")
        gap = pile['accuracy'] - synth['accuracy']
        if abs(gap) > 0.05:
            direction = "Pile AHEAD" if gap > 0 else "Synthetic AHEAD"
            print(f"  >>> GAP: {gap:+.3f} ({direction})")
        del model
        torch.cuda.empty_cache()
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Pile vs Synthetic")
    print(f"{'='*60}")
    print(f"  {'Step':>8} {'Syn LD':>10} {'Syn Acc':>10} {'Pile LD':>10} {'Pile Acc':>10} {'Gap':>8}")
    for r in results:
        s, p = r["synthetic"], r["pile"]
        gap = p["accuracy"] - s["accuracy"]
        marker = " <--" if abs(gap) > 0.1 else ""
        print(f"  {r['step']:>8} {s['mean_ld']:>+9.3f} {s['accuracy']:>9.1%} {p['mean_ld']:>+9.3f} {p['accuracy']:>9.1%} {gap:>+7.3f}{marker}")
    pile_70 = next((r["step"] for r in results if r["pile"]["accuracy"] > 0.7), None)
    synth_70 = next((r["step"] for r in results if r["synthetic"]["accuracy"] > 0.7), None)
    print(f"\n  Pile reaches 70% at step: {pile_70}")
    print(f"  Synthetic reaches 70% at step: {synth_70}")
    if pile_70 and synth_70:
        if pile_70 < synth_70: print(f"  GROKKING SIGNAL: Pile leads by {synth_70 - pile_70} steps")
        elif pile_70 == synth_70: print(f"  SIMULTANEOUS emergence")
        else: print(f"  REVERSE: Synthetic leads")
    m_safe = args.model.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"dev_interp_grokking_{m_safe}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "n_pile_prompts": len(pile_prompts), "results": results,
                   "pile_70_step": pile_70, "synth_70_step": synth_70}, f, indent=2)
    print(f"\n  Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pile-prompts", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ppt", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    t0 = time.time()
    main(args)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
