#!/usr/bin/env python3
import argparse, json, time, os, sys, torch
import numpy as np
from collections import defaultdict
try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except ImportError:
    from src.circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
from transformer_lens import HookedTransformer

CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1000, 2000, 3000, 4000, 5000, 6000, 8000,
               10000, 16000, 33000, 66000, 100000, 143000]
TAU = 0.02

def compute_ioi_metrics(model, templates, ppt, seed, device):
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
    return {"mean_ld": all_lds.mean().item(), "median_ld": all_lds.median().item(),
            "std_ld": all_lds.std().item(), "accuracy": all_correct.mean().item(), "n_prompts": len(all_lds)}

def scan_heads(model, templates, ppt, seed, device):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    base_lds, base_sls = [], []
    for tmpl in templates[:10]:
        ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl], symmetric=True, seed=seed)
        tokens = model.to_tokens(ds.prompts)
        if device == "cuda": tokens = tokens.cuda()
        io_ids = torch.tensor(ds.io_token_ids, device=tokens.device)
        s_ids = torch.tensor(ds.s_token_ids, device=tokens.device)
        logits = model(tokens)
        last = logits[:, -1, :]
        ld = last[torch.arange(len(io_ids)), io_ids] - last[torch.arange(len(s_ids)), s_ids]
        sl = last[torch.arange(len(s_ids)), s_ids]
        base_lds.append(ld.detach().cpu())
        base_sls.append(sl.detach().cpu())
    base_ld = torch.cat(base_lds).mean().item()
    base_sl = torch.cat(base_sls).mean().item()
    head_results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            def hook_fn(value, hook, h=head):
                value[:, :, h, :] = 0.0
                return value
            hook_name = f"blocks.{layer}.attn.hook_z"
            abl_lds, abl_sls = [], []
            for tmpl in templates[:10]:
                ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl], symmetric=True, seed=seed)
                tokens = model.to_tokens(ds.prompts)
                if device == "cuda": tokens = tokens.cuda()
                io_ids = torch.tensor(ds.io_token_ids, device=tokens.device)
                s_ids = torch.tensor(ds.s_token_ids, device=tokens.device)
                logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
                last = logits[:, -1, :]
                ld = last[torch.arange(len(io_ids)), io_ids] - last[torch.arange(len(s_ids)), s_ids]
                sl = last[torch.arange(len(s_ids)), s_ids]
                abl_lds.append(ld.detach().cpu())
                abl_sls.append(sl.detach().cpu())
            abl_ld = torch.cat(abl_lds).mean().item()
            abl_sl = torch.cat(abl_sls).mean().item()
            head_results.append({"layer": layer, "head": head, "component": f"L{layer}H{head}",
                                "delta_ioi": abl_ld - base_ld, "delta_anti": abl_sl - base_sl})
    return head_results, base_ld, base_sl

def classify_heads(head_results, tau):
    roles = {"name_movers": [], "other_ioi": [], "subject_promoters": [], "copy_suppression": [], "anti_ioi": []}
    for h in head_results:
        di, da = h["delta_ioi"], h["delta_anti"]
        if di < -tau and da > tau: roles["name_movers"].append(h)
        elif di < -tau: roles["other_ioi"].append(h)
        if di > tau and da < -tau: roles["subject_promoters"].append(h)
        if di > tau: roles["anti_ioi"].append(h)
        if da > tau: roles["copy_suppression"].append(h)
    return roles

def run_checkpoint(model_name, step, device, ppt, seed):
    try:
        model = HookedTransformer.from_pretrained(model_name, center_writing_weights=True,
            center_unembed=True, fold_ln=True, device=device, checkpoint_value=step)
    except Exception as e:
        print(f"  Step {step}: FAILED to load ({e})")
        return None
    perf = compute_ioi_metrics(model, ALL_TEMPLATES, ppt, seed, device)
    head_results, base_ld, base_sl = scan_heads(model, ALL_TEMPLATES, ppt, seed, device)
    roles = classify_heads(head_results, TAU)
    top_ioi = min(head_results, key=lambda x: x["delta_ioi"]) if head_results else None
    top_anti = max(head_results, key=lambda x: x["delta_ioi"]) if head_results else None
    result = {"step": step, "performance": perf,
        "n_name_movers": len(roles["name_movers"]), "n_other_ioi": len(roles["other_ioi"]),
        "n_subject_promoters": len(roles["subject_promoters"]),
        "n_copy_suppression": len(roles["copy_suppression"]),
        "n_anti_ioi": len(roles["anti_ioi"]),
        "n_total_ioi": len(roles["name_movers"]) + len(roles["other_ioi"]),
        "top_ioi_head": top_ioi["component"] if top_ioi else None,
        "top_ioi_delta": top_ioi["delta_ioi"] if top_ioi else None,
        "top_anti_head": top_anti["component"] if top_anti else None,
        "top_anti_delta": top_anti["delta_ioi"] if top_anti else None,
        "name_mover_heads": [(h["component"], h["delta_ioi"], h["delta_anti"]) for h in roles["name_movers"]]}
    del model
    torch.cuda.empty_cache()
    return result

def main(args):
    print(f"\n{'='*60}")
    print(f"  Dev Interp: {args.model}")
    print(f"{'='*60}")
    all_results = []
    for step in CHECKPOINTS:
        print(f"\n--- Step {step} ---")
        result = run_checkpoint(args.model, step, args.device, args.ppt, args.seed)
        if result is None: continue
        all_results.append(result)
        p = result["performance"]
        print(f"  Performance: LD={p['mean_ld']:+.4f}, accuracy={p['accuracy']:.3f}")
        print(f"  Components: NM={result['n_name_movers']}, other_IOI={result['n_other_ioi']}, SP={result['n_subject_promoters']}, CS={result['n_copy_suppression']}")
        print(f"  Top IOI head: {result['top_ioi_head']} ({result['top_ioi_delta']:+.4f})")
        if result['name_mover_heads']:
            print(f"  Name-movers: {', '.join(h[0] for h in result['name_mover_heads'][:5])}")
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  {'Step':>8} {'LD':>8} {'Acc':>6} {'NM':>4} {'IOI':>4} {'SP':>4} {'CS':>4} {'Top Head':<12}")
    for r in all_results:
        p = r["performance"]
        print(f"  {r['step']:>8} {p['mean_ld']:>+7.3f} {p['accuracy']:>5.1%} {r['n_name_movers']:>4} {r['n_total_ioi']:>4} {r['n_subject_promoters']:>4} {r['n_copy_suppression']:>4} {r['top_ioi_head'] or 'N/A':<12}")
    m_safe = args.model.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"dev_interp_{m_safe}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "tau": TAU, "checkpoints": CHECKPOINTS, "results": all_results}, f, indent=2)
    print(f"\n  Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ppt", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    t0 = time.time()
    main(args)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
