import torch, json, os, shutil, time, numpy as np
from transformer_lens import HookedTransformer
try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except:
    from src.circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

templates = ALL_TEMPLATES[:15]
ppt, seed = 20, 42

def clear_cache():
    cache_dir = '/workspace/.hf_home/hub'
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith('models--'):
                shutil.rmtree(os.path.join(cache_dir, d), ignore_errors=True)

def get_ioi_accuracy(model, templates, ppt, seed, hooks=[]):
    all_correct, all_lds = [], []
    for tmpl in templates:
        ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl], symmetric=True, seed=seed)
        tokens = model.to_tokens(ds.prompts).cuda()
        io_ids = torch.tensor(ds.io_token_ids, device='cuda')
        s_ids = torch.tensor(ds.s_token_ids, device='cuda')
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
        last = logits[:, -1, :]
        ld = last[torch.arange(len(io_ids)), io_ids] - last[torch.arange(len(s_ids)), s_ids]
        all_correct.append((ld > 0).float().detach().cpu())
        all_lds.append(ld.detach().cpu())
    return torch.cat(all_correct).mean().item(), torch.cat(all_lds).mean().item()

###############################################################
# PART A: ABLATION EXPERIMENTS ON 410M AND 1B
###############################################################

print("=" * 60)
print("  PART A: CROSS-SCALE ABLATION EXPERIMENTS")
print("=" * 60)

configs = [
    {
        'model': 'EleutherAI/pythia-410m-deduped',
        'steps': [1000, 2000, 3000],
        'early_nms': [(5,11)],
        'step3000_top': (5,2),
    },
    {
        'model': 'EleutherAI/pythia-1b-deduped',
        'steps': [1000, 2000, 3000],
        'early_nms': [(0,0),(0,4),(0,5),(0,6),(1,3)],
        'step3000_top': (8,7),
    },
]

all_ablation = {}

for cfg in configs:
    model_name = cfg['model']
    print("\n" + "=" * 60)
    print("  ABLATION: " + model_name)
    print("=" * 60)
    
    for step in cfg['steps']:
        clear_cache()
        print("\n--- Step %d ---" % step)
        
        try:
            model = HookedTransformer.from_pretrained(model_name,
                center_writing_weights=True, center_unembed=True, fold_ln=True,
                device='cuda', checkpoint_value=step)
        except Exception as e:
            print("  FAILED: %s" % str(e))
            continue
        
        base_acc, base_ld = get_ioi_accuracy(model, templates, ppt, seed)
        print("  BASELINE: acc=%.3f, LD=%+.4f" % (base_acc, base_ld))
        
        results = {"baseline_acc": base_acc, "baseline_ld": base_ld, "ablations": {}}
        
        for layer, head in cfg['early_nms']:
            def hook_fn(value, hook, h=head):
                value[:, :, h, :] = 0.0
                return value
            hook = ("blocks.%d.attn.hook_z" % layer, hook_fn)
            acc, ld = get_ioi_accuracy(model, templates, ppt, seed, hooks=[hook])
            diff = acc - base_acc
            if diff > 0.01:
                direction = "HELPS"
            elif diff < -0.01:
                direction = "HURTS"
            else:
                direction = "NEUTRAL"
            print("  Ablate L%dH%d: acc=%.3f (%+.3f) %s" % (layer, head, acc, diff, direction))
            results["ablations"]["L%dH%d" % (layer, head)] = {"acc": acc, "diff": diff, "effect": direction}
        
        if step >= 3000:
            layer, head = cfg['step3000_top']
            def hook_fn(value, hook, h=head):
                value[:, :, h, :] = 0.0
                return value
            hook = ("blocks.%d.attn.hook_z" % layer, hook_fn)
            acc, ld = get_ioi_accuracy(model, templates, ppt, seed, hooks=[hook])
            diff = acc - base_acc
            print("  Ablate L%dH%d (dominant): acc=%.3f (%+.3f)" % (layer, head, acc, diff))
            results["ablations"]["L%dH%d_dominant" % (layer, head)] = {"acc": acc, "diff": diff}
        
        if len(cfg['early_nms']) > 1:
            hooks = []
            for layer, head in cfg['early_nms']:
                def make_hook(h):
                    def hook_fn(value, hook):
                        value[:, :, h, :] = 0.0
                        return value
                    return hook_fn
                hooks.append(("blocks.%d.attn.hook_z" % layer, make_hook(head)))
            acc, ld = get_ioi_accuracy(model, templates, ppt, seed, hooks=hooks)
            diff = acc - base_acc
            print("  Ablate ALL early NMs: acc=%.3f (%+.3f)" % (acc, diff))
            results["ablations"]["ALL_early"] = {"acc": acc, "diff": diff}
        
        key = "%s_step%d" % (model_name, step)
        all_ablation[key] = results
        
        del model
        torch.cuda.empty_cache()
    
    with open("results/ablation_cross_scale.json", "w") as f:
        json.dump(all_ablation, f, indent=2)
    print("  [saved ablation_cross_scale.json]")

print("\nPART A COMPLETE")

###############################################################
# PART B: INDUCTION HEAD EMERGENCE
###############################################################

print("\n" + "=" * 60)
print("  PART B: INDUCTION HEAD EMERGENCE")
print("=" * 60)

CHECKPOINTS = [0, 512, 1000, 2000, 3000, 4000, 5000, 8000,
               10000, 16000, 33000, 66000, 143000]

MODELS = [
    'EleutherAI/pythia-160m-deduped',
    'EleutherAI/pythia-410m-deduped',
    'EleutherAI/pythia-1b-deduped',
]

def measure_induction_score(model, seq_len=200, n_batches=4, batch_size=4):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    half = seq_len // 2
    
    all_scores = torch.zeros(n_layers, n_heads)
    
    for _ in range(n_batches):
        first_half = torch.randint(1000, 50000, (batch_size, half))
        tokens = torch.cat([first_half, first_half], dim=1).cuda()
        bos = torch.full((batch_size, 1), model.tokenizer.bos_token_id, dtype=torch.long).cuda()
        tokens = torch.cat([bos, tokens], dim=1)
        
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        
        for layer in range(n_layers):
            attn = cache["blocks.%d.attn.hook_pattern" % layer]
            
            for i in range(1, half):
                dest_pos = half + 1 + i
                src_pos = 1 + i
                if dest_pos < attn.shape[2] and src_pos < attn.shape[3]:
                    all_scores[layer] += attn[:, :, dest_pos, src_pos].mean(dim=0).detach().cpu()
        
        del cache
        torch.cuda.empty_cache()
    
    all_scores /= (n_batches * (half - 1))
    return all_scores

induction_results = {}

for model_name in MODELS:
    print("\n" + "=" * 60)
    print("  INDUCTION: " + model_name)
    print("=" * 60)
    
    model_results = []
    
    for step in CHECKPOINTS:
        clear_cache()
        print("\n--- Step %d ---" % step)
        
        try:
            model = HookedTransformer.from_pretrained(model_name,
                center_writing_weights=True, center_unembed=True, fold_ln=True,
                device='cuda', checkpoint_value=step)
        except Exception as e:
            print("  FAILED: %s" % str(e))
            continue
        
        scores = measure_induction_score(model, seq_len=200, n_batches=4, batch_size=4)
        
        flat = scores.flatten()
        top_indices = flat.argsort(descending=True)[:5]
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // model.cfg.n_heads
            head = idx.item() % model.cfg.n_heads
            score = flat[idx].item()
            top_heads.append({"head": "L%dH%d" % (layer, head), "score": round(score, 4)})
        
        n_strong = (scores > 0.1).sum().item()
        n_medium = (scores > 0.05).sum().item()
        max_score = scores.max().item()
        
        result = {
            "step": step,
            "max_induction_score": round(max_score, 4),
            "n_strong_induction": n_strong,
            "n_medium_induction": n_medium,
            "top_5_heads": top_heads,
        }
        model_results.append(result)
        
        top_str = ", ".join([h["head"] + "(%.3f)" % h["score"] for h in top_heads[:3]])
        print("  Max: %.4f, Strong: %d, Medium: %d" % (max_score, n_strong, n_medium))
        print("  Top: %s" % top_str)
        
        del model
        torch.cuda.empty_cache()
    
    induction_results[model_name] = model_results
    
    with open("results/induction_emergence.json", "w") as f:
        json.dump(induction_results, f, indent=2)
    print("  [saved induction_emergence.json]")

print("\n" + "=" * 60)
print("  ALL EXPERIMENTS COMPLETE")
print("=" * 60)
