import torch, json, os, shutil
import numpy as np
from transformer_lens import HookedTransformer
try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except:
    from src.circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

def clear_cache():
    cache_dir = '/workspace/.hf_home/hub'
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith('models--'):
                shutil.rmtree(os.path.join(cache_dir, d), ignore_errors=True)

TEMPLATES = ALL_TEMPLATES[:15]
PPT, SEED = 20, 42
EARLY_NMS = [(0, 5), (0, 6), (0, 10), (8, 9)]

results = {}

for step in [1000, 3000, 143000]:
    print("--- Step %d ---" % step)
    clear_cache()
    model = HookedTransformer.from_pretrained('EleutherAI/pythia-160m-deduped',
        center_writing_weights=True, center_unembed=True, fold_ln=True,
        device='cuda', checkpoint_value=step)

    W_U = model.W_U
    W_O = model.W_O

    step_results = {}

    for layer, head in EARLY_NMS:
        io_projections = []
        s_projections = []

        for tmpl in TEMPLATES[:10]:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                            symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).cuda()
            io_ids = torch.tensor(ds.io_token_ids, device='cuda')
            s_ids = torch.tensor(ds.s_token_ids, device='cuda')

            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
            z = cache["blocks.%d.attn.hook_z" % layer][:, -1, head, :]
            head_out = z @ W_O[layer, head]

            for i in range(len(io_ids)):
                io_dir = W_U[:, io_ids[i].item()]
                s_dir = W_U[:, s_ids[i].item()]
                io_projections.append(torch.dot(head_out[i], io_dir).item())
                s_projections.append(torch.dot(head_out[i], s_dir).item())

            del cache
            torch.cuda.empty_cache()

        mean_io = float(np.mean(io_projections))
        mean_s = float(np.mean(s_projections))
        diff = mean_io - mean_s
        head_name = "L%dH%d" % (layer, head)
        step_results[head_name] = {
            "mean_io_projection": round(mean_io, 4),
            "mean_s_projection": round(mean_s, 4),
            "io_minus_s": round(diff, 4),
            "promotes": "IO" if diff > 0 else "S",
        }
        print("  %s: IO=%.4f, S=%.4f, diff=%.4f -> promotes %s" % (
            head_name, mean_io, mean_s, diff, "IO" if diff > 0 else "S"))

    results["step_%d" % step] = step_results
    del model
    torch.cuda.empty_cache()

mega_path = "results/mega_experiments.json"
if os.path.exists(mega_path):
    with open(mega_path) as f:
        mega = json.load(f)
else:
    mega = {}
mega["exp_a_output_projection"] = results
with open(mega_path, 'w') as f:
    json.dump(mega, f, indent=2)
print("\nSaved to %s" % mega_path)
