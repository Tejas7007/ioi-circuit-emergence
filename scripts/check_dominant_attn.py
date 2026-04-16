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

configs = [
    ('EleutherAI/pythia-410m-deduped', [(5, 2), (12, 12), (4, 6)]),
    ('EleutherAI/pythia-1b-deduped', [(8, 7), (11, 0)]),
]

results = {}

for model_name, heads in configs:
    print("=== %s ===" % model_name)
    clear_cache()
    model = HookedTransformer.from_pretrained(model_name,
        center_writing_weights=True, center_unembed=True, fold_ln=True,
        device='cuda', checkpoint_value=143000)

    model_results = {}
    for layer, head in heads:
        attn_to_io = []
        attn_to_s1 = []
        attn_to_s2 = []

        for tmpl in TEMPLATES[:10]:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                            symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).cuda()
            io_ids = torch.tensor(ds.io_token_ids, device='cuda')
            s_ids = torch.tensor(ds.s_token_ids, device='cuda')

            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
            attn = cache["blocks.%d.attn.hook_pattern" % layer]
            final_pos = tokens.shape[1] - 1

            for i in range(tokens.shape[0]):
                io_tok = io_ids[i].item()
                s_tok = s_ids[i].item()
                io_pos = -1
                s1_pos = -1
                s2_pos = -1
                s_count = 0
                for j in range(1, tokens.shape[1]):
                    if tokens[i, j].item() == io_tok and io_pos == -1:
                        io_pos = j
                    if tokens[i, j].item() == s_tok:
                        s_count += 1
                        if s_count == 1: s1_pos = j
                        elif s_count == 2: s2_pos = j
                if io_pos > 0:
                    attn_to_io.append(attn[i, head, final_pos, io_pos].item())
                if s1_pos > 0:
                    attn_to_s1.append(attn[i, head, final_pos, s1_pos].item())
                if s2_pos > 0:
                    attn_to_s2.append(attn[i, head, final_pos, s2_pos].item())

            del cache
            torch.cuda.empty_cache()

        head_name = "L%dH%d" % (layer, head)
        model_results[head_name] = {
            "attn_to_IO": round(float(np.mean(attn_to_io)), 4),
            "attn_to_S1": round(float(np.mean(attn_to_s1)), 4),
            "attn_to_S2": round(float(np.mean(attn_to_s2)), 4),
        }
        print("  %s: IO=%.4f S1=%.4f S2=%.4f" % (
            head_name,
            float(np.mean(attn_to_io)),
            float(np.mean(attn_to_s1)),
            float(np.mean(attn_to_s2))))

    m_key = model_name.split("/")[1]
    results[m_key] = model_results
    del model
    torch.cuda.empty_cache()

mega_path = "results/mega_experiments.json"
with open(mega_path) as f:
    mega = json.load(f)
mega["exp_f_dominant_head_attn"] = results
with open(mega_path, 'w') as f:
    json.dump(mega, f, indent=2)
print("\nSaved. DONE.")
