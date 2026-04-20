"""
read_alien_features.py
======================
Finds the top 3 most "alien" Mamba features (lowest Jaccard similarity) 
and extracts the exact text snippets from Wikipedia that cause them to fire.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from mamba_lens import HookedMamba
from sae_lens import SAE

# ── Config ────────────────────────────────────────────────────────────────────
JSON_PATH = Path("all_mapped_features.json")
DATA_PATH = Path("wiki_10m_chunks.npy")
MAMBA_SAE_PATH = "sae_weights.safetensors" 
MAMBA_MODEL_ID = "state-spaces/mamba-130m"
TOKENIZER_ID = "EleutherAI/pythia-70m-deduped" # Assuming standard GPT-NeoX tokenizer was used

BATCH_SIZE = 16
# We only need to search a fraction of the dataset to find good examples
SEARCH_CHUNKS = 1024 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_top_alien_features(json_path, top_k=3):
    with open(json_path, "r") as f:
        features = json.load(f)
    # The JSON is already sorted lowest to highest similarity!
    return [f["mamba_feature_id"] for f in features[:top_k]]

def main():
    alien_feature_ids = load_top_alien_features(JSON_PATH, top_k=3)
    print(f"Investigating the most alien Mamba features: {alien_feature_ids}")

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = HookedMamba.from_pretrained(MAMBA_MODEL_ID, device=DEVICE)
    model.eval()
    
    # Load SAE using the V6 API
    sae_dir = str(Path(MAMBA_SAE_PATH).parent)
    sae = SAE.load_from_disk(sae_dir, device=DEVICE)
    sae.eval()

    print("Loading dataset snippet...")
    chunks_np = np.load(DATA_PATH, mmap_mode="r")
    
    # Data structures to track the max activating text for each feature
    max_acts = {fid: 0.0 for fid in alien_feature_ids}
    best_contexts = {fid: "" for fid in alien_feature_ids}

    print(f"Scanning the first {SEARCH_CHUNKS} chunks for feature triggers...")
    hook_name = f"blocks.12.hook_resid_post"

    for i in range(0, SEARCH_CHUNKS, BATCH_SIZE):
        batch = chunks_np[i : i + BATCH_SIZE]
        tokens = torch.tensor(batch, dtype=torch.long, device=DEVICE)

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            resid = cache[hook_name]          
            sae_acts = sae.encode(resid) # Shape: [Batch, Seq, d_sae]

            # Check our specific alien features
            for fid in alien_feature_ids:
                # Get the activations for just this feature across the batch
                feature_acts = sae_acts[:, :, fid]
                batch_max_val = feature_acts.max().item()

                if batch_max_val > max_acts[fid]:
                    max_acts[fid] = batch_max_val
                    
                    # Find exactly which token in which sequence caused the max fire
                    b_idx, s_idx = torch.where(feature_acts == batch_max_val)
                    b_idx, s_idx = b_idx[0].item(), s_idx[0].item()

                    # Grab a window of text around the firing token (e.g., 10 tokens before and after)
                    start_idx = max(0, s_idx - 15)
                    end_idx = min(tokens.shape[1], s_idx + 5)
                    
                    context_tokens = tokens[b_idx, start_idx:end_idx]
                    firing_token = tokens[b_idx, s_idx:s_idx+1]
                    
                    # Decode to English
                    prefix = tokenizer.decode(tokens[b_idx, start_idx:s_idx])
                    trigger = tokenizer.decode(firing_token)
                    suffix = tokenizer.decode(tokens[b_idx, s_idx+1:end_idx])
                    
                    best_contexts[fid] = f"{prefix} >>> {trigger.upper()} <<< {suffix}"

    print("\n" + "="*50)
    print("👽 ALIEN FEATURE DECODING COMPLETE 👽")
    print("="*50)
    
    for fid in alien_feature_ids:
        print(f"\nFeature ID: {fid}")
        print(f"Max Activation Strength: {max_acts[fid]:.2f}")
        print(f"Trigger Context:")
        print(f"... {best_contexts[fid]} ...")
        print("-" * 50)

if __name__ == "__main__":
    main()