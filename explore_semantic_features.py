"""
explore_semantic_features.py
============================
Samples Mamba features from the "Semantic Grey Matter" zone (0.28 - 0.35 similarity)
and extracts the Top 4 contexts for each to identify the underlying concept.
"""

import json
import random
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
TOKENIZER_ID = "EleutherAI/pythia-70m-deduped"

BATCH_SIZE = 16
SEARCH_CHUNKS = 1024  # Scanning ~1 million tokens
TOP_K_CONTEXTS = 4    # How many examples to extract per feature
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sample_grey_matter_features(json_path, min_sim=0.28, max_sim=0.35, sample_size=4):
    """Pulls a random sample of features from our target similarity zone."""
    with open(json_path, "r") as f:
        all_features = json.load(f)
        
    valid_features = [
        f for f in all_features 
        if min_sim <= f["max_jaccard_similarity"] <= max_sim
        and f["total_activations"] > 100 # Ignore dead or ultra-rare features
    ]
    
    print(f"Found {len(valid_features):,} features in the {min_sim}-{max_sim} range.")
    sampled = random.sample(valid_features, min(sample_size, len(valid_features)))
    
    # Sort them so we investigate the lowest similarity ones first
    sampled.sort(key=lambda x: x["max_jaccard_similarity"])
    return sampled

def main():
    sampled_features = sample_grey_matter_features(JSON_PATH)
    fids = [f["mamba_feature_id"] for f in sampled_features]
    
    print(f"\nInvestigating Features: {fids}")
    for f in sampled_features:
        print(f"  - Feature {f['mamba_feature_id']}: {f['max_jaccard_similarity']} similarity")

    print("\nLoading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = HookedMamba.from_pretrained(MAMBA_MODEL_ID, device=DEVICE)
    model.eval()
    
    sae_dir = str(Path(MAMBA_SAE_PATH).parent)
    sae = SAE.load_from_disk(sae_dir, device=DEVICE)
    sae.eval()

    chunks_np = np.load(DATA_PATH, mmap_mode="r")
    
    # Dictionary to track the top K activations: {fid: [(value, global_batch_idx, seq_idx)]}
    top_activations = {fid: [] for fid in fids}
    hook_name = f"blocks.12.hook_resid_post"

    print(f"\nScanning the first {SEARCH_CHUNKS} chunks for feature triggers...")
    
    for i in range(0, SEARCH_CHUNKS, BATCH_SIZE):
        batch = chunks_np[i : i + BATCH_SIZE]
        tokens = torch.tensor(batch, dtype=torch.long, device=DEVICE)

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            resid = cache[hook_name]          
            sae_acts = sae.encode(resid) # Shape: [B, S, d_sae]

            for fid in fids:
                feature_acts = sae_acts[:, :, fid]
                
                # Get the highest values in this specific batch
                vals, indices = torch.topk(feature_acts.flatten(), min(TOP_K_CONTEXTS, feature_acts.numel()))
                
                for val, idx in zip(vals.cpu().tolist(), indices.cpu().tolist()):
                    if val > 1.0: # We only care if the feature actually fired meaningfully
                        b_idx = idx // feature_acts.shape[1]
                        s_idx = idx % feature_acts.shape[1]
                        
                        global_batch_idx = i + b_idx
                        top_activations[fid].append((val, global_batch_idx, s_idx))
                
                # Sort the running list descending and keep only the Top K globally
                top_activations[fid].sort(key=lambda x: x[0], reverse=True)
                top_activations[fid] = top_activations[fid][:TOP_K_CONTEXTS]

    print("\n" + "="*60)
    print("GREY MATTER SEMANTIC DECODING COMPLETE")
    print("="*60)
    
    # Now that we found the coordinates, fetch the text from the array and print it
    for feature in sampled_features:
        fid = feature["mamba_feature_id"]
        sim = feature["max_jaccard_similarity"]
        
        print(f"\nFeature ID: {fid} | Similarity to Pythia: {sim}")
        print("-" * 60)
        
        acts = top_activations[fid]
        if not acts:
            print("  [No strong activations found in this snippet of the dataset]")
            continue
            
        for rank, (val, global_b, s_idx) in enumerate(acts):
            # Grab the specific token sequence from our numpy array
            seq_tokens = chunks_np[global_b]
            
            # Create a 30-token window around the trigger
            start_idx = max(0, s_idx - 15)
            end_idx = min(len(seq_tokens), s_idx + 5)
            
            prefix = tokenizer.decode(seq_tokens[start_idx:s_idx])
            trigger = tokenizer.decode(seq_tokens[s_idx:s_idx+1])
            suffix = tokenizer.decode(seq_tokens[s_idx+1:end_idx])
            
            print(f"  #{rank+1} (Strength: {val:.2f})")
            print(f"  ... {prefix} >>> {trigger.upper()} <<< {suffix} ...\n")

if __name__ == "__main__":
    main()