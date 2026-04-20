import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from mamba_lens import HookedMamba
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
JSON_PATH = Path("all_mapped_features.json")
DATA_PATH = Path("wiki_10m_chunks.npy")

TARGET_MAMBA_FEATURE = 4454
TOP_K_CONTEXTS = 5
SEARCH_CHUNKS = 1024
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mamba
MAMBA_MODEL_ID = "state-spaces/mamba-130m"
MAMBA_SAE_PATH = "sae_weights.safetensors"
MAMBA_HOOK = "blocks.12.hook_resid_post"

# Pythia
PYTHIA_MODEL_ID = "EleutherAI/pythia-70m-deduped"
PYTHIA_SAE_RELEASE = "pythia-70m-deduped-res-sm"
PYTHIA_SAE_ID = "blocks.3.hook_resid_post"
PYTHIA_HOOK = "blocks.3.hook_resid_post"


def get_pythia_twin(mamba_fid):
    with open(JSON_PATH, "r") as f:
        features = json.load(f)
    for feat in features:
        if feat["mamba_feature_id"] == mamba_fid:
            return feat["best_pythia_match_id"], feat["max_jaccard_similarity"]
    raise ValueError(f"Feature {mamba_fid} not found in JSON.")


def main():
    print(f"Looking up Pythia twin for Mamba Feature {TARGET_MAMBA_FEATURE}...")
    pythia_twin_id, sim_score = get_pythia_twin(TARGET_MAMBA_FEATURE)
    print(f"Found Twin! Pythia Feature {pythia_twin_id} (Jaccard Similarity: {sim_score})\n")

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PYTHIA_MODEL_ID)

    print("Loading Pythia and its SAE...")
    pythia_model = HookedTransformer.from_pretrained(PYTHIA_MODEL_ID, device=DEVICE)
    pythia_model.eval()
    pythia_sae = SAE.from_pretrained(release=PYTHIA_SAE_RELEASE, sae_id=PYTHIA_SAE_ID, device=DEVICE)
    pythia_sae.eval()

    print("Loading Mamba and its SAE...")
    mamba_model = HookedMamba.from_pretrained(MAMBA_MODEL_ID, device=DEVICE)
    mamba_model.eval()
    mamba_sae_dir = str(Path(MAMBA_SAE_PATH).parent)
    mamba_sae = SAE.load_from_disk(mamba_sae_dir, device=DEVICE)
    mamba_sae.eval()

    chunks_np = np.load(DATA_PATH, mmap_mode="r")
    
    mamba_top_acts = []
    pythia_top_acts = []

    print(f"\nScanning the first {SEARCH_CHUNKS} chunks of Wikipedia...")
    
    for i in tqdm(range(0, SEARCH_CHUNKS, BATCH_SIZE)):
        batch = chunks_np[i : i + BATCH_SIZE]
        tokens = torch.tensor(batch, dtype=torch.long, device=DEVICE)

        with torch.inference_mode():
            # Get Mamba activations
            _, m_cache = mamba_model.run_with_cache(tokens, names_filter=MAMBA_HOOK)
            m_sae_acts = mamba_sae.encode(m_cache[MAMBA_HOOK])
            m_feat_acts = m_sae_acts[:, :, TARGET_MAMBA_FEATURE]

            # Get Pythia activations
            _, p_cache = pythia_model.run_with_cache(tokens, names_filter=PYTHIA_HOOK)
            p_sae_acts = pythia_sae.encode(p_cache[PYTHIA_HOOK])
            p_feat_acts = p_sae_acts[:, :, pythia_twin_id]

            # Track Mamba top acts
            m_vals, m_indices = torch.topk(m_feat_acts.flatten(), min(TOP_K_CONTEXTS, m_feat_acts.numel()))
            for val, idx in zip(m_vals.cpu().tolist(), m_indices.cpu().tolist()):
                if val > 1.0:
                    b_idx, s_idx = idx // m_feat_acts.shape[1], idx % m_feat_acts.shape[1]
                    mamba_top_acts.append((val, i + b_idx, s_idx))

            # Track Pythia top acts
            p_vals, p_indices = torch.topk(p_feat_acts.flatten(), min(TOP_K_CONTEXTS, p_feat_acts.numel()))
            for val, idx in zip(p_vals.cpu().tolist(), p_indices.cpu().tolist()):
                if val > 1.0:
                    b_idx, s_idx = idx // p_feat_acts.shape[1], idx % p_feat_acts.shape[1]
                    pythia_top_acts.append((val, i + b_idx, s_idx))

        # Sort and truncate continuously to save memory
        mamba_top_acts = sorted(mamba_top_acts, key=lambda x: x[0], reverse=True)[:TOP_K_CONTEXTS]
        pythia_top_acts = sorted(pythia_top_acts, key=lambda x: x[0], reverse=True)[:TOP_K_CONTEXTS]

    def print_contexts(acts, model_name):
        print(f"\n=== {model_name} TOP ACTIVATIONS ===")
        if not acts:
            print("  [No strong activations found]")
        for rank, (val, global_b, s_idx) in enumerate(acts):
            seq_tokens = chunks_np[global_b]
            start_idx, end_idx = max(0, s_idx - 15), min(len(seq_tokens), s_idx + 5)
            
            prefix = tokenizer.decode(seq_tokens[start_idx:s_idx])
            trigger = tokenizer.decode(seq_tokens[s_idx:s_idx+1])
            suffix = tokenizer.decode(seq_tokens[s_idx+1:end_idx])
            
            print(f"  #{rank+1} (Strength: {val:.2f})")
            print(f"  ... {prefix} >>> {trigger.upper()} <<< {suffix} ...\n")

    print("\n" + "="*70)
    print("🔬 FEATURE SPLITTING ANALYSIS COMPLETE 🔬")
    print("="*70)
    
    print_contexts(mamba_top_acts, f"MAMBA FEATURE {TARGET_MAMBA_FEATURE}")
    print_contexts(pythia_top_acts, f"PYTHIA TWIN FEATURE {pythia_twin_id}")

if __name__ == "__main__":
    main()