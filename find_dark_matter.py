import json
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
H5_PATH = Path("activations.h5")
OUT_JSON = Path("dark_matter_features.json")

SIMILARITY_THRESHOLD = 0.15
BATCH_CHUNKS = 32
PYTHIA_FEAT_CHUNK = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Safe tensor helper ────────────────────────────────────────────────────────

def bool_slice_to_cuda(arr: np.ndarray, device: str) -> torch.Tensor:
    """Zero-copy CPU transfer to avoid system RAM spikes."""
    return torch.from_numpy(arr.astype(np.float32)).to(device)

def main():
    print(f"Opening {H5_PATH} ...")

    with h5py.File(H5_PATH, "r") as h5:
        mamba_data  = h5["mamba"]
        pythia_data = h5["pythia"]

        n_chunks, seq_len, d_mamba  = mamba_data.shape
        _,        _,        d_pythia = pythia_data.shape

        n_tokens = n_chunks * seq_len
        print(f"Tokens        : {n_tokens:,}")
        print(f"Mamba features: {d_mamba:,}")
        print(f"Pythia features: {d_pythia:,}")

        # The intersection matrix is 12288 x 32768 floats = ~1.6 GB VRAM. 
        # This easily fits on the L4, so we can hold it entirely on the GPU.
        intersection = torch.zeros((d_mamba, d_pythia), dtype=torch.float32, device=DEVICE)
        mamba_counts = torch.zeros(d_mamba,  dtype=torch.float32, device=DEVICE)
        pythia_counts = torch.zeros(d_pythia, dtype=torch.float32, device=DEVICE)

        pythia_col_starts = range(0, d_pythia, PYTHIA_FEAT_CHUNK)

        print("\nComputing Jaccard similarity (Streaming batches from disk) ...")
        
        # FIX: The batch loop is now on the OUTSIDE. We read from disk exactly once.
        for i in tqdm(range(0, n_chunks, BATCH_CHUNKS), desc="Processing Batches"):
            
            # Read full batch into System RAM (Takes ~1 GB, perfectly safe)
            m_slice = mamba_data[i : i + BATCH_CHUNKS]
            p_slice = pythia_data[i : i + BATCH_CHUNKS] 
            
            m_t = bool_slice_to_cuda(m_slice.reshape(-1, d_mamba), DEVICE)
            mamba_counts += m_t.sum(dim=0)
            
            # Flatten Pythia batch in system RAM
            p_slice_flat = p_slice.reshape(-1, d_pythia)

            # FIX: The slab loop is now on the INSIDE. Slices from RAM, sends to VRAM.
            for p_start in pythia_col_starts:
                p_end  = min(p_start + PYTHIA_FEAT_CHUNK, d_pythia)
                
                # Move just this 512-feature slab to the GPU (~67 MB VRAM)
                p_t = bool_slice_to_cuda(p_slice_flat[:, p_start:p_end], DEVICE)

                # Dot product
                intersection[:, p_start:p_end] += m_t.T @ p_t
                pythia_counts[p_start:p_end] += p_t.sum(dim=0)

                del p_t

            del m_slice, p_slice, p_slice_flat, m_t

        print("\nCalculating final Jaccard metrics...")
        # Jaccard = Intersection / (Set M + Set P - Intersection)
        union = mamba_counts.unsqueeze(1) + pythia_counts.unsqueeze(0) - intersection
        union[union == 0] = 1.0   # avoid div/0 for dead features

        jaccard = intersection / union

        max_sims, best_match_indices = jaccard.max(dim=1)

        print("\nSaving ALL feature alignments ...")
        
        max_sims_cpu    = max_sims.cpu().tolist()
        best_match_cpu  = best_match_indices.cpu().tolist()
        mamba_fires_cpu = mamba_counts.cpu().tolist()

        all_features = []

        for mamba_idx, (sim, match_idx, fires) in enumerate(
            zip(max_sims_cpu, best_match_cpu, mamba_fires_cpu)
        ):
            if fires == 0:
                continue  # skip dead features

            all_features.append({
                "mamba_feature_id":     mamba_idx,
                "total_activations":    int(fires),
                "best_pythia_match_id": int(match_idx),
                "max_jaccard_similarity": round(sim, 4),
            })

        # Sort by similarity ASCENDING so the most "alien" Mamba features are at the top
        all_features.sort(key=lambda x: x["max_jaccard_similarity"])

        OUT_ALL_JSON = Path("all_mapped_features.json")
        
        with open(OUT_ALL_JSON, "w") as f:
            json.dump(all_features, f, indent=2)

        print(f"\n--- Results ---")
        print(f"Total Active Features Saved: {len(all_features):,}")
        print(f"Lowest Jaccard Similarity: {all_features[0]['max_jaccard_similarity']}")
        print(f"Highest Jaccard Similarity: {all_features[-1]['max_jaccard_similarity']}")
        print(f"Saved full feature map to {OUT_ALL_JSON}")

if __name__ == "__main__":
    main()