"""
harvest_activations.py
======================
Streams tokenized chunks through Pythia-70m-deduped and Mamba-130m, intercepts
the residual stream at each model's middle layer, extracts SAE features, and
saves the boolean activation masks directly to disk via HDF5.
"""

import gc
import sys
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer
from mamba_lens import HookedMamba  # Required to load Mamba architecture!
from sae_lens import SAE

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = Path("wiki_10m_chunks.npy")
OUT_PATH  = Path("activations.h5")

BATCH_SIZE = 8       # Safe for 24 GB L4 GPU at seq_len=1024
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Pythia config ─────────────────────────────────────────────────────────────
PYTHIA_MODEL_ID   = "EleutherAI/pythia-70m-deduped"
PYTHIA_SAE_RELEASE = "pythia-70m-deduped-res-sm"
PYTHIA_SAE_ID     = "blocks.3.hook_resid_post"   # middle of 6 layers
PYTHIA_LAYER      = 3

# ── Mamba config ──────────────────────────────────────────────────────────────
MAMBA_MODEL_ID  = "state-spaces/mamba-130m"
# Pointing directly to the weights you just trained!
MAMBA_SAE_PATH  = "sae_weights.safetensors" 
MAMBA_LAYER     = 12      # middle of Mamba-130m's 24 layers


# ── Helpers ───────────────────────────────────────────────────────────────────

def free_memory():
    """Forces aggressive garbage collection and empties GPU VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_mamba_sae(sae_path: str, device: str) -> SAE:
    """Load the custom Mamba SAE we just trained from the local path."""
    p = Path(sae_path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find SAE weights at {sae_path}")
    
    # Load from local directory (V6 API handles the safetensors file)
    parent_dir = str(p.parent)
    sae = SAE.load_from_pretrained(parent_dir, device=device)
    return sae

def process_model(
    model_id:    str,
    layer:       int,
    chunks_np:   np.ndarray,
    h5_file:     h5py.File,
    dataset_name: str,
    sae:         SAE,
):
    """Loads a model, processes all chunks, and saves boolean masks to HDF5."""

    print(f"\n--- Starting Pipeline for {model_id} (layer {layer}) ---")

    print(f"  Loading base model {model_id} ...")
    # CRITICAL FIX: We must use HookedMamba for Mamba, HookedTransformer for Pythia
    if "mamba" in model_id.lower():
        model = HookedMamba.from_pretrained(model_id, device=DEVICE)
    else:
        model = HookedTransformer.from_pretrained(model_id, device=DEVICE)
        
    model.eval()

    n_chunks, seq_len = chunks_np.shape
    n_features        = sae.cfg.d_sae
    hook_name         = f"blocks.{layer}.hook_resid_post"

    # ── Create HDF5 dataset ──────────────────────────────────────────────────
    print(f"  Creating HDF5 dataset '{dataset_name}' -> shape ({n_chunks}, {seq_len}, {n_features})")
    if dataset_name in h5_file:
        del h5_file[dataset_name]

    h5_dataset = h5_file.create_dataset(
        name=dataset_name,
        shape=(n_chunks, seq_len, n_features),
        dtype=bool,
        chunks=(1, seq_len, n_features),   
        compression="gzip",
        compression_opts=4,
    )

    # ── Inference loop ───────────────────────────────────────────────────────
    print(f"  Harvesting activations via hook '{hook_name}' ...")

    for i in tqdm(range(0, n_chunks, BATCH_SIZE), desc=f"  {dataset_name}"):
        batch = chunks_np[i : i + BATCH_SIZE]
        tokens = torch.tensor(batch, dtype=torch.long, device=DEVICE)

        with torch.inference_mode():
            _, cache   = model.run_with_cache(tokens, names_filter=hook_name)
            resid      = cache[hook_name]          # [B, S, d_model]
            sae_acts   = sae.encode(resid)         # [B, S, d_sae]
            
            # Binarize to save RAM: > 0 means the feature fired
            fired_mask = (sae_acts > 0).cpu().numpy()

            h5_dataset[i : i + len(batch)] = fired_mask

            del tokens, cache, resid, sae_acts, fired_mask

    print(f"  Finished {model_id}. Clearing memory ...")
    del model
    free_memory()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Run prepare_dataset.py first.")

    print(f"Loading chunk array from {DATA_PATH} (memory-mapped)")
    chunks_np = np.load(DATA_PATH, mmap_mode="r")

    print(f"\n[1/4] Loading Pythia SAE: {PYTHIA_SAE_RELEASE} / {PYTHIA_SAE_ID}")
    pythia_sae, _, _ = SAE.from_pretrained(
        release=PYTHIA_SAE_RELEASE,
        sae_id=PYTHIA_SAE_ID,
        device=DEVICE,
    )
    pythia_sae.eval()

    print(f"[2/4] Loading Mamba SAE from: {MAMBA_SAE_PATH}")
    mamba_sae = load_mamba_sae(MAMBA_SAE_PATH, DEVICE)
    mamba_sae.eval()

    # ── Harvest ──────────────────────────────────────────────────────────────
    with h5py.File(OUT_PATH, "a") as h5_file:

        print("\n[3/4] Pass 1 - Pythia")
        process_model(
            model_id=PYTHIA_MODEL_ID,
            layer=PYTHIA_LAYER,
            chunks_np=chunks_np,
            h5_file=h5_file,
            dataset_name="pythia",
            sae=pythia_sae,
        )
        del pythia_sae
        free_memory()

        print("\n[4/4] Pass 2 - Mamba")
        process_model(
            model_id=MAMBA_MODEL_ID,
            layer=MAMBA_LAYER,
            chunks_np=chunks_np,
            h5_file=h5_file,
            dataset_name="mamba",
            sae=mamba_sae,
        )
        del mamba_sae
        free_memory()

    print(f"\n✅ Activation harvesting complete! Data saved to {OUT_PATH}")

if __name__ == "__main__":
    main()