import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# ── Config ───────────────────────────────────────────────────────────────────

TOKENIZER_ID   = "EleutherAI/gpt-neox-20b"   # shared by mamba-130m & Pythia-160m
DATASET_ID     = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"                # script-free English Wikipedia snapshot
DATASET_SPLIT  = "train"

DEFAULT_TARGET = 10_000_000   # tokens
DEFAULT_SEQ    = 512          # tokens per chunk fed into the model
BOS            = True         # prepend <BOS> at each document boundary
SEED           = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_tokens", type=int, default=DEFAULT_TARGET,
                   help="Stop after this many tokens (default 10 000 000)")
    p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ,
                   help="Sequence length for each chunk (default 512)")
    p.add_argument("--out_dir", type=str, default="data",
                   help="Output directory (default: ./data)")
    p.add_argument("--streaming", action="store_true", default=True,
                   help="Stream the dataset — avoids downloading all of Wikipedia")
    return p.parse_args()


def make_chunks(token_buffer: list[int], seq_len: int) -> tuple[list[list[int]], list[int]]:
    """Slice a flat token buffer into non-overlapping fixed-length chunks.

    Returns:
        chunks      – list of token lists, each exactly seq_len long
        remainder   – leftover tokens (< seq_len) to carry forward
    """
    n = len(token_buffer) // seq_len
    chunks    = [token_buffer[i * seq_len:(i + 1) * seq_len] for i in range(n)]
    remainder = token_buffer[n * seq_len:]
    return chunks, remainder


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    bos_id    = tokenizer.bos_token_id   # 0 for gpt-neox

    print(f"[2/5] Streaming dataset: {DATASET_ID} / {DATASET_CONFIG}")
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=args.streaming,
    )
    # NOTE — sequence packing: chunks are sliced from a continuous token stream,
    # so a single chunk may span two Wikipedia articles (end of one + BOS + start
    # of next). This is compute-optimal but has an interpretability implication:
    # any SAE feature that fires at topic-change boundaries or on the BOS token
    # is reacting to packed document seams, not within-document semantics.
    # For Mamba specifically, the hidden state h_t carries prior-document context
    # across the BOS boundary until the selective-update gate flushes it — a
    # potential source of "dark matter" features worth flagging in Week 2 analysis.
    dataset = dataset.shuffle(seed=SEED, buffer_size=10_000)

    # ── Tokenize & collect ───────────────────────────────────────────────────
    print(f"[3/5] Tokenising — target: {args.target_tokens:,} tokens  "
          f"(seq_len={args.seq_len})")

    all_chunks:      list[list[int]] = []
    token_buffer:    list[int]       = []
    total_tokens                     = 0
    n_docs                           = 0
    t0                               = time.time()

    pbar = tqdm(total=args.target_tokens, unit="tok", dynamic_ncols=True)

    for doc in dataset:
        text   = (doc.get("text") or "").strip()
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if BOS and bos_id is not None:
            ids = [bos_id] + ids

        token_buffer.extend(ids)
        n_docs += 1

        # Slice completed chunks out of the buffer
        new_chunks, token_buffer = make_chunks(token_buffer, args.seq_len)
        all_chunks.extend(new_chunks)
        total_tokens += len(new_chunks) * args.seq_len
        pbar.update(len(new_chunks) * args.seq_len)

        if total_tokens >= args.target_tokens:
            break

    pbar.close()

    # Trim to exactly target_tokens (drop the last partial batch if any)
    max_chunks = args.target_tokens // args.seq_len
    all_chunks = all_chunks[:max_chunks]
    total_tokens = len(all_chunks) * args.seq_len

    elapsed = time.time() - t0
    print(f"    ✓ {total_tokens:,} tokens  |  {len(all_chunks):,} chunks  "
          f"|  {n_docs:,} documents  |  {elapsed:.1f}s")

    # ── Save raw token ids (.bin) ────────────────────────────────────────────
    print("[4/5] Writing output files …")

    bin_path = out_dir / "wiki_10m_tokens.bin"
    flat_ids  = [tok for chunk in all_chunks for tok in chunk]
    with open(bin_path, "wb") as f:
        # Header: magic (4B) + n_tokens (8B) + dtype_code (1B = uint16)
        f.write(b"WTOK")
        f.write(struct.pack("<Q", total_tokens))
        f.write(struct.pack("<B", 2))   # 2 = uint16
        arr = np.array(flat_ids, dtype=np.uint16)
        arr.tofile(f)
    print(f"    → {bin_path}  ({bin_path.stat().st_size / 1e6:.1f} MB)")

    # ── Save chunked numpy array (.npy) ─────────────────────────────────────
    npy_path = out_dir / "wiki_10m_chunks.npy"
    chunks_np = np.array(all_chunks, dtype=np.int32)   # int32 for PyTorch compat
    np.save(npy_path, chunks_np)
    print(f"    → {npy_path}  shape={chunks_np.shape}  "
          f"({npy_path.stat().st_size / 1e6:.1f} MB)")

    # ── Save plaintext JSONL (for auto-interp in Week 2) ────────────────────
    jsonl_path = out_dir / "wiki_10m_chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks):
            text = tokenizer.decode(chunk, skip_special_tokens=True)
            f.write(json.dumps({"chunk_id": i, "text": text}) + "\n")
    print(f"    → {jsonl_path}  ({jsonl_path.stat().st_size / 1e6:.1f} MB)")

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        "tokenizer":     TOKENIZER_ID,
        "dataset":       f"{DATASET_ID}/{DATASET_CONFIG}",
        "split":         DATASET_SPLIT,
        "seed":          SEED,
        "bos_prepended": BOS,
        "target_tokens": args.target_tokens,
        "actual_tokens": total_tokens,
        "seq_len":       args.seq_len,
        "n_chunks":      len(all_chunks),
        "n_documents":   n_docs,
        "elapsed_s":     round(elapsed, 2),
        "files": {
            "bin":   str(bin_path),
            "npy":   str(npy_path),
            "jsonl": str(jsonl_path),
        },
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"    → {meta_path}")

    # ── Quick sanity check ───────────────────────────────────────────────────
    print("[5/5] Sanity check …")
    loaded = np.load(npy_path)
    assert loaded.shape == (len(all_chunks), args.seq_len), \
        f"Shape mismatch: {loaded.shape}"
    assert loaded.min() >= 0 and loaded.max() < len(tokenizer), \
        "Token ids out of vocab range"
    sample_text = tokenizer.decode(loaded[0].tolist(), skip_special_tokens=True)
    print(f'    Chunk[0] preview: "{sample_text[:120].strip()} …"')
    print("\n✅  Dataset ready.  Load with:\n")
    print("    import numpy as np")
    print(f"    chunks = np.load('{npy_path}')   # shape ({len(all_chunks)}, {args.seq_len})")
    print()


if __name__ == "__main__":
    main()