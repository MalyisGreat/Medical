"""
Data Preparation Script for Large-Scale Training

Downloads and tokenizes data into sharded binary files for efficient training.
Run this BEFORE training on H100s to avoid download bottlenecks.

FAST DOWNLOAD: Uses snapshot_download with parallel workers (much faster than streaming!)
FAST TOKENIZATION: Uses multiprocessing + batch encoding (~10x faster!)

Usage:
    # Prepare 2B tokens (FAST - uses parallel download + parallel tokenization)
    python prepare_data.py --max_tokens 2000000000 --datasets synth,guidelines

    # Prepare 100M tokens (for testing)
    python prepare_data.py --max_tokens 100000000 --datasets synth
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Iterator, Dict, Any
import random
import multiprocessing as mp
from functools import partial

import tiktoken
from huggingface_hub import snapshot_download
from datasets import load_dataset

# Global tokenizer for multiprocessing (initialized per worker)
_tokenizer = None
_eos_token = None

def _init_tokenizer():
    """Initialize tokenizer in worker process."""
    global _tokenizer, _eos_token
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    _eos_token = _tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

def _tokenize_batch(texts: List[str]) -> List[List[int]]:
    """Tokenize a batch of texts (called in worker process)."""
    global _tokenizer, _eos_token
    if _tokenizer is None:
        _init_tokenizer()

    results = []
    for text in texts:
        tokens = _tokenizer.encode(text, disallowed_special=())
        tokens.append(_eos_token)
        results.append(tokens)
    return results

# Dataset configurations
DATASET_CONFIGS = {
    "synth": {
        "repo_id": "PleIAs/SYNTH",
        "weight": 0.40,
    },
    "guidelines": {
        "repo_id": "epfl-llm/guidelines",
        "weight": 0.15,
    },
    "openmedtext": {
        "repo_id": "ywchoi/OpenMedText",
        "weight": 0.10,
    },
    "wikitext": {
        "repo_id": "wikitext",
        "name": "wikitext-2-raw-v1",
        "weight": 1.0,
    },
}


def format_synth(example: Dict[str, Any]) -> str:
    """Format SYNTH with reasoning chain (English only)."""
    lang = example.get("language", "")
    if lang != "en":
        return ""

    query = example.get("query", "")
    reasoning = example.get("synthetic_reasoning", "")
    answer = example.get("synthetic_answer", "")

    if reasoning and answer:
        return f"<|question|>\n{query}\n<|thinking|>\n{reasoning}\n<|answer|>\n{answer}"
    elif reasoning:
        return f"<|question|>\n{query}\n<|thinking|>\n{reasoning}"
    return query


def format_guidelines(example: Dict[str, Any]) -> str:
    """Format clinical guidelines."""
    title = example.get("title", "")
    text = example.get("clean_text", example.get("text", ""))
    if title and text:
        return f"Clinical Guideline: {title}\n\n{text}"
    return text


def format_example(example: Dict[str, Any], dataset_name: str) -> str:
    """Format example based on dataset type."""
    if dataset_name == "synth":
        return format_synth(example)
    elif dataset_name == "guidelines":
        return format_guidelines(example)
    else:
        return example.get("text", "")


def fast_download_dataset(
    dataset_name: str,
    target_tokens: int,
    seq_length: int,
    tokenizer,
    download_workers: int = 8,
    tokenize_workers: int = None,
) -> List[int]:
    """
    Download dataset FAST using snapshot_download with parallel workers.
    Then load and tokenize with PARALLEL multiprocessing.
    """
    if tokenize_workers is None:
        tokenize_workers = min(mp.cpu_count(), 16)  # Use up to 16 cores

    config = DATASET_CONFIGS[dataset_name]
    repo_id = config["repo_id"]

    print(f"\n[{dataset_name}] Downloading with {download_workers} parallel workers...")
    start_time = time.time()

    # Step 1: Fast parallel download of all files
    try:
        cache_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            max_workers=download_workers,  # Parallel file downloads!
        )
        download_time = time.time() - start_time
        print(f"[{dataset_name}] Download complete in {download_time:.1f}s")
    except Exception as e:
        print(f"[{dataset_name}] snapshot_download failed: {e}")
        print(f"[{dataset_name}] Falling back to streaming...")
        cache_dir = None

    # Step 2: Load dataset
    print(f"[{dataset_name}] Loading dataset...")
    load_start = time.time()

    try:
        if cache_dir:
            ds = load_dataset(repo_id, config.get("name"), split="train")
        else:
            ds = load_dataset(repo_id, config.get("name"), split="train", streaming=True)
    except Exception as e:
        print(f"[{dataset_name}] Error loading: {e}")
        return []

    # Step 3: Collect texts (estimate how many we need based on avg tokens per text)
    # Assume ~200 tokens per text on average, collect 2x what we need
    estimated_texts_needed = (target_tokens // 200) * 2
    print(f"[{dataset_name}] Collecting texts (target: ~{estimated_texts_needed:,} texts)...")

    texts = []
    collect_start = time.time()
    last_print = time.time()

    for example in ds:
        text = format_example(example, dataset_name)
        if len(text) < 50:
            continue
        texts.append(text)

        if len(texts) >= estimated_texts_needed:
            break

        # Progress every 5 seconds
        now = time.time()
        if now - last_print > 5:
            print(f"[{dataset_name}] Collected {len(texts):,} texts...")
            last_print = now

    collect_time = time.time() - collect_start
    print(f"[{dataset_name}] Collected {len(texts):,} texts in {collect_time:.1f}s")

    if not texts:
        return []

    # Step 4: Parallel tokenization
    print(f"[{dataset_name}] Tokenizing with {tokenize_workers} parallel workers...")
    tokenize_start = time.time()

    # Split texts into batches for workers
    batch_size = max(1000, len(texts) // tokenize_workers)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    tokens = []
    total_tokenized = 0

    # Use multiprocessing pool (spawn context for Windows compatibility)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=tokenize_workers, initializer=_init_tokenizer) as pool:
        for batch_results in pool.imap(_tokenize_batch, batches):
            for token_list in batch_results:
                tokens.extend(token_list)
                total_tokenized += 1

                if len(tokens) >= target_tokens:
                    break

            if len(tokens) >= target_tokens:
                break

            # Progress
            now = time.time()
            if now - last_print > 3:
                progress = len(tokens) / target_tokens * 100
                tok_per_sec = len(tokens) / (now - tokenize_start)
                eta = (target_tokens - len(tokens)) / tok_per_sec / 60 if tok_per_sec > 0 else 0
                print(f"[{dataset_name}] Tokenizing: {progress:.1f}% | {len(tokens)/1e6:.1f}M tokens | {tok_per_sec/1e6:.2f}M tok/s | ETA: {eta:.1f}min")
                last_print = now

    tokenize_time = time.time() - tokenize_start
    total_time = time.time() - start_time

    print(f"[{dataset_name}] DONE: {len(tokens)/1e6:.1f}M tokens")
    print(f"[{dataset_name}]   Tokenize: {tokenize_time:.1f}s ({len(tokens)/tokenize_time/1e6:.2f}M tok/s)")
    print(f"[{dataset_name}]   Total: {total_time:.1f}s")

    return tokens[:target_tokens]


def prepare_sharded_data(
    datasets: List[str],
    max_tokens: int,
    output_dir: str,
    shard_size: int = 100_000_000,
    seq_length: int = 512,
    seed: int = 42,
    download_workers: int = 8,
    tokenize_workers: int = None,
):
    """
    Prepare training data as sharded binary files with FAST parallel downloads.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    if tokenize_workers is None:
        tokenize_workers = min(mp.cpu_count(), 16)

    print("=" * 60)
    print("Preparing Training Data (FAST PARALLEL DOWNLOAD + TOKENIZE)")
    print("=" * 60)
    print(f"Datasets: {datasets}")
    print(f"Target tokens: {max_tokens:,}")
    print(f"Download workers: {download_workers}")
    print(f"Tokenize workers: {tokenize_workers}")
    print(f"Shard size: {shard_size:,} tokens")
    print(f"Sequence length: {seq_length}")
    print(f"Output: {output_dir}")

    # Calculate tokens per dataset based on weights
    total_weight = sum(DATASET_CONFIGS[d]["weight"] for d in datasets)
    dataset_targets = {}
    for d in datasets:
        weight_ratio = DATASET_CONFIGS[d]["weight"] / total_weight
        dataset_targets[d] = int(max_tokens * weight_ratio)

    print(f"\nDataset allocation:")
    for d, tokens in dataset_targets.items():
        print(f"  {d}: {tokens:,} tokens ({tokens/max_tokens*100:.1f}%)")

    print(f"\n" + "-" * 60)
    print("Downloading and tokenizing...")
    print("-" * 60)

    start_time = time.time()
    all_tokens = []

    # Download each dataset with fast parallel download + parallel tokenization
    for name in datasets:
        target = dataset_targets[name]
        tokens = fast_download_dataset(
            name, target, seq_length, tokenizer, download_workers, tokenize_workers
        )
        all_tokens.extend(tokens)
        print(f"  {name}: {len(tokens):,} tokens collected")

    download_time = time.time() - start_time
    print(f"\n" + "-" * 60)
    print(f"All downloads complete in {download_time:.1f}s ({download_time/60:.1f}min)")
    print("-" * 60)

    print(f"\nTotal tokens collected: {len(all_tokens):,}")

    # Create chunks
    print(f"\nCreating chunks and shuffling...")

    chunks = []
    for i in range(0, len(all_tokens) - seq_length, seq_length):
        chunk = all_tokens[i:i + seq_length]
        if len(chunk) == seq_length:
            chunks.append(chunk)

    print(f"Created {len(chunks):,} chunks")

    # Shuffle chunks
    random.shuffle(chunks)

    # Convert to numpy array
    data = np.array(chunks, dtype=np.uint32)
    print(f"Data shape: {data.shape}")
    print(f"Data size: {data.nbytes / 1e9:.2f} GB")

    # Save as shards
    print(f"\n" + "-" * 60)
    print("Saving shards...")
    print("-" * 60)

    chunks_per_shard = shard_size // seq_length
    num_shards = (len(chunks) + chunks_per_shard - 1) // chunks_per_shard

    for shard_idx in range(num_shards):
        start_idx = shard_idx * chunks_per_shard
        end_idx = min((shard_idx + 1) * chunks_per_shard, len(chunks))

        shard_data = data[start_idx:end_idx]
        shard_file = output_path / f"shard_{shard_idx:04d}.bin"

        shard_data.tofile(shard_file)

        shard_tokens = shard_data.shape[0] * seq_length
        print(f"  Shard {shard_idx}: {shard_data.shape[0]:,} chunks, {shard_tokens:,} tokens")

    # Save metadata
    import json
    metadata = {
        "datasets": datasets,
        "total_tokens": len(all_tokens),
        "total_chunks": len(chunks),
        "seq_length": seq_length,
        "num_shards": num_shards,
        "shard_files": [f.name for f in output_path.glob("shard_*.bin")],
        "dtype": "uint32",
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time

    print(f"\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"Total chunks: {len(chunks):,}")
    print(f"Total tokens: {len(chunks) * seq_length:,}")
    print(f"Shards: {num_shards}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Output: {output_dir}")
    print(f"\nTo train, use:")
    print(f"  python train.py --data_dir {output_dir} --model medium --batch_size 256 --compile")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare training data (fast parallel downloads)")

    parser.add_argument("--datasets", type=str, default="synth,guidelines",
                        help="Comma-separated dataset names")
    parser.add_argument("--max_tokens", type=int, default=100_000_000,
                        help="Total tokens to prepare")
    parser.add_argument("--output", type=str, default="./data_shards",
                        help="Output directory for shards")
    parser.add_argument("--shard_size", type=int, default=100_000_000,
                        help="Tokens per shard (default 100M)")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--download_workers", type=int, default=8,
                        help="Number of parallel download workers (default: 8)")
    parser.add_argument("--tokenize_workers", type=int, default=None,
                        help="Number of parallel tokenization workers (default: auto)")

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    prepare_sharded_data(
        datasets=datasets,
        max_tokens=args.max_tokens,
        output_dir=args.output,
        shard_size=args.shard_size,
        seq_length=args.seq_length,
        seed=args.seed,
        download_workers=args.download_workers,
        tokenize_workers=args.tokenize_workers,
    )


if __name__ == "__main__":
    main()
