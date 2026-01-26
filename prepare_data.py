"""
Data Preparation Script for Large-Scale Training

Downloads and tokenizes data into sharded binary files for efficient training.
Run this BEFORE training on H100s to avoid download bottlenecks.

Usage:
    # Prepare 5B tokens (for full training)
    python prepare_data.py --max_tokens 5000000000 --datasets synth,guidelines,openmedtext

    # Prepare 100M tokens (for testing)
    python prepare_data.py --max_tokens 100000000 --datasets synth
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Iterator
import random

from data_loader import MedicalPretrainDataLoader, MEDICAL_DATASETS


def prepare_sharded_data(
    datasets: List[str],
    max_tokens: int,
    output_dir: str,
    shard_size: int = 100_000_000,  # 100M tokens per shard
    seq_length: int = 512,
    seed: int = 42,
):
    """
    Prepare training data as sharded binary files.

    Args:
        datasets: List of dataset names to use
        max_tokens: Total tokens to prepare
        output_dir: Directory to save shards
        shard_size: Tokens per shard file
        seq_length: Sequence length for training
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing Training Data")
    print("=" * 60)
    print(f"Datasets: {datasets}")
    print(f"Target tokens: {max_tokens:,}")
    print(f"Shard size: {shard_size:,} tokens")
    print(f"Sequence length: {seq_length}")
    print(f"Output: {output_dir}")

    # Initialize data loader
    loader = MedicalPretrainDataLoader(
        datasets=datasets,
        max_seq_length=seq_length,
        seed=seed,
    )

    # Calculate tokens per dataset based on weights
    total_weight = sum(MEDICAL_DATASETS[d].weight for d in datasets)
    dataset_targets = {}
    for d in datasets:
        weight_ratio = MEDICAL_DATASETS[d].weight / total_weight
        dataset_targets[d] = int(max_tokens * weight_ratio)

    print(f"\nDataset allocation:")
    for d, tokens in dataset_targets.items():
        print(f"  {d}: {tokens:,} tokens ({tokens/max_tokens*100:.1f}%)")

    # Stream and tokenize data
    print(f"\n" + "-" * 60)
    print("Downloading and tokenizing...")
    print("-" * 60)

    all_tokens = []
    total_collected = 0

    for dataset_name in datasets:
        target_tokens = dataset_targets[dataset_name]
        target_samples = (target_tokens // seq_length) * 2  # 2x to account for filtering

        print(f"\n{MEDICAL_DATASETS[dataset_name].name}:")
        print(f"  Target: {target_tokens:,} tokens")

        dataset_tokens = []
        token_count = 0
        sample_count = 0
        start_time = time.time()

        try:
            for text in loader.load_dataset_samples(dataset_name, max_samples=target_samples):
                tokens = loader.tokenize_text(text)
                # Add EOS token
                eos = loader.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
                tokens.append(eos)

                dataset_tokens.extend(tokens)
                token_count += len(tokens)
                sample_count += 1

                if token_count >= target_tokens:
                    break

                # Progress update every 500 samples with speed
                if sample_count % 500 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                    progress_pct = (token_count / target_tokens) * 100
                    eta_sec = (target_tokens - token_count) / tokens_per_sec if tokens_per_sec > 0 else 0
                    eta_min = eta_sec / 60
                    print(f"    {progress_pct:5.1f}% | {token_count:,} tokens | {tokens_per_sec:,.0f} tok/s | ETA: {eta_min:.1f}min", end="\r")

            elapsed = time.time() - start_time
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
            print(f"  Collected: {token_count:,} tokens from {sample_count:,} samples ({tokens_per_sec:,.0f} tok/s, {elapsed:.1f}s)")
            all_tokens.extend(dataset_tokens[:target_tokens])
            total_collected += min(token_count, target_tokens)

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nTotal tokens collected: {total_collected:,}")

    # Shuffle all tokens at chunk level
    print(f"\nCreating chunks and shuffling...")

    # Create chunks
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

    shard_files = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * chunks_per_shard
        end_idx = min((shard_idx + 1) * chunks_per_shard, len(chunks))

        shard_data = data[start_idx:end_idx]
        shard_file = output_path / f"shard_{shard_idx:04d}.bin"

        shard_data.tofile(shard_file)
        shard_files.append(str(shard_file))

        shard_tokens = shard_data.shape[0] * seq_length
        print(f"  Shard {shard_idx}: {shard_data.shape[0]:,} chunks, {shard_tokens:,} tokens")

    # Save metadata
    metadata = {
        "datasets": datasets,
        "total_tokens": total_collected,
        "total_chunks": len(chunks),
        "seq_length": seq_length,
        "num_shards": num_shards,
        "shard_files": [f.name for f in output_path.glob("shard_*.bin")],
        "dtype": "uint32",
    }

    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"Total chunks: {len(chunks):,}")
    print(f"Total tokens: {len(chunks) * seq_length:,}")
    print(f"Shards: {num_shards}")
    print(f"Output: {output_dir}")
    print(f"\nTo train, use:")
    print(f"  python train.py --data_dir {output_dir} ...")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")

    parser.add_argument("--datasets", type=str, default="synth,guidelines,openmedtext",
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

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    prepare_sharded_data(
        datasets=datasets,
        max_tokens=args.max_tokens,
        output_dir=args.output,
        shard_size=args.shard_size,
        seq_length=args.seq_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
