"""
Data Preparation Script for Large-Scale Training

Downloads and tokenizes data into sharded binary files for efficient training.
Run this BEFORE training on H100s to avoid download bottlenecks.

PARALLEL DOWNLOADS: Uses multiprocessing to download datasets simultaneously.
SPLIT LARGE DATASETS: Can split a single large dataset (like SYNTH) across multiple workers.

Usage:
    # Prepare 5B tokens (for full training)
    python prepare_data.py --max_tokens 5000000000 --datasets synth,guidelines,openmedtext

    # Prepare 100M tokens (for testing)
    python prepare_data.py --max_tokens 100000000 --datasets synth

    # FAST: Split SYNTH across 4 parallel workers
    python prepare_data.py --max_tokens 2000000000 --datasets synth,guidelines --split_workers 4
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import random
from multiprocessing import Pool, cpu_count
import tempfile

from data_loader import MedicalPretrainDataLoader, MEDICAL_DATASETS


def download_dataset_chunk(args: Tuple[str, int, int, str, int, int]) -> Tuple[str, str, int]:
    """
    Download and tokenize a chunk of a dataset (runs in parallel).

    Args:
        args: (dataset_name, target_tokens, seq_length, temp_dir, worker_id, skip_samples)

    Returns:
        (worker_name, temp_file_path, token_count)
    """
    dataset_name, target_tokens, seq_length, temp_dir, worker_id, skip_samples = args

    worker_name = f"{dataset_name}_{worker_id}" if worker_id > 0 else dataset_name

    print(f"\n[{worker_name}] Starting download (skip {skip_samples:,} samples)...")

    # Initialize loader for this process
    loader = MedicalPretrainDataLoader(
        datasets=[dataset_name],
        max_seq_length=seq_length,
        seed=42 + worker_id,  # Different seed per worker
    )

    target_samples = (target_tokens // seq_length) * 2

    dataset_tokens = []
    token_count = 0
    sample_count = 0
    skipped = 0
    start_time = time.time()
    last_print = start_time

    try:
        for text in loader.load_dataset_samples(dataset_name, max_samples=skip_samples + target_samples):
            # Skip samples assigned to other workers
            if skipped < skip_samples:
                skipped += 1
                continue

            tokens = loader.tokenize_text(text)
            eos = loader.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
            tokens.append(eos)

            dataset_tokens.extend(tokens)
            token_count += len(tokens)
            sample_count += 1

            if token_count >= target_tokens:
                break

            # Progress update every 5 seconds
            now = time.time()
            if now - last_print > 5:
                elapsed = now - start_time
                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                progress_pct = (token_count / target_tokens) * 100
                eta_min = ((target_tokens - token_count) / tokens_per_sec / 60) if tokens_per_sec > 0 else 0
                print(f"[{worker_name}] {progress_pct:5.1f}% | {token_count/1e6:.1f}M tokens | {tokens_per_sec/1e6:.2f}M tok/s | ETA: {eta_min:.1f}min")
                last_print = now

        elapsed = time.time() - start_time
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
        print(f"[{worker_name}] DONE: {token_count/1e6:.1f}M tokens in {elapsed:.1f}s ({tokens_per_sec/1e6:.2f}M tok/s)")

        # Save to temp file
        temp_file = os.path.join(temp_dir, f"{worker_name}_tokens.npy")
        np.array(dataset_tokens[:target_tokens], dtype=np.uint32).tofile(temp_file)

        return worker_name, temp_file, min(token_count, target_tokens)

    except Exception as e:
        print(f"[{worker_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return worker_name, None, 0


def prepare_sharded_data(
    datasets: List[str],
    max_tokens: int,
    output_dir: str,
    shard_size: int = 100_000_000,
    seq_length: int = 512,
    seed: int = 42,
    num_workers: int = None,
    split_workers: int = 1,  # How many workers to split large datasets across
):
    """
    Prepare training data as sharded binary files with PARALLEL downloads.

    Args:
        split_workers: Split large datasets (like SYNTH) across this many workers for faster download.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing Training Data (PARALLEL MODE)")
    print("=" * 60)
    print(f"Datasets: {datasets}")
    print(f"Target tokens: {max_tokens:,}")
    print(f"Split workers per dataset: {split_workers}")
    print(f"Shard size: {shard_size:,} tokens")
    print(f"Sequence length: {seq_length}")
    print(f"Output: {output_dir}")

    # Calculate tokens per dataset based on weights
    total_weight = sum(MEDICAL_DATASETS[d].weight for d in datasets)
    dataset_targets = {}
    for d in datasets:
        weight_ratio = MEDICAL_DATASETS[d].weight / total_weight
        dataset_targets[d] = int(max_tokens * weight_ratio)

    print(f"\nDataset allocation:")
    for d, tokens in dataset_targets.items():
        print(f"  {d}: {tokens:,} tokens ({tokens/max_tokens*100:.1f}%)")

    print(f"\n" + "-" * 60)
    print("Downloading datasets in PARALLEL...")
    print("-" * 60)

    start_time = time.time()

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare args for parallel download
        # Split large datasets across multiple workers
        download_args = []

        for name in datasets:
            target = dataset_targets[name]

            # Only split large datasets (SYNTH) and if split_workers > 1
            if split_workers > 1 and name == "synth" and target > 100_000_000:
                # Split SYNTH across multiple workers
                tokens_per_worker = target // split_workers
                samples_per_worker = tokens_per_worker // seq_length * 2

                print(f"\n  Splitting {name} across {split_workers} workers ({tokens_per_worker/1e6:.0f}M tokens each)")

                for i in range(split_workers):
                    skip = i * samples_per_worker
                    worker_tokens = tokens_per_worker if i < split_workers - 1 else target - (tokens_per_worker * i)
                    download_args.append((name, worker_tokens, seq_length, temp_dir, i, skip))
            else:
                # Single worker for this dataset
                download_args.append((name, target, seq_length, temp_dir, 0, 0))

        total_workers = len(download_args)
        if num_workers is None:
            num_workers = min(total_workers, cpu_count())

        print(f"\nTotal parallel downloads: {total_workers}")
        print(f"Worker pool size: {num_workers}")

        # Download in parallel
        if num_workers > 1 and total_workers > 1:
            with Pool(num_workers) as pool:
                results = pool.map(download_dataset_chunk, download_args)
        else:
            # Sequential for single dataset
            results = [download_dataset_chunk(args) for args in download_args]

        download_time = time.time() - start_time
        print(f"\n" + "-" * 60)
        print(f"Downloads complete in {download_time:.1f}s")
        print("-" * 60)

        # Merge results
        print("\nMerging tokens...")
        all_tokens = []
        total_collected = 0

        for name, temp_file, count in results:
            if temp_file and os.path.exists(temp_file):
                tokens = np.fromfile(temp_file, dtype=np.uint32)
                all_tokens.extend(tokens.tolist())
                total_collected += count
                print(f"  {name}: {count:,} tokens")

    print(f"\nTotal tokens collected: {total_collected:,}")

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
        "total_tokens": total_collected,
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
    parser = argparse.ArgumentParser(description="Prepare training data (parallel downloads)")

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
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--split_workers", type=int, default=1,
                        help="Split large datasets (SYNTH) across N workers (default: 1, try 4 for faster)")

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    prepare_sharded_data(
        datasets=datasets,
        max_tokens=args.max_tokens,
        output_dir=args.output,
        shard_size=args.shard_size,
        seq_length=args.seq_length,
        seed=args.seed,
        num_workers=args.workers,
        split_workers=args.split_workers,
    )


if __name__ == "__main__":
    main()
