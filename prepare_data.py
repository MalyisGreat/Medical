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
from typing import List, Iterator, Dict, Any, Optional
import random
import multiprocessing as mp
from functools import partial
import fnmatch
import math

import tiktoken
from huggingface_hub import snapshot_download, HfApi
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
        "file_pattern": "synth_*.parquet",
    },
    "pubmed": {
        "repo_id": "ncbi/pubmed",
        "weight": 0.30,
    },
    "guidelines": {
        "repo_id": "epfl-llm/guidelines",
        "weight": 0.15,
    },
    "openmedtext": {
        "repo_id": "ywchoi/OpenMedText",
        "weight": 0.10,
    },
    "fineweb_edu": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "weight": 0.05,
    },
    "wikitext": {
        "repo_id": "wikitext",
        "name": "wikitext-2-raw-v1",
        "weight": 1.0,
        "file_pattern": None,
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

def format_pubmed(example: Dict[str, Any]) -> str:
    """Format PubMed abstract with title."""
    try:
        citation = example.get("MedlineCitation", {})
        article = citation.get("Article", {})

        title = article.get("ArticleTitle", "")
        abstract = article.get("Abstract", {})
        abstract_text = abstract.get("AbstractText", "")

        if isinstance(abstract_text, list):
            abstract_text = " ".join(str(t) for t in abstract_text if t)

        if title and abstract_text:
            return f"Title: {title}\n\nAbstract: {abstract_text}"
        return abstract_text or title or ""
    except Exception:
        return ""


def format_example(example: Dict[str, Any], dataset_name: str) -> str:
    """Format example based on dataset type."""
    if dataset_name == "synth":
        return format_synth(example)
    elif dataset_name == "pubmed":
        return format_pubmed(example)
    elif dataset_name == "guidelines":
        return format_guidelines(example)
    else:
        return example.get("text", "")


class ShardWriter:
    """Stream tokens into sharded binary files without large memory spikes."""

    def __init__(
        self,
        output_dir: str,
        seq_length: int,
        shard_size: int,
        seed: int = 42,
        shuffle_buffer: int = 4096,
    ):
        self.output_path = Path(output_dir)
        self.seq_length = seq_length
        self.shard_size = shard_size
        self.chunks_per_shard = max(1, shard_size // seq_length)
        self.shuffle_buffer = shuffle_buffer
        self.rng = random.Random(seed)

        self.token_buffer: List[int] = []
        self.chunk_buffer: List[List[int]] = []
        self.shard_idx = 0
        self.current_chunks = 0
        self.total_chunks = 0
        self.total_tokens = 0
        self.shard_files: List[str] = []
        self.shard_fh = None
        self.current_shard_path: Optional[Path] = None

        self._open_shard()

    def _open_shard(self):
        self.current_chunks = 0
        self.current_shard_path = self.output_path / f"shard_{self.shard_idx:04d}.bin"
        self.shard_fh = open(self.current_shard_path, "wb")
        self.shard_files.append(self.current_shard_path.name)

    def _close_shard(self, remove_if_empty: bool = False):
        if self.shard_fh is not None:
            self.shard_fh.close()
            self.shard_fh = None
        if remove_if_empty or self.current_chunks == 0:
            try:
                if self.current_shard_path and self.current_shard_path.exists():
                    os.remove(self.current_shard_path)
            except OSError:
                pass
            if self.shard_files and self.current_shard_path:
                if self.shard_files[-1] == self.current_shard_path.name:
                    self.shard_files.pop()

    def _flush_chunks(self):
        if not self.chunk_buffer:
            return

        self.rng.shuffle(self.chunk_buffer)

        while self.chunk_buffer:
            remaining = self.chunks_per_shard - self.current_chunks
            to_write = self.chunk_buffer[:remaining]
            if not to_write:
                break

            arr = np.asarray(to_write, dtype=np.uint32)
            arr.tofile(self.shard_fh)

            wrote = len(to_write)
            self.current_chunks += wrote
            self.total_chunks += wrote
            self.total_tokens += wrote * self.seq_length
            self.chunk_buffer = self.chunk_buffer[remaining:]

            if self.current_chunks >= self.chunks_per_shard:
                self._close_shard(remove_if_empty=False)
                self.shard_idx += 1
                self._open_shard()

    def add_tokens(self, tokens: List[int]):
        self.token_buffer.extend(tokens)

        while len(self.token_buffer) >= self.seq_length:
            chunk = self.token_buffer[:self.seq_length]
            self.token_buffer = self.token_buffer[self.seq_length:]
            self.chunk_buffer.append(chunk)

            if len(self.chunk_buffer) >= self.shuffle_buffer:
                self._flush_chunks()

    def finalize(self):
        # Drop leftover tokens that don't fill a chunk
        self.token_buffer = []
        self._flush_chunks()
        if self.current_chunks == 0:
            self._close_shard(remove_if_empty=True)
        else:
            self._close_shard(remove_if_empty=False)


def _select_repo_files(repo_id: str, pattern: str, pct: float) -> List[str]:
    """Select the first pct% of repo files matching a pattern."""
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    matched = sorted([f for f in files if fnmatch.fnmatch(f, pattern)])
    if not matched:
        return []
    if pct <= 0 or pct > 100:
        raise ValueError(f"subset pct must be in (0, 100], got {pct}")
    k = max(1, int(math.floor(len(matched) * (pct / 100.0))))
    return matched[:k]


def fast_download_dataset(
    dataset_name: str,
    target_tokens: int,
    seq_length: int,
    tokenizer,
    download_workers: int = 8,
    tokenize_workers: int = None,
    writer: Optional[ShardWriter] = None,
    subset_pct: Optional[float] = None,
) -> int:
    """
    Download dataset FAST using snapshot_download with parallel workers.
    Then load and tokenize with PARALLEL multiprocessing.
    """
    if tokenize_workers is None:
        tokenize_workers = min(mp.cpu_count(), 16)  # Use up to 16 cores

    config = DATASET_CONFIGS[dataset_name]
    repo_id = config["repo_id"]
    file_pattern = config.get("file_pattern")
    subset_files = None

    print(f"\n[{dataset_name}] Downloading with {download_workers} parallel workers...")
    start_time = time.time()

    # Step 1: Fast parallel download of selected files
    try:
        allow_patterns = None
        if subset_pct and file_pattern:
            subset_files = _select_repo_files(repo_id, file_pattern, subset_pct)
            if not subset_files:
                raise ValueError(f"No files matched pattern {file_pattern}")
            allow_patterns = subset_files

        cache_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            max_workers=download_workers,  # Parallel file downloads!
            allow_patterns=allow_patterns,
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
            if subset_files:
                ds = load_dataset(
                    repo_id,
                    config.get("name"),
                    split="train",
                    data_files={"train": subset_files},
                )
            else:
                ds = load_dataset(repo_id, config.get("name"), split="train")
        else:
            if subset_files:
                ds = load_dataset(
                    repo_id,
                    config.get("name"),
                    split="train",
                    streaming=True,
                    data_files={"train": subset_files},
                )
            else:
                ds = load_dataset(repo_id, config.get("name"), split="train", streaming=True)
    except Exception as e:
        print(f"[{dataset_name}] Error loading: {e}")
        return []

    # Step 3: Stream texts and tokenize in parallel
    # Assume ~200 tokens per text on average, collect 2x what we need
    estimated_texts_needed = (target_tokens // 200) * 2
    print(f"[{dataset_name}] Streaming texts (target: ~{estimated_texts_needed:,} texts)...")

    def iter_text_batches(batch_size: int, max_texts: int):
        batch = []
        count = 0
        for example in ds:
            text = format_example(example, dataset_name)
            if len(text) < 50:
                continue
            batch.append(text)
            count += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

            if max_texts and count >= max_texts:
                break

        if batch:
            yield batch

    print(f"[{dataset_name}] Tokenizing with {tokenize_workers} parallel workers...")
    tokenize_start = time.time()
    last_print = time.time()
    tokens_collected = 0

    batch_size = 1000
    batches = iter_text_batches(batch_size, estimated_texts_needed)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=tokenize_workers, initializer=_init_tokenizer)

    try:
        stop = False
        for batch_results in pool.imap(_tokenize_batch, batches):
            for token_list in batch_results:
                if not token_list:
                    continue
                remaining = target_tokens - tokens_collected
                if remaining <= 0:
                    stop = True
                    break

                if len(token_list) > remaining:
                    token_list = token_list[:remaining]

                if writer is not None:
                    writer.add_tokens(token_list)
                tokens_collected += len(token_list)

                if tokens_collected >= target_tokens:
                    stop = True
                    break

            if stop:
                break

            now = time.time()
            if now - last_print > 3:
                progress = tokens_collected / target_tokens * 100
                tok_per_sec = tokens_collected / (now - tokenize_start) if now > tokenize_start else 0
                eta = (target_tokens - tokens_collected) / tok_per_sec / 60 if tok_per_sec > 0 else 0
                print(f"[{dataset_name}] Tokenizing: {progress:.1f}% | {tokens_collected/1e6:.1f}M tokens | {tok_per_sec/1e6:.2f}M tok/s | ETA: {eta:.1f}min")
                last_print = now

        if stop:
            pool.terminate()
        else:
            pool.close()
    finally:
        pool.join()

    tokenize_time = time.time() - tokenize_start
    total_time = time.time() - start_time

    print(f"[{dataset_name}] DONE: {tokens_collected/1e6:.1f}M tokens")
    if tokenize_time > 0:
        print(f"[{dataset_name}]   Tokenize: {tokenize_time:.1f}s ({tokens_collected/tokenize_time/1e6:.2f}M tok/s)")
    print(f"[{dataset_name}]   Total: {total_time:.1f}s")

    return tokens_collected


def prepare_sharded_data(
    datasets: List[str],
    max_tokens: int,
    output_dir: str,
    shard_size: int = 100_000_000,
    seq_length: int = 512,
    seed: int = 42,
    download_workers: int = 8,
    tokenize_workers: int = None,
    synth_pct: Optional[float] = None,
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
    writer = ShardWriter(
        output_dir=output_dir,
        seq_length=seq_length,
        shard_size=shard_size,
        seed=seed,
        shuffle_buffer=4096,
    )

    # Download each dataset with fast parallel download + parallel tokenization
    for name in datasets:
        target = dataset_targets[name]
        subset_pct = synth_pct if name == "synth" else None
        tokens_collected = fast_download_dataset(
            name,
            target,
            seq_length,
            tokenizer,
            download_workers,
            tokenize_workers,
            writer=writer,
            subset_pct=subset_pct,
        )
        print(f"  {name}: {tokens_collected:,} tokens collected")

    download_time = time.time() - start_time
    print(f"\n" + "-" * 60)
    print(f"All downloads complete in {download_time:.1f}s ({download_time/60:.1f}min)")
    print("-" * 60)

    print(f"\nFinalizing shards...")
    writer.finalize()

    # Save metadata
    import json
    metadata = {
        "datasets": datasets,
        "total_tokens": writer.total_tokens,
        "total_chunks": writer.total_chunks,
        "seq_length": seq_length,
        "num_shards": len(writer.shard_files),
        "shard_files": writer.shard_files,
        "dtype": "uint32",
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time

    print(f"\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"Total chunks: {writer.total_chunks:,}")
    print(f"Total tokens: {writer.total_tokens:,}")
    print(f"Shards: {len(writer.shard_files)}")
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
    parser.add_argument("--synth_pct", type=float, default=None,
                        help="Download only the first N%% of SYNTH shards (e.g., 30 for 30%)")

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
        synth_pct=args.synth_pct,
    )


if __name__ == "__main__":
    main()
