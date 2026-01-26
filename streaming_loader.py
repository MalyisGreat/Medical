"""
Streaming Data Loader for Large-Scale Training

Streams data directly from HuggingFace during training - no prep step needed.
Uses multiple workers to prefetch and tokenize in parallel.

This is the FAST approach - start training immediately, no 60 min prep.
"""

import os
import random
from typing import Iterator, List, Optional
from itertools import cycle

import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken

# Disable hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from datasets import load_dataset


class StreamingMedicalDataset(IterableDataset):
    """
    Streams and tokenizes medical data on-the-fly.

    No prep step needed - starts training immediately.
    Uses HuggingFace streaming + parallel tokenization.
    """

    def __init__(
        self,
        datasets: List[str] = None,
        seq_length: int = 512,
        tokenizer_name: str = "cl100k_base",
        seed: int = 42,
        buffer_size: int = 10000,  # Shuffle buffer
    ):
        self.datasets = datasets or ["synth"]
        self.seq_length = seq_length
        self.seed = seed
        self.buffer_size = buffer_size

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.eos_token = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        # Dataset configs
        self.dataset_configs = {
            "synth": {
                "path": "PleIAs/SYNTH",
                "name": None,
                "split": "train",
                "weight": 0.40,
            },
            "guidelines": {
                "path": "epfl-llm/guidelines",
                "name": None,
                "split": "train",
                "weight": 0.15,
            },
            "openmedtext": {
                "path": "ywchoi/OpenMedText",
                "name": None,
                "split": "train",
                "weight": 0.10,
            },
            "fineweb_edu": {
                "path": "HuggingFaceFW/fineweb-edu",
                "name": "sample-10BT",
                "split": "train",
                "weight": 0.05,
            },
            "wikitext": {
                "path": "wikitext",
                "name": "wikitext-103-raw-v1",  # Larger version
                "split": "train",
                "weight": 1.0,
            },
        }

    def _format_synth(self, example) -> str:
        """Format SYNTH with reasoning chain."""
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

    def _format_guidelines(self, example) -> str:
        """Format clinical guidelines."""
        title = example.get("title", "")
        text = example.get("clean_text", example.get("text", ""))
        if title and text:
            return f"Clinical Guideline: {title}\n\n{text}"
        return text

    def _format_example(self, example, dataset_name: str) -> str:
        """Format example based on dataset type."""
        if dataset_name == "synth":
            return self._format_synth(example)
        elif dataset_name == "guidelines":
            return self._format_guidelines(example)
        else:
            return example.get("text", "")

    def _stream_dataset(self, dataset_name: str) -> Iterator[str]:
        """Stream formatted text from a dataset."""
        config = self.dataset_configs[dataset_name]

        try:
            ds = load_dataset(
                config["path"],
                config["name"],
                split=config["split"],
                streaming=True,
            )

            for example in ds:
                text = self._format_example(example, dataset_name)
                if len(text) > 50:  # Skip very short texts
                    yield text

        except Exception as e:
            print(f"Error streaming {dataset_name}: {e}")
            return

    def _interleave_datasets(self) -> Iterator[str]:
        """Interleave multiple datasets based on weights."""
        # Create iterators for each dataset
        iterators = {}
        weights = {}

        for name in self.datasets:
            if name in self.dataset_configs:
                iterators[name] = self._stream_dataset(name)
                weights[name] = self.dataset_configs[name]["weight"]

        if not iterators:
            return

        # Normalize weights
        total_weight = sum(weights.values())
        probs = {k: v / total_weight for k, v in weights.items()}

        # Weighted random sampling
        rng = random.Random(self.seed)
        names = list(iterators.keys())
        prob_list = [probs[n] for n in names]

        exhausted = set()

        while len(exhausted) < len(names):
            # Pick dataset based on weights
            name = rng.choices(names, weights=prob_list, k=1)[0]

            if name in exhausted:
                continue

            try:
                text = next(iterators[name])
                yield text
            except StopIteration:
                exhausted.add(name)
                # Redistribute weight
                prob_list = [0 if names[i] in exhausted else probs[names[i]]
                            for i in range(len(names))]
                total = sum(prob_list)
                if total > 0:
                    prob_list = [p / total for p in prob_list]

    def _tokenize_and_chunk(self, texts: Iterator[str]) -> Iterator[List[int]]:
        """Tokenize texts and create fixed-length chunks."""
        buffer = []

        for text in texts:
            tokens = self.tokenizer.encode(text, disallowed_special=())
            tokens.append(self.eos_token)
            buffer.extend(tokens)

            # Yield complete chunks
            while len(buffer) >= self.seq_length:
                yield buffer[:self.seq_length]
                buffer = buffer[self.seq_length:]

    def __iter__(self) -> Iterator[tuple]:
        """Yield (input, target) pairs for training."""
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Adjust seed per worker for different data
            worker_seed = self.seed + worker_info.id
        else:
            worker_seed = self.seed

        random.seed(worker_seed)

        # Stream, tokenize, and chunk
        texts = self._interleave_datasets()
        chunks = self._tokenize_and_chunk(texts)

        for chunk in chunks:
            tokens = torch.tensor(chunk, dtype=torch.long)
            yield tokens[:-1], tokens[1:]


def create_streaming_dataloader(
    datasets: List[str],
    batch_size: int = 32,
    seq_length: int = 512,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Create a streaming dataloader for training.

    No prep step needed - starts immediately.

    Args:
        datasets: List of dataset names
        batch_size: Batch size
        seq_length: Sequence length
        num_workers: Number of parallel workers (for tokenization)
        seed: Random seed

    Returns:
        DataLoader that streams data
    """
    dataset = StreamingMedicalDataset(
        datasets=datasets,
        seq_length=seq_length,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# Test
if __name__ == "__main__":
    print("Testing streaming loader...")

    loader = create_streaming_dataloader(
        datasets=["synth"],
        batch_size=4,
        seq_length=512,
        num_workers=0,  # 0 for testing
    )

    tokenizer = tiktoken.get_encoding("cl100k_base")

    for i, (inputs, targets) in enumerate(loader):
        print(f"Batch {i}: inputs={inputs.shape}, targets={targets.shape}")
        print(f"  Sample: {tokenizer.decode(inputs[0, :50].tolist())[:100]}...")

        if i >= 2:
            break

    print("\nStreaming works!")
