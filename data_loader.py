"""
Medical LLM Pretraining Data Loader

Loads and combines medical datasets for pretraining:
- SYNTH (reasoning chains from PleIAs)
- PubMed abstracts (ncbi/pubmed)
- Clinical guidelines (epfl-llm/guidelines)
- OpenMedText (medical textbooks)
- FineWeb-Edu (general education)

IMPORTANT: Avoids benchmark datasets (MedQA, MedMCQA, PubMedQA) to prevent contamination.
"""

import os
import random
from typing import Optional, List, Iterator, Dict, Any
from dataclasses import dataclass

# Disable hf_transfer to avoid issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from datasets import load_dataset, Dataset, IterableDataset, concatenate_datasets
import tiktoken
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""
    name: str
    hf_path: str
    hf_name: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    weight: float = 1.0
    max_samples: Optional[int] = None
    streaming: bool = True


# Dataset configurations (avoiding benchmarks)
MEDICAL_DATASETS = {
    # Test dataset - simple wikitext for initial testing
    "wikitext": DatasetConfig(
        name="WikiText (test)",
        hf_path="wikitext",
        hf_name="wikitext-2-raw-v1",
        text_field="text",
        weight=1.0,
        streaming=False,
    ),
    "synth": DatasetConfig(
        name="SYNTH Reasoning",
        hf_path="PleIAs/SYNTH",
        text_field="synthetic_reasoning",  # Use reasoning chains
        weight=0.40,
        streaming=True,
    ),
    "pubmed": DatasetConfig(
        name="PubMed Abstracts",
        hf_path="ncbi/pubmed",
        text_field="MedlineCitation.Article.Abstract.AbstractText",
        weight=0.30,
        streaming=True,
    ),
    "guidelines": DatasetConfig(
        name="Clinical Guidelines",
        hf_path="epfl-llm/guidelines",
        text_field="clean_text",
        weight=0.15,
        streaming=False,  # Small dataset
    ),
    "openmedtext": DatasetConfig(
        name="OpenMedText",
        hf_path="ywchoi/OpenMedText",
        text_field="text",
        weight=0.10,
        streaming=False,
    ),
    "fineweb_edu": DatasetConfig(
        name="FineWeb-Edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_name="sample-10BT",  # Use the 10B token sample
        text_field="text",
        weight=0.05,
        streaming=True,
    ),
}


class MedicalPretrainDataLoader:
    """
    Loads and combines medical datasets for LLM pretraining.

    Features:
    - Streaming support for large datasets
    - Weighted sampling from multiple sources
    - Tokenization with tiktoken
    - No benchmark contamination
    """

    def __init__(
        self,
        datasets: List[str] = None,
        tokenizer_name: str = "cl100k_base",  # GPT-4 tokenizer
        max_seq_length: int = 2048,
        seed: int = 42,
    ):
        """
        Initialize the data loader.

        Args:
            datasets: List of dataset names to use (default: all)
            tokenizer_name: tiktoken tokenizer name
            max_seq_length: Maximum sequence length for training
            seed: Random seed for reproducibility
        """
        self.datasets_to_use = datasets or list(MEDICAL_DATASETS.keys())
        self.max_seq_length = max_seq_length
        self.seed = seed

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

        random.seed(seed)

        print(f"Initialized MedicalPretrainDataLoader")
        print(f"  Tokenizer: {tokenizer_name} (vocab size: {self.vocab_size})")
        print(f"  Max seq length: {max_seq_length}")
        print(f"  Datasets: {self.datasets_to_use}")

    def _extract_text(self, example: Dict[str, Any], config: DatasetConfig) -> str:
        """Extract text from an example using the configured field path."""
        text_field = config.text_field

        # Handle nested fields (e.g., "MedlineCitation.Article.Abstract.AbstractText")
        value = example
        for key in text_field.split("."):
            if isinstance(value, dict):
                value = value.get(key, "")
            else:
                return ""

        # Handle lists
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)

        return str(value) if value else ""

    def _format_synth(self, example: Dict[str, Any]) -> str:
        """
        Format SYNTH dataset with question and reasoning (English only).

        Uses structured format with clear markers so model learns:
        1. To think step-by-step (reasoning before answer)
        2. The pattern: question -> think -> answer
        """
        # Filter for English only
        lang = example.get("language", "")
        if lang != "en":
            return ""  # Skip non-English

        query = example.get("query", "")
        reasoning = example.get("synthetic_reasoning", "")
        answer = example.get("synthetic_answer", "")

        # Structured format with clear markers for reasoning chain learning
        # The model learns: given a question, first reason, then answer
        if reasoning and answer:
            return f"""<|question|>
{query}
<|thinking|>
{reasoning}
<|answer|>
{answer}"""
        elif reasoning:
            return f"""<|question|>
{query}
<|thinking|>
{reasoning}"""
        return query

    def _format_pubmed(self, example: Dict[str, Any]) -> str:
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

    def _format_guidelines(self, example: Dict[str, Any]) -> str:
        """Format clinical guidelines."""
        title = example.get("title", "")
        text = example.get("clean_text", example.get("text", ""))
        source = example.get("source", "")

        if title and text:
            return f"Clinical Guideline: {title}\nSource: {source}\n\n{text}"
        return text

    def load_dataset_samples(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Load samples from a single dataset.

        Args:
            dataset_name: Name of the dataset to load
            max_samples: Maximum samples to yield

        Yields:
            Formatted text samples
        """
        if dataset_name not in MEDICAL_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = MEDICAL_DATASETS[dataset_name]
        print(f"\nLoading {config.name}...")

        try:
            # Load dataset (no trust_remote_code - deprecated)
            if config.streaming:
                ds = load_dataset(
                    config.hf_path,
                    config.hf_name,
                    split=config.split,
                    streaming=True,
                )
            else:
                ds = load_dataset(
                    config.hf_path,
                    config.hf_name,
                    split=config.split,
                )

            count = 0
            effective_max = max_samples or config.max_samples

            for example in ds:
                # Format based on dataset type
                if dataset_name == "synth":
                    text = self._format_synth(example)
                elif dataset_name == "pubmed":
                    text = self._format_pubmed(example)
                elif dataset_name == "guidelines":
                    text = self._format_guidelines(example)
                else:
                    text = self._extract_text(example, config)

                # Skip empty or very short texts
                if len(text) < 50:
                    continue

                yield text
                count += 1

                if effective_max and count >= effective_max:
                    break

            print(f"  Loaded {count} samples from {config.name}")

        except Exception as e:
            print(f"  Error loading {config.name}: {e}")
            return

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using tiktoken."""
        return self.tokenizer.encode(text, allowed_special=set())

    def create_training_chunks(
        self,
        texts: Iterator[str],
        add_eos: bool = True,
    ) -> Iterator[List[int]]:
        """
        Create fixed-length training chunks from texts.

        Args:
            texts: Iterator of text samples
            add_eos: Whether to add EOS token between documents

        Yields:
            Token chunks of max_seq_length
        """
        buffer = []
        eos_token = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        for text in texts:
            tokens = self.tokenize_text(text)

            if add_eos:
                tokens.append(eos_token)

            buffer.extend(tokens)

            # Yield complete chunks
            while len(buffer) >= self.max_seq_length:
                yield buffer[:self.max_seq_length]
                buffer = buffer[self.max_seq_length:]

        # Yield final partial chunk if substantial
        if len(buffer) > self.max_seq_length // 2:
            # Pad to full length
            buffer.extend([eos_token] * (self.max_seq_length - len(buffer)))
            yield buffer

    def create_mixed_dataset(
        self,
        max_tokens: int = 1_000_000,  # 1M tokens for testing
        shuffle_buffer: int = 10000,
    ) -> List[List[int]]:
        """
        Create a mixed dataset with weighted sampling from all sources.

        Args:
            max_tokens: Maximum total tokens to collect
            shuffle_buffer: Buffer size for shuffling

        Returns:
            List of tokenized chunks ready for training
        """
        print(f"\nCreating mixed dataset ({max_tokens:,} tokens target)...")

        # Calculate samples per dataset based on weights
        total_weight = sum(
            MEDICAL_DATASETS[d].weight
            for d in self.datasets_to_use
        )

        all_chunks = []
        total_tokens = 0

        for dataset_name in self.datasets_to_use:
            config = MEDICAL_DATASETS[dataset_name]

            # Calculate target tokens for this dataset
            weight_ratio = config.weight / total_weight
            target_tokens = int(max_tokens * weight_ratio)
            target_samples = target_tokens // self.max_seq_length + 1

            print(f"\n{config.name}: targeting {target_tokens:,} tokens ({weight_ratio:.1%})")

            try:
                texts = self.load_dataset_samples(dataset_name, max_samples=target_samples * 2)
                chunks = list(self.create_training_chunks(texts))

                # Limit to target
                if len(chunks) > target_samples:
                    random.shuffle(chunks)
                    chunks = chunks[:target_samples]

                all_chunks.extend(chunks)
                total_tokens += len(chunks) * self.max_seq_length

                print(f"  Added {len(chunks)} chunks ({len(chunks) * self.max_seq_length:,} tokens)")

            except Exception as e:
                print(f"  Skipping {config.name}: {e}")

        # Final shuffle
        random.shuffle(all_chunks)

        print(f"\nTotal: {len(all_chunks)} chunks, {total_tokens:,} tokens")
        return all_chunks

    def save_binary(self, chunks: List[List[int]], output_path: str):
        """Save tokenized data as binary file for fast loading."""
        import numpy as np

        data = np.array(chunks, dtype=np.uint16)
        data.tofile(output_path)

        print(f"Saved {len(chunks)} chunks to {output_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def test_data_loading():
    """Test loading a small sample from each dataset."""
    print("=" * 60)
    print("Testing Medical Dataset Loading")
    print("=" * 60)

    loader = MedicalPretrainDataLoader(
        max_seq_length=512,  # Shorter for testing
    )

    # Test each dataset individually
    for dataset_name in ["guidelines", "openmedtext"]:  # Start with smaller ones
        print(f"\n{'='*40}")
        print(f"Testing: {dataset_name}")
        print("=" * 40)

        try:
            samples = list(loader.load_dataset_samples(dataset_name, max_samples=3))

            for i, sample in enumerate(samples):
                print(f"\n--- Sample {i+1} ---")
                print(sample[:500] + "..." if len(sample) > 500 else sample)

        except Exception as e:
            print(f"Error: {e}")

    # Test mixed dataset creation
    print(f"\n{'='*60}")
    print("Testing Mixed Dataset Creation")
    print("=" * 60)

    try:
        # Very small test
        chunks = loader.create_mixed_dataset(max_tokens=50_000)
        print(f"\nSuccess! Created {len(chunks)} training chunks")

        # Show token distribution
        print(f"\nSample chunk tokens (first 50): {chunks[0][:50]}")
        print(f"Decoded: {loader.tokenizer.decode(chunks[0][:100])[:200]}...")

    except Exception as e:
        print(f"Error creating mixed dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loading()
