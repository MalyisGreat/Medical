"""
Medical Dataset Loader

Loads and preprocesses medical QA datasets for LLM fine-tuning:
- PubMedQA: Biomedical research question answering
- MedQA: Medical board exam questions (USMLE)
- MedMCQA: Medical entrance exam questions

Each dataset is formatted into instruction-following format for fine-tuning.
"""

from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Dict, List, Optional, Tuple
import random


class MedicalDatasetLoader:
    """
    Loads and formats medical QA datasets for instruction fine-tuning.

    Supports multiple datasets:
    - pubmedqa: Research-based QA from PubMed abstracts
    - medqa: USMLE-style medical board questions
    - medmcqa: Medical entrance exam MCQs
    """

    # Instruction templates for different question types
    TEMPLATES = {
        "mcq": """Answer the following medical question by selecting the correct option.

Question: {question}

Options:
{options}

Provide your answer with a brief explanation.""",

        "yes_no": """Based on the context provided, answer the question with Yes, No, or Maybe, and explain your reasoning.

Context: {context}

Question: {question}

Answer:""",

        "open": """Answer the following medical question thoroughly and accurately.

Question: {question}

Answer:"""
    }

    RESPONSE_TEMPLATE = """{answer}

Explanation: {explanation}"""

    def __init__(self, seed: int = 42, system_prompt: str = "You are a medical expert."):
        self.seed = seed
        self.system_prompt = system_prompt
        random.seed(seed)

    def load_pubmedqa(
        self,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load PubMedQA dataset.

        PubMedQA contains questions derived from PubMed article titles,
        with contexts from abstracts and yes/no/maybe answers.

        Args:
            split: 'train' or 'test'
            max_samples: Limit number of samples (useful for testing)
        """
        print(f"Loading PubMedQA ({split})...")

        # Load the labeled subset (has human annotations)
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        def format_pubmedqa(example):
            # Build context from the abstract
            context = " ".join(example.get("context", {}).get("contexts", []))
            if not context:
                context = example.get("abstract", "")

            question = example["question"]
            answer = example["final_decision"]  # yes/no/maybe
            long_answer = example.get("long_answer", "")

            # Format as instruction
            instruction = self.TEMPLATES["yes_no"].format(
                context=context[:2000],  # Truncate long contexts
                question=question
            )

            # Format response
            if long_answer:
                response = f"{answer.capitalize()}.\n\n{long_answer}"
            else:
                response = answer.capitalize()

            return {
                "instruction": instruction,
                "response": response,
                "source": "pubmedqa"
            }

        formatted = dataset.map(format_pubmedqa, remove_columns=dataset.column_names)
        print(f"  Loaded {len(formatted)} samples from PubMedQA")
        return formatted

    def load_medqa(
        self,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load MedQA dataset (USMLE-style questions).

        MedQA contains multiple-choice questions from medical licensing exams.

        Args:
            split: 'train', 'validation', or 'test'
            max_samples: Limit number of samples
        """
        print(f"Loading MedQA ({split})...")

        try:
            # Try the 4-option USMLE version first
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        except Exception:
            # Fallback to bigbio version
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        def format_medqa(example):
            question = example.get("question", example.get("sent1", ""))

            # Handle different dataset formats for options
            if "options" in example:
                options = example["options"]
                if isinstance(options, dict):
                    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
                else:
                    options_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
            elif "ending0" in example:
                # Alternative format
                options_text = "\n".join([
                    f"A) {example.get('ending0', '')}",
                    f"B) {example.get('ending1', '')}",
                    f"C) {example.get('ending2', '')}",
                    f"D) {example.get('ending3', '')}"
                ])
            else:
                options_text = ""

            # Get the answer
            answer_idx = example.get("answer_idx", example.get("label", 0))
            if isinstance(answer_idx, str):
                answer_letter = answer_idx.upper()
            else:
                answer_letter = chr(65 + int(answer_idx))

            # Get the answer text
            if "options" in example and isinstance(example["options"], dict):
                answer_text = example["options"].get(answer_letter, "")
            elif "options" in example:
                answer_text = example["options"][int(answer_idx)] if isinstance(answer_idx, int) else ""
            else:
                answer_text = example.get(f"ending{answer_idx}", "")

            instruction = self.TEMPLATES["mcq"].format(
                question=question,
                options=options_text
            )

            response = f"The correct answer is {answer_letter}) {answer_text}"

            return {
                "instruction": instruction,
                "response": response,
                "source": "medqa"
            }

        formatted = dataset.map(format_medqa, remove_columns=dataset.column_names)
        print(f"  Loaded {len(formatted)} samples from MedQA")
        return formatted

    def load_medmcqa(
        self,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load MedMCQA dataset.

        MedMCQA contains MCQs from Indian medical entrance exams (AIIMS, NEET).

        Args:
            split: 'train', 'validation', or 'test'
            max_samples: Limit number of samples
        """
        print(f"Loading MedMCQA ({split})...")

        dataset = load_dataset("openlifescienceai/medmcqa", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        def format_medmcqa(example):
            question = example["question"]

            options_text = "\n".join([
                f"A) {example['opa']}",
                f"B) {example['opb']}",
                f"C) {example['opc']}",
                f"D) {example['opd']}"
            ])

            answer_idx = example["cop"]  # 0-3
            answer_letter = chr(65 + answer_idx)
            answer_options = [example['opa'], example['opb'], example['opc'], example['opd']]
            answer_text = answer_options[answer_idx]

            # Include explanation if available
            explanation = example.get("exp", "")

            instruction = self.TEMPLATES["mcq"].format(
                question=question,
                options=options_text
            )

            if explanation:
                response = f"The correct answer is {answer_letter}) {answer_text}\n\nExplanation: {explanation}"
            else:
                response = f"The correct answer is {answer_letter}) {answer_text}"

            return {
                "instruction": instruction,
                "response": response,
                "source": "medmcqa"
            }

        formatted = dataset.map(format_medmcqa, remove_columns=dataset.column_names)
        print(f"  Loaded {len(formatted)} samples from MedMCQA")
        return formatted

    def load_combined(
        self,
        datasets: List[str] = ["pubmedqa", "medqa", "medmcqa"],
        split: str = "train",
        max_samples_per_dataset: Optional[int] = None,
        shuffle: bool = True
    ) -> Dataset:
        """
        Load and combine multiple medical datasets.

        Args:
            datasets: List of dataset names to include
            split: Data split to load
            max_samples_per_dataset: Max samples from each dataset
            shuffle: Whether to shuffle the combined dataset
        """
        all_datasets = []

        loaders = {
            "pubmedqa": self.load_pubmedqa,
            "medqa": self.load_medqa,
            "medmcqa": self.load_medmcqa
        }

        for name in datasets:
            if name in loaders:
                try:
                    ds = loaders[name](split=split, max_samples=max_samples_per_dataset)
                    all_datasets.append(ds)
                except Exception as e:
                    print(f"  Warning: Failed to load {name}: {e}")

        if not all_datasets:
            raise ValueError("No datasets could be loaded!")

        combined = concatenate_datasets(all_datasets)

        if shuffle:
            combined = combined.shuffle(seed=self.seed)

        print(f"\nCombined dataset: {len(combined)} total samples")
        return combined

    def format_for_training(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 2048,
        add_eos: bool = True
    ) -> Dataset:
        """
        Format dataset for training with proper chat template.

        Args:
            dataset: The dataset to format
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            add_eos: Whether to add EOS token
        """
        def format_example(example):
            if getattr(tokenizer, "chat_template", None):
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["response"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""

            if add_eos and tokenizer.eos_token:
                text += tokenizer.eos_token

            return {"text": text}

        return dataset.map(format_example)


def create_medical_dataset(
    max_samples: int = 5000,
    split: str = "train",
    datasets: List[str] = ["pubmedqa", "medqa", "medmcqa"]
) -> Dataset:
    """
    Convenience function to create a combined medical dataset.

    Args:
        max_samples: Total max samples (distributed across datasets)
        split: Data split
        datasets: Which datasets to include
    """
    loader = MedicalDatasetLoader()
    samples_per_dataset = max_samples // len(datasets)

    return loader.load_combined(
        datasets=datasets,
        split=split,
        max_samples_per_dataset=samples_per_dataset,
        shuffle=True
    )


if __name__ == "__main__":
    # Test the data loader
    print("Testing Medical Dataset Loader\n")
    print("=" * 50)

    loader = MedicalDatasetLoader()

    # Test loading individual datasets with small samples
    print("\n1. Testing PubMedQA...")
    try:
        pubmed = loader.load_pubmedqa(split="train", max_samples=5)
        print(f"   Sample instruction:\n{pubmed[0]['instruction'][:200]}...")
        print(f"   Sample response:\n{pubmed[0]['response'][:200]}...")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n2. Testing MedMCQA...")
    try:
        mcqa = loader.load_medmcqa(split="train", max_samples=5)
        print(f"   Sample instruction:\n{mcqa[0]['instruction'][:200]}...")
        print(f"   Sample response:\n{mcqa[0]['response'][:200]}...")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)
    print("Data loader test complete!")
