"""
Medical Benchmark Evaluation Script

Evaluates trained models on medical benchmarks:
- MedQA (USMLE-style questions)
- MedMCQA (Medical entrance exam)
- PubMedQA (Yes/No/Maybe from abstracts)

Usage:
    # Quick test (10 questions per benchmark)
    python evaluate.py --checkpoint output/checkpoint_10000.pt --num_questions 10

    # Full evaluation
    python evaluate.py --checkpoint output/checkpoint_10000.pt --num_questions 500
"""

import os
import argparse
import random
from typing import List, Dict, Optional, Tuple
import json

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn.functional as F
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

from model import (
    MedicalLLM, ModelConfig,
    get_tiny_config, get_small_config, get_medium_config, get_baguette_config
)


def get_config(model_name: str) -> ModelConfig:
    """Get model config by name."""
    configs = {
        "tiny": get_tiny_config,
        "small": get_small_config,
        "medium": get_medium_config,
        "baguette": get_baguette_config,
    }
    return configs[model_name]()


class MedicalBenchmarkEvaluator:
    """
    Evaluates language models on medical benchmarks.

    Uses likelihood-based scoring: the model assigns probability to each
    answer choice, and we pick the most likely one.
    """

    def __init__(
        self,
        model: MedicalLLM,
        tokenizer,
        device: str = "cuda",
        max_seq_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_length = max_seq_length
        self.model.eval()

    def score_completion(self, prompt: str, completion: str) -> float:
        """
        Score how likely the model thinks the completion follows the prompt.

        Returns log-probability of the completion given the prompt.
        """
        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt, allowed_special=set())
        completion_tokens = self.tokenizer.encode(completion, allowed_special=set())

        # Combine and truncate if needed
        all_tokens = prompt_tokens + completion_tokens
        if len(all_tokens) > self.max_seq_length:
            # Keep end of prompt + completion
            overflow = len(all_tokens) - self.max_seq_length
            prompt_tokens = prompt_tokens[overflow:]
            all_tokens = prompt_tokens + completion_tokens

        # Create input tensor
        input_ids = torch.tensor([all_tokens], device=self.device)

        with torch.no_grad():
            logits, _ = self.model(input_ids)

        # Get log probabilities for completion tokens only
        log_probs = F.log_softmax(logits, dim=-1)

        # Sum log probs for completion tokens
        prompt_len = len(prompt_tokens)
        total_log_prob = 0.0

        for i, token_id in enumerate(completion_tokens):
            pos = prompt_len + i - 1  # Position of the token we're predicting
            if pos >= 0 and pos < log_probs.size(1):
                total_log_prob += log_probs[0, pos, token_id].item()

        # Normalize by length
        return total_log_prob / max(len(completion_tokens), 1)

    def evaluate_multiple_choice(
        self,
        question: str,
        choices: List[str],
        correct_idx: int,
    ) -> Tuple[bool, int]:
        """
        Evaluate a multiple choice question.

        Returns (is_correct, predicted_idx)
        """
        # Score each choice
        scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: "
            score = self.score_completion(prompt, choice)
            scores.append(score)

        # Pick best
        predicted_idx = max(range(len(scores)), key=lambda i: scores[i])
        is_correct = (predicted_idx == correct_idx)

        return is_correct, predicted_idx

    def evaluate_with_reasoning(
        self,
        question: str,
        choices: List[str],
        correct_idx: int,
    ) -> Tuple[bool, int]:
        """
        Evaluate using the model's reasoning format.

        Uses <|question|> <|thinking|> <|answer|> format.
        """
        # Format with reasoning markers
        prompt = f"<|question|>\n{question}\n"

        # Add choices to prompt
        choice_labels = ["A", "B", "C", "D", "E"]
        for i, choice in enumerate(choices):
            if i < len(choice_labels):
                prompt += f"{choice_labels[i]}. {choice}\n"

        prompt += "<|thinking|>\nLet me analyze each option carefully.\n<|answer|>\nThe correct answer is "

        # Score each answer label
        scores = []
        for i, label in enumerate(choice_labels[:len(choices)]):
            score = self.score_completion(prompt, label)
            scores.append(score)

        predicted_idx = max(range(len(scores)), key=lambda i: scores[i])
        is_correct = (predicted_idx == correct_idx)

        return is_correct, predicted_idx


def load_medqa(num_questions: int = 10) -> List[Dict]:
    """Load MedQA (USMLE) questions."""
    print(f"Loading MedQA ({num_questions} questions)...")

    try:
        # Try the main MedQA dataset
        ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="test", trust_remote_code=True)
    except Exception:
        try:
            # Alternative source
            ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        except Exception as e:
            print(f"Could not load MedQA: {e}")
            return []

    questions = []
    for i, item in enumerate(ds):
        if i >= num_questions:
            break

        # Handle different dataset formats
        if "question" in item:
            letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

            # Get choices
            choices = item.get("options", item.get("choices", []))
            if isinstance(choices, dict):
                # Options is dict like {'A': '...', 'B': '...'}
                choices = [choices.get(k, "") for k in ["A", "B", "C", "D", "E"] if k in choices]

            # Get correct answer index
            answer_idx = item.get("answer_idx", "A")
            if isinstance(answer_idx, str):
                correct_idx = letter_to_idx.get(answer_idx.upper(), 0)
            else:
                correct_idx = int(answer_idx) if answer_idx else 0

            q = {
                "question": item["question"],
                "choices": choices,
                "correct_idx": correct_idx,
            }

            questions.append(q)

    print(f"  Loaded {len(questions)} questions")
    return questions


def load_medmcqa(num_questions: int = 10) -> List[Dict]:
    """Load MedMCQA questions."""
    print(f"Loading MedMCQA ({num_questions} questions)...")

    try:
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    except Exception as e:
        print(f"Could not load MedMCQA: {e}")
        return []

    questions = []
    for i, item in enumerate(ds):
        if i >= num_questions:
            break

        # MedMCQA format: question, opa, opb, opc, opd, cop (correct option 0-3)
        choices = [
            item.get("opa", ""),
            item.get("opb", ""),
            item.get("opc", ""),
            item.get("opd", ""),
        ]

        correct_idx = item.get("cop", 0)  # cop is 0-indexed

        questions.append({
            "question": item["question"],
            "choices": choices,
            "correct_idx": max(0, min(correct_idx, 3)),  # Ensure valid range
        })

    print(f"  Loaded {len(questions)} questions")
    return questions


def load_pubmedqa(num_questions: int = 10) -> List[Dict]:
    """Load PubMedQA questions."""
    print(f"Loading PubMedQA ({num_questions} questions)...")

    try:
        ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        except Exception as e:
            print(f"Could not load PubMedQA: {e}")
            return []

    questions = []
    for i, item in enumerate(ds):
        if i >= num_questions:
            break

        # PubMedQA: context, question, answer (yes/no/maybe)
        context = item.get("context", item.get("CONTEXTS", ""))
        if isinstance(context, dict):
            # Some formats have context as dict with 'contexts' key
            context = context.get("contexts", str(context))
        if isinstance(context, list):
            context = " ".join(str(c) for c in context)
        context = str(context) if context else ""

        question = item.get("question", item.get("QUESTION", ""))
        answer = item.get("answer", item.get("final_decision", item.get("LONG_ANSWER", "")))

        # Map answer to index
        answer_map = {"yes": 0, "no": 1, "maybe": 2}
        if isinstance(answer, str):
            correct_idx = answer_map.get(answer.lower(), 0)
        else:
            correct_idx = 0

        context_preview = context[:500] if len(context) > 500 else context
        full_question = f"Context: {context_preview}...\n\nQuestion: {question}"

        questions.append({
            "question": full_question,
            "choices": ["Yes", "No", "Maybe"],
            "correct_idx": correct_idx,
        })

    print(f"  Loaded {len(questions)} questions")
    return questions


def run_evaluation(
    evaluator: MedicalBenchmarkEvaluator,
    questions: List[Dict],
    benchmark_name: str,
    use_reasoning: bool = True,
) -> Dict:
    """Run evaluation on a set of questions."""
    if not questions:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = 0
    total = len(questions)

    print(f"\nEvaluating {benchmark_name}...")

    for q in tqdm(questions, desc=benchmark_name):
        if use_reasoning:
            is_correct, _ = evaluator.evaluate_with_reasoning(
                q["question"],
                q["choices"],
                q["correct_idx"],
            )
        else:
            is_correct, _ = evaluator.evaluate_multiple_choice(
                q["question"],
                q["choices"],
                q["correct_idx"],
            )

        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate medical LLM on benchmarks")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["tiny", "small", "medium", "baguette"],
                        help="Model configuration")
    parser.add_argument("--num_questions", type=int, default=10,
                        help="Number of questions per benchmark (default: 10)")
    parser.add_argument("--benchmarks", type=str, default="medqa,medmcqa,pubmedqa",
                        help="Comma-separated list of benchmarks")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--no_reasoning", action="store_true",
                        help="Don't use reasoning format")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 60)
    print("Medical LLM Benchmark Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Questions per benchmark: {args.num_questions}")
    print(f"Device: {args.device}")
    print(f"Use reasoning format: {not args.no_reasoning}")

    # Load model
    print("\nLoading model...")
    config = get_config(args.model)
    model = MedicalLLM(config)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # Assume it's just the state dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Create evaluator
    evaluator = MedicalBenchmarkEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_seq_length=config.max_seq_length,
    )

    # Load benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    benchmark_loaders = {
        "medqa": load_medqa,
        "medmcqa": load_medmcqa,
        "pubmedqa": load_pubmedqa,
    }

    results = {}

    print("\n" + "-" * 60)
    print("Loading Benchmarks")
    print("-" * 60)

    for benchmark in benchmarks:
        if benchmark in benchmark_loaders:
            questions = benchmark_loaders[benchmark](args.num_questions)

            result = run_evaluation(
                evaluator,
                questions,
                benchmark,
                use_reasoning=not args.no_reasoning,
            )

            results[benchmark] = result

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    total_correct = 0
    total_questions = 0

    for benchmark, result in results.items():
        acc = result["accuracy"] * 100
        correct = result["correct"]
        total = result["total"]

        print(f"{benchmark:15s}: {acc:5.1f}% ({correct}/{total})")

        total_correct += correct
        total_questions += total

    if total_questions > 0:
        overall_acc = total_correct / total_questions * 100
        print("-" * 40)
        print(f"{'Overall':15s}: {overall_acc:5.1f}% ({total_correct}/{total_questions})")

    # Random baseline for comparison
    print("\n(Random baseline: ~25% for 4-choice, ~33% for 3-choice)")

    # Save results
    output_file = args.checkpoint.replace(".pt", "_eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "model": args.model,
            "num_questions": args.num_questions,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
