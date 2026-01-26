#!/usr/bin/env python3
"""
Holdout benchmark evaluation for medical QA.

Scores models on held-out splits using likelihood-based multiple-choice scoring.
"""

from typing import Dict, List, Tuple, Optional
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_prompt(tokenizer, system_prompt: str, question: str, options: List[str]) -> str:
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    user = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n{user}"


def _score_completion(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None or max_len <= 0:
        max_len = getattr(tokenizer, "model_max_length", 2048)

    all_tokens = prompt_tokens + completion_tokens
    if len(all_tokens) > max_len:
        overflow = len(all_tokens) - max_len
        prompt_tokens = prompt_tokens[overflow:]
        all_tokens = prompt_tokens + completion_tokens

    input_ids = torch.tensor([all_tokens], device=device)
    with torch.inference_mode():
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_tokens)
    total_log_prob = 0.0
    for i, token_id in enumerate(completion_tokens):
        pos = prompt_len + i - 1
        if 0 <= pos < log_probs.size(1):
            total_log_prob += log_probs[0, pos, token_id].item()

    return total_log_prob / max(len(completion_tokens), 1)


def _evaluate_mcq(model, tokenizer, device: str, system_prompt: str, questions: List[Dict]) -> Dict:
    correct = 0
    total = len(questions)
    for q in questions:
        prompt = _build_prompt(tokenizer, system_prompt, q["question"], q["choices"])
        scores = []
        for i in range(len(q["choices"])):
            label = chr(65 + i)
            score = _score_completion(model, tokenizer, prompt, label, device)
            scores.append(score)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        if pred == q["correct_idx"]:
            correct += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def _evaluate_yes_no(model, tokenizer, device: str, system_prompt: str, questions: List[Dict]) -> Dict:
    correct = 0
    total = len(questions)
    labels = ["Yes", "No", "Maybe"]
    for q in questions:
        prompt = _build_prompt(tokenizer, system_prompt, q["question"], labels)
        scores = []
        for label in labels:
            score = _score_completion(model, tokenizer, prompt, label, device)
            scores.append(score)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        if pred == q["correct_idx"]:
            correct += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def load_medqa(num_questions: Optional[int] = 100) -> List[Dict]:
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception:
        ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test")

    questions = []
    for i, item in enumerate(ds):
        if num_questions is not None and i >= num_questions:
            break
        if "question" in item:
            choices = item.get("options", item.get("choices", []))
            if isinstance(choices, dict):
                choices = [choices.get(k, "") for k in ["A", "B", "C", "D", "E"] if k in choices]
            answer_idx = item.get("answer_idx", item.get("label", "A"))
            if isinstance(answer_idx, str):
                correct_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(answer_idx.upper(), 0)
            else:
                correct_idx = int(answer_idx) if answer_idx is not None else 0
            questions.append({"question": item["question"], "choices": choices, "correct_idx": correct_idx})
    return questions


def load_medmcqa(num_questions: Optional[int] = 100) -> List[Dict]:
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    questions = []
    for i, item in enumerate(ds):
        if num_questions is not None and i >= num_questions:
            break
        choices = [item.get("opa", ""), item.get("opb", ""), item.get("opc", ""), item.get("opd", "")]
        correct_idx = int(item.get("cop", 0))
        questions.append({"question": item["question"], "choices": choices, "correct_idx": correct_idx})
    return questions


def load_pubmedqa(num_questions: Optional[int] = 100) -> List[Dict]:
    try:
        ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="test")
    except Exception:
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="test")
        except Exception:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    questions = []
    for i, item in enumerate(ds):
        if num_questions is not None and i >= num_questions:
            break
        context = item.get("context", item.get("CONTEXTS", ""))
        if isinstance(context, dict):
            context = context.get("contexts", str(context))
        if isinstance(context, list):
            context = " ".join(str(c) for c in context)
        context = str(context) if context else ""
        question = item.get("question", item.get("QUESTION", ""))
        answer = item.get("answer", item.get("final_decision", item.get("LONG_ANSWER", "")))
        answer_map = {"yes": 0, "no": 1, "maybe": 2}
        correct_idx = answer_map.get(str(answer).lower(), 0)
        full_question = f"Context: {context[:500]}...\n\nQuestion: {question}"
        questions.append({"question": full_question, "choices": ["Yes", "No", "Maybe"], "correct_idx": correct_idx})
    return questions


def evaluate_model(
    model,
    tokenizer,
    device: str,
    benchmarks: List[str],
    num_questions: Optional[int],
    system_prompt: str,
) -> Dict[str, Dict]:
    model.eval()
    results = {}
    for name in benchmarks:
        if name == "medqa":
            qs = load_medqa(num_questions)
            results[name] = _evaluate_mcq(model, tokenizer, device, system_prompt, qs)
        elif name == "medmcqa":
            qs = load_medmcqa(num_questions)
            results[name] = _evaluate_mcq(model, tokenizer, device, system_prompt, qs)
        elif name == "pubmedqa":
            qs = load_pubmedqa(num_questions)
            results[name] = _evaluate_yes_no(model, tokenizer, device, system_prompt, qs)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate medical QA benchmarks")
    parser.add_argument("--model", required=True, help="Base model or fine-tuned model path")
    parser.add_argument("--num_questions", type=int, default=100,
                        help="Questions per benchmark (0 = full split)")
    parser.add_argument("--benchmarks", type=str, default="medqa,medmcqa,pubmedqa",
                        help="Comma-separated benchmark names")
    parser.add_argument("--system_prompt", type=str, default="You are a medical expert.",
                        help="System prompt used for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    args = parser.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    ).to(device)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    num_questions = None if args.num_questions <= 0 else args.num_questions
    results = evaluate_model(model, tokenizer, device, benchmarks, num_questions, args.system_prompt)

    for name, res in results.items():
        acc = res["accuracy"] * 100
        print(f"{name:10s}: {acc:5.1f}% ({res['correct']}/{res['total']})")


if __name__ == "__main__":
    main()
