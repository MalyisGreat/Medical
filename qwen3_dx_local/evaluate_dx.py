#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

DATASET_VERSIONS = ["v2", "v1"]

def dataset_files(version: str) -> Dict[str, str]:
    prefix = f"qwen3_verifiable_dx_dataset_{version}"
    return {
        "train": f"{prefix}_train.jsonl",
        "val": f"{prefix}_val.jsonl",
        "test": f"{prefix}_test.jsonl",
    }

def try_apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    # Qwen3 supports enable_thinking switch; we disable it for 1-letter answers. :contentReference[oaicite:3]{index=3}
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizer signature fallback
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

def parse_choice_letter(text: str) -> Optional[str]:
    m = re.search(r"[A-E]", text.strip().upper())
    return m.group(0) if m else None

def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def resolve_data_files(data_dir: str, dataset_version: str) -> Dict[str, str]:
    def files_exist(base: Path, files: Dict[str, str]) -> bool:
        return all((base / name).exists() for name in files.values())

    base = Path(data_dir).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent

    versions = DATASET_VERSIONS if dataset_version == "auto" else [dataset_version]
    for v in versions:
        files = dataset_files(v)
        if files_exist(base, files):
            return {k: str(base / v) for k, v in files.items()}
        if files_exist(script_dir, files):
            return {k: str(script_dir / v) for k, v in files.items()}

    raise FileNotFoundError(
        f"Dataset files not found in '{base}' or '{script_dir}'. "
        "Expected qwen3_verifiable_dx_dataset_v2_*.jsonl or v1. "
        "Pass --data_dir pointing to the folder with the dataset files."
    )

@torch.no_grad()
def generate_completions(model, tokenizer, prompts: List[str], max_new_tokens: int) -> List[str]:
    device = pick_device()
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completions = []
    for j in range(gen.size(0)):
        in_len = int(enc["attention_mask"][j].sum().item())
        out_ids = gen[j, in_len:]
        completions.append(tokenizer.decode(out_ids, skip_special_tokens=True))
    return completions

@torch.no_grad()
def eval_mcq(model, tokenizer, ds, task_type: str, max_samples: int, batch_size: int, max_new_tokens: int):
    device = pick_device()
    model.eval()

    # Filter
    ds = ds.filter(lambda x: x["task_type"] == task_type)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0

    t0 = time.time()
    # Simple batching
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i+batch_size]
        prompts = []
        gold = []
        for ex in batch:
            msgs = ex["messages"]
            # prompt = system+user only; generation prompt appended
            prompt_msgs = msgs[:-1]
            prompt_text = try_apply_chat_template(tokenizer, prompt_msgs, add_generation_prompt=True)
            prompts.append(prompt_text)
            gold.append(ex["messages"][-1]["content"].strip())

        completions = generate_completions(model, tokenizer, prompts, max_new_tokens)

        for out_text, g in zip(completions, gold):
            pred = parse_choice_letter(out_text)
            g = parse_choice_letter(g)

            if pred is not None and g is not None and pred == g:
                correct += 1
            total += 1

    dt_s = time.time() - t0
    acc = correct / max(total, 1)
    return {
        "task": task_type,
        "n": total,
        "correct": correct,
        "accuracy": acc,
        "seconds": dt_s,
        "samples_per_sec": total / max(dt_s, 1e-9),
    }

@torch.no_grad()
def eval_label(model, tokenizer, ds, task_type: str, max_samples: int, batch_size: int, max_new_tokens: int):
    model.eval()
    ds = ds.filter(lambda x: x["task_type"] == task_type)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    t0 = time.time()

    for i in range(0, len(ds), batch_size):
        batch = ds[i:i+batch_size]
        prompts = []
        gold = []
        for ex in batch:
            msgs = ex["messages"]
            prompt_text = try_apply_chat_template(tokenizer, msgs[:-1], add_generation_prompt=True)
            prompts.append(prompt_text)
            gold.append(ex.get("dx_label", ex["messages"][-1]["content"]))

        completions = generate_completions(model, tokenizer, prompts, max_new_tokens)
        for out_text, g in zip(completions, gold):
            pred = normalize_label(out_text)
            gt = normalize_label(g)
            if pred and pred == gt:
                correct += 1
            total += 1

    dt_s = time.time() - t0
    acc = correct / max(total, 1)
    return {
        "task": task_type,
        "n": total,
        "correct": correct,
        "accuracy": acc,
        "seconds": dt_s,
        "samples_per_sec": total / max(dt_s, 1e-9),
    }

@torch.no_grad()
def eval_json(model, tokenizer, ds, task_type: str, max_samples: int, batch_size: int, max_new_tokens: int):
    model.eval()
    ds = ds.filter(lambda x: x["task_type"] == task_type)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    dx_correct = 0
    triage_correct = 0
    both_correct = 0
    total = 0
    t0 = time.time()

    for i in range(0, len(ds), batch_size):
        batch = ds[i:i+batch_size]
        prompts = []
        gold_dx = []
        gold_triage = []
        for ex in batch:
            msgs = ex["messages"]
            prompt_text = try_apply_chat_template(tokenizer, msgs[:-1], add_generation_prompt=True)
            prompts.append(prompt_text)
            gold_dx.append(ex.get("dx_label", ""))
            gold_triage.append(ex.get("triage", ""))

        completions = generate_completions(model, tokenizer, prompts, max_new_tokens)
        for out_text, gd, gt in zip(completions, gold_dx, gold_triage):
            obj = extract_json_obj(out_text)
            if isinstance(obj, dict):
                pred_dx = normalize_label(str(obj.get("dx", "")))
                pred_tri = normalize_label(str(obj.get("triage", "")))
                gdx = normalize_label(gd)
                gtri = normalize_label(gt)
                dx_ok = pred_dx and pred_dx == gdx
                tri_ok = pred_tri and pred_tri == gtri
                if dx_ok:
                    dx_correct += 1
                if tri_ok:
                    triage_correct += 1
                if dx_ok and tri_ok:
                    both_correct += 1
            total += 1

    dt_s = time.time() - t0
    return {
        "task": task_type,
        "n": total,
        "dx_accuracy": dx_correct / max(total, 1),
        "triage_accuracy": triage_correct / max(total, 1),
        "both_accuracy": both_correct / max(total, 1),
        "seconds": dt_s,
        "samples_per_sec": total / max(dt_s, 1e-9),
    }

def load_model_and_tokenizer(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    return model, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id or local folder")
    ap.add_argument("--data_dir", type=str, default=".", help="Folder containing qwen3_verifiable_dx_dataset_v*_*.jsonl")
    ap.add_argument("--split", type=str, choices=["train","val","test"], default="test")
    ap.add_argument("--task_type", type=str, default="dx_mcq_letter_v2",
                    choices=[
                        "dx_mcq", "dx_label", "dx_abstain_label", "dx_triage_json", "dx_triage_json_abstain",
                        "dx_mcq_letter_v2", "dx_mcq_label_v2", "dx_free_label_v2",
                        "dx_abstain_options_v2", "dx_abstain_free_v2",
                        "all",
                    ])
    ap.add_argument("--dataset_version", type=str, default="v2", choices=["auto", "v1", "v2"])
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = pick_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    data_files = resolve_data_files(args.data_dir, args.dataset_version)
    ds = load_dataset("json", data_files=data_files, split=args.split)
    chosen_version = "v2" if "v2" in Path(data_files["train"]).name else "v1"

    metrics = []
    if args.task_type in {"dx_mcq", "dx_mcq_letter_v2"}:
        metrics.append(eval_mcq(model, tokenizer, ds, args.task_type, args.max_samples, args.batch_size, args.max_new_tokens))
    elif args.task_type in {"dx_label", "dx_abstain_label", "dx_mcq_label_v2", "dx_free_label_v2", "dx_abstain_options_v2", "dx_abstain_free_v2"}:
        metrics.append(eval_label(model, tokenizer, ds, args.task_type, args.max_samples, args.batch_size, args.max_new_tokens))
    elif args.task_type in {"dx_triage_json", "dx_triage_json_abstain"}:
        metrics.append(eval_json(model, tokenizer, ds, args.task_type, args.max_samples, args.batch_size, args.max_new_tokens))
    else:
        if chosen_version == "v1":
            metrics.append(eval_mcq(model, tokenizer, ds, "dx_mcq", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_label(model, tokenizer, ds, "dx_label", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_label(model, tokenizer, ds, "dx_abstain_label", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_json(model, tokenizer, ds, "dx_triage_json", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_json(model, tokenizer, ds, "dx_triage_json_abstain", args.max_samples, args.batch_size, args.max_new_tokens))
        else:
            metrics.append(eval_mcq(model, tokenizer, ds, "dx_mcq_letter_v2", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_label(model, tokenizer, ds, "dx_mcq_label_v2", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_label(model, tokenizer, ds, "dx_free_label_v2", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_label(model, tokenizer, ds, "dx_abstain_options_v2", args.max_samples, args.batch_size, args.max_new_tokens))
            metrics.append(eval_label(model, tokenizer, ds, "dx_abstain_free_v2", args.max_samples, args.batch_size, args.max_new_tokens))

    if len(metrics) == 1:
        print(json.dumps(metrics[0], indent=2))
    else:
        print(json.dumps({"results": metrics}, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        out_obj = metrics[0] if len(metrics) == 1 else {"results": metrics}
        Path(args.out_json).write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
