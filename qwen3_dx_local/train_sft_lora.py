#!/usr/bin/env python3
import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import inspect
from peft import LoraConfig, get_peft_model

DATASET_VERSIONS = ["v2", "v1"]

def dataset_files(version: str) -> Dict[str, str]:
    prefix = f"qwen3_verifiable_dx_dataset_{version}"
    return {
        "train": f"{prefix}_train.jsonl",
        "val": f"{prefix}_val.jsonl",
        "test": f"{prefix}_test.jsonl",
    }

DEFAULT_TASKS = ",".join([
    "dx_mcq_letter_v2",
    "dx_mcq_label_v2",
    "dx_free_label_v2",
    "dx_abstain_options_v2",
    "dx_abstain_free_v2",
])

PROFILE_DEFAULTS = {
    "h100": {
        "max_len": 2048,
        "batch_size": 32,
        "eval_batch_size": 32,
        "grad_accum": 1,
    }
}

def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def system_info() -> Dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        p = torch.cuda.get_device_properties(0)
        info["gpu_vram_gb"] = round(p.total_memory / (1024**3), 2)
    return info

def try_apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    # Qwen3: disable thinking for classification. :contentReference[oaicite:7]{index=7}
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

def tokenize_row(row: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
    messages = row["messages"]
    prompt_text = try_apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = try_apply_chat_template(tokenizer, messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    enc = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_len)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", [1] * len(input_ids))

    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def parse_task_types(val: str) -> Set[str]:
    items = [v.strip() for v in val.split(",") if v.strip()]
    return set(items)

def filter_kwargs_for_signature(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def apply_profile(args) -> None:
    if args.profile and args.profile in PROFILE_DEFAULTS:
        cfg = PROFILE_DEFAULTS[args.profile]
        args.max_len = cfg.get("max_len", args.max_len)
        args.batch_size = cfg.get("batch_size", args.batch_size)
        args.eval_batch_size = cfg.get("eval_batch_size", args.eval_batch_size)
        args.grad_accum = cfg.get("grad_accum", args.grad_accum)

def resolve_dtype(args, device: str) -> Tuple[torch.dtype, bool, bool]:
    if device != "cuda":
        return torch.float32, False, False

    if args.dtype == "bf16":
        return torch.bfloat16, True, False
    if args.dtype == "fp16":
        return torch.float16, False, True
    if args.dtype == "fp32":
        return torch.float32, False, False

    # auto
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--data_dir", type=str, default=".", help="Folder containing qwen3_verifiable_dx_dataset_v*_*.jsonl")
    ap.add_argument("--dataset_version", type=str, default="v2", choices=["auto", "v1", "v2"])
    ap.add_argument("--benchmark_after_sft", action="store_true", default=True,
                    help="Run MedQA/MedMCQA/PubMedQA after SFT (default: on)")
    ap.add_argument("--benchmark_before_sft", action="store_true", help="Run MedQA/MedMCQA/PubMedQA before SFT")
    ap.add_argument("--benchmark_num_questions", type=int, default=20)
    ap.add_argument("--benchmark_list", type=str, default="medqa,medmcqa,pubmedqa")
    ap.add_argument("--benchmark_system_prompt", type=str, default="You are a medical expert.")
    ap.add_argument("--benchmark_allow_train_fallback", action="store_true")
    ap.add_argument("--output_dir", type=str, default="runs/sft_dx")
    ap.add_argument("--max_len", type=int, default=384)

    # Light compute defaults for 2080 Ti
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--eval_steps", type=int, default=250)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj")

    ap.add_argument("--task_types", type=str, default=DEFAULT_TASKS, help="Comma-separated task types to include")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=0, help="0 = no limit")
    ap.add_argument("--merge_and_save", action="store_true", help="Save a merged fp16 model for easy inference")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Reduce VRAM at cost of speed")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--profile", type=str, default="", choices=["", "h100"], help="Apply H100-friendly defaults")

    args = ap.parse_args()
    apply_profile(args)

    run_dir = Path(args.output_dir) / now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "system_info.json").write_text(json.dumps(system_info(), indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"[train_sft] device={device}")
    print("[train_sft] system info:", json.dumps(system_info(), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype, use_bf16, use_fp16 = resolve_dtype(args, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # LoRA config
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Load data
    data_files = resolve_data_files(args.data_dir, args.dataset_version)
    ds_train = load_dataset("json", data_files=data_files, split="train")
    ds_val = load_dataset("json", data_files=data_files, split="val")

    task_set = parse_task_types(args.task_types)
    print(f"[train_sft] tasks: {sorted(task_set)}")
    ds_train = ds_train.filter(lambda x: x["task_type"] in task_set)
    ds_val = ds_val.filter(lambda x: x["task_type"] in task_set)
    if len(ds_train) == 0:
        raise ValueError("No training examples after filtering. Check --dataset_version and --task_types.")
    if len(ds_val) == 0:
        raise ValueError("No validation examples after filtering. Check --dataset_version and --task_types.")

    if args.limit_train > 0:
        ds_train = ds_train.select(range(min(args.limit_train, len(ds_train))))

    # Tokenize
    ds_train_tok = ds_train.map(lambda r: tokenize_row(r, tokenizer, args.max_len), remove_columns=ds_train.column_names)
    ds_val_tok = ds_val.map(lambda r: tokenize_row(r, tokenizer, args.max_len), remove_columns=ds_val.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, label_pad_token_id=-100, pad_to_multiple_of=8)

    # transformers renamed evaluation_strategy -> eval_strategy in newer versions
    ta_kwargs = dict(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=["tensorboard"],
        logging_dir=str(run_dir / "tb"),
        seed=args.seed,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"

    ta_kwargs = filter_kwargs_for_signature(TrainingArguments.__init__, ta_kwargs)
    training_args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok,
        data_collator=collator,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    def _run_benchmarks(tag: str):
        try:
            import sys
            repo_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repo_root / "medical-llm-finetune"))
            import benchmark as med_bench
        except Exception as exc:
            print(f"[train_sft] benchmark import failed: {exc}")
            return

        benchmarks = [b.strip() for b in args.benchmark_list.split(",") if b.strip()]
        num_q = args.benchmark_num_questions
        strict_holdout = not args.benchmark_allow_train_fallback
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()

        # preload questions once per call to avoid repeated dataset downloads per benchmark
        questions = {}
        for name in benchmarks:
            if name == "medqa":
                questions[name] = med_bench.load_medqa(num_q)
            elif name == "medmcqa":
                questions[name] = med_bench.load_medmcqa(num_q)
            elif name == "pubmedqa":
                questions[name] = med_bench.load_pubmedqa(num_q, strict_holdout=strict_holdout)

        results = {}
        for name in benchmarks:
            qs = questions.get(name, [])
            if name in ("medqa", "medmcqa"):
                results[name] = med_bench._evaluate_mcq(model, tokenizer, device, args.benchmark_system_prompt, qs)
            elif name == "pubmedqa":
                results[name] = med_bench._evaluate_yes_no(model, tokenizer, device, args.benchmark_system_prompt, qs)

        out_path = run_dir / f"benchmark_{tag}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[train_sft] benchmark {tag}: {json.dumps(results)}")

    if args.benchmark_before_sft:
        _run_benchmarks("before_sft")

    # Train
    trainer.train()

    # Save adapter + tokenizer
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save a merged model (optional; convenient for inference/eval)
    if args.merge_and_save:
        merged_dir = run_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

    print(f"[train_sft] done. Run dir: {run_dir}")
    print(f"[train_sft] adapter: {adapter_dir}")
    if args.merge_and_save:
        print(f"[train_sft] merged:  {run_dir / 'merged'}")

    if args.benchmark_after_sft:
        _run_benchmarks("after_sft")

if __name__ == "__main__":
    main()
