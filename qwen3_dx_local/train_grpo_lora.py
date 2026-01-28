#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

DATASET_VERSIONS = ["v2", "v1"]

def dataset_files(version: str) -> Dict[str, str]:
    prefix = f"qwen3_verifiable_dx_dataset_{version}"
    return {
        "train": f"{prefix}_train.jsonl",
        "val": f"{prefix}_val.jsonl",
        "test": f"{prefix}_test.jsonl",
    }

DEFAULT_TASKS = "dx_mcq_letter_v2"

PROFILE_DEFAULTS = {
    "h100": {
        "batch_size": 32,
        "grad_accum": 1,
        "num_generations": 8,
        "max_prompt_length": 1024,
        "max_completion_length": 32,
    }
}

def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def try_apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    # Disable thinking for verifiable 1-letter completions. :contentReference[oaicite:10]{index=10}
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

def parse_choice_letter(text: str) -> Optional[str]:
    m = re.search(r"[A-E]", text.strip().upper())
    return m.group(0) if m else None

def parse_strict_letter(text: str) -> Optional[str]:
    m = re.fullmatch(r"\s*([A-E])\s*", text.strip().upper())
    return m.group(1) if m else None

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

def score_one(task_type: str, completion: str, ground_truth: str, dx_label: str, triage: str) -> float:
    task_type = (task_type or "").strip()
    completion = completion or ""
    ground_truth = ground_truth or ""
    dx_label = dx_label or ""
    triage = triage or ""

    if task_type in {"dx_mcq", "dx_mcq_letter_v2"}:
        pred = parse_strict_letter(completion)
        gt = parse_strict_letter(ground_truth)
        if pred is None or gt is None:
            return -0.5
        return 1.0 if pred == gt else 0.0

    if task_type in {
        "dx_label",
        "dx_abstain_label",
        "dx_mcq_label_v2",
        "dx_free_label_v2",
        "dx_abstain_options_v2",
        "dx_abstain_free_v2",
    }:
        pred = normalize_label(completion)
        gt = normalize_label(dx_label or ground_truth)
        if not pred:
            return -0.5
        return 1.0 if pred == gt else 0.0

    if task_type in {"dx_triage_json", "dx_triage_json_abstain"}:
        obj = extract_json_obj(completion)
        if not isinstance(obj, dict):
            return -0.5
        pred_dx = normalize_label(str(obj.get("dx", "")))
        pred_triage = normalize_label(str(obj.get("triage", "")))
        gt_dx = normalize_label(dx_label)
        gt_triage = normalize_label(triage)
        reward = 0.0
        if pred_dx and pred_dx == gt_dx:
            reward += 0.5
        if pred_triage and pred_triage == gt_triage:
            reward += 0.5
        if set(obj.keys()) == {"dx", "triage"}:
            reward += 0.1
        return reward

    return 0.0

def reward_dx(prompts=None, completions=None, task_type=None, ground_truth=None, dx_label=None, triage=None, **kwargs) -> List[float]:
    rewards = []
    for i, c in enumerate(completions):
        t = task_type[i] if task_type is not None else ""
        gt = ground_truth[i] if ground_truth is not None else ""
        lbl = dx_label[i] if dx_label is not None else ""
        trg = triage[i] if triage is not None else ""
        rewards.append(score_one(t, c, gt, lbl, trg))
    return rewards

def parse_task_types(val: str) -> Set[str]:
    items = [v.strip() for v in val.split(",") if v.strip()]
    return set(items)

def apply_profile(args) -> None:
    if args.profile and args.profile in PROFILE_DEFAULTS:
        cfg = PROFILE_DEFAULTS[args.profile]
        args.batch_size = cfg.get("batch_size", args.batch_size)
        args.grad_accum = cfg.get("grad_accum", args.grad_accum)
        args.num_generations = cfg.get("num_generations", args.num_generations)
        args.max_prompt_length = cfg.get("max_prompt_length", args.max_prompt_length)
        args.max_completion_length = cfg.get("max_completion_length", args.max_completion_length)

def resolve_dtype(args, device: str) -> Tuple[torch.dtype, bool, bool]:
    if device != "cuda":
        return torch.float32, False, False
    if args.dtype == "bf16":
        return torch.bfloat16, True, False
    if args.dtype == "fp16":
        return torch.float16, False, True
    if args.dtype == "fp32":
        return torch.float32, False, False
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
    ap.add_argument("--output_dir", type=str, default="runs/grpo_dx")

    # GRPO settings (single GPU friendly)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=8)         # must be divisible by num_generations :contentReference[oaicite:12]{index=12}
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_generations", type=int, default=4)    # groups of 4
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)

    # LoRA (GRPOTrainer supports peft_config) :contentReference[oaicite:13]{index=13}
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj")

    ap.add_argument("--task_types", type=str, default=DEFAULT_TASKS, help="Comma-separated task types to include")
    ap.add_argument("--dataset_version", type=str, default="v2", choices=["auto", "v1", "v2"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--profile", type=str, default="", choices=["", "h100"], help="Apply H100-friendly defaults")
    ap.add_argument("--init_adapter", type=str, default="", help="Path to LoRA adapter to initialize GRPO from (e.g., SFT adapter)")
    ap.add_argument("--benchmark_every_steps", type=int, default=100)
    ap.add_argument("--benchmark_num_questions", type=int, default=20)
    ap.add_argument("--benchmark_list", type=str, default="medqa,medmcqa,pubmedqa")
    ap.add_argument("--benchmark_system_prompt", type=str, default="You are a medical expert.")
    ap.add_argument("--benchmark_allow_train_fallback", action="store_true")
    args = ap.parse_args()

    apply_profile(args)

    if args.batch_size % args.num_generations != 0:
        raise ValueError("--batch_size must be divisible by --num_generations")

    task_set = parse_task_types(args.task_types)
    print(f"[train_grpo] tasks: {sorted(task_set)}")
    if any("json" in t for t in task_set) and args.max_completion_length < 16:
        print("[train_grpo] bumping --max_completion_length to 32 for JSON tasks")
        args.max_completion_length = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    dtype, use_bf16, use_fp16 = resolve_dtype(args, device)

    run_dir = Path(args.output_dir) / now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"  # required by GRPO processing class :contentReference[oaicite:14]{index=14}
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = resolve_data_files(args.data_dir, args.dataset_version)
    ds = load_dataset("json", data_files=data_files, split="train")
    ds = ds.filter(lambda x: x["task_type"] in task_set)
    if len(ds) == 0:
        raise ValueError("No training examples after filtering. Check --dataset_version and --task_types.")

    def to_prompt_row(ex):
        msgs = ex["messages"]
        prompt_msgs = msgs[:-1]  # system+user
        prompt_text = try_apply_chat_template(tokenizer, prompt_msgs, add_generation_prompt=True)
        gt = ex["messages"][-1]["content"].strip()
        return {
            "prompt": prompt_text,
            "ground_truth": gt,
            "task_type": ex.get("task_type", ""),
            "dx_label": ex.get("dx_label", ""),
            "triage": ex.get("triage", ""),
        }

    keep_cols = {"task_type", "dx_label", "triage"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.map(to_prompt_row, remove_columns=remove_cols)

    peft_config = None
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]

    grpo_args = GRPOConfig(
        output_dir=str(run_dir),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,             # :contentReference[oaicite:15]{index=15}
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=["tensorboard"],
        remove_unused_columns=False,  # keep ground_truth for reward func :contentReference[oaicite:16]{index=16}
        seed=args.seed,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )

    if args.init_adapter:
        model = PeftModel.from_pretrained(base_model, args.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = base_model

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=ds,
        reward_funcs=reward_dx,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if args.benchmark_every_steps > 0:
        try:
            import sys
            repo_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repo_root / "medical-llm-finetune"))
            import benchmark as med_bench
        except Exception as exc:
            print(f"[train_grpo] benchmark import failed: {exc}")
            med_bench = None

        if med_bench is not None:
            benchmarks = [b.strip() for b in args.benchmark_list.split(",") if b.strip()]
            num_q = args.benchmark_num_questions
            strict_holdout = not args.benchmark_allow_train_fallback
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # preload questions once
            questions = {}
            for name in benchmarks:
                if name == "medqa":
                    questions[name] = med_bench.load_medqa(num_q)
                elif name == "medmcqa":
                    questions[name] = med_bench.load_medmcqa(num_q)
                elif name == "pubmedqa":
                    questions[name] = med_bench.load_pubmedqa(num_q, strict_holdout=strict_holdout)

            class BenchmarkCallback(TrainerCallback):
                def __init__(self):
                    self.last_step = 0

                def on_step_end(self, args_cb, state, control, **kwargs):
                    if state.global_step == 0:
                        return control
                    if state.global_step % args.benchmark_every_steps != 0:
                        return control
                    if state.global_step == self.last_step:
                        return control

                    self.last_step = state.global_step
                    model.eval()
                    results = {}
                    for name in benchmarks:
                        qs = questions.get(name, [])
                        if name in ("medqa", "medmcqa"):
                            results[name] = med_bench._evaluate_mcq(model, tokenizer, device, args.benchmark_system_prompt, qs)
                        elif name == "pubmedqa":
                            results[name] = med_bench._evaluate_yes_no(model, tokenizer, device, args.benchmark_system_prompt, qs)
                    out_path = run_dir / "benchmark_progress.jsonl"
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": state.global_step, "results": results}) + "\n")
                    print(f"[train_grpo] benchmark step {state.global_step}: {json.dumps(results)}")
                    model.train()
                    return control

            trainer.add_callback(BenchmarkCallback())

    trainer.train()

    # Save final artifacts
    trainer.model.save_pretrained(run_dir / "adapter")
    tokenizer.save_pretrained(run_dir / "adapter")
    print(f"[train_grpo] done. Run dir: {run_dir}")

if __name__ == "__main__":
    main()
