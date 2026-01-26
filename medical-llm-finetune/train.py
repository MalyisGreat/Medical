#!/usr/bin/env python3
"""
Medical LLM Fine-tuning (Baguettotron-first)

Default: full fine-tune on medical QA with a held-out benchmark
before and after training.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

from data_loader import MedicalDatasetLoader
from benchmark import evaluate_model

MODEL_REGISTRY = [
    {
        "id": "Qwen/Qwen3-0.6B",
        "params_b": 0.6,
        "family": "Qwen3",
        "instruct": True,
    },
    {
        "id": "Qwen/Qwen3-0.6B-Base",
        "params_b": 0.6,
        "family": "Qwen3",
        "instruct": False,
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.49,
        "family": "Qwen2.5",
        "instruct": True,
    },
]

# CLI defaults (used for preset override detection)
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LEARNING_RATE = 2e-4


def select_best_model(max_params_b: float, prefer_family: str) -> str:
    candidates = [c for c in MODEL_REGISTRY if c["params_b"] <= max_params_b]
    if not candidates:
        raise ValueError(f"No models found under {max_params_b}B params.")

    if prefer_family:
        fam = [c for c in candidates if c["family"].lower() == prefer_family.lower()]
        if fam:
            candidates = fam

    instruct = [c for c in candidates if c["instruct"]]
    if instruct:
        candidates = instruct

    # Pick the largest param count in the filtered set
    candidates.sort(key=lambda c: c["params_b"], reverse=True)

    # Verify availability, fall back if not reachable
    for c in candidates:
        try:
            AutoConfig.from_pretrained(c["id"], trust_remote_code=True)
            return c["id"]
        except Exception:
            continue

    raise RuntimeError("No candidate model could be resolved from Hugging Face.")


def supports_flash_attention() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def setup_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def setup_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list = None,
) -> LoraConfig:
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(
    model_name: str,
    use_lora: bool,
    use_qlora: bool,
    lora_r: int,
    lora_alpha: int,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_config = setup_quantization_config() if use_qlora else None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    attn_impl = "flash_attention_2" if supports_flash_attention() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    # Report attention implementation for debugging
    attn_impl_actual = getattr(model.config, "attn_implementation", None)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"[rank {local_rank}] attn_impl={attn_impl_actual}")

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora_cfg = setup_lora_config(r=lora_r, alpha=lora_alpha)
        model = get_peft_model(model, lora_cfg)

    return model, tokenizer


def create_training_arguments(
    output_dir: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    max_seq_length: int,
    tf32: bool,
    max_steps: int = -1,
) -> SFTConfig:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=False,
        bf16=use_bf16,
        tf32=tf32,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        group_by_length=True,
        report_to="none",
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
    )


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


class PeriodicBenchmarkCallback(TrainerCallback):
    def __init__(
        self,
        eval_every_percent: float,
        output_dir: str,
        benchmarks: List[str],
        num_questions: int,
        system_prompt: str,
        allow_train_fallback: bool,
    ):
        self.eval_every_percent = eval_every_percent
        self.output_dir = output_dir
        self.benchmarks = benchmarks
        self.num_questions = num_questions
        self.system_prompt = system_prompt
        self.strict_holdout = not allow_train_fallback
        self.next_percent = eval_every_percent

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_every_percent <= 0:
            return control
        if state.max_steps is None or state.max_steps <= 0:
            return control

        progress = 100.0 * state.global_step / state.max_steps
        if progress + 1e-6 < self.next_percent:
            return control

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")

        if model is None or tokenizer is None:
            return control

        # Ensure all ranks pause while rank 0 evaluates
        if is_main_process():
            was_training = model.training
            model.eval()
            device = next(model.parameters()).device
            results = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                device=str(device),
                benchmarks=self.benchmarks,
                num_questions=self.num_questions,
                system_prompt=self.system_prompt,
                strict_holdout=self.strict_holdout,
            )
            os.makedirs(self.output_dir, exist_ok=True)
            out_path = os.path.join(self.output_dir, "benchmark_progress.jsonl")
            payload = {
                "global_step": state.global_step,
                "progress_percent": round(progress, 2),
                "results": results,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.next_percent += self.eval_every_percent
        return control


def train(
    model_name: str,
    output_dir: str,
    max_samples: int,
    epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    max_seq_length: int,
    datasets: List[str],
    eval_split: float,
    use_lora: bool,
    use_qlora: bool,
    eval_before: bool,
    eval_after: bool,
    eval_questions: int,
    eval_benchmarks: List[str],
    system_prompt: str,
    max_params_b: float,
    prefer_family: str,
    tf32: bool,
    eval_interval_percent: float,
    allow_train_fallback: bool,
):
    if model_name == "auto":
        model_name = select_best_model(max_params_b=max_params_b, prefer_family=prefer_family)

    print("=" * 60)
    print("Medical LLM Fine-tuning (Baguettotron)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Base Model: {model_name}")
    print(f"  Datasets: {datasets}")
    print(f"  Max Samples: {max_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gradient Accumulation: {gradient_accumulation}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Sequence Length: {max_seq_length}")
    print(f"  LoRA: {use_lora} | QLoRA: {use_qlora}")
    print(f"  TF32: {tf32}")

    if torch.cuda.is_available() and is_main_process():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({gpu_memory:.1f} GB)")

    if torch.cuda.is_available() and tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loader = MedicalDatasetLoader(system_prompt=system_prompt)
    samples_per_dataset = None
    if max_samples and max_samples > 0:
        samples_per_dataset = max_samples // len(datasets)
    train_dataset = loader.load_combined(
        datasets=datasets,
        split="train",
        max_samples_per_dataset=samples_per_dataset,
        shuffle=True,
    )

    split_data = train_dataset.train_test_split(test_size=eval_split, seed=42)
    train_dataset = split_data["train"]
    eval_dataset = split_data["test"]

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        use_lora=use_lora,
        use_qlora=use_qlora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    if use_lora:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    train_dataset = loader.format_for_training(train_dataset, tokenizer, max_length=max_seq_length)
    eval_dataset = loader.format_for_training(eval_dataset, tokenizer, max_length=max_seq_length)

    training_args = create_training_arguments(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        tf32=tf32,
    )

    eval_device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_main_process() and eval_before:
        print("\n" + "-" * 60)
        print("Benchmark Before Fine-tuning")
        print("-" * 60)
        model.to(eval_device)
        results_before = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            device=eval_device,
            benchmarks=eval_benchmarks,
            num_questions=eval_questions,
            system_prompt=system_prompt,
            strict_holdout=not allow_train_fallback,
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "benchmark_before.json"), "w") as f:
            json.dump(results_before, f, indent=2)
        for name, res in results_before.items():
            acc = res["accuracy"] * 100
            print(f"{name:10s}: {acc:5.1f}% ({res['correct']}/{res['total']})")

    print("\n" + "-" * 60)
    print("Starting Training")
    print("-" * 60)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    if eval_interval_percent and eval_interval_percent > 0:
        trainer.add_callback(
            PeriodicBenchmarkCallback(
                eval_every_percent=eval_interval_percent,
                output_dir=output_dir,
                benchmarks=eval_benchmarks,
                num_questions=eval_questions,
                system_prompt=system_prompt,
                allow_train_fallback=allow_train_fallback,
            )
        )

    trainer.train()

    print("\n" + "-" * 60)
    print("Saving Model")
    print("-" * 60)

    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    if is_main_process() and eval_after:
        print("\n" + "-" * 60)
        print("Benchmark After Fine-tuning")
        print("-" * 60)
        model.to(eval_device)
        results_after = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            device=eval_device,
            benchmarks=eval_benchmarks,
            num_questions=eval_questions,
            system_prompt=system_prompt,
            strict_holdout=not allow_train_fallback,
        )
        with open(os.path.join(output_dir, "benchmark_after.json"), "w") as f:
            json.dump(results_after, f, indent=2)
        for name, res in results_after.items():
            acc = res["accuracy"] * 100
            print(f"{name:10s}: {acc:5.1f}% ({res['correct']}/{res['total']})")

    print("\nTraining Complete!")
    return trainer


def apply_hardware_preset(args):
    if args.hardware != "h100":
        return args

    # Only override if user left defaults
    if args.batch_size == DEFAULT_BATCH_SIZE:
        args.batch_size = 32
    if args.grad_accum == DEFAULT_GRAD_ACCUM:
        args.grad_accum = 1
    if args.learning_rate == DEFAULT_LEARNING_RATE:
        args.learning_rate = 2e-4
    args.tf32 = True
    return args


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Baguettotron on medical data")

    parser.add_argument("--hardware", type=str, default="auto", choices=["auto", "h100"],
                        help="Hardware preset for training (default: auto)")
    parser.add_argument("--model", "-m", default="auto",
                        help="Base model to fine-tune, or 'auto' to pick the best <1B model")
    parser.add_argument("--max_params_b", type=float, default=1.0,
                        help="Max model size (billions of params) for auto selection")
    parser.add_argument("--prefer_family", type=str, default="Qwen3",
                        help="Preferred model family when auto-selecting (e.g., Qwen3)")
    parser.add_argument("--output", "-o", default="./medical_llm_output",
                        help="Output directory for checkpoints")
    parser.add_argument("--max_samples", "-n", type=int, default=20000,
                        help="Maximum training samples (0 = uncapped)")
    parser.add_argument("--epochs", "-e", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--datasets", nargs="+", default=["pubmedqa", "medmcqa"],
                        help="Datasets to use (pubmedqa, medqa, medmcqa)")
    parser.add_argument("--eval_split", type=float, default=0.1,
                        help="Fraction of training data for eval")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA adapters (default: full fine-tune)")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use QLoRA 4-bit quantization (implies LoRA)")
    parser.add_argument("--eval_before", action="store_true",
                        help="Run holdout benchmark before training")
    parser.add_argument("--eval_after", action="store_true",
                        help="Run holdout benchmark after training")
    parser.add_argument("--eval_questions", type=int, default=100,
                        help="Questions per benchmark")
    parser.add_argument("--eval_interval_percent", type=float, default=0,
                        help="Run holdout benchmark every N%% of training (0 = off)")
    parser.add_argument("--eval_benchmarks", type=str, default="medqa,medmcqa,pubmedqa",
                        help="Comma-separated benchmarks")
    parser.add_argument("--system_prompt", type=str, default="You are a medical expert.",
                        help="System prompt for chat formatting")
    parser.add_argument("--allow_train_fallback", action="store_true",
                        help="Allow PubMedQA benchmark to fall back to train split (not strict holdout)")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 matmul (recommended on H100)")

    args = parser.parse_args()
    args = apply_hardware_preset(args)

    use_lora = args.use_lora or args.use_qlora
    eval_benchmarks = [b.strip() for b in args.eval_benchmarks.split(",") if b.strip()]

    train(
        model_name=args.model,
        output_dir=args.output,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        datasets=args.datasets,
        eval_split=args.eval_split,
        use_lora=use_lora,
        use_qlora=args.use_qlora,
        eval_before=args.eval_before,
        eval_after=args.eval_after,
        eval_questions=args.eval_questions,
        eval_benchmarks=eval_benchmarks,
        system_prompt=args.system_prompt,
        max_params_b=args.max_params_b,
        prefer_family=args.prefer_family,
        tf32=args.tf32,
        eval_interval_percent=args.eval_interval_percent,
        allow_train_fallback=args.allow_train_fallback,
    )


if __name__ == "__main__":
    main()
