"""
Medical LLM Training Script

Optimized for:
- Local testing (tiny models, CPU/single GPU)
- Cloud training (8xH100 for full runs)

Based on modded-nanogpt training optimizations.
"""

import os
import sys
import time
import math
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Local imports
from model import (
    MedicalLLM, ModelConfig,
    get_tiny_config, get_small_config, get_medium_config, get_baguette_config
)
from data_loader import MedicalPretrainDataLoader


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_size: str = "tiny"  # tiny, small, medium, baguette

    # Data
    max_tokens: int = 1_000_000  # Total training tokens
    batch_size: int = 8
    max_seq_length: int = 512

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 100
    lr_decay_steps: Optional[int] = None  # Default to total steps

    # Training
    max_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 10

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    compile: bool = False  # torch.compile (requires PyTorch 2.0+)
    mixed_precision: bool = True

    # Paths
    output_dir: str = "./output"
    data_cache: str = "./data_cache"


class TokenDataset(Dataset):
    """Dataset wrapper for tokenized chunks."""

    def __init__(self, chunks: List[List[int]]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.chunks[idx], dtype=torch.long)
        # Input is all but last token, target is all but first
        return tokens[:-1], tokens[1:]


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(config: TrainConfig):
    """Main training loop."""

    print("=" * 60)
    print("Medical LLM Pretraining")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.device

    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.data_cache, exist_ok=True)

    # Get model config
    model_configs = {
        "tiny": get_tiny_config,
        "small": get_small_config,
        "medium": get_medium_config,
        "baguette": get_baguette_config,
    }
    model_config = model_configs[config.model_size]()
    model_config.max_seq_length = config.max_seq_length

    print(f"\nModel: {config.model_size}")
    print(f"  Layers: {model_config.n_layer}")
    print(f"  Heads: {model_config.n_head}")
    print(f"  Dim: {model_config.n_embd}")
    print(f"  Seq length: {model_config.max_seq_length}")

    # Create model
    model = MedicalLLM(model_config)
    model = model.to(device)

    # Compile if requested (PyTorch 2.0+)
    if config.compile and hasattr(torch, 'compile'):
        print("\nCompiling model with torch.compile...")
        model = torch.compile(model)

    # Setup mixed precision
    use_amp = config.mixed_precision and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load data
    print("\n" + "-" * 60)
    print("Loading Training Data")
    print("-" * 60)

    data_loader = MedicalPretrainDataLoader(
        datasets=["wikitext"],  # Use wikitext for initial testing (reliable)
        max_seq_length=config.max_seq_length,
    )

    # Create or load cached data
    cache_file = os.path.join(config.data_cache, f"train_chunks_{config.max_tokens}.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        chunks = torch.load(cache_file)
    else:
        print("Creating training data...")
        chunks = data_loader.create_mixed_dataset(max_tokens=config.max_tokens)
        torch.save(chunks, cache_file)
        print(f"Saved cache to {cache_file}")

    # Create dataset and dataloader
    train_dataset = TokenDataset(chunks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for testing
        pin_memory=device == "cuda",
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Setup optimizer
    # Use AdamW with weight decay only on non-embedding, non-norm parameters
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'ln_' in name or 'norm' in name or 'wte' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate, betas=(config.beta1, config.beta2))

    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Decayed params: {sum(p.numel() for p in decay_params):,}")
    print(f"  Non-decayed params: {sum(p.numel() for p in no_decay_params):,}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    model.train()
    step = 0
    total_loss = 0.0
    best_loss = float('inf')
    start_time = time.time()
    tokens_processed = 0

    max_steps = min(config.max_steps, len(train_loader) * 10)  # Cap at 10 epochs
    lr_decay_steps = config.lr_decay_steps or max_steps

    data_iter = iter(train_loader)

    while step < max_steps:
        # Get batch
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Update learning rate
        lr = get_lr(step, config.warmup_steps, lr_decay_steps, config.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                logits, loss = model(inputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Accumulate stats
        total_loss += loss.item()
        tokens_processed += inputs.numel()
        step += 1

        # Logging
        if step % config.log_interval == 0:
            avg_loss = total_loss / config.log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed

            print(f"Step {step:5d}/{max_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.0f} | "
                  f"Time: {elapsed:.1f}s")

            total_loss = 0.0

        # Evaluation
        if step % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Quick eval on a batch
                eval_inputs, eval_targets = next(iter(train_loader))
                eval_inputs = eval_inputs.to(device)
                eval_targets = eval_targets.to(device)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        _, eval_loss = model(eval_inputs, eval_targets)
                else:
                    _, eval_loss = model(eval_inputs, eval_targets)

                print(f"  Eval loss: {eval_loss.item():.4f}")

                if eval_loss.item() < best_loss:
                    best_loss = eval_loss.item()
                    # Save best model
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'config': model_config,
                    }, os.path.join(config.output_dir, 'best_model.pt'))
                    print(f"  New best! Saved checkpoint.")

            model.train()

        # Periodic save
        if step % config.save_interval == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
            }, os.path.join(config.output_dir, f'checkpoint_{step}.pt'))
            print(f"  Saved checkpoint_{step}.pt")

    # Final stats
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total steps: {step}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total tokens: {tokens_processed:,}")
    print(f"Avg tokens/sec: {tokens_processed / total_time:.0f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Output: {config.output_dir}")

    return model


def test_generation(model, data_loader, device="cpu"):
    """Test text generation with the trained model."""
    model.eval()

    # Create a simple prompt
    prompt = "The patient presents with"
    prompt_tokens = data_loader.tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print("Generated:")

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
        generated = data_loader.tokenizer.decode(output[0].tolist())
        print(generated)


def main():
    parser = argparse.ArgumentParser(description="Train Medical LLM")

    parser.add_argument("--model", type=str, default="tiny",
                        choices=["tiny", "small", "medium", "baguette"],
                        help="Model size")
    parser.add_argument("--max_tokens", type=int, default=100_000,
                        help="Total training tokens")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test data loading, don't train")

    args = parser.parse_args()

    if args.test_only:
        # Just test data loading with wikitext (reliable test dataset)
        print("Testing data loading with WikiText...")
        loader = MedicalPretrainDataLoader(
            datasets=["wikitext"],  # Use wikitext for testing
            max_seq_length=256,
        )
        chunks = loader.create_mixed_dataset(max_tokens=10_000)
        if len(chunks) == 0:
            print("\nNo chunks created - check dataset loading")
            return
        print(f"\nSuccess! Created {len(chunks)} chunks")
        print(f"Sample tokens: {chunks[0][:20]}")
        print(f"Decoded: {loader.tokenizer.decode(chunks[0][:50])}")
        return

    config = TrainConfig(
        model_size=args.model,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        device=args.device,
        compile=args.compile,
        output_dir=args.output,
    )

    model = train(config)

    # Test generation
    print("\n" + "-" * 60)
    print("Testing Generation")
    print("-" * 60)

    loader = MedicalPretrainDataLoader(max_seq_length=config.max_seq_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_generation(model, loader, device)


if __name__ == "__main__":
    main()
