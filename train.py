"""
Medical LLM Training Script

Optimized for:
- Local testing (tiny models, CPU/single GPU)
- Cloud training (8xH100 with DDP for full runs)

Based on modded-nanogpt training optimizations.

Usage:
  # Local testing (downloads data on-the-fly)
  python train.py --model tiny --max_tokens 100000 --max_steps 100

  # Resume from checkpoint
  python train.py --model tiny --resume output/checkpoint_500.pt

  # Large-scale training with pre-downloaded data (RECOMMENDED for H100s)
  # Step 1: Prepare data
  python prepare_data.py --max_tokens 5000000000 --datasets synth,guidelines,openmedtext

  # Step 2: Train with sharded data
  torchrun --nproc_per_node=8 train.py --model medium --data_dir ./data_shards --max_steps 50000
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local imports
from model import (
    MedicalLLM, ModelConfig,
    get_tiny_config, get_small_config, get_medium_config, get_baguette_config
)
from data_loader import MedicalPretrainDataLoader
from streaming_loader import StreamingMedicalDataset


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_size: str = "tiny"  # tiny, small, medium, baguette

    # Data
    datasets: List[str] = None  # Dataset names to use
    max_tokens: int = 1_000_000  # Total training tokens
    batch_size: int = 8
    max_seq_length: int = 512
    eval_split: float = 0.05  # 5% for evaluation

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

    # Distributed
    ddp: bool = False  # Set automatically if using torchrun

    # Paths
    output_dir: str = "./output"
    data_cache: str = "./data_cache"
    data_dir: Optional[str] = None  # Path to pre-prepared sharded data
    streaming: bool = False  # Stream data directly (no prep needed, fastest start)
    resume: Optional[str] = None  # Path to checkpoint to resume from


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


class ShardedDataset(Dataset):
    """Dataset that loads from pre-prepared sharded binary files."""

    def __init__(self, data_dir: str, seq_length: int = 512):
        """
        Args:
            data_dir: Directory containing shard files and metadata.json
            seq_length: Sequence length (must match preparation)
        """
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length

        # Load metadata
        metadata_file = self.data_dir / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"No metadata.json found in {data_dir}. Run prepare_data.py first.")

        with open(metadata_file) as f:
            self.metadata = json.load(f)

        # Verify seq_length matches
        if self.metadata["seq_length"] != seq_length:
            print(f"Warning: seq_length mismatch. Data: {self.metadata['seq_length']}, Config: {seq_length}")
            self.seq_length = self.metadata["seq_length"]

        # Load all shards into memory-mapped arrays
        self.shards = []
        self.shard_sizes = []
        total_chunks = 0

        for shard_file in sorted(self.data_dir.glob("shard_*.bin")):
            # Memory-map the file for efficient loading
            mmap = np.memmap(shard_file, dtype=np.uint32, mode='r')
            num_chunks = len(mmap) // self.seq_length
            mmap = mmap[:num_chunks * self.seq_length].reshape(num_chunks, self.seq_length)

            self.shards.append(mmap)
            self.shard_sizes.append(num_chunks)
            total_chunks += num_chunks

        self.total_chunks = total_chunks
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)

        print(f"Loaded {len(self.shards)} shards with {self.total_chunks:,} chunks")

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        # Find which shard this index belongs to
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[shard_idx]

        tokens = torch.from_numpy(self.shards[shard_idx][local_idx].astype(np.int64))
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


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        # Launched with torchrun
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def print_main(msg: str, rank: int):
    """Print only on main process."""
    if is_main_process(rank):
        print(msg)


def train(config: TrainConfig):
    """Main training loop with DDP support."""

    # Setup distributed
    rank, local_rank, world_size, ddp = setup_distributed()
    config.ddp = ddp

    print_main("=" * 60, rank)
    print_main("Medical LLM Pretraining", rank)
    print_main("=" * 60, rank)
    print_main(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", rank)

    if ddp:
        print_main(f"\nDistributed Training: {world_size} GPUs", rank)
        device = f"cuda:{local_rank}"
    else:
        # Setup device for single GPU/CPU
        if config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = config.device

    print_main(f"\nDevice: {device}", rank)
    if "cuda" in device:
        gpu_idx = local_rank if ddp else 0
        print_main(f"  GPU: {torch.cuda.get_device_name(gpu_idx)}", rank)
        print_main(f"  Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.1f} GB", rank)

    # Create output directory
    if is_main_process(rank):
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.data_cache, exist_ok=True)

    # Sync before proceeding
    if ddp:
        dist.barrier()

    # Get model config
    model_configs = {
        "tiny": get_tiny_config,
        "small": get_small_config,
        "medium": get_medium_config,
        "baguette": get_baguette_config,
    }
    model_config = model_configs[config.model_size]()
    model_config.max_seq_length = config.max_seq_length

    print_main(f"\nModel: {config.model_size}", rank)
    print_main(f"  Layers: {model_config.n_layer}", rank)
    print_main(f"  Heads: {model_config.n_head}", rank)
    print_main(f"  Dim: {model_config.n_embd}", rank)
    print_main(f"  Seq length: {model_config.max_seq_length}", rank)

    # Create model
    model = MedicalLLM(model_config)
    model = model.to(device)

    # Resume from checkpoint if specified
    start_step = 0
    best_loss = float('inf')

    if config.resume:
        print_main(f"\nResuming from checkpoint: {config.resume}", rank)
        checkpoint = torch.load(config.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('step', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        print_main(f"  Resumed at step {start_step}, best loss: {best_loss:.4f}", rank)

    # Wrap with DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module  # For saving
    else:
        raw_model = model

    # Compile if requested (PyTorch 2.0+)
    if config.compile and hasattr(torch, 'compile'):
        print_main("\nCompiling model with torch.compile...", rank)
        model = torch.compile(model)

    # Setup mixed precision
    use_amp = config.mixed_precision and "cuda" in device
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load data
    print_main("\n" + "-" * 60, rank)
    print_main("Loading Training Data", rank)
    print_main("-" * 60, rank)

    # Track datasets used (for checkpoint metadata)
    datasets_to_use = config.datasets or ["wikitext"]

    if config.streaming:
        # FASTEST: Stream directly from HuggingFace (no prep needed)
        print_main(f"Streaming mode: datasets={datasets_to_use}", rank)
        print_main("  No prep step - training starts immediately!", rank)

        train_dataset = StreamingMedicalDataset(
            datasets=datasets_to_use,
            seq_length=config.max_seq_length,
            seed=42,
        )
        eval_dataset = None  # No separate eval in streaming mode

        # Streaming doesn't use samplers
        train_sampler = None
        eval_sampler = None
        shuffle = False  # Streaming handles its own randomization

        # Create train loader (streaming is infinite)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=4 if ddp else 2,
            pin_memory="cuda" in device,
            prefetch_factor=2,
        )
        eval_loader = None

        print_main(f"  Streaming dataloader ready", rank)

    elif config.data_dir:
        # Use pre-prepared sharded data (recommended for large-scale training)
        print_main(f"Loading sharded data from: {config.data_dir}", rank)

        full_dataset = ShardedDataset(config.data_dir, seq_length=config.max_seq_length)

        # Split into train/eval
        n_eval = max(1, int(len(full_dataset) * config.eval_split))
        n_train = len(full_dataset) - n_eval

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_eval],
            generator=torch.Generator().manual_seed(42)
        )

        print_main(f"\nTrain samples: {n_train:,}", rank)
        print_main(f"Eval samples: {n_eval:,}", rank)

    else:
        # On-the-fly data loading (for testing/small runs)
        print_main(f"Datasets: {datasets_to_use}", rank)

        data_loader = MedicalPretrainDataLoader(
            datasets=datasets_to_use,
            max_seq_length=config.max_seq_length,
        )

        # Create or load cached data
        dataset_hash = "_".join(sorted(datasets_to_use))
        cache_file = os.path.join(config.data_cache, f"train_chunks_{dataset_hash}_{config.max_tokens}.pt")

        if os.path.exists(cache_file):
            print_main(f"Loading cached data from {cache_file}", rank)
            chunks = torch.load(cache_file, weights_only=False)
        else:
            print_main("Creating training data...", rank)
            chunks = data_loader.create_mixed_dataset(max_tokens=config.max_tokens)
            if is_main_process(rank):
                torch.save(chunks, cache_file)
                print_main(f"Saved cache to {cache_file}", rank)

        if len(chunks) == 0:
            print_main("ERROR: No training data loaded!", rank)
            cleanup_distributed()
            return None

        # Split into train/eval
        n_eval = max(1, int(len(chunks) * config.eval_split))
        eval_chunks = chunks[:n_eval]
        train_chunks = chunks[n_eval:]

        print_main(f"\nTrain samples: {len(train_chunks)}", rank)
        print_main(f"Eval samples: {len(eval_chunks)}", rank)

        # Create datasets
        train_dataset = TokenDataset(train_chunks)
        eval_dataset = TokenDataset(eval_chunks)

    # Create dataloaders (skip if streaming mode already created them)
    if not config.streaming:
        if ddp:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42)
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
            shuffle = False  # Sampler handles shuffling
        else:
            train_sampler = None
            eval_sampler = None
            shuffle = True

        # Adjust batch size for DDP (total batch = batch_size * world_size)
        per_device_batch = config.batch_size
        if ddp:
            print_main(f"Per-device batch size: {per_device_batch}", rank)
            print_main(f"Effective batch size: {per_device_batch * world_size}", rank)

        train_loader = DataLoader(
            train_dataset,
            batch_size=per_device_batch,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=4 if ddp else 0,
            pin_memory="cuda" in device,
            drop_last=True,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=per_device_batch,
            shuffle=False,
            sampler=eval_sampler,
            num_workers=0,
            pin_memory="cuda" in device,
        )

        print_main(f"Batches per epoch: {len(train_loader)}", rank)
    else:
        print_main(f"Streaming mode: infinite batches", rank)

    # Setup optimizer
    decay_params = []
    no_decay_params = []

    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if 'ln_' in name or 'norm' in name or 'wte' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate, betas=(config.beta1, config.beta2))

    # Load optimizer state if resuming
    if config.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print_main("  Loaded optimizer state", rank)

    print_main(f"\nOptimizer: AdamW", rank)
    print_main(f"  Learning rate: {config.learning_rate}", rank)
    print_main(f"  Weight decay: {config.weight_decay}", rank)
    print_main(f"  Decayed params: {sum(p.numel() for p in decay_params):,}", rank)
    print_main(f"  Non-decayed params: {sum(p.numel() for p in no_decay_params):,}", rank)

    # Training loop
    print_main("\n" + "=" * 60, rank)
    print_main("Starting Training", rank)
    print_main("=" * 60, rank)

    model.train()
    step = start_step
    total_loss = 0.0
    start_time = time.time()
    tokens_processed = 0

    # Cap at 10 epochs for non-streaming, or just use max_steps for streaming
    if config.streaming:
        max_steps = config.max_steps
    else:
        max_steps = min(config.max_steps, len(train_loader) * 10)
    lr_decay_steps = config.lr_decay_steps or max_steps

    epoch = 0
    while step < max_steps:
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for inputs, targets in train_loader:
            if step >= max_steps:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Update learning rate
            lr = get_lr(step, config.warmup_steps, lr_decay_steps, config.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
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
            tokens_processed += inputs.numel() * world_size  # Account for all GPUs
            step += 1

            # Logging
            if step % config.log_interval == 0:
                avg_loss = total_loss / config.log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed

                print_main(f"Step {step:5d}/{max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Tokens/s: {tokens_per_sec:.0f} | "
                      f"Time: {elapsed:.1f}s", rank)

                total_loss = 0.0

            # Evaluation
            if step % config.eval_interval == 0:
                if eval_loader is not None:
                    model.eval()
                    eval_losses = []

                    with torch.no_grad():
                        for eval_inputs, eval_targets in eval_loader:
                            eval_inputs = eval_inputs.to(device)
                            eval_targets = eval_targets.to(device)

                            if use_amp:
                                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                                    _, eval_loss = model(eval_inputs, eval_targets)
                            else:
                                _, eval_loss = model(eval_inputs, eval_targets)

                            eval_losses.append(eval_loss.item())

                    avg_eval_loss = sum(eval_losses) / len(eval_losses)

                    # Sync eval loss across GPUs
                    if ddp:
                        eval_tensor = torch.tensor([avg_eval_loss], device=device)
                        dist.all_reduce(eval_tensor, op=dist.ReduceOp.AVG)
                        avg_eval_loss = eval_tensor.item()

                    print_main(f"  Eval loss: {avg_eval_loss:.4f} (on {len(eval_losses)} batches)", rank)

                    if avg_eval_loss < best_loss:
                        best_loss = avg_eval_loss
                else:
                    # Streaming mode: use train loss for checkpointing
                    avg_eval_loss = loss.item()
                    print_main(f"  Train loss: {avg_eval_loss:.4f} (streaming mode)", rank)
                    if avg_eval_loss < best_loss:
                        best_loss = avg_eval_loss

                # Save best model if improved (only on main process)
                if avg_eval_loss <= best_loss and is_main_process(rank):
                    torch.save({
                        'step': step,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'config': model_config,
                        'train_config': {
                            'model_size': config.model_size,
                            'datasets': datasets_to_use,
                            'max_tokens': config.max_tokens,
                        },
                    }, os.path.join(config.output_dir, 'best_model.pt'))
                    print_main(f"  New best! Saved checkpoint.", rank)

                model.train()

            # Periodic save
            if step % config.save_interval == 0 and is_main_process(rank):
                torch.save({
                    'step': step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': model_config,
                    'train_config': {
                        'model_size': config.model_size,
                        'datasets': datasets_to_use,
                        'max_tokens': config.max_tokens,
                    },
                }, os.path.join(config.output_dir, f'checkpoint_{step}.pt'))
                print_main(f"  Saved checkpoint_{step}.pt", rank)

        epoch += 1

    # Final stats
    total_time = time.time() - start_time
    print_main("\n" + "=" * 60, rank)
    print_main("Training Complete!", rank)
    print_main("=" * 60, rank)
    print_main(f"Total steps: {step}", rank)
    print_main(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)", rank)
    print_main(f"Total tokens: {tokens_processed:,}", rank)
    print_main(f"Avg tokens/sec: {tokens_processed / total_time:.0f}", rank)
    print_main(f"Best loss: {best_loss:.4f}", rank)
    print_main(f"Output: {config.output_dir}", rank)

    # Cleanup DDP
    cleanup_distributed()

    return raw_model


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
        # Handle unicode safely for Windows console
        try:
            print(generated)
        except UnicodeEncodeError:
            print(generated.encode('ascii', errors='replace').decode('ascii'))


def main():
    parser = argparse.ArgumentParser(description="Train Medical LLM")

    parser.add_argument("--model", type=str, default="tiny",
                        choices=["tiny", "small", "medium", "baguette"],
                        help="Model size")
    parser.add_argument("--datasets", type=str, default="wikitext",
                        help="Comma-separated dataset names (wikitext,synth,guidelines,openmedtext,pubmed,fineweb_edu)")
    parser.add_argument("--max_tokens", type=int, default=100_000,
                        help="Total training tokens")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluation interval (steps)")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Checkpoint save interval (steps)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to pre-prepared sharded data (from prepare_data.py)")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream data directly from HuggingFace (no prep needed, fastest start)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test data loading, don't train")

    args = parser.parse_args()

    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.test_only:
        # Just test data loading
        print(f"Testing data loading with: {datasets}")
        loader = MedicalPretrainDataLoader(
            datasets=datasets,
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
        datasets=datasets if datasets else None,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
        compile=args.compile,
        output_dir=args.output,
        data_dir=args.data_dir,
        streaming=args.streaming,
        resume=args.resume,
    )

    model = train(config)

    if model is not None:
        # Test generation (only on single GPU)
        if not config.ddp or is_main_process(0):
            print("\n" + "-" * 60)
            print("Testing Generation")
            print("-" * 60)

            loader = MedicalPretrainDataLoader(max_seq_length=config.max_seq_length)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            test_generation(model, loader, device)


if __name__ == "__main__":
    main()
