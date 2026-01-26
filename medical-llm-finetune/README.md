# Medical LLM Fine-tune (Baguettotron)

Fine-tune Baguettotron on medical QA datasets and run a holdout benchmark
before and after training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .\\.venv\\Scripts\\Activate.ps1 on Windows
pip install -r requirements.txt
```

## Train + Benchmark

```bash
python train.py \
  --model auto \
  --datasets pubmedqa medqa medmcqa \
  --max_samples 20000 \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum 4 \
  --eval_before \
  --eval_after
```

Outputs:
- `medical_llm_output/benchmark_before.json`
- `medical_llm_output/benchmark_after.json`
- `medical_llm_output/final_model/`

## LoRA / QLoRA

```bash
python train.py --use_lora
python train.py --use_qlora
```

## Auto-select a sub-1B model

```bash
python train.py --model auto --max_params_b 1.0 --prefer_family Qwen3
```

## Qwen on H100 (preset)

```bash
python train.py \
  --hardware h100 \
  --model Qwen/Qwen3-0.6B \
  --use_lora \
  --datasets pubmedqa medqa medmcqa \
  --eval_before --eval_after
```

## Uncapped data + periodic benchmarks

Run on full available data (no cap) and evaluate every 10% of training:

```bash
python train.py \
  --model Qwen/Qwen3-0.6B \
  --datasets pubmedqa medqa medmcqa \
  --max_samples 0 \
  --eval_before --eval_after \
  --eval_interval_percent 10
```

Periodic results are appended to:
- `medical_llm_output/benchmark_progress.jsonl`

## Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=8 train.py \
  --model Qwen/Qwen3-0.6B \
  --datasets pubmedqa medqa medmcqa \
  --max_samples 0 \
  --eval_before --eval_after \
  --eval_interval_percent 10
```

## Standalone Benchmark

```bash
python benchmark.py --model PleIAs/Baguettotron --num_questions 100
```

## Datasets

Training uses `train` splits. Holdout benchmark uses:
- MedQA: `test`
- MedMCQA: `validation`
- PubMedQA: `test` (fallback to `train` if `test` is unavailable)
