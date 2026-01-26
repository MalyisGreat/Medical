# Medical LLM Pretraining

Train a small but powerful medical reasoning model from scratch using modern techniques.

## Key Features

- **Reasoning-focused training**: Uses SYNTH dataset with structured `<|question|>` → `<|thinking|>` → `<|answer|>` format
- **Modern architecture**: RoPE, RMSNorm, SwiGLU, QK-Norm (based on modded-nanogpt)
- **No benchmark contamination**: Avoids MedQA/MedMCQA/PubMedQA for clean evaluation
- **Efficient**: Designed for $20 training budget on 8xH100

## Architecture

| Config | Params | Layers | Hidden | Use Case |
|--------|--------|--------|--------|----------|
| tiny | 87M | 6 | 384 | Testing |
| small | 162M | 12 | 768 | Local training |
| medium | 406M | 24 | 1024 | Cloud training |
| baguette | 308M | 80 | 512 | Deep reasoning |

## Data Sources

| Dataset | Content | Weight |
|---------|---------|--------|
| SYNTH | Reasoning chains (Wikipedia-derived) | 40% |
| PubMed | Medical abstracts | 30% |
| Clinical Guidelines | CDC, WHO, NICE guidelines | 15% |
| OpenMedText | Medical textbooks | 10% |
| FineWeb-Edu | General education | 5% |

## Reasoning Format

The model learns chain-of-thought reasoning through structured training:

```
<|question|>
What are the symptoms of diabetes?
<|thinking|>
Let me break this down systematically...
- Type 1 vs Type 2 considerations
- Common symptoms include polyuria, polydipsia...
- Risk factors to consider...
<|answer|>
The main symptoms of diabetes include increased thirst (polydipsia),
frequent urination (polyuria), unexplained weight loss...
```

## Quick Start

### Test locally
```bash
pip install -r requirements.txt

# Test data loading
python train.py --test_only --datasets synth

# Train tiny model (testing)
python train.py --model tiny --max_tokens 100000 --max_steps 100 --datasets wikitext
```

### Full training (8xH100)

**Option 1: STREAMING (fastest start, recommended)**
```bash
# No prep step - starts training immediately!
torchrun --nproc_per_node=8 train.py \
    --model medium \
    --streaming \
    --datasets synth,guidelines,openmedtext \
    --batch_size 64 \
    --max_steps 40000 \
    --compile

# Resume from checkpoint
torchrun --nproc_per_node=8 train.py \
    --model medium \
    --streaming \
    --datasets synth,guidelines,openmedtext \
    --resume output/checkpoint_10000.pt
```

**Option 2: Pre-prepared shards (if network is slow)**
```bash
# Step 1: Prepare data once
python prepare_data.py \
    --max_tokens 5000000000 \
    --datasets synth,guidelines,openmedtext \
    --output ./data_shards

# Step 2: Train from shards
torchrun --nproc_per_node=8 train.py \
    --model medium \
    --data_dir ./data_shards \
    --batch_size 64 \
    --max_steps 40000 \
    --compile
```

### Local testing
```bash
# Quick streaming test
python train.py --model tiny --streaming --datasets synth --max_steps 100

# Or with cached data
python train.py --model tiny --max_tokens 1000000 --datasets wikitext
```

### Available datasets
- `wikitext` - WikiText-2 (for testing)
- `synth` - SYNTH reasoning chains (40% weight)
- `pubmed` - PubMed abstracts (30% weight)
- `guidelines` - Clinical guidelines (15% weight)
- `openmedtext` - Medical textbooks (10% weight)
- `fineweb_edu` - General education (5% weight)

## Cost Estimates

With `--compile` and streaming on 8xH100 (~$12/hr on Lambda/Vast):

| Model | Tokens | Time | Cost |
|-------|--------|------|------|
| medium (406M) | 2B | ~45 min | ~$9 |
| medium (406M) | 5B | ~2 hours | ~$24 |
| baguette (308M) | 5B | ~2.5 hours | ~$30 |

**Note:** SYNTH data is high-quality reasoning chains - you may get good results with fewer tokens than traditional pretraining.

## Evaluation

Test on medical benchmarks (not used in training):
- MedQA (USMLE)
- MedMCQA
- PubMedQA

### Quick evaluation (10 questions)
```bash
python evaluate.py --checkpoint output/checkpoint_10000.pt --num_questions 10
```

### Full evaluation
```bash
python evaluate.py --checkpoint output/checkpoint_10000.pt --num_questions 500
```

### Options
- `--model`: Model config (tiny/small/medium/baguette)
- `--benchmarks`: Which benchmarks to run (default: medqa,medmcqa,pubmedqa)
- `--no_reasoning`: Don't use reasoning format (just raw completion)
- `--device`: cuda or cpu

## References

- [SYNTH Dataset](https://huggingface.co/datasets/PleIAs/SYNTH) - Reasoning chains from PleIAs
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - Training optimizations
- [Baguettotron](https://huggingface.co/PleIAs/Baguettotron) - Deep architecture inspiration
