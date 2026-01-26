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
python train.py --test_only

# Train tiny model (testing)
python train.py --model tiny --max_tokens 100000 --max_steps 100
```

### Full training (8xH100)
```bash
# Edit train.py line 161 to use medical datasets:
# datasets=["synth", "guidelines", "openmedtext"]

python train.py \
    --model medium \
    --max_tokens 5000000000 \
    --batch_size 32 \
    --max_steps 50000
```

## Cost Estimates

| Model | Tokens | Time (8xH100) | Cost |
|-------|--------|---------------|------|
| medium (406M) | 5B | ~2 hours | ~$24 |
| baguette (308M) | 5B | ~3 hours | ~$36 |

## Evaluation

Test on medical benchmarks (not used in training):
- MedQA (USMLE)
- MedMCQA
- PubMedQA
- MMLU medical subset

## References

- [SYNTH Dataset](https://huggingface.co/datasets/PleIAs/SYNTH) - Reasoning chains from PleIAs
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - Training optimizations
- [Baguettotron](https://huggingface.co/PleIAs/Baguettotron) - Deep architecture inspiration
