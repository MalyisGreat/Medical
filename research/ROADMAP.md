# Medical LLM Pretraining Research Roadmap

Goal: find the cheapest stable recipe that maximizes tokens/sec and quality for small models.

## Core metrics to log
- tokens/sec
- train loss (and eval loss if available)
- step time
- peak VRAM (if GPU)
- wall time per 1k steps

## Overnight loop (repeat daily)
1) Run smoke tests (data load, model forward)
2) Run a short sweep (same token budget across variants)
3) Pick winners by tokens/sec and loss
4) Queue the next sweep focused on the top 2 settings
5) Document any changes to code or data

## Rolling task queue (add more as results come in)
- [ ] Compare AdamW vs Muon on tiny model (same tokens, seq=128)
- [ ] Seq length sweep: 128 vs 256 vs 512 (same tokens)
- [ ] Batch size vs grad accumulation tradeoff (same effective batch)
- [ ] Streaming vs cached shards (tokens/sec + stability)
- [ ] Compile on/off (CUDA only)
- [ ] Data mix: synth-only vs synth+guidelines vs synth+pubmed
- [ ] RMSNorm vs LayerNorm (if we add LayerNorm switch)
- [ ] Tie embeddings vs untied (quality vs speed)
- [ ] QK-norm on/off (stability)
- [ ] Add gradient checkpointing for longer contexts
- [ ] Evaluate on 50-question subsets of MedQA/MedMCQA/PubMedQA

## Upgrade path (when a winner emerges)
- Run 5x longer on the best 2 configs
- Switch to real eval sets
- Lock settings into a baseline config
