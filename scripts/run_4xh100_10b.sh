#!/usr/bin/env bash
set -euo pipefail

# 10B-token training on 4x H100, full dataset download.
# Customize via environment variables if needed.

TOTAL_TOKENS="${TOTAL_TOKENS:-10000000000}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH="${BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORLD_SIZE="${WORLD_SIZE:-4}"
MODEL="${MODEL:-medium}"
DATASETS="${DATASETS:-synth,guidelines,openmedtext}"
OUTPUT_DIR="${OUTPUT_DIR:-./data_shards}"
SHARD_SIZE="${SHARD_SIZE:-100000000}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-16}"

if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi

# Optional: faster HF downloads if installed
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

MAX_STEPS="$(python - <<PY
import math
total = int("${TOTAL_TOKENS}")
seq = int("${SEQ_LEN}")
batch = int("${BATCH}")
accum = int("${GRAD_ACCUM}")
world = int("${WORLD_SIZE}")
tps = seq * batch * accum * world
print(math.ceil(total / tps))
PY
)"

echo "Config:"
echo "  TOTAL_TOKENS=${TOTAL_TOKENS}"
echo "  SEQ_LEN=${SEQ_LEN}"
echo "  BATCH=${BATCH}"
echo "  GRAD_ACCUM=${GRAD_ACCUM}"
echo "  WORLD_SIZE=${WORLD_SIZE}"
echo "  MODEL=${MODEL}"
echo "  DATASETS=${DATASETS}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  SHARD_SIZE=${SHARD_SIZE}"
echo "  DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS}"
echo "  TOKENIZE_WORKERS=${TOKENIZE_WORKERS}"
echo "  TOKENS_PER_STEP=$((SEQ_LEN * BATCH * GRAD_ACCUM * WORLD_SIZE))"
echo "  MAX_STEPS=${MAX_STEPS}"

python prepare_data.py \
  --max_tokens "${TOTAL_TOKENS}" \
  --datasets "${DATASETS}" \
  --output "${OUTPUT_DIR}" \
  --seq_length "${SEQ_LEN}" \
  --shard_size "${SHARD_SIZE}" \
  --download_workers "${DOWNLOAD_WORKERS}" \
  --tokenize_workers "${TOKENIZE_WORKERS}"

torchrun --nproc_per_node="${WORLD_SIZE}" train.py \
  --model "${MODEL}" \
  --data_dir "${OUTPUT_DIR}" \
  --max_seq_length "${SEQ_LEN}" \
  --batch_size "${BATCH}" \
  --max_steps "${MAX_STEPS}" \
  --optimizer muon \
  --compile
