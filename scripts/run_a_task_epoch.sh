#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=${DATASET_PATH:-opening_dataset_train.json}
EVAL_DATASET=${EVAL_DATASET:-opening_dataset_eval.json}
MODEL_NAME=${MODEL_NAME:-jrahn/RookWorld-LM-124M}
LR=${LR:-2e-7}
BETA=${BETA:-0.02}
BATCH_SIZE=${BATCH_SIZE:-16}
GENERATIONS=${GENERATIONS:-16}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_SAMPLES=${EVAL_SAMPLES:-200}
LOGGING_STEPS=${LOGGING_STEPS:-5}
# Example overrides (uncomment / export before running):
# export DATASET_PATH="opening_dataset_train.json"
# export EVAL_DATASET="opening_dataset_eval.json"
# export MODEL_NAME="jrahn/RookWorld-LM-124M"
# export LR="2e-7"
# export BETA="0.02"
# export BATCH_SIZE="16"
# export GENERATIONS="16"
# export EVAL_INTERVAL="10"
# export EVAL_SAMPLES="200"
# export LOGGING_STEPS="5"
# export TRAIN_LIMIT="4096"
# export OUTPUT_ROOT="grpo_runs"
# export RUN_ID="a_task_$(date +%y%m%d-%H%M%S)"

OUTPUT_ROOT=${OUTPUT_ROOT:-grpo_runs}

export DATASET_PATH
export EVAL_DATASET

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "✗ training dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_DATASET}" ]]; then
  echo "✗ eval dataset not found: ${EVAL_DATASET}" >&2
  exit 1
fi

TRAIN_SAMPLES=$(python3 - <<'PY'
import json, os
path = os.environ['DATASET_PATH']
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
PY
)
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  TRAIN_SAMPLES=${TRAIN_LIMIT}
fi

RUN_ID=${RUN_ID:-"a_task_$(date +%y%m%d-%H%M%S)"}
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"

export WANDB_DISABLED=1

set -x
uv run python train_a_min.py \
  --dataset "${DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL_NAME}" \
  --limit "${TRAIN_SAMPLES}" \
  --learning_rate "${LR}" \
  --beta "${BETA}" \
  --generations "${GENERATIONS}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs 1 \
  --max_steps -1 \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_dataset "${EVAL_DATASET}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --eval_samples "${EVAL_SAMPLES}" | tee "${LOG_FILE}"
set +x

echo "✓ run complete: ${OUTPUT_DIR}"
