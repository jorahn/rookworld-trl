#!/bin/bash
#
# RookWorld TRL - Default GRPO Training Script
#
# This script runs GRPO training with sensible default parameters
# for the RookWorld chess dataset using Transformers + TRL.
#

set -e  # Exit on any error

# Default training parameters (can be overridden with env vars)
MODEL_NAME="${MODEL_NAME:-jrahn/RookWorld-LM-124M}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_output}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
BETA="${BETA:-0.1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
DATASET_SIZE="${DATASET_SIZE:-500}"

# Hardware optimizations
USE_BF16="${USE_BF16:-true}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-false}"

echo "🏆 RookWorld TRL - GRPO Training"
echo "=================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Generations per prompt: ${NUM_GENERATIONS}"
echo "Beta (KL coef): ${BETA}"
echo "Max completion length: ${MAX_COMPLETION_LENGTH}"
echo "Dataset size: ${DATASET_SIZE}"
echo "BF16: ${USE_BF16}"
echo "Torch compile: ${USE_TORCH_COMPILE}"
echo "=================================="

# Build command arguments
ARGS=(
    --model_name "${MODEL_NAME}"
    --output_dir "${OUTPUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --num_epochs "${NUM_EPOCHS}"
    --num_generations "${NUM_GENERATIONS}"
    --beta "${BETA}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --dataset_size "${DATASET_SIZE}"
)

# Add optional arguments
if [[ "${USE_BF16}" == "true" ]]; then
    ARGS+=(--bf16)
fi

if [[ "${USE_TORCH_COMPILE}" == "true" ]]; then
    ARGS+=(--compile)
fi

if [[ -n "${STOCKFISH_PATH}" ]]; then
    ARGS+=(--stockfish_path "${STOCKFISH_PATH}")
fi

# Run training
echo "🚀 Starting GRPO training..."
echo "Command: uv run rookworld-train ${ARGS[*]}"
echo ""

exec uv run rookworld-train "${ARGS[@]}"