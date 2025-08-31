#!/bin/bash
#
# RookWorld TRL - GRPO Training Script
#
# Optimized configuration with stability improvements:
# - Batch size 16 for optimal throughput
# - Gradient clipping and warmup for stability  
# - Tensorboard logging enabled
# - Evaluation every 100 steps
#

set -e  # Exit on any error

# Optimized training parameters (can be overridden with env vars)
MODEL_NAME="${MODEL_NAME:-jrahn/RookWorld-LM-124M}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_output}"
BATCH_SIZE="${BATCH_SIZE:-16}"  # Optimized batch size from benchmarking
LEARNING_RATE="${LEARNING_RATE:-1e-6}"  # Increased from 1e-7
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
BETA="${BETA:-1.0}"  # High KL penalty to prevent catastrophic forgetting
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
DATASET_SIZE="${DATASET_SIZE:-5000}"  # Larger dataset for substantial training

# Evaluation and logging (optimized)
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# Stability parameters
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

# Generation parameters (focused sampling)
TEMPERATURE="${TEMPERATURE:-0.5}"  # Lower temperature for focused sampling
TOP_P="${TOP_P:-0.9}"              # Keep nucleus sampling

# Hardware optimizations
USE_BF16="${USE_BF16:-true}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-false}"

echo "🚀 RookWorld TRL - GRPO Training (Optimized + Stable)"
echo "====================================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE} (optimized)"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Dataset size: ${DATASET_SIZE} samples"
echo "Max steps: ~$(( DATASET_SIZE / BATCH_SIZE )) steps"
echo "Generations per prompt: ${NUM_GENERATIONS}"
echo "Beta (KL coef): ${BETA} (high penalty)"
echo "Max completion length: ${MAX_COMPLETION_LENGTH}"
echo "Gradient clipping: ${MAX_GRAD_NORM}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Temperature: ${TEMPERATURE} (focused)"
echo "Top-p: ${TOP_P}"
echo "Eval every: ${EVAL_STEPS} steps"
echo "Save every: ${SAVE_STEPS} steps"
echo "Log every: ${LOGGING_STEPS} steps"
echo "BF16: ${USE_BF16}"
echo "Torch compile: ${USE_TORCH_COMPILE}"
echo "====================================================="

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
    --eval_steps "${EVAL_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --warmup_steps "${WARMUP_STEPS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --tensorboard
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
echo "🚀 Starting optimized GRPO training..."
echo "Command: uv run rookworld-train ${ARGS[*]}"
echo ""
echo "📊 Monitor training with:"
echo "   tensorboard --logdir ${OUTPUT_DIR}/runs"
echo ""
echo "🛡️  Knowledge preservation features enabled:"
echo "   • Conservative learning rate: ${LEARNING_RATE} (preserve pretrained knowledge)"
echo "   • High KL penalty: ${BETA} (prevent catastrophic forgetting)"
echo "   • Gradient clipping: ${MAX_GRAD_NORM}"
echo "   • Learning rate warmup: ${WARMUP_STEPS} steps"  
echo "   • Frequent checkpoints: every ${SAVE_STEPS} steps"
echo "   • Use --stable flag for ultra-conservative settings"
echo ""

exec uv run rookworld-train "${ARGS[@]}"