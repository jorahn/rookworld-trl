#!/bin/bash

# Large hyperparameter sweep launcher for manual GRPO
# Runs 48 experiments with 50 steps each, targeting comprehensive parameter coverage

set -euo pipefail

# Configuration
RUNS=48
STEPS=50
EVAL_EVERY=10
TASK_TYPE="A"  # Environment tasks (A:)
SWEEP_SEED=20250919

# UV cache optimization
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

echo "🚀 Launching large sweep: ${RUNS} runs × ${STEPS} steps each (2 GPUs × 4 concurrent = 8 max parallel)"
echo "   Task type: ${TASK_TYPE}"
echo "   Eval every: ${EVAL_EVERY} steps"
echo "   UV cache: ${UV_CACHE_DIR}"
echo "   Memory-safe: 4 concurrent per GPU (accounts for gradient update spikes)"
echo "   Sweeping: LR (1e-7 to 2e-6), batch size (4,8,12), generations (16-40), entropy (0.001-0.005)"
echo ""

# Launch the sweep
uv run python scripts/run_random_sweep.py \
    --runs "${RUNS}" \
    --steps "${STEPS}" \
    --eval-every "${EVAL_EVERY}" \
    --task-type "${TASK_TYPE}" \
    --sweep-seed "${SWEEP_SEED}" \
    --parallel-gpus 2 \
    --max-concurrent-per-gpu 4

echo ""
echo "✅ Sweep completed! Check logs/sweeps/ for results."