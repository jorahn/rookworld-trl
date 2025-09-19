#!/bin/bash

# Focused hyperparameter sweep launcher for manual GRPO
# 24 runs × 100 steps each - refined parameter ranges based on 48-run analysis
# Focuses on proven high-performer parameter ranges

set -euo pipefail

# Configuration - REFINED based on sweep analysis
RUNS=24
STEPS=100
EVAL_EVERY=20
TASK_TYPE="A"  # Environment tasks (A:)
SWEEP_SEED=20250919

# UV cache optimization
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

echo "🎯 Launching FOCUSED sweep: ${RUNS} runs × ${STEPS} steps each (2 GPUs × 4 concurrent = 8 max parallel)"
echo "   Task type: ${TASK_TYPE}"
echo "   Eval every: ${EVAL_EVERY} steps"
echo "   UV cache: ${UV_CACHE_DIR}"
echo "   Memory-safe: 4 concurrent per GPU (accounts for gradient update spikes)"
echo ""
echo "   🔬 REFINED PARAMETER RANGES (based on 48-run analysis):"
echo "   - Learning rate: 8e-8 to 4e-7 (focused on proven optimal range)"
echo "   - Batch size: 8 (fixed - ALL high performers used this)"
echo "   - Generations: 16, 24, 32 (from successful runs)"
echo "   - Advanced schedule: 65% weight (preferred by top performers)"
echo "   - Entropy: 0.001-0.005 (all values appeared in best runs)"
echo ""

# Launch the focused sweep
uv run python scripts/run_random_sweep.py \
    --runs "${RUNS}" \
    --steps "${STEPS}" \
    --eval-every "${EVAL_EVERY}" \
    --task-type "${TASK_TYPE}" \
    --sweep-seed "${SWEEP_SEED}" \
    --parallel-gpus 2 \
    --max-concurrent-per-gpu 4

echo ""
echo "✅ Focused sweep completed! Check logs/sweeps/ for results."
echo "💡 This sweep explores refined parameter ranges based on evidence from 48-run analysis."