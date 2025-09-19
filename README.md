# RookWorld TRL - Experimental GRPO Chess Training

Co-authors: Claude Code, OpenAI Codex

An **experimental implementation** of **Group Relative Policy Optimization (GRPO)** for chess using HuggingFace Transformers and TRL. This project explores GRPO training dynamics, debugging model degradation issues, and optimizing parameters for chess language models.

## 🎯 Overview

This experimental package explores **GRPO training for chess language models** with:

- **🏆 Chess-Accurate Rewards**: Sophisticated Stockfish-based evaluation system  
- **📊 Mixed Task Learning**: Both Policy (P:) and Environment (A:) tasks from RookWorld dataset
- **⚡ Optimized Performance**: Batch size 16, BF16 precision, efficient Stockfish integration
- **🛡️ Experimental Stability**: Gradient clipping, learning rate warmup, data quality validation
- **📈 Research Monitoring**: Tensorboard logging, frequent checkpoints, detailed metrics
- **🔬 Learning-Focused**: Manual debugging tools and algorithmic analysis

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Install Stockfish (Ubuntu/Debian)
sudo apt install stockfish
```

### Stable Training (Recommended)

```bash
# Optimal GRPO training based on 48-run hyperparameter sweep
uv run python manual_grpo_debug.py \
  --steps 200 \
  --task_type A \
  --batch_size 8 \
  --gens 16 \
  --learning_rate 2e-07 \
  --lr_schedule advanced \
  --lr_warmup_steps 15 \
  --entropy_coef 0.005 \
  --eval_every 20 \
  --seed 42

# Run comprehensive hyperparameter sweep (2 GPUs × 4 concurrent = 8 parallel)
./scripts/launch_sweep.sh

# Monitor training progress
tensorboard --logdir grpo_output_*/runs
```

### Legacy Training (May be unstable)

```bash
# Original training script (may have stability issues)
./train.sh

# Conservative training for stability testing
uv run rookworld-train --stable

# Custom parameter exploration
uv run rookworld-train --batch_size 8 --learning_rate 5e-6 --beta 0.2
```

### Testing & Inspection

```bash
# Test package installation
uv run python example.py

# Inspect reward function behavior
uv run rookworld-inspect --batch_size 4

# Debug reward calculations
uv run rookworld-inspect --batch_size 2 --model_name jrahn/RookWorld-LM-124M

# Manual GRPO debugging (step-by-step analysis)
uv run python manual_grpo_debug.py --seed 42

# Manual GRPO: sequential steps with beta adaptation and checkpoints
uv run python manual_grpo_debug.py --steps 3 --beta_adapt --target_kl 0.5 --checkpoint_every 2 --seed 42

# Manual GRPO: overfit a single batch (useful diagnostic)
uv run python manual_grpo_debug.py --steps 10 --overfit_single_batch --seed 42
```

## 🏗️ Architecture

### Enhanced Reward System
- **Best Move Verification**: Stockfish ground truth comparison (50% weight)
- **Move Candidates Quality**: Set overlap with top 5 Stockfish moves (30% weight)  
- **Evaluation Accuracy**: MAE regression against true pawn values (20% weight)

### RookWorld Dataset Integration
- **Mixed Task Support**: Both P: (policy analysis) and A: (environment prediction)
- **Data Quality Validation**: Malformed tasks automatically filtered out
- **Format Verification**: Only valid chess positions and moves included

### GRPO Training with Stability
- **Optimized Batch Size**: 16 samples per batch for optimal throughput
- **Gradient Clipping**: Prevents explosive gradient updates
- **Learning Rate Warmup**: 100 steps for stable training initialization
- **Conservative Mode**: `--stable` flag for extra stability (LR=1e-6, beta=0.2)

## 📊 Experimental Configuration

### Hyperparameter Sweep Results (48 Runs - Evidence-Based)
```bash
# OPTIMAL CONFIGURATION (Top performer from systematic sweep)
Learning rate: 1-3e-07      # Sweet spot - higher rates (>1e-6) often harmful
LR schedule: advanced       # Outperforms cosine in most cases
Batch size: 8               # Memory-efficient, stable performance
Generations: 16-24          # Balanced GRPO signal vs memory usage
Entropy coef: 0.005         # Exploration vs exploitation balance
Beta (KL penalty): 0.005    # Fixed (hardcoded in manual_grpo_debug.py)
Warmup steps: 10-20         # Moderate warmup provides stability
Evaluation: every 10-20     # Frequent monitoring for early detection

# DANGER ZONES (Avoid - causes performance degradation)
Learning rate: >1e-6        # Consistently causes -0.65 to -0.70 performance drop
High batch sizes: >12       # Diminishing returns, memory constraints
Low entropy: <0.001         # Reduces exploration, mode collapse risk
```

### Generation Parameters & Units
- Evaluations (`E:`) use pawn units throughout (e.g., `0.30` = +0.30 pawns). Ground-truth engine scores are converted from centipawns to pawns before MAE.
- Tooling (example, inspect, manual debug) applies task-conditional generation for better behavior:
  - P: tasks → `temperature=0.5`, `top_p=0.9` (focused)
  - A: tasks → `temperature=0.95`, `top_p=0.95` (slightly more permissive)
- Attention masks are passed explicitly during generation to avoid tokenizer warnings and ensure stable behavior.

### Reproducibility & Determinism
- Use `--seed` with manual debug to make runs reproducible (default `42`).
- Scoring uses fixed-depth Stockfish analysis for deterministic rewards.
- Sampling determinism: RNG is seeded per generate call; repeated runs with the same `--seed` match closely.
- Example:
```bash
uv run python manual_grpo_debug.py --steps 2 --overfit_single_batch --seed 42
```

### Task-Conditional Training (rookworld-train)
- Enable alternating P/A phases with distinct sampling:
```bash
uv run rookworld-train --task_conditional_gen \
  --p_temperature 0.5 --p_top_p 0.9 \
  --a_temperature 0.95 --a_top_p 0.95
```

### Stability Features
- **Gradient Explosion Protection**: `max_grad_norm=1.0`
- **Learning Rate Warmup**: Prevents early instability
- **Data Quality Validation**: Malformed samples automatically filtered
- **Frequent Checkpoints**: Save every 100 steps
- **Conservative Mode**: `--stable` flag for maximum stability

### Performance Optimizations
- **Batch Size 16**: Benchmarked optimal for RTX 4090 (4x improvement over default)
- **BF16 Precision**: Faster training with minimal quality loss
- **Efficient Stockfish Integration**: Cached evaluations, optimized depth
- **Tensorboard Integration**: Real-time monitoring of GRPO-specific metrics

## 📈 Monitoring & Analysis

### Tensorboard Metrics
```bash
# Start tensorboard
tensorboard --logdir grpo_output/runs

# Monitor key metrics:
# - loss trends and stability
# - reward progression
# - KL divergence (should stay < 1000)
# - gradient norms (should stay < 100)
# - completion quality metrics
```

### Experimental Indicators
- **Successful experiments**: Loss decreases steadily, rewards improve
- **Training issues**: Large loss spikes, high gradient norms
- **Interesting results**: Stable loss around 0.5-2.0 after 500+ steps

## 🔧 Advanced Usage

### Hyperparameter Sweep Infrastructure

```bash
# Launch comprehensive multi-GPU sweep (evidence-based parameter exploration)
./scripts/launch_sweep.sh

# Custom sweep parameters
uv run python scripts/run_random_sweep.py \
  --runs 24 \
  --steps 100 \
  --eval-every 20 \
  --task-type A \
  --parallel-gpus 2 \
  --max-concurrent-per-gpu 4

# Dry run to preview sweep commands
uv run python scripts/run_random_sweep.py --runs 4 --dry-run
```

### Experimental Configurations

```bash
# Evidence-based optimal training (from 48-run sweep)
uv run python manual_grpo_debug.py \
  --learning_rate 2e-07 \
  --lr_schedule advanced \
  --batch_size 8 \
  --gens 16 \
  --entropy_coef 0.005

# Conservative parameter testing (safe learning rate range)
uv run python manual_grpo_debug.py \
  --learning_rate 1e-07 \
  --lr_warmup_steps 20 \
  --steps 200

# Quick testing/debugging
uv run python manual_grpo_debug.py \
  --steps 20 \
  --batch_size 4 \
  --gens 8 \
  --eval_every 5
```

### Environment Variables
All training parameters can be overridden:
```bash
MODEL_NAME="your-model"           # Base model to fine-tune
OUTPUT_DIR="./custom_output"      # Training output directory
BATCH_SIZE=12                     # Batch size
LEARNING_RATE=5e-6               # Learning rate
BETA=0.1                         # KL penalty coefficient (optimal balance)
DATASET_SIZE=2000                # Number of training samples
MAX_GRAD_NORM=0.5                # Gradient clipping threshold
WARMUP_STEPS=200                 # Warmup duration

# Task-conditional generation for train.sh
TASK_CONDITIONAL_GEN=true \
P_TEMPERATURE=0.5 P_TOP_P=0.9 \
A_TEMPERATURE=0.95 A_TOP_P=0.95 \
./train.sh
```

### Troubleshooting

**If training is unstable:**
```bash
# Use stable mode
uv run rookworld-train --stable

# Or manually set conservative parameters
uv run rookworld-train --learning_rate 1e-6 --max_grad_norm 0.3 --warmup_steps 300

# Adjust generation parameters for better completion quality
uv run rookworld-train --temperature 0.5 --top_p 0.9
```

**If loss spikes occur:**
- Check tensorboard for gradient norm spikes
- Reduce learning rate or increase warmup steps
- Consider lower batch size if memory constrained

**For researching GRPO training dynamics:**

## Manual Debug vs. Trainer Hparams (Temporary)

The manual debug script (`manual_grpo_debug.py`) uses the following intuitive, group‑agnostic hparam names:

- `--batch_size`: prompts per microbatch.
- `--num_generations/--gens`: completions per prompt (GRPO group size).
- `--grad_accum_steps/--ga`: number of chunks the manual debug splits the flattened (prompt × generation) samples into within a batch; it applies an optimizer step per chunk. This is optimized for fast, memory‑lean local iteration and is NOT “true GA” in the HuggingFace trainer sense.

Note: This semantics is currently inconsistent with `example.py` and the main training package (`src/rookworld_trl/train.py`), where GA and batch sizing follow standard trainer conventions. We’ll converge these in a future cleanup; for now, the manual debug script prioritizes clarity and hackability for single‑batch overfit investigations.
```bash
# Step-by-step manual GRPO analysis (experimental debugging)
uv run python manual_grpo_debug.py

# Research model behavior changes during training
# Analyze reward calculation, advantages, and gradient updates  
# Explore where and why pretrained model performance changes
```

## 📊 Reward System Example

```python
# Policy Task Example
prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
response = "M: e2e4 d2d4 g1f3 E: 0.30 0.35 0.28 B: e2e4"

# Component scoring:
# Best move (50%): 1.0 (e2e4 matches Stockfish #1)
# Candidates (30%): 0.6 (3/5 moves in Stockfish top 5)  
# Evaluations (20%): 0.95 (low MAE, correct signs)
# Final: 0.5*1.0 + 0.3*0.6 + 0.2*0.95 = 0.868
```

## 🎮 Package Commands

### Training Commands
- `rookworld-train`: Main GRPO training with experimental features
- `./train.sh`: Convenience script with tested parameters

### Research Commands  
- `rookworld-inspect`: Reward function analysis and batch debugging
- `python example.py`: Package functionality demonstration
- `python manual_grpo_debug.py`: Step-by-step algorithm analysis

### Experimental Parameters
- `--stable`: Conservative mode for stability experiments
- `--tensorboard`: Enable metrics logging for analysis
- `--eval_steps`: Evaluation frequency (default: 100)
- `--max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `--warmup_steps`: Learning rate warmup duration (default: 100)
- `--temperature`: Generation temperature (default: 0.5, experimentally tuned)
- `--top_p`: Nucleus sampling threshold (default: 0.9)
- `--beta`: KL penalty coefficient (default: 0.1, found optimal)

## 🔧 Debugging Tools

### Manual GRPO Analysis
The `manual_grpo_debug.py` script provides step-by-step analysis of the GRPO training process:

```bash
# Run detailed GRPO debugging
uv run python manual_grpo_debug.py
```

**What it analyzes:**
- Model performance before/after GRPO steps
- Completion generation quality and diversity
- Reward calculation and advantage computation  
- KL divergence and policy gradient calculations
- Exact loss breakdown and gradient updates

**Use cases:**
- Debugging why pretrained model performance degrades during training
- Analyzing reward distributions and advantage calculations
- Identifying optimal generation parameters (temperature, top_p)
- Comparing manual implementation with TRL training logs

## 📄 License

MIT License - See LICENSE file for details.

---

**Experimental GRPO training setup for exploring chess language model optimization! 🧪**
