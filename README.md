# RookWorld TRL - Production GRPO Chess Training

A **production-ready** implementation of **Group Relative Policy Optimization (GRPO)** for chess using HuggingFace Transformers and TRL, featuring optimized batch sizes, stability improvements, and comprehensive monitoring.

## 🎯 Overview

This package delivers **stable, high-performance GRPO training** for chess language models with:

- **🏆 Chess-Accurate Rewards**: Sophisticated Stockfish-based evaluation system  
- **📊 Mixed Task Learning**: Both Policy (P:) and Environment (A:) tasks from RookWorld dataset
- **⚡ Optimized Performance**: Batch size 16, BF16 precision, efficient Stockfish integration
- **🛡️ Training Stability**: Gradient clipping, learning rate warmup, data quality validation
- **📈 Comprehensive Monitoring**: Tensorboard logging, frequent checkpoints, detailed metrics
- **🔧 Production Ready**: Self-contained package with battle-tested defaults

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Install Stockfish (Ubuntu/Debian)
sudo apt install stockfish
```

### Training

```bash
# Production training (recommended) - optimized + stable
./train.sh

# Monitor training progress
tensorboard --logdir grpo_output/runs

# Extra stable training for difficult setups
uv run rookworld-train --stable

# Custom training parameters
uv run rookworld-train --batch_size 8 --learning_rate 5e-6 --stable
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
uv run python manual_grpo_debug.py
```

## 🏗️ Architecture

### Enhanced Reward System
- **Best Move Verification**: Stockfish ground truth comparison (50% weight)
- **Move Candidates Quality**: Set overlap with top 5 Stockfish moves (30% weight)  
- **Evaluation Accuracy**: MAE regression against true centipawn values (20% weight)

### RookWorld Dataset Integration
- **Mixed Task Support**: Both P: (policy analysis) and A: (environment prediction)
- **Data Quality Validation**: Malformed tasks automatically filtered out
- **Format Verification**: Only valid chess positions and moves included

### GRPO Training with Stability
- **Optimized Batch Size**: 16 samples per batch for optimal throughput
- **Gradient Clipping**: Prevents explosive gradient updates
- **Learning Rate Warmup**: 100 steps for stable training initialization
- **Conservative Mode**: `--stable` flag for extra stability (LR=1e-6, beta=0.2)

## 📊 Training Configuration

### Default Settings (Optimized + Stable)
```bash
Batch size: 16              # Optimal for RTX 4090 (4x improvement)
Dataset size: 5000 samples  # Substantial training data
Learning rate: 1e-6         # Optimal for chess knowledge preservation
Beta (KL penalty): 0.1      # Optimal KL/PG balance from testing
Temperature: 0.5            # Focused sampling vs TRL default 1.0
Top-p: 0.9                  # Nucleus sampling vs TRL default 1.0
Gradient clipping: 1.0      # Stability protection
Warmup steps: 100          # Stable initialization
Evaluation: every 100 steps # Frequent monitoring
Tensorboard: enabled       # Full metrics tracking
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

### Key Indicators
- **Healthy Training**: Loss decreases steadily, rewards improve
- **Stability Issues**: Large loss spikes, high gradient norms
- **Convergence**: Stable loss around 0.5-2.0 after 500+ steps

## 🔧 Advanced Usage

### Custom Training Configurations

```bash
# High-performance training
BATCH_SIZE=24 LEARNING_RATE=2e-5 ./train.sh

# Ultra-stable training  
uv run rookworld-train --stable --learning_rate 5e-7 --warmup_steps 300

# Large-scale training
DATASET_SIZE=10000 BATCH_SIZE=32 ./train.sh

# Development/testing
DATASET_SIZE=100 BATCH_SIZE=4 ./train.sh
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

**For debugging GRPO training issues:**
```bash
# Step-by-step manual GRPO analysis
uv run python manual_grpo_debug.py

# Compare before/after model performance
# Analyze reward calculation, advantages, and gradient updates
# Identify where pretrained model performance is lost
```

## 📊 Reward System Example

```python
# Policy Task Example
prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
response = "M: e2e4 d2d4 g1f3 E: 30 35 28 B: e2e4"

# Component scoring:
# Best move (50%): 1.0 (e2e4 matches Stockfish #1)
# Candidates (30%): 0.6 (3/5 moves in Stockfish top 5)  
# Evaluations (20%): 0.95 (low MAE, correct signs)
# Final: 0.5*1.0 + 0.3*0.6 + 0.2*0.95 = 0.868
```

## 🎮 Package Commands

### Training Commands
- `rookworld-train`: Main GRPO training with all stability features
- `./train.sh`: Convenience script with production defaults

### Analysis Commands  
- `rookworld-inspect`: Reward function debugging and batch analysis
- `python example.py`: Package functionality demonstration

### Key Parameters
- `--stable`: Ultra-conservative training (prevents instability)
- `--tensorboard`: Enable comprehensive metrics logging
- `--eval_steps`: Evaluation frequency (default: 100)
- `--max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `--warmup_steps`: Learning rate warmup duration (default: 100)
- `--temperature`: Generation temperature (default: 0.5, focused sampling)
- `--top_p`: Nucleus sampling threshold (default: 0.9)

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

**Ready for production chess language model training with optimal performance and stability! 🚀**