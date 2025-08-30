# RookWorld TRL - GRPO Chess Training

A sophisticated implementation of **Group Relative Policy Optimization (GRPO)** for chess using HuggingFace Transformers and TRL, featuring advanced chess-accurate reward functions and RookWorld dataset integration.

## 🎯 Overview

This package implements state-of-the-art GRPO training for chess language models, focusing on:

- **🏆 Chess-Accurate Rewards**: Sophisticated Stockfish-based evaluation system
- **📊 Mixed Task Learning**: Both Policy (P:) and Environment (A:) tasks from RookWorld dataset  
- **🚀 Continuous Rewards**: Fine-grained feedback with -1.0 to 1.0 range
- **⚡ Performance Optimized**: BF16, torch.compile, and efficient Stockfish integration
- **🔧 Production Ready**: Self-contained package with comprehensive tooling

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Install Stockfish (Ubuntu/Debian)
sudo apt install stockfish
```

### Basic Training

```bash
# Basic GRPO training
uv run rookworld-train

# With custom parameters
uv run rookworld-train --batch_size 8 --learning_rate 2e-5 --bf16 --num_generations 6

# Using convenience script with defaults
./train.sh

# Inspect reward calculations
uv run rookworld-inspect --batch_size 4

# Run example to test installation
uv run python example.py
```

## 🏗️ Architecture

### Enhanced Reward System
- **Best Move Verification**: Stockfish ground truth comparison (50% weight)
- **Move Candidates Quality**: Set overlap with top 5 Stockfish moves (30% weight)  
- **Evaluation Accuracy**: MAE regression against true centipawn values (20% weight)

### RookWorld Dataset Integration
- **Mixed Task Support**: Both P: (policy analysis) and A: (environment prediction)
- **Robust Parsing**: Handles various RookWorld dataset formats
- **Intelligent Fallbacks**: Synthetic data when dataset unavailable

### GRPO Training
- **True GRPO Algorithm**: Uses TRL's GRPOTrainer (not DPO)
- **Group-Based Learning**: Multiple completions per prompt with relative advantages
- **Performance Optimized**: BF16 precision, torch.compile support

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

## 🔧 Configuration

### Key Parameters
- `--num_generations`: Completions per prompt for GRPO (default: 4)
- `--beta`: KL divergence penalty coefficient (default: 0.1)
- `--max_completion_length`: Min 144 tokens per completion
- `--stockfish_path`: Auto-detects if not specified

## 📄 License

MIT License
