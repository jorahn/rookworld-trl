# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- `uv sync` - Install dependencies (uses uv for fast package management)
- `uv run python example.py` - Test package installation

### Training Commands
- `./train.sh` - Main GRPO training script with optimized parameters
- `uv run rookworld-train` - Direct training with custom arguments
- `uv run rookworld-train --stable` - Conservative training mode for stability

### Analysis & Debugging
- `uv run rookworld-inspect --batch_size 4` - Inspect reward function behavior
- `uv run python manual_grpo_debug.py --seed 42` - Step-by-step GRPO debugging
- `uv run python manual_grpo_debug.py --overfit_single_batch` - Diagnostic overfit mode

### Development Tools
- `black src/` - Code formatting
- `isort src/` - Import sorting
- `flake8 src/` - Linting
- `mypy src/` - Type checking
- `pytest` - Run tests (when available)

### Monitoring
- `tensorboard --logdir grpo_output_*/runs` - Monitor training progress

## Code Architecture

### Core Package Structure (`src/rookworld_trl/`)
- `train.py` - Main GRPO training implementation with TrainingConfig
- `rewards.py` - Chess-specific reward system using Stockfish evaluation
- `dataset.py` - RookWorld dataset loading with P:/A: task handling
- `inspect.py` - Batch reward analysis and debugging tools
- `utils.py` - Shared utilities for text normalization

### Training System
- **GRPO Implementation**: Uses HuggingFace TRL's GRPOTrainer
- **Chess Reward Function**: Multi-component scoring (best move 50%, candidates 30%, evaluations 20%)
- **Stockfish Integration**: Automated path detection with depth-based evaluation
- **Task Types**: P: (Policy analysis) and A: (Environment prediction) from RookWorld

### Key Configuration
- **Model**: `jrahn/RookWorld-LM-124M` (default base model)
- **Optimal Parameters**: Batch size 16, LR 1e-6, Beta 0.1, Temperature 0.5
- **Stability Features**: Gradient clipping (1.0), warmup (100 steps), BF16 precision

### Dataset Handling
- **RookWorld Integration**: Loads from HuggingFace hub with preprocessing
- **Task Prefixing**: Auto-detects and adds "A: " prefix for environment tasks
- **Quality Validation**: Filters malformed chess positions and moves

### Training Features
- **Task-Conditional Generation**: Different sampling params for P: vs A: tasks
- **Deterministic Debug Mode**: Seeded runs with fixed-depth Stockfish
- **Overfit Diagnostics**: Single-batch training for debugging model degradation
- **Tensorboard Logging**: Real-time metrics and loss tracking

### Entry Points (pyproject.toml)
- `rookworld-train` - Main training command
- `rookworld-inspect` - Reward analysis tool

### Environment Variables for train.sh
All training parameters are configurable via env vars:
- `MODEL_NAME`, `OUTPUT_DIR`, `BATCH_SIZE`, `LEARNING_RATE`
- `BETA`, `DATASET_SIZE`, `TEMPERATURE`, `TOP_P`
- `TASK_CONDITIONAL_GEN=true` for P:/A: specific sampling

## Important Notes

### Dependencies
- Requires `stockfish` binary installed (`sudo apt install stockfish`)
- Uses `uv` for package management (not pip)
- Python 3.9+ required

### Model Training Considerations
- Chess knowledge preservation requires conservative learning rates (1e-6)
- Gradient clipping essential for training stability
- Temperature 0.5 prevents gibberish vs TRL default 1.0
- Beta 0.1 provides optimal KL penalty balance

### Debugging Workflow
1. Use `manual_grpo_debug.py` for step-by-step algorithm analysis
2. Enable `--overfit_single_batch` to diagnose model degradation
3. Monitor Tensorboard for gradient norms and KL divergence
4. Use `rookworld-inspect` for reward function verification