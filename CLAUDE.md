# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- `uv sync` - Install dependencies (uses uv for fast package management)
- `uv run python example.py` - Test package installation

### Training Commands
- `./scripts/launch_sweep.sh` - **RECOMMENDED**: Comprehensive hyperparameter sweep (48 runs, 2 GPUs)
- `uv run python manual_grpo_debug.py --learning_rate 2e-07 --lr_schedule advanced --batch_size 8 --gens 16` - **Evidence-based optimal training**
- `./train.sh` - Legacy GRPO training script (may be unstable)
- `uv run rookworld-train --stable` - Conservative training mode for stability

### Analysis & Debugging
- `uv run rookworld-inspect --batch_size 4` - Inspect reward function behavior
- `uv run python manual_grpo_debug.py --seed 42` - Step-by-step GRPO debugging
- `uv run python manual_grpo_debug.py --overfit_single_batch` - Diagnostic overfit mode

### Key Performance Indicators (KPIs) for A: Tasks
- **PRIMARY KPI**: Evaluation accuracy (% correct on held-out set) - target 80%+
- **SECONDARY KPI**: Evaluation reward (average reward on held-out set) - higher is better
- **CRITICAL CHALLENGE**: Move 1 accuracy (starting position transitions) - currently 0-8%
- **Note**: Training performance metrics can improve while eval metrics worsen (overfitting)
- **Evidence-based parameters for eval performance** (corrected analysis, 2025-09-19):
  - `--learning_rate 1.2e-07` - **CRITICAL**: Conservative rates (1-2.4e-07) best for eval accuracy
  - `--lr_schedule cosine` - Slightly outperforms advanced for eval metrics
  - `--lr_warmup_steps 10` - Moderate warmup (10-20 range)
  - `--batch_size 8` - Confirmed optimal across all experiments
  - `--gens 16` - Moderate GRPO signal, good generalization
  - `--entropy_coef 0.001` - **Low entropy best for eval performance** (not 0.005!)
  - `--task_type A|P|mixed` - Focus on environment (A:) or policy (P:) tasks
  - `--eval_every 10` - **CRITICAL**: Monitor both training AND eval to detect overfitting
  - `--checkpoint_every -1` - Only checkpoint at end (-1, default) or every N steps
- Manual debug flags of interest:
  - `--gens/--num_generations <int>`: completions per prompt (GRPO group size)
  - `--ga/--grad_accum_steps <int>`: number of within-batch chunks processed (default 1)
  - `--beta_warmup_steps <int>`: warmup steps with beta=0 (default 20)
  - `--entropy_coef <float>`: entropy regularization (default 0.005)
  - `--learning_rate/--lr <float>`: base learning rate (default 1e-7)
  - Full run logs are written to `logs/manual_grpo_debug_run-<YYMMDD-HHMMSS>.log` and streamed to stdout
  - Important: Do not reduce `max_new_tokens` — the reward schema depends on this length budget; changing it breaks evaluation comparability.

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
- **Stable Parameters**: Batch size 4, LR 1e-7, Beta 0.005, Advanced LR schedule
- **Stability Features**: Gradient clipping (1.0), LR warmup (20 steps), Beta warmup (20 steps)
- **Critical**: Use `generate_eval()` for evaluation to prevent training collapse

### Dataset Handling
- **RookWorld Integration**: Loads from HuggingFace hub with preprocessing
- **Task Prefixing**: Auto-detects and adds "A: " prefix for environment tasks
- **Quality Validation**: Filters malformed chess positions and moves

### Training Features
- **Task-Conditional Generation**: Different sampling params for P: vs A: tasks
- **Deterministic Debug Mode**: Seeded runs with fixed-depth Stockfish
- **Overfit Diagnostics**: Single-batch training for debugging model degradation
- **Tensorboard Logging**: Real-time metrics and loss tracking

### Entry Points & Scripts
- `./scripts/launch_sweep.sh` - **RECOMMENDED**: Multi-GPU hyperparameter sweep
- `scripts/run_random_sweep.py` - Customizable sweep infrastructure with parallel execution
- `rookworld-train` - Main training command (legacy, may be unstable)
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
