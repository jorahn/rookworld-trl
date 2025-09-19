# Repository Guidelines

## Project Structure & Module Organization
- `src/rookworld_trl/`: Python package.
  - `train.py`: GRPO training entrypoint (`rookworld-train`).
  - `inspect.py`: Reward/GRPO inspection CLI (`rookworld-inspect`).
  - `dataset.py`: RookWorld dataset parsing and batching.
  - `rewards.py`: Stockfish-backed reward scoring (fallback without engine).
  - `utils.py`: Spacing normalization utilities.
- `train.sh`: Opinionated training wrapper with env overrides.
- `example.py`: Simple sanity check script.
- `README.md`, `manual_*.md`: Background and experiments.

## Build, Test, and Development Commands
- Install deps: `uv sync`
- **RECOMMENDED training**: `./scripts/launch_sweep.sh` (48-run hyperparameter sweep, 2 GPUs)
- **Evidence-based optimal training**: `uv run python manual_grpo_debug.py --learning_rate 2e-07 --lr_schedule advanced --batch_size 8 --gens 16 --entropy_coef 0.005`
- Legacy training: `./train.sh` or `uv run rookworld-train --stable` (may be unstable)
- Custom sweep: `uv run python scripts/run_random_sweep.py --runs 24 --parallel-gpus 2`
- Inspect rewards: `uv run rookworld-inspect --batch_size 4`
- TensorBoard: `tensorboard --logdir grpo_output_*/runs`
- Lint/format: `uv run black . && uv run isort . && uv run flake8`
- Type check: `uv run mypy src`
- Tests: `uv run pytest -q`

## Recent Critical Updates (2025-09-19)

### Comprehensive Hyperparameter Sweep Completed ✅
- **48 successful runs**: Complete systematic exploration of GRPO parameter space
- **Multi-GPU infrastructure**: 2 GPUs × 4 concurrent = 8 parallel experiments (6x speedup)
- **Evidence-based optimization**: Clear learning rate sensitivity patterns identified

### Key Findings - Learning Rate Sensitivity (CRITICAL)
- **Optimal range**: 1e-7 to 3e-7 provides stable positive gains (+0.12 avg ΔPerf)
- **Danger zone**: >1e-6 learning rates consistently harmful (-0.35 avg ΔPerf, up to -0.70)
- **Sweet spot confirmed**: 2e-07 with advanced schedule = top performer (+0.1762 performance gain)

### Infrastructure Improvements
- **Sweep automation**: `./scripts/launch_sweep.sh` for comprehensive parameter exploration
- **Parallel execution**: Memory-safe multi-GPU training with VRAM management
- **Enhanced logging**: Full log file parsing with metrics extraction from manual_grpo_debug runs

### Stability Achievements (2025-09-18)
- **Fixed tensor crashes**: PR #7 comprehensive error handling (100% success rate)
- **Extended training**: Validated up to 500 steps without crashes
- **Move 1 breakthrough**: First progress on hardest opening positions (0% → 8%)

### Key Performance Indicators (KPIs) for A: Tasks
- **PRIMARY KPI**: Evaluation accuracy (% correct on held-out set) - target 80%+
- **SECONDARY KPI**: Evaluation reward (average reward on held-out set) - higher is better
- **CRITICAL CHALLENGE**: Move 1 accuracy (starting position transitions) - currently 0-8%
- **WARNING**: Training metrics can improve while eval metrics worsen (overfitting risk)

### Evidence-Based Best Practices (CORRECTED)
- **Learning rate**: 1.5-2.4e-07 (only range showing eval accuracy improvement in focused sweep)
- **LR schedule**: Both advanced and cosine show similar eval performance
- **Batch/generations**: 8×16 or 8×24 configurations
- **Entropy coefficient**: 0.002 shows best eval accuracy results
- **Evaluation frequency**: Every 10-20 steps for early overfitting detection
- **Overfitting concern**: Tighter parameter ranges may cause training/eval divergence

## Reproducibility & Determinism
- Manual debug: use `--seed` (default `42`) for reproducible sampling and batching.
- Reward scoring is depth-based (fixed-depth Stockfish) to reduce timing jitter.
- Overfit mode: `--overfit_single_batch` reuses the same batch each step to observe update dynamics.

## Coding Style & Naming Conventions
- Python 3.9+; Black line length 88; Isort profile “black”.
- Use type hints; `mypy` strictness: disallow untyped defs.
- Modules/functions: `snake_case`; constants: `UPPER_SNAKE_CASE`; classes: `CamelCase`.
- Keep public CLI flags mirrored in `TrainingConfig` defaults.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`, `tests/` root; `test_*.py` or `*_test.py`).
- Prefer fast, engine-free unit tests; mock Stockfish when needed.
- Add tests for: parsing in `dataset.py`, reward helpers in `rewards.py`, and spacing utils.
- If adding a CLI, include a smoke test using `uv run -m` or entrypoint.

## Commit & Pull Request Guidelines
- Conventional Commits: `feat:`, `fix:`, `docs:`, etc. (see recent history).
- Write focused commits; reference issues with `#123` when applicable.
- PRs must include: summary, rationale, runnable examples/commands, and notes on perf/compat risks. Attach logs or TensorBoard screenshots if relevant.
- CI expectations: code formatted, lint/type checks pass, tests pass locally.

## Security & Configuration Tips
- Stockfish: install via OS package manager or set `STOCKFISH_PATH` env var. Example: `STOCKFISH_PATH=/usr/bin/stockfish uv run rookworld-train`.
- Training knobs via env in `train.sh`: `MODEL_NAME`, `OUTPUT_DIR`, `BATCH_SIZE`, `LEARNING_RATE`, `BETA`, `WARMUP_STEPS`.
- Task-conditional sampling (train.sh): set `TASK_CONDITIONAL_GEN=true` and provide per-task params `P_TEMPERATURE`, `P_TOP_P`, `A_TEMPERATURE`, `A_TOP_P`.
- Normalize spacing before scoring/generation to avoid KL inflation (see `utils.normalize_spacing`).
 - Do not reduce `max_new_tokens` in manual debug or training runs: the reward schema expects this sequence budget and altering it breaks comparability and evaluation assumptions.
 - Logs: `logs/` is ignored by default; only curated full-run logs may be force-added for reproducibility and are referenced from experiments logs.
