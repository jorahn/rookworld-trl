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
- Run training: `./train.sh` or `uv run rookworld-train --stable`
- **Manual debug training**: `uv run python manual_grpo_debug.py --steps 500 --task_type A --lr_schedule advanced`
- Inspect rewards: `uv run rookworld-inspect --batch_size 4`
- TensorBoard: `tensorboard --logdir grpo_output_*/runs`
- Lint/format: `uv run black . && uv run isort . && uv run flake8`
- Type check: `uv run mypy src`
- Tests: `uv run pytest -q`

## Recent Critical Updates (2025-09-18)

### Major Breakthrough: Stable GRPO Training Achieved
- **Fixed catastrophic collapse bug**: Eval mode switching issue resolved
- **Successful 435-step training run**: Longest stable training achieved
- **Performance maintained**: 66% accuracy (vs 68% baseline) without model degradation

### Learning Rate Scheduler Improvements
- **Advanced 3-phase schedule**: `--lr_schedule advanced` (new default)
  - Phase 1: Linear warmup from 0 to base_lr
  - Phase 2: Cosine decay to 5% of base_lr (70% of training)
  - Phase 3: Linear annealing from 5% to 0 (30% of training)
- **Fixed warmup bug**: Warmup now applies to ALL schedules including "constant"
- **Ultra-low base LR**: Default changed from 2e-6 to 1e-7 to prevent training collapse

### Critical Bug Fixes
- **Eval mode fix**: Fixed evaluation function to properly switch between eval/training modes
  - Previously caused catastrophic model collapse after first training step
  - Now uses `generate_eval()` which preserves training state correctly
- **Max tokens**: Evaluation now uses 144 tokens (was 100) for consistency

### New Parameters
- `--lr_schedule`: Choose from constant, cosine, linear, step, advanced (default: advanced)
- `--lr_warmup_steps`: Warmup steps for LR schedule (default: 20, was 0)
- `--task_type`: Focus training on P: (policy) or A: (environment) tasks
- `--eval_every`: Evaluate on held-out set every N steps (0=disabled)
- `--checkpoint_every`: Create checkpoints every N steps (-1=only at end)
- `--save_eval_samples`: Save detailed eval predictions to JSONL files (default: false)

### Known Issues
- **Tensor crash at step 435**: Empty rewards tensor causes reshape failure (GitHub issue #6)
- Training is stable for 400+ steps but may crash near completion

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
