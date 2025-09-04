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
- Inspect rewards: `uv run rookworld-inspect --batch_size 4`
- TensorBoard: `tensorboard --logdir grpo_output_*/runs`
- Lint/format: `uv run black . && uv run isort . && uv run flake8`
- Type check: `uv run mypy src`
- Tests: `uv run pytest -q`

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
