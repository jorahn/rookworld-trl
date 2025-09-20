# Experiments — 2025-09-20

## Reward Function Overhaul
- **Change**: Replaced A-task reward in `train_a_min.py` with Levenshtein-shaped
  scoring. Exact match earns `+1.0`; perfect FEN but suffix errors start at `+0.8`
  and degrade via suffix edit distance; any FEN error yields `[−0.8, −0.2]`
  depending on normalized edit distance with suffix adding up to ±0.2. Returns
  clamp to `[-1, 1]`.
- **Verification**: `uv run --no-sync python - <<'PY' ...` smoke tests confirmed
  expected rewards for exact matches, suffix errors, and FEN mistakes.

## Diagnostic Run — 10 Steps (Beta Warmup + 256 Tokens)
- **Command**:
  ```
  uv run --no-sync python scripts/run_beta_warmup_eval.py
  ```
- **Config highlights**: 10% dataset slices (87 train / 23 eval), 10 GRPO steps,
  `max_new_tokens=256`, adaptive β warmup (target β 0.05 → ≈0.01), grad clip 0.2.
- **Artifacts**: `logs/ten_step_beta_warmup_20250920_113941/` (per-step JSONL,
  summary, checkpoint).
- **Metrics**:
  - Train accuracy ~69%, avg reward ↑ 0.675 → 0.678.
  - Eval accuracy ~61%, avg reward ↑ 0.589 → 0.597.
  - KL spikes early (≈5.9e6) but decay as β tightens; longer completions routinely
    hit the 256-token cap.

## Script: `scripts/run_beta_warmup_eval.py`
- **Purpose**: Reusable harness for these diagnostics (fractional dataset, beta
  warmup, logging). `--steps <= 0` now triggers a full epoch, storing outputs
  under `logs/epoch_beta_warmup_<timestamp>/`.

## Full-Epoch Diagnostic (10% data)
- **Command**:
  ```
  uv run --no-sync python scripts/run_beta_warmup_eval.py --steps 0
  ```
- **Artifacts**: `logs/epoch_beta_warmup_20250920_114830/` (168-step run).
- **Metrics**:
  - Train accuracy oscillates 65–70%; eval accuracy plateaus at 56–61% with avg
    reward ≈0.54–0.60.
  - β warmup pegs β≈0 after early KL bursts; large KL spikes (≥5e6) still occur
    around the first dozen updates, indicating we may need tighter LR or KL
    targets.
  - Loss values explode (∼1e18) but reward shaping keeps evaluations bounded.

## Notes & Next Steps
- Investigate further KL mitigation (lower LR, stronger clipping, adjusted
  target_kl) to curb the early-step blowups seen in both short and full runs.
- Consider expanding eval slice or adding move-specific diagnostics to validate
  the new reward shaping on harder opening positions.
