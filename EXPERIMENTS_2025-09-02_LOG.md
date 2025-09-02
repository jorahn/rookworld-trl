# RookWorld TRL — Experiments Log (2025‑09‑02)

This log captures today’s changes, hyperparameter revisions, observed behavior, and conclusions for the overfit‑one‑batch sanity check and related training plumbing.

## Context
- Goal: Make the manual GRPO overfit‑one‑batch test show a stable, upward reward trend while avoiding KL runaway and mode collapse.
- Secondary: Remove task‑conditional trainer duplication and add a dry‑run path to the main training entrypoint.

## Code Changes
- Train (`src/rookworld_trl/train.py`)
  - Reuse a single `GRPOTrainer` across P/A phases (removed duplicate trainer instances).
  - Switch phases by reassigning `train_dataset` and generation params on the same trainer.
  - Add `--dry_run` to validate dataset splits and phase scheduling without loading model/rewards.
  - Fix `save_model` call for task‑conditional branch.

- Manual debug (`manual_grpo_debug.py`)
  - Added overfit‑focused defaults and iterative fixes across revisions (r1→r4):
    - Advantage handling: std clamp (0.05) and advantage clipping (±2.0).
    - KL term:
      - Replaced sign‑ambiguous log‑prob difference and squared diff surrogate with true token‑level KL(p||q) computed over the vocabulary (policy with grads, reference no‑grad).
    - Beta schedule:
      - Warmup β=0 for 20 steps, then fixed β after warmup.
      - Final setting: β_after_warmup=0.005 (with adaptation OFF by default; when ON, clamp β∈[0, 0.05] and `target_kl=1.0`).
    - Sampling and variance:
      - Entropy regularization added; final entropy_coef=0.005 (was 0.01).
      - Task‑conditional generation tuned (P lower temp, A higher), and P‑only batch preference for the overfit test.
      - num_generations increased from 4→8→12 for steadier PG.
    - Eval stability: switched post‑step metric to greedy decoding (no sampling).
    - Learning rate: 1e‑6→3e‑6, then stabilized at 2e‑6 for smoother steps.
    - Logging: config banner with revision tag (r4) and explicit KL/advantage settings.

## Hyperparameter Revisions (Manual Debug)
- r1 (baseline issues observed):
  - LR=1e‑6, gens=4, entropy=0.0, β=0.1, squared/ambiguous KL, no advantage clipping, sampled eval.
  - Behavior: KL dominated, frequent identical completions, advantage std≈0, noisy/regressive steps.

- r2 (stability pass):
  - LR=3e‑6, gens=8, entropy=0.01, clamp std=0.05, advantage clip=±2.0, greedy eval, squared KL, β warmup 20 → β=0.02.
  - Behavior: Better stability but KL still large; with adaptation ON, β ran away; with fixed β=0.02, KL% still high.

- r3 (proper KL + lower β):
  - True token‑level KL(p||q), β after warmup=0.02→0.005, adaptation OFF by default; config banner added.
  - Behavior: KL% ~15–60%, net upward trend over 50 steps (e.g., ~0.394→~0.432), still oscillatory.

- r4 (smoother defaults):
  - LR=2e‑6, gens=12, entropy=0.005, β after warmup=0.005, greedy eval, P‑only overfit batch.
  - Behavior (50 steps): stable grads, multiple peaks at 0.44–0.442; modest net gain with oscillations typical of small‑batch PG.

## Key Observations
- KL runaway was the primary blocker initially. Fixed by: true KL, β warmup + small fixed β, adaptation OFF (or clamped).
- Advantage collapse (std≈0) and identical completions killed PG signal. Fixed by: std clamp, advantage clipping, entropy bonus, higher generations.
- Greedy eval is essential for a clear read on learning (removes sampling noise from the metric).
- With these in place, 50‑step runs show stable dynamics and incremental gains but no “breakthrough” — reasonable for a pretrained LM that already captured easy patterns.

## Representative Results (Summaries)
- Pre‑fix (squared KL, β≈0.02–1.0): KL% ≈ 90–100%, grad norms spiking, reward oscillates without clear gains.
- Post‑fix (true KL, β=0.005): KL≈0.055 after warmup, KL% mostly ≤ ~60%, grads stable, reward trends up with periodic dips; peaks ~0.44–0.442.

## Conclusions
- Overfit‑one‑batch sanity check is now “green” in terms of training dynamics: KL controlled, stable grads, non‑collapsed sampling, and upward reward trend.
- 50 steps are insufficient for large breakthroughs on chess; expect clearer gains at 200–500 steps.
- The manual harness now supports confident iteration and small sweeps.

## Next Steps
- Run 500‑step overfit with r4 defaults to probe longer‑horizon improvements.
- If needed, run a compact sweep to lock a smoother preset:
  - LR ∈ {2e‑6, 3e‑6}
  - β_after_warmup ∈ {0.0, 0.0025, 0.005}
  - entropy ∈ {0.0, 0.005}
  - num_generations ∈ {12}
- Success criteria: 20‑step moving average of greedy reward with positive slope, KL% not persistently > ~70%, grad norms stable.

## Operational Notes
- Task‑conditional training in `train.py` now uses a single GRPOTrainer across phases (memory and state continuity improved).
- `--dry_run` in `rookworld-train` validates dataset splits and phase scheduling without heavy deps.
- Manual debug revision tags (rX) printed at start help confirm the exact configuration used in a run.

