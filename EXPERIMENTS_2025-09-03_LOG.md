# RookWorld TRL — Experiments Log (2025‑09‑03)

## Summary
- Objective: Stabilize overfit‑one‑batch learning and quantify whether larger effective batch is required for steady reward gains.
- Result: Training dynamics remain stable with r4 settings (true KL, low β, modest entropy), showing repeated peaks but modest net gains in 50‑step runs. A 500‑step remote run confirms no KL runaway, but progress is incremental as expected for chess. Local capacity indicates bs16 fits; bs32 OOMs under the current single‑backward scheme.

## Findings Since 2025‑09‑02
- Overfit (r4) 50‑step local runs:
  - β warmup 20 steps → β_after_warmup=0.005
  - True token‑level KL(p||q), entropy=0.005, gens=12
  - Reward shows multiple peaks ≈0.44–0.442 with modest net improvement; KL% mostly 15–60%; grads stable.
- 500‑step remote run (different HW):
  - Start/end: 0.244 → 0.260 (+0.016), with peaks ≈0.306 later in the run.
  - KL≈0.055 post‑warmup; KL% within 15–60% most steps; grads stable.
- Capacity & batch size:
  - Manual debug accumulates all completion graphs until a single backward → peak activation memory scales with batch_size × num_generations × seq_len.
  - Locally: bs16 "barely fits"; bs32 OOMs around sample ~31 (consistent with single‑backward accumulation).

## Conclusions
- Current r4 settings produce healthy training dynamics (KL controlled, stable grads, periodic reward upticks) but do not guarantee monotonic gains over 50 steps — expected for this task and setup.
- Larger effective batch can reduce PG variance and yield a steadier slope, but is not strictly required for learning. It is, however, the most efficient lever for visibly smoother curves.
- Single‑backward accumulation limits batch scaling locally. To explore larger batches, gradient accumulation is the right change in the manual debug code.

## Plan (Sept 3)
- Implement gradient accumulation in `manual_grpo_debug.py`:
  - For each completion, compute `sample_loss = pg + β·kl − c·H` and call `(sample_loss / (batch_size·num_generations)).backward()` immediately.
  - Accumulate grads across all samples; after the last sample, clip gradients, `optimizer.step()`, and `optimizer.zero_grad()`.
  - Continue logging with `.item()` to avoid retaining graphs.
- Keep other r4 parameters steady while introducing GA:
  - LR=2e‑6; entropy=0.005; gens=12; warmup=20; β_after_warmup=0.005; true token‑level KL; adv std clamp=0.05; adv clip=±2.0; greedy post‑step eval; overfit fixed P‑only batch.
- Evaluate whether effective batch matters materially:
  - A/B test 50 steps on the same fixed batch:
    - A: eff_batch=8 (e.g., bs=8, GA=1)
    - B: eff_batch=32 (e.g., bs=8, GA=4)
  - Track: 20‑step moving average of greedy reward (slope), average KL%, GradN stability; compare deltas.
- If B >> A in slope and stability, adopt GA for local iterations and scale further on remote (2× VRAM × 2 GPUs); otherwise, keep eff_batch small and proceed with targeted parameter nudges only.

## Explicitly Not Doing (for now)
- Drastic plan from Issue #4 (deterministic generation, LR=1e‑5, entropy=0, reward simplification). Rationale:
  - Deterministic training generation eliminates within‑group variance (advantages vanish), breaking GRPO.
  - Very high LR risks instability without clear benefit given current stability.
  - Simplified reward may overfit to a proxy target, misrepresenting true objective gains.

## Suggested Commands
- Local A/B (after GA implementation):
  - A: `uv run python manual_grpo_debug.py --overfit_single_batch --steps 50 --batch_size 8`
  - B: `uv run python manual_grpo_debug.py --overfit_single_batch --steps 50 --batch_size 8` (with GA=4 in code)
- Remote long run: `uv run python manual_grpo_debug.py --overfit_single_batch --steps 500 --batch_size 8`

## Risks & Mitigations
- Risk: GA introduces subtle differences in numerical behavior.
  - Mitigation: Scale losses, keep per‑sample backward; compare one‑step results vs. single‑backward path for consistency.
- Risk: Larger eff_batch reduces exploration diversity if entropy too low.
  - Mitigation: Maintain entropy=0.005 initially; optionally anneal late (≥200 steps).

