# RookWorld TRL — Experiments Log (2025‑09‑03)

## Summary
- Objective: Stabilize overfit‑one‑batch learning and quantify whether larger effective batch is required for steady reward gains.
- Result: Training dynamics remain stable with r4 settings (true KL, low β, modest entropy), showing repeated peaks but modest net gains in 50‑step runs. A 500‑step remote run confirms no KL runaway, but progress is incremental as expected for chess. Local capacity indicates bs16 fits; bs32 OOMs under the current single‑backward scheme.

## Findings Since 2025-09-02
- Overfit (r4) 50‑step local runs:
  - β warmup 20 steps → β_after_warmup=0.005
  - True token‑level KL(p||q), entropy=0.005, gens=12
  - Reward shows multiple peaks ≈0.44–0.442 with modest net improvement; KL% mostly 15–60%; grads stable.
- 500‑step remote run (different HW):
  - Start/end: 0.244 → 0.260 (+0.016), with peaks ≈0.306 later in the run.
  - KL≈0.055 post‑warmup; KL% within 15–60% most steps; grads stable.
- Capacity & batch size:
  - Manual debug accumulates all completion graphs until a single backward → peak activation memory scales with batch_size × num_generations × seq_len.
  - Important clarification: the earlier bs=32 OOM was observed BEFORE introducing gradient accumulation; with GA enabled, a later bs=32 probe did NOT OOM (details in Update below).
  - Locally (pre-GA): bs16 "barely fits"; bs32 OOMs around sample ~31 (consistent with single-backward accumulation).

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



## Update (late Sept 3)
- Capacity vs. utilization (WITH GA enabled): Two-step bs=32 run shows low VRAM (~3 GB of 12 GB) and high compute utilization (>90%), but throughput is not the goal right now. We could leverage higher VRAM via chunked grad accumulation to speed iteration, but we will keep complexity low for now.
- Effect of larger batch: Increasing batch size to 32 did not materially change training dynamics (reward slope/stability) yet significantly slows experiment iteration. Decision: continue with lower batch size for local iterations.
- Next focus: Increase num_generations (within-prompt) to reduce variance where GRPO signal lives.
  - Suggested equal-compute probes on the same fixed batch:
    - A1: lower bs, higher gens (e.g., bs=8, gens=16)
    - A2: current baseline (e.g., bs=16, gens=12)
  - Keep r5 params: true KL, warmup 20 → β=0.005, entropy 0.005, LR 2e-6, greedy eval, P-only batch.
- Timing: Phase timings are now logged (r5); use them to monitor where time is spent.

## 2-step GA Comparison (Sept 3)
- Setup: steps=2, overfit_single_batch, batch_size=4 (prompts per microbatch), num_generations=12 (completions per prompt), same fixed P-only batch.
- Naming clarification (manual debug):
  - batch_size: prompts per microbatch; gens: per-prompt completions (group size); ga: number of within-batch chunks to process (manual debug splits flattened samples into ga chunks and steps each chunk).
  - Note: This manual-debug GA is not “true GA” (accumulate then single step). See README “Manual Debug vs. Trainer Hparams (Temporary)”.

- Results (final avg reward, performance delta, step-2 total time):
  - GA=1 (single update after full batch; microbatch_size=48): 0.3960, +0.0018, ~12.26s
  - GA=48 (per-sample micro-update; microbatch_size=1): 0.3962, +0.0020, ~13.61s
  - GA=4 (minibatches; microbatch_size=12): 0.3900, −0.0042, ~12.65s

- Takeaways:
  - Over only 2 steps, behavior is similar across GA settings; GA>1 adds small overhead from more frequent step/zero/clip.
  - For fast local iteration, keep GA=1 by default; increase GA when memory headroom is tight or when probing different update cadences.
  - Groups (gens) remain orthogonal: they affect advantage computation, not the update cadence.

## High-Gens Follow-up (Sept 3, later)
- Runs on the same dev machine with per-sample backward (low peak VRAM), bs=8, GA=1, overfit_single_batch=True.
- 20 steps, gens=16 → Final 0.4108 (+0.0059). Typical step time ≈26–28s. Full log: logs/manual_grpo_debug_run-250903-141055.log
- 20 steps, gens=24 → Final 0.4529 (+0.0479). Typical step time ≈33–34s. Full log: logs/manual_grpo_debug_run-250903-142047.log
- Longer run (latest full log, bs=8, gens=24): Full log: logs/manual_grpo_debug_run-250903-151059.log
  - StartPost=0.4310 → EndPost=0.4010 (Δ −0.0300) over ~30 steps; PosSteps=16, NegSteps=14.
  - MA10: 0.4236 → 0.4101 (Δ −0.0135); BestPost=0.4550@step6; WorstPost=0.3490@step23.
  - Timing: Avg step ≈33.7s; Last step ≈33.3s.
- Interpretation: Higher gens shows promise (clear gain at 20 steps), but over ~30 steps the curve oscillates. Next probes: more steps at gens=24 or bump to gens=32 with a shorter warmup and slightly lower entropy to stabilize later steps.

## 100-step High-Gens (Sept 3, evening)
- Run: bs=8, gens=32, GA=1, steps=100, warmup=10, entropy=0.003. Full log: logs/manual_grpo_debug_run-250903-155627.log
- Result summary:
  - StartPost=0.4310 → EndPost=0.4370 (Δ +0.0060); PosSteps=55, NegSteps=45.
  - MA20: 0.4221 → 0.4121 (Δ −0.0100); indicates oscillations despite small net gain.
  - Timing: Avg step ≈39.2s; Last step ≈38.5s (as expected, slower than gens=24).
- Read: Increasing gens to 32 at 100 steps produced a modest net gain but the moving average slope was slightly negative over the last window, suggesting lingering variance. Next: extend horizon further (200–300 steps) or pair higher gens with earlier KL engagement and slightly lower entropy to smooth late-curve behavior.
 
## True GA Applied + Quick Validation (Sept 3, evening)
- Implementation change:
  - Converted `manual_grpo_debug.py` to true gradient accumulation: accumulate per-sample losses (scaled by 1/(batch_size·gens)) and perform a single clip/step/zero per batch per GRPO step. Removed intra-batch micro-updates.
  - Rationale: Make per-step updates equivalent to a frozen batch objective and avoid optimizer/clipping cadence artifacts.

- Sanity tests (2 steps on P-only fixed batch):
  - A: `bs=2, gens=4, ga=1` → 0.0786 → 0.0786 (Δ +0.0000). One optimizer update per step (confirmed).
  - B: `bs=2, gens=4, ga=4` → Same single-update behavior and identical outcome (as intended with true GA).
  - C: `bs=2, gens=8, ga=1` → 0.0786 → 0.0932 (Δ +0.0146). Slight positive movement with higher gens even over 2 steps.

- Interpretation:
  - The script now reflects true GA semantics. Very short runs with tiny batch/gens show little change, as expected. Increasing gens reduces variance and can nudge rewards upward even at 2 steps.

- Next actions for overfit success:
  - Run 50-step overfit on the same fixed P-only batch with two configs:
    - Baseline: `bs=8, gens=24, warmup=20, entropy=0.005, beta_after=0.005, lr=2e-6`.
    - High-gens: `bs=8, gens=32, warmup=10, entropy=0.003, beta_after=0.0025–0.005, lr=2e-6`.
  - Keep stochastic training generation; monitor 20-step MA of greedy reward, KL%, grad norms, and advantage variance. If slope is flat and stability holds, bump LR to 3e-6.

- Misc:
  - Updated `pyproject.toml` URLs to `https://github.com/jorahn/rookworld-trl`.

## 300-step Remote Run (Sept 3, UTC)
- File: `logs/manual_grpo_debug_run-250903-164312.log` (filename normalized +2h to UTC; remote TZ was −2).
- Setup: overfit_single_batch (fixed P-only batch), steps=300, warmup=10 → β=0.005, true token-level KL(p||q), entropy=0.003, greedy post-step eval.
- Aggregate results:
  - StartPost=0.335 → EndPost=0.349 (Δ +0.0137).
  - Step deltas: pos=149, neg=139, zero=12 (oscillatory but slight positive drift).
  - Averages across steps: Pre=0.34185, Post=0.34189, Δ≈+5.7e−05 (net near-zero due to cancellation across ups/downs).
  - KL: avg 0.0539 (post‑warmup steady ≈0.056); KL% avg ≈44.7% (typical 25–60%, spikes up to ~80%).
  - GradN: avg ≈5.15; Advantages: avg ≈3.17; β avg ≈0.00483.
  - Timing: per‑step total ≈27s (first step ~43s incl. setup and initial eval). Generation ~10.7s; rewards ~3.4–3.7s; loss update ~7.4s; post‑eval ~5.5s.
- Stability & notes:
  - No NaNs, OOMs, or divergence. One early "low std" warning before β engaged; healthy reward variance thereafter.
  - KL remained controlled after warmup; gradients stable; no runaway behavior observed.
- Conclusions (for this run):
  - Training is stable with β=0.005 and true token‑level KL. Overfit‑batch reward shows incremental gains with expected oscillations.
  - Variance appears to limit monotonic improvement more than regularization. Increasing within‑prompt generations (groups) or effective batch via GA should smooth curves.
  - If more smoothing is needed, consider slightly earlier/smaller warmup or β in the 0.005–0.01 range paired with entropy 0.003–0.005; keep true KL.
