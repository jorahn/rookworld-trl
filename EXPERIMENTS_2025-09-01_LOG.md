# RookWorld TRL — Experiments Log (2025‑09‑01)

This entry summarizes early manual GRPO baseline runs prior to the Sept 2 fixes. It captures the specific measurements and observations from the predecessor manual debug analysis that are not included in the 2025‑09‑02 log.

## Summary
- Objective: establish a baseline for the overfit‑one‑batch GRPO test and identify blockers (KL dominance, advantage collapse, identical completions).
- Result: with TRL‑matched settings, KL penalty dominated updates; reward did not improve and sometimes regressed after a step. This guided the Sept 2 fixes (true KL, lower β after warmup, std clamp, entropy, task‑conditional sampling).

## Setup (TRL‑matched baseline)
- Batch size: 4 prompts; Generations per prompt: 4; Max new tokens: 256.
- Learning rate: 1e‑6; Beta (KL penalty): 0.1.
- Sampling: P: temp=0.5/top_p=0.9; A: temp=0.95/top_p=0.95.
- Engine: Stockfish enabled; ground‑truth evals converted cp→pawns before MAE.

## Batch & Baseline
- Composition: P: 3/4, A: 1/4.
- Initial per‑prompt rewards (example run): [0.495, 0.429, 0.910, 0.149]; overall mean ≈0.496.
- Note: A‑task frequently reached ≈0.910 on this sample, indicating an easy deterministic target; useful for sanity but low learning signal within the group.

## Generation & Rewards (Training Step)
- P‑task completions varied; A‑task often produced identical completions (acceptable for trivial A prompts; zero variance implies no PG signal within that group).
- Example flattened rewards (16):
  [0.5012, 0.4648, 0.4996, 0.7040, 0.1986, 0.1250, 0.1786, 0.1786,
   0.9100, 0.9100, 0.9100, 0.9100, 0.0750, -0.0350, 0.1950, 0.0750]

## Advantages (TRL‑exact)
- Group means: [0.5424, 0.1702, 0.9100, 0.0775]
- Std per group: [0.1090, 0.0316, 0.0000, 0.0939] → clamp ≥1e‑6 to avoid div‑by‑zero.
- Normalized advantages (examples):
  - Prompt 1: [−0.378, −0.712, −0.393, +1.482] (good range 2.194)
  - Prompt 2: [+0.900, −1.432, +0.266, +0.266] (good range 2.331)
  - Prompt 3 (A): all 0.000 (zero std → no learning signal)
  - Prompt 4: [−0.027, −1.198, +1.251, −0.027] (good range 2.448)

## Loss Decomposition (baseline)
- Per‑prompt PG averages: P1=0.000, P2=−0.145, P3=0.000, P4=+0.273.
- Per‑prompt KL averages: P1=−0.633, P2=−0.539, P3=−0.520, P4=−0.508.
- Aggregates: PG=+0.032, KL=−0.547 → Total=−0.516; KL ratio ≈94.5% (dominant).
- Gradient norm (example): ≈20.35 (clipped downstream).

## Post‑Update Performance (representative)
- Per‑prompt avgs: [0.406, 0.239, 0.910, 0.055]; overall 0.4027 (Δ −0.0931 vs baseline of 0.4958).
- Interpretation: with β=0.1 and this sampling, KL dominance constrained learning; easy A‑task remained stable; P‑tasks regressed slightly.

## Early Findings & Recommendations (pre‑Sept 2)
- KL balance: KL dominated; lower β or use warmup then small fixed β to allow PG to move the model.
- Variance: increase generations within prompt to stabilize group baselines; add mild entropy.
- Monitoring: alert on zero‑std groups (e.g., identical A completions) and resample/adjust prompts to maintain learning signal density.

## Relation to Sept 2
- The Sept 2 entry documents the implementation of these remedies: true token‑level KL, β warmup → β_after_warmup≈0.005, std clamp and advantage clipping, entropy, task‑conditional generation, greedy eval metric, and raising generations. Those changes produced stable dynamics and modest net gains over 50‑step runs.

