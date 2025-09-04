# RookWorld TRL — Experiments Log (2025‑09‑04)

## Context & Purpose
- Goal: improve chess best‑move accuracy via RLVR (GRPO) and validate training with an overfit‑one‑batch test that shows a clear upward reward trend.
- Current status (as of Sept 3):
  - Stable training: true token‑level KL, β warmup→small fixed β, entropy regularization, advantage std clamp/clip; no KL runaway.
  - Signal from higher gens: 20‑step bs=8 runs showed a meaningful gain at gens=24 (+0.0479), while a 100‑step bs=8, gens=32 run had a modest net gain (+0.0060) with oscillations.
  - Constraint: do not reduce `max_new_tokens` — reward schema depends on the length budget.

## Step‑Back Analysis (today)
- Primary limiter: variance of the GRPO signal within prompt groups, not regularization or LR.
- Most promising lever: increase generations (32–40) to reduce within‑prompt variance while keeping stochasticity;
  pair with shorter warmup (earlier KL engagement) and slightly lower entropy to damp exploration noise.
- Keep overfit diagnostic simple and informative:
  - P‑only batch for cleaner signal
  - GA=1 (single optimizer step per batch); per‑sample backward for low VRAM; greedy eval for the metric
  - Monitor MA10/MA20 slopes and KL% band

## Run 1 — 300×40 (remote)
- Command:
```
uv run python manual_grpo_debug.py \
  --overfit_single_batch \
  --steps 300 \
  --batch_size 8 \
  --gens 40 \
  --ga 1 \
  --beta_warmup_steps 5 \
  --entropy_coef 0.002 \
  --seed 42
```
- Expectation:
  - Positive MA20 slope through the mid and late windows; net Δ ≥ +0.05 over 300 steps
  - KL engaged after step 5 with small β; entropy low but non‑zero for healthy variance
  - Grad norms stable; no KL dominance post‑warmup
- Full log: logs/manual_grpo_debug_run-250904-061520.log

### Result Summary
- Config: bs=8, gens=40, steps=300, GA=1, warmup=5, entropy=0.002, seed=42
- Reward (greedy, P‑only batch):
  - StartPost=0.2830 → EndPost=0.2910 (Δ +0.0080)
  - MA20: 0.2876 → 0.2953 (Δ +0.0078)
  - PosSteps=141, NegSteps=151; BestPost=0.3380 @ step 627215; WorstPost=0.2200 @ step 627187
- Timing: Avg step ≈15.1s; Last step ≈15.1s

### Read & Next
- MA20 slope is positive with a small net gain over 300 steps — better than earlier 100×32 (which showed MA20 decline). This supports the “more gens + longer horizon” lever to overcome variance.
- Next:
  - Replicate with a second seed to confirm trend stability.
  - Optionally test warmup=10 vs 5 to see if slightly later KL improves the mid‑run slope; keep β small.
  - Consider adaptive β to maintain a KL band if oscillations return in longer runs (500 steps).

## Next (if needed)
- If 300×40 remains oscillatory with weak net gains:
  - Keep gens=40, increase horizon to 500 steps OR
  - Introduce gentle adaptive β to maintain a KL band while preserving small average β
- If 300×40 shows clean improvement: replicate on a second seed and proceed to trainer‑level sweep.
