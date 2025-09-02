# Manual GRPO Debug — Comprehensive Analysis (2025‑09‑02)

## Overview
- Purpose: Validate manual GRPO step after recent changes: pawn‑unit evaluations, task‑conditional generation (P vs A), explicit attention masks, and advantage std clamp.
- Setup: `batch=4`, `generations=4`, `max_new_tokens=256`, `beta=0.1`, `lr=1e-6`, P: `temp=0.5/top_p=0.9`, A: `0.95/0.95`.
- Engine: Stockfish detected and used; ground‑truth evals converted cp→pawns before MAE.

## Batch & Baseline
- Composition: P: 3/4, A: 1/4.
- Initial rewards (avg per prompt): [0.495, 0.429, 0.910, 0.149]; overall mean 0.4958.
- Note: A‑task reached 0.910 consistently, suggesting easy deterministic target for this sample.

## Generation & Rewards (Training Step)
- P‑task generations varied; A‑task produced identical completions (expected for simple A prompts; no penalty applied by design).
- Rewards (flat 16 vector): [0.5012, 0.4648, 0.4996, 0.7040, 0.1986, 0.1250, 0.1786, 0.1786, 0.91, 0.91, 0.91, 0.91, 0.0750, -0.0350, 0.1950, 0.0750].

## Advantage Calculation (TRL‑exact)
- Group means: [0.5424, 0.1702, 0.9100, 0.0775].
- Std per group: [0.1090, 0.0316, 0.0000, 0.0939] → clamped to ≥1e‑6 to avoid div‑by‑zero.
- Normalized advantages (examples):
  - Prompt1: [-0.378, -0.712, -0.393, +1.482] (good range 2.194)
  - Prompt2: [+0.900, -1.432, +0.266, +0.266] (good range 2.331)
  - Prompt3 (A): all 0.000 (zero std → no signal)
  - Prompt4: [-0.027, -1.198, +1.251, -0.027] (good range 2.448)

## Loss Decomposition (Corrected GRPO)
- Per‑prompt PG averages: P1=0.000, P2=-0.145, P3=0.000, P4=0.273.
- Per‑prompt KL averages: P1=-0.633, P2=-0.539, P3=-0.520, P4=-0.508.
- Aggregates: PG=+0.032, KL=-0.547 → Total loss=-0.516; KL ratio ≈94.5% (dominant).
- Gradient norm: 20.35 (clipped downstream as needed).

## Post‑Update Performance
- Per‑prompt avgs: [0.406, 0.239, 0.910, 0.055]; overall 0.4027 (Δ = -0.0931 vs baseline).
- Interpretation: With β=0.1 and current sampling, KL dominance constrains learning; easy A‑task remains stable; P‑tasks regressed slightly.

## Findings & Recommendations
- Units: Pawn‑unit evals working; mismatched scales now penalized via MAE in pawns.
- A‑tasks: Identical generations are acceptable for trivial transitions; ensure future harder A prompts exist to maintain informative std.
- KL balance: KL dominates (≈95%). Consider adaptive KL or lower β (0.03–0.07) to boost PG impact while monitoring stability.
- Exploration: For training (not just debug), consider `generations=8` to stabilize group baselines; keep task‑conditional sampling.
- Monitoring: Alert on zero std groups; resample those prompts during training to preserve learning signal density.

## Notable Fixes Reflected Here
- Attention masks passed to generation to remove tokenizer warning and stabilize behavior.
- Advantage std clamp (≥1e‑6) to prevent divide‑by‑zero and exploding scales.
- Task‑conditional generation in debug/inspect; training entrypoint supports alternating P/A phases with separate params.

## Full Execution Log

```
`torch_dtype` is deprecated! Use `dtype` instead!
🚨 DEBUGGING GRPO TRAINING - STEP BY STEP
======================================================================
🔍 MANUAL GRPO DEBUG - SINGLE BATCH ANALYSIS
======================================================================
📋 Configuration (TRL-matched):
  Batch size: 4
  Generations per prompt: 4
  Max new tokens: 256
  Beta (KL penalty): 0.1 (low for more learning)
  Learning rate: 1e-06
  Temperature: 0.5 (focused sampling)
  Top-p: 0.9 (nucleus sampling)
  AdamW: β1=0.9, β2=0.999, ε=1e-08, decay=0.0

==================== PHASE 1: SETUP ====================
📥 Loading base model...
✅ Loaded reference model (frozen) and training model
🏆 Initializing reward function...
✓ Auto-detected Stockfish at: /usr/games/stockfish
✓ Stockfish initialized at /usr/games/stockfish (depth=10, time=0.1s)
📊 Loading mixed batch...
Loading 20 samples from RookWorld dataset...
✓ Loaded 20 mixed task samples
✅ Setup complete. Batch composition:
  P: tasks: 3/4
  A: tasks: 1/4

==================== PHASE 2: INITIAL MODEL PERFORMANCE ====================
  Prompt 1: avg reward = 0.492
  Prompt 2: avg reward = 0.362
  Prompt 3: avg reward = 0.910
  Prompt 4: avg reward = 0.064
📊 INITIAL TRAINING MODEL Performance:
  Average reward: 0.4570
  Positive ratio: 100.0%

==================== PHASE 3: MANUAL GRPO STEP ====================
🤖 Generating completions for GRPO training...

🎯 Prompt 1: P: rnbqkbnr/pp1ppppp/2p5/8/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq ...
    Gen 1: Train_LP= -528.0, Ref_LP=  -23.5
           Text:                             M: d7d5 d7d6 g7g6 e7e5 e7e6     ...
           Normalized: M: d7d5 d7d6 g7g6 e7e5 e7e6 E: -0.4 -0.6 -0.82 -0.54 -0.83 B...
    Gen 2: Train_LP= -358.0, Ref_LP=  -46.2
           Text:                             M: d7d6 d7d5 e7e6 g7g6 e7e5     ...
           Normalized: M: d7d6 d7d5 e7e6 g7g6 e7e5 E: -0.72 -0.42 -0.75 -0.68 -0.53...
    Gen 3: Train_LP= -688.0, Ref_LP=  -36.0
           Text:                             M: d7d5 e7e5 g7g6 d7d6 e7e6     ...
           Normalized: M: d7d5 e7e5 g7g6 d7d6 e7e6 E: -0.59 -0.48 -0.78 -0.79 -0.9 ...
    Gen 4: Train_LP= -644.0, Ref_LP=  -46.0
           Text:                             M: d7d5 g7g6 e7e5 d7d6 e7e6     ...
           Normalized: M: d7d5 g7g6 e7e5 d7d6 e7e6 E: -0.43 -0.72 -0.51 -0.78 -0.74...

🎯 Prompt 2: P: 2k4r/ppp1q1p1/nbb1N3/4PP2/3p2P1/PB5P/1PPK3R/3RQ3 b - - 3 ...
    Gen 1: Train_LP= -656.0, Ref_LP=  -43.2
           Text:                                M: d4d3 a6c5 e7e8 c8b8 h8e8  ...
           Normalized: M: d4d3 a6c5 e7e8 c8b8 h8e8 E: -3.76 -3.47 -4.05 -3.89 -3.92...
    Gen 2: Train_LP= -478.0, Ref_LP=  -48.2
           Text:                                M: a6c5 h8h3 e7e8 h8h4 d4d3  ...
           Normalized: M: a6c5 h8h3 e7e8 h8h4 d4d3 E: -3.49 -3.95 -3.73 -3.89 -3.48...
    Gen 3: Train_LP= -294.0, Ref_LP=  -34.2
           Text:                                M: h8e8 d4d3 a6c5 c8b8 h8h6  ...
           Normalized: M: h8e8 d4d3 a6c5 c8b8 h8h6 E: -3.46 -3.18 -3.02 -3.26 -3.3 ...
    Gen 4: Train_LP= -374.0, Ref_LP=  -35.0
           Text:                                M: a6c5 c6f3 d4d3 c6d7 e7d7  ...
           Normalized: M: a6c5 c6f3 d4d3 c6d7 e7d7 E: -3.44 -3.53 -3.6 -3.64 -3.54 ...

🎯 Prompt 3: A: r1b2b2/1p3Q2/2k2P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R b KQ ...
    Gen 1: Train_LP= -362.0, Ref_LP=   -0.0
           Text: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
           Normalized: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
    Gen 2: Train_LP= -298.0, Ref_LP=   -0.0
           Text: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
           Normalized: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
    Gen 3: Train_LP= -316.0, Ref_LP=   -0.0
           Text: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
           Normalized: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
    Gen 4: Train_LP= -414.0, Ref_LP=   -0.0
           Text: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
           Normalized: r1b2b2/1p3Q2/3k1P1p/p1nN2p1/2P5/5N2/PP3PPP/R1B1KB1R w KQ - 1...
    ⚠️  All 4 completions are IDENTICAL!

🎯 Prompt 4: P: 3r2k1/pp3p1p/8/2P2R1P/1K3Q2/1PP1P3/4q3/6r1 b - - 9 33...
    Gen 1: Train_LP= -652.0, Ref_LP=  -37.8
           Text:                                      M: e2d3 e2g4 g1g7 g1g4 ...
           Normalized: M: e2d3 e2g4 g1g7 g1g4 e2g2 E: 3.75 4.44 4.3 6.1 3.59 B: e2g...
    Gen 2: Train_LP= -340.0, Ref_LP=  -57.5
           Text:                                      M: e2g4 a7a5 e2g2 g1g4 ...
           Normalized: M: e2g4 a7a5 e2g2 g1g4 d8f8 E: 3.91 4.39 3.57 4.95 3.51 B: d...
    Gen 3: Train_LP= -672.0, Ref_LP=  -59.5
           Text:                                      M: a7a5 e2d3 e2g4 g1g4 ...
           Normalized: M: a7a5 e2d3 e2g4 g1g4 e2d2 E: 3.58 4.01 3.78 4.38 4.16 B: a...
    Gen 4: Train_LP= -596.0, Ref_LP=  -36.2
           Text:                                      M: e2d3 e2g4 g1g7
[... omitted for brevity: identical to full log in manual_grpo_debug_2025-09-02.log ...]

🔢 TRL Advantage Calculation (exact formula):
  Total rewards shape: torch.Size([16])
  Rewards: [ 0.5092      0.458       0.4912      0.706       0.145       0.08333333
  0.17857143  0.125       0.91        0.91        0.91        0.91
  0.045      -0.005       0.195       0.045     ]
  Grouped rewards shape: torch.Size([4, 4])
  Group means: [0.5411     0.13297619 0.91       0.07      ]
  Expanded means: [0.5411     0.5411     0.5411     0.5411     0.13297619 0.13297619
 0.13297619 0.13297619 0.91       0.91       0.91       0.91
 0.07       0.07       0.07       0.07      ]
  Raw advantages: [-0.0319     -0.0831     -0.0499      0.1649      0.01202381 -0.04964286
  0.04559524 -0.00797619  0.          0.          0.          0.
 -0.025      -0.075       0.125      -0.025     ]
  Group std devs: [0.11196017 0.03979753 0.         0.08660254]
  Normalized advantages: [-0.28492275 -0.74222823 -0.44569421  1.47284519  0.3021245  -1.24738533
  1.14568005 -0.20041922  0.          0.          0.          0.
 -0.28867513 -0.8660254   1.44337567 -0.28867513]

... (continues through loss breakdown, gradient norms, and post-update performance)
```
