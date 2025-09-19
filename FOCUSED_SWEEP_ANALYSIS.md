# Focused Hyperparameter Sweep Design

## Analysis of Previous 48-Run Sweep

### High Performer Patterns (ΔPerf > 0.1, n=16)
- **Learning Rate Range**: 1.02e-07 to 1.37e-06
- **Batch Size**: ALL used batch_size=8 (100% consistency)
- **Generations**: 16, 24, 32 (even distribution among top performers)
- **LR Schedule**: Advanced (56%) slightly preferred over Cosine (44%)
- **Entropy Coefficients**: 0.001, 0.002, 0.005 (all three appeared in best runs)

### Learning Rate Analysis by Range
```
Range                    | Count | Avg ΔPerf | Success Rate
Optimal (1e-7 to 3e-7)  |   19  |  -0.132   |   31.6%
Moderate (3e-7 to 1e-6) |   19  |  -0.151   |   26.3%
High (1e-6 to 2e-6)     |   10  |  -0.189   |   50.0%
```

**Key Insight**: Despite higher success rate in 1e-6 to 2e-6 range, the average performance change is worse, indicating potential overfitting or instability.

## Focused Sweep Parameter Refinements

### 1. Learning Rate (TIGHTENED)
- **Previous**: 1e-7 to 2e-6 (log-uniform, 3 orders of magnitude)
- **Refined**: 8e-8 to 4e-7 (log-uniform, ~0.7 orders of magnitude)
- **Rationale**: Focus on proven optimal range, exclude problematic high LRs

### 2. Batch Size (FIXED)
- **Previous**: [4, 8, 12] (random choice)
- **Refined**: 8 (fixed)
- **Rationale**: 100% of high performers used batch_size=8

### 3. Generations (FOCUSED)
- **Previous**: 16-40 (adaptive to batch size)
- **Refined**: [16, 24, 32] (random choice)
- **Rationale**: Only values that appeared in high performer runs

### 4. LR Schedule (WEIGHTED)
- **Previous**: ["advanced", "cosine"] (50/50 split)
- **Refined**: ["advanced", "cosine"] with 65/35 weighting
- **Rationale**: Advanced schedule appeared in 9/16 high performers vs 7/16 for cosine

### 5. Entropy Coefficient (UNCHANGED)
- **Previous**: [0.001, 0.002, 0.005] (random choice)
- **Refined**: [0.001, 0.002, 0.005] (unchanged)
- **Rationale**: All three values appeared in high performers, good coverage

## Expected Outcomes

### Statistical Power
- **24 runs × 100 steps**: 2400 total training steps vs 2400 in previous sweep
- **Doubled duration per run**: Better signal on training dynamics and convergence
- **Halved parameter variations**: More focused exploration of proven ranges

### Hypothesis
By focusing on parameter ranges that produced the top performers in the 48-run sweep:
1. **Higher success rate**: More runs with positive ΔPerf
2. **Better average performance**: Higher mean ΔPerf across runs
3. **Reduced variance**: Less extreme negative performance outcomes
4. **Better convergence**: 100 steps allows more training dynamics to emerge

### Key Questions to Answer
1. Can we achieve >50% success rate (vs 31.6% in optimal range)?
2. Do longer runs (100 steps) show continued improvement or plateau?
3. Is the learning rate sweet spot narrower than 8e-8 to 4e-7?
4. Does batch_size=8 remain optimal across longer training duration?

## Launch Command
```bash
./scripts/launch_focused_sweep.sh
```

This focused sweep trades breadth for depth, concentrating exploration on the most promising parameter regions identified from systematic evidence.