# Tensor Handling Bug - Empty Rewards Tensor Crash

## Issue Description
Training crashes at step 435/500 with a tensor reshaping error when the rewards tensor becomes empty.

## Error Details
```
20:33:57 | 🔢 TRL Advantage Calculation (exact formula):
20:33:57 |   Total rewards shape: torch.Size([0])
20:33:57 |   Rewards: []
20:33:57 | ❌ Error in manual GRPO analysis: shape '[4, 12]' is invalid for input of size 0
20:33:57 | Traceback (most recent call last):
20:33:57 |   File "/home/jrahn/dev/rookworld-trl/manual_grpo_debug.py", line 1092, in manual_grpo_single_batch
    rewards_grouped = all_rewards_tensor.view(batch_size, num_generations)
20:33:57 | RuntimeError: shape '[4, 12]' is invalid for input of size 0
```

## Context
- **Step**: 435/500 (87% complete)
- **Configuration**: A: task training, batch_size=4, num_generations=12
- **Expected tensor shape**: [4, 12] = 48 rewards
- **Actual tensor shape**: [0] = empty
- **Training was stable** for 435 steps before this crash

## Root Cause Analysis
The rewards tensor (`all_rewards_tensor`) becomes empty at step 435, but the code attempts to reshape it to `[batch_size, num_generations]` = `[4, 12]`. This suggests:

1. **Reward calculation failure**: No rewards were generated for the batch
2. **Tensor accumulation bug**: Rewards tensor wasn't properly populated
3. **Edge case handling**: Missing validation for empty rewards before reshaping

## Location
File: `manual_grpo_debug.py`
Line: 1092
Function: `manual_grpo_single_batch()`
Code: `rewards_grouped = all_rewards_tensor.view(batch_size, num_generations)`

## Reproduction
```bash
uv run python manual_grpo_debug.py \
  --steps 500 \
  --task_type A \
  --lr_schedule advanced \
  --lr_warmup_steps 20 \
  --eval_every 10 \
  --checkpoint_every -1
```

Training will be stable for ~435 steps, then crash with empty rewards tensor.

## Impact
- **High**: Prevents completion of long training runs
- **Training stability**: Excellent until crash (435 successful steps)
- **Data loss**: Training progress lost due to crash near completion

## Suggested Fix
@claude Please review the reward calculation and tensor handling code around line 1092 in `manual_grpo_debug.py`.

Specifically investigate:
1. Why `all_rewards_tensor` becomes empty at step 435
2. Add validation before tensor reshaping operations
3. Handle edge case where no rewards are generated for a batch
4. Ensure proper error handling and recovery

The crash occurs during `rewards_grouped = all_rewards_tensor.view(batch_size, num_generations)` when `all_rewards_tensor` has shape `[0]` instead of the expected `[48]`.

## Priority
**High** - This prevents completion of production training runs and needs to be fixed for reliable long-term training.

## Related Files
- `manual_grpo_debug.py:1092` (crash location)
- `src/rookworld_trl/rewards.py` (reward calculation)
- Training log: `logs/manual_grpo_debug_run-250918-200610.log`