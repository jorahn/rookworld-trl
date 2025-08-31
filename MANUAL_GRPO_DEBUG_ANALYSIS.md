# Manual GRPO Debug Code Analysis

## Overview

The `manual_grpo_debug.py` script is a detailed implementation of a single GRPO (Group Relative Policy Optimization) training step, designed to debug and understand why the pretrained model's performance degrades during training. This analysis breaks down every component, parameter choice, and calculation to identify potential issues in the GRPO training pipeline.

## ⚠️ Important Technical Corrections

**Note**: This document analyzes a manual GRPO implementation that contains some deviations from TRL's exact GRPO algorithm. The manual implementation demonstrates GRPO principles but may not exactly match TRL training dynamics due to the following key differences:

### Key Technical Differences from TRL GRPO

1. **KL Divergence Calculation**: 
   - **Manual Implementation**: Uses `(tok_logp_pol - tok_logp_ref).mean()` 
   - **TRL's GRPO**: Uses f-divergence surrogate `exp(ref_logp - pol_logp) - (ref_logp - pol_logp) - 1`
   - **Impact**: Different magnitude and gradient properties for KL penalty

2. **Policy Gradient Term**: 
   - **Manual Implementation**: Sequence-level REINFORCE `-advantage * seq_logprob`
   - **TRL's GRPO**: Per-token likelihood ratio approach with token weighting
   - **Impact**: Different variance properties in gradient estimation

3. **Length Normalization**: 
   - **Manual Implementation**: Simple division by sequence length `sum() / Tgen`
   - **TRL's GRPO**: Masked token normalization `sum(loss*mask) / sum(mask)`
   - **Impact**: Different handling of variable-length sequences

4. **Shape Handling Bug**: The manual implementation contains a tensor gather operation that may cause shape mismatches in log probability extraction.

These differences mean the manual implementation serves as a conceptual demonstration of GRPO mechanics rather than an exact TRL replica.

## Phase-by-Phase Analysis

### Phase 1: Setup and Configuration

**Purpose**: Initialize models, tokenizer, and training parameters with deterministic behavior.

#### Model Architecture Choice
- **Model**: `jrahn/RookWorld-LM-124M` 
- **Why**: A domain-specific 124M parameter transformer specifically trained for chess/RookWorld tasks
- **Precision**: `torch.bfloat16` for memory efficiency while maintaining numerical stability
- **Device mapping**: `"auto"` for automatic GPU/CPU allocation

#### Dual Model Strategy
```python
reference_model = AutoModelForCausalLM.from_pretrained(...)  # Frozen baseline
training_model = AutoModelForCausalLM.from_pretrained(...)   # Trainable copy
```

**Why Two Models**: 
- Reference model provides stable baseline for KL divergence calculation
- Training model accumulates updates while reference remains unchanged
- Essential for GRPO's policy gradient with KL regularization

#### Critical Configuration Parameters

| Parameter | Value | Impact of Higher Values | Impact of Lower Values |
|-----------|-------|------------------------|------------------------|
| `batch_size` | 4 | More stable gradients, slower iteration | Faster iteration, noisier gradients |
| `num_generations` | 4 | Better advantage estimation, more compute | Faster training, less stable advantages |
| `max_new_tokens` | 256 | Longer completions, more context | Shorter responses, faster generation |
| `beta` | 0.1 | Stronger KL penalty, less exploration | Weaker regularization, potential overfitting |
| `learning_rate` | 1e-6 | Faster convergence, risk of instability | Slower but more stable learning |
| `temperature` | 0.5 | More focused sampling, less diversity | More diverse but potentially unfocused outputs |

#### Deterministic Setup
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42
```
**Purpose**: Ensures reproducible results across runs for debugging consistency.

### Phase 2: Initial Model Performance Test

**Purpose**: Establish baseline performance before any training updates.

#### Performance Testing Strategy
- **Generations per prompt**: 2 (reduced for speed during testing)
- **Temperature**: 0.8 (higher than training for more diverse evaluation)
- **Evaluation mode**: Model set to `eval()` to eliminate dropout randomness

#### Why This Matters
- Quantifies the starting point for performance comparison
- Identifies if the model already has good performance that shouldn't be degraded
- Provides a concrete metric for measuring training impact

### Phase 3: Generation and Log Probability Calculation

**Purpose**: Generate completions and calculate log probabilities for both training and reference models.

#### Generation Parameters Deep Dive

**Temperature (0.5)**:
- Lower than evaluation (0.8) for more focused training data
- Higher values (0.8-1.0): More creative but potentially off-task completions
- Lower values (0.1-0.3): Very focused but potentially repetitive outputs

**Top-p (0.9)**:
- Nucleus sampling that keeps 90% of probability mass
- Higher values (0.95-1.0): Include more low-probability tokens
- Lower values (0.7-0.85): More conservative token selection

#### Critical Implementation Details

**Eval Mode Generation**:
```python
def generate_eval(model, **kwargs):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**kwargs)
    if was_training:
        model.train()
    return outputs
```
**Why**: Eliminates dropout randomness during generation while preserving training state.

**Log Probability Calculation**:
```python
completion_logits = train_logits[prompt_length-1:-1]
completion_log_probs = F.log_softmax(completion_logits, dim=-1)
token_log_probs = completion_log_probs.gather(1, completion_tokens.unsqueeze(0)).squeeze()
```
**Critical Details**:
- `prompt_length-1:-1`: Aligns logits with target tokens (logit at position i predicts token at i+1)
- Separate calculation for training and reference models
- Sum of token log probabilities gives sequence likelihood

### Phase 4: Reward Calculation

**Purpose**: Score generated completions using the domain-specific reward function.

#### Reward Function Integration
- Uses `create_reward_function()` from the project's reward system
- Evaluates chess move quality, syntax correctness, and task completion
- Handles both "P:" (position analysis) and "A:" (action selection) tasks

#### Error Handling
```python
try:
    scores = reward_fn(completions, prompts=[prompt] * len(completions))
except Exception as e:
    dummy_scores = [-0.5] * num_generations
```
**Why Dummy Scores**: Prevents training from stopping due to reward function errors while providing negative signal.

### Phase 5: TRL Exact Advantage Calculation

**Purpose**: Implement the exact advantage calculation used by TRL (Transformers Reinforcement Learning) library.

#### TRL Advantage Formula Breakdown

**Step 1: Group Rewards**
```python
rewards_grouped = all_rewards_tensor.view(batch_size, num_generations)
```
Reshapes flat reward list into (batch_size, num_generations) matrix.

**Step 2: Calculate Group Baselines**
```python
mean_grouped_rewards = rewards_grouped.mean(dim=1)
```
Computes average reward for each prompt's generations (baseline for that prompt).

**Step 3: Expand Baselines**
```python
mean_grouped_rewards_expanded = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
```
Repeats each baseline to match the original flat structure.

**Step 4: Raw Advantages**
```python
advantages_tensor = all_rewards_tensor - mean_grouped_rewards_expanded
```
Subtracts the prompt-specific baseline from each reward.

**Step 5: Normalization**
```python
std_rewards_safe = std_rewards + 1e-4  # Numerical stability
advantages_normalized = advantages_tensor / std_expanded
```

#### Why This Calculation Method
- **Per-prompt baselines**: Accounts for varying prompt difficulty
- **Standard deviation normalization**: Ensures advantages have consistent scale across prompts
- **Numerical stability**: 1e-4 addition prevents division by zero

#### Parameter Impact Analysis

**Higher num_generations (8 vs 4)**:
- More stable advantage estimates (larger sample for baseline)
- Better exploration of action space
- Increased computational cost

**Different beta values**:
- `beta = 0.01`: Weak KL penalty, more exploration but potential distribution drift
- `beta = 0.1`: Balanced regularization (current choice)
- `beta = 1.0`: Strong KL penalty, conservative updates, slower learning

### Phase 6-8: Corrected GRPO Loss Calculation

**Purpose**: Implement the complete GRPO loss with proper KL divergence calculation including gradients.

#### Key Corrections from Review

**1. KL Divergence Implementation**:
```python
seq_kl = (tok_logp_pol - tok_logp_ref).mean()  # Simplified KL approximation
```
**Note**: This is a simplified approximation. TRL's actual GRPO uses an f-divergence surrogate:
```python
# TRL's approach:
per_token_kl = torch.exp(ref_logp - pol_logp) - (ref_logp - pol_logp) - 1
```
The manual implementation's approach changes both the magnitude and gradients of the KL penalty compared to TRL.

**2. Length Normalization**:
```python
seq_logprob = tok_logp_pol.sum() / Tgen  # Simple length normalization
```
**Note**: This is a simplified approach. TRL normalizes by the number of valid (non-masked) completion tokens:
```python
# TRL's approach:
loss = (loss_tokens * completion_mask).sum() / completion_mask.sum()
```

**3. Proper Sequence Alignment**:
```python
pol_logits = logits[completion_start:completion_end]  # [Tgen, V]
targets = full_tokens.input_ids[0][prompt_length:completion_end+1]
```
**Critical**: Ensures logits at position i predict token at position i+1.

**4. Tensor Shape Handling**:
```python
# Manual implementation has a shape bug in gather operation:
token_log_probs = completion_log_probs.gather(1, completion_tokens.unsqueeze(0)).squeeze()
# Correct form should be:
token_log_probs = completion_log_probs.gather(dim=1, index=completion_tokens.unsqueeze(1)).squeeze(1)
```

#### GRPO Loss Components

**Policy Gradient Term**:
```python
pg_term = -advantage * seq_logprob  # Sequence-level REINFORCE
```
**Note**: This uses standard REINFORCE. TRL's GRPO uses a per-token likelihood ratio approach:
```python
# TRL's approach:
token_weight = exp(per_token_logps - per_token_logps.detach())
per_token_pg = -token_weight * advantages.unsqueeze(1)
```
Both are valid policy gradient estimators but have different variance properties.

**KL Regularization Term**:
```python
kl_term = beta * seq_kl
```
- Prevents policy from deviating too far from reference
- `beta = 0.1` provides moderate regularization

**Total Loss**:
```python
sample_loss = pg_term + kl_term
```

#### Parameter Sensitivity Analysis

**Beta (KL Penalty) Impact**:
- `β = 0.01`: Weak regularization, faster learning, higher risk of mode collapse
- `β = 0.1`: Current choice - balanced exploration vs exploitation
- `β = 1.0`: Strong regularization, slower learning, more conservative updates

**Learning Rate Impact**:
- `lr = 1e-7`: Very conservative, minimal risk but slow progress
- `lr = 1e-6`: Current choice - moderate updates with stability
- `lr = 1e-5`: Aggressive learning, faster convergence but instability risk

### Phase 9: Post-Update Performance Testing

**Purpose**: Measure the immediate impact of a single GRPO training step.

#### Testing Methodology
- Same prompts and evaluation setup as initial test
- Model in eval mode for consistent evaluation
- Multiple generations per prompt for robust scoring

### Phase 10: Comprehensive Analysis

**Purpose**: Analyze the training step's effectiveness and diagnose potential issues.

#### Key Metrics Tracked

**Performance Change**:
- Direct comparison of pre/post training performance
- Positive change indicates successful learning
- Negative change suggests overfitting or poor advantage estimation

**Loss Component Balance**:
```python
kl_ratio = abs(avg_kl_loss) / (abs(avg_pg_loss) + abs(avg_kl_loss) + 1e-8) * 100
```
**Note**: The "10-30% KL contribution" is a heuristic guideline, not a canonical standard. TRL typically uses adaptive KL (tuning β to track a target KL divergence) rather than maintaining a fixed loss ratio.
- Too high: KL dominates, learning is overly conservative
- Too low: Insufficient regularization, potential overfitting

**Gradient Analysis**:
- Total gradient norm indicates update magnitude
- Gradient clipping (max_norm=1.0) prevents explosive updates
- Zero gradients indicate no learning signal

## Critical Design Decisions and Alternatives

### 1. Advantage Normalization Strategy

**Current Choice**: Per-prompt group normalization with standard deviation
**Alternative**: Global batch normalization
**Impact**: Per-prompt normalization handles varying prompt difficulties better but may amplify noise in small batches.

### 2. KL Divergence Implementation

**Current Choice**: Simplified KL approximation `(tok_logp_pol - tok_logp_ref).mean()`
**TRL's Choice**: F-divergence surrogate `exp(ref_logp - pol_logp) - (ref_logp - pol_logp) - 1`
**Impact**: TRL's approach provides better approximation properties for small policy shifts and matches theoretical GRPO formulation.

### 3. Length Normalization

**Current Choice**: Divide log probabilities by sequence length
**Alternative**: Raw log probabilities
**Impact**: Prevents bias toward shorter completions in policy gradient updates.

### 4. Generation Mode Management

**Current Choice**: Force eval mode during generation, restore training mode
**Alternative**: Generate in training mode
**Impact**: Eval mode eliminates dropout randomness for consistent evaluation and comparison.

## Debugging Insights

### Signs of Healthy Training
1. **Diverse advantages**: Range > 0.01, standard deviation > 0.01
2. **Balanced loss components**: KL penalty 10-30% of total loss
3. **Moderate gradient norms**: Not zero, not explosive (clipped at 1.0)
4. **Performance improvement**: Positive change after updates

### Red Flags
1. **Identical completions**: Indicates collapsed exploration
2. **Zero gradients**: No learning signal reaching the model
3. **Extreme KL ratios**: Either overly conservative or unregularized learning
4. **Performance degradation**: Negative performance change suggests overfitting

## Hyperparameter Interaction Effects

### Beta × Learning Rate Interaction
- **High β, High LR**: Conservative but potentially unstable updates
- **High β, Low LR**: Very conservative, slow but stable learning
- **Low β, High LR**: Aggressive learning, high overfitting risk
- **Low β, Low LR**: Moderate exploration with stability (current choice)

### Temperature × Advantage Calculation
- **High temp**: More diverse completions → better advantage estimates
- **Low temp**: Focused completions → cleaner signal but less exploration
- **Current choice (0.5)**: Balanced approach for training data quality

### Batch Size × Advantage Stability
- **Large batches**: More stable advantage estimates, better baselines
- **Small batches**: Noisier advantages but faster iteration
- **Current choice (4)**: Minimal viable batch for detailed debugging

## Expected Outcomes and Validation

### Success Indicators
1. **Performance Improvement**: Post-update performance > initial performance
2. **Proper Loss Decomposition**: Both PG and KL terms contributing meaningfully
3. **Gradient Flow**: Non-zero gradient norms indicating active learning
4. **Advantage Diversity**: Clear separation between good and bad completions

### Failure Modes and Diagnostics
1. **Mode Collapse**: All generations identical → increase temperature or reduce beta
2. **No Learning**: Zero gradients → check loss calculation and model states
3. **Performance Degradation**: Overfitting → reduce learning rate or increase beta
4. **Reward Function Errors**: Dummy scores used → fix reward function implementation

This manual implementation serves as a conceptual demonstration of GRPO mechanics and can be used to understand the algorithm's components, though it differs from TRL's exact implementation.

## TRL-Correct Implementation Snippets

For reference, here are the corrected implementations that would match TRL's GRPO:

### Corrected Per-Token Log Probability Extraction
```python
# For logits [L, V] over prompt+completion; ids [L]
lp = F.log_softmax(logits[:-1], dim=-1)                      # [L-1, V]
tok_lp = lp.gather(dim=1, index=ids[1:].unsqueeze(1)).squeeze(1)  # [L-1]
tok_lp = tok_lp[prompt_len-1:]                               # keep completion part
```

### TRL-Style KL + Loss Calculation
```python
# Per-token KL using f-divergence surrogate
per_token_kl = torch.exp(ref_tok_lp - tok_lp) - (ref_tok_lp - tok_lp) - 1

# Per-token policy gradient with likelihood ratio
per_token_pg = -torch.exp(tok_lp - tok_lp.detach()) * advantage  # broadcast over tokens

# Combined loss with proper masking
loss_tokens = per_token_pg + beta * per_token_kl
loss = (loss_tokens * completion_mask).sum() / completion_mask.sum()
```

### Key Advantages of TRL's Approach
1. **F-divergence KL**: Better approximation for small policy changes
2. **Per-token weighting**: Lower variance gradient estimates
3. **Proper masking**: Correct handling of variable-length sequences
4. **Numerical stability**: Robust to edge cases in tokenization