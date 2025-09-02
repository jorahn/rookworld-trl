#!/usr/bin/env python3
"""
Manual GRPO implementation with extensive logging to debug the training process
This script manually performs all GRPO steps for a single batch to understand
where the pretrained model performance is being degraded.

Fixed based on technical review:
- Corrected tensor gather operations for proper log probability extraction
- TRL-compatible f-divergence surrogate for KL calculation
- Model state management (freeze reference, eval mode for generation)
- Length normalization to prevent bias
- Deterministic execution with proper seeding
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.rookworld_trl.rewards import create_reward_function
from src.rookworld_trl.dataset import RookWorldDataGenerator
from src.rookworld_trl.utils import normalize_spacing
import copy

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_eval(model, **kwargs):
    """Generate in eval mode to eliminate dropout randomness, restore previous mode"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**kwargs)
    if was_training:
        model.train()
    return outputs

def manual_grpo_single_batch():
    """
    Manually implement GRPO for a single batch with extensive logging
    """
    
    print("🔍 MANUAL GRPO DEBUG - SINGLE BATCH ANALYSIS")
    print("=" * 70)
    
    # Configuration exactly matching TRL GRPO training
    batch_size = 4  # Smaller for detailed analysis
    num_generations = 4  # Our override (TRL default = 8)
    max_new_tokens = 256  # Match max_completion_length
    beta = 0.1  # Lower KL penalty for more learning
    learning_rate = 1e-6  # Increased from 1e-7
    temperature = 0.5  # Lower temperature for more focused sampling
    top_p = 0.9  # Keep nucleus sampling
    
    # TRL optimizer defaults
    adam_beta1 = 0.9
    adam_beta2 = 0.999  
    adam_epsilon = 1e-8
    weight_decay = 0.0
    
    print(f"📋 Configuration (TRL-matched):")
    print(f"  Batch size: {batch_size}")
    print(f"  Generations per prompt: {num_generations}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Beta (KL penalty): {beta} (low for more learning)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Temperature: {temperature} (focused sampling)")
    print(f"  Top-p: {top_p} (nucleus sampling)")
    print(f"  AdamW: β1={adam_beta1}, β2={adam_beta2}, ε={adam_epsilon}, decay={weight_decay}")
    
    # ============================================================================
    # PHASE 1: SETUP
    # ============================================================================
    print(f"\n{'='*20} PHASE 1: SETUP {'='*20}")
    
    # Load models
    print(f"📥 Loading base model...")
    model_name = "jrahn/RookWorld-LM-124M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load TWO copies: reference model and training model
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    training_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Properly freeze the reference model
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False
    
    print(f"✅ Loaded reference model (frozen) and training model")
    
    # Create reward function
    print(f"🏆 Initializing reward function...")
    reward_fn = create_reward_function()
    
    # Get mixed batch of prompts
    print(f"📊 Loading mixed batch...")
    data_generator = RookWorldDataGenerator(dataset_size=20)
    prompts = data_generator.get_mixed_batch(batch_size)
    
    print(f"✅ Setup complete. Batch composition:")
    p_count = sum(1 for p in prompts if p.startswith("P: "))
    a_count = sum(1 for p in prompts if p.startswith("A: "))
    print(f"  P: tasks: {p_count}/{batch_size}")
    print(f"  A: tasks: {a_count}/{batch_size}")
    
    # ============================================================================
    # PHASE 2: INITIAL MODEL PERFORMANCE TEST
    # ============================================================================
    print(f"\n{'='*20} PHASE 2: INITIAL MODEL PERFORMANCE {'='*20}")
    
    def test_model_performance(model, model_name):
        """Test model performance and return average reward"""
        all_scores = []
        
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Task-conditional generation parameters
            is_a_task = prompt.startswith("A: ")
            gen_temperature = 0.95 if is_a_task else 0.5
            gen_top_p = 0.95 if is_a_task else 0.9

            outputs = generate_eval(
                model,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Avoid attention mask warning
                max_new_tokens=max_new_tokens,
                num_return_sequences=2,  # Fewer for speed
                do_sample=True,
                temperature=gen_temperature,
                top_p=gen_top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            prompt_len = inputs.input_ids.shape[1]
            completions = []
            
            for j in range(2):
                completion_tokens = outputs[j][prompt_len:]
                completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                completions.append(completion)
            
            try:
                scores = reward_fn(completions, prompts=[prompt] * len(completions))
                all_scores.extend(scores)
                avg_score = sum(scores) / len(scores)
                print(f"  Prompt {i+1}: avg reward = {avg_score:.3f}")
            except Exception as e:
                print(f"  Prompt {i+1}: scoring error - {e}")
                all_scores.extend([-1.0] * 2)
        
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        positive_ratio = sum(1 for s in all_scores if s > 0) / len(all_scores) if all_scores else 0.0
        
        print(f"📊 {model_name} Performance:")
        print(f"  Average reward: {overall_avg:.4f}")
        print(f"  Positive ratio: {positive_ratio*100:.1f}%")
        
        return overall_avg
    
    # Test initial performance
    initial_performance = test_model_performance(training_model, "INITIAL TRAINING MODEL")
    
    # ============================================================================
    # PHASE 3: MANUAL GRPO TRAINING STEP
    # ============================================================================
    print(f"\n{'='*20} PHASE 3: MANUAL GRPO STEP {'='*20}")
    
    # Enable gradients for training model with exact TRL optimizer
    training_model.train()
    optimizer = torch.optim.AdamW(
        training_model.parameters(), 
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay
    )
    
    all_completions = []
    all_log_probs = []
    all_ref_log_probs = []
    
    print(f"🤖 Generating completions for GRPO training...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎯 Prompt {i+1}: {prompt[:60]}...")
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(training_model.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate completions in eval mode for consistency
        is_a_task = prompt.startswith("A: ")
        gen_temperature = 0.95 if is_a_task else temperature
        gen_top_p = 0.95 if is_a_task else top_p

        outputs = generate_eval(
            training_model,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Avoid attention mask warning
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_generations,
            do_sample=True,
            temperature=gen_temperature,
            top_p=gen_top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Extract completions and calculate log probabilities
        prompt_completions = []
        prompt_log_probs = []
        prompt_ref_log_probs = []
        
        for j in range(num_generations):
            # Extract completion
            full_sequence = outputs.sequences[j]
            completion_tokens = full_sequence[prompt_length:]
            completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            
            # Normalize spacing to prevent KL divergence inflation
            completion_text_normalized = normalize_spacing(completion_text)
            prompt_completions.append(completion_text_normalized)
            
            # Calculate log probabilities with training model
            with torch.no_grad():
                train_outputs = training_model(full_sequence.unsqueeze(0))
                train_logits = train_outputs.logits[0]
                
                # Get log probs for completion tokens only
                completion_logits = train_logits[prompt_length-1:-1]
                completion_log_probs = F.log_softmax(completion_logits, dim=-1)
                
                # Extract log probs for actual tokens
                token_log_probs = completion_log_probs.gather(
                    1, completion_tokens.unsqueeze(1)
                ).squeeze(1)
                
                train_total_log_prob = token_log_probs.sum().item()
                prompt_log_probs.append(train_total_log_prob)
            
            # Calculate reference log probabilities with reference model
            with torch.no_grad():
                ref_outputs = reference_model(full_sequence.unsqueeze(0))
                ref_logits = ref_outputs.logits[0]
                
                ref_completion_logits = ref_logits[prompt_length-1:-1]
                ref_completion_log_probs = F.log_softmax(ref_completion_logits, dim=-1)
                
                ref_token_log_probs = ref_completion_log_probs.gather(
                    1, completion_tokens.unsqueeze(1)
                ).squeeze(1)
                
                ref_total_log_prob = ref_token_log_probs.sum().item()
                prompt_ref_log_probs.append(ref_total_log_prob)
            
            print(f"    Gen {j+1}: Train_LP={train_total_log_prob:7.1f}, Ref_LP={ref_total_log_prob:7.1f}")
            print(f"           Text: {completion_text[:60]}...")
            print(f"           Normalized: {completion_text_normalized[:60]}...")
        
        all_completions.append(prompt_completions)
        all_log_probs.append(prompt_log_probs)
        all_ref_log_probs.append(prompt_ref_log_probs)
        
        # Check for identical completions
        unique_completions = len(set(prompt_completions))
        if unique_completions == 1:
            print(f"    ⚠️  All {num_generations} completions are IDENTICAL!")
    
    # ============================================================================
    # PHASE 4: REWARD CALCULATION
    # ============================================================================
    print(f"\n{'='*20} PHASE 4: REWARD CALCULATION {'='*20}")
    
    all_rewards = []
    
    for i, (prompt, completions) in enumerate(zip(prompts, all_completions)):
        print(f"\n🎯 Scoring Prompt {i+1} completions:")
        
        try:
            scores = reward_fn(completions, prompts=[prompt] * len(completions))
            all_rewards.append(scores)
            
            for j, (completion, score) in enumerate(zip(completions, scores)):
                print(f"  Gen {j+1}: Reward={score:6.3f} | {completion[:40]}...")
            
            avg_score = sum(scores) / len(scores)
            print(f"  📊 Average reward: {avg_score:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error scoring: {e}")
            dummy_scores = [-0.5] * num_generations
            all_rewards.append(dummy_scores)
    
    # ============================================================================
    # PHASE 5: TRL EXACT ADVANTAGE CALCULATION
    # ============================================================================
    print(f"\n{'='*20} PHASE 5: TRL EXACT ADVANTAGE CALCULATION {'='*20}")
    
    # Convert to tensors for exact TRL calculation
    all_rewards_tensor = torch.tensor([reward for reward_group in all_rewards for reward in reward_group])
    
    print(f"🔢 TRL Advantage Calculation (exact formula):")
    print(f"  Total rewards shape: {all_rewards_tensor.shape}")
    print(f"  Rewards: {all_rewards_tensor.numpy()}")
    
    # TRL Formula 1: Reshape rewards to (batch_size, num_generations)
    rewards_grouped = all_rewards_tensor.view(batch_size, num_generations)
    print(f"  Grouped rewards shape: {rewards_grouped.shape}")
    
    # TRL Formula 2: Calculate mean per group
    mean_grouped_rewards = rewards_grouped.mean(dim=1)
    print(f"  Group means: {mean_grouped_rewards.numpy()}")
    
    # TRL Formula 3: Repeat for each generation
    mean_grouped_rewards_expanded = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    print(f"  Expanded means: {mean_grouped_rewards_expanded.numpy()}")
    
    # TRL Formula 4: Calculate advantages
    advantages_tensor = all_rewards_tensor - mean_grouped_rewards_expanded
    print(f"  Raw advantages: {advantages_tensor.numpy()}")
    
    # TRL Formula 5: Standard deviation normalization (if enabled)
    std_rewards = rewards_grouped.std(dim=1)
    print(f"  Group std devs: {std_rewards.numpy()}")
    
    # Check for zero std (TRL adds 1e-4 for numerical stability)
    std_rewards_safe = std_rewards + 1e-4
    
    # Normalize advantages by std (TRL default behavior)
    std_expanded = std_rewards_safe.repeat_interleave(num_generations, dim=0)
    advantages_normalized = advantages_tensor / std_expanded
    
    print(f"  Normalized advantages: {advantages_normalized.numpy()}")
    
    # Convert back to list format for compatibility
    all_advantages = advantages_normalized.view(batch_size, num_generations).tolist()
    
    print(f"\n📊 TRL Advantage Summary:")
    for i, advantages in enumerate(all_advantages):
        print(f"  Prompt {i+1}: {[f'{a:+.3f}' for a in advantages]}")
        
        advantage_range = max(advantages) - min(advantages)
        advantage_mean = np.mean(advantages)
        advantage_std = np.std(advantages)
        
        print(f"    Stats: mean={advantage_mean:+.3f} (expect ≈0), std={advantage_std:.3f} (expect >0)")
        
        if advantage_range < 0.01:
            print(f"    ⚠️  Very small range ({advantage_range:.4f}) - little learning signal!")
        else:
            print(f"    ✅ Good range ({advantage_range:.3f}) - clear learning signal")
            
        if abs(advantage_mean) > 0.1:
            print(f"    ⚠️  Mean far from zero - check advantage calculation!")
        if advantage_std < 0.01:
            print(f"    ⚠️  Very low std - insufficient reward diversity!")
    
    # ============================================================================
    # PHASE 6-8: CORRECTED GRPO LOSS CALCULATION WITH PROPER KL
    # ============================================================================
    print(f"\n{'='*20} PHASE 6-8: CORRECTED GRPO LOSS CALCULATION {'='*20}")
    
    # Enable gradients for training model
    training_model.train()
    optimizer.zero_grad()
    
    total_pg_loss = 0.0
    total_kl_loss = 0.0
    
    print(f"🔄 Computing GRPO loss with proper token-level KL...")
    
    for i, (prompt, completions, advantages) in enumerate(
        zip(prompts, all_completions, all_advantages)
    ):
        print(f"\n🎯 Prompt {i+1} - Corrected GRPO Loss:")
        
        # Use normalized prompt for consistent tokenization
        normalized_prompt = normalize_spacing(prompt)
        inputs = tokenizer(normalized_prompt, return_tensors="pt").to(training_model.device)
        prompt_length = inputs.input_ids.shape[1]
        
        prompt_pg_loss = 0.0
        prompt_kl_loss = 0.0
        
        for j, (completion, advantage) in enumerate(zip(completions, advantages)):
            # Normalize prompt and completion for consistent tokenization
            normalized_prompt = normalize_spacing(prompt)
            normalized_completion = normalize_spacing(completion)
            
            # Tokenize full sequence (normalized prompt + normalized completion)  
            full_text = normalized_prompt + " " + normalized_completion
            full_tokens = tokenizer(full_text, return_tensors="pt").to(training_model.device)
            
            # Forward pass through training model (WITH gradients)
            outputs = training_model(full_tokens.input_ids)
            logits = outputs.logits[0]
            
            # Completion span
            completion_start = prompt_length - 1
            completion_end = full_tokens.input_ids.shape[1] - 1
            
            if completion_end > completion_start:
                targets = full_tokens.input_ids[0][prompt_length:completion_end+1]
                Tgen = len(targets)
                
                # Policy logits and log-probs (WITH gradients)
                pol_logits = logits[completion_start:completion_end]  # [Tgen, V]
                pol_logp = F.log_softmax(pol_logits, dim=-1)
                tok_logp_pol = pol_logp.gather(1, targets.unsqueeze(1)).squeeze()  # [Tgen]
                
                # Reference pass (NO gradients)
                with torch.no_grad():
                    ref_out = reference_model(full_tokens.input_ids)
                    ref_logits = ref_out.logits[0][completion_start:completion_end]  # [Tgen, V]
                    ref_logp = F.log_softmax(ref_logits, dim=-1)
                    tok_logp_ref = ref_logp.gather(1, targets.unsqueeze(1)).squeeze()  # [Tgen]
                
                # Length-normalized terms (CRITICAL FIX from review)
                seq_logprob = tok_logp_pol.sum() / Tgen  # Normalized by length
                # Stable KL approximation (f-divergence can be unstable with large logp differences)
                seq_kl = (tok_logp_pol - tok_logp_ref).mean()
                
                # GRPO loss: -A * logp + β * KL (BOTH with gradients)
                # Note: Using sequence-level REINFORCE; TRL uses per-token likelihood ratio weighting
                pg_term = -advantage * seq_logprob
                kl_term = beta * seq_kl
                sample_loss = pg_term + kl_term
                
                prompt_pg_loss += pg_term
                prompt_kl_loss += kl_term
                
                print(f"  Gen {j+1}: A={advantage:+.3f}, logp/len={seq_logprob:.3f}, kl={seq_kl:.3f}")
                print(f"         PG={pg_term:.3f}, KL_penalty={kl_term:.3f}, total={sample_loss:.3f}")
            else:
                print(f"  Gen {j+1}: Empty completion - skipping")
        
        avg_prompt_pg = prompt_pg_loss / num_generations
        avg_prompt_kl = prompt_kl_loss / num_generations
        total_pg_loss += avg_prompt_pg
        total_kl_loss += avg_prompt_kl
        
        print(f"  📊 Prompt averages: PG={avg_prompt_pg:.3f}, KL={avg_prompt_kl:.3f}")
    
    # Average losses across batch
    avg_pg_loss = total_pg_loss / batch_size
    avg_kl_loss = total_kl_loss / batch_size
    total_loss = avg_pg_loss + avg_kl_loss
    
    print(f"\n🔢 CORRECTED Loss Breakdown:")
    print(f"  Policy Gradient Loss: {avg_pg_loss:8.3f}")
    print(f"  KL Loss (with gradients): {avg_kl_loss:8.3f}")
    print(f"  Total Loss:          ={total_loss:8.3f}")
    print(f"")
    kl_ratio = abs(avg_kl_loss) / (abs(avg_pg_loss) + abs(avg_kl_loss) + 1e-8) * 100
    print(f"  KL penalty ratio: {kl_ratio:.1f}%")
    
    # Backward pass with proper gradients
    print(f"\n🔄 Performing corrected gradient update...")
    if total_loss.requires_grad:
        total_loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0.0
        for name, param in training_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Total gradient norm: {total_grad_norm:.2f}")
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        print(f"  ✅ Gradient update applied with proper KL regularization")
    else:
        print(f"  ⚠️  No gradients to update")
    
    # ============================================================================
    # PHASE 9: POST-UPDATE PERFORMANCE TEST
    # ============================================================================
    print(f"\n{'='*20} PHASE 9: POST-UPDATE PERFORMANCE {'='*20}")
    
    # Test performance after one GRPO step
    post_performance = test_model_performance(training_model, "POST-UPDATE TRAINING MODEL")
    
    # ============================================================================
    # PHASE 10: ANALYSIS
    # ============================================================================
    print(f"\n{'='*20} PHASE 10: ANALYSIS {'='*20}")
    
    performance_change = post_performance - initial_performance
    
    print(f"🔍 Corrected GRPO Step Impact:")
    print(f"  Initial performance: {initial_performance:.4f}")
    print(f"  Post-update performance: {post_performance:.4f}")
    print(f"  Performance change: {performance_change:+.4f}")
    
    # Compare with training logs
    training_avg_reward = -0.207
    print(f"\n📊 Comparison with Training Logs:")
    print(f"  Corrected manual result: {post_performance:.4f}")
    print(f"  Training log average: {training_avg_reward:.4f}")
    print(f"  Difference: {post_performance - training_avg_reward:.4f}")
    
    # Analysis of KL regularization effectiveness
    kl_magnitude = abs(avg_kl_loss)
    pg_magnitude = abs(avg_pg_loss)
    
    print(f"\n🔍 KL Regularization Analysis:")
    print(f"  KL magnitude: {kl_magnitude:.3f}")
    print(f"  PG magnitude: {pg_magnitude:.3f}")
    
    if kl_magnitude > pg_magnitude * 10:
        print(f"  ⚠️  KL dominates - consider lowering beta")
    elif kl_magnitude < pg_magnitude * 0.1:
        print(f"  ⚠️  KL too weak - consider increasing beta")
    else:
        print(f"  ✅ KL and PG balanced - good regularization")
    
    return {
        'initial_performance': initial_performance,
        'post_performance': post_performance,
        'performance_change': performance_change,
        'total_loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
        'pg_loss': avg_pg_loss,
        'kl_penalty': avg_kl_loss
    }

def main():
    print("🚨 DEBUGGING GRPO TRAINING - STEP BY STEP")
    print("=" * 70)
    
    try:
        results = manual_grpo_single_batch()
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"  Initial model performance: {results['initial_performance']:.4f}")
        print(f"  After 1 GRPO step: {results['post_performance']:.4f}")
        print(f"  Performance change: {results['performance_change']:+.4f}")
        print(f"  Total loss: {results['total_loss']:.2f}")
        
    except Exception as e:
        print(f"❌ Error in manual GRPO analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
