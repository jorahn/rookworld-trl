#!/usr/bin/env python3
"""
Manual GRPO implementation with extensive logging to debug the training process
This script manually performs all GRPO steps for a single batch to understand
where the pretrained model performance is being degraded.

Overfit Defaults Patch (2025-09-02):
- steps=50, overfit_single_batch=True
- learning_rate=3e-6 (was 1e-6)
- num_generations=8 (was 4), max_new_tokens=128 (was 256)
- sampling: P(temp=0.7, top_p=0.95), A(temp=1.0, top_p=0.98)
- prefer P-only prompts for overfit batch
- beta schedule: warmup β=0.0 for 10 steps, then β=0.03 with adaptation toward target_KL≈0.15
- advantage std clamp: min std=0.05 (was 1e-6)
- entropy bonus: add −entropy_coef*H with entropy_coef=0.01

To revert: reset constants to previous values and remove entropy term/P-only batching.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import time
import math
import logging
import os
import re
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.rookworld_trl.rewards import create_reward_function
from src.rookworld_trl.dataset import RookWorldDataGenerator
from src.rookworld_trl.utils import normalize_spacing
import copy

# Set deterministic behavior (seed set in main)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def generate_eval(model, **kwargs):
    """Generate in eval mode to eliminate dropout randomness, restore previous mode"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**kwargs)
    if was_training:
        model.train()
    return outputs

def evaluate_on_eval_set(model, tokenizer, eval_file="opening_dataset_eval.json", max_samples=50):
    """Evaluate model on held-out eval set and return accuracy."""
    import json

    try:
        with open(eval_file) as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"  ⚠️ Eval file {eval_file} not found, skipping evaluation")
        return None

    # Sample subset for speed
    if max_samples and max_samples < len(eval_data):
        import random
        eval_data = random.sample(eval_data, max_samples)

    exact_matches = 0
    by_move = {}

    for sample in eval_data:
        prompt = sample["prompt"]
        expected = sample["expected_fen"]
        move_num = sample.get("move_num", 0)

        if move_num not in by_move:
            by_move[move_num] = {"correct": 0, "total": 0}
        by_move[move_num]["total"] += 1

        # Generate with greedy decoding for consistency
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs.input_ids.shape[1]
        completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        # Extract FEN
        parts = completion.split("+")
        if parts and parts[0].strip():
            generated_fen = parts[0].strip()
            if generated_fen == expected:
                exact_matches += 1
                by_move[move_num]["correct"] += 1

    accuracy = exact_matches / len(eval_data) * 100

    # Report results
    print(f"  📊 Eval accuracy: {exact_matches}/{len(eval_data)} ({accuracy:.1f}%)")

    # Report move 1 specifically if present
    if 1 in by_move:
        move1_acc = by_move[1]["correct"] / by_move[1]["total"] * 100
        print(f"     Move 1: {by_move[1]['correct']}/{by_move[1]['total']} ({move1_acc:.0f}%)")

    return accuracy

def save_checkpoint(model, optimizer, step, checkpoint_dir="checkpoints"):
    """Save model and optimizer checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model
    model.save_pretrained(checkpoint_path)
    print(f"  💾 Saved model checkpoint to {checkpoint_path}")

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

    # Save metadata
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return checkpoint_path

def get_lr_with_schedule(step, total_steps, base_lr, schedule="constant", warmup_steps=0):
    """Calculate learning rate based on schedule."""
    if schedule == "constant":
        return base_lr

    # Warmup phase
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    # Post-warmup scheduling
    if schedule == "cosine":
        # Cosine decay from base_lr to 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == "linear":
        # Linear decay from base_lr to 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * (1 - progress)
    elif schedule == "step":
        # Step decay: halve every 1/3 of training
        if step > total_steps * 2/3:
            return base_lr * 0.25
        elif step > total_steps * 1/3:
            return base_lr * 0.5
        return base_lr

    return base_lr

def manual_grpo_single_batch(
    steps: int = 1,
    beta_adapt: bool = False,
    target_kl: float = 0.5,
    checkpoint_every: int = 0,
    overfit_single_batch: bool = False,
    seed: int = 42,
    batch_size: int = 4,
    num_generations: int = 12,
    grad_accum_steps: int = 1,
    beta_warmup_steps: int = 20,
    entropy_coef: float = 0.005,
    task_type: str = "P",
    checkpoint_dir: str = "checkpoints",
    eval_every: int = 0,
    eval_file: str = "opening_dataset_eval.json",
    early_stop_threshold: float = 95.0,
    lr_schedule: str = "constant",
    lr_warmup_steps: int = 0,
):
    """
    Manually implement GRPO for a single batch with extensive logging
    """
    
    print("🔍 MANUAL GRPO DEBUG - SINGLE BATCH ANALYSIS")
    print("=" * 70)
    
    # Configuration tuned for overfitting a single batch
    max_new_tokens = 128
    base_learning_rate = 2e-6
    learning_rate = base_learning_rate  # Will be adjusted by scheduler
    # Beta schedule
    beta_after_warmup = 0.005
    beta = 0.0
    # Task-conditional sampling
    p_temperature = 0.7
    p_top_p = 0.95
    a_temperature = 1.0
    a_top_p = 0.98
    # Entropy bonus (configurable)
    
    # TRL optimizer defaults
    adam_beta1 = 0.9
    adam_beta2 = 0.999  
    adam_epsilon = 1e-8
    weight_decay = 0.0
    
    print(f"📋 Configuration (overfit defaults, 2025-09-03r5):")
    print(f"  Task type: {task_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  Generations per prompt: {num_generations}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Beta (KL penalty): {beta} (warmup → {beta_after_warmup} after {beta_warmup_steps} steps)")
    print(f"  KL: token-level KL(p||q) over generated tokens; Adv clip=±2.0; Entropy coef={entropy_coef}")
    print(f"  Steps: {steps} (sequential GRPO updates)")
    if checkpoint_every > 0:
        print(f"  Checkpoints: Every {checkpoint_every} steps to {checkpoint_dir}/")
    elif checkpoint_every == -1:
        print(f"  Checkpoints: Only at end of training to {checkpoint_dir}/")
    else:
        print(f"  Checkpoints: Disabled")
    if eval_every > 0:
        print(f"  Evaluation: Every {eval_every} steps on {eval_file}")
        print(f"  Early stopping: At {early_stop_threshold}% eval accuracy")
    print(f"  LR schedule: {lr_schedule} (base={base_learning_rate})")
    # Gradient accumulation / micro-updates per batch
    total_samples = batch_size * num_generations
    if grad_accum_steps < 1:
        print(f"  ⚠️ Invalid --grad_accum_steps={grad_accum_steps}; resetting to 1")
        grad_accum_steps = 1
    if grad_accum_steps > total_samples:
        print(f"  ⚠️ --grad_accum_steps ({grad_accum_steps}) > total_samples ({total_samples}); capping to {total_samples}")
        grad_accum_steps = total_samples
    microbatch_size = math.ceil(total_samples / grad_accum_steps)
    print(f"  GA: true accumulation; one optimizer step per batch; effective_batch={total_samples}")
    if grad_accum_steps != 1:
        print(f"  Note: --grad_accum_steps={grad_accum_steps} is ignored for stepping (kept for future memory chunking)")
    if beta_adapt:
        print(f"  Beta adaptation: ON (target_KL≈{target_kl})")
    if overfit_single_batch:
        print(f"  Overfit mode: ON (reuse the same batch each step)")
    print(f"  Seed: {seed}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Sampling (task-conditional): P(temp={p_temperature}, top_p={p_top_p}) | A(temp={a_temperature}, top_p={a_top_p})")
    print(f"  AdamW: β1={adam_beta1}, β2={adam_beta2}, ε={adam_epsilon}, decay={weight_decay}")
    
    # ============================================================================
    # PHASE 1: SETUP
    # ============================================================================
    print(f"\n{'='*20} PHASE 1: SETUP {'='*20}")
    _step_t0 = time.time()
    _t_setup0 = time.time()
    
    # Load models
    print(f"📥 Loading base model...")
    model_name = "jrahn/RookWorld-LM-124M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load TWO copies: reference model and training model
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    training_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16,
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
    
    # Load dataset samples once (already shuffled in generator)
    print(f"📊 Loading batch for task type: {task_type}...")

    # Check if we should use opening dataset for A: tasks
    import os
    import json
    if task_type == "A" and os.path.exists("opening_dataset_train.json"):
        print(f"📂 Loading opening dataset for A: task training...")
        with open("opening_dataset_train.json") as f:
            opening_data = json.load(f)
        ordered_prompts = [sample["prompt"] for sample in opening_data]
        print(f"🎯 Loaded {len(ordered_prompts)} opening position prompts")
        # Focus on move 1 positions for initial training
        move_1_prompts = [sample["prompt"] for sample in opening_data if sample["move_num"] == 1]
        print(f"  → {len(move_1_prompts)} from starting position (move 1)")
    else:
        # Default RookWorld dataset loading
        dg_size = max(100, batch_size * 5)  # Increased size for better task selection
        data_generator = RookWorldDataGenerator(dataset_size=dg_size, seed=seed)

        # Filter prompts based on task type
        if task_type == "P":
            ordered_prompts = [prompt for t, prompt, _, _ in data_generator.samples if t == 'P']
            print(f"🧪 P-only mode: using {len(ordered_prompts)} P: task prompts")
        elif task_type == "A":
            ordered_prompts = [prompt for t, prompt, _, _ in data_generator.samples if t == 'A']
            print(f"🧪 A-only mode: using {len(ordered_prompts)} A: task prompts")
        else:  # mixed
            ordered_prompts = [prompt for _, prompt, _, _ in data_generator.samples]
            print(f"🧪 Mixed mode: using all {len(ordered_prompts)} prompts")

        if len(ordered_prompts) < batch_size:
            print(f"⚠️ Warning: Only {len(ordered_prompts)} prompts available for task type {task_type}, need {batch_size}")
            # Fallback to all prompts if not enough task-specific ones
            ordered_prompts = [prompt for _, prompt, _, _ in data_generator.samples]

    total_prompts = len(ordered_prompts)
    
    # Step 1 batch (fixed): first slice of ordered prompts
    start_idx = 0
    end_idx = start_idx + batch_size
    if end_idx <= total_prompts:
        prompts = ordered_prompts[start_idx:end_idx]
        wrap = False
    else:
        wrap = True
        overflow = end_idx % total_prompts
        prompts = ordered_prompts[start_idx:] + ordered_prompts[:overflow]
        prompts = prompts[:batch_size]
    print(f"🧾 Using dataset prompts [{start_idx}:{end_idx}) of {total_prompts} (wrap-around: {'yes' if wrap else 'no'})")
    
    print(f"✅ Setup complete. Batch composition:")
    _t_setup = time.time() - _t_setup0
    print(f"[timing] setup: {_t_setup:.2f}s")
    p_count = sum(1 for p in prompts if p.startswith("P: "))
    a_count = sum(1 for p in prompts if p.startswith("A: "))
    print(f"  P: tasks: {p_count}/{batch_size}")
    print(f"  A: tasks: {a_count}/{batch_size}")
    
    # ============================================================================
    # PHASE 2: INITIAL MODEL PERFORMANCE TEST
    # ============================================================================
    print(f"\n{'='*20} PHASE 2: INITIAL MODEL PERFORMANCE {'='*20}")
    _t_init_eval0 = time.time()
    
    def test_model_performance(model, model_name):
        """Test model performance and return average reward (deterministic, greedy)"""
        all_scores = []
        a_task_metrics = {"exact_match": 0, "total_distance": 0, "valid_fens": 0, "count": 0}

        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Deterministic greedy evaluation for stability
            outputs = generate_eval(
                model,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = inputs.input_ids.shape[1]
            completions = []
            completion_tokens = outputs[0][prompt_len:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(completion)

            # Track A: task specific metrics
            if prompt.startswith("A: "):
                a_task_metrics["count"] += 1
                # Extract expected FEN and compare
                try:
                    import chess
                    import Levenshtein
                    # Parse prompt to get expected transition
                    env_match = re.search(r'A:\s*([^+]+)\+([^+]+)\+', prompt)
                    if env_match:
                        fen = env_match.group(1).strip()
                        move_uci = env_match.group(2).strip()
                        board = chess.Board(fen)
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            expected_fen = board.fen()
                            # Extract predicted FEN from response
                            parts = completion.split("+")
                            if parts and parts[0].strip():
                                predicted_fen = parts[0].strip()

                                # Compare full FENs for exact match
                                if predicted_fen == expected_fen:
                                    a_task_metrics["exact_match"] += 1

                                # Calculate Levenshtein distance on full FEN
                                distance = Levenshtein.distance(expected_fen, predicted_fen)
                                a_task_metrics["total_distance"] += distance

                                # Check if valid FEN
                                try:
                                    chess.Board(predicted_fen)
                                    a_task_metrics["valid_fens"] += 1
                                except:
                                    pass
                except Exception as e:
                    # Debug: print exception to understand what's failing
                    print(f"    Metric calculation error: {e}")

            try:
                scores = reward_fn(completions, prompts=[prompt] * len(completions))
                all_scores.extend(scores)
                avg_score = sum(scores) / len(scores)
                print(f"  Prompt {i+1}: avg reward = {avg_score:.3f}")
            except Exception as e:
                print(f"  Prompt {i+1}: scoring error - {e}")
                all_scores.extend([-1.0])

        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        positive_ratio = sum(1 for s in all_scores if s > 0) / len(all_scores) if all_scores else 0.0

        print(f"📊 {model_name} Performance:")
        print(f"  Average reward: {overall_avg:.4f}")
        print(f"  Positive ratio: {positive_ratio*100:.1f}%")

        # Report A: task metrics if applicable
        if a_task_metrics["count"] > 0:
            exact_rate = a_task_metrics["exact_match"] / a_task_metrics["count"]
            valid_rate = a_task_metrics["valid_fens"] / a_task_metrics["count"]
            avg_distance = a_task_metrics["total_distance"] / a_task_metrics["count"]
            print(f"  A: Task Metrics:")
            print(f"    - Exact FEN match: {exact_rate*100:.1f}%")
            print(f"    - Valid FEN rate: {valid_rate*100:.1f}%")
            print(f"    - Avg Levenshtein distance: {avg_distance:.2f}")

        return overall_avg
    
    # Test initial performance
    initial_performance = test_model_performance(training_model, "INITIAL TRAINING MODEL")
    _t_init_eval = time.time() - _t_init_eval0
    print(f"[timing] initial_eval: {_t_init_eval:.2f}s")
    
    # Collect per-step metrics for overview
    step_metrics = []
    
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
    _t_gen0 = time.time()
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎯 Prompt {i+1}: {prompt[:60]}...")
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(training_model.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate completions in eval mode for consistency
        is_a_task = prompt.startswith("A: ")
        gen_temperature = a_temperature if is_a_task else p_temperature
        gen_top_p = a_top_p if is_a_task else p_top_p

        # Reseed RNG deterministically per generate block
        torch.manual_seed(seed + 2000 + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 2000 + i)

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
            output_scores=True,
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
    _t_generation = time.time() - _t_gen0
    print(f"[timing] generation: {_t_generation:.2f}s")
    
    # ============================================================================
    # PHASE 4: REWARD CALCULATION
    # ============================================================================
    print(f"\n{'='*20} PHASE 4: REWARD CALCULATION {'='*20}")
    _t_rewards0 = time.time()
    
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
    _t_rewards = time.time() - _t_rewards0
    print(f"[timing] rewards: {_t_rewards:.2f}s")
    
    # ============================================================================
    # PHASE 5: TRL EXACT ADVANTAGE CALCULATION
    # ============================================================================
    print(f"\n{'='*20} PHASE 5: TRL EXACT ADVANTAGE CALCULATION {'='*20}")
    _t_adv0 = time.time()
    
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
    
    # Numerical safety: clamp std to avoid divide-by-zero or extreme scaling
    std_rewards_safe = torch.clamp(std_rewards, min=0.05)
    
    # Normalize advantages by std (TRL default behavior)
    std_expanded = std_rewards_safe.repeat_interleave(num_generations, dim=0)
    advantages_normalized = advantages_tensor / std_expanded
    advantages_normalized = advantages_normalized.clamp(-2.0, 2.0)
    
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

    # Aggregate advantage metric for table: mean per-prompt advantage range
    adv_range_values = []
    for advantages in all_advantages:
        adv_range_values.append(max(advantages) - min(advantages))
    step_adv_metric = float(np.mean(adv_range_values)) if adv_range_values else 0.0
    _t_adv = time.time() - _t_adv0
    print(f"[timing] advantages: {_t_adv:.2f}s")
    
    # ============================================================================
    # PHASE 6-8: CORRECTED GRPO LOSS CALCULATION WITH PROPER KL (grad accumulation)
    # ============================================================================
    print(f"\n{'='*20} PHASE 6-8: CORRECTED GRPO LOSS CALCULATION {'='*20}")
    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    _t_loss0 = time.time()
    
    total_pg_loss = 0.0  # floats for logging
    total_kl_loss = 0.0
    scale = 1.0 / (batch_size * num_generations)
    print(f"🔄 Computing GRPO loss with proper token-level KL and accumulating grads...")
    processed = 0
    steps_done = 0
    for i, (prompt, completions, advantages) in enumerate(zip(prompts, all_completions, all_advantages)):
        print(f"\n🎯 Prompt {i+1} - Corrected GRPO Loss:")
        normalized_prompt = normalize_spacing(prompt)
        inputs = tokenizer(normalized_prompt, return_tensors="pt").to(training_model.device)
        prompt_length = inputs.input_ids.shape[1]
        prompt_pg_loss = 0.0
        prompt_kl_loss = 0.0
        for j, (completion, advantage) in enumerate(zip(completions, advantages)):
            normalized_completion = normalize_spacing(completion)
            full_text = normalized_prompt + " " + normalized_completion
            full_tokens = tokenizer(full_text, return_tensors="pt").to(training_model.device)
            outputs = training_model(full_tokens.input_ids)
            logits = outputs.logits[0]
            completion_start = prompt_length - 1
            completion_end = full_tokens.input_ids.shape[1] - 1
            if completion_end > completion_start:
                targets = full_tokens.input_ids[0][prompt_length:completion_end+1]
                Tgen = len(targets)
                pol_logits = logits[completion_start:completion_end]
                pol_logp = F.log_softmax(pol_logits, dim=-1)
                tok_logp_pol = pol_logp.gather(1, targets.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    ref_out = reference_model(full_tokens.input_ids)
                    ref_logits = ref_out.logits[0][completion_start:completion_end]
                    ref_logp = F.log_softmax(ref_logits, dim=-1)
                    tok_logp_ref = ref_logp.gather(1, targets.unsqueeze(1)).squeeze()
                seq_logprob = tok_logp_pol.sum() / Tgen
                probs = torch.exp(pol_logp)
                tok_kl = (probs * (pol_logp - ref_logp)).sum(dim=-1)
                seq_kl = tok_kl.mean()
                tok_entropy = -(probs * pol_logp).sum(dim=-1).mean()
                pg_term = -advantage * seq_logprob
                kl_term = beta * seq_kl
                ent_term = -entropy_coef * tok_entropy
                sample_loss = (pg_term + kl_term + ent_term) * scale
                sample_loss.backward()
                pg_v = float(pg_term.detach().item())
                kl_v = float(kl_term.detach().item())
                ent_v = float(ent_term.detach().item())
                prompt_pg_loss += pg_v
                prompt_kl_loss += kl_v
                print(f"  Gen {j+1}: A={advantage:+.3f}, logp/len={float(seq_logprob.detach().item()):.3f}, kl={float(seq_kl.detach().item()):.3f}, H={float(tok_entropy.detach().item()):.3f}")
                print(f"         PG={pg_v:.3f}, KL_penalty={kl_v:.3f}, EntReg={ent_v:.3f}, total={(pg_v+kl_v+ent_v):.3f}")
                processed += 1
                # True GA: defer optimizer step until all samples processed
            else:
                print(f"  Gen {j+1}: Empty completion - skipping")
        avg_prompt_pg = prompt_pg_loss / num_generations
        avg_prompt_kl = prompt_kl_loss / num_generations
        total_pg_loss += avg_prompt_pg
        total_kl_loss += avg_prompt_kl
        print(f"  📊 Prompt averages: PG={avg_prompt_pg:.3f}, KL={avg_prompt_kl:.3f}")
    
    # Average losses across batch (floats)
    avg_pg_loss_val = total_pg_loss / batch_size
    avg_kl_loss_val = total_kl_loss / batch_size
    total_loss_val = avg_pg_loss_val + avg_kl_loss_val
    
    print(f"\n🔢 CORRECTED Loss Breakdown (accumulated):")
    print(f"  Policy Gradient Loss: {avg_pg_loss_val:8.3f}")
    print(f"  KL Loss (with gradients): {avg_kl_loss_val:8.3f}")
    print(f"  Total Loss:          ={total_loss_val:8.3f}")
    kl_ratio = abs(avg_kl_loss_val) / (abs(avg_pg_loss_val) + abs(avg_kl_loss_val) + 1e-8) * 100
    print(f"  KL penalty ratio: {kl_ratio:.1f}%")
    
    # Update learning rate for step 1
    current_lr = get_lr_with_schedule(1, steps, base_learning_rate, lr_schedule, lr_warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    if current_lr != base_learning_rate:
        print(f"  📈 LR schedule: {lr_schedule} → lr={current_lr:.2e} (base={base_learning_rate:.2e})")

    # If no micro-updates were performed (e.g., empty completions), ensure one update
    if steps_done == 0:
        print(f"\n🔄 Performing gradient update...")
        total_grad_norm = 0.0
        for name, param in training_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Total gradient norm: {total_grad_norm:.2f}")
        torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    _t_loss = time.time() - _t_loss0
    print(f"[timing] loss_update: {_t_loss:.2f}s")
    print(f"  ✅ Gradient update applied with proper KL regularization")
    
    # ============================================================================
    # PHASE 9: POST-UPDATE PERFORMANCE TEST
    # ============================================================================
    print(f"\n{'='*20} PHASE 9: POST-UPDATE PERFORMANCE {'='*20}")
    _t_post_eval0 = time.time()
    # Test performance after one GRPO step
    post_performance = test_model_performance(training_model, "POST-UPDATE TRAINING MODEL (step 1)")
    _t_post_eval = time.time() - _t_post_eval0
    print(f"[timing] post_eval: {_t_post_eval:.2f}s")
    
    # ============================================================================
    # PHASE 10: ANALYSIS
    # ============================================================================
    print(f"\n{'='*20} PHASE 10: ANALYSIS {'='*20}")
    _t_analysis0 = time.time()
    
    performance_change = post_performance - initial_performance
    
    _t_analysis = time.time() - _t_analysis0
    _step_secs = time.time() - _step_t0
    print(f"\n⏱️ Timing (step 1): setup={_t_setup:.2f}s, initial_eval={_t_init_eval:.2f}s, generation={_t_generation:.2f}s, rewards={_t_rewards:.2f}s, advantages={_t_adv:.2f}s, loss_update={_t_loss:.2f}s, post_eval={_t_post_eval:.2f}s, analysis={_t_analysis:.2f}s, total={_step_secs:.2f}s")
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
    kl_magnitude = abs(avg_kl_loss_val)
    pg_magnitude = abs(avg_pg_loss_val)
    
    print(f"\n🔍 KL Regularization Analysis:")
    print(f"  KL magnitude: {kl_magnitude:.3f}")
    print(f"  PG magnitude: {pg_magnitude:.3f}")
    
    if kl_magnitude > pg_magnitude * 10:
        print(f"  ⚠️  KL dominates - consider lowering beta")
    elif kl_magnitude < pg_magnitude * 0.1:
        print(f"  ⚠️  KL too weak - consider increasing beta")
    else:
        print(f"  ✅ KL and PG balanced - good regularization")
    
    # Evaluation for step 1
    eval_accuracy = None
    if eval_every > 0 and eval_every == 1:
        print(f"\n📊 Evaluating on held-out set...")
        eval_accuracy = evaluate_on_eval_set(training_model, tokenizer, eval_file, max_samples=50)

        if eval_accuracy is not None and eval_accuracy >= early_stop_threshold:
            print(f"\n✅ Early stopping: Eval accuracy {eval_accuracy:.1f}% >= {early_stop_threshold}%")
            # Save final checkpoint on early stop
            if checkpoint_every != 0:
                checkpoint_path = save_checkpoint(training_model, optimizer, 1, checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"  Final checkpoint saved at {checkpoint_path}")
            # Set flag to skip further steps
            steps = 1

    # Only checkpoint if checkpoint_every > 0 and step 1 is a multiple
    if checkpoint_every > 0 and (1 % checkpoint_every == 0):
        checkpoint_path = save_checkpoint(training_model, optimizer, 1, checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_path)

    # Record metrics for step 1
    # Safely convert tensors to scalars for metrics
    _pg_val = float(avg_pg_loss_val)
    _kl_val = float(avg_kl_loss_val)
    _kl_ratio_val = float(kl_ratio)

    step_metrics.append({
        'step': 1,
        'pre': float(initial_performance),
        'post': float(post_performance),
        'delta': float(performance_change),
        'pg': _pg_val,
        'kl': _kl_val,
        'kl_ratio': _kl_ratio_val,
        'adv': step_adv_metric,
        'grad_norm': float(total_grad_norm) if 'total_grad_norm' in locals() else None,
        'beta': float(beta),
        'sec': float(_step_secs),
    })

    # Prepare for optional additional steps
    last_post_performance = post_performance

    # Track if we early stopped
    early_stopped = False

    # Sequential extra steps (flat style, explicit)
    if steps > 1:
        for step_idx in range(2, steps + 1):
            _step_t0 = time.time()
            print(f"\n{'='*70}")
            print(f"🧭 SEQUENTIAL STEP {step_idx}/{steps}")
            print(f"{'='*70}")

            # Select batch for this step
            if overfit_single_batch:
                print(f"🧾 Overfit mode: reusing initial batch (no dataset slice advance)")
            else:
                start_idx = (step_idx - 1) * batch_size
                end_idx = start_idx + batch_size
                if end_idx <= total_prompts:
                    prompts = ordered_prompts[start_idx:end_idx]
                    wrap = False
                else:
                    wrap = True
                    overflow = end_idx % total_prompts
                    prompts = ordered_prompts[start_idx:] + ordered_prompts[:overflow]
                    prompts = prompts[:batch_size]
                print(f"🧾 Using dataset prompts [{start_idx}:{end_idx}) of {total_prompts} (wrap-around: {'yes' if wrap else 'no'})")

            # ===== Generation (same as PHASE 3) =====
            all_completions = []
            all_log_probs = []
            all_ref_log_probs = []
            print(f"\n🤖 Generating completions for GRPO training (step {step_idx})...")
            _t_gen0 = time.time()
            for i, prompt in enumerate(prompts):
                print(f"\n🎯 Prompt {i+1}: {prompt[:60]}...")
                inputs = tokenizer(prompt, return_tensors='pt').to(training_model.device)
                prompt_length = inputs.input_ids.shape[1]
                is_a_task = prompt.startswith("A: ")
                gen_temperature = a_temperature if is_a_task else p_temperature
                gen_top_p = a_top_p if is_a_task else p_top_p
                # Reseed RNG deterministically per generate block in sequential steps
                torch.manual_seed(seed + 3000 + (step_idx * 100) + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + 3000 + (step_idx * 100) + i)

                outputs = generate_eval(
                    training_model,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_generations,
                    do_sample=True,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                prompt_completions = []
                prompt_log_probs = []
                prompt_ref_log_probs = []
                for j in range(num_generations):
                    full_sequence = outputs.sequences[j]
                    completion_tokens = full_sequence[prompt_length:]
                    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                    completion_text_normalized = normalize_spacing(completion_text)
                    prompt_completions.append(completion_text_normalized)
                    with torch.no_grad():
                        train_outputs = training_model(full_sequence.unsqueeze(0))
                        train_logits = train_outputs.logits[0]
                        completion_logits = train_logits[prompt_length-1:-1]
                        completion_log_probs = F.log_softmax(completion_logits, dim=-1)
                        token_log_probs = completion_log_probs.gather(1, completion_tokens.unsqueeze(1)).squeeze(1)
                        train_total_log_prob = token_log_probs.sum().item()
                        prompt_log_probs.append(train_total_log_prob)
                    with torch.no_grad():
                        ref_outputs = reference_model(full_sequence.unsqueeze(0))
                        ref_logits = ref_outputs.logits[0]
                        ref_completion_logits = ref_logits[prompt_length-1:-1]
                        ref_completion_log_probs = F.log_softmax(ref_completion_logits, dim=-1)
                        ref_token_log_probs = ref_completion_log_probs.gather(1, completion_tokens.unsqueeze(1)).squeeze(1)
                        ref_total_log_prob = ref_token_log_probs.sum().item()
                        prompt_ref_log_probs.append(ref_total_log_prob)
                    print(f"    Gen {j+1}: Train_LP={train_total_log_prob:7.1f}, Ref_LP={ref_total_log_prob:7.1f}")
                    print(f"           Text: {completion_text[:60]}...")
                    print(f"           Normalized: {completion_text_normalized[:60]}...")
                all_completions.append(prompt_completions)
                all_log_probs.append(prompt_log_probs)
                all_ref_log_probs.append(prompt_ref_log_probs)
            _t_generation = time.time() - _t_gen0
            print(f"[timing] generation: {_t_generation:.2f}s")

            # ===== Rewards (same as PHASE 4) =====
            print(f"\n{'='*20} PHASE 4: REWARD CALCULATION (step {step_idx}) {'='*20}")
            _t_rewards0 = time.time()
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
                    all_rewards.append([-0.5] * num_generations)
            _t_rewards = time.time() - _t_rewards0
            print(f"[timing] rewards: {_t_rewards:.2f}s")

            # ===== Advantages (same as PHASE 5) =====
            print(f"\n{'='*20} PHASE 5: TRL EXACT ADVANTAGE CALCULATION (step {step_idx}) {'='*20}")
            _t_adv0 = time.time()
            all_rewards_tensor = torch.tensor([r for g in all_rewards for r in g])
            print(f"🔢 TRL Advantage Calculation (exact formula):")
            print(f"  Total rewards shape: {all_rewards_tensor.shape}")
            print(f"  Rewards: {all_rewards_tensor.numpy()}")
            rewards_grouped = all_rewards_tensor.view(batch_size, num_generations)
            mean_grouped_rewards = rewards_grouped.mean(dim=1)
            mean_grouped_rewards_expanded = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            advantages_tensor = all_rewards_tensor - mean_grouped_rewards_expanded
            std_rewards = rewards_grouped.std(dim=1)
            std_rewards_safe = torch.clamp(std_rewards, min=1e-6)
            std_expanded = std_rewards_safe.repeat_interleave(num_generations, dim=0)
            advantages_normalized = advantages_tensor / std_expanded
            advantages_normalized = advantages_normalized.clamp(-2.0, 2.0)
            all_advantages = advantages_normalized.view(batch_size, num_generations).tolist()

            # Aggregate advantage metric for table: mean per-prompt advantage range
            adv_range_values = []
            for advantages in all_advantages:
                adv_range_values.append(max(advantages) - min(advantages))
            step_adv_metric = float(np.mean(adv_range_values)) if adv_range_values else 0.0

            _t_adv = time.time() - _t_adv0
            print(f"[timing] advantages: {_t_adv:.2f}s")
            # ===== Loss & update (same as PHASE 6-8) =====
            print(f"\n{'='*20} PHASE 6-8: CORRECTED GRPO LOSS CALCULATION (step {step_idx}) {'='*20}")
            _t_loss0 = time.time()
            training_model.train()
            optimizer.zero_grad(set_to_none=True)
            total_pg_loss = 0.0
            total_kl_loss = 0.0
            scale = 1.0 / (batch_size * num_generations)
            processed = 0
            steps_done = 0
            print(f"🔄 Computing GRPO loss with proper token-level KL and accumulating grads...")
            for i, (prompt, completions, advantages) in enumerate(zip(prompts, all_completions, all_advantages)):
                print(f"\n🎯 Prompt {i+1} - Corrected GRPO Loss:")
                normalized_prompt = normalize_spacing(prompt)
                inputs = tokenizer(normalized_prompt, return_tensors='pt').to(training_model.device)
                prompt_length = inputs.input_ids.shape[1]
                prompt_pg_loss = 0.0
                prompt_kl_loss = 0.0
                for j, (completion, advantage) in enumerate(zip(completions, advantages)):
                    normalized_completion = normalize_spacing(completion)
                    normalized_prompt = normalize_spacing(prompt)
                    full_text = normalized_prompt + " " + normalized_completion
                    full_tokens = tokenizer(full_text, return_tensors='pt').to(training_model.device)
                    outputs = training_model(full_tokens.input_ids)
                    logits = outputs.logits[0]
                    completion_start = prompt_length - 1
                    completion_end = full_tokens.input_ids.shape[1] - 1
                    if completion_end > completion_start:
                        targets = full_tokens.input_ids[0][prompt_length:completion_end+1]
                        Tgen = len(targets)
                        pol_logits = logits[completion_start:completion_end]
                        pol_logp = F.log_softmax(pol_logits, dim=-1)
                        tok_logp_pol = pol_logp.gather(1, targets.unsqueeze(1)).squeeze()
                        with torch.no_grad():
                            ref_out = reference_model(full_tokens.input_ids)
                            ref_logits = ref_out.logits[0][completion_start:completion_end]
                            ref_logp = F.log_softmax(ref_logits, dim=-1)
                            tok_logp_ref = ref_logp.gather(1, targets.unsqueeze(1)).squeeze()
                        seq_logprob = tok_logp_pol.sum() / Tgen
                        probs = torch.exp(pol_logp)
                        tok_kl = (probs * (pol_logp - ref_logp)).sum(dim=-1)
                        seq_kl = tok_kl.mean()
                        tok_entropy = -(probs * pol_logp).sum(dim=-1).mean()
                        pg_term = -advantage * seq_logprob
                        kl_term = beta * seq_kl
                        ent_term = -entropy_coef * tok_entropy
                        sample_loss = (pg_term + kl_term + ent_term) * scale
                        sample_loss.backward()
                        pg_v = float(pg_term.detach().item())
                        kl_v = float(kl_term.detach().item())
                        ent_v = float(ent_term.detach().item())
                        prompt_pg_loss += pg_v
                        prompt_kl_loss += kl_v
                        print(f"  Gen {j+1}: A={advantage:+.3f}, logp/len={float(seq_logprob.detach().item()):.3f}, kl={float(seq_kl.detach().item()):.3f}, H={float(tok_entropy.detach().item()):.3f}")
                        print(f"         PG={pg_v:.3f}, KL_penalty={kl_v:.3f}, EntReg={ent_v:.3f}, total={(pg_v+kl_v+ent_v):.3f}")
                        processed += 1
                        # True GA: defer optimizer step until all samples processed
                    else:
                        print(f"  Gen {j+1}: Empty completion - skipping")
                avg_prompt_pg = prompt_pg_loss / num_generations
                avg_prompt_kl = prompt_kl_loss / num_generations
                total_pg_loss += avg_prompt_pg
                total_kl_loss += avg_prompt_kl
                print(f"  📊 Prompt averages: PG={avg_prompt_pg:.3f}, KL={avg_prompt_kl:.3f}")
            avg_pg_loss_val = total_pg_loss / batch_size
            avg_kl_loss_val = total_kl_loss / batch_size
            total_loss_val = avg_pg_loss_val + avg_kl_loss_val
            print(f"\n🔢 CORRECTED Loss Breakdown:")
            print(f"  Policy Gradient Loss: {avg_pg_loss_val:8.3f}")
            print(f"  KL Loss (with gradients): {avg_kl_loss_val:8.3f}")
            print(f"  Total Loss:          ={total_loss_val:8.3f}")
            kl_ratio = abs(avg_kl_loss_val) / (abs(avg_pg_loss_val) + abs(avg_kl_loss_val) + 1e-8) * 100
            print(f"  KL penalty ratio: {kl_ratio:.1f}%")
            if steps_done == 0:
                print(f"\n🔄 Performing gradient update (step {step_idx})...")
                total_grad_norm = 0.0
                for name, param in training_model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        total_grad_norm += grad_norm ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"  Total gradient norm: {total_grad_norm:.2f}")
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                print(f"  ✅ Gradient update applied with proper KL regularization")
            _t_loss = time.time() - _t_loss0
            print(f"[timing] loss_update: {_t_loss:.2f}s")

            # Update learning rate with schedule
            current_lr = get_lr_with_schedule(step_idx, steps, base_learning_rate, lr_schedule, lr_warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            if current_lr != base_learning_rate:
                print(f"  📈 LR schedule: {lr_schedule} → lr={current_lr:.2e} (base={base_learning_rate:.2e})")

            # Beta warmup + optional adaptation
            if step_idx <= beta_warmup_steps:
                beta = 0.0
                print(f"  🔧 Beta warmup: step {step_idx}/{beta_warmup_steps} → beta={beta:.3f}")
            else:
                if beta < beta_after_warmup:
                    beta = beta_after_warmup
                if beta_adapt:
                    kl_mag = float(abs(avg_kl_loss_val))
                    if kl_mag > target_kl * 1.05:
                        beta = beta * 1.1
                    elif kl_mag < target_kl * 0.95:
                        beta = max(0.0, beta * 0.9)
                    # Clamp beta to a safe range when adapting
                    beta = max(0.0, min(beta, 0.05))
                    print(f"  🔧 Beta adaptation: KL≈{kl_mag:.3f}, target≈{target_kl:.3f} → beta={beta:.3f}")

            # Post-update eval
            print(f"\n{'='*20} PHASE 9: POST-UPDATE PERFORMANCE (step {step_idx}) {'='*20}")
            _t_post_eval0 = time.time()
            post_performance = test_model_performance(training_model, f"POST-UPDATE (step {step_idx})")
            _t_post_eval = time.time() - _t_post_eval0
            print(f"[timing] post_eval: {_t_post_eval:.2f}s")
            perf_delta = post_performance - last_post_performance
            last_post_performance = post_performance
            print(f"\n{'='*20} PHASE 10: ANALYSIS (step {step_idx}) {'='*20}")
            _t_analysis0 = time.time()
            print(f"🔍 GRPO Step Impact (step {step_idx}):")
            print(f"  Post-update performance: {post_performance:.4f}")
            print(f"  Step performance change: {perf_delta:+.4f}")
            _t_analysis = time.time() - _t_analysis0
            _step_secs = time.time() - _step_t0
            print(f"\n⏱️ Timing (step {step_idx}): generation={_t_generation:.2f}s, rewards={_t_rewards:.2f}s, advantages={_t_adv:.2f}s, loss_update={_t_loss:.2f}s, post_eval={_t_post_eval:.2f}s, analysis={_t_analysis:.2f}s, total={_step_secs:.2f}s")

            # Evaluation on held-out set
            eval_accuracy = None
            if eval_every > 0 and (step_idx % eval_every == 0):
                print(f"\n📊 Evaluating on held-out set...")
                eval_accuracy = evaluate_on_eval_set(training_model, tokenizer, eval_file, max_samples=50)

                if eval_accuracy is not None:
                    if eval_accuracy >= early_stop_threshold:
                        print(f"\n✅ Early stopping: Eval accuracy {eval_accuracy:.1f}% >= {early_stop_threshold}%")
                        # Save final checkpoint before stopping
                        if checkpoint_every != 0:
                            checkpoint_path = save_checkpoint(training_model, optimizer, step_idx, checkpoint_dir)
                            tokenizer.save_pretrained(checkpoint_path)
                            print(f"  Final checkpoint saved at {checkpoint_path}")
                        early_stopped = True
                        last_post_performance = post_performance
                        break

            # Checkpoint only at specified intervals (not at every step)
            if checkpoint_every > 0 and (step_idx % checkpoint_every == 0):
                checkpoint_path = save_checkpoint(training_model, optimizer, step_idx, checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_path)

            # Store per-step metrics (overview)
            pre_val = float(post_performance - perf_delta)
            gradn = float(total_grad_norm) if 'total_grad_norm' in locals() else None
            _pg_val = float(avg_pg_loss_val)
            _kl_val = float(avg_kl_loss_val)
            _kl_ratio_val = float(kl_ratio)

            step_metrics.append({
                'step': int(step_idx),
                'pre': pre_val,
                'post': float(post_performance),
                'delta': float(perf_delta),
                'pg': _pg_val,
                'kl': _kl_val,
                'kl_ratio': _kl_ratio_val,
                'adv': step_adv_metric,
                'grad_norm': gradn,
                'beta': float(beta),
                'sec': float(_step_secs),
            })

    # Save final checkpoint if checkpoint_every == -1 (only at end)
    final_step = len(step_metrics) if step_metrics else 1
    if checkpoint_every == -1 and not early_stopped:
        print(f"\n💾 Saving final checkpoint at end of training (step {final_step})...")
        checkpoint_path = save_checkpoint(training_model, optimizer, final_step, checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"  Final checkpoint saved at {checkpoint_path}")

    return {
        'initial_performance': initial_performance,
        'post_performance': last_post_performance,
        'performance_change': last_post_performance - initial_performance,
        'total_loss': total_loss_val,
        'pg_loss': avg_pg_loss_val,
        'kl_penalty': avg_kl_loss_val,
        'step_metrics': step_metrics,
        # metadata for logging
        'batch_size': batch_size,
        'num_generations': num_generations,
        'grad_accum_steps': grad_accum_steps,
        'effective_batch': batch_size * num_generations,
        'microbatch_size': microbatch_size,
    }

def main():
    # Configure logging: stream to stdout and write full log to logs/
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime('%y%m%d-%H%M%S')
    logfile = f"logs/manual_grpo_debug_run-{ts}.log"

    logger = logging.getLogger('manual_grpo')
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if main() somehow re-enters
    if not logger.handlers:
        sh = logging.StreamHandler()
        fh = logging.FileHandler(logfile, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)

    # Redirect print to logger.info for full capture
    import builtins as _builtins
    _orig_print = _builtins.print

    def _print(*args, sep=' ', end='\n', **kwargs):
        try:
            msg = sep.join(str(a) for a in args)
        except Exception:
            msg = ' '.join(map(str, args))
        logger.info(msg)

    _builtins.print = _print

    print("🚨 DEBUGGING GRPO TRAINING - STEP BY STEP")
    print("=" * 70)
    print(f"Full log: {logfile}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--beta_adapt", action="store_true", default=False)
    parser.add_argument("--target_kl", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=0, help="Save checkpoint every N steps (0=disabled, -1=only at end)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--eval_every", type=int, default=0, help="Evaluate on held-out set every N steps (0=disabled)")
    parser.add_argument("--eval_file", type=str, default="opening_dataset_eval.json", help="Evaluation dataset file")
    parser.add_argument("--early_stop_threshold", type=float, default=95.0, help="Early stop if eval accuracy >= this")
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine", "linear", "step"], help="Learning rate schedule")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="LR warmup steps for schedules")
    parser.add_argument("--overfit_single_batch", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4, help="prompts per microbatch")
    parser.add_argument("--num_generations", "--gens", type=int, default=12, help="completions per prompt (GRPO group size)")
    parser.add_argument(
        "--grad_accum_steps", "--ga", type=int, default=1,
        help="reserved: currently ignored for stepping (true GA does a single optimizer step per batch)"
    )
    parser.add_argument("--beta_warmup_steps", type=int, default=20, help="warmup steps with beta=0 before switching to beta_after_warmup")
    parser.add_argument("--entropy_coef", type=float, default=0.005, help="entropy regularization coefficient")
    parser.add_argument("--task_type", type=str, default="P", choices=["P", "A", "mixed"],
                      help="Task type: P (policy), A (environment), or mixed")
    args = parser.parse_args()

    try:
        # Global seeding for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        results = manual_grpo_single_batch(
            steps=args.steps,
            beta_adapt=args.beta_adapt,
            target_kl=args.target_kl,
            checkpoint_every=args.checkpoint_every,
            overfit_single_batch=args.overfit_single_batch,
            seed=args.seed,
            batch_size=args.batch_size,
            num_generations=args.num_generations,
            grad_accum_steps=args.grad_accum_steps,
            beta_warmup_steps=args.beta_warmup_steps,
            entropy_coef=args.entropy_coef,
            task_type=args.task_type,
            checkpoint_dir=args.checkpoint_dir,
            eval_every=args.eval_every,
            eval_file=args.eval_file,
            early_stop_threshold=args.early_stop_threshold,
            lr_schedule=args.lr_schedule,
            lr_warmup_steps=args.lr_warmup_steps,
        )
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"  Initial model performance: {results['initial_performance']:.4f}")
        print(f"  After {args.steps} GRPO step(s): {results['post_performance']:.4f}")
        print(f"  Performance change: {results['performance_change']:+.4f}")
        print(f"  Total loss: {results['total_loss']:.2f}")
        
        # Overview table per step (always, if metrics available)
        rows = results.get('step_metrics', [])
        header = f"{'Step':>4} | {'Pre':>6} | {'Post':>6} | {'Δ':>6} | {'PG':>7} | {'KL':>7} | {'KL%':>5} | {'Adv':>6} | {'GradN':>6} | {'β':>5} | {'Sec':>6}"
        if rows:
            print("\n📋 Per-step overview:")
            print(header)
            print('-' * len(header))
            for r in rows:
                gradn = f"{r['grad_norm']:.2f}" if r['grad_norm'] is not None else "-"
                secs = f"{r.get('sec', 0.0):.2f}"
                print(f"{r['step']:>4} | {r['pre']:>6.3f} | {r['post']:>6.3f} | {r['delta']:>6.3f} | {r['pg']:>7.3f} | {r['kl']:>7.3f} | {r['kl_ratio']:>5.1f} | {r['adv']:>6.3f} | {gradn:>6} | {r['beta']:>5.3f} | {secs:>6}")

        # No separate summary file: full console output is mirrored to logs/manual_grpo_debug_run-*.log
        
    except Exception as e:
        print(f"❌ Error in manual GRPO analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
