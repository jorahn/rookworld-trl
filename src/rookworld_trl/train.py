"""
RookWorld GRPO training script using HuggingFace Transformers and TRL.

Complete implementation with enhanced chess-accurate reward system and 
RookWorld dataset integration for training GPT-2 models on chess tasks.
"""

import argparse
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import chess
import chess.pgn
from pathlib import Path
import random

from .rewards import create_reward_function
from .dataset import RookWorldDataGenerator, load_and_prepare_samples


@dataclass
class TrainingConfig:
    """Centralized configuration for GRPO training with optimal tested values."""
    
    # Model & Output
    model_name: str = "jrahn/RookWorld-LM-124M"
    output_dir: str = "./grpo_output"
    
    # Training Parameters (tested optimal values)
    learning_rate: float = 1e-6      # Optimal for chess knowledge preservation
    batch_size: int = 16             # Optimal from benchmarking (4x improvement)
    num_train_epochs: int = 1        # Single epoch training
    dataset_size: int = 5000         # Substantial training data
    gradient_accumulation_steps: int = 1  # TRL default
    max_length: int = 256            # Token sequence length
    
    # GRPO Algorithm Parameters (tested optimal values)
    beta: float = 0.1                # Optimal KL/PG balance from testing
    num_generations: int = 4         # Generations per prompt (good diversity)
    max_completion_length: int = 256 # Max completion tokens
    
    # Generation Parameters (working values that preserve chess format)
    temperature: float = 0.5         # Focused sampling (vs TRL default 1.0)
    top_p: float = 0.9              # Nucleus sampling (vs TRL default 1.0)
    
    # Stability Parameters (prevent model degradation)
    max_grad_norm: float = 1.0       # Gradient clipping threshold
    warmup_steps: int = 100          # Learning rate warmup (vs TRL default 0)
    
    # Logging & Evaluation (frequent monitoring)
    eval_steps: int = 100            # Evaluation frequency
    save_steps: int = 100            # Save frequency (vs TRL default 500)
    logging_steps: int = 10          # Logging frequency
    tensorboard: bool = False        # Enable via CLI flag
    
    # Hardware Optimizations
    bf16: bool = True               # Mixed precision training
    use_torch_compile: bool = False # Optional optimization
    
    # Optional Parameters
    stockfish_path: Optional[str] = None  # Auto-detect if None
    stable: bool = False            # Ultra-conservative mode via CLI
    dataset_name: str = "jrahn/rookworld_7m"


def create_dataset(data_generator: RookWorldDataGenerator, size: int = 1000) -> Dataset:
    """Create a dataset for training using RookWorld data."""
    # Use the generator to create mixed prompts
    prompts = data_generator.get_mixed_batch(size)
    
    # Ensure we have enough data
    while len(prompts) < size and len(data_generator.samples) > 0:
        additional_prompts = data_generator.get_mixed_batch(
            min(size - len(prompts), len(data_generator.samples))
        )
        prompts.extend(additional_prompts)
    
    # Remove duplicates but keep order diversity
    unique_prompts = []
    seen = set()
    for prompt in prompts:
        if prompt not in seen:
            unique_prompts.append(prompt)
            seen.add(prompt)
    
    return Dataset.from_dict({"prompt": unique_prompts[:size]})


def main():
    parser = argparse.ArgumentParser(description="Train GRPO on RookWorld chess tasks")
    parser.add_argument("--model_name", default="jrahn/RookWorld-LM-124M", help="Model to fine-tune")
    parser.add_argument("--output_dir", default="./grpo_output", help="Output directory")
    # Use TrainingConfig defaults for all CLI arguments
    default_config = TrainingConfig()
    
    parser.add_argument("--batch_size", type=int, default=default_config.batch_size, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=default_config.learning_rate, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=default_config.num_train_epochs, help="Number of epochs")
    parser.add_argument("--stockfish_path", default=default_config.stockfish_path, help="Path to Stockfish binary (auto-detect if None)")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--max_completion_length", type=int, default=default_config.max_completion_length, help="Max completion length (min 144)")
    parser.add_argument("--dataset_size", type=int, default=default_config.dataset_size, help="Number of samples to load from dataset")
    parser.add_argument("--num_generations", type=int, default=default_config.num_generations, help="Number of completions per prompt")
    parser.add_argument("--beta", type=float, default=default_config.beta, help="KL penalty coefficient")
    parser.add_argument("--eval_steps", type=int, default=default_config.eval_steps, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=default_config.save_steps, help="Save frequency") 
    parser.add_argument("--logging_steps", type=int, default=default_config.logging_steps, help="Logging frequency")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")
    parser.add_argument("--max_grad_norm", type=float, default=default_config.max_grad_norm, help="Gradient clipping threshold")
    parser.add_argument("--warmup_steps", type=int, default=default_config.warmup_steps, help="Learning rate warmup steps")
    parser.add_argument("--stable", action="store_true", help="Use stable training configuration")
    parser.add_argument("--temperature", type=float, default=default_config.temperature, help="Generation temperature (lower for focused sampling)")
    parser.add_argument("--top_p", type=float, default=default_config.top_p, help="Nucleus sampling top_p")
    
    args = parser.parse_args()
    
    # Ensure max completion length is at least 144
    max_completion_length = max(144, args.max_completion_length)
    if max_completion_length > args.max_completion_length:
        print(f"⚠️ Increased max_completion_length from {args.max_completion_length} to {max_completion_length} (minimum required)")
    
    # Apply stable configuration if requested
    if args.stable:
        print("🛡️ Applying stable training configuration...")
        # Override settings for stability
        args.learning_rate = min(args.learning_rate, 1e-6)  # Cap learning rate
        args.beta = max(args.beta, 0.2)  # Moderate KL penalty for stability
        args.max_grad_norm = min(args.max_grad_norm, 0.5)  # Lower gradient clipping
        args.warmup_steps = max(args.warmup_steps, 200)  # Longer warmup
        print(f"  • Learning rate: {args.learning_rate:.2e}")
        print(f"  • KL penalty (beta): {args.beta}")  
        print(f"  • Max grad norm: {args.max_grad_norm}")
        print(f"  • Warmup steps: {args.warmup_steps}")
    
    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        stockfish_path=args.stockfish_path,
        bf16=args.bf16,
        use_torch_compile=args.compile,
        max_completion_length=max_completion_length,
        dataset_size=args.dataset_size,
        num_generations=args.num_generations,
        beta=args.beta,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        tensorboard=args.tensorboard,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        stable=args.stable,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    print("=" * 80)
    print("🏆 ROOKWORLD GRPO TRAINING WITH TRL")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name} ({config.dataset_size} samples)")
    print(f"Output: {config.output_dir}")
    print(f"Generations per prompt: {config.num_generations}")
    print(f"Max completion length: {config.max_completion_length}")
    print(f"BF16: {config.bf16}")
    print("=" * 80)
    
    # Load tokenizer and model
    print("\n📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Properly set pad token without warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Don't change the model config here - let TRL handle it
    
    # Model loading with optimizations
    model_kwargs = {}
    if config.bf16 and torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16
        print("✓ Using BF16 precision")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    if config.use_torch_compile:
        model = torch.compile(model, mode="reduce-overhead")
        print("✓ Using torch.compile optimization")
    
    # Create reward function with auto-detection
    print("\n🏆 Initializing reward function...")
    reward_function = create_reward_function(config.stockfish_path)
    
    # Generate training data from RookWorld dataset
    print(f"\n📊 Loading RookWorld dataset...")
    data_generator = RookWorldDataGenerator(
        max_length=config.max_length,
        dataset_size=config.dataset_size
    )
    
    # Show dataset composition
    info = data_generator.get_samples_info()
    print(f"✓ Dataset composition: {info['total']} total ({info['P']} P: tasks, {info['A']} A: tasks)")
    
    train_dataset = create_dataset(data_generator, size=config.dataset_size)
    print(f"✓ Created training dataset with {len(train_dataset)} unique samples (requested {config.dataset_size})")
    
    # Training configuration - minimal working configuration
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        report_to=["tensorboard"] if config.tensorboard else [],
        logging_dir=f"{config.output_dir}/runs" if config.tensorboard else None,
        max_steps=max(1, len(train_dataset) // config.batch_size) if len(train_dataset) < 1000 else -1,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        temperature=config.temperature,
        top_p=config.top_p
    )
    
    # Initialize trainer
    print("\n🚀 Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
    )
    
    print("\n🎯 Starting GRPO training...")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Beta (KL coef): {config.beta}")
    print(f"   Generations per prompt: {config.num_generations}")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    print(f"\n💾 Saving model to {config.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Model saved to: {config.output_dir}")
    print("🏆 Ready for chess inference and further training!")


if __name__ == "__main__":
    main()