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
    """Configuration for GRPO training."""
    model_name: str = "jrahn/RookWorld-LM-124M"
    output_dir: str = "./grpo_output"
    
    # Training parameters
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 256
    
    # GRPO specific
    beta: float = 0.1  # KL penalty coefficient
    num_generations: int = 4  # Number of completions per prompt (group size)
    max_completion_length: int = 256  # Max length for generated completions (minimum 144)
    
    # Hardware optimizations
    bf16: bool = True
    use_torch_compile: bool = False
    
    # Stockfish path (optional - will auto-detect if None)
    stockfish_path: Optional[str] = None
    
    # Dataset parameters
    dataset_size: int = 500
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--stockfish_path", default=None, help="Path to Stockfish binary (auto-detect if None)")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--max_completion_length", type=int, default=256, help="Max completion length (min 144)")
    parser.add_argument("--dataset_size", type=int, default=500, help="Number of samples to load from dataset")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    
    args = parser.parse_args()
    
    # Ensure max completion length is at least 144
    max_completion_length = max(144, args.max_completion_length)
    if max_completion_length > args.max_completion_length:
        print(f"⚠️ Increased max_completion_length from {args.max_completion_length} to {max_completion_length} (minimum required)")
    
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading with optimizations
    model_kwargs = {}
    if config.bf16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
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
    
    train_dataset = create_dataset(data_generator, size=200)
    print(f"✓ Created training dataset with {len(train_dataset)} samples")
    
    # Training configuration - minimal working configuration
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_generations=config.num_generations,
        model_init_kwargs={},  # Required by GRPOTrainer
        max_completion_length=config.max_completion_length,
        logging_steps=10,
        save_steps=100,
        bf16=config.bf16,
        report_to=[],  # Disable wandb and other external logging
        logging_dir=None,  # Disable tensorboard logging
    )
    
    # Initialize trainer
    print("\n🚀 Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
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