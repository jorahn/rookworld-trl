"""Specialized GRPO training script that focuses on A: tasks only.

The script intentionally mirrors the manual-debug checklist while relying on
TRL's GRPOTrainer for the heavy lifting. Dataset preparation filters the
HuggingFace rookworld corpus down to well-formed environment transitions so
every batch contains grouped move targets for the A-task evaluation KPI.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState
from trl import GRPOConfig, GRPOTrainer

from .dataset import load_and_prepare_samples
from .rewards import create_reward_function


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    """Return unique items in insertion order without extra allocations."""
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


@dataclass
class ATDatasetSummary:
    requested: int
    kept: int
    distinct: int
    skipped_malformed: int
    skipped_duplicates: int

    def to_pretty_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


def build_a_task_dataset(
    limit: int,
    dataset_name: str,
    split: str,
    seed: int,
    min_required: int = 64,
) -> Tuple[Dataset, ATDatasetSummary]:
    """Create a Dataset containing only valid A: prompts."""
    raw_samples = load_and_prepare_samples(
        n_samples=max(limit * 3, min_required),
        dataset_name=dataset_name,
        split=split,
        seed=seed,
    )

    prompts: List[str] = []
    skipped_malformed = 0
    for task_type, prompt, _, _ in raw_samples:
        if task_type != "A":
            continue
        if not prompt or not prompt.endswith("+"):
            skipped_malformed += 1
            continue
        prompts.append(prompt)
        if len(prompts) >= limit:
            break

    if not prompts:
        raise RuntimeError("No valid A: prompts were recovered from the dataset")

    unique_prompts = _unique_preserve_order(prompts)
    skipped_duplicates = len(prompts) - len(unique_prompts)

    dataset = Dataset.from_dict({"prompt": unique_prompts})
    summary = ATDatasetSummary(
        requested=limit,
        kept=len(unique_prompts),
        distinct=len(unique_prompts),
        skipped_malformed=skipped_malformed,
        skipped_duplicates=skipped_duplicates,
    )

    if summary.kept < min_required:
        raise RuntimeError(
            f"Only {summary.kept} A: prompts available (min required {min_required})"
        )

    return dataset, summary


# -----------------------------------------------------------------------------
# Adaptive β controller
# -----------------------------------------------------------------------------

class AdaptiveBetaCallback(TrainerCallback):
    """Simple β warmup + bounded KL adaptation for GRPO Trainer."""

    KL_KEYS = (
        "kl_divergence",
        "train/kl_divergence",
        "kl",
        "train/kl",
    )

    def __init__(
        self,
        target_beta: float,
        warmup_steps: int,
        target_kl: float,
        max_beta: float,
        min_beta: float = 0.0,
        tolerance: float = 0.05,
        growth: float = 0.10,
        shrink: float = 0.10,
        verbose: bool = True,
    ) -> None:
        self._target_beta = target_beta
        self._warmup_steps = max(0, warmup_steps)
        self._target_kl = target_kl
        self._max_beta = max_beta
        self._min_beta = min_beta
        self._tolerance = tolerance
        self._growth = growth
        self._shrink = shrink
        self._verbose = verbose
        self._current_beta = 0.0

    def _maybe_log(self, message: str) -> None:
        if self._verbose:
            print(message)

    def on_step_begin(
        self,
        args: GRPOConfig,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if state.global_step < self._warmup_steps:
            beta = 0.0
        elif self._current_beta == 0.0:
            beta = self._target_beta
            self._current_beta = beta
        else:
            beta = self._current_beta
        args.beta = float(np.clip(beta, self._min_beta, self._max_beta))
        return control

    def on_log(
        self,
        args: GRPOConfig,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> TrainerControl:
        if logs is None:
            return control
        if state.global_step < self._warmup_steps:
            return control

        raw_kl: Optional[float] = None
        for key in self.KL_KEYS:
            if key in logs:
                raw_kl = float(logs[key])
                break
        if raw_kl is None:
            return control

        if self._current_beta == 0.0:
            self._current_beta = self._target_beta

        upper = self._target_kl * (1 + self._tolerance)
        lower = self._target_kl * (1 - self._tolerance)

        beta = self._current_beta
        if raw_kl > upper:
            beta = min(self._max_beta, beta * (1 + self._growth))
        elif raw_kl < lower:
            beta = max(self._min_beta, beta * (1 - self._shrink))

        if beta != self._current_beta:
            self._maybe_log(f"  🔧 Adaptive β update: KL={raw_kl:.4f} → β={beta:.5f}")
        self._current_beta = beta
        args.beta = float(np.clip(beta, self._min_beta, self._max_beta))
        return control


# -----------------------------------------------------------------------------
# Training configuration and launcher
# -----------------------------------------------------------------------------

@dataclass
class ATaskTrainingConfig:
    model_name: str = "jrahn/RookWorld-LM-124M"
    output_dir: str = "grpo_output_a_task"
    dataset_name: str = "jrahn/rookworld_7m"
    dataset_split: str = "train"
    dataset_limit: int = 2048
    seed: int = 42
    learning_rate: float = 2e-7
    beta: float = 0.02
    beta_max: float = 0.05
    beta_warmup_steps: int = 10
    target_kl: float = 0.15
    num_train_epochs: int = 1
    max_steps: int = -1
    batch_size: int = 8
    num_generations: int = 16
    max_completion_length: int = 144
    temperature: float = 0.7
    top_p: float = 0.98
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 0
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    bf16: bool = True
    tensorboard: bool = False
    stockfish_path: Optional[str] = None
    disable_beta_adaptation: bool = False


def parse_args() -> ATaskTrainingConfig:
    parser = argparse.ArgumentParser(description="GRPO training focused on A: tasks")
    parser.add_argument("--model_name", default=ATaskTrainingConfig.model_name)
    parser.add_argument("--output_dir", default=ATaskTrainingConfig.output_dir)
    parser.add_argument("--dataset_name", default=ATaskTrainingConfig.dataset_name)
    parser.add_argument("--dataset_split", default=ATaskTrainingConfig.dataset_split)
    parser.add_argument("--dataset_limit", type=int, default=ATaskTrainingConfig.dataset_limit)
    parser.add_argument("--seed", type=int, default=ATaskTrainingConfig.seed)
    parser.add_argument("--learning_rate", type=float, default=ATaskTrainingConfig.learning_rate)
    parser.add_argument("--beta", type=float, default=ATaskTrainingConfig.beta)
    parser.add_argument("--beta_max", type=float, default=ATaskTrainingConfig.beta_max)
    parser.add_argument("--beta_warmup_steps", type=int, default=ATaskTrainingConfig.beta_warmup_steps)
    parser.add_argument("--target_kl", type=float, default=ATaskTrainingConfig.target_kl)
    parser.add_argument("--num_train_epochs", type=int, default=ATaskTrainingConfig.num_train_epochs)
    parser.add_argument("--max_steps", type=int, default=ATaskTrainingConfig.max_steps)
    parser.add_argument("--batch_size", type=int, default=ATaskTrainingConfig.batch_size)
    parser.add_argument("--num_generations", type=int, default=ATaskTrainingConfig.num_generations)
    parser.add_argument("--max_completion_length", type=int, default=ATaskTrainingConfig.max_completion_length)
    parser.add_argument("--temperature", type=float, default=ATaskTrainingConfig.temperature)
    parser.add_argument("--top_p", type=float, default=ATaskTrainingConfig.top_p)
    parser.add_argument("--logging_steps", type=int, default=ATaskTrainingConfig.logging_steps)
    parser.add_argument("--save_steps", type=int, default=ATaskTrainingConfig.save_steps)
    parser.add_argument("--eval_steps", type=int, default=ATaskTrainingConfig.eval_steps)
    parser.add_argument("--max_grad_norm", type=float, default=ATaskTrainingConfig.max_grad_norm)
    parser.add_argument("--warmup_steps", type=int, default=ATaskTrainingConfig.warmup_steps)
    parser.add_argument("--bf16", action="store_true", default=ATaskTrainingConfig.bf16)
    parser.add_argument("--no_bf16", action="store_true", help="Disable BF16 even if available")
    parser.add_argument("--tensorboard", action="store_true", default=ATaskTrainingConfig.tensorboard)
    parser.add_argument("--stockfish_path", default=ATaskTrainingConfig.stockfish_path)
    parser.add_argument("--disable_beta_adaptation", action="store_true")
    args = parser.parse_args()

    config = ATaskTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_limit=args.dataset_limit,
        seed=args.seed,
        learning_rate=args.learning_rate,
        beta=args.beta,
        beta_max=args.beta_max,
        beta_warmup_steps=args.beta_warmup_steps,
        target_kl=args.target_kl,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=max(144, args.max_completion_length),
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        bf16=False if args.no_bf16 else args.bf16,
        tensorboard=args.tensorboard,
        stockfish_path=args.stockfish_path,
        disable_beta_adaptation=args.disable_beta_adaptation,
    )
    return config


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    config = parse_args()
    set_global_seed(config.seed)

    print("=" * 88)
    print("🏁 GRPO TRAINING FOR A: TASKS")
    print("=" * 88)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}/{config.dataset_split}")
    print(f"Requested prompts: {config.dataset_limit}")
    print(f"Output dir: {config.output_dir}")
    print(f"Hyperparams → lr={config.learning_rate:.2e}, β={config.beta:.3f}, gens={config.num_generations}")
    print("=" * 88)

    print("\n📊 Preparing A: task dataset...")
    train_dataset, summary = build_a_task_dataset(
        limit=config.dataset_limit,
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        seed=config.seed,
    )
    print(f"✓ Dataset summary: {summary.to_pretty_json()}")

    os.makedirs(config.output_dir, exist_ok=True)

    print("\n📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, object] = {}
    if config.bf16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        print("✓ Using BF16 precision")

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    print("\n🏆 Initializing reward function...")
    reward_fn = create_reward_function(config.stockfish_path)

    # Configure GRPO
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        report_to=["tensorboard"] if config.tensorboard else [],
        logging_dir=f"{config.output_dir}/runs" if config.tensorboard else None,
        temperature=config.temperature,
        top_p=config.top_p,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
    )

    if hasattr(training_args, "per_device_train_batch_size"):
        training_args.per_device_train_batch_size = config.batch_size
    elif hasattr(training_args, "train_batch_size"):
        training_args.train_batch_size = config.batch_size

    print("\n🚀 Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    if not getattr(config, "disable_beta_adaptation", False):
        print(
            f"\n🔁 β warmup+adaptation enabled: warmup={config.beta_warmup_steps} steps, "
            f"target_KL={config.target_kl}, β∈[0,{config.beta_max}]"
        )
        trainer.add_callback(
            AdaptiveBetaCallback(
                target_beta=config.beta,
                warmup_steps=config.beta_warmup_steps,
                target_kl=config.target_kl,
                max_beta=config.beta_max,
                min_beta=0.0,
            )
        )
    else:
        print("\n⚙️ β adaptation disabled via flag; using constant β")

    print("\n🎯 Starting GRPO training on A: prompts...")
    trainer.train()

    print("\n💾 Saving fine-tuned model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    print("\n✅ Training complete. Ready for evaluation.")


if __name__ == "__main__":
    main()
