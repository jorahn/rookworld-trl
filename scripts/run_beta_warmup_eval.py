#!/usr/bin/env python3
"""Run a 10-step GRPO diagnostic with beta warmup on the A-task dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.rookworld_trl.train_a import AdaptiveBetaCallback
from src.rookworld_trl.utils import normalize_spacing
from train_a_min import reward_factory


def _extract_generated_fen(text: str, pattern_prompt_prefix: re.Pattern[str]) -> str:
    normalized = normalize_spacing(text)
    normalized = pattern_prompt_prefix.sub("", normalized)
    return normalized.split("+")[0].strip()


def evaluate_subset(
    model: AutoModelForCausalLM,
    dataset: List[Dict],
    label: str,
    step: int,
    tokenizer: AutoTokenizer,
    reward_fn,
    log_dir: Path,
    max_new_tokens: int,
    pattern_prompt_prefix: re.Pattern[str],
) -> Tuple[float, float]:
    records = []
    matches = 0
    rewards: List[float] = []
    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    try:
        for idx, sample in enumerate(dataset):
            prompt = sample["prompt"]
            expected = sample.get("expected_fen", "")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs.input_ids.shape[1]
            completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            reward = reward_fn([completion], prompts=[prompt])[0]
            predicted = _extract_generated_fen(completion, pattern_prompt_prefix)
            match = bool(expected) and predicted == expected
            if match:
                matches += 1
            rewards.append(reward)
            records.append(
                {
                    "step": step,
                    "index": idx,
                    "move_num": sample.get("move_num"),
                    "prompt": prompt,
                    "expected_fen": expected,
                    "generated_fen": predicted,
                    "raw_completion": completion,
                    "reward": reward,
                    "match": match,
                    "dataset": label,
                }
            )
    finally:
        if model_was_training:
            model.train()

    accuracy = matches / len(dataset) * 100 if dataset else 0.0
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    out_file = log_dir / f"{label}_step_{step:03d}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    print(
        f"[{label}] step {step:03d}: accuracy={accuracy:.2f}% avg_reward={avg_reward:.3f} -> {out_file}"
    )
    return accuracy, avg_reward


class LoggingEvalCallback(TrainerCallback):
    def __init__(
        self,
        datasets: Dict[str, List[Dict]],
        tokenizer: AutoTokenizer,
        reward_fn,
        log_dir: Path,
        summary_records: List[Dict],
        max_new_tokens: int,
        pattern_prompt_prefix: re.Pattern[str],
        eval_every: int,
    ) -> None:
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.log_dir = log_dir
        self.summary_records = summary_records
        self.max_new_tokens = max_new_tokens
        self.pattern_prompt_prefix = pattern_prompt_prefix
        self.eval_every = eval_every

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        step = state.global_step
        if step == 0 or step % self.eval_every != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        for label, dataset in self.datasets.items():
            acc, reward = evaluate_subset(
                model,
                dataset,
                label,
                step,
                self.tokenizer,
                self.reward_fn,
                self.log_dir,
                self.max_new_tokens,
                self.pattern_prompt_prefix,
            )
            self.summary_records.append(
                {"dataset": label, "step": step, "accuracy": acc, "avg_reward": reward}
            )

        return control


def build_train_dataset(subset: List[Dict[str, str]]) -> Dataset:
    return Dataset.from_list([{ "prompt": sample["prompt"] } for sample in subset])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 10-step A-task diagnostic with beta warmup")
    parser.add_argument("--train-path", default="opening_dataset_train.json")
    parser.add_argument("--eval-path", default="opening_dataset_eval.json")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10, help="Limit to N updates (<=0 runs full epoch)")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=0.1)
    parser.add_argument("--beta-warmup", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=5e5)
    parser.add_argument("--kl-tolerance", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1.5e-7)
    parser.add_argument("--max-grad-norm", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--output-root", default="logs")
    parser.add_argument(
        "--growth", type=float, default=0.25, help="adaptive beta growth factor when KL > target"
    )
    parser.add_argument(
        "--shrink", type=float, default=0.2, help="adaptive beta shrink factor when KL < target"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.train_path, "r", encoding="utf-8") as f:
        train_data_full = json.load(f)
    with open(args.eval_path, "r", encoding="utf-8") as f:
        eval_data_full = json.load(f)

    subset_size = max(1, math.ceil(len(train_data_full) * args.fraction))
    eval_size = max(1, math.ceil(len(eval_data_full) * args.fraction))
    train_subset = train_data_full[:subset_size]
    eval_subset = eval_data_full[:eval_size]

    print(f"Using {subset_size}/{len(train_data_full)} training samples ({args.fraction:.0%}).")
    print(f"Using {eval_size}/{len(eval_data_full)} eval samples ({args.fraction:.0%}).")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this diagnostic run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required")

    tokenizer = AutoTokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "jrahn/RookWorld-LM-124M",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    reward_fn = reward_factory()
    pattern_prompt_prefix = re.compile(r"^\s*[PA]:\s*")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = "epoch_beta_warmup" if args.steps <= 0 else "ten_step_beta_warmup"
    log_dir = Path(args.output_root) / f"{run_tag}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging detailed generations under {log_dir}")

    summary_records: List[Dict] = []

    train_dataset = build_train_dataset(train_subset)
    max_steps = args.steps if args.steps > 0 else -1

    config = GRPOConfig(
        output_dir=str(log_dir / "checkpoint"),
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.generations,
        num_train_epochs=1,
        max_completion_length=args.max_new_tokens,
        bf16=True,
        report_to=[],
        dataloader_pin_memory=True,
        max_steps=max_steps,
        generation_batch_size=min(args.batch_size, len(train_subset)) * args.generations,
        logging_steps=1,
        max_grad_norm=args.max_grad_norm,
    )

    if hasattr(config, "per_device_train_batch_size"):
        config.per_device_train_batch_size = min(args.batch_size, len(train_subset))

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    datasets = {"train": train_subset, "eval": eval_subset}

    # Baseline evaluations (step 0)
    for label, dataset in datasets.items():
        acc, reward = evaluate_subset(
            model,
            dataset,
            label,
            0,
            tokenizer,
            reward_fn,
            log_dir,
            args.max_new_tokens,
            pattern_prompt_prefix,
        )
        summary_records.append({"dataset": label, "step": 0, "accuracy": acc, "avg_reward": reward})

    trainer.add_callback(
        AdaptiveBetaCallback(
            target_beta=args.beta,
            warmup_steps=args.beta_warmup,
            target_kl=args.target_kl,
            max_beta=args.beta_max,
            min_beta=0.0,
            tolerance=args.kl_tolerance,
            growth=args.growth,
            shrink=args.shrink,
            verbose=True,
        )
    )
    trainer.add_callback(
        LoggingEvalCallback(
            datasets,
            tokenizer,
            reward_fn,
            log_dir,
            summary_records,
            args.max_new_tokens,
            pattern_prompt_prefix,
            args.eval_every,
        )
    )

    print("\n=== Starting diagnostic run ===")
    trainer.train()

    final_step = args.steps if args.steps > 0 else trainer.state.global_step
    for label, dataset in datasets.items():
        acc, reward = evaluate_subset(
            trainer.model,
            dataset,
            label,
            final_step,
            tokenizer,
            reward_fn,
            log_dir,
            args.max_new_tokens,
            pattern_prompt_prefix,
        )
        summary_records.append(
            {"dataset": label, "step": final_step, "accuracy": acc, "avg_reward": reward}
        )

    summary_path = log_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"Summary written to {summary_path}")

    trainer.save_model(log_dir / "final_model")
    tokenizer.save_pretrained(log_dir / "final_model")
    print("Run complete.")


if __name__ == "__main__":
    main()
