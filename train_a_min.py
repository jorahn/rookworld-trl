#!/usr/bin/env python3
"""Minimal GRPO training loop for A: tasks using TRL and a JSON dataset."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from typing import Iterable, List, Tuple

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import chess
import torch
import Levenshtein


def load_prompts(path: str, limit: int | None) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    prompts = []
    for sample in data:
        prompt = sample["prompt"].strip()
        prompts.append({"prompt": prompt})
    return Dataset.from_list(prompts)


def normalize_spacing(text: str) -> str:
    if not text:
        return text
    return " ".join(text.split())


class EnvironmentReward:
    """Reward function tailored for A-task evaluation with edit-distance shaping."""

    def score(self, completions: List[str], *, prompts: Iterable[str] | None = None, **_: object) -> List[float]:
        if prompts is None:
            prompts = [""] * len(completions)
        rewards: List[float] = []
        for completion, prompt in zip(completions, prompts):
            rewards.append(self._score_single(prompt, completion))
        return rewards

    def _score_single(self, prompt: str, completion: str) -> float:
        prompt_norm = normalize_spacing(prompt)
        completion_norm = normalize_spacing(completion)

        expected = self._extract_expected_response(prompt_norm)
        if expected is None:
            return -1.0

        expected_fen, expected_suffix, expected_full = expected

        if completion_norm.strip() == expected_full:
            return 1.0

        parts = completion_norm.split("+")
        predicted_fen = parts[0].strip() if parts else ""
        predicted_suffix = "+".join(part.strip() for part in parts[1:]).strip()

        suffix_similarity = self._normalized_similarity(predicted_suffix, expected_suffix)
        suffix_adjustment = -0.2 + 0.4 * suffix_similarity

        if predicted_fen == expected_fen:
            reward = 0.8 + suffix_adjustment
        else:
            fen_similarity = self._normalized_similarity(predicted_fen, expected_fen)
            fen_reward = -0.8 + 0.6 * fen_similarity
            reward = fen_reward + suffix_adjustment

        return max(-1.0, min(1.0, reward))

    def _normalized_similarity(self, predicted: str, expected: str) -> float:
        if not predicted and not expected:
            return 1.0
        max_len = max(len(predicted), len(expected), 1)
        distance = Levenshtein.distance(predicted, expected)
        similarity = 1.0 - (distance / max_len)
        return max(0.0, min(1.0, similarity))

    def _extract_expected_response(self, prompt: str) -> Tuple[str, str, str] | None:
        match = re.search(r"A:\s*([^+]+)\+([^+]+)\+([^+]*)\+?", prompt)
        if not match:
            return None

        fen_str = match.group(1).strip()
        move_str = match.group(2).strip()

        try:
            board = chess.Board(fen_str)
            move = chess.Move.from_uci(move_str)
        except (ValueError, chess.InvalidMoveError):
            return None

        if move not in board.legal_moves:
            return None

        mover_is_white = board.turn
        next_board = board.copy()
        next_board.push(move)
        expected_fen = next_board.fen()

        terminated = int(next_board.is_game_over())
        truncated = 0

        if terminated:
            result = next_board.result()
            if result == "1-0":
                reward_value = 1.0 if mover_is_white else 0.0
            elif result == "0-1":
                reward_value = 1.0 if not mover_is_white else 0.0
            else:
                reward_value = 0.5
        else:
            reward_value = 0.001

        reward_str = self._format_reward_value(reward_value)
        expected_suffix = f"{reward_str}+{terminated}+{truncated}"
        expected_full = f"{expected_fen}+{expected_suffix}"

        return expected_fen, expected_suffix, expected_full

    def _format_reward_value(self, value: float) -> str:
        if abs(value - 0.001) < 1e-9:
            return "0.001"
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip('0').rstrip('.')


def reward_factory():
    scorer = EnvironmentReward()
    return scorer.score


def load_eval_records(path: str, max_samples: int | None) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    return data


def extract_generated_fen(text: str) -> str:
    normalized = normalize_spacing(text)
    normalized = re.sub(r"^\s*[PA]:\s*", "", normalized)
    return normalized.split("+")[0].strip()


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_records: List[dict],
    reward_fn,
    max_new_tokens: int = 196,
) -> Tuple[float, float]:
    if not eval_records:
        return 0.0, 0.0

    device = next(model.parameters()).device
    completions: List[str] = []
    prompts: List[str] = []
    exact_matches = 0

    model_was_training = model.training
    model.eval()
    try:
        for record in eval_records:
            prompt = record.get("prompt", "")
            expected = record.get("expected_fen", "")
            prompts.append(prompt)

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
            completion = tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            )
            completions.append(completion)

            predicted = extract_generated_fen(completion)
            if predicted and expected and predicted == expected:
                exact_matches += 1

    finally:
        if model_was_training:
            model.train()

    rewards = reward_fn(completions, prompts=prompts)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    accuracy = exact_matches / len(eval_records) * 100.0
    return accuracy, avg_reward


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        eval_records: List[dict],
        reward_fn,
        interval: int,
        max_new_tokens: int = 196,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_records = eval_records
        self.reward_fn = reward_fn
        self.interval = max(1, interval)
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.interval != 0:
            return control

        model = kwargs.get("model")
        if model is None or not self.eval_records:
            return control

        timestamp = datetime.now().strftime("%H:%M:%S")
        acc, reward = evaluate_model(
            model,
            self.tokenizer,
            self.eval_records,
            self.reward_fn,
            max_new_tokens=self.max_new_tokens,
        )
        print(
            f"[{timestamp}] 🔍 Eval @ step {step}: accuracy={acc:.2f}% | avg_reward={reward:.3f}"
        )

        return control


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO trainer for A tasks")
    parser.add_argument("--dataset", default="opening_dataset_train.json")
    parser.add_argument("--output_dir", default="grpo_a_min")
    parser.add_argument("--model", default="jrahn/RookWorld-LM-124M")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-7)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_dataset", default="opening_dataset_eval.json")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--kl_epsilon", type=float, default=0.1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this training script")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU must support BF16 for this training script")

    print("✓ CUDA device with BF16 support detected")

    dataset = load_prompts(args.dataset, args.limit)
    print(f"✓ Loaded {len(dataset)} training prompts from {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch_dtype,
    )
    if not hasattr(model, "hf_device_map"):
        model = model.to("cuda")

    eval_records = load_eval_records(args.eval_dataset, args.eval_samples)
    print(f"✓ Loaded {len(eval_records)} eval samples from {args.eval_dataset}")

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.generations,
        num_train_epochs=args.epochs,
        max_completion_length=196,
        max_grad_norm=args.max_grad_norm,
        epsilon=args.kl_epsilon,
        bf16=True,
        report_to=[],
        dataloader_pin_memory=True,
        max_steps=args.max_steps,
        generation_batch_size=args.batch_size * args.generations,
        logging_steps=args.logging_steps,
    )
    if hasattr(config, "per_device_train_batch_size"):
        config.per_device_train_batch_size = args.batch_size
    if hasattr(config, "generation_batch_size"):
        config.generation_batch_size = args.batch_size * args.generations

    reward_fn = reward_factory()

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    if eval_records:
        trainer.add_callback(
            EvalCallback(
                tokenizer=tokenizer,
                eval_records=eval_records,
                reward_fn=reward_fn,
                interval=args.eval_interval,
                max_new_tokens=196,
            )
        )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
