#!/usr/bin/env python3
"""
Quick evaluation script for checking model performance on held-out eval set.
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys
from src.rookworld_trl.utils import normalize_spacing

def generate_eval(model, **kwargs):
    """Generate in eval mode to eliminate dropout randomness, restore previous mode"""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model.generate(**kwargs)
        if was_training:
            model.train()
        return outputs
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        if was_training:
            model.train()
        return None

def evaluate_model(model_path="jrahn/RookWorld-LM-124M", eval_file="opening_dataset_eval.json", sample_size=50):
    """Evaluate model on held-out eval set using EXACT FEN matching."""

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval dataset
    with open(eval_file) as f:
        eval_data = json.load(f)

    # Sample subset
    if sample_size and sample_size < len(eval_data):
        import random
        random.seed(42)
        eval_data = random.sample(eval_data, sample_size)

    print(f"Evaluating on {len(eval_data)} samples...")

    exact_matches = 0
    by_move = {}

    def _extract_generated_fen(text: str) -> str:
        # Normalize spacing and strip task markers
        norm = normalize_spacing(text)
        norm = re.sub(r"^\s*[PA]:\s*", "", norm)
        # Take content before first '+' as FEN (handles A-task schema: FEN+reward+terminates+truncated)
        return norm.split("+")[0].strip()

    def _fen_exact_match(pred: str, exp: str) -> bool:
        # EXACT match - all fields must match
        return pred == exp

    for sample in tqdm(eval_data, desc="Evaluating"):
        prompt = sample["prompt"]
        expected = sample["expected_fen"]
        move_num = sample["move_num"]

        if move_num not in by_move:
            by_move[move_num] = {"correct": 0, "total": 0}
        by_move[move_num]["total"] += 1

        # Generate with greedy decoding using generate_eval for consistency
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Use generate_eval to properly handle eval mode
        outputs = generate_eval(
            model,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=196,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Handle generation failure
        if outputs is None:
            print(f"  ⚠️  Generation failed for sample {sample['prompt'][:50]}...")
            continue

        prompt_len = inputs.input_ids.shape[1]
        completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        generated_fen = _extract_generated_fen(completion)

        if generated_fen and _fen_exact_match(generated_fen, expected):
            exact_matches += 1
            by_move[move_num]["correct"] += 1

    # Calculate metrics
    accuracy = exact_matches / len(eval_data) * 100

    print(f"\n=== EVALUATION RESULTS (EXACT FEN MATCH) ===")
    print(f"Overall accuracy: {exact_matches}/{len(eval_data)} ({accuracy:.1f}%)")
    print("Note: Using exact FEN matching - all 6 fields must match exactly")

    print("\nBy move number:")
    for move in sorted(by_move.keys()):
        stats = by_move[move]
        move_acc = stats["correct"] / stats["total"] * 100
        print(f"  Move {move}: {stats['correct']}/{stats['total']} ({move_acc:.0f}%)")

    # Focus on problematic move 1
    if 1 in by_move:
        print(f"\n⚠️ Move 1 accuracy: {by_move[1]['correct']}/{by_move[1]['total']} ({by_move[1]['correct']/by_move[1]['total']*100:.0f}%)")

    return accuracy

if __name__ == "__main__":
    # Check if custom model path provided
    model_path = sys.argv[1] if len(sys.argv) > 1 else "jrahn/RookWorld-LM-124M"

    # Run evaluation
    accuracy = evaluate_model(model_path=model_path)

    # Early stopping recommendation
    if accuracy >= 95.0:
        print("\n✅ Model accuracy >= 95% - Consider early stopping!")
    elif accuracy >= 90.0:
        print("\n📈 Model accuracy >= 90% - Significant improvement!")
    else:
        print(f"\n📊 Current accuracy: {accuracy:.1f}% - Training should continue")
