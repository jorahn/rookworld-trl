#!/usr/bin/env python3
"""
Quick evaluation script for checking model performance on held-out eval set.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

def evaluate_model(model_path="jrahn/RookWorld-LM-124M", eval_file="opening_dataset_eval.json", sample_size=50):
    """Evaluate model on held-out eval set."""

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

    for sample in tqdm(eval_data, desc="Evaluating"):
        prompt = sample["prompt"]
        expected = sample["expected_fen"]
        move_num = sample["move_num"]

        if move_num not in by_move:
            by_move[move_num] = {"correct": 0, "total": 0}
        by_move[move_num]["total"] += 1

        # Generate with greedy decoding
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

    # Calculate metrics
    accuracy = exact_matches / len(eval_data) * 100

    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Overall accuracy: {exact_matches}/{len(eval_data)} ({accuracy:.1f}%)")

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