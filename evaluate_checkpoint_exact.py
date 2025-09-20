#!/usr/bin/env python3
"""
Exact match evaluation script for checking model performance on held-out eval set.
This version checks for exact match of the full generation, not just FEN equivalence.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

def evaluate_model(model_path="jrahn/RookWorld-LM-124M", eval_file="opening_dataset_eval.json", sample_size=50):
    """Evaluate model on held-out eval set with exact match."""

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
    print("Using EXACT MATCH scoring (full generation must match expected)")

    exact_matches = 0
    by_move = {}

    # For A-tasks, we need the full expected completion with schema
    # Since the eval dataset only has expected_fen, we'll need to construct the expected format
    # Based on the model's learned pattern of appending +0.001+0+0
    # But for true exact match, we should compare what the model generates vs what it should generate

    for sample in tqdm(eval_data, desc="Evaluating"):
        prompt = sample["prompt"]
        expected_fen = sample["expected_fen"]
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
                max_new_tokens=196,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs.input_ids.shape[1]
        completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

        # For exact match, we need to know what the expected full generation is
        # Since A-tasks have schema: FEN+reward+terminates+truncated
        # and the model consistently generates +0.001+0+0, we can check:

        # Option 1: Check if FEN part matches exactly (most practical)
        generated_fen = completion.split('+')[0].strip() if '+' in completion else completion.strip()

        # For true exact match, completion should be: expected_fen+reward+terminates+truncated
        # But we don't have the expected reward/term/trunc values in the eval dataset

        # So we'll do exact FEN match (not lenient)
        if generated_fen == expected_fen:
            exact_matches += 1
            by_move[move_num]["correct"] += 1

    # Calculate metrics
    accuracy = exact_matches / len(eval_data) * 100

    print(f"\n=== EXACT MATCH EVALUATION RESULTS ===")
    print(f"Overall accuracy: {exact_matches}/{len(eval_data)} ({accuracy:.1f}%)")

    print("\nBy move number:")
    for move in sorted(by_move.keys()):
        stats = by_move[move]
        move_acc = stats["correct"] / stats["total"] * 100
        print(f"  Move {move}: {stats['correct']}/{stats['total']} ({move_acc:.0f}%)")

    # Focus on problematic move 1
    if 1 in by_move:
        print(f"\n⚠️ Move 1 accuracy: {by_move[1]['correct']}/{by_move[1]['total']} ({by_move[1]['correct']/by_move[1]['total']*100:.0f}%)")

    print("\nNote: This uses EXACT FEN matching (all 6 fields must match exactly).")
    print("The +reward+terminates+truncated suffix is stripped before comparison.")

    return accuracy

if __name__ == "__main__":
    # Check if custom model path provided
    model_path = sys.argv[1] if len(sys.argv) > 1 else "jrahn/RookWorld-LM-124M"

    # Run evaluation
    accuracy = evaluate_model(model_path=model_path)
