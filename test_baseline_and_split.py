#!/usr/bin/env python3
"""
Test baseline model performance on opening dataset and split into train/eval sets.
"""

import json
import torch
import chess
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def test_baseline_performance(dataset_file="opening_dataset.json", sample_size=100):
    """Test model performance on opening positions."""

    print("Loading model...")
    model_name = "jrahn/RookWorld-LM-124M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset from {dataset_file}...")
    with open(dataset_file) as f:
        dataset = json.load(f)

    # Sample subset for testing
    test_samples = random.sample(dataset, min(sample_size, len(dataset)))

    print(f"Testing on {len(test_samples)} samples...")

    results = {
        "exact_match": 0,
        "board_correct": 0,
        "en_passant_correct": 0,
        "missing_en_passant": 0,
        "total": len(test_samples),
        "by_move": {}
    }

    for sample in tqdm(test_samples):
        prompt = sample["prompt"]
        expected_fen = sample["expected_fen"]
        move_num = sample["move_num"]

        # Initialize move stats
        if move_num not in results["by_move"]:
            results["by_move"][move_num] = {
                "exact_match": 0,
                "board_correct": 0,
                "total": 0
            }

        results["by_move"][move_num]["total"] += 1

        # Generate completion (greedy for consistency)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs.input_ids.shape[1]
        completion_tokens = outputs[0][prompt_len:]
        completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

        # Parse generated FEN
        parts = completion.split("+")
        if parts and parts[0].strip():
            generated_fen = parts[0].strip()

            # Check exact match
            if generated_fen == expected_fen:
                results["exact_match"] += 1
                results["by_move"][move_num]["exact_match"] += 1

            # Check board position only
            gen_board = generated_fen.split()[0] if generated_fen.split() else ""
            exp_board = expected_fen.split()[0]

            if gen_board == exp_board:
                results["board_correct"] += 1
                results["by_move"][move_num]["board_correct"] += 1

            # Check en passant specifically
            if len(generated_fen.split()) >= 4 and len(expected_fen.split()) >= 4:
                gen_ep = generated_fen.split()[3]
                exp_ep = expected_fen.split()[3]

                if exp_ep != "-" and exp_ep in ["e3", "d3", "e6", "d6", "c3", "c6", "f3", "f6"]:
                    # Expected en passant
                    if gen_ep == exp_ep:
                        results["en_passant_correct"] += 1
                    else:
                        results["missing_en_passant"] += 1

    return results

def split_dataset(dataset_file="opening_dataset.json", train_ratio=0.8, seed=42):
    """Split dataset into train and eval sets."""

    random.seed(seed)

    with open(dataset_file) as f:
        dataset = json.load(f)

    # Shuffle
    random.shuffle(dataset)

    # Split
    split_idx = int(len(dataset) * train_ratio)
    train_set = dataset[:split_idx]
    eval_set = dataset[split_idx:]

    # Save splits
    with open("opening_dataset_train.json", "w") as f:
        json.dump(train_set, f, indent=2)

    with open("opening_dataset_eval.json", "w") as f:
        json.dump(eval_set, f, indent=2)

    print(f"Split dataset: {len(train_set)} train, {len(eval_set)} eval")

    # Analyze splits
    for name, data in [("Train", train_set), ("Eval", eval_set)]:
        move_counts = {}
        pawn_moves = 0
        starting_pos = 0

        for sample in data:
            move_num = sample["move_num"]
            if move_num not in move_counts:
                move_counts[move_num] = 0
            move_counts[move_num] += 1

            if sample.get("is_pawn_move"):
                pawn_moves += 1

            if "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq" in sample["prompt"]:
                starting_pos += 1

        print(f"\n{name} set:")
        print(f"  Total: {len(data)}")
        print(f"  From starting position: {starting_pos}")
        print(f"  Pawn moves: {pawn_moves}")
        print(f"  Move distribution: {sorted(move_counts.items())[:5]}...")

def main():
    # Test baseline
    print("=" * 60)
    print("BASELINE PERFORMANCE TEST")
    print("=" * 60)

    results = test_baseline_performance("opening_dataset.json", sample_size=50)

    print("\nResults:")
    print(f"  Exact match: {results['exact_match']}/{results['total']} ({results['exact_match']/results['total']*100:.1f}%)")
    print(f"  Board correct: {results['board_correct']}/{results['total']} ({results['board_correct']/results['total']*100:.1f}%)")

    if results["missing_en_passant"] > 0 or results["en_passant_correct"] > 0:
        ep_total = results["missing_en_passant"] + results["en_passant_correct"]
        print(f"  En passant: {results['en_passant_correct']}/{ep_total} correct ({results['en_passant_correct']/ep_total*100:.1f}%)")

    print("\nBy move number:")
    for move_num in sorted(results["by_move"].keys()):
        stats = results["by_move"][move_num]
        print(f"  Move {move_num}: {stats['exact_match']}/{stats['total']} exact ({stats['exact_match']/stats['total']*100:.0f}%), "
              f"{stats['board_correct']}/{stats['total']} board ({stats['board_correct']/stats['total']*100:.0f}%)")

    # Test on problem set
    print("\n" + "=" * 60)
    print("PROBLEM SET PERFORMANCE")
    print("=" * 60)

    problem_results = test_baseline_performance("problem_opening_dataset.json", sample_size=10)

    print(f"\nProblem set results:")
    print(f"  Exact match: {problem_results['exact_match']}/{problem_results['total']} ({problem_results['exact_match']/problem_results['total']*100:.1f}%)")
    print(f"  Board correct: {problem_results['board_correct']}/{problem_results['total']} ({problem_results['board_correct']/problem_results['total']*100:.1f}%)")

    # Split dataset if performance is low enough to warrant training
    exact_match_rate = results['exact_match'] / results['total']

    if exact_match_rate < 0.5:  # Less than 50% exact match
        print("\n" + "=" * 60)
        print("SPLITTING DATASET")
        print("=" * 60)
        print(f"Exact match rate {exact_match_rate*100:.1f}% is low enough to warrant training")
        split_dataset("opening_dataset.json")
    else:
        print(f"\n⚠️ Exact match rate {exact_match_rate*100:.1f}% is quite high - model may not need much training")

if __name__ == "__main__":
    main()