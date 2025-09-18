#!/usr/bin/env python3
"""
Test the model on problematic opening positions mentioned in PROMPTING_GUIDE.md
Specifically: d2d4 and e2e4 from starting position
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import chess
import Levenshtein

def test_opening_moves():
    # Load model
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

    # Test problematic opening moves
    test_cases = [
        # Starting position with d2d4
        {
            "name": "d2d4 from start",
            "prompt": "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+d2d4+d2d4+",
            "expected_fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
        },
        # Starting position with e2e4
        {
            "name": "e2e4 from start",
            "prompt": "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+e2e4+",
            "expected_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        },
        # After 1.e4 e5, 2.Nf3
        {
            "name": "Nf3 after e4 e5",
            "prompt": "A: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2+g1f3+e2e4 e7e5 g1f3+",
            "expected_fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        },
        # After 1.d4 d5, 2.c4 (Queen's Gambit)
        {
            "name": "c4 Queen's Gambit",
            "prompt": "A: rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2+c2c4+d2d4 d7d5 c2c4+",
            "expected_fen": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2"
        }
    ]

    print(f"\nTesting {len(test_cases)} problematic opening positions...")
    print("=" * 70)

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Prompt: {test['prompt'][:80]}...")

        # Generate completion
        inputs = tokenizer(test['prompt'], return_tensors="pt").to(model.device)

        with torch.no_grad():
            # Test with different sampling strategies
            for sampling_name, sampling_params in [
                ("Greedy", {"do_sample": False}),
                ("Sample (T=0.6)", {"do_sample": True, "temperature": 0.6, "top_k": 5}),
                ("Sample (T=1.0)", {"do_sample": True, "temperature": 1.0, "top_p": 0.98})
            ]:
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **sampling_params
                )

                prompt_len = inputs.input_ids.shape[1]
                completion_tokens = outputs[0][prompt_len:]
                completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

                # Parse generated FEN
                parts = completion.split("+")
                if parts and parts[0].strip():
                    generated_fen = parts[0].strip()

                    # Check exact match
                    exact_match = generated_fen == test['expected_fen']

                    # Calculate Levenshtein distance
                    distance = Levenshtein.distance(test['expected_fen'], generated_fen)

                    # Check if valid FEN
                    try:
                        chess.Board(generated_fen)
                        valid = True
                    except:
                        valid = False

                    print(f"  {sampling_name:15} - Match: {'✓' if exact_match else '✗'}, Valid: {'✓' if valid else '✗'}, Distance: {distance}")

                    if not exact_match:
                        print(f"    Expected: {test['expected_fen']}")
                        print(f"    Got:      {generated_fen}")

                    results.append({
                        "test": test['name'],
                        "sampling": sampling_name,
                        "exact_match": exact_match,
                        "valid": valid,
                        "distance": distance
                    })
                else:
                    print(f"  {sampling_name:15} - Failed to generate FEN")
                    results.append({
                        "test": test['name'],
                        "sampling": sampling_name,
                        "exact_match": False,
                        "valid": False,
                        "distance": 999
                    })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for sampling in ["Greedy", "Sample (T=0.6)", "Sample (T=1.0)"]:
        sampling_results = [r for r in results if r['sampling'] == sampling]
        exact_matches = sum(1 for r in sampling_results if r['exact_match'])
        valid_fens = sum(1 for r in sampling_results if r['valid'])
        avg_distance = sum(r['distance'] for r in sampling_results) / len(sampling_results)

        print(f"\n{sampling}:")
        print(f"  Exact matches: {exact_matches}/{len(sampling_results)} ({exact_matches/len(sampling_results)*100:.0f}%)")
        print(f"  Valid FENs:    {valid_fens}/{len(sampling_results)} ({valid_fens/len(sampling_results)*100:.0f}%)")
        print(f"  Avg distance:  {avg_distance:.2f}")

if __name__ == "__main__":
    test_opening_moves()