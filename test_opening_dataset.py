#!/usr/bin/env python3
"""
Create and test an opening-focused dataset for A: tasks with pawn moves
"""

import chess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_opening_prompts():
    """Create A: task prompts for problematic opening positions."""

    prompts = []

    # Starting position moves - the problematic ones
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Most common first moves that cause issues
    first_moves = [
        ("e2e4", "e2e4"),  # 1.e4
        ("d2d4", "d2d4"),  # 1.d4
        ("g1f3", "g1f3"),  # 1.Nf3
        ("c2c4", "c2c4"),  # 1.c4 English
        ("e2e3", "e2e3"),  # 1.e3
        ("d2d3", "d2d3"),  # 1.d3
        ("f2f4", "f2f4"),  # 1.f4 Bird's
        ("b2b3", "b2b3"),  # 1.b3 Nimzo-Larsen
    ]

    for move, history in first_moves:
        prompt = f"A: {starting_fen}+{move}+{history}+"
        prompts.append(prompt)

    # Common second moves after 1.e4 e5
    e4e5_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    second_moves_e4 = [
        ("g1f3", "e2e4 e7e5 g1f3"),  # 2.Nf3
        ("f1c4", "e2e4 e7e5 f1c4"),  # 2.Bc4 Italian
        ("f2f4", "e2e4 e7e5 f2f4"),  # 2.f4 King's Gambit
        ("b1c3", "e2e4 e7e5 b1c3"),  # 2.Nc3 Vienna
    ]

    for move, history in second_moves_e4:
        prompt = f"A: {e4e5_fen}+{move}+{history}+"
        prompts.append(prompt)

    # Common second moves after 1.d4 d5
    d4d5_fen = "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2"
    second_moves_d4 = [
        ("c2c4", "d2d4 d7d5 c2c4"),  # 2.c4 Queen's Gambit
        ("g1f3", "d2d4 d7d5 g1f3"),  # 2.Nf3
        ("b1c3", "d2d4 d7d5 b1c3"),  # 2.Nc3
        ("c1f4", "d2d4 d7d5 c1f4"),  # 2.Bf4 London
    ]

    for move, history in second_moves_d4:
        prompt = f"A: {d4d5_fen}+{move}+{history}+"
        prompts.append(prompt)

    return prompts


def test_variance_in_generations():
    """Test if model generates varied outputs with sampling."""

    print("Testing generation variance on opening positions...")
    print("=" * 70)

    # Load model
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

    # Test the most problematic prompt
    prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+e2e4+"

    print(f"Testing prompt: {prompt[:60]}...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate multiple completions with different temperatures
    for temp, top_p, top_k in [(0.6, 0.95, None), (1.0, 0.98, None), (0.8, None, 10)]:
        print(f"\nSampling with temp={temp}, top_p={top_p}, top_k={top_k}:")

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            num_return_sequences=5,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs.input_ids.shape[1]
        completions = []

        for i in range(5):
            completion_tokens = outputs[i][prompt_len:]
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            # Extract just the FEN part
            fen_part = completion.split("+")[0].strip() if "+" in completion else completion
            completions.append(fen_part[:40])  # First 40 chars of FEN

        # Check for uniqueness
        unique_completions = set(completions)

        if len(unique_completions) == 1:
            print(f"  ⚠️ All 5 completions are IDENTICAL: {completions[0]}")
        else:
            print(f"  ✓ {len(unique_completions)}/5 unique completions:")
            for comp in unique_completions:
                count = completions.count(comp)
                print(f"    {comp} (x{count})")


def test_opening_prompts_rewards():
    """Test rewards on opening position dataset."""

    from src.rookworld_trl.rewards import create_reward_function

    prompts = create_opening_prompts()
    reward_fn = create_reward_function()

    print("\n" + "=" * 70)
    print("Testing rewards on opening positions dataset")
    print("=" * 70)

    # Simulate some responses (both correct and incorrect)
    for i, prompt in enumerate(prompts[:4]):
        print(f"\n{i+1}. {prompt[:60]}...")

        # Parse expected FEN
        import re
        match = re.search(r'A:\s*([^+]+)\+([^+]+)\+', prompt)
        if match:
            fen = match.group(1).strip()
            move = match.group(2).strip()

            board = chess.Board(fen)
            board.push(chess.Move.from_uci(move))
            expected_fen = board.fen()

            # Create test responses
            responses = [
                f"{expected_fen}+0.001+0+0",  # Perfect
                f"{expected_fen.replace(' d3 ', ' - ')}+0.001+0+0",  # Missing en passant
                f"{expected_fen.replace('PPP1PPPP', 'PPPPPPPP')}+0.001+0+0",  # Missing pawn removal
            ]

            scores = reward_fn(responses, prompts=[prompt] * len(responses))

            print(f"  Expected: {expected_fen}")
            print(f"  Rewards: Perfect={scores[0]:+.3f}, NoEP={scores[1]:+.3f}, WrongPawn={scores[2]:+.3f}")
            print(f"  Average: {sum(scores)/len(scores):+.3f}")


if __name__ == "__main__":
    # Test variance
    test_variance_in_generations()

    # Test rewards on opening positions
    test_opening_prompts_rewards()

    # Output the prompts for use in training
    prompts = create_opening_prompts()
    print(f"\n✓ Created {len(prompts)} opening position prompts for training")