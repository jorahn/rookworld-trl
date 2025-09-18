#!/usr/bin/env python3
"""
Create a proper opening position dataset for A: tasks by playing random legal moves
from the starting position, ensuring we capture move history correctly.
"""

import chess
import random
import json

def generate_opening_positions(num_positions=100, max_moves=15, seed=42):
    """
    Generate opening positions by playing random legal moves from the starting position.

    Returns list of A: task prompts with proper move history.
    """
    random.seed(seed)
    prompts = []

    for game_idx in range(num_positions):
        board = chess.Board()
        move_history = []

        # Play random number of moves (1 to max_moves)
        num_moves = random.randint(1, max_moves)

        for move_num in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Bias towards pawn moves in the opening (50% chance)
            pawn_moves = [m for m in legal_moves if board.piece_at(m.from_square) and
                         board.piece_at(m.from_square).piece_type == chess.PAWN]

            if pawn_moves and random.random() < 0.5:
                move = random.choice(pawn_moves)
            else:
                move = random.choice(legal_moves)

            # Store the state BEFORE the move for A: task
            fen_before = board.fen()
            move_uci = move.uci()

            # Play the move
            board.push(move)
            move_history.append(move_uci)

            # Create A: task prompt
            # Format: A: [state_before]+[move]+[move_history_including_current]+
            # The move history should have max 10 moves with current move as LAST
            history_for_prompt = move_history[-10:]  # Last 10 moves including current
            history_str = " ".join(history_for_prompt)

            prompt = f"A: {fen_before}+{move_uci}+{history_str}+"

            # Store prompt with metadata
            prompts.append({
                "prompt": prompt,
                "move_num": move_num + 1,
                "expected_fen": board.fen(),
                "move": move_uci,
                "is_pawn_move": move in pawn_moves if pawn_moves else False
            })

    return prompts

def analyze_dataset(prompts):
    """Analyze the generated dataset."""
    print("Dataset Analysis:")
    print("=" * 50)
    print(f"Total prompts: {len(prompts)}")

    # Move number distribution
    move_counts = {}
    pawn_moves = 0
    starting_pos = 0

    for p in prompts:
        move_num = p["move_num"]
        if move_num not in move_counts:
            move_counts[move_num] = 0
        move_counts[move_num] += 1

        if p["is_pawn_move"]:
            pawn_moves += 1

        if "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq" in p["prompt"]:
            starting_pos += 1

    print(f"\nMove distribution:")
    for move_num in sorted(move_counts.keys()):
        print(f"  Move {move_num}: {move_counts[move_num]} positions")

    print(f"\nPawn moves: {pawn_moves}/{len(prompts)} ({pawn_moves*100/len(prompts):.1f}%)")
    print(f"From starting position: {starting_pos}")

    # Sample some prompts
    print("\nSample prompts:")
    for i, p in enumerate(prompts[:3]):
        print(f"\n{i+1}. Move {p['move_num']}: {p['move']}")
        print(f"   Prompt: {p['prompt'][:80]}...")
        print(f"   Expected: {p['expected_fen'][:60]}...")

def save_dataset(prompts, filename="opening_dataset.json"):
    """Save dataset to JSON file."""
    with open(filename, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"\nSaved {len(prompts)} prompts to {filename}")

def create_focused_problem_set():
    """Create a small focused set of the most problematic moves."""
    problem_prompts = []

    # The specific problematic moves from starting position
    board = chess.Board()
    starting_fen = board.fen()

    problematic_first_moves = [
        ("e2e4", "King's pawn"),
        ("d2d4", "Queen's pawn"),
        ("c2c4", "English Opening"),
        ("g1f3", "Reti Opening"),
    ]

    for move_uci, name in problematic_first_moves:
        move = chess.Move.from_uci(move_uci)
        board_copy = chess.Board()

        # First move
        prompt = f"A: {starting_fen}+{move_uci}+{move_uci}+"
        board_copy.push(move)
        expected = board_copy.fen()

        problem_prompts.append({
            "prompt": prompt,
            "move_num": 1,
            "expected_fen": expected,
            "move": move_uci,
            "name": name,
            "is_pawn_move": move_uci[0] in 'abcdefgh'
        })

        # Common second moves
        if move_uci == "e2e4":
            # After 1.e4 e5
            board_copy.push(chess.Move.from_uci("e7e5"))
            fen_after_e5 = board_copy.fen()

            second_moves = [("g1f3", "Italian prep"), ("f2f4", "King's Gambit")]
            for second_move, second_name in second_moves:
                prompt = f"A: {fen_after_e5}+{second_move}+e2e4 e7e5 {second_move}+"
                board_temp = board_copy.copy()
                board_temp.push(chess.Move.from_uci(second_move))

                problem_prompts.append({
                    "prompt": prompt,
                    "move_num": 3,
                    "expected_fen": board_temp.fen(),
                    "move": second_move,
                    "name": f"{name} → {second_name}",
                    "is_pawn_move": second_move[0] in 'abcdefgh'
                })

    return problem_prompts

if __name__ == "__main__":
    # Generate main dataset
    print("Generating opening positions dataset...")
    prompts = generate_opening_positions(num_positions=200, max_moves=10, seed=42)

    # Analyze
    analyze_dataset(prompts)

    # Save
    save_dataset(prompts, "opening_dataset.json")

    # Create focused problem set
    print("\n" + "=" * 50)
    print("Creating focused problem set...")
    problem_prompts = create_focused_problem_set()

    print(f"Created {len(problem_prompts)} problematic positions:")
    for p in problem_prompts:
        print(f"  - {p['name']}: {p['move']}")

    save_dataset(problem_prompts, "problem_opening_dataset.json")