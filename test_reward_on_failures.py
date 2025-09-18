#!/usr/bin/env python3
"""
Test that our reward function properly penalizes the opening move failures.
"""

from src.rookworld_trl.rewards import create_reward_function

def test_reward_on_failures():
    """Test reward function on actual model failures from opening positions."""

    reward_fn = create_reward_function()

    # Test cases from our actual model outputs
    test_cases = [
        {
            "name": "d2d4 - Extra king (greedy)",
            "prompt": "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+d2d4+d2d4+",
            "response": "rnbqkbnr/pppppppp/8/8/3PP3/8/PPPPPP/RNBQKBNRK1 b KQkq - 0 1+0.001+0+0",
            "expected": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
        },
        {
            "name": "d2d4 - Missing pawn removal (T=0.6)",
            "prompt": "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+d2d4+d2d4+",
            "response": "rnbqkbnr/pppppppp/8/8/3P4/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1+0.001+0+0",
            "expected": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
        },
        {
            "name": "e2e4 - Missing en passant (greedy)",
            "prompt": "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+e2e4+",
            "response": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2+0.001+0+0",
            "expected": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        },
        {
            "name": "e2e4 - Completely wrong (T=1.0)",
            "prompt": "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+e2e4+",
            "response": "rnbqkbnr/pppppppb1/3p4/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 2+0.001+0+0",
            "expected": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        },
        {
            "name": "Perfect response (for comparison)",
            "prompt": "A: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2+g1f3+e2e4 e7e5 g1f3+",
            "response": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2+0.001+0+0",
            "expected": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        }
    ]

    print("Testing reward function on actual model failures:")
    print("=" * 70)

    for test in test_cases:
        # Get reward
        scores = reward_fn([test["response"]], prompts=[test["prompt"]])
        reward = scores[0]

        # Determine if this is a failure case
        response_fen = test["response"].split("+")[0].strip()
        is_exact = response_fen == test["expected"]

        print(f"\n{test['name']}:")
        print(f"  Exact match: {is_exact}")
        print(f"  Reward: {reward:+.3f}")

        if is_exact:
            if reward < 0:
                print(f"  ❌ ERROR: Perfect response got negative reward!")
        else:
            if reward >= 0:
                print(f"  ❌ ERROR: Incorrect response got positive reward!")
            else:
                print(f"  ✓ Correctly penalized")

    print("\n" + "=" * 70)
    print("SUMMARY:")

    # Generate multiple completions to see reward distribution
    print("\nTesting with multiple generations (sampling variation):")

    # d2d4 from start - generate variations
    prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+d2d4+d2d4+"

    # Simulate various model outputs (from our test results)
    variations = [
        "rnbqkbnr/pppppppp/8/8/3PP3/8/PPPPPP/RNBQKBNRK1 b KQkq - 0 1+0.001+0+0",  # Extra king
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1+0.001+0+0",    # Missing pawn removal
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1+0.001+0+0",    # Missing en passant
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1+0.001+0+0",   # Perfect!
        "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2+0.001+0+0",   # Extra P in position
    ]

    prompts = [prompt] * len(variations)
    scores = reward_fn(variations, prompts=prompts)

    print(f"\nRewards for d2d4 variations (n={len(variations)}):")
    for i, (var, score) in enumerate(zip(variations, scores)):
        fen = var.split("+")[0].strip()
        print(f"  {i+1}. {fen[:40]}... → {score:+.3f}")

    avg_reward = sum(scores) / len(scores)
    positive_rewards = sum(1 for s in scores if s > 0)
    negative_rewards = sum(1 for s in scores if s < 0)

    print(f"\nDistribution:")
    print(f"  Average reward: {avg_reward:+.3f}")
    print(f"  Positive rewards: {positive_rewards}/{len(scores)} ({positive_rewards/len(scores)*100:.0f}%)")
    print(f"  Negative rewards: {negative_rewards}/{len(scores)} ({negative_rewards/len(scores)*100:.0f}%)")

    if avg_reward < 0:
        print(f"  ✓ Group gets negative average reward - GRPO will learn to avoid these!")
    else:
        print(f"  ❌ Group gets positive average - might reinforce errors!")

if __name__ == "__main__":
    test_reward_on_failures()