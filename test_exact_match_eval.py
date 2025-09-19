#!/usr/bin/env python3
"""Test exact match evaluation on sample completions."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def load_eval_dataset():
    """Load the evaluation dataset."""
    eval_path = Path("opening_dataset_eval.json")
    with open(eval_path, "r") as f:
        data = json.load(f)
    return data

def generate_completion(model, tokenizer, prompt, max_new_tokens=128):
    """Generate a completion for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.01,  # Near-deterministic
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (exclude prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion.strip()

def exact_match(pred, expected):
    """Check if prediction exactly matches expected (after stripping whitespace)."""
    return pred.strip() == expected.strip()

def main():
    # Load model and tokenizer
    model_name = "jrahn/RookWorld-LM-124M"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    # Load eval dataset
    eval_data = load_eval_dataset()

    # Sample first 20 items for detailed analysis
    sample_size = min(20, len(eval_data))
    samples = eval_data[:sample_size]

    exact_matches = 0
    results = []

    print(f"\nGenerating completions for {sample_size} samples...\n")
    print("=" * 80)

    for i, item in enumerate(samples):
        prompt = item["prompt"]
        expected = item["expected_fen"]  # For A-tasks, the expected completion is the FEN

        # Generate completion
        generated = generate_completion(model, tokenizer, prompt)

        # Check exact match
        is_exact = exact_match(generated, expected)
        exact_matches += is_exact

        # Store result
        results.append({
            "index": i,
            "prompt": prompt,
            "expected": expected,
            "generated": generated,
            "exact_match": is_exact
        })

        # Print detailed comparison
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Expected:  '{expected}'")
        print(f"Generated: '{generated}'")
        print(f"Exact Match: {is_exact}")

        # If not exact, show differences
        if not is_exact:
            if len(expected) != len(generated):
                print(f"  Length diff: expected {len(expected)}, got {len(generated)}")

            # Character-level comparison for short strings
            if len(expected) < 100 and len(generated) < 100:
                min_len = min(len(expected), len(generated))
                for j in range(min_len):
                    if expected[j] != generated[j]:
                        print(f"  First diff at position {j}: expected '{expected[j]}', got '{generated[j]}'")
                        break

        print("-" * 40)

    # Summary statistics
    accuracy = exact_matches / sample_size * 100
    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"Exact Match Accuracy: {exact_matches}/{sample_size} = {accuracy:.1f}%")

    # Analyze error patterns
    print(f"\n{'=' * 80}")
    print("ERROR ANALYSIS:")

    # Categorize errors
    errors = [r for r in results if not r["exact_match"]]
    if errors:
        print(f"\nFound {len(errors)} mismatches. Analyzing patterns...")

        # Check for common patterns
        fen_format_errors = 0
        move_format_errors = 0
        length_mismatches = 0
        minor_differences = 0

        for err in errors:
            exp = err["expected"]
            gen = err["generated"]

            # Check if it's a FEN (contains slashes)
            if "/" in exp:
                # FEN response
                if "/" not in gen:
                    fen_format_errors += 1
                elif len(exp.split()) != len(gen.split()):
                    fen_format_errors += 1
                else:
                    # Check if only counters differ
                    exp_parts = exp.split()
                    gen_parts = gen.split()
                    if len(exp_parts) >= 4 and len(gen_parts) >= 4:
                        if exp_parts[:4] == gen_parts[:4]:
                            minor_differences += 1
                        else:
                            fen_format_errors += 1
            elif len(exp) != len(gen):
                length_mismatches += 1
            else:
                # Might be a move
                if exp and gen and (exp[0] != gen[0] or exp[-1] != gen[-1]):
                    move_format_errors += 1
                else:
                    minor_differences += 1

        print(f"  FEN format errors: {fen_format_errors}")
        print(f"  Move format errors: {move_format_errors}")
        print(f"  Length mismatches: {length_mismatches}")
        print(f"  Minor differences: {minor_differences}")

        # Show a few specific error examples
        print(f"\nDetailed error examples (first 5):")
        for err in errors[:5]:
            print(f"\n  Index {err['index']}:")
            print(f"    Expected:  '{err['expected']}'")
            print(f"    Generated: '{err['generated']}'")
    else:
        print("No errors found - all samples matched exactly!")

    # Save results for further analysis
    with open("exact_match_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to exact_match_eval_results.json")

if __name__ == "__main__":
    main()