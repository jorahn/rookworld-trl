#!/usr/bin/env python3
"""Analyze evaluation errors considering the A-task schema."""

import json

def extract_fen_from_generation(gen):
    """Extract just the FEN part from the generation (before the +reward+term+trunc)."""
    # Split on '+' and take the first part
    parts = gen.split('+')
    if parts:
        return parts[0].strip()
    return gen.strip()

def main():
    # Load results
    with open('exact_match_eval_results.json') as f:
        results = json.load(f)

    print("=" * 80)
    print("ANALYSIS CONSIDERING A-TASK SCHEMA (FEN+reward+terminates+truncated)")
    print("=" * 80)

    # Check FEN accuracy (just the position part)
    fen_matches = 0
    position_only_matches = 0
    errors_by_type = {
        'position_wrong': [],
        'counters_only_wrong': [],
        'malformed': []
    }

    for r in results:
        expected_fen = r['expected']
        generated = r['generated']
        generated_fen = extract_fen_from_generation(generated)

        # Exact FEN match
        if generated_fen == expected_fen:
            fen_matches += 1
        else:
            # Check if just position is correct (first 4 fields)
            exp_parts = expected_fen.split()
            gen_parts = generated_fen.split()

            if len(exp_parts) >= 4 and len(gen_parts) >= 4:
                if exp_parts[:4] == gen_parts[:4]:
                    position_only_matches += 1
                    errors_by_type['counters_only_wrong'].append(r['index'])
                else:
                    errors_by_type['position_wrong'].append(r['index'])
            else:
                errors_by_type['malformed'].append(r['index'])

    n = len(results)
    print(f"\n1. FEN ACCURACY (excluding +reward+term+trunc suffix):")
    print(f"   - Exact FEN match: {fen_matches}/{n} = {fen_matches/n*100:.1f}%")
    print(f"   - Position correct (counters wrong): {position_only_matches}/{n} = {position_only_matches/n*100:.1f}%")
    print(f"   - Total position accuracy: {(fen_matches + position_only_matches)/n*100:.1f}%")

    print(f"\n2. ERROR BREAKDOWN:")
    print(f"   - Position wrong: {len(errors_by_type['position_wrong'])} samples")
    print(f"   - Counters only wrong: {len(errors_by_type['counters_only_wrong'])} samples")
    print(f"   - Malformed FEN: {len(errors_by_type['malformed'])} samples")

    # Show examples of position errors
    if errors_by_type['position_wrong']:
        print(f"\n3. POSITION ERRORS (first 5):")
        for idx in errors_by_type['position_wrong'][:5]:
            r = results[idx]
            expected_fen = r['expected']
            generated_fen = extract_fen_from_generation(r['generated'])

            print(f"\n   Sample {idx}:")
            print(f"   Expected:  {expected_fen}")
            print(f"   Generated: {generated_fen}")

            # Show position difference
            exp_pos = expected_fen.split()[0] if expected_fen.split() else ""
            gen_pos = generated_fen.split()[0] if generated_fen.split() else ""
            if exp_pos != gen_pos:
                print(f"   Position diff:")
                print(f"     Expected:  {exp_pos}")
                print(f"     Generated: {gen_pos}")

    # Check for exact match including the schema suffix
    print(f"\n4. FULL GENERATION ANALYSIS:")

    # Check if all have the same suffix pattern
    suffix_pattern = "+0.001+0+0"
    all_have_suffix = all(r['generated'].endswith(suffix_pattern) for r in results)

    if all_have_suffix:
        print(f"   All generations end with '{suffix_pattern}'")
        print(f"   This appears to be: +reward+terminates+truncated")
        print(f"   The model has learned this schema consistently")

    # For true exact match, we'd need to know the expected reward/term/trunc values
    print(f"\n5. EXACT MATCH ACCURACY (full generation):")
    print(f"   Since the eval dataset only has expected FEN, not reward/term/trunc,")
    print(f"   we cannot measure true exact match for the full A-task schema.")
    print(f"   The model is correctly following the schema format though.")

    print(f"\n6. RECOMMENDATION:")
    print(f"   - For A-task evaluation, compare only the FEN portion")
    print(f"   - Current FEN accuracy: {fen_matches/n*100:.1f}%")
    print(f"   - Position accuracy: {(fen_matches + position_only_matches)/n*100:.1f}%")
    print(f"   - The evaluation should either:")
    print(f"     a) Strip the +reward+term+trunc suffix before comparison, OR")
    print(f"     b) Include expected reward/term/trunc values in the eval dataset")

if __name__ == "__main__":
    main()