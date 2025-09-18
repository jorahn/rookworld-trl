#!/usr/bin/env python3
"""Analyze patterns in how generated FEN and raw completions change during training."""

import json
from pathlib import Path
from collections import Counter
import re

def load_step_data(step_file: Path):
    """Load JSONL data from a step file."""
    data = []
    with open(step_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_completion_patterns(completions):
    """Analyze patterns in raw completions."""
    patterns = {
        'has_valid_suffix': 0,  # Ends with +X.XXX+X+X format
        'has_spaces': 0,
        'has_multiple_slashes': 0,
        'abnormal_length': 0,
        'character_distribution': Counter(),
        'starts_with_valid_rank': 0,
        'ends_with_move_count': 0,
    }

    suffix_pattern = r'\+[\d\.]+\+\d+\+\d+$'
    valid_rank_pattern = r'^[rnbqkpRNBQKP1-8]+/'

    for comp in completions:
        if re.search(suffix_pattern, comp):
            patterns['has_valid_suffix'] += 1

        if ' ' in comp and '+' not in comp:  # Spaces not in the suffix
            patterns['has_spaces'] += 1

        if comp.count('/') > 10:
            patterns['has_multiple_slashes'] += 1

        if len(comp) > 150 or len(comp) < 50:
            patterns['abnormal_length'] += 1

        for char in comp:
            patterns['character_distribution'][char] += 1

        if re.match(valid_rank_pattern, comp):
            patterns['starts_with_valid_rank'] += 1

        if re.search(r'\d+$', comp.split('+')[0] if '+' in comp else comp):
            patterns['ends_with_move_count'] += 1

    return patterns

def main():
    log_dir = Path('logs/manual_debug_eval_predictions')
    step_files = sorted(log_dir.glob('step_*.jsonl'))

    print("="*80)
    print("OUTPUT PATTERN ANALYSIS ACROSS TRAINING STEPS")
    print("="*80)

    for step_file in step_files:
        step_num = int(step_file.stem.split('_')[1])
        data = load_step_data(step_file)

        raw_completions = [d.get('raw_completion', '') for d in data]
        generated_fens = [d.get('generated_fen', '') for d in data]

        print(f"\n--- Step {step_num} ---")

        # Analyze raw completions
        comp_patterns = analyze_completion_patterns(raw_completions)
        total = len(raw_completions)

        print(f"Raw Completion Patterns:")
        print(f"  Has valid suffix (+X.XXX+X+X): {comp_patterns['has_valid_suffix']}/{total} ({comp_patterns['has_valid_suffix']/total*100:.1f}%)")
        print(f"  Has spaces (not in suffix): {comp_patterns['has_spaces']}/{total} ({comp_patterns['has_spaces']/total*100:.1f}%)")
        print(f"  Too many slashes (>10): {comp_patterns['has_multiple_slashes']}/{total} ({comp_patterns['has_multiple_slashes']/total*100:.1f}%)")
        print(f"  Abnormal length: {comp_patterns['abnormal_length']}/{total} ({comp_patterns['abnormal_length']/total*100:.1f}%)")
        print(f"  Starts with valid rank: {comp_patterns['starts_with_valid_rank']}/{total} ({comp_patterns['starts_with_valid_rank']/total*100:.1f}%)")

        # Most common non-standard characters
        char_dist = comp_patterns['character_distribution']
        standard_chars = set('rnbqkpRNBQKP1234567890/-+ wKQkq.')
        non_standard = {c: count for c, count in char_dist.items() if c not in standard_chars}

        if non_standard:
            print(f"  Non-standard characters found:")
            for char, count in sorted(non_standard.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    '{char}': {count} occurrences")

        # Sample problematic outputs
        print(f"\nSample outputs (first 3):")
        for i, comp in enumerate(raw_completions[:3], 1):
            if len(comp) > 100:
                comp = comp[:100] + "..."
            print(f"  {i}: {comp}")

        # Check for specific degradation patterns
        if step_num > 0:
            print(f"\n  Degradation indicators:")
            if comp_patterns['has_valid_suffix'] < total * 0.5:
                print(f"    ⚠️ Lost suffix format (only {comp_patterns['has_valid_suffix']}/{total})")
            if comp_patterns['has_spaces'] > total * 0.2:
                print(f"    ⚠️ Excessive spaces in output ({comp_patterns['has_spaces']}/{total})")
            if comp_patterns['starts_with_valid_rank'] < total * 0.5:
                print(f"    ⚠️ Lost valid FEN structure ({comp_patterns['starts_with_valid_rank']}/{total})")

    # Compare first and last step
    print("\n" + "="*80)
    print("DEGRADATION SUMMARY")
    print("="*80)

    first_data = load_step_data(step_files[0])
    last_data = load_step_data(step_files[-1])

    first_comps = [d.get('raw_completion', '') for d in first_data]
    last_comps = [d.get('raw_completion', '') for d in last_data]

    first_patterns = analyze_completion_patterns(first_comps)
    last_patterns = analyze_completion_patterns(last_comps)

    print(f"\nKey metric changes (Step 0 → Step {len(step_files)-1}):")
    print(f"  Valid suffix: {first_patterns['has_valid_suffix']}/{len(first_comps)} → {last_patterns['has_valid_suffix']}/{len(last_comps)}")
    print(f"  Valid rank start: {first_patterns['starts_with_valid_rank']}/{len(first_comps)} → {last_patterns['starts_with_valid_rank']}/{len(last_comps)}")
    print(f"  Has spaces: {first_patterns['has_spaces']}/{len(first_comps)} → {last_patterns['has_spaces']}/{len(last_comps)}")

    # Average length change
    first_avg_len = sum(len(c) for c in first_comps) / len(first_comps)
    last_avg_len = sum(len(c) for c in last_comps) / len(last_comps)
    print(f"  Average length: {first_avg_len:.1f} → {last_avg_len:.1f} chars")

if __name__ == "__main__":
    main()