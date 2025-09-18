#!/usr/bin/env python3
"""Analyze training progression from step-by-step JSONL evaluation files."""

import json
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd

def load_step_data(step_file: Path) -> List[Dict]:
    """Load JSONL data from a step file."""
    data = []
    with open(step_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_fen_quality(fen: str) -> Dict:
    """Analyze the quality of a generated FEN string."""
    # Standard FEN pattern
    fen_pattern = r'^[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+\/[rnbqkpRNBQKP1-8]+ [wb] [-KQkq]+ [-a-h1-8]+ \d+ \d+$'

    analysis = {
        'valid_structure': bool(re.match(fen_pattern, fen)),
        'has_spaces': ' ' in fen.split()[0] if fen else False,
        'has_invalid_chars': bool(re.search(r'[^rnbqkpRNBQKP1-8/\s\-]', fen.split()[0] if fen else '')),
        'length': len(fen) if fen else 0,
        'num_slashes': fen.count('/') if fen else 0,
    }

    return analysis

def compute_step_metrics(data: List[Dict]) -> Dict:
    """Compute metrics for a single step."""
    total = len(data)
    matches = sum(1 for d in data if d.get('match', False))

    fen_analyses = [analyze_fen_quality(d.get('generated_fen', '')) for d in data]

    # Analyze raw completion patterns
    completion_lengths = [len(d.get('raw_completion', '')) for d in data]

    metrics = {
        'total_samples': total,
        'matches': matches,
        'accuracy': matches / total if total > 0 else 0,
        'avg_completion_length': sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0,
        'valid_fen_structure': sum(1 for a in fen_analyses if a['valid_structure']) / total if total > 0 else 0,
        'has_invalid_chars': sum(1 for a in fen_analyses if a['has_invalid_chars']) / total if total > 0 else 0,
        'correct_slash_count': sum(1 for a in fen_analyses if a['num_slashes'] == 7) / total if total > 0 else 0,
    }

    # Sample some raw completions for manual inspection
    metrics['sample_completions'] = [d.get('raw_completion', '')[:100] for d in data[:3]]

    return metrics

def main():
    log_dir = Path('logs/manual_debug_eval_predictions')

    if not log_dir.exists():
        print(f"Directory {log_dir} not found!")
        return

    step_files = sorted(log_dir.glob('step_*.jsonl'))

    results = []
    for step_file in step_files:
        step_num = int(step_file.stem.split('_')[1])
        data = load_step_data(step_file)
        metrics = compute_step_metrics(data)
        metrics['step'] = step_num
        results.append(metrics)

    # Sort by step number
    results = sorted(results, key=lambda x: x['step'])

    # Print summary table
    print("\n" + "="*80)
    print("TRAINING PROGRESSION ANALYSIS")
    print("="*80)

    # Create DataFrame for better display
    df_data = []
    for r in results:
        df_data.append({
            'Step': r['step'],
            'Accuracy': f"{r['accuracy']:.1%}",
            'Matches': f"{r['matches']}/{r['total_samples']}",
            'Avg_Len': f"{r['avg_completion_length']:.0f}",
            'Valid_FEN': f"{r['valid_fen_structure']:.1%}",
            'Invalid_Chars': f"{r['has_invalid_chars']:.1%}",
            'Correct_Slashes': f"{r['correct_slash_count']:.1%}",
        })

    df = pd.DataFrame(df_data)
    print("\n" + df.to_string(index=False))

    # Detailed breakdown of degradation
    print("\n" + "="*80)
    print("DETAILED DEGRADATION ANALYSIS")
    print("="*80)

    for r in results:
        print(f"\n--- Step {r['step']} ---")
        print(f"Exact match accuracy: {r['accuracy']:.1%}")
        print(f"Valid FEN structure: {r['valid_fen_structure']:.1%}")
        print(f"Has invalid characters: {r['has_invalid_chars']:.1%}")
        print(f"Average completion length: {r['avg_completion_length']:.0f} chars")
        print(f"\nSample completions (first 100 chars):")
        for i, comp in enumerate(r['sample_completions'], 1):
            print(f"  {i}: {comp}")

    # Check for catastrophic degradation
    if len(results) >= 2:
        initial_acc = results[0]['accuracy']
        final_acc = results[-1]['accuracy']

        print("\n" + "="*80)
        print("PERFORMANCE CHANGE SUMMARY")
        print("="*80)
        print(f"Initial accuracy (Step {results[0]['step']}): {initial_acc:.1%}")
        print(f"Final accuracy (Step {results[-1]['step']}): {final_acc:.1%}")
        print(f"Absolute change: {(final_acc - initial_acc)*100:+.1f}%")
        print(f"Relative change: {((final_acc / initial_acc) - 1)*100:+.1f}%" if initial_acc > 0 else "N/A")

        # Find when degradation started
        for i in range(1, len(results)):
            if results[i]['accuracy'] < results[i-1]['accuracy'] * 0.5:  # 50% drop
                print(f"\n⚠️  CATASTROPHIC DEGRADATION detected between Step {results[i-1]['step']} and Step {results[i]['step']}")
                print(f"   Accuracy dropped from {results[i-1]['accuracy']:.1%} to {results[i]['accuracy']:.1%}")
                break

if __name__ == "__main__":
    main()