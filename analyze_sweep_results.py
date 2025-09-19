#!/usr/bin/env python3
"""Analyze sweep results to identify optimal parameter ranges for next focused sweep."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def load_sweep_results(sweep_path: str) -> List[Dict[str, Any]]:
    """Load aggregate results from sweep."""
    results_file = Path(sweep_path) / "aggregate_results.json"
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_performance_by_param(results: List[Dict[str, Any]]) -> None:
    """Analyze performance patterns by each parameter."""

    # Extract data for analysis
    data = []
    for result in results:
        if result["status"] != "ok":
            continue

        params = result["params"]
        metrics = result["metrics"]

        # Get final evaluation if available
        final_eval = None
        if result["eval_history"]:
            final_eval = result["eval_history"][-1]["accuracy_percent"]

        row = {
            "run_id": result["run_id"],
            "performance_change": metrics.get("performance_change", 0),
            "final_performance": metrics.get("post_performance", metrics.get("initial_performance", 0)),
            "final_eval_acc": final_eval,
            "learning_rate": params["learning_rate"],
            "lr_schedule": params["lr_schedule"],
            "lr_warmup_steps": params["lr_warmup_steps"],
            "batch_size": params["batch_size"],
            "num_generations": params["num_generations"],
            "entropy_coef": params["entropy_coef"],
        }
        data.append(row)

    df = pd.DataFrame(data)

    print("=== SWEEP RESULTS ANALYSIS ===")
    print(f"Total successful runs: {len(df)}")
    print()

    # Top performers by performance change
    print("🏆 TOP 10 PERFORMERS (by performance change)")
    top_performers = df.nlargest(10, 'performance_change')
    print(top_performers[['run_id', 'performance_change', 'learning_rate', 'lr_schedule',
                         'batch_size', 'num_generations', 'entropy_coef']].to_string(index=False))
    print()

    # Analyze learning rate patterns
    print("📊 LEARNING RATE ANALYSIS")
    lr_bins = [
        (0, 1e-7, "Very Low (0-1e-7)"),
        (1e-7, 3e-7, "Optimal (1e-7 to 3e-7)"),
        (3e-7, 1e-6, "Moderate (3e-7 to 1e-6)"),
        (1e-6, 2e-6, "High (1e-6 to 2e-6)"),
        (2e-6, float('inf'), "Very High (>2e-6)")
    ]

    for min_lr, max_lr, label in lr_bins:
        mask = (df['learning_rate'] > min_lr) & (df['learning_rate'] <= max_lr)
        subset = df[mask]
        if len(subset) > 0:
            avg_perf = subset['performance_change'].mean()
            success_rate = (subset['performance_change'] > 0).mean() * 100
            print(f"{label:20} | Count: {len(subset):2} | Avg ΔPerf: {avg_perf:+6.3f} | Success Rate: {success_rate:4.1f}%")
    print()

    # Analyze other parameters for top performers (performance_change > 0.1)
    high_performers = df[df['performance_change'] > 0.1]
    print(f"🎯 HIGH PERFORMER PATTERNS (ΔPerf > 0.1, n={len(high_performers)})")

    if len(high_performers) > 0:
        print(f"Learning Rate range: {high_performers['learning_rate'].min():.2e} to {high_performers['learning_rate'].max():.2e}")
        print(f"LR Schedule: {high_performers['lr_schedule'].value_counts().to_dict()}")
        print(f"LR Warmup Steps: {sorted(high_performers['lr_warmup_steps'].unique())}")
        print(f"Batch Sizes: {sorted(high_performers['batch_size'].unique())}")
        print(f"Generations: {sorted(high_performers['num_generations'].unique())}")
        print(f"Entropy Coefs: {sorted(high_performers['entropy_coef'].unique())}")
    print()

    # Parameter recommendations
    print("🎯 RECOMMENDED PARAMETER RANGES FOR NEXT SWEEP")

    # Learning rate: focus on proven optimal range
    good_lr_runs = df[(df['learning_rate'] >= 1e-7) & (df['learning_rate'] <= 3e-7)]
    if len(good_lr_runs) > 0:
        print(f"Learning Rate: {good_lr_runs['learning_rate'].min():.2e} to {good_lr_runs['learning_rate'].max():.2e}")
        print(f"  (Current range shows {good_lr_runs['performance_change'].mean():+.3f} avg ΔPerf vs {df['performance_change'].mean():+.3f} overall)")

    # Other parameters from high performers
    if len(high_performers) > 0:
        print(f"LR Warmup Steps: {list(high_performers['lr_warmup_steps'].unique())}")
        print(f"Batch Sizes: {list(high_performers['batch_size'].unique())}")
        print(f"Generations: {list(high_performers['num_generations'].unique())}")
        print(f"Entropy Coefficients: {list(high_performers['entropy_coef'].unique())}")

        # Most common schedule among high performers
        best_schedule = high_performers['lr_schedule'].mode().iloc[0]
        print(f"Best LR Schedule: {best_schedule}")

if __name__ == "__main__":
    import sys

    # Accept sweep path as command line argument or use default
    if len(sys.argv) > 1:
        sweep_path = sys.argv[1]
    else:
        sweep_path = "logs/sweeps/20250919_143505"

    print(f"Analyzing sweep results from: {sweep_path}")
    results = load_sweep_results(sweep_path)
    analyze_performance_by_param(results)