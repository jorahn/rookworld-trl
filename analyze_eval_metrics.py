#!/usr/bin/env python3
"""Analyze sweep results focusing on eval accuracy and eval reward - the actual KPIs."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def load_and_analyze_eval_metrics(sweep_path: str) -> None:
    """Analyze evaluation accuracy and reward patterns from sweep results."""
    results_file = Path(sweep_path) / "aggregate_results.json"

    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"=== EVALUATION METRICS ANALYSIS ===")
    print(f"Sweep path: {sweep_path}")
    print(f"Total successful runs: {len([r for r in results if r['status'] == 'ok'])}")
    print()

    # Extract evaluation metrics for each run
    eval_data = []
    for result in results:
        if result["status"] != "ok" or not result["eval_history"]:
            continue

        params = result["params"]

        # Get baseline and final evaluation metrics
        baseline = result["eval_history"][0]
        final = result["eval_history"][-1]

        eval_data.append({
            "run_id": result["run_id"],
            "learning_rate": params["learning_rate"],
            "lr_schedule": params["lr_schedule"],
            "lr_warmup_steps": params["lr_warmup_steps"],
            "batch_size": params["batch_size"],
            "num_generations": params["num_generations"],
            "entropy_coef": params["entropy_coef"],
            "baseline_accuracy": baseline["accuracy_percent"],
            "final_accuracy": final["accuracy_percent"],
            "accuracy_change": final["accuracy_percent"] - baseline["accuracy_percent"],
            "baseline_reward": baseline["avg_reward"],
            "final_reward": final["avg_reward"],
            "reward_change": final["avg_reward"] - baseline["avg_reward"],
            "baseline_move1": baseline["move1_percent"],
            "final_move1": final["move1_percent"],
            "move1_change": final["move1_percent"] - baseline["move1_percent"],
        })

    df = pd.DataFrame(eval_data)

    print("🎯 PRIMARY KPI: EVALUATION ACCURACY")
    print(f"Baseline accuracy: {df['baseline_accuracy'].mean():.1f}% (range: {df['baseline_accuracy'].min():.1f}%-{df['baseline_accuracy'].max():.1f}%)")
    print(f"Final accuracy: {df['final_accuracy'].mean():.1f}% (range: {df['final_accuracy'].min():.1f}%-{df['final_accuracy'].max():.1f}%)")
    print(f"Accuracy change: {df['accuracy_change'].mean():+.1f}% (range: {df['accuracy_change'].min():+.1f}% to {df['accuracy_change'].max():+.1f}%)")
    print(f"Accuracy variance: {df['accuracy_change'].std():.2f}% (0 = no learning, >2 = meaningful variation)")
    print()

    print("🏆 SECONDARY KPI: EVALUATION REWARD")
    print(f"Baseline reward: {df['baseline_reward'].mean():.3f} (range: {df['baseline_reward'].min():.3f}-{df['baseline_reward'].max():.3f})")
    print(f"Final reward: {df['final_reward'].mean():.3f} (range: {df['final_reward'].min():.3f}-{df['final_reward'].max():.3f})")
    print(f"Reward change: {df['reward_change'].mean():+.3f} (range: {df['reward_change'].min():+.3f} to {df['reward_change'].max():+.3f})")
    print(f"Reward variance: {df['reward_change'].std():.4f} (0 = no learning, >0.01 = meaningful)")
    print()

    print("🎪 MOVE 1 ACCURACY (Hardest Challenge)")
    print(f"Baseline Move 1: {df['baseline_move1'].mean():.1f}% (range: {df['baseline_move1'].min():.1f}%-{df['baseline_move1'].max():.1f}%)")
    print(f"Final Move 1: {df['final_move1'].mean():.1f}% (range: {df['final_move1'].min():.1f}%-{df['final_move1'].max():.1f}%)")
    print(f"Move 1 change: {df['move1_change'].mean():+.1f}% (range: {df['move1_change'].min():+.1f}% to {df['move1_change'].max():+.1f}%)")
    print()

    # Identify runs with meaningful improvements in key metrics
    accuracy_improvers = df[df['accuracy_change'] > 0]
    reward_improvers = df[df['reward_change'] > 0]
    move1_improvers = df[df['move1_change'] > 0]

    print("📊 IMPROVEMENT PATTERNS")
    print(f"Runs with accuracy improvement: {len(accuracy_improvers)}/{len(df)} ({len(accuracy_improvers)/len(df)*100:.1f}%)")
    print(f"Runs with reward improvement: {len(reward_improvers)}/{len(df)} ({len(reward_improvers)/len(df)*100:.1f}%)")
    print(f"Runs with Move 1 improvement: {len(move1_improvers)}/{len(df)} ({len(move1_improvers)/len(df)*100:.1f}%)")
    print()

    if len(accuracy_improvers) > 0:
        print("🏆 BEST EVAL ACCURACY PERFORMERS")
        best_accuracy = df.nlargest(5, 'accuracy_change')[['run_id', 'accuracy_change', 'final_accuracy', 'learning_rate', 'lr_schedule', 'entropy_coef']]
        print(best_accuracy.to_string(index=False))
        print()

    if len(reward_improvers) > 0:
        print("🎁 BEST EVAL REWARD PERFORMERS")
        best_reward = df.nlargest(5, 'reward_change')[['run_id', 'reward_change', 'final_reward', 'learning_rate', 'lr_schedule', 'entropy_coef']]
        print(best_reward.to_string(index=False))
        print()

    # Parameter correlation analysis for key metrics
    print("🔍 PARAMETER CORRELATIONS WITH KEY METRICS")

    # Learning rate vs metrics
    lr_corr_acc = df['learning_rate'].corr(df['accuracy_change'])
    lr_corr_reward = df['learning_rate'].corr(df['reward_change'])
    print(f"Learning rate correlation with accuracy change: {lr_corr_acc:.3f}")
    print(f"Learning rate correlation with reward change: {lr_corr_reward:.3f}")
    print()

    # Analysis by parameter values
    print("📈 EVAL ACCURACY BY PARAMETER VALUES")

    # By LR schedule
    acc_by_schedule = df.groupby('lr_schedule')['accuracy_change'].agg(['mean', 'std', 'count'])
    print("By LR Schedule:")
    print(acc_by_schedule)
    print()

    # By entropy coefficient
    acc_by_entropy = df.groupby('entropy_coef')['accuracy_change'].agg(['mean', 'std', 'count'])
    print("By Entropy Coefficient:")
    print(acc_by_entropy)
    print()

    # By number of generations
    acc_by_gens = df.groupby('num_generations')['accuracy_change'].agg(['mean', 'std', 'count'])
    print("By Number of Generations:")
    print(acc_by_gens)
    print()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sweep_path = sys.argv[1]
    else:
        # Default to latest sweep
        sweep_path = "logs/sweeps/20250919_170016"

    load_and_analyze_eval_metrics(sweep_path)