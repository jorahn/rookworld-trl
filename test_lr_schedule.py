#!/usr/bin/env python3
"""Test and visualize the learning rate schedule."""

import matplotlib.pyplot as plt
import numpy as np
from manual_grpo_debug import get_lr_with_schedule

def test_lr_schedule():
    total_steps = 500
    base_lr = 2e-6
    warmup_steps = 20

    schedules = ["constant", "cosine", "linear", "step", "advanced"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, schedule in enumerate(schedules):
        steps = []
        lrs = []

        for step in range(total_steps):
            lr = get_lr_with_schedule(step, total_steps, base_lr, schedule, warmup_steps)
            steps.append(step)
            lrs.append(lr)

        ax = axes[idx]
        ax.plot(steps, lrs, linewidth=2)
        ax.set_title(f"Schedule: {schedule}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5, label='End of warmup')

        if schedule == "advanced":
            # Mark the phase transitions
            remaining = total_steps - warmup_steps
            cosine_end = warmup_steps + int(remaining * 0.7)
            ax.axvline(x=cosine_end, color='g', linestyle='--', alpha=0.5, label='End of cosine')
            ax.legend()

        # Add min/max annotations
        max_lr = max(lrs)
        min_lr = min(lrs)
        ax.text(0.02, 0.98, f"Max: {max_lr:.2e}\nMin: {min_lr:.2e}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle(f"Learning Rate Schedules (base_lr={base_lr:.1e}, warmup={warmup_steps}, total={total_steps})")
    plt.tight_layout()
    plt.savefig("lr_schedules.png", dpi=150)
    print("Learning rate schedules saved to lr_schedules.png")

    # Print detailed info for advanced schedule
    print("\nAdvanced Schedule Details:")
    print(f"  Warmup steps: 0-{warmup_steps} (linear 0 → {base_lr:.2e})")
    remaining = total_steps - warmup_steps
    cosine_steps = int(remaining * 0.7)
    linear_steps = remaining - cosine_steps
    print(f"  Cosine decay: {warmup_steps}-{warmup_steps + cosine_steps} ({cosine_steps} steps, {base_lr:.2e} → {0.05*base_lr:.2e})")
    print(f"  Linear annealing: {warmup_steps + cosine_steps}-{total_steps} ({linear_steps} steps, {0.05*base_lr:.2e} → 0)")

    # Test specific points
    print("\nSample learning rates (advanced schedule):")
    test_points = [0, 10, 20, 50, 100, 200, 350, 450, 499]
    for step in test_points:
        if step < total_steps:
            lr = get_lr_with_schedule(step, total_steps, base_lr, "advanced", warmup_steps)
            print(f"  Step {step:3d}: {lr:.3e}")

if __name__ == "__main__":
    test_lr_schedule()