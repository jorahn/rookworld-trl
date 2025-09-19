#!/usr/bin/env python3
"""Run a randomized hyperparameter sweep for manual_grpo_debug.py and summarize results.

The script executes manual GRPO debug runs with randomly sampled hyperparameters,
collects key metrics from the generated logs, and produces an aggregated Markdown
report (plus JSON metadata) for further analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Regular expressions for parsing manual_grpo_debug.py output
LOG_PATH_RE = re.compile(r"Full log: (logs/manual_grpo_debug_run-[\d\-]+\.log)")
INITIAL_PERF_RE = re.compile(r"Initial model performance:\s*([-+]?\d+(?:\.\d+)?)")
POST_PERF_RE = re.compile(r"After\s+\d+\s+GRPO step\(s\):\s*([-+]?\d+(?:\.\d+)?)")
DELTA_PERF_RE = re.compile(r"Performance change:\s*([-+]?\d+(?:\.\d+)?)")
TOTAL_LOSS_RE = re.compile(r"Total loss:\s*([-+]?\d+(?:\.\d+)?)")
TRAIN_AVG_RE = re.compile(r"Average reward:\s*([-+]?\d+(?:\.\d+)?)")
EVAL_ACC_RE = re.compile(r"Eval accuracy:\s*(\d+)/(\d+)\s*\(([-+]?\d+(?:\.\d+)?)%\)")
EVAL_REWARD_RE = re.compile(r"Eval avg reward:\s*([-+]?\d+(?:\.\d+)?)")
MOVE1_RE = re.compile(r"Move 1:\s*(\d+)/(\d+)\s*\(([-+]?\d+(?:\.\d+)?)%\)")

# Static configuration
DEFAULT_SEED_POOL = [42, 56, 123, 314, 512, 777, 1337, 2025, 2718, 9001]
DEFAULT_RNG_SEED = 20250919
DEFAULT_RUNS = 4


@dataclass
class EvalSnapshot:
    """Container for evaluation metrics at a given checkpoint."""

    step_index: int
    accuracy_percent: float
    accuracy_hit: int
    accuracy_total: int
    avg_reward: Optional[float] = None
    move1_percent: Optional[float] = None
    move1_hit: Optional[int] = None
    move1_total: Optional[int] = None


@dataclass
class RunResult:
    """Captures metadata, metrics, and artifacts for a single sweep run."""

    run_id: int
    params: Dict[str, Any]
    command: List[str]
    status: str
    return_code: int
    duration_s: float
    stdout_path: Path
    manual_log_path: Optional[Path] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    eval_history: List[EvalSnapshot] = field(default_factory=list)
    error: Optional[str] = None

    def short_status(self) -> str:
        if self.status == "ok":
            return "ok"
        return f"error({self.return_code})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomized sweep runner for manual GRPO debug script")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of randomized runs to execute")
    parser.add_argument("--steps", type=int, default=200, help="Number of GRPO steps per run")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Evaluation frequency to pass through to manual_grpo_debug.py",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per batch")
    parser.add_argument("--generations", type=int, default=40, help="Generations per prompt for GRPO groups")
    parser.add_argument(
        "--task-type",
        type=str,
        default="A",
        choices=["P", "A", "mixed"],
        help="Task type to target; defaults to environment tasks (A)",
    )
    parser.add_argument(
        "--sweep-seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help="Seed for the sweep sampler (affects hyperparameter selection order)",
    )
    parser.add_argument(
        "--seed-pool",
        type=int,
        nargs="*",
        default=DEFAULT_SEED_POOL,
        help="Seed values to sample from for manual runs (with replacement)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/sweeps",
        help="Directory to store sweep artifacts (stdout, metadata, report)",
    )
    parser.add_argument(
        "--uv-bin",
        type=str,
        default="uv",
        help="Path to the uv executable (defaults to 'uv' on PATH)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample hyperparameters and print planned commands without executing them",
    )
    parser.add_argument(
        "--parallel-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel execution (default: 1)",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu",
        type=int,
        default=4,
        help="Maximum concurrent runs per GPU (default: 4, safe for gradient updates)",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_learning_rate(rng: random.Random) -> float:
    log_min = math.log10(1e-7)
    log_max = math.log10(2e-6)
    value = 10 ** rng.uniform(log_min, log_max)
    # Round to two significant figures for readability while keeping value precise enough
    return float(f"{value:.2e}")


def sample_schedule_and_warmup(rng: random.Random) -> Dict[str, Any]:
    schedule = rng.choice(["advanced", "cosine"])
    if schedule == "advanced":
        warmup = rng.choice([10, 20, 30])
    else:
        warmup = rng.choice([0, 10, 20])
    return {"lr_schedule": schedule, "lr_warmup_steps": warmup}


def sample_batch_and_generations(rng: random.Random) -> Dict[str, Any]:
    """Sample batch size and generations with memory-aware constraints."""
    # Batch size: affects gradient estimation quality and memory
    batch_size = rng.choice([4, 8, 12])

    # Generations per prompt: affects GRPO signal quality
    # Balance between signal quality and memory usage
    if batch_size <= 6:
        num_generations = rng.choice([20, 30, 40])
    else:
        num_generations = rng.choice([16, 24, 32])

    return {"batch_size": batch_size, "num_generations": num_generations}


def sample_entropy(rng: random.Random) -> Dict[str, Any]:
    """Sample entropy coefficient - exploration vs exploitation balance."""
    # Entropy coefficient: exploration vs exploitation balance
    entropy_coef = rng.choice([0.001, 0.002, 0.005])

    return {"entropy_coef": entropy_coef}


def sample_hyperparameters(rng: random.Random, args: argparse.Namespace) -> Dict[str, Any]:
    params = {
        "steps": args.steps,
        "task_type": args.task_type,
        "eval_every": args.eval_every,
        "learning_rate": sample_learning_rate(rng),
        "seed": rng.choice(args.seed_pool) if args.seed_pool else rng.randint(1, 10_000),
        "beta_warmup_steps": 20,  # Keep reasonable default, not critical to sweep
    }
    # Add high-impact parameter sampling
    params.update(sample_schedule_and_warmup(rng))
    params.update(sample_batch_and_generations(rng))
    params.update(sample_entropy(rng))
    return params


def build_command(args: argparse.Namespace, params: Dict[str, Any], uv_bin: str) -> List[str]:
    cmd = [
        uv_bin,
        "run",
        "python",
        "manual_grpo_debug.py",
        "--steps",
        str(params["steps"]),
        "--task_type",
        params["task_type"],
        "--batch_size",
        str(params["batch_size"]),
        "--gens",
        str(params["num_generations"]),
        "--beta_warmup_steps",
        str(params["beta_warmup_steps"]),
        "--entropy_coef",
        f"{params['entropy_coef']}",
        "--eval_every",
        str(params["eval_every"]),
        "--lr_schedule",
        params["lr_schedule"],
        "--lr_warmup_steps",
        str(params["lr_warmup_steps"]),
        "--learning_rate",
        f"{params['learning_rate']}",
        "--seed",
        str(params["seed"]),
    ]
    return cmd


def parse_log_file(log_path: Path) -> Dict[str, Any]:
    """Parse the manual_grpo_debug log file for metrics and evaluation data."""
    metrics: Dict[str, Any] = {}
    eval_history: List[EvalSnapshot] = []

    if not log_path.exists():
        return {"metrics": metrics, "eval_history": eval_history}

    try:
        with log_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return {"metrics": metrics, "eval_history": eval_history}

    eval_count = -1

    for line in lines:
        # Parse summary metrics
        if "Initial model performance" in line:
            match = INITIAL_PERF_RE.search(line)
            if match:
                metrics["initial_performance"] = float(match.group(1))
            continue

        if "After" in line and "GRPO step" in line:
            match = POST_PERF_RE.search(line)
            if match:
                metrics["post_performance"] = float(match.group(1))
            continue

        if "Performance change" in line:
            match = DELTA_PERF_RE.search(line)
            if match:
                metrics["performance_change"] = float(match.group(1))
            continue

        if "Total loss" in line:
            match = TOTAL_LOSS_RE.search(line)
            if match:
                metrics["total_loss"] = float(match.group(1))
            continue

        # Parse evaluation metrics
        acc_match = EVAL_ACC_RE.search(line)
        if acc_match:
            eval_count += 1
            snapshot = EvalSnapshot(
                step_index=eval_count,
                accuracy_hit=int(acc_match.group(1)),
                accuracy_total=int(acc_match.group(2)),
                accuracy_percent=float(acc_match.group(3)),
            )
            eval_history.append(snapshot)
            continue

        reward_match = EVAL_REWARD_RE.search(line)
        if reward_match and eval_history:
            eval_history[-1].avg_reward = float(reward_match.group(1))
            continue

        move1_match = MOVE1_RE.search(line)
        if move1_match and eval_history:
            eval_history[-1].move1_hit = int(move1_match.group(1))
            eval_history[-1].move1_total = int(move1_match.group(2))
            eval_history[-1].move1_percent = float(move1_match.group(3))
            continue

    return {"metrics": metrics, "eval_history": eval_history}


def parse_manual_output(stdout_text: str, stderr_text: str) -> Dict[str, Any]:
    """Extract log path from stderr and parse the actual log file for metrics."""
    manual_log_path: Optional[Path] = None

    # Look for log path in both stdout and stderr
    all_text = stdout_text + "\n" + stderr_text
    for line in all_text.splitlines():
        log_match = LOG_PATH_RE.search(line)
        if log_match:
            manual_log_path = Path(log_match.group(1))
            break

    # Parse the actual log file if found
    if manual_log_path and manual_log_path.exists():
        log_data = parse_log_file(manual_log_path)
        return {
            "manual_log_path": manual_log_path,
            "metrics": log_data["metrics"],
            "eval_history": log_data["eval_history"],
        }

    # Fallback to parsing stderr/stdout directly (legacy behavior)
    metrics: Dict[str, Any] = {}
    eval_history: List[EvalSnapshot] = []
    eval_count = -1

    for line in all_text.splitlines():
        if "Initial model performance" in line:
            match = INITIAL_PERF_RE.search(line)
            if match:
                metrics.setdefault("initial_performance", float(match.group(1)))
            continue

        if "After" in line and "GRPO step" in line:
            match = POST_PERF_RE.search(line)
            if match:
                metrics["post_performance"] = float(match.group(1))
            continue

        if "Performance change" in line:
            match = DELTA_PERF_RE.search(line)
            if match:
                metrics["performance_change"] = float(match.group(1))
            continue

        if "Total loss" in line:
            match = TOTAL_LOSS_RE.search(line)
            if match:
                metrics["total_loss"] = float(match.group(1))
            continue

        acc_match = EVAL_ACC_RE.search(line)
        if acc_match:
            eval_count += 1
            snapshot = EvalSnapshot(
                step_index=eval_count,
                accuracy_hit=int(acc_match.group(1)),
                accuracy_total=int(acc_match.group(2)),
                accuracy_percent=float(acc_match.group(3)),
            )
            eval_history.append(snapshot)
            continue

        reward_match = EVAL_REWARD_RE.search(line)
        if reward_match and eval_history:
            eval_history[-1].avg_reward = float(reward_match.group(1))
            continue

        move1_match = MOVE1_RE.search(line)
        if move1_match and eval_history:
            eval_history[-1].move1_hit = int(move1_match.group(1))
            eval_history[-1].move1_total = int(move1_match.group(2))
            eval_history[-1].move1_percent = float(move1_match.group(3))
            continue

    return {
        "manual_log_path": manual_log_path,
        "metrics": metrics,
        "eval_history": eval_history,
    }


def run_single_sweep(
    run_id: int,
    args: argparse.Namespace,
    params: Dict[str, Any],
    output_root: Path,
    uv_bin: str,
    dry_run: bool,
    gpu_id: Optional[int] = None,
) -> RunResult:
    run_dir = output_root / f"run_{run_id:02d}"
    ensure_directory(run_dir)

    command = build_command(args, params, uv_bin)
    command_str = " ".join(command)

    stdout_path = run_dir / "stdout.log"
    metadata_path = run_dir / "metadata.json"
    (run_dir / "command.txt").write_text(command_str + "\n", encoding="utf-8")

    if dry_run:
        stdout_path.write_text("[dry run] command not executed\n", encoding="utf-8")
        result = RunResult(
            run_id=run_id,
            params=params,
            command=command,
            status="skipped",
            return_code=0,
            duration_s=0.0,
            stdout_path=stdout_path,
        )
        json.dump(result.__dict__, metadata_path.open("w", encoding="utf-8"), default=str, indent=2)
        return result

    gpu_info = f" (GPU {gpu_id})" if gpu_id is not None else ""
    print(f"[run {run_id}]{gpu_info} starting: {command_str}")
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(repo_root() / ".uv-cache"))

    # Set GPU visibility if specified
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.run(
        command,
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    duration = time.perf_counter() - start

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_text = proc.stderr or ""
    if stderr_text:
        (run_dir / "stderr.log").write_text(stderr_text, encoding="utf-8")

    parse_result = parse_manual_output(proc.stdout, stderr_text)
    metrics = parse_result["metrics"]
    eval_history = parse_result["eval_history"]
    manual_log_path = parse_result["manual_log_path"]

    status = "ok" if proc.returncode == 0 else "failed"
    error_msg = None if status == "ok" else (proc.stderr.strip() or "manual_grpo_debug exited with non-zero status")

    result = RunResult(
        run_id=run_id,
        params=params,
        command=command,
        status=status,
        return_code=proc.returncode,
        duration_s=duration,
        stdout_path=stdout_path,
        manual_log_path=manual_log_path,
        metrics=metrics,
        eval_history=eval_history,
        error=error_msg,
    )

    json.dump(
        {
            "run_id": run_id,
            "params": params,
            "command": command,
            "status": status,
            "return_code": proc.returncode,
            "duration_seconds": duration,
            "metrics": metrics,
            "manual_log_path": str(manual_log_path) if manual_log_path else None,
            "eval_history": [snapshot.__dict__ for snapshot in eval_history],
            "error": error_msg,
        },
        metadata_path.open("w", encoding="utf-8"),
        indent=2,
    )

    if status == "ok":
        print(f"[run {run_id}]{gpu_info} completed in {duration/60:.1f} min — Δperf={metrics.get('performance_change')}")
    else:
        print(f"[run {run_id}]{gpu_info} FAILED (rc={proc.returncode}) — see {stdout_path}")

    return result


def summarize_results(results: List[RunResult], report_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_runs = len(results)
    successes = [r for r in results if r.status == "ok"]
    failures = [r for r in results if r.status != "ok"]

    def maybe_best(key: str, default: float = float("-inf")) -> Optional[RunResult]:
        candidates = [r for r in successes if key in r.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda rr: rr.metrics.get(key, default))

    best_delta = maybe_best("performance_change")

    best_final_eval: Optional[RunResult] = None
    for run in successes:
        if not run.eval_history:
            continue
        final_eval = run.eval_history[-1].accuracy_percent
        if best_final_eval is None or final_eval > best_final_eval.eval_history[-1].accuracy_percent:
            best_final_eval = run

    lines: List[str] = []
    lines.append(f"# Manual GRPO Randomized Sweep Report\n")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Runs attempted: {total_runs}")
    lines.append(f"- Successful runs: {len(successes)}")
    lines.append(f"- Failed runs: {len(failures)}\n")

    if best_delta:
        lines.append("## Top Performer — Performance Gain")
        lines.append(
            textwrap.dedent(
                f"""
                - Run: {best_delta.run_id}
                - Δ Performance: {best_delta.metrics.get('performance_change')}
                - Final performance: {best_delta.metrics.get('post_performance')}
                - Params: {json.dumps(best_delta.params, indent=2)}
                - Manual log: {best_delta.manual_log_path or 'n/a'}
                """
            ).strip()
        )
        lines.append("")

    if best_final_eval:
        final_snapshot = best_final_eval.eval_history[-1]
        lines.append("## Top Performer — Final Evaluation Accuracy")
        lines.append(
            textwrap.dedent(
                f"""
                - Run: {best_final_eval.run_id}
                - Final eval: {final_snapshot.accuracy_percent:.1f}% ({final_snapshot.accuracy_hit}/{final_snapshot.accuracy_total})
                - Move 1: {final_snapshot.move1_percent or 'n/a'}%
                - Params: {json.dumps(best_final_eval.params, indent=2)}
                - Manual log: {best_final_eval.manual_log_path or 'n/a'}
                """
            ).strip()
        )
        lines.append("")

    if successes:
        lines.append("## Run Summary")
        header = (
            "| Run | Seed | LR | Schedule | Warmup | Steps | ΔPerf | FinalPerf | BaselineEval | FinalEval | FinalMove1 | Status | Log |"
        )
        separator = "|" + " --- |" * 13
        lines.append(header)
        lines.append(separator)
        for run in results:
            params = run.params
            metrics = run.metrics
            seed = params.get("seed", "-")
            lr = params.get("learning_rate", "-")
            schedule = params.get("lr_schedule", "-")
            warmup = params.get("lr_warmup_steps", "-")
            steps = params.get("steps", "-")
            delta = metrics.get("performance_change", "-")
            final_perf = metrics.get("post_performance", "-")

            baseline_eval = final_eval = final_move1 = "-"
            if run.eval_history:
                baseline_snapshot = run.eval_history[0]
                baseline_eval = f"{baseline_snapshot.accuracy_percent:.1f}%"
                final_snapshot = run.eval_history[-1]
                final_eval = f"{final_snapshot.accuracy_percent:.1f}%"
                if final_snapshot.move1_percent is not None:
                    final_move1 = f"{final_snapshot.move1_percent:.1f}%"

            status = run.short_status()
            log_path = run.manual_log_path.name if run.manual_log_path else "n/a"

            lines.append(
                f"| {run.run_id} | {seed} | {lr} | {schedule} | {warmup} | {steps} | {delta} | {final_perf} | {baseline_eval} | {final_eval} | {final_move1} | {status} | {log_path} |"
            )
        lines.append("")

    if failures:
        lines.append("## Failures")
        for run in failures:
            lines.append(
                f"- Run {run.run_id} failed (rc={run.return_code}): {run.error or 'see stderr'}"
            )
        lines.append("")

    # Parameter coverage summary
    if successes:
        schedules = sorted({run.params.get("lr_schedule") for run in successes})
        seeds_used = sorted({run.params.get("seed") for run in successes})
        lines.append("## Parameter Coverage")
        lines.append(f"- Schedules explored: {', '.join(map(str, schedules))}")
        lines.append(f"- Seeds explored: {', '.join(map(str, seeds_used))}")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.sweep_seed)

    output_root = repo_root() / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_directory(output_root)

    planned_runs: List[Dict[str, Any]] = []
    for _ in range(args.runs):
        params = sample_hyperparameters(rng, args)
        planned_runs.append(params)

    if args.dry_run:
        print("Dry run mode — planned commands:\n")
        for idx, params in enumerate(planned_runs, start=1):
            cmd = build_command(args, params, args.uv_bin)
            print(f"[{idx}] {' '.join(cmd)}")
        return

    results: List[RunResult] = []
    max_workers = args.parallel_gpus * args.max_concurrent_per_gpu

    if max_workers > 1:
        # Parallel execution with GPU-aware concurrency control
        print(f"Running {len(planned_runs)} experiments: {args.parallel_gpus} GPUs × {args.max_concurrent_per_gpu} concurrent/GPU = {max_workers} max parallel\n")

        def run_with_gpu(task):
            idx, params = task
            gpu_id = (idx - 1) % args.parallel_gpus  # Distribute runs across GPUs
            return run_single_sweep(idx, args, params, output_root, args.uv_bin, args.dry_run, gpu_id)

        tasks = [(idx, params) for idx, params in enumerate(planned_runs, start=1)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {executor.submit(run_with_gpu, task): task[0] for task in tasks}

            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"[run {run_id}] generated an exception: {exc}")

        # Sort results by run_id to maintain order
        results.sort(key=lambda r: r.run_id)
    else:
        # Sequential execution (original behavior)
        for idx, params in enumerate(planned_runs, start=1):
            result = run_single_sweep(idx, args, params, output_root, args.uv_bin, args.dry_run)
            results.append(result)

    # Persist aggregate metadata
    aggregate_path = output_root / "aggregate_results.json"
    json.dump(
        [
            {
                "run_id": r.run_id,
                "params": r.params,
                "status": r.status,
                "return_code": r.return_code,
                "duration_seconds": r.duration_s,
                "metrics": r.metrics,
                "manual_log_path": str(r.manual_log_path) if r.manual_log_path else None,
                "eval_history": [snapshot.__dict__ for snapshot in r.eval_history],
                "stdout_path": str(r.stdout_path),
                "error": r.error,
            }
            for r in results
        ],
        aggregate_path.open("w", encoding="utf-8"),
        indent=2,
    )

    report_path = output_root / "report.md"
    summarize_results(results, report_path)
    print(f"\nSweep complete. Report written to {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(130)
