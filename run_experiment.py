from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import research
from robotbench.config import load_task
from robotbench.evaluate import evaluate_policy
from robotbench.logging import (
    RUNS_DIR,
    append_ledger,
    git_commit,
    git_diff_summary,
    utc_now_id,
    write_json,
)
from robotbench.train import TrainBudget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=["reach", "push"])
    parser.add_argument("--backend", default="toy", choices=["toy", "mujoco", "aloha", "mobile_aloha_mock"])
    parser.add_argument("--budget-seconds", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--rollouts-per-iteration", type=int, default=24)
    parser.add_argument("--change-note", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    result = run_experiment(
        task_name=args.task,
        backend=args.backend,
        budget_seconds=args.budget_seconds,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_iterations=args.max_iterations,
        rollouts_per_iteration=args.rollouts_per_iteration,
        change_note=args.change_note,
        run_id=args.run_id or None,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def run_experiment(
    task_name: str,
    backend: str,
    budget_seconds: float,
    seeds: list[int],
    episodes_per_seed: int,
    max_iterations: int,
    rollouts_per_iteration: int,
    change_note: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    task = load_task(task_name)
    run_id = run_id or f"{utc_now_id()}-{backend}-{task_name}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    budget = TrainBudget(
        seconds=budget_seconds,
        max_iterations=max_iterations,
        rollouts_per_iteration=rollouts_per_iteration,
        eval_episodes=episodes_per_seed,
    )

    per_seed = []
    for seed in seeds:
        train_start = time.monotonic()
        policy = research.train_policy(task=task, budget=budget, seed=seed, backend=backend)
        train_elapsed_seconds = time.monotonic() - train_start
        policy_path = run_dir / f"policy_seed{seed}.npz"
        if hasattr(research, "save_policy"):
            research.save_policy(policy, str(policy_path))
        train_metrics = evaluate_policy(
            task=task,
            world="train",
            seeds=[seed],
            episodes_per_seed=episodes_per_seed,
            policy_fn=lambda obs, p=policy: research.act(p, obs),
            backend=backend,
        )
        eval_metrics = evaluate_policy(
            task=task,
            world="eval",
            seeds=[seed],
            episodes_per_seed=episodes_per_seed,
            policy_fn=lambda obs, p=policy: research.act(p, obs),
            backend=backend,
        )
        per_seed.append(
            {
                "seed": seed,
                "policy_path": str(policy_path),
                "train_elapsed_seconds": train_elapsed_seconds,
                "train": train_metrics,
                "eval": eval_metrics,
            }
        )

    summary = summarize(task_name=task_name, backend=backend, per_seed=per_seed)
    payload = {
        "run_id": run_id,
        "task": task_name,
        "backend": backend,
        "budget_seconds": budget_seconds,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "change_note": change_note,
        "git_commit": git_commit(),
        "git_diff_summary": git_diff_summary(),
        "per_seed": per_seed,
        "summary": summary,
    }
    write_json(run_dir / "metrics.json", payload)

    ledger_entry = {
        "run_id": run_id,
        "task": task_name,
        "backend": backend,
        "git_commit": payload["git_commit"],
        "change_note": change_note,
        "git_diff_summary": payload["git_diff_summary"],
        "eval_score": summary["eval_score"],
        "eval_success_rate": summary["eval_success_rate"],
        "eval_catastrophe_rate": summary["eval_catastrophe_rate"],
        "accepted": None,
        "run_dir": str(run_dir),
    }
    append_ledger(ledger_entry)
    return payload


def summarize(task_name: str, backend: str, per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(path: tuple[str, str]) -> float:
        return sum(float(row[path[0]][path[1]]) for row in per_seed) / len(per_seed)

    return {
        "task": task_name,
        "backend": backend,
        "seed_count": len(per_seed),
        "train_elapsed_seconds": sum(float(row["train_elapsed_seconds"]) for row in per_seed),
        "train_score": avg(("train", "score")),
        "eval_score": avg(("eval", "score")),
        "eval_success_rate": avg(("eval", "success_rate")),
        "eval_catastrophe_rate": avg(("eval", "catastrophe_rate")),
        "eval_avg_return": avg(("eval", "avg_return")),
        "eval_avg_distance": avg(("eval", "avg_distance")),
        "eval_joint_limit_violations": avg(("eval", "joint_limit_violations")),
        "eval_torque_limit_violations": avg(("eval", "torque_limit_violations")),
    }


if __name__ == "__main__":
    main()
