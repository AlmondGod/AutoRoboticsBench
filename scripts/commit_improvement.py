#!/usr/bin/env python3
"""Commit tracked source changes when a run improves eval score."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PRIMARY_SCORE_KEYS = (
    "score",
    "bc1_reliability_speed_score",
    "success_rate",
    "peak_final_success",
    "hidden_final_success",
    "video_transfer_success",
    "language_success_rate",
    "offlinerl_final_success",
    "visual_world_model_score",
    "reward_model_benchmark_score",
    "world_model_benchmark_score",
)


def git(repo_root: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        text=True,
        capture_output=True,
    )


def read_score(results_path: Path) -> float:
    if not results_path.exists():
        raise SystemExit(f"Eval results not found: {results_path}")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    metric = data.get("metric")
    if isinstance(metric, str):
        try:
            return float(data[metric])
        except (KeyError, TypeError, ValueError):
            pass
    for key in PRIMARY_SCORE_KEYS:
        try:
            return float(data[key])
        except (KeyError, TypeError, ValueError):
            continue
    raise SystemExit(f"Eval results do not contain a known numeric primary score: {results_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--message")
    parser.add_argument("--allow-equal", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / args.run_id
    results_path = run_dir / "eval" / "results.json"
    best_path = run_dir / "best_committed_score.json"

    branch = git(repo_root, ["branch", "--show-current"]).stdout.strip()
    if branch in {"main", "master"}:
        raise SystemExit("Refusing to commit benchmark changes directly on main/master.")
    if not branch:
        raise SystemExit("Refusing to commit from detached HEAD.")

    score = read_score(results_path)
    previous = None
    if best_path.exists():
        previous_payload = json.loads(best_path.read_text(encoding="utf-8"))
        previous = float(previous_payload.get("score", previous_payload.get("success_rate")))

    improved = previous is None or score > previous or (args.allow_equal and score == previous)
    if not improved:
        print(f"No commit: score {score:.6f} did not improve previous {previous:.6f}.")
        return 0

    status_before = git(repo_root, ["status", "--porcelain"]).stdout.strip()
    if not status_before:
        print("No tracked source changes to commit.")
        best_path.write_text(
            json.dumps({"task": args.task, "score": score}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return 0

    git(repo_root, ["add", "-A", "--", "."])
    git(repo_root, ["reset", "--", "runs"])

    staged = git(repo_root, ["diff", "--cached", "--name-only"]).stdout.strip()
    if not staged:
        print("No non-run source changes staged for commit.")
        return 0

    message = args.message or f"Improve {args.task} eval to {score:.4f}"
    git(repo_root, ["commit", "-m", message, "-m", f"run_id: {args.run_id}"])

    best_path.write_text(
        json.dumps({"task": args.task, "score": score}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Committed improvement on {branch}: score={score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
