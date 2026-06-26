#!/usr/bin/env python3
"""Finalize a benchmark run by evaluating, judging, collecting artifacts, and writing metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRIMARY_SCORE_KEYS = (
    "score",
    "bc1_reliability_speed_val_mse_score",
    "bc1_reliability_speed_score",
    "bc5_success_val_mse_score",
    "video_transfer_val_mse_score",
    "long_horizon_success_val_mse_score",
    "language_success_val_mse_score",
    "visual_world_model_score",
    "reward_model_benchmark_score",
    "world_model_benchmark_score",
    "success_rate",
    "peak_final_success",
    "hidden_final_success",
    "video_transfer_success",
    "language_success_rate",
    "offlinerl_final_success",
)


def run_step(name: str, command: list[str]) -> dict:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, text=True, capture_output=True)
    finished = datetime.now(timezone.utc)
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": finished.isoformat().replace("+00:00", "Z"),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def read_float(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def primary_score(eval_results: dict | None) -> float | None:
    if not isinstance(eval_results, dict):
        return None
    metric = eval_results.get("metric")
    if isinstance(metric, str):
        value = to_float(eval_results.get(metric))
        if value is not None:
            return value
    for key in PRIMARY_SCORE_KEYS:
        value = to_float(eval_results.get(key))
        if value is not None:
            return value
    return None


def count_flags(eval_results: dict | None, judge_report: dict | None) -> int:
    total = 0
    if isinstance(eval_results, dict) and isinstance(eval_results.get("flags"), list):
        total += len(eval_results["flags"])
    if isinstance(judge_report, dict) and isinstance(judge_report.get("warnings"), list):
        total += len(judge_report["warnings"])
    if isinstance(judge_report, dict) and isinstance(judge_report.get("flags"), list):
        total += len(judge_report["flags"])
    return total


def build_run_summary(
    *,
    run_id: str,
    task: str,
    mode: str,
    run_dir: Path,
    steps: list[dict],
    finished_at: str,
) -> dict[str, Any]:
    metadata = read_json(run_dir / "run_metadata.json") or {}
    usage = read_json(run_dir / "run_usage.json") or {}
    eval_results = read_json(run_dir / "eval" / "results.json") or {}
    judge_report = read_json(run_dir / "judge_report.json") or {}
    commands = read_jsonl(run_dir / "commands.jsonl")

    started_epoch = read_float(run_dir / "started_epoch.txt")
    finished_epoch = datetime.now(timezone.utc).timestamp()
    wall_seconds = finished_epoch - started_epoch if started_epoch is not None else None
    container_seconds = sum(float(row.get("duration_seconds") or 0) for row in commands) if commands else None

    success_rate = to_float(eval_results.get("success_rate"))
    score = primary_score(eval_results)
    baseline_score = to_float(metadata.get("baseline_score"))
    improvement = None
    normalized_score = None
    if score is not None and baseline_score is not None:
        improvement = score - baseline_score
        if baseline_score != 1.0:
            normalized_score = (score - baseline_score) / (1.0 - baseline_score)
    elif score is not None:
        normalized_score = score

    step_failed = any(int(step.get("returncode", 0)) != 0 for step in steps)
    valid = eval_results.get("valid")
    if valid is None:
        valid = judge_report.get("valid")
    fatal = bool(step_failed or valid is False)

    return {
        "run_id": run_id,
        "agent": metadata.get("agent"),
        "model": metadata.get("model"),
        "scaffold": metadata.get("scaffold"),
        "task": metadata.get("task") or task,
        "base": metadata.get("base"),
        "seed": metadata.get("seed"),
        "status": "failed" if fatal else "complete",
        "valid": valid,
        "wall_seconds": wall_seconds,
        "container_seconds": container_seconds,
        "gpu_seconds": to_float(usage.get("gpu_seconds")),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "estimated_usd": usage.get("estimated_usd"),
        "score": score,
        "success_rate": success_rate,
        "baseline_score": baseline_score,
        "improvement": improvement,
        "normalized_score": normalized_score,
        "num_eval_episodes": eval_results.get("num_episodes") or eval_results.get("num_eval_episodes"),
        "num_flags": count_flags(eval_results, judge_report),
        "fatal": fatal,
        "mode": mode,
        "branch": metadata.get("branch"),
        "finished_at": finished_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=["runpod", "docker"], default="runpod")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    steps: list[dict] = []
    if args.mode == "runpod":
        steps.append(
            run_step(
                "eval",
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_eval_runpod.py"),
                    "--run-id",
                    args.run_id,
                    "--task",
                    args.task,
                ],
            )
        )
    else:
        steps.append(
            run_step(
                "eval",
                ["bash", "-lc", f"RUN_ID={args.run_id} TASK={args.task} ./docker/run_eval_container.sh"],
            )
        )

    steps.append(
        run_step(
            "judge",
            [
                sys.executable,
                str(repo_root / "scripts" / "judge_run.py"),
                "--run-id",
                args.run_id,
                "--task",
                args.task,
            ],
        )
    )
    steps.append(
        run_step(
            "collect_artifacts",
            [
                sys.executable,
                str(repo_root / "scripts" / "collect_artifacts.py"),
                "--run-id",
                args.run_id,
            ],
        )
    )

    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (run_dir / "finished_at.txt").write_text(finished_at + "\n", encoding="utf-8")

    final_report = {
        "run_id": args.run_id,
        "task": args.task,
        "mode": args.mode,
        "finished_at": finished_at,
        "steps": steps,
        "eval_results": read_json(run_dir / "eval" / "results.json"),
        "judge_report": read_json(run_dir / "judge_report.json"),
    }
    run_summary = build_run_summary(
        run_id=args.run_id,
        task=args.task,
        mode=args.mode,
        run_dir=run_dir,
        steps=steps,
        finished_at=finished_at,
    )
    final_report["run_summary"] = run_summary
    out_path = run_dir / "final_report.json"
    out_path.write_text(json.dumps(final_report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote final report to {out_path}")
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote run summary to {summary_path}")

    timing_path = run_dir / "timing.jsonl"
    if not timing_path.exists() and run_summary.get("wall_seconds") is not None:
        timing_point = {
            "wall_seconds": run_summary.get("wall_seconds"),
            "container_seconds": run_summary.get("container_seconds"),
            "score": run_summary.get("score"),
            "success_rate": run_summary.get("success_rate"),
            "normalized_score": run_summary.get("normalized_score"),
        }
        timing_path.write_text(json.dumps(timing_point, sort_keys=True) + "\n", encoding="utf-8")

    failed = [step for step in steps if step["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
