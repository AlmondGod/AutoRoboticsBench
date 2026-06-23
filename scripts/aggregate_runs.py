#!/usr/bin/env python3
"""Aggregate RoboAutoresearch run summaries into analysis CSVs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PRIMARY_SCORE_KEYS = (
    "score",
    "bc1_reliability_speed_score",
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


RUN_COLUMNS = [
    "run_id",
    "agent",
    "model",
    "scaffold",
    "task",
    "base",
    "seed",
    "status",
    "valid",
    "wall_seconds",
    "container_seconds",
    "gpu_seconds",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "estimated_usd",
    "score",
    "success_rate",
    "baseline_score",
    "improvement",
    "normalized_score",
    "num_eval_episodes",
    "num_flags",
    "fatal",
]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line, "fatal": "invalid_json"})
    return rows


def nested_get(data: dict[str, Any], paths: list[str], default: Any = None) -> Any:
    for path in paths:
        cur: Any = data
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def primary_score(payload: dict[str, Any]) -> float | None:
    metric = payload.get("metric")
    if isinstance(metric, str):
        value = to_float(payload.get(metric))
        if value is not None:
            return value
    for key in PRIMARY_SCORE_KEYS:
        value = to_float(payload.get(key))
        if value is not None:
            return value
    return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def count_flags(summary: dict[str, Any]) -> int | None:
    flags = coalesce(
        nested_get(summary, ["flags"]),
        nested_get(summary, ["eval.flags"]),
        nested_get(summary, ["eval_results.flags"]),
        nested_get(summary, ["judge.flags"]),
        nested_get(summary, ["judge_report.flags"]),
    )
    warnings = coalesce(
        nested_get(summary, ["warnings"]),
        nested_get(summary, ["judge.warnings"]),
        nested_get(summary, ["judge_report.warnings"]),
    )
    total = 0
    found = False
    for value in (flags, warnings):
        if isinstance(value, list):
            total += len(value)
            found = True
        elif value:
            total += 1
            found = True
    return total if found else None


def flatten_run(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    run_dir = summary_path.parent
    final_report = load_json(run_dir / "final_report.json")
    eval_results = load_json(run_dir / "eval" / "results.json")
    judge_report = load_json(run_dir / "judge_report.json")

    merged = {
        **summary,
        "final_report": final_report,
        "eval_results": eval_results,
        "judge_report": judge_report,
    }

    run_id = coalesce(nested_get(merged, ["run_id"]), run_dir.name)
    score = coalesce(
        nested_get(merged, ["score"]),
        nested_get(merged, ["eval.score"]),
        nested_get(merged, ["eval_results.score"]),
        nested_get(merged, ["final_report.eval_results.score"]),
        primary_score(eval_results),
        primary_score(nested_get(merged, ["final_report.eval_results"], {}) or {}),
    )
    success_rate = coalesce(
        nested_get(merged, ["success_rate"]),
        nested_get(merged, ["eval.success_rate"]),
        nested_get(merged, ["eval_results.success_rate"]),
        nested_get(merged, ["final_report.eval_results.success_rate"]),
    )
    baseline_score = nested_get(merged, ["baseline_score", "baseline.score"])
    normalized_score = nested_get(merged, ["normalized_score", "metrics.normalized_score"])

    numeric_score = to_float(coalesce(normalized_score, score, success_rate))
    if normalized_score is None:
        raw = to_float(coalesce(score, success_rate))
        base = to_float(baseline_score)
        if raw is not None and base is not None and base != 1.0:
            normalized_score = (raw - base) / (1.0 - base)
    improvement = nested_get(merged, ["improvement", "metrics.improvement"])
    if improvement is None:
        raw = to_float(coalesce(score, success_rate))
        base = to_float(baseline_score)
        if raw is not None and base is not None:
            improvement = raw - base

    row = {
        "run_id": run_id,
        "agent": nested_get(merged, ["agent"]),
        "model": nested_get(merged, ["model"]),
        "scaffold": nested_get(merged, ["scaffold"]),
        "task": nested_get(merged, ["task"]),
        "base": nested_get(merged, ["base"]),
        "seed": nested_get(merged, ["seed"]),
        "status": nested_get(merged, ["status"]),
        "valid": coalesce(
            nested_get(merged, ["valid"]),
            nested_get(merged, ["eval_results.valid"]),
            nested_get(merged, ["final_report.eval_results.valid"]),
            nested_get(merged, ["judge_report.valid"]),
        ),
        "wall_seconds": nested_get(merged, ["wall_seconds", "timing.wall_seconds"]),
        "container_seconds": nested_get(merged, ["container_seconds", "timing.container_seconds"]),
        "gpu_seconds": nested_get(merged, ["gpu_seconds", "timing.gpu_seconds"]),
        "input_tokens": nested_get(merged, ["input_tokens", "tokens.input", "usage.input_tokens"]),
        "output_tokens": nested_get(merged, ["output_tokens", "tokens.output", "usage.output_tokens"]),
        "reasoning_tokens": nested_get(
            merged,
            ["reasoning_tokens", "tokens.reasoning", "usage.reasoning_tokens"],
        ),
        "total_tokens": nested_get(merged, ["total_tokens", "tokens.total", "usage.total_tokens"]),
        "estimated_usd": nested_get(merged, ["estimated_usd", "cost.estimated_usd"]),
        "score": score,
        "success_rate": success_rate,
        "baseline_score": baseline_score,
        "improvement": improvement,
        "normalized_score": normalized_score,
        "num_eval_episodes": coalesce(
            nested_get(merged, ["num_eval_episodes"]),
            nested_get(merged, ["eval_results.num_episodes"]),
            nested_get(merged, ["final_report.eval_results.num_episodes"]),
        ),
        "num_flags": count_flags(merged),
        "fatal": nested_get(merged, ["fatal"]),
    }
    if row["score"] is None and numeric_score is not None:
        row["score"] = numeric_score
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return
    except ImportError:
        pass

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def timing_rows(runs_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = ["run_id"]
    seen = set(columns)
    for timing_path in sorted(runs_dir.glob("*/timing.jsonl")):
        run_id = timing_path.parent.name
        for item in load_jsonl(timing_path):
            row = {"run_id": run_id, **item}
            rows.append(row)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    return rows, columns


def leaderboard_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(row.get("agent"), row.get("model"), row.get("scaffold"))].append(row)

    leaderboard: list[dict[str, Any]] = []
    for (agent, model, scaffold), rows in groups.items():
        scores = [
            to_float(coalesce(row.get("normalized_score"), row.get("score"), row.get("success_rate")))
            for row in rows
        ]
        scores = [score for score in scores if score is not None]
        costs = [to_float(row.get("estimated_usd")) for row in rows]
        costs = [cost for cost in costs if cost is not None]
        leaderboard.append(
            {
                "agent": agent,
                "model": model,
                "scaffold": scaffold,
                "num_runs": len(rows),
                "mean_normalized_score": sum(scores) / len(scores) if scores else None,
                "mean_estimated_usd": sum(costs) / len(costs) if costs else None,
            }
        )
    leaderboard.sort(
        key=lambda row: (
            row["mean_normalized_score"] is None,
            -(row["mean_normalized_score"] or 0),
            str(row.get("agent") or ""),
        )
    )
    return leaderboard


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs"
    analysis_dir = repo_root / "analysis"

    run_rows = [flatten_run(path) for path in sorted(runs_dir.glob("*/run_summary.json"))]
    write_csv(analysis_dir / "runs.csv", run_rows, RUN_COLUMNS)

    curve_rows, curve_columns = timing_rows(runs_dir)
    if curve_rows:
        write_csv(analysis_dir / "curves.csv", curve_rows, curve_columns)

    leaderboard = leaderboard_rows(run_rows)
    write_csv(
        analysis_dir / "leaderboard.csv",
        leaderboard,
        ["agent", "model", "scaffold", "num_runs", "mean_normalized_score", "mean_estimated_usd"],
    )

    print(f"Wrote {analysis_dir / 'runs.csv'} ({len(run_rows)} rows)")
    if curve_rows:
        print(f"Wrote {analysis_dir / 'curves.csv'} ({len(curve_rows)} rows)")
    print(f"Wrote {analysis_dir / 'leaderboard.csv'} ({len(leaderboard)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
