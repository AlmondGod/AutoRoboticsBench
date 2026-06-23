#!/usr/bin/env python3
"""Append an eval checkpoint to runs/<RUN_ID>/timing.jsonl."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=os.environ.get("ROBOAUTORESEARCH_RUN_ID", ""))
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--experiment", type=int, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if not args.run_id:
        raise SystemExit("--run-id is required when ROBOAUTORESEARCH_RUN_ID is unset")

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / args.run_id
    eval_path = Path(args.eval_json)
    if not eval_path.is_absolute():
        eval_path = repo_root / eval_path
    if not eval_path.exists():
        raise SystemExit(f"Eval JSON not found: {eval_path}")

    payload = read_json(eval_path)
    commands = read_jsonl(run_dir / "commands.jsonl")
    started_epoch = to_float((run_dir / "started_epoch.txt").read_text().strip()) if (run_dir / "started_epoch.txt").exists() else None
    now_epoch = datetime.now(timezone.utc).timestamp()
    wall_seconds = now_epoch - started_epoch if started_epoch is not None else None
    container_seconds = sum(float(row.get("duration_seconds") or 0) for row in commands) if commands else None

    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "eval_json": str(eval_path),
        "score": primary_score(payload),
        "success_rate": to_float(payload.get("success_rate")),
        "normalized_score": primary_score(payload),
        "wall_seconds": wall_seconds,
        "container_seconds": container_seconds,
        "episodes": payload.get("episodes") or payload.get("num_episodes") or payload.get("num_eval_episodes"),
        "successes": payload.get("successes"),
    }
    if args.label:
        row["label"] = args.label
    if args.experiment is not None:
        row["experiment"] = int(args.experiment)
    if args.notes:
        row["notes"] = args.notes

    timing_path = run_dir / "timing.jsonl"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with timing_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
