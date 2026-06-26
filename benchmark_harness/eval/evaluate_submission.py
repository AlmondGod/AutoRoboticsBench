#!/usr/bin/env python3
"""Placeholder clean-container evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .privileged_state_checks import check_privileged_state_use
    from .replay_detection import detect_replay
    from .validators import validate_submission
except ImportError:
    from privileged_state_checks import check_privileged_state_use
    from replay_detection import detect_replay
    from validators import validate_submission


def clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


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


def primary_score(data: dict) -> float:
    metric = data.get("metric")
    if isinstance(metric, str) and metric in data:
        return clamp_score(data.get(metric))
    for key in PRIMARY_SCORE_KEYS:
        if key in data:
            return clamp_score(data.get(key))
    return 0.0


def evaluate_result_json(submission: Path) -> tuple[float, float | None, int]:
    result_path = submission / "result.json"
    if not result_path.exists():
        return 0.0, 0.0, 0
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0, 0.0, 0
    episodes = data.get("num_eval_episodes", data.get("num_episodes", data.get("episodes", 0)))
    try:
        num_episodes = int(episodes)
    except (TypeError, ValueError):
        num_episodes = 0
    success_rate = clamp_score(data.get("success_rate")) if "success_rate" in data else None
    return primary_score(data), success_rate, num_episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    submission = Path(args.submission)
    valid, flags = validate_submission(submission)
    flags.extend(detect_replay(submission))
    flags.extend(check_privileged_state_use(submission))

    score = 0.0
    success_rate = 0.0
    num_episodes = 0
    if valid:
        score, success_rate, num_episodes = evaluate_result_json(submission)

    result = {
        "task": args.task,
        "score": score,
        "success_rate": success_rate,
        "valid": valid,
        "flags": flags,
        "num_episodes": num_episodes,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
