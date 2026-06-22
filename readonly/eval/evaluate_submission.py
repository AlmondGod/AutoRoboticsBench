#!/usr/bin/env python3
"""Placeholder clean-container evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from privileged_state_checks import check_privileged_state_use
from replay_detection import detect_replay
from validators import validate_submission


def clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def evaluate_result_json(submission: Path) -> tuple[float, int]:
    result_path = submission / "result.json"
    if not result_path.exists():
        return 0.0, 0
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0, 0
    episodes = data.get("num_eval_episodes", data.get("num_episodes", 0))
    try:
        num_episodes = int(episodes)
    except (TypeError, ValueError):
        num_episodes = 0
    return clamp_score(data.get("score", data.get("success_rate"))), num_episodes


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

    success = 0.0
    num_episodes = 0
    if valid:
        success, num_episodes = evaluate_result_json(submission)

    result = {
        "task": args.task,
        "success_rate": success,
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
