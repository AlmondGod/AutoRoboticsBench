#!/usr/bin/env python3
"""Evaluate a final_submission directory with a task's real evaluator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CHECKPOINT_NAMES = (
    "policy_best.pt",
    "policy.pt",
    "policy_last.pt",
    "checkpoint.pt",
    "model.pt",
    "world_model.pt",
)


def find_checkpoint(submission: Path) -> Path | None:
    for name in CHECKPOINT_NAMES:
        candidate = submission / name
        if candidate.exists() and candidate.is_file():
            return candidate
    checkpoints = sorted(submission.glob("*.pt"))
    return checkpoints[0] if checkpoints else None


def write_invalid(out_path: Path, task: str, flags: list[str]) -> None:
    result = {
        "task": task,
        "score": 0.0,
        "success_rate": 0.0,
        "valid": False,
        "flags": flags,
        "num_episodes": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def annotate_result(out_path: Path, task: str, valid: bool, flags: list[str]) -> None:
    if not out_path.exists():
        write_invalid(out_path, task, [*flags, "eval_result_missing"])
        return
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        write_invalid(out_path, task, [*flags, "eval_result_invalid_json"])
        return
    payload.setdefault("task", task)
    payload["valid"] = bool(valid)
    payload["flags"] = list(flags)
    if "num_episodes" not in payload and "episodes" in payload:
        payload["num_episodes"] = payload.get("episodes")
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    submission = Path(args.submission).resolve()
    out_path = Path(args.out).resolve()
    task_dir = repo_root / "tasks" / args.task
    eval_script = task_dir / "eval_parallel.py"
    if not eval_script.exists():
        eval_script = task_dir / "eval.py"
    if not eval_script.exists():
        write_invalid(out_path, args.task, ["eval_script_missing"])
        return 1

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from benchmark_harness.eval.privileged_state_checks import check_privileged_state_use
    from benchmark_harness.eval.replay_detection import detect_replay
    from benchmark_harness.eval.validators import validate_submission

    valid, flags = validate_submission(submission)
    flags.extend(detect_replay(submission))
    flags.extend(check_privileged_state_use(submission))
    if flags:
        valid = False
    if not valid:
        write_invalid(out_path, args.task, flags)
        return 1

    checkpoint = find_checkpoint(submission)
    if checkpoint is None:
        write_invalid(out_path, args.task, [*flags, "checkpoint_missing"])
        return 1

    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out_path),
    ]
    env = os.environ.copy()
    env["ROBOAUTORESEARCH_REPO_ROOT"] = str(repo_root)
    if (submission / "inference.py").exists():
        cmd.extend(["--inference", "inference"])
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(submission) if not existing_pythonpath else f"{submission}{os.pathsep}{existing_pythonpath}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=repo_root, env=env)
    annotate_result(out_path, args.task, valid, flags)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
