#!/usr/bin/env python3
"""Evaluate a RunPod dockerless submission in-process."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / args.run_id
    submission = run_dir / "output" / "final_submission"
    out_path = run_dir / "eval" / "results.json"

    cmd = [
        sys.executable,
        str(repo_root / "readonly" / "eval" / "evaluate_submission.py"),
        "--task",
        args.task,
        "--submission",
        str(submission),
        "--out",
        str(out_path),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
