#!/usr/bin/env python3
"""Rule-based judge for a benchmark run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SUSPICIOUS_STRINGS = [
    "heldout",
    "eval_seed",
    "test_episodes",
    "evaluate_submission.py",
    "success_detector.py",
]


def scan_tree(root: Path) -> list[dict]:
    warnings: list[dict] = []
    if not root.exists():
        return warnings

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = text.lower()
        for needle in SUSPICIOUS_STRINGS:
            if needle.lower() in lowered:
                warnings.append({"file": str(path), "match": needle})
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / args.run_id
    final_submission = run_dir / "output" / "final_submission"

    warnings = scan_tree(run_dir / "task") + scan_tree(run_dir / "output")
    report = {
        "run_id": args.run_id,
        "task": args.task,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "valid": final_submission.exists(),
        "final_submission_exists": final_submission.exists(),
        "warnings": warnings,
    }

    out_path = run_dir / "judge_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote judge report to {out_path}")

    return 0 if final_submission.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
