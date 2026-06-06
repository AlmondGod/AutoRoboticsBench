from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robotbench.logging import mark_ledger_decision, write_json


MIN_RELATIVE_IMPROVEMENT = 0.02
MAX_CATASTROPHE_REGRESSION = 0.0
MAX_SAFETY_REGRESSION = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = judge(Path(args.baseline), Path(args.candidate))
    write_json(Path(args.candidate) / "decision.json", result)
    mark_ledger_decision(result["candidate_run"], bool(result["accepted"]))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["accepted"] else 1)


def judge(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline = _load_metrics(baseline_dir)
    candidate = _load_metrics(candidate_dir)
    base_summary = baseline["summary"]
    cand_summary = candidate["summary"]

    baseline_score = float(base_summary["eval_score"])
    candidate_score = float(cand_summary["eval_score"])
    required = baseline_score + max(abs(baseline_score) * MIN_RELATIVE_IMPROVEMENT, 0.01)
    catastrophe_ok = (
        float(cand_summary["eval_catastrophe_rate"])
        <= float(base_summary["eval_catastrophe_rate"]) + MAX_CATASTROPHE_REGRESSION
    )
    safety_ok = (
        float(cand_summary["eval_torque_limit_violations"])
        <= float(base_summary["eval_torque_limit_violations"]) + MAX_SAFETY_REGRESSION
    )
    accepted = candidate_score >= required and catastrophe_ok and safety_ok

    return {
        "accepted": accepted,
        "baseline_run": baseline["run_id"],
        "candidate_run": candidate["run_id"],
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "required_score": required,
        "score_delta": candidate_score - baseline_score,
        "catastrophe_ok": catastrophe_ok,
        "safety_ok": safety_ok,
    }


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing metrics file: {path}")
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
