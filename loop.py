from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from judge import judge
from robotbench.logging import mark_ledger_decision, write_json
from run_experiment import run_experiment


RESEARCH_FILE = Path("research.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=["reach", "push"])
    parser.add_argument("--backend", default="toy", choices=["toy", "mujoco", "aloha", "mobile_aloha_mock"])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--budget-seconds", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--baseline", default="")
    args = parser.parse_args()

    baseline = Path(args.baseline) if args.baseline else None
    for idx in range(args.iterations):
        backup = RESEARCH_FILE.read_text()
        run = run_experiment(
            task_name=args.task,
            backend=args.backend,
            budget_seconds=args.budget_seconds,
            seeds=args.seeds,
            episodes_per_seed=5,
            max_iterations=30,
            rollouts_per_iteration=24,
            change_note=f"loop iteration {idx}",
        )
        run_dir = Path("runs") / run["run_id"]
        if baseline is None:
            baseline = run_dir
            print(f"initialized baseline: {baseline}")
            continue

        decision = judge(baseline, run_dir)
        write_json(run_dir / "decision.json", decision)
        mark_ledger_decision(run["run_id"], bool(decision["accepted"]))
        print(decision)
        if decision["accepted"]:
            baseline = run_dir
        else:
            RESEARCH_FILE.write_text(backup)
            subprocess.run(["git", "diff", "--", str(RESEARCH_FILE)], check=False)
            shutil.rmtree(run_dir, ignore_errors=False)


if __name__ == "__main__":
    main()
