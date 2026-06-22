from __future__ import annotations

from tasks.robocasa_bc5.visualize import main_with_defaults


if __name__ == "__main__":
    main_with_defaults(
        task_name="robocasa_bc1",
        eval_script="tasks/robocasa_bc1/eval.py",
        inference="tasks.robocasa_bc1.inference",
        manifest="data/autorobobench/robocasa_bc1_manifest.json",
        split="data/autorobobench/robocasa_bc1_splits.json",
        max_steps="750",
        commit_steps="8",
    )
