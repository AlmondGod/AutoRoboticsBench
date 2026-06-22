from __future__ import annotations

from tasks.robocasa_bc5.visualize import main_with_defaults


if __name__ == "__main__":
    main_with_defaults(
        task_name="robocasa_language_following",
        eval_script="tasks/robocasa_language_following/eval.py",
        inference="tasks.robocasa_language_following.inference",
        manifest="data/autorobobench/robocasa_language_following_manifest.json",
        split="data/autorobobench/robocasa_language_following_splits.json",
        max_steps="900",
        commit_steps="8",
    )
