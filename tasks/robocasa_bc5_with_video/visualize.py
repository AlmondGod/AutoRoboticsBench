from __future__ import annotations

from tasks.robocasa_bc5.visualize import main_with_defaults


if __name__ == "__main__":
    main_with_defaults(
        task_name="robocasa_bc5_with_video",
        eval_script="tasks/robocasa_bc5_with_video/eval.py",
        inference="tasks.robocasa_bc5_with_video.inference",
        manifest="data/robocasa5/manifest.json",
        split="data/autorobobench/robocasa_bc5_with_video_splits.json",
        max_steps="260",
        commit_steps="16",
    )
