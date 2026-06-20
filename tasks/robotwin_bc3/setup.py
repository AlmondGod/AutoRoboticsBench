from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robotwin_bc3.common import (
    DEFAULT_DATASET_REPO,
    DEFAULT_SPLIT,
    DEFAULT_TRAIN_DEMOS_PER_TASK,
    EPISODES_PER_TASK,
    RENAME_MAP,
    TASKS,
    require_command,
    task_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and verify RoboTwin BC-3 metadata.")
    parser.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--train-demos-per-task", type=int, default=DEFAULT_TRAIN_DEMOS_PER_TASK)
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    split = build_split(str(args.repo_id), int(args.train_demos_per_task))
    out = Path(args.split)
    if not args.no_write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")

    payload = {
        "task": "robotwin_bc3",
        "repo_id": str(args.repo_id),
        "split": str(out),
        "wrote_split": not args.no_write,
        "runtime": verify_runtime() if args.verify_runtime else None,
        "tasks": [
            {
                "alias": task["alias"],
                "robotwin_task": task["robotwin_task"],
                "train_episodes": len(task["train_episode_ids"]),
                "heldout_demo_episodes": len(task["heldout_demo_episode_ids"]),
                "sample_instructions": task["sample_instructions"],
            }
            for task in split["tasks"]
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_split(repo_id: str, train_demos_per_task: int) -> dict:
    import pandas as pd

    info_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/info.json")
    episodes_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="meta/episodes/chunk-000/file-000.parquet",
    )
    info = json.loads(Path(info_path).read_text())
    episodes = pd.read_parquet(episodes_path)
    total_episodes = int(info["total_episodes"])
    if total_episodes < max(task.end_episode_exclusive for task in TASKS):
        raise ValueError(f"{repo_id} has only {total_episodes} episodes, not enough for BC-3 task ranges")
    if train_demos_per_task <= 0 or train_demos_per_task > EPISODES_PER_TASK:
        raise ValueError(f"train_demos_per_task must be in 1..{EPISODES_PER_TASK}")

    split_tasks = []
    for task in TASKS:
        task_rows = episodes.iloc[task.start_episode : task.end_episode_exclusive]
        episode_ids = [int(item) for item in task_rows["episode_index"].tolist()]
        if len(episode_ids) != EPISODES_PER_TASK:
            raise ValueError(f"{task.alias} expected {EPISODES_PER_TASK} episodes, got {len(episode_ids)}")
        sample_instructions = [
            str(task_rows.iloc[idx]["tasks"][0])
            for idx in sorted({0, min(49, len(task_rows) - 1), min(50, len(task_rows) - 1), len(task_rows) - 1})
        ]
        _validate_language(task.alias, sample_instructions, task.language_check)
        split_tasks.append(
            {
                "alias": task.alias,
                "robotwin_task": task.robotwin_task,
                "task_index": int(task.task_index),
                "description": task.description,
                "episode_range": [task.start_episode, task.end_episode_exclusive],
                "train_episode_ids": episode_ids[:train_demos_per_task],
                "heldout_demo_episode_ids": episode_ids[train_demos_per_task:],
                "sample_instructions": sample_instructions,
            }
        )

    return {
        "task": "robotwin_bc3",
        "repo_id": repo_id,
        "format": "lerobot_v3",
        "episodes_per_task": EPISODES_PER_TASK,
        "train_demos_per_task": int(train_demos_per_task),
        "rename_map": RENAME_MAP,
        "robot": "Aloha-AgileX bimanual",
        "action_dim": 14,
        "video_columns": [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        "tasks": split_tasks,
    }


def _validate_language(alias: str, samples: list[str], required_terms: tuple[str, ...]) -> None:
    text = " ".join(samples).lower()
    missing = [term for term in required_terms if term.lower() not in text]
    if missing:
        raise ValueError(f"{alias} sample language is missing expected terms: {missing}; samples={samples}")


def verify_runtime() -> dict:
    runtime = {
        "lerobot_train": shutil.which("lerobot-train"),
        "lerobot_eval": shutil.which("lerobot-eval"),
        "torch_import": False,
        "lerobot_import": False,
        "robotwin_env_import": False,
        "notes": [],
    }
    require_command("lerobot-train")
    require_command("lerobot-eval")
    try:
        import torch  # noqa: F401

        runtime["torch_import"] = True
    except Exception as exc:  # pragma: no cover - runtime diagnostic
        runtime["notes"].append(f"torch import failed: {type(exc).__name__}: {exc}")
    try:
        import lerobot  # noqa: F401

        runtime["lerobot_import"] = True
    except Exception as exc:  # pragma: no cover - runtime diagnostic
        runtime["notes"].append(f"lerobot import failed: {type(exc).__name__}: {exc}")
    try:
        import lerobot.envs.robotwin  # noqa: F401

        runtime["robotwin_env_import"] = True
    except Exception as exc:  # pragma: no cover - runtime diagnostic
        runtime["notes"].append(f"lerobot.envs.robotwin import failed: {type(exc).__name__}: {exc}")
    runtime["bc3_tasks"] = task_payload()
    return runtime


if __name__ == "__main__":
    main()
