from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / "data" / "autorobobench" / "robotwin_bc3_splits.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "autorobobench" / "robotwin_bc3"
DEFAULT_DATASET_REPO = "lerobot/robotwin_unified"
DEFAULT_PRETRAINED_POLICY = "lerobot/smolvla_robotwin"
DEFAULT_BASE_POLICY = "lerobot/smolvla_base"
EPISODES_PER_TASK = 550
DEFAULT_TRAIN_DEMOS_PER_TASK = 50
DEFAULT_FINETUNE_SECONDS = 300

RENAME_MAP = {
    "observation.images.cam_high": "observation.images.camera1",
    "observation.images.cam_left_wrist": "observation.images.camera2",
    "observation.images.cam_right_wrist": "observation.images.camera3",
}


@dataclass(frozen=True)
class RobotwinTaskSpec:
    alias: str
    robotwin_task: str
    task_index: int
    description: str
    language_check: tuple[str, ...]

    @property
    def start_episode(self) -> int:
        return self.task_index * EPISODES_PER_TASK

    @property
    def end_episode_exclusive(self) -> int:
        return self.start_episode + EPISODES_PER_TASK


TASKS = (
    RobotwinTaskSpec(
        alias="blocks_ranking_rgb",
        robotwin_task="blocks_ranking_rgb",
        task_index=2,
        description="Arrange the red, green, and blue blocks from left to right.",
        language_check=("red", "green", "blue"),
    ),
    RobotwinTaskSpec(
        alias="place_a2b_left",
        robotwin_task="place_a2b_left",
        task_index=20,
        description="Place object A to the left of object B.",
        language_check=("left",),
    ),
    RobotwinTaskSpec(
        alias="place_object_basket",
        robotwin_task="place_object_basket",
        task_index=32,
        description="Place the target object into a basket and move the basket.",
        language_check=("basket",),
    ),
)


def task_names() -> list[str]:
    return [task.robotwin_task for task in TASKS]


def load_split(path: str | Path = DEFAULT_SPLIT) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def split_train_episode_ids(split: dict[str, Any], *, demos_per_task: int | None = None) -> list[int]:
    episodes: list[int] = []
    for task in split["tasks"]:
        ids = list(task["train_episode_ids"])
        if demos_per_task is not None:
            ids = ids[: int(demos_per_task)]
        episodes.extend(int(item) for item in ids)
    return episodes


def split_eval_task_names(split: dict[str, Any]) -> list[str]:
    return [str(task["robotwin_task"]) for task in split["tasks"]]


def require_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"required command not found on PATH: {name}")
    return path


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"cmd": cmd}, indent=2), flush=True)
    return subprocess.run(cmd, check=False, text=True, env=env)


def parse_eval_info(path: str | Path) -> dict[str, Any]:
    info = json.loads(Path(path).read_text())
    overall = info.get("overall") or info.get("aggregated") or {}
    pc_success = overall.get("pc_success")
    success_rate = None if pc_success is None else float(pc_success) / 100.0
    per_task = {}
    for task in info.get("per_group", {}):
        row = info["per_group"][task]
        pc = row.get("pc_success")
        per_task[task] = {
            "episodes": int(row.get("n_episodes", 0)),
            "success_rate": None if pc is None else float(pc) / 100.0,
            "pc_success": pc,
        }
    return {
        "success_rate": success_rate,
        "pc_success": pc_success,
        "episodes": int(overall.get("n_episodes", 0)),
        "per_task": per_task,
        "raw": info,
    }


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    checkpoint_root = Path(output_dir) / "checkpoints"
    last = checkpoint_root / "last" / "pretrained_model"
    if last.exists():
        return last
    candidates = sorted(checkpoint_root.glob("*/pretrained_model"))
    return candidates[-1] if candidates else None


def task_payload() -> list[dict[str, Any]]:
    return [asdict(task) | {"episode_range": [task.start_episode, task.end_episode_exclusive]} for task in TASKS]


def ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

