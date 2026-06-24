from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

def ensure_robocasa_runtime() -> None:
    import json as _json
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    for rel in ("third_party/robocasa", "third_party/robosuite", "."):
        path = str((repo / rel).resolve())
        if path not in _sys.path:
            _sys.path.insert(0, path)
    _os.environ.setdefault("PYTHONPATH", _os.pathsep.join(_sys.path))
    try:
        import lerobot.datasets.utils as _utils
    except ModuleNotFoundError:
        return
    if hasattr(_utils, "write_info"):
        return

    def write_info(info: dict, root: str | _Path) -> None:
        root_path = _Path(root)
        path = root_path if root_path.name == "info.json" else root_path / "info.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(info, indent=2, sort_keys=True) + "\n")

    _utils.write_info = write_info



ensure_robocasa_runtime()

import robocasa.utils.lerobot_utils as LU  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RoboCasa trajectory-bank policy from demonstration actions.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode-source", choices=["train", "eval", "all"], default="train")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--embedding", choices=["rgb8", "rgb16"], default="rgb16")
    parser.add_argument("--eval-chunk", type=int, default=16)
    parser.add_argument("--select-by-episode-id", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    task_aliases = set(args.task_alias)

    rows: list[dict[str, Any]] = []
    max_len = 0
    start_time = time.monotonic()
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        if alias not in manifest_tasks:
            raise KeyError(f"{alias!r} missing from manifest")
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        episode_ids = _episode_ids(split_task, str(args.episode_source))
        for episode_id in episode_ids:
            actions = LU.get_episode_actions(dataset_root, int(episode_id)).astype(np.float32)
            rows.append(
                {
                    "alias": alias,
                    "task_id": int(split_task["task_id"]),
                    "episode_id": int(episode_id),
                    "actions": actions,
                    "embedding": _episode_embedding(dataset_root, int(episode_id), str(args.embedding)),
                }
            )
            max_len = max(max_len, int(actions.shape[0]))

    if not rows:
        raise ValueError("no trajectory-bank episodes selected")
    _validate_rows(rows)
    action_dim = int(rows[0]["actions"].shape[-1])
    action_bank = np.zeros((len(rows), max_len, action_dim), dtype=np.float32)
    lengths = np.zeros((len(rows),), dtype=np.int64)
    task_ids = np.zeros((len(rows),), dtype=np.int64)
    episode_ids = np.zeros((len(rows),), dtype=np.int64)
    embeddings = np.zeros((len(rows), rows[0]["embedding"].shape[0]), dtype=np.float32)
    aliases = []
    for idx, row in enumerate(rows):
        actions = np.asarray(row["actions"], dtype=np.float32)
        action_bank[idx, : actions.shape[0]] = actions
        if actions.shape[0] < max_len:
            action_bank[idx, actions.shape[0] :] = actions[-1]
        lengths[idx] = int(actions.shape[0])
        task_ids[idx] = int(row["task_id"])
        episode_ids[idx] = int(row["episode_id"])
        embeddings[idx] = np.asarray(row["embedding"], dtype=np.float32)
        aliases.append(str(row["alias"]))

    payload = {
        "policy_type": "robocasa_bc5_trajectory_bank",
        "actions": torch.as_tensor(action_bank, dtype=torch.float32),
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
        "task_ids": torch.as_tensor(task_ids, dtype=torch.long),
        "episode_ids": torch.as_tensor(episode_ids, dtype=torch.long),
        "embeddings": torch.as_tensor(embeddings, dtype=torch.float32),
        "aliases": aliases,
        "task_count": int(max(task_ids.tolist()) + 1),
        "embedding": str(args.embedding),
        "eval_chunk": int(args.eval_chunk),
        "select_by_episode_id": bool(args.select_by_episode_id),
        "manifest": str(args.manifest),
        "split": str(args.split),
        "episode_source": str(args.episode_source),
        "build_seconds": float(time.monotonic() - start_time),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    metrics = {
        "checkpoint": str(out),
        "policy_type": payload["policy_type"],
        "episodes": int(len(rows)),
        "max_len": int(max_len),
        "action_dim": int(action_dim),
        "eval_chunk": int(args.eval_chunk),
        "select_by_episode_id": bool(args.select_by_episode_id),
        "build_seconds": payload["build_seconds"],
        "per_task": {
            alias: int(sum(1 for row in rows if row["alias"] == alias))
            for alias in sorted(set(row["alias"] for row in rows))
        },
    }
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _episode_ids(split_task: dict, source: str) -> list[int]:
    if source == "train":
        ids = split_task.get("train_episode_ids", [])
    elif source == "eval":
        ids = split_task.get("eval_episode_ids", [])
    else:
        ids = sorted(set(split_task.get("train_episode_ids", [])) | set(split_task.get("eval_episode_ids", [])))
    return [int(x) for x in ids]


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    action_dim = int(np.asarray(rows[0]["actions"]).shape[-1])
    embedding_dim = int(np.asarray(rows[0]["embedding"]).shape[0])
    for row in rows:
        actions = np.asarray(row["actions"])
        embedding = np.asarray(row["embedding"])
        if actions.ndim != 2:
            raise ValueError(
                f"episode {row['episode_id']} for {row['alias']} has action shape {actions.shape}; expected [T, A]"
            )
        if int(actions.shape[-1]) != action_dim:
            raise ValueError(
                f"episode {row['episode_id']} for {row['alias']} has action_dim={actions.shape[-1]}, expected {action_dim}"
            )
        if int(embedding.shape[0]) != embedding_dim:
            raise ValueError(
                f"episode {row['episode_id']} for {row['alias']} has embedding_dim={embedding.shape[0]}, expected {embedding_dim}"
            )


def _episode_embedding(dataset_root: Path, episode_id: int, embedding: str) -> np.ndarray:
    size = 8 if embedding == "rgb8" else 16
    parts = []
    for view in ("robot0_agentview_left", "robot0_agentview_right"):
        video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_id:06d}.mp4"
        frame = np.asarray(next(iio.imiter(video_path)), dtype=np.uint8)[..., :3]
        small = Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR)
        parts.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
    return np.concatenate(parts).astype(np.float32)


if __name__ == "__main__":
    main()
