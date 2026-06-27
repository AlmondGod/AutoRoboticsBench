from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


DEFAULT_MANIFEST = ROOT / "data" / "robocasa5" / "manifest.json"
DEFAULT_SPLIT = ROOT / "data" / "autorobobench" / "robocasa_bc5_splits.json"
DEFAULT_POLICY_SET = ROOT / "data" / "autorobobench" / "robocasa_world_model_policy_set.json"
DEFAULT_VIDEO_POOL = ROOT / "data" / "autorobobench" / "robocasa_world_model_video_pool.json"
DEFAULT_FAILED_ROLLOUT_SPLIT = ROOT / "data" / "autorobobench" / "robocasa_visual_world_model_failed_rollouts.json"


@dataclass
class TransitionData:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    progress: np.ndarray
    next_progress: np.ndarray
    success: np.ndarray
    task_id: np.ndarray
    episode_id: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.state.shape[0])


@dataclass(frozen=True)
class VideoOnlyEpisode:
    alias: str
    task_id: int
    split: str
    episode_id: int
    view: str
    video_path: Path


def load_transition_data(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    split_path: str | Path = DEFAULT_SPLIT,
    train_episodes_per_task: int = 20,
    val_episodes_per_task: int = 5,
    task_aliases: set[str] | None = None,
    frame_stride: int = 1,
    failed_rollout_split_path: str | Path | None = DEFAULT_FAILED_ROLLOUT_SPLIT,
) -> tuple[TransitionData, TransitionData, list[dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_path).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    aliases = task_aliases or set()
    train_parts = []
    val_parts = []
    summary = []
    for split_task in split["tasks"]:
        alias = str(split_task["alias"])
        if aliases and alias not in aliases:
            continue
        task_id = int(split_task["task_id"])
        dataset_root = _resolve_dataset_root(manifest_tasks[alias]["dataset_path"])
        all_train_ids = [int(x) for x in split_task["train_episode_ids"]]
        all_val_ids = [int(x) for x in split_task["val_episode_ids"]]
        train_limit = int(train_episodes_per_task)
        val_limit = int(val_episodes_per_task)
        train_ids = all_train_ids if train_limit <= 0 else all_train_ids[:train_limit]
        val_ids = all_val_ids if val_limit <= 0 else all_val_ids[:val_limit]
        train_count = _append_episodes(train_parts, dataset_root, train_ids, task_id, int(frame_stride))
        val_count = _append_episodes(val_parts, dataset_root, val_ids, task_id, int(frame_stride))
        summary.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str(dataset_root),
                "train_episode_ids": train_ids,
                "val_episode_ids": val_ids,
                "train_transitions": int(train_count),
                "val_transitions": int(val_count),
            }
        )
    _append_failed_rollout_split(
        train_parts,
        summary,
        failed_rollout_split_path=failed_rollout_split_path,
        task_aliases=aliases,
        frame_stride=int(frame_stride),
    )
    return _concat(train_parts), _concat(val_parts), summary


def _append_failed_rollout_split(
    train_parts: list[dict[str, np.ndarray]],
    summary: list[dict[str, Any]],
    *,
    failed_rollout_split_path: str | Path | None,
    task_aliases: set[str],
    frame_stride: int,
) -> None:
    if not failed_rollout_split_path:
        return
    path = Path(failed_rollout_split_path)
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    for row in payload.get("tasks", []):
        alias = str(row["alias"])
        if task_aliases and alias not in task_aliases:
            continue
        task_id = int(row["task_id"])
        dataset_root = _resolve_dataset_root(row["dataset_path"])
        train_ids = [int(x) for x in row.get("train_episode_ids", [])]
        train_count = _append_episodes(train_parts, dataset_root, train_ids, task_id, int(frame_stride))
        summary.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str(dataset_root),
                "train_episode_ids": train_ids,
                "val_episode_ids": [],
                "failed_rollout_train_episode_ids": train_ids,
                "train_transitions": int(train_count),
                "val_transitions": 0,
                "source": "failed_policy_rollout",
            }
        )


def load_video_only_pool(
    video_pool_path: str | Path = DEFAULT_VIDEO_POOL,
    *,
    max_episodes_per_task: int = 0,
    task_aliases: set[str] | None = None,
    splits: set[str] | None = None,
) -> list[VideoOnlyEpisode]:
    """Return RGB video-only records without reading action/state parquet data."""
    pool = json.loads(Path(video_pool_path).read_text())
    aliases = task_aliases or set()
    wanted_splits = splits or set()
    template = str(pool.get("video_path_template", "videos/chunk-000/observation.images.{view}/episode_{episode_id:06d}.mp4"))
    records: list[VideoOnlyEpisode] = []
    for task in pool.get("tasks", []):
        alias = str(task["alias"])
        split = str(task.get("split", ""))
        if aliases and alias not in aliases:
            continue
        if wanted_splits and split not in wanted_splits:
            continue
        start, end = [int(x) for x in task["video_episode_range"]]
        episode_ids = list(range(start, end + 1))
        if int(max_episodes_per_task) > 0:
            episode_ids = episode_ids[: int(max_episodes_per_task)]
        dataset_root = Path(str(task["dataset_path"]))
        if not dataset_root.is_absolute():
            dataset_root = ROOT / dataset_root
        for episode_id in episode_ids:
            for view in pool.get("views", []):
                rel = template.format(view=str(view), episode_id=int(episode_id))
                records.append(
                    VideoOnlyEpisode(
                        alias=alias,
                        task_id=int(task["task_id"]),
                        split=split,
                        episode_id=int(episode_id),
                        view=str(view),
                        video_path=dataset_root / rel,
                    )
                )
    return records


def _resolve_dataset_root(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    parts = path.parts
    marker = ("third_party", "robocasa", "datasets")
    for idx in range(0, max(0, len(parts) - len(marker) + 1)):
        if tuple(parts[idx : idx + len(marker)]) == marker:
            remapped = ROOT.joinpath(*parts[idx:])
            if remapped.exists():
                return remapped
            return remapped
    if not path.is_absolute():
        return ROOT / path
    return path


def summarize_video_only_pool(records: list[VideoOnlyEpisode]) -> dict[str, Any]:
    by_task: dict[tuple[str, str], set[int]] = {}
    existing_videos = 0
    for record in records:
        by_task.setdefault((record.alias, record.split), set()).add(int(record.episode_id))
        if record.video_path.exists():
            existing_videos += 1
    return {
        "video_records": len(records),
        "video_files_existing": existing_videos,
        "video_episodes": sum(len(ids) for ids in by_task.values()),
        "tasks": [
            {
                "alias": alias,
                "split": split,
                "video_episodes": len(ids),
            }
            for (alias, split), ids in sorted(by_task.items())
        ],
    }


def load_video_frames(video_path: str | Path, *, stride: int = 1, max_frames: int = 0) -> np.ndarray:
    """Load RGB frames from a video-only record for optional self-supervised methods."""
    path = Path(video_path)
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        frames = []
        index = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if index % max(1, int(stride)) == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if int(max_frames) > 0 and len(frames) >= int(max_frames):
                    break
            index += 1
        cap.release()
        return np.asarray(frames, dtype=np.uint8)
    except ModuleNotFoundError:
        import imageio.v3 as iio

        frames = []
        for index, frame in enumerate(iio.imiter(path)):
            if index % max(1, int(stride)) == 0:
                frames.append(np.asarray(frame, dtype=np.uint8))
                if int(max_frames) > 0 and len(frames) >= int(max_frames):
                    break
        return np.asarray(frames, dtype=np.uint8)


def load_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    path = Path(video_path)
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise IndexError(f"could not read frame {frame_idx} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    except ModuleNotFoundError:
        import imageio.v3 as iio

        for index, frame in enumerate(iio.imiter(path)):
            if index == int(frame_idx):
                return np.asarray(frame, dtype=np.uint8)
        raise IndexError(f"could not read frame {frame_idx} from {path}")


def _append_episodes(parts: list[dict[str, np.ndarray]], dataset_root: Path, episode_ids: list[int], task_id: int, frame_stride: int) -> int:
    count = 0
    for episode_id in episode_ids:
        part = load_episode_transitions(dataset_root, int(episode_id), int(task_id), frame_stride=max(1, frame_stride))
        if part["state"].shape[0] > 0:
            parts.append(part)
            count += int(part["state"].shape[0])
    return count


def load_episode_transitions(dataset_root: Path, episode_id: int, task_id: int, *, frame_stride: int = 1) -> dict[str, np.ndarray]:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    frame = pd.read_parquet(episode_path)
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    action = episode_actions(dataset_root, episode_id, frame).astype(np.float32)
    n = min(len(state), len(action))
    if n <= 1:
        return _empty_part(state_dim=state.shape[-1] if state.ndim == 2 else 1, action_dim=action.shape[-1] if action.ndim == 2 else 1)
    rows = np.arange(0, n - 1, max(1, frame_stride), dtype=np.int32)
    progress = rows.astype(np.float32) / max(1, n - 1)
    next_progress = (rows + 1).astype(np.float32) / max(1, n - 1)
    success = _episode_success(frame, rows, n)
    if not _trajectory_succeeded(frame, success):
        progress = np.zeros_like(progress, dtype=np.float32)
        next_progress = np.zeros_like(next_progress, dtype=np.float32)
    return {
        "state": state[rows].astype(np.float32),
        "action": action[rows].astype(np.float32),
        "next_state": state[rows + 1].astype(np.float32),
        "progress": progress[:, None].astype(np.float32),
        "next_progress": next_progress[:, None].astype(np.float32),
        "success": success[:, None].astype(np.float32),
        "task_id": np.full((len(rows),), int(task_id), dtype=np.int64),
        "episode_id": np.full((len(rows),), int(episode_id), dtype=np.int32),
        "frame_idx": rows.astype(np.int32),
    }


def _episode_success(frame: pd.DataFrame, rows: np.ndarray, n: int) -> np.ndarray:
    for key in ("next.success", "success", "is_success"):
        if key in frame:
            values = np.asarray(frame[key].to_numpy(), dtype=np.float32).reshape(-1)
            return values[np.minimum(rows + 1, len(values) - 1)]
    success = np.zeros((len(rows),), dtype=np.float32)
    if len(success):
        success[-1] = 1.0
    return success


def _trajectory_succeeded(frame: pd.DataFrame, transition_success: np.ndarray) -> bool:
    for key in ("next.success", "success", "is_success"):
        if key in frame:
            values = np.asarray(frame[key].to_numpy(), dtype=np.float32).reshape(-1)
            return bool(len(values) and float(np.nanmax(values)) > 0.5)
    return bool(len(transition_success) and float(np.nanmax(transition_success)) > 0.5)


def episode_actions(dataset_root: Path, episode_id: int, frame: pd.DataFrame | None = None) -> np.ndarray:
    if frame is None:
        episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        frame = pd.read_parquet(episode_path)
    if "action" in frame:
        return np.stack(frame["action"].to_numpy()).astype(np.float32)
    try:
        ensure_robocasa_runtime()
        import robocasa.utils.lerobot_utils as LU

        return LU.get_episode_actions(dataset_root, episode_id).astype(np.float32)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "episode parquet has no action column and RoboCasa is not importable for lerobot_utils fallback"
        ) from exc


def _concat(parts: list[dict[str, np.ndarray]]) -> TransitionData:
    if not parts:
        return TransitionData(
            state=np.zeros((0, 1), dtype=np.float32),
            action=np.zeros((0, 1), dtype=np.float32),
            next_state=np.zeros((0, 1), dtype=np.float32),
            progress=np.zeros((0, 1), dtype=np.float32),
            next_progress=np.zeros((0, 1), dtype=np.float32),
            success=np.zeros((0, 1), dtype=np.float32),
            task_id=np.zeros((0,), dtype=np.int64),
            episode_id=np.zeros((0,), dtype=np.int32),
            frame_idx=np.zeros((0,), dtype=np.int32),
        )
    return TransitionData(**{key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]})


def _empty_part(state_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    return {
        "state": np.zeros((0, int(state_dim)), dtype=np.float32),
        "action": np.zeros((0, int(action_dim)), dtype=np.float32),
        "next_state": np.zeros((0, int(state_dim)), dtype=np.float32),
        "progress": np.zeros((0, 1), dtype=np.float32),
        "next_progress": np.zeros((0, 1), dtype=np.float32),
        "success": np.zeros((0, 1), dtype=np.float32),
        "task_id": np.zeros((0,), dtype=np.int64),
        "episode_id": np.zeros((0,), dtype=np.int32),
        "frame_idx": np.zeros((0,), dtype=np.int32),
    }


def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def normalize_data(data: TransitionData, stats: dict[str, np.ndarray]) -> TransitionData:
    return TransitionData(
        state=((data.state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
        action=((data.action - stats["action_mean"]) / stats["action_std"]).astype(np.float32),
        next_state=((data.next_state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
        progress=data.progress.astype(np.float32),
        next_progress=data.next_progress.astype(np.float32),
        success=data.success.astype(np.float32),
        task_id=data.task_id.astype(np.int64),
        episode_id=data.episode_id.astype(np.int32),
        frame_idx=data.frame_idx.astype(np.int32),
    )


def make_stats(train: TransitionData) -> dict[str, np.ndarray]:
    state_mean, state_std = mean_std(np.concatenate([train.state, train.next_state], axis=0))
    action_mean, action_std = mean_std(train.action)
    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
