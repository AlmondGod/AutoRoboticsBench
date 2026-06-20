from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from tasks.robotwin_bc3.common import DEFAULT_DATASET_REPO, DEFAULT_SPLIT, load_split


STATE_KEY = "observation.state"
ACTION_KEY = "action"


def load_robotwin_arrays(
    *,
    split_path: str | Path = DEFAULT_SPLIT,
    repo_id: str | None = None,
    train_demos_per_task: int | None = None,
    val_demos_per_task: int = 50,
    chunk_horizon: int = 16,
    frame_stride: int = 2,
) -> dict[str, Any]:
    split = load_split(split_path)
    repo = str(repo_id or split.get("repo_id") or DEFAULT_DATASET_REPO)
    episode_meta = _load_episode_meta(repo)
    train_ids: list[int] = []
    val_ids: list[int] = []
    task_id_by_episode: dict[int, int] = {}
    for task_idx, task in enumerate(split["tasks"]):
        train = [int(item) for item in task["train_episode_ids"]]
        if train_demos_per_task is not None:
            train = train[: int(train_demos_per_task)]
        heldout = [int(item) for item in task["heldout_demo_episode_ids"][: int(val_demos_per_task)]]
        train_ids.extend(train)
        val_ids.extend(heldout)
        for episode_id in train + heldout:
            task_id_by_episode[int(episode_id)] = int(task_idx)

    selected_ids = sorted(set(train_ids + val_ids))
    frames = _load_episode_frames(repo, episode_meta, selected_ids)
    train = _build_chunk_dataset(frames, train_ids, task_id_by_episode, chunk_horizon, frame_stride)
    val = _build_chunk_dataset(frames, val_ids, task_id_by_episode, chunk_horizon, frame_stride)
    stats = _fit_stats(train)
    return {
        "repo_id": repo,
        "split": split,
        "train": train,
        "val": val,
        "stats": stats,
        "task_names": [task["robotwin_task"] for task in split["tasks"]],
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "chunk_horizon": int(chunk_horizon),
        "frame_stride": int(frame_stride),
    }


def _load_episode_meta(repo_id: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="meta/episodes/chunk-000/file-000.parquet",
    )
    return pd.read_parquet(path)


def _load_episode_frames(repo_id: str, episode_meta: pd.DataFrame, episode_ids: list[int]) -> dict[int, pd.DataFrame]:
    rows = episode_meta.iloc[episode_ids]
    by_file: dict[int, list[tuple[int, int, int]]] = {}
    for _, row in rows.iterrows():
        by_file.setdefault(int(row["data/file_index"]), []).append(
            (int(row["episode_index"]), int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        )

    out: dict[int, pd.DataFrame] = {}
    for file_index, ranges in sorted(by_file.items()):
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"data/chunk-000/file-{file_index:03d}.parquet",
        )
        shard = pd.read_parquet(path, columns=[STATE_KEY, ACTION_KEY, "episode_index", "index"])
        for episode_id, start, end in ranges:
            mask = (shard["index"] >= start) & (shard["index"] < end)
            episode = shard.loc[mask].sort_values("index").reset_index(drop=True)
            if len(episode) == 0:
                raise ValueError(f"episode {episode_id} loaded zero rows from file {file_index}")
            out[episode_id] = episode
    return out


def _build_chunk_dataset(
    frames: dict[int, pd.DataFrame],
    episode_ids: list[int],
    task_id_by_episode: dict[int, int],
    chunk_horizon: int,
    frame_stride: int,
) -> dict[str, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    task_ids: list[int] = []
    progress: list[float] = []
    episode_out: list[int] = []
    horizon = int(chunk_horizon)
    stride = max(1, int(frame_stride))
    for episode_id in episode_ids:
        df = frames[int(episode_id)]
        ep_states = np.stack(df[STATE_KEY].to_numpy()).astype(np.float32)
        ep_actions = np.stack(df[ACTION_KEY].to_numpy()).astype(np.float32)
        limit = len(ep_actions) - horizon
        if limit <= 0:
            continue
        for t in range(0, limit, stride):
            states.append(ep_states[t])
            actions.append(ep_actions[t : t + horizon])
            task_ids.append(int(task_id_by_episode[int(episode_id)]))
            progress.append(float(t) / float(max(1, len(ep_actions) - 1)))
            episode_out.append(int(episode_id))
    return {
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "task_id": np.asarray(task_ids, dtype=np.int64),
        "progress": np.asarray(progress, dtype=np.float32)[:, None],
        "episode_id": np.asarray(episode_out, dtype=np.int64),
    }


def _fit_stats(train: dict[str, np.ndarray]) -> dict[str, list[float]]:
    eps = 1e-6
    return {
        "state_mean": train["state"].mean(axis=0).astype(float).tolist(),
        "state_std": (train["state"].std(axis=0) + eps).astype(float).tolist(),
        "action_mean": train["action"].reshape(-1, train["action"].shape[-1]).mean(axis=0).astype(float).tolist(),
        "action_std": (
            train["action"].reshape(-1, train["action"].shape[-1]).std(axis=0) + eps
        ).astype(float).tolist(),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
