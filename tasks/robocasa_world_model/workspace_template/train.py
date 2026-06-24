from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

from dataclasses import dataclass
from typing import Any

import pandas as pd

# Inlined dataset helpers from the reward-model task; keep train.py self-contained.
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
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
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
    return _concat(train_parts), _concat(val_parts), summary


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


# Inlined inverse-dynamics loader; keep train.py self-contained.
class VideoInverseDynamics(nn.Module):
    def __init__(self, *, action_dim: int, task_count: int, task_dim: int = 32, width: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task = nn.Embedding(int(task_count), int(task_dim))
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(192 + int(task_dim) + 1, int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(action_dim)),
        )

    def encode_pair(self, image_pair: torch.Tensor) -> torch.Tensor:
        return self.encoder(image_pair)

    def forward(self, image_pair: torch.Tensor, task_id: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        if progress.ndim == 1:
            progress = progress[:, None]
        h = self.encode_pair(image_pair)
        h = torch.cat([h, self.task(task_id.long()), progress.float()], dim=-1)
        return self.head(h)


def load_inverse_dynamics(checkpoint: str | Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = VideoInverseDynamics(
        action_dim=int(cfg["action_dim"]),
        task_count=int(cfg["task_count"]),
        task_dim=int(cfg["task_dim"]),
        width=int(cfg["width"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return {
        "model": model,
        "config": cfg,
        "action_mean": torch.as_tensor(payload["action_mean"], dtype=torch.float32, device=device),
        "action_std": torch.as_tensor(payload["action_std"], dtype=torch.float32, device=device),
        "device": device,
    }
# Inlined from tasks/robocasa_world_model/model.py; keep this file self-contained.
class RoboCasaWorldModel(nn.Module):
    """State/action-conditioned dynamics model with optional latent VAE state."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        task_count: int,
        width: int = 512,
        depth: int = 4,
        task_dim: int = 64,
        latent_dim: int = 0,
        condition_on_task: bool = False,
        condition_on_progress: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task_dim = int(task_dim) if bool(condition_on_task) else 0
        self.latent_dim = int(latent_dim)
        self.condition_on_progress = bool(condition_on_progress)
        state_width = self.latent_dim if self.latent_dim > 0 else self.state_dim

        if self.latent_dim > 0:
            self.encoder = _mlp(self.state_dim, 2 * self.latent_dim, width, max(1, depth // 2), dropout)
            self.decoder = _mlp(self.latent_dim, self.state_dim, width, max(1, depth // 2), dropout)
        else:
            self.encoder = None
            self.decoder = None

        if self.task_dim > 0:
            self.task = nn.Embedding(max(1, self.task_count), self.task_dim)
        else:
            self.task = None

        inp = state_width + self.action_dim + self.task_dim + (1 if self.condition_on_progress else 0)
        self.trunk = _mlp(inp, width, width, depth, dropout, final_norm=True)
        self.delta = nn.Linear(width, state_width)
        self.progress = nn.Linear(width, 1)
        self.success = nn.Linear(width, 1)

    def encode_state(self, state: torch.Tensor, *, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            zero = torch.zeros((), dtype=state.dtype, device=state.device)
            return state, zero, zero
        stats = self.encoder(state)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        if sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            return latent
        return self.decoder(latent)

    def conditioned_input(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        *,
        task_id: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [latent, action]
        if self.task is not None:
            if task_id is None:
                task_id = torch.zeros((latent.shape[0],), dtype=torch.long, device=latent.device)
            task_id = task_id.reshape(-1).long().clamp(0, max(0, self.task_count - 1))
            parts.append(self.task(task_id))
        if self.condition_on_progress:
            if progress is None:
                progress = torch.zeros((latent.shape[0], 1), dtype=latent.dtype, device=latent.device)
            else:
                progress = progress.reshape(latent.shape[0], -1)[:, :1].to(dtype=latent.dtype, device=latent.device)
            parts.append(progress)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        task_id: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        z, mu, logvar = self.encode_state(state, sample=sample_latent)
        h = self.trunk(self.conditioned_input(z, action, task_id=task_id, progress=progress))
        next_z = z + self.delta(h)
        next_state = self.decode_state(next_z)
        return {
            "next_state": next_state,
            "next_latent": next_z,
            "next_progress": torch.sigmoid(self.progress(h)),
            "success_logit": self.success(h),
            "latent_mu": mu,
            "latent_logvar": logvar,
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        state_weight: float = 1.0,
        progress_weight: float = 0.25,
        success_weight: float = 0.25,
        kl_weight: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(
            batch["state"],
            batch["action"],
            task_id=batch.get("task_id"),
            progress=batch.get("progress"),
            sample_latent=True,
        )
        state_loss = F.mse_loss(out["next_state"], batch["next_state"])
        progress_loss = F.mse_loss(out["next_progress"], batch["next_progress"])
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        if self.latent_dim > 0:
            mu = out["latent_mu"]
            logvar = out["latent_logvar"]
            kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
        else:
            kl = torch.zeros((), dtype=state_loss.dtype, device=state_loss.device)
        total = (
            float(state_weight) * state_loss
            + float(progress_weight) * progress_loss
            + float(success_weight) * success_loss
            + float(kl_weight) * kl
        )
        metrics = {
            "loss": total.detach(),
            "state_mse": state_loss.detach(),
            "progress_mse": progress_loss.detach(),
            "success_bce": success_loss.detach(),
            "kl": kl.detach(),
        }
        return total, metrics


def _mlp(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    dropout: float,
    *,
    final_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(depth)):
        layers.extend(
            [
                nn.Linear(dim, int(width)),
                nn.LayerNorm(int(width)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
        )
        dim = int(width)
    if final_norm:
        layers.append(nn.LayerNorm(dim))
    layers.append(nn.Linear(dim, int(out_dim)))
    return nn.Sequential(*layers)




# Inlined video representation loader; keep train.py self-contained.
class VideoProgressEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, int(embed_dim)),
            nn.LayerNorm(int(embed_dim)),
        )
        self.progress = nn.Linear(int(embed_dim), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(self.encoder(image), dim=-1)
        return {"embedding": z, "progress": torch.sigmoid(self.progress(z))}


def load_video_encoder(checkpoint: str | Path, device: torch.device) -> VideoProgressEncoder:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    model = VideoProgressEncoder(embed_dim=int(cfg.get("embed_dim", 64))).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
def device_from_arg(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train RoboCasa learned reward/progress model.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--video-pool", default=str(DEFAULT_VIDEO_POOL))
    parser.add_argument("--video-episodes-per-task", type=int, default=0)
    parser.add_argument("--video-pool-split", action="append", default=[])
    parser.add_argument("--video-repr-checkpoint", default="")
    parser.add_argument("--video-align-weight", type=float, default=0.0)
    parser.add_argument("--video-align-view", default="robot0_agentview_right")
    parser.add_argument("--video-align-image-size", type=int, default=96)
    parser.add_argument("--inverse-dynamics-checkpoint", default="")
    parser.add_argument("--inverse-align-weight", type=float, default=0.0)
    parser.add_argument("--inverse-align-view", default="robot0_agentview_right")
    parser.add_argument("--inverse-align-image-size", type=int, default=64)
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_reward_model/base")
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-train-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=0, help="Set >0 to train a VAE latent dynamics model.")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=0.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")
    if float(args.max_train_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = device_from_arg(str(args.device))
    train_raw, val_raw, summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        task_aliases=set(args.task_alias),
        frame_stride=int(args.frame_stride),
    )
    if len(train_raw) == 0 or len(val_raw) == 0:
        raise ValueError("need both train and val transitions for world-model training")
    video_summary = {"enabled": False, "reason": "video_episodes_per_task is 0"}
    if int(args.video_episodes_per_task) > 0:
        video_records = load_video_only_pool(
            args.video_pool,
            max_episodes_per_task=int(args.video_episodes_per_task),
            task_aliases=set(args.task_alias),
            splits=set(args.video_pool_split),
        )
        video_summary = {
            "enabled": True,
            "video_pool": str(args.video_pool),
            "max_episodes_per_task": int(args.video_episodes_per_task),
            "splits": list(args.video_pool_split),
            **summarize_video_only_pool(video_records),
            "notes": [
                "Default baseline records availability only and does not train on video-only data.",
                "Mutable training methods can use load_video_only_pool/load_video_frames for inverse dynamics or self-supervised video losses.",
            ],
        }
    stats = make_stats(train_raw)
    train = normalize_data(train_raw, stats)
    val = normalize_data(val_raw, stats)
    task_count = int(max(train.task_id.max(initial=0), val.task_id.max(initial=0)) + 1)
    model = RoboCasaWorldModel(
        state_dim=int(train.state.shape[-1]),
        action_dim=int(train.action.shape[-1]),
        task_count=task_count,
        width=int(args.width),
        depth=int(args.depth),
        task_dim=int(args.task_dim),
        latent_dim=int(args.latent_dim),
        dropout=float(args.dropout),
    ).to(device)
    params = list(model.parameters())
    video_align, video_align_head = _build_video_alignment(args, device)
    inverse_align, inverse_align_head = _build_inverse_alignment(args, device, width=int(args.width))
    if video_align_head is not None:
        params.extend(video_align_head.parameters())
    if inverse_align_head is not None:
        params.extend(inverse_align_head.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    if video_align is not None and float(video_align["weight"]) > 0:
        video_align["train_targets"] = _precompute_video_targets(train, summary, video_align, device)
    if inverse_align is not None and float(inverse_align["weight"]) > 0:
        inverse_align["train_targets"] = _precompute_inverse_targets(train, summary, inverse_align, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    start_time = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        step += 1
        model.train()
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        loss, metrics = model.loss(
            batch,
            state_weight=float(args.state_weight),
            progress_weight=float(args.progress_weight),
            success_weight=float(args.success_weight),
            kl_weight=float(args.kl_weight),
        )
        if video_align is not None and float(video_align["weight"]) > 0:
            align_loss = _video_alignment_loss(model, batch, train, idx, summary, video_align, device)
            loss = loss + float(video_align["weight"]) * align_loss
            metrics["video_align_loss"] = align_loss.detach()
        if inverse_align is not None and float(inverse_align["weight"]) > 0:
            inverse_loss = _inverse_alignment_loss(model, batch, idx, inverse_align, device)
            loss = loss + float(inverse_align["weight"]) * inverse_loss
            metrics["inverse_align_loss"] = inverse_loss.detach()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            val_metrics = _eval(model, val, int(args.batch_size), device)
            row = {
                "step": int(step),
                "elapsed_seconds": time.monotonic() - start_time,
                **{key: float(value.detach().cpu()) for key, value in metrics.items()},
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["val_reward_score_loss"] < best_val:
                best_val = row["val_reward_score_loss"]
                _save_checkpoint(
                    out_dir / "policy_best.pt",
                    model,
                    stats,
                    args,
                    summary,
                    video_summary,
                    video_align,
                    inverse_align,
                    history,
                    step,
                )

    final_metrics = _eval(model, val, int(args.batch_size), device)
    _save_checkpoint(
        out_dir / "policy_last.pt",
        model,
        stats,
        args,
        summary,
        video_summary,
        video_align,
        inverse_align,
        history,
        len(history),
    )
    payload = {
        "task": "robocasa_reward_model",
        "checkpoint": str(out_dir / "policy_best.pt"),
        "last_checkpoint": str(out_dir / "policy_last.pt"),
        "train_transitions": len(train),
        "val_transitions": len(val),
        "video_only_pool": video_summary,
        "video_alignment": _video_alignment_summary(args, video_align),
        "inverse_alignment": _inverse_alignment_summary(args, inverse_align),
        "summary": summary,
        "final_val": final_metrics,
        "best_val_reward_score_loss": best_val,
        "history": history,
        "seconds": time.monotonic() - start_time,
    }
    save_json(out_dir / "train_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _batch(data: TransitionData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state": torch.as_tensor(data.state[idx], dtype=torch.float32, device=device),
        "action": torch.as_tensor(data.action[idx], dtype=torch.float32, device=device),
        "next_state": torch.as_tensor(data.next_state[idx], dtype=torch.float32, device=device),
        "progress": torch.as_tensor(data.progress[idx], dtype=torch.float32, device=device),
        "next_progress": torch.as_tensor(data.next_progress[idx], dtype=torch.float32, device=device),
        "success": torch.as_tensor(data.success[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


def _build_video_alignment(args: argparse.Namespace, device: torch.device) -> tuple[dict | None, nn.Module | None]:
    if not args.video_repr_checkpoint and float(args.video_align_weight) <= 0:
        return None, None
    if not args.video_repr_checkpoint:
        raise ValueError("--video-align-weight requires --video-repr-checkpoint")
    if int(args.latent_dim) <= 0:
        raise ValueError("video representation alignment requires --latent-dim > 0")
    encoder = load_video_encoder(args.video_repr_checkpoint, device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    head = nn.Linear(int(args.latent_dim), int(encoder.embed_dim)).to(device)
    return (
        {
            "encoder": encoder,
            "head": head,
            "weight": float(args.video_align_weight),
            "view": str(args.video_align_view),
            "image_size": int(args.video_align_image_size),
        },
        head,
    )


def _build_inverse_alignment(
    args: argparse.Namespace,
    device: torch.device,
    *,
    width: int,
) -> tuple[dict | None, nn.Module | None]:
    if not args.inverse_dynamics_checkpoint and float(args.inverse_align_weight) <= 0:
        return None, None
    if not args.inverse_dynamics_checkpoint:
        raise ValueError("--inverse-align-weight requires --inverse-dynamics-checkpoint")
    inverse = load_inverse_dynamics(args.inverse_dynamics_checkpoint, device)
    inverse_model = inverse["model"]
    for param in inverse_model.parameters():
        param.requires_grad_(False)
    feature_dim = 192
    head = nn.Linear(int(width), feature_dim).to(device)
    return (
        {
            "model": inverse_model,
            "head": head,
            "feature_dim": feature_dim,
            "weight": float(args.inverse_align_weight),
            "view": str(args.inverse_align_view),
            "image_size": int(args.inverse_align_image_size),
            "checkpoint": str(args.inverse_dynamics_checkpoint),
        },
        head,
    )


def _video_alignment_loss(
    model: RoboCasaWorldModel,
    batch: dict[str, torch.Tensor],
    data: TransitionData,
    idx: np.ndarray,
    summary: list[dict],
    video_align: dict,
    device: torch.device,
) -> torch.Tensor:
    if "train_targets" in video_align:
        target = video_align["train_targets"][torch.as_tensor(idx, dtype=torch.long, device=device)]
    else:
        frames = []
        dataset_by_task = {int(row["task_id"]): Path(row["dataset_path"]) for row in summary}
        view = str(video_align["view"])
        image_size = int(video_align["image_size"])
        for task_id, episode_id, frame_idx in zip(data.task_id[idx], data.episode_id[idx], data.frame_idx[idx]):
            root = dataset_by_task[int(task_id)]
            video = root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{int(episode_id):06d}.mp4"
            frames.append(_preprocess_frame(load_video_frame(video, int(frame_idx)), image_size))
        image = torch.as_tensor(np.stack(frames), dtype=torch.float32, device=device)
        with torch.no_grad():
            target = video_align["encoder"](image)["embedding"]
    z, _, _ = model.encode_state(batch["state"], sample=False)
    pred = F.normalize(video_align["head"](z), dim=-1)
    return F.mse_loss(pred, target)


def _inverse_alignment_loss(
    model: RoboCasaWorldModel,
    batch: dict[str, torch.Tensor],
    idx: np.ndarray,
    inverse_align: dict,
    device: torch.device,
) -> torch.Tensor:
    target = inverse_align["train_targets"][torch.as_tensor(idx, dtype=torch.long, device=device)]
    hidden = _transition_hidden(model, batch)
    pred = F.normalize(inverse_align["head"](hidden), dim=-1)
    return F.mse_loss(pred, target)


def _transition_hidden(model: RoboCasaWorldModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    z, _, _ = model.encode_state(batch["state"], sample=False)
    h = torch.cat([z, batch["action"]], dim=-1)
    return model.trunk(h)


@torch.no_grad()
def _precompute_video_targets(
    data: TransitionData,
    summary: list[dict],
    video_align: dict,
    device: torch.device,
) -> torch.Tensor:
    encoder = video_align["encoder"]
    encoder.eval()
    targets = torch.empty((len(data), int(encoder.embed_dim)), dtype=torch.float32, device=device)
    dataset_by_task = {int(row["task_id"]): Path(row["dataset_path"]) for row in summary}
    view = str(video_align["view"])
    image_size = int(video_align["image_size"])
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        groups.setdefault((int(task_id), int(episode_id)), []).append(index)
    for (task_id, episode_id), indices in sorted(groups.items()):
        root = dataset_by_task[int(task_id)]
        video = root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{int(episode_id):06d}.mp4"
        frames = load_video_frames(video)
        frame_indices = np.clip(data.frame_idx[np.asarray(indices, dtype=np.int64)], 0, max(0, len(frames) - 1))
        for start in range(0, len(indices), 256):
            batch_indices = indices[start : start + 256]
            batch_frames = [_preprocess_frame(frames[int(frame_idx)], image_size) for frame_idx in frame_indices[start : start + 256]]
            image = torch.as_tensor(np.stack(batch_frames), dtype=torch.float32, device=device)
            targets[torch.as_tensor(batch_indices, dtype=torch.long, device=device)] = encoder(image)["embedding"]
    return targets


@torch.no_grad()
def _precompute_inverse_targets(
    data: TransitionData,
    summary: list[dict],
    inverse_align: dict,
    device: torch.device,
) -> torch.Tensor:
    inverse_model = inverse_align["model"]
    inverse_model.eval()
    feature_dim = int(inverse_align["feature_dim"])
    targets = torch.empty((len(data), feature_dim), dtype=torch.float32, device=device)
    dataset_by_task = {int(row["task_id"]): Path(row["dataset_path"]) for row in summary}
    view = str(inverse_align["view"])
    image_size = int(inverse_align["image_size"])
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        groups.setdefault((int(task_id), int(episode_id)), []).append(index)
    for (task_id, episode_id), indices in sorted(groups.items()):
        root = dataset_by_task[int(task_id)]
        video = root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{int(episode_id):06d}.mp4"
        frames = load_video_frames(video)
        frame_indices = np.clip(data.frame_idx[np.asarray(indices, dtype=np.int64)], 0, max(0, len(frames) - 2))
        for start in range(0, len(indices), 256):
            batch_indices = indices[start : start + 256]
            pairs = []
            for frame_idx in frame_indices[start : start + 256]:
                frame_i = int(frame_idx)
                pairs.append(
                    np.concatenate(
                        [
                            _preprocess_frame(frames[frame_i], image_size),
                            _preprocess_frame(frames[min(frame_i + 1, len(frames) - 1)], image_size),
                        ],
                        axis=0,
                    )
                )
            image_pair = torch.as_tensor(np.stack(pairs), dtype=torch.float32, device=device)
            encoded = F.normalize(inverse_model.encode_pair(image_pair), dim=-1)
            targets[torch.as_tensor(batch_indices, dtype=torch.long, device=device)] = encoded
    return targets


def _preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((int(image_size), int(image_size))))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


def _video_alignment_summary(args: argparse.Namespace, video_align: dict | None) -> dict:
    return {
        "enabled": video_align is not None and float(args.video_align_weight) > 0,
        "checkpoint": str(args.video_repr_checkpoint),
        "weight": float(args.video_align_weight),
        "view": str(args.video_align_view),
        "image_size": int(args.video_align_image_size),
        "requires_latent_dim": True,
    }


def _inverse_alignment_summary(args: argparse.Namespace, inverse_align: dict | None) -> dict:
    return {
        "enabled": inverse_align is not None and float(args.inverse_align_weight) > 0,
        "checkpoint": str(args.inverse_dynamics_checkpoint),
        "weight": float(args.inverse_align_weight),
        "view": str(args.inverse_align_view),
        "image_size": int(args.inverse_align_image_size),
        "target": "frozen_inverse_dynamics_pair_encoder",
    }


@torch.no_grad()
def _eval(model: RoboCasaWorldModel, data: TransitionData, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {
        "progress_mse": 0.0,
        "success_bce": 0.0,
        "reward_score_loss": 0.0,
    }
    count = 0
    for start in range(0, len(data), batch_size):
        idx = np.arange(start, min(len(data), start + batch_size))
        batch = _batch(data, idx, device)
        total, metrics = model.loss(batch)
        n = len(idx)
        for key in ("progress_mse", "success_bce"):
            sums[key] += float(metrics[key].detach().cpu()) * n
        reward_score_loss = metrics["progress_mse"] + metrics["success_bce"]
        sums["reward_score_loss"] += float(reward_score_loss.detach().cpu()) * n
        count += n
    return {key: value / max(1, count) for key, value in sums.items()}


def _save_checkpoint(
    path: Path,
    model: RoboCasaWorldModel,
    stats: dict[str, np.ndarray],
    args: argparse.Namespace,
    summary: list[dict],
    video_summary: dict,
    video_align: dict | None,
    inverse_align: dict | None,
    history: list[dict],
    step: int,
) -> None:
    cfg = {
        "state_dim": int(model.state_dim),
        "action_dim": int(model.action_dim),
        "task_count": int(model.task_count),
        "width": int(args.width),
        "depth": int(args.depth),
        "task_dim": int(args.task_dim),
        "latent_dim": int(args.latent_dim),
        "dropout": float(args.dropout),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "stats": stats,
            "summary": summary,
            "video_only_pool": video_summary,
            "video_alignment": _video_alignment_summary(args, video_align),
            "inverse_alignment": _inverse_alignment_summary(args, inverse_align),
            "history": history,
            "step": int(step),
            "task": "robocasa_reward_model",
        },
        path,
    )


if __name__ == "__main__":
    main()
