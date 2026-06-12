from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image

import robocasa.utils.lerobot_utils as LU
from models.robocasa_tiny_evaluator import RoboCasaTinyEvaluator, tiny_evaluator_loss
from train.common import device_from_arg


@dataclass
class TransitionData:
    agent: np.ndarray
    wrist: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    next_agent: np.ndarray
    next_wrist: np.ndarray
    next_proprio: np.ndarray
    task_id: np.ndarray
    progress: np.ndarray
    next_progress: np.ndarray
    success: np.ndarray
    next_success: np.ndarray
    episode_idx: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.action.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa/world_evaluator/tiny")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--condition-on-robocasa-task-index", action="store_true")
    parser.add_argument("--train-demos-per-task", type=int, default=80)
    parser.add_argument("--val-episode-id", action="append", type=int, default=[])
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--success-window", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.5)
    parser.add_argument("--success-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    train, val = _load_data(
        manifest,
        train_demos_per_task=int(args.train_demos_per_task),
        val_episode_ids=set(args.val_episode_id),
        robocasa_task_indices=set(args.robocasa_task_index),
        condition_on_robocasa_task_index=bool(args.condition_on_robocasa_task_index),
        frame_stride=int(args.frame_stride),
        success_window=float(args.success_window),
    )
    if len(train) == 0 or len(val) == 0:
        raise ValueError("need non-empty train and val transition data")

    proprio_mean, proprio_std = _mean_std(np.concatenate([train.proprio, train.next_proprio], axis=0))
    train.proprio = ((train.proprio - proprio_mean) / proprio_std).astype(np.float32)
    train.next_proprio = ((train.next_proprio - proprio_mean) / proprio_std).astype(np.float32)
    val.proprio = ((val.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val.next_proprio = ((val.next_proprio - proprio_mean) / proprio_std).astype(np.float32)
    action_mean, action_std = _mean_std(train.action)
    train.action = ((train.action - action_mean) / action_std).astype(np.float32)
    val.action = ((val.action - action_mean) / action_std).astype(np.float32)

    device = device_from_arg(args.device)
    task_count = int(max(train.task_id.max(initial=0), val.task_id.max(initial=0)) + 1)
    model = RoboCasaTinyEvaluator(
        proprio_dim=int(train.proprio.shape[-1]),
        action_dim=int(train.action.shape[-1]),
        task_count=task_count,
        latent_dim=int(args.latent_dim),
        width=int(args.width),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val = math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        out = model(batch)
        loss, parts = tiny_evaluator_loss(
            out,
            batch,
            latent_weight=float(args.latent_weight),
            proprio_weight=float(args.proprio_weight),
            progress_weight=float(args.progress_weight),
            success_weight=float(args.success_weight),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        record = {"step": step, **parts}
        history.append(record)
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_metrics = _eval(model, val, device, int(args.batch_size), args)
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["loss"] < best_val:
                best_val = float(val_metrics["loss"])
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(
                f"step={step} loss={parts['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_progress_mae={val_metrics['progress_mae']:.4f}",
                flush=True,
            )

    val_metrics = _eval(model, val, device, int(args.batch_size), args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_type": "robocasa_tiny_evaluator",
        "proprio_dim": int(train.proprio.shape[-1]),
        "action_dim": int(train.action.shape[-1]),
        "task_count": task_count,
        "latent_dim": int(args.latent_dim),
        "width": int(args.width),
        "dropout": float(args.dropout),
        "manifest": str(Path(args.manifest)),
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "condition_on_robocasa_task_index": bool(args.condition_on_robocasa_task_index),
    }
    torch.save(checkpoint, out_dir / "tiny_evaluator.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_loss"] = float(best_val)
    torch.save(best_checkpoint, out_dir / "tiny_evaluator_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "tiny_evaluator.pt"),
        "best_checkpoint": str(out_dir / "tiny_evaluator_best.pt"),
        "best_step": int(best_step),
        "best_val_loss": float(best_val),
        "val": val_metrics,
        "train_samples": len(train),
        "val_samples": len(val),
        "train_demos_per_task": int(args.train_demos_per_task),
        "val_episode_ids": [int(ep) for ep in args.val_episode_id],
        "robocasa_task_indices": [int(idx) for idx in args.robocasa_task_index],
        "frame_stride": int(args.frame_stride),
        "success_window": float(args.success_window),
        "latent_dim": int(args.latent_dim),
        "width": int(args.width),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_seconds": time.time() - started,
        "tasks": [task["alias"] for task in manifest["tasks"]],
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _filtered_manifest(path: Path, task_aliases: list[str]) -> dict:
    manifest = json.loads(path.read_text())
    if task_aliases:
        keep = set(task_aliases)
        manifest["tasks"] = [task for task in manifest["tasks"] if task["alias"] in keep]
        if not manifest["tasks"]:
            raise ValueError(f"no tasks left after filtering for aliases={sorted(keep)}")
    for task_id, task in enumerate(manifest["tasks"]):
        task["task_id"] = task_id
    manifest["task_count"] = len(manifest["tasks"])
    return manifest


def _load_data(
    manifest: dict,
    *,
    train_demos_per_task: int,
    val_episode_ids: set[int],
    robocasa_task_indices: set[int],
    condition_on_robocasa_task_index: bool,
    frame_stride: int,
    success_window: float,
) -> tuple[TransitionData, TransitionData]:
    train_parts: list[dict[str, np.ndarray]] = []
    val_parts: list[dict[str, np.ndarray]] = []
    for task in manifest["tasks"]:
        dataset_root = Path(task["dataset_path"])
        episode_paths = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
        train_count = min(train_demos_per_task, max(1, len(episode_paths) - 1))
        for ordinal, episode_path in enumerate(episode_paths):
            episode_idx = int(episode_path.stem.split("_")[-1])
            robocasa_idx = _episode_task_index(episode_path)
            if robocasa_task_indices and robocasa_idx not in robocasa_task_indices:
                continue
            part = _episode_transitions(
                dataset_root=dataset_root,
                episode_path=episode_path,
                episode_idx=episode_idx,
                task_id=robocasa_idx if condition_on_robocasa_task_index else int(task["task_id"]),
                frame_stride=frame_stride,
                success_window=success_window,
            )
            is_val = episode_idx in val_episode_ids if val_episode_ids else ordinal >= train_count
            (val_parts if is_val else train_parts).append(part)
            print(
                f"loaded {task['alias']} episode={episode_idx} split={'val' if is_val else 'train'} transitions={len(part['action'])}",
                flush=True,
            )
    return _concat(train_parts), _concat(val_parts)


def _episode_task_index(episode_path: Path) -> int:
    frame = pd.read_parquet(episode_path, columns=["task_index"])
    return int(frame["task_index"].iloc[0])


def _episode_transitions(
    *,
    dataset_root: Path,
    episode_path: Path,
    episode_idx: int,
    task_id: int,
    frame_stride: int,
    success_window: float,
) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(episode_path)
    agent = _read_video64(dataset_root, episode_idx, "robot0_agentview_left")
    wrist = _read_video64(dataset_root, episode_idx, "robot0_agentview_right")
    proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    actions = LU.get_episode_actions(dataset_root, episode_idx).astype(np.float32)
    n = min(len(agent), len(wrist), len(proprio), len(actions))
    starts = np.arange(0, max(0, n - 1), max(1, frame_stride), dtype=np.int32)
    next_idx = np.minimum(starts + 1, n - 1)
    progress = starts.astype(np.float32) / max(1, n - 1)
    next_progress = next_idx.astype(np.float32) / max(1, n - 1)
    success = (progress >= float(success_window)).astype(np.float32)
    next_success = (next_progress >= float(success_window)).astype(np.float32)
    return {
        "agent": agent[starts],
        "wrist": wrist[starts],
        "proprio": proprio[starts],
        "action": actions[starts],
        "next_agent": agent[next_idx],
        "next_wrist": wrist[next_idx],
        "next_proprio": proprio[next_idx],
        "task_id": np.full((len(starts),), task_id, dtype=np.int64),
        "progress": progress,
        "next_progress": next_progress,
        "success": success,
        "next_success": next_success,
        "episode_idx": np.full((len(starts),), episode_idx, dtype=np.int32),
        "frame_idx": starts.astype(np.int32),
    }


def _read_video64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frames = [_resize64(np.asarray(frame, dtype=np.uint8)) for frame in iio.imiter(path)]
    return np.stack(frames).astype(np.uint8)


def _resize64(image: np.ndarray) -> np.ndarray:
    if image.shape[0] == 64 and image.shape[1] == 64:
        return image[..., :3]
    return np.asarray(Image.fromarray(image[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def _concat(parts: list[dict[str, np.ndarray]]) -> TransitionData:
    if not parts:
        return TransitionData(*(np.zeros((0,), dtype=np.float32) for _ in range(14)))  # type: ignore[arg-type]
    return TransitionData(
        agent=np.concatenate([part["agent"] for part in parts], axis=0),
        wrist=np.concatenate([part["wrist"] for part in parts], axis=0),
        proprio=np.concatenate([part["proprio"] for part in parts], axis=0),
        action=np.concatenate([part["action"] for part in parts], axis=0),
        next_agent=np.concatenate([part["next_agent"] for part in parts], axis=0),
        next_wrist=np.concatenate([part["next_wrist"] for part in parts], axis=0),
        next_proprio=np.concatenate([part["next_proprio"] for part in parts], axis=0),
        task_id=np.concatenate([part["task_id"] for part in parts], axis=0),
        progress=np.concatenate([part["progress"] for part in parts], axis=0),
        next_progress=np.concatenate([part["next_progress"] for part in parts], axis=0),
        success=np.concatenate([part["success"] for part in parts], axis=0),
        next_success=np.concatenate([part["next_success"] for part in parts], axis=0),
        episode_idx=np.concatenate([part["episode_idx"] for part in parts], axis=0),
        frame_idx=np.concatenate([part["frame_idx"] for part in parts], axis=0),
    )


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def _batch(data: TransitionData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent": torch.as_tensor(data.agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(data.wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "action": torch.as_tensor(data.action[idx], dtype=torch.float32, device=device),
        "next_agent": torch.as_tensor(data.next_agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "next_wrist": torch.as_tensor(data.next_wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "next_proprio": torch.as_tensor(data.next_proprio[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
        "progress": torch.as_tensor(data.progress[idx], dtype=torch.float32, device=device),
        "next_progress": torch.as_tensor(data.next_progress[idx], dtype=torch.float32, device=device),
        "success": torch.as_tensor(data.success[idx], dtype=torch.float32, device=device),
        "next_success": torch.as_tensor(data.next_success[idx], dtype=torch.float32, device=device),
    }


def _eval(model: RoboCasaTinyEvaluator, data: TransitionData, device: torch.device, batch_size: int, args) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "latent_loss": 0.0, "proprio_loss": 0.0, "progress_loss": 0.0, "success_loss": 0.0}
    progress_abs = 0.0
    success_correct = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            out = model(batch)
            loss, parts = tiny_evaluator_loss(
                out,
                batch,
                latent_weight=float(args.latent_weight),
                proprio_weight=float(args.proprio_weight),
                progress_weight=float(args.progress_weight),
                success_weight=float(args.success_weight),
            )
            n = len(idx)
            for key in totals:
                totals[key] += float((parts[key] if key != "loss" else loss.detach().cpu().item()) * n)
            progress_abs += float((torch.sigmoid(out["progress"]) - batch["progress"]).abs().sum().detach().cpu())
            success_pred = torch.sigmoid(out["success_logit"]) >= 0.5
            success_correct += float((success_pred == (batch["success"] >= 0.5)).sum().detach().cpu())
            count += n
    model.train()
    return {
        **{key: value / max(1, count) for key, value in totals.items()},
        "progress_mae": progress_abs / max(1, count),
        "success_acc": success_correct / max(1, count),
    }


if __name__ == "__main__":
    main()
