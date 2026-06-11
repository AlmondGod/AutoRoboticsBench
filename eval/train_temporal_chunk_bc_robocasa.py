from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

import robocasa.utils.lerobot_utils as LU
from train.common import device_from_arg


@dataclass
class TemporalChunkData:
    agent: np.ndarray
    wrist: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    task_id: np.ndarray
    episode_idx: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.agent.shape[0])


class RoboCasaTemporalChunkBC(nn.Module):
    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 512,
        dropout: float = 0.05,
        task_dim: int = 32,
    ) -> None:
        super().__init__()
        self.chunk_horizon = chunk_horizon
        self.action_dim = action_dim
        self.image = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, width),
            nn.SiLU(),
        )
        prop_width = max(128, width // 2)
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, prop_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(prop_width, prop_width),
            nn.SiLU(),
        )
        self.task = nn.Embedding(task_count, task_dim)
        self.head = nn.Sequential(
            nn.Linear(width + prop_width + task_dim, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, chunk_horizon * action_dim),
        )
        self.action_in = nn.Linear(chunk_horizon * action_dim, 2 * width)
        self.flow_time = nn.Sequential(
            nn.Linear(1, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, 2 * width),
        )
        self.flow_decoder = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, chunk_horizon * action_dim),
        )

    def forward(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        features = self.encode_obs(agent, wrist, proprio, task_id)
        out = self.head(features)
        return out.reshape(agent.shape[0], self.chunk_horizon, self.action_dim)

    def encode_obs(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        image_feat = self.image(torch.cat([agent, wrist], dim=1))
        proprio_feat = self.proprio(proprio)
        task_feat = self.task(task_id)
        return torch.cat([image_feat, proprio_feat, task_feat], dim=-1)

    def flow_velocity(self, obs_h: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = action_t.shape[0]
        action_flat = action_t.reshape(batch, self.chunk_horizon * self.action_dim)
        t = t.reshape(batch, 1).to(dtype=obs_h.dtype, device=obs_h.device)
        h = self.head[0](obs_h)
        velocity = self.flow_decoder(h + self.action_in(action_flat) + self.flow_time(t))
        return velocity.reshape(batch, self.chunk_horizon, self.action_dim)

    def sample_flow(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 8,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_h = self.encode_obs(agent, wrist, proprio, task_id)
        if initial_noise is None:
            action = torch.zeros((obs_h.shape[0], self.chunk_horizon, self.action_dim), dtype=obs_h.dtype, device=obs_h.device)
        else:
            action = initial_noise.to(dtype=obs_h.dtype, device=obs_h.device)
        steps = max(1, int(steps))
        dt = 1.0 / steps
        for idx in range(steps):
            t = torch.full((obs_h.shape[0],), (idx + 0.5) * dt, dtype=obs_h.dtype, device=obs_h.device)
            action = action + dt * self.flow_velocity(obs_h, action, t)
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa/opendrawer_temporal_chunk_bc")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--chunk-horizon", type=int, default=48)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--train-demos-per-task", type=int, default=80)
    parser.add_argument("--val-episode-id", action="append", type=int, default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--condition-on-robocasa-task-index", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-noise", type=float, default=0.01)
    parser.add_argument("--proprio-noise", type=float, default=0.01)
    parser.add_argument("--action-smooth", type=float, default=0.001)
    parser.add_argument("--policy-kind", choices=["bc", "flow"], default="bc")
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-sigma", type=float, default=1.0)
    parser.add_argument("--chunk-decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    train_data, val_data = _load_data(
        manifest,
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
        train_demos_per_task=int(args.train_demos_per_task),
        val_episode_ids=set(args.val_episode_id),
        robocasa_task_indices=set(args.robocasa_task_index),
        condition_on_robocasa_task_index=bool(args.condition_on_robocasa_task_index),
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("need both train and val temporal samples")

    proprio_mean, proprio_std = _mean_std(train_data.proprio)
    action_mean, action_std = _masked_mean_std(train_data.actions, train_data.mask)
    train_data.proprio = ((train_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val_data.proprio = ((val_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    train_data.actions = ((train_data.actions - action_mean) / action_std).astype(np.float32)
    val_data.actions = ((val_data.actions - action_mean) / action_std).astype(np.float32)

    device = device_from_arg(args.device)
    model = RoboCasaTemporalChunkBC(
        proprio_dim=int(train_data.proprio.shape[-1]),
        chunk_horizon=int(args.chunk_horizon),
        action_dim=int(train_data.actions.shape[-1]),
        task_count=int(max(train_data.task_id.max(initial=0), val_data.task_id.max(initial=0)) + 1),
        width=int(args.width),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val_loss = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(train_data), size=int(args.batch_size))
        batch = _batch(train_data, idx, device)
        batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
        if args.policy_kind == "flow":
            loss = _flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                chunk_decay=float(args.chunk_decay),
            )
        else:
            pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_value = float(loss.detach().cpu())
        history.append({"step": step, "train_loss": loss_value})
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_loss = _eval_loss(
                model,
                val_data,
                device,
                batch_size=max(128, int(args.batch_size)),
                policy_kind=str(args.policy_kind),
                flow_steps=int(args.flow_steps),
            )
            history[-1]["val_loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"step={step} temporal_chunk_loss={loss_value:.6f} val_loss={val_loss:.6f}", flush=True)

    val_loss = _eval_loss(
        model,
        val_data,
        device,
        batch_size=max(128, int(args.batch_size)),
        policy_kind=str(args.policy_kind),
        flow_steps=int(args.flow_steps),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
            "state_dict": model.state_dict(),
            "policy_type": "robocasa_temporal_chunk_bc",
            "chunk_horizon": int(args.chunk_horizon),
            "action_dim": int(train_data.actions.shape[-1]),
            "proprio_dim": int(train_data.proprio.shape[-1]),
            "task_count": int(max(train_data.task_id.max(initial=0), val_data.task_id.max(initial=0)) + 1),
            "width": int(args.width),
            "dropout": float(args.dropout),
            "policy_kind": str(args.policy_kind),
            "flow_steps": int(args.flow_steps),
            "flow_sigma": float(args.flow_sigma),
            "chunk_decay": float(args.chunk_decay),
            "condition_on_robocasa_task_index": bool(args.condition_on_robocasa_task_index),
            "views": ["robot0_agentview_left", "robot0_agentview_right"],
            "manifest": str(Path(args.manifest)),
            "proprio_mean": proprio_mean,
            "proprio_std": proprio_std,
            "action_mean": action_mean,
            "action_std": action_std,
        }
    torch.save(checkpoint, out_dir / "temporal_chunk.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_action_mse_normalized"] = float(best_val_loss)
    torch.save(best_checkpoint, out_dir / "temporal_chunk_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "temporal_chunk.pt"),
        "best_checkpoint": str(out_dir / "temporal_chunk_best.pt"),
        "best_step": int(best_step),
        "best_val_action_mse_normalized": float(best_val_loss),
        "chunk_horizon": int(args.chunk_horizon),
        "frame_stride": int(args.frame_stride),
        "train_demos": int(args.train_demos_per_task),
        "val_episode_ids": [int(ep) for ep in args.val_episode_id],
        "robocasa_task_indices": [int(idx) for idx in args.robocasa_task_index],
        "condition_on_robocasa_task_index": bool(args.condition_on_robocasa_task_index),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "val_action_mse_normalized": val_loss,
        "width": int(args.width),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "image_noise": float(args.image_noise),
        "proprio_noise": float(args.proprio_noise),
        "action_smooth": float(args.action_smooth),
        "policy_kind": str(args.policy_kind),
        "flow_steps": int(args.flow_steps),
        "flow_sigma": float(args.flow_sigma),
        "chunk_decay": float(args.chunk_decay),
        "seed": int(args.seed),
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
    chunk_horizon: int,
    frame_stride: int,
    train_demos_per_task: int,
    val_episode_ids: set[int],
    robocasa_task_indices: set[int],
    condition_on_robocasa_task_index: bool,
) -> tuple[TemporalChunkData, TemporalChunkData]:
    train_parts: list[dict[str, np.ndarray]] = []
    val_parts: list[dict[str, np.ndarray]] = []
    for task in manifest["tasks"]:
        dataset_root = Path(task["dataset_path"])
        episode_paths = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
        train_count = min(train_demos_per_task, max(1, len(episode_paths) - 1))
        for ordinal, episode_path in enumerate(episode_paths):
            episode_idx = int(episode_path.stem.split("_")[-1])
            if robocasa_task_indices and _episode_task_index(episode_path) not in robocasa_task_indices:
                continue
            part = _episode_samples(
                dataset_root,
                episode_path,
                episode_idx,
                int(task["task_id"]),
                chunk_horizon,
                frame_stride,
                condition_on_robocasa_task_index,
            )
            is_val = episode_idx in val_episode_ids if val_episode_ids else ordinal >= train_count
            if not is_val:
                train_parts.append(part)
            else:
                val_parts.append(part)
            print(
                f"loaded {task['alias']} episode={episode_idx} split={'val' if is_val else 'train'} samples={len(part['task_id'])}",
                flush=True,
            )
    return _concat_parts(train_parts), _concat_parts(val_parts)


def _episode_task_index(episode_path: Path) -> int:
    frame = pd.read_parquet(episode_path, columns=["task_index"])
    return int(frame["task_index"].iloc[0])


def _episode_samples(
    dataset_root: Path,
    episode_path: Path,
    episode_idx: int,
    task_id: int,
    chunk_horizon: int,
    frame_stride: int,
    condition_on_robocasa_task_index: bool,
) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(episode_path)
    robocasa_task_index = int(frame["task_index"].iloc[0])
    sample_task_id = robocasa_task_index if condition_on_robocasa_task_index else task_id
    agent = _read_video64(dataset_root, episode_idx, "robot0_agentview_left")
    wrist = _read_video64(dataset_root, episode_idx, "robot0_agentview_right")
    proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    actions = LU.get_episode_actions(dataset_root, episode_idx).astype(np.float32)
    n = min(len(agent), len(wrist), len(proprio), len(actions))
    starts = np.arange(0, n, max(1, frame_stride), dtype=np.int32)

    out_actions = np.zeros((len(starts), chunk_horizon, actions.shape[-1]), dtype=np.float32)
    mask = np.zeros((len(starts), chunk_horizon), dtype=np.float32)
    for row_idx, start in enumerate(starts):
        end = min(n, int(start) + chunk_horizon)
        length = end - int(start)
        out_actions[row_idx, :length] = actions[int(start) : end]
        mask[row_idx, :length] = 1.0

    return {
        "agent": agent[starts],
        "wrist": wrist[starts],
        "proprio": proprio[starts],
        "actions": out_actions,
        "mask": mask,
        "task_id": np.full((len(starts),), sample_task_id, dtype=np.int64),
        "episode_idx": np.full((len(starts),), episode_idx, dtype=np.int32),
        "frame_idx": starts.astype(np.int32),
    }


def _read_video64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frames = [_resize64(np.asarray(frame, dtype=np.uint8)) for frame in iio.imiter(video_path)]
    return np.stack(frames).astype(np.uint8)


def _resize64(image: np.ndarray) -> np.ndarray:
    if image.shape[0] == 64 and image.shape[1] == 64:
        return image[..., :3]
    return np.asarray(Image.fromarray(image[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def _concat_parts(parts: list[dict[str, np.ndarray]]) -> TemporalChunkData:
    if not parts:
        return TemporalChunkData(
            agent=np.zeros((0, 64, 64, 3), dtype=np.uint8),
            wrist=np.zeros((0, 64, 64, 3), dtype=np.uint8),
            proprio=np.zeros((0, 16), dtype=np.float32),
            actions=np.zeros((0, 1, 12), dtype=np.float32),
            mask=np.zeros((0, 1), dtype=np.float32),
            task_id=np.zeros((0,), dtype=np.int64),
            episode_idx=np.zeros((0,), dtype=np.int32),
            frame_idx=np.zeros((0,), dtype=np.int32),
        )
    return TemporalChunkData(
        agent=np.concatenate([part["agent"] for part in parts], axis=0),
        wrist=np.concatenate([part["wrist"] for part in parts], axis=0),
        proprio=np.concatenate([part["proprio"] for part in parts], axis=0),
        actions=np.concatenate([part["actions"] for part in parts], axis=0),
        mask=np.concatenate([part["mask"] for part in parts], axis=0),
        task_id=np.concatenate([part["task_id"] for part in parts], axis=0),
        episode_idx=np.concatenate([part["episode_idx"] for part in parts], axis=0),
        frame_idx=np.concatenate([part["frame_idx"] for part in parts], axis=0),
    )


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def _masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1])
    keep = mask.reshape(-1) > 0
    valid = flat[keep]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def _batch(data: TemporalChunkData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent": torch.as_tensor(data.agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(data.wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data.actions[idx], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(data.mask[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


def _augment(batch: dict[str, torch.Tensor], image_noise: float, proprio_noise: float) -> dict[str, torch.Tensor]:
    if image_noise > 0:
        scale = 255.0 * image_noise
        batch["agent"] = (batch["agent"] + torch.randn_like(batch["agent"]) * scale).clamp(0.0, 255.0)
        batch["wrist"] = (batch["wrist"] + torch.randn_like(batch["wrist"]) * scale).clamp(0.0, 255.0)
    if proprio_noise > 0:
        batch["proprio"] = batch["proprio"] + torch.randn_like(batch["proprio"]) * proprio_noise
    return batch


def _eval_loss(
    model: RoboCasaTemporalChunkBC,
    data: TemporalChunkData,
    device: torch.device,
    batch_size: int,
    *,
    policy_kind: str = "bc",
    flow_steps: int = 8,
) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            if policy_kind == "flow":
                pred = model.sample_flow(
                    batch["agent"],
                    batch["wrist"],
                    batch["proprio"],
                    batch["task_id"],
                    steps=flow_steps,
                )
            else:
                pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _masked_chunk_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, chunk_decay: float = 1.0) -> torch.Tensor:
    per_step = (pred - target).square().mean(dim=-1)
    weights = _chunk_weights(pred.shape[1], chunk_decay, pred.device, pred.dtype)
    return (per_step * mask * weights).sum() / (mask * weights).sum().clamp_min(1.0)


def _flow_matching_loss(
    model: RoboCasaTemporalChunkBC,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    chunk_decay: float,
) -> torch.Tensor:
    actions = batch["actions"]
    noise = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    view_t = t.reshape(-1, 1, 1)
    action_t = (1.0 - view_t) * noise + view_t * actions
    target_velocity = actions - noise
    obs_h = model.encode_obs(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
    pred_velocity = model.flow_velocity(obs_h, action_t, t)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    return (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)


def _chunk_weights(horizon: int, decay: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.ones((horizon,), dtype=dtype, device=device)
    if decay != 1.0:
        idx = torch.arange(horizon, dtype=dtype, device=device)
        weights = decay**idx
    return weights.reshape(1, horizon) / weights.mean().clamp_min(1e-6)


if __name__ == "__main__":
    main()
