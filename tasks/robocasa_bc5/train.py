from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

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


def device_from_arg(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


from tasks.robocasa_bc5.model import (
    RoboCasaFrozenCLIPFlowPolicy,
    RoboCasaFrozenR3MFlowPolicy,
    RoboCasaFrozenSmolVLMFlowPolicy,
    RoboCasaHistoryACTFlowPolicy,
    RoboCasaHistoryACTPolicy,
    RoboCasaHistoryFlowPolicy,
    RoboCasaMiniPi0ACTPolicy,
    RoboCasaMiniPi0ACTResNetPolicy,
    RoboCasaMiniPi0Policy,
    RoboCasaMiniPi0ResNetPolicy,
    RoboCasaPatchViTACTPolicy,
    RoboCasaSequenceFlowPolicy,
    RoboCasaTemporalChunkBC,
)


ensure_robocasa_runtime()

import robocasa.utils.lerobot_utils as LU  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AutoroboBench RoboCasa BC-5 baseline policy.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--video-pool", default="")
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_bc5/baseline")
    parser.add_argument("--train-episodes-per-task", type=int, default=4)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-noise", type=float, default=0.01)
    parser.add_argument("--proprio-noise", type=float, default=0.01)
    parser.add_argument("--action-smooth", type=float, default=0.001)
    parser.add_argument(
        "--policy-kind",
        choices=[
            "bc",
            "flow",
            "sequence_flow",
            "history_act",
            "history_flow",
            "history_act_flow",
            "frozen_clip_flow",
            "frozen_r3m_flow",
            "frozen_smolvlm_flow",
            "mini_pi0_act",
            "mini_pi0_act_resnet",
            "mini_pi0",
            "mini_pi0_resnet",
            "vit_act",
        ],
        default="bc",
    )
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-sigma", type=float, default=1.0)
    parser.add_argument("--flow-source", choices=["noise", "bc"], default="noise")
    parser.add_argument("--flow-eval-start", choices=["zero", "noise", "bc"], default="noise")
    parser.add_argument("--flow-residual-scale", type=float, default=1.0)
    parser.add_argument("--flow-time-sampling", choices=["uniform", "beta_low_noise"], default="uniform")
    parser.add_argument("--bc-aux-weight", type=float, default=0.1)
    parser.add_argument("--vlm-encoder-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--r3m-encoder-name", default="resnet50")
    parser.add_argument("--vlm-cache-batch-size", type=int, default=32)
    parser.add_argument("--frozen-feature-cache-dir", default="data/autorobobench/feature_cache")
    parser.add_argument("--chunk-decay", type=float, default=1.0)
    parser.add_argument("--transformer-depth", type=int, default=3)
    parser.add_argument("--action-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--history-stride", type=int, default=16)
    parser.add_argument("--progress-conditioning", action="store_true")
    parser.add_argument("--progress-scale", type=float, default=260.0)
    parser.add_argument("--task-action-normalization", action="store_true")
    parser.add_argument("--eval-commit-steps", type=int, default=16)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--freeze-non-flow", action="store_true")
    parser.add_argument("--video-pretrain-seconds", type=float, default=0.0)
    parser.add_argument("--video-pretrain-episodes-per-task", type=int, default=0)
    parser.add_argument("--video-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--video-pretrain-gap", type=int, default=8)
    parser.add_argument("--video-pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--video-pretrain-temperature", type=float, default=0.1)
    parser.add_argument(
        "--video-pretrain-objective",
        choices=["temporal_infonce", "latent_dynamics", "hybrid"],
        default="temporal_infonce",
    )
    parser.add_argument("--video-latent-dynamics-weight", type=float, default=1.0)
    parser.add_argument("--pidm-pretrain-seconds", type=float, default=0.0)
    parser.add_argument("--pidm-video-episodes-per-task", type=int, default=0)
    parser.add_argument("--pidm-batch-size", type=int, default=128)
    parser.add_argument("--pidm-lr", type=float, default=3e-4)
    parser.add_argument("--pidm-gap", type=int, default=4)
    parser.add_argument("--pidm-action-weight", type=float, default=1.0)
    parser.add_argument("--pidm-latent-weight", type=float, default=1.0)
    parser.add_argument("--vpt-idm-seconds", type=float, default=0.0)
    parser.add_argument("--vpt-pseudo-episodes-per-task", type=int, default=0)
    parser.add_argument("--vpt-idm-batch-size", type=int, default=128)
    parser.add_argument("--vpt-idm-lr", type=float, default=3e-4)
    parser.add_argument("--vpt-pseudo-weight", type=float, default=1.0)
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")
    if float(args.max_train_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task_aliases = set(args.task_alias)
    train_data, val_data, split_summary = load_split_data(
        manifest,
        split,
        task_aliases=task_aliases,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("need both train and val samples for RoboCasa BC-5")

    vpt_metrics = _maybe_add_vpt_pseudo_labels(
        train_data=train_data,
        manifest=manifest,
        split=split,
        task_aliases=task_aliases,
        idm_seconds=float(args.vpt_idm_seconds),
        pseudo_episodes_per_task=int(args.vpt_pseudo_episodes_per_task),
        batch_size=int(args.vpt_idm_batch_size),
        lr=float(args.vpt_idm_lr),
        pseudo_weight=float(args.vpt_pseudo_weight),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
        seed=int(args.seed),
        device=device_from_arg(args.device),
    )

    raw_proprio_dim = int(train_data.proprio.shape[-1])
    if args.progress_conditioning:
        _append_progress_features(train_data, float(args.progress_scale))
        _append_progress_features(val_data, float(args.progress_scale))

    proprio_mean, proprio_std = _mean_std(train_data.proprio)
    action_mean, action_std = _weighted_masked_mean_std(train_data.actions, train_data.mask)
    task_action_mean = None
    task_action_std = None
    if args.task_action_normalization:
        task_action_mean, task_action_std = _per_task_action_stats(train_data)
    train_data.proprio = ((train_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val_data.proprio = ((val_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    if args.task_action_normalization:
        train_data.actions = _normalize_actions_by_task(
            train_data.actions,
            train_data.task_id,
            task_action_mean,
            task_action_std,
        )
        val_data.actions = _normalize_actions_by_task(
            val_data.actions,
            val_data.task_id,
            task_action_mean,
            task_action_std,
        )
    else:
        train_data.actions = ((train_data.actions - action_mean) / action_std).astype(np.float32)
        val_data.actions = ((val_data.actions - action_mean) / action_std).astype(np.float32)
    if args.policy_kind in {
        "history_act",
        "history_flow",
        "history_act_flow",
        "frozen_clip_flow",
        "frozen_r3m_flow",
        "frozen_smolvlm_flow",
        "mini_pi0_act",
        "mini_pi0_act_resnet",
        "mini_pi0",
        "mini_pi0_resnet",
        "vit_act",
    }:
        _attach_history(train_data, int(args.history_stride))
        _attach_history(val_data, int(args.history_stride))

    device = device_from_arg(args.device)
    task_count = max(1, int(max(train_data.task_id.max(initial=0), val_data.task_id.max(initial=0)) + 1))
    task_texts = _task_texts_for_split(manifest, split, task_aliases)
    if args.policy_kind == "frozen_clip_flow":
        model = RoboCasaFrozenCLIPFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            task_texts=task_texts,
            encoder_name=str(args.vlm_encoder_name),
            width=int(args.width),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "frozen_smolvlm_flow":
        model = RoboCasaFrozenSmolVLMFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            task_texts=task_texts,
            encoder_name=str(args.vlm_encoder_name),
            width=int(args.width),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "frozen_r3m_flow":
        model = RoboCasaFrozenR3MFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            task_texts=task_texts,
            encoder_name=str(args.r3m_encoder_name),
            width=int(args.width),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0_act_resnet":
        model = RoboCasaMiniPi0ACTResNetPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0_act":
        model = RoboCasaMiniPi0ACTPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0_resnet":
        model = RoboCasaMiniPi0ResNetPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0":
        model = RoboCasaMiniPi0Policy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "vit_act":
        model = RoboCasaPatchViTACTPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "history_act_flow":
        model = RoboCasaHistoryACTFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "history_flow":
        model = RoboCasaHistoryFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "history_act":
        model = RoboCasaHistoryACTPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "sequence_flow":
        model = RoboCasaSequenceFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    else:
        model = RoboCasaTemporalChunkBC(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            dropout=float(args.dropout),
        ).to(device)
    init_info = _load_init_checkpoint(model, str(args.init_checkpoint), device)
    freeze_info = _freeze_non_flow(model) if args.freeze_non_flow else _parameter_trainability(model)
    training_budget_start = time.monotonic()
    pidm_metrics = _maybe_pidm_pretrain(
        model=model,
        manifest=manifest,
        split=split,
        video_pool_path=Path(args.video_pool) if args.video_pool else None,
        task_aliases=task_aliases,
        max_train_seconds=float(args.pidm_pretrain_seconds),
        video_episodes_per_task=int(args.pidm_video_episodes_per_task),
        batch_size=int(args.pidm_batch_size),
        gap=int(args.pidm_gap),
        lr=float(args.pidm_lr),
        action_weight=float(args.pidm_action_weight),
        latent_weight=float(args.pidm_latent_weight),
        seed=int(args.seed),
        device=device,
    )
    video_pretrain_metrics = _maybe_video_pretrain(
        model=model,
        manifest=manifest,
        split=split,
        video_pool_path=Path(args.video_pool) if args.video_pool else None,
        task_aliases=task_aliases,
        max_train_seconds=float(args.video_pretrain_seconds),
        episodes_per_task=int(args.video_pretrain_episodes_per_task),
        batch_size=int(args.video_pretrain_batch_size),
        gap=int(args.video_pretrain_gap),
        lr=float(args.video_pretrain_lr),
        temperature=float(args.video_pretrain_temperature),
        objective=str(args.video_pretrain_objective),
        latent_dynamics_weight=float(args.video_latent_dynamics_weight),
        seed=int(args.seed),
        device=device,
    )
    clip_train_data = None
    clip_val_data = None
    clip_cache_metrics = {"enabled": False}
    if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
        cache_start = time.monotonic()
        feature_cache_dir = Path(args.frozen_feature_cache_dir) if args.frozen_feature_cache_dir else None
        encoder_name = str(
            args.r3m_encoder_name
            if args.policy_kind == "frozen_r3m_flow"
            else args.vlm_encoder_name
        )
        clip_train_data = _cache_clip_features(
            model,
            train_data,
            device=device,
            batch_size=int(args.vlm_cache_batch_size),
            label="train",
            cache_path=_feature_cache_path(
                feature_cache_dir,
                label="train",
                data=train_data,
                policy_kind=str(args.policy_kind),
                encoder_name=encoder_name,
                feature_dim=int(model.feature_dim),
                manifest_path=str(args.manifest),
                split_path=str(args.split),
                chunk_horizon=int(args.chunk_horizon),
                frame_stride=int(args.frame_stride),
                history_stride=int(args.history_stride),
                task_aliases=sorted(task_aliases),
                train_episodes_per_task=int(args.train_episodes_per_task),
                val_episodes_per_task=int(args.val_episodes_per_task),
            ),
        )
        clip_val_data = _cache_clip_features(
            model,
            val_data,
            device=device,
            batch_size=int(args.vlm_cache_batch_size),
            label="val",
            cache_path=_feature_cache_path(
                feature_cache_dir,
                label="val",
                data=val_data,
                policy_kind=str(args.policy_kind),
                encoder_name=encoder_name,
                feature_dim=int(model.feature_dim),
                manifest_path=str(args.manifest),
                split_path=str(args.split),
                chunk_horizon=int(args.chunk_horizon),
                frame_stride=int(args.frame_stride),
                history_stride=int(args.history_stride),
                task_aliases=sorted(task_aliases),
                train_episodes_per_task=int(args.train_episodes_per_task),
                val_episodes_per_task=int(args.val_episodes_per_task),
            ),
        )
        clip_cache_metrics = {
            "enabled": True,
            "encoder_name": encoder_name,
            "cache_dir": str(feature_cache_dir or ""),
            "policy_kind": str(args.policy_kind),
            "feature_dim": int(model.feature_dim),
            "train_samples": int(len(clip_train_data)),
            "val_samples": int(len(clip_val_data)),
            "seconds": float(time.monotonic() - cache_start),
        }
    opt = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val_loss = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = training_budget_start

    step = 0
    while True:
        if time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        step += 1
        sample_data = (
            clip_train_data
            if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"} and clip_train_data is not None
            else train_data
        )
        idx = _sample_indices(sample_data, int(args.batch_size), rng, balanced=bool(args.balanced_sampling))
        if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
            batch = _clip_feature_batch(clip_train_data, idx, device)
            batch = _augment_clip_features(batch, float(args.proprio_noise))
            loss = _frozen_clip_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_weight=float(args.bc_aux_weight),
                time_sampling=str(args.flow_time_sampling),
            )
        elif args.policy_kind in {"mini_pi0", "mini_pi0_resnet"}:
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _mini_pi0_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                chunk_decay=float(args.chunk_decay),
                time_sampling=str(args.flow_time_sampling),
            )
        elif args.policy_kind in {"mini_pi0_act", "mini_pi0_act_resnet", "vit_act"}:
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(
                batch["prev_agent"],
                batch["prev_wrist"],
                batch["agent"],
                batch["wrist"],
                batch["prev_proprio"],
                batch["proprio"],
                batch["task_id"],
            )
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        elif args.policy_kind == "history_act_flow":
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _history_act_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_weight=float(args.bc_aux_weight),
            )
        elif args.policy_kind == "history_flow":
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _history_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_aux_weight=float(args.bc_aux_weight),
            )
        elif args.policy_kind == "history_act":
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(
                batch["prev_agent"],
                batch["prev_wrist"],
                batch["agent"],
                batch["wrist"],
                batch["prev_proprio"],
                batch["proprio"],
                batch["task_id"],
            )
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        elif args.policy_kind == "sequence_flow":
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _sequence_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_aux_weight=float(args.bc_aux_weight),
            )
        elif args.policy_kind == "flow":
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _flow_matching_loss(model, batch, sigma=float(args.flow_sigma), chunk_decay=float(args.chunk_decay))
        else:
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        row = {"step": step, "train_loss": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time}
        if step == 1 or step % int(args.log_interval) == 0:
            if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
                val_loss = _eval_clip_feature_loss(
                    model,
                    clip_val_data,
                    device,
                    batch_size=max(64, int(args.batch_size)),
                    flow_steps=int(args.flow_steps),
                    flow_eval_start=str(args.flow_eval_start),
                    flow_residual_scale=float(args.flow_residual_scale),
                )
            else:
                val_loss = _eval_policy_loss(
                    model,
                    val_data,
                    device,
                    batch_size=max(64, int(args.batch_size)),
                    policy_kind=str(args.policy_kind),
                    flow_steps=int(args.flow_steps),
                    flow_eval_start=str(args.flow_eval_start),
                    flow_residual_scale=float(args.flow_residual_scale),
                )
            row["val_loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = _checkpoint_state_dict(model, str(args.policy_kind))
            print(f"step={step} train_loss={row['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
        history.append(row)

    if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
        final_val_loss = _eval_clip_feature_loss(
            model,
            clip_val_data,
            device,
            batch_size=max(64, int(args.batch_size)),
            flow_steps=int(args.flow_steps),
            flow_eval_start=str(args.flow_eval_start),
            flow_residual_scale=float(args.flow_residual_scale),
        )
    else:
        final_val_loss = _eval_policy_loss(
            model,
            val_data,
            device,
            batch_size=max(64, int(args.batch_size)),
            policy_kind=str(args.policy_kind),
            flow_steps=int(args.flow_steps),
            flow_eval_start=str(args.flow_eval_start),
            flow_residual_scale=float(args.flow_residual_scale),
        )
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_step = int(history[-1]["step"] if history else 0)
        best_state = _checkpoint_state_dict(model, str(args.policy_kind))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": _checkpoint_state_dict(model, str(args.policy_kind)),
        "policy_type": (
            "autorobobench_robocasa_bc5_frozen_clip_flow"
            if args.policy_kind == "frozen_clip_flow"
            else "autorobobench_robocasa_bc5_frozen_smolvlm_flow"
            if args.policy_kind == "frozen_smolvlm_flow"
            else "autorobobench_robocasa_bc5_frozen_r3m_flow"
            if args.policy_kind == "frozen_r3m_flow"
            else "autorobobench_robocasa_bc5_history_act"
            if args.policy_kind == "history_act"
            else "autorobobench_robocasa_bc5_mini_pi0_act_resnet"
            if args.policy_kind == "mini_pi0_act_resnet"
            else "autorobobench_robocasa_bc5_mini_pi0_act"
            if args.policy_kind == "mini_pi0_act"
            else "autorobobench_robocasa_bc5_mini_pi0_resnet"
            if args.policy_kind == "mini_pi0_resnet"
            else "autorobobench_robocasa_bc5_mini_pi0"
            if args.policy_kind == "mini_pi0"
            else "autorobobench_robocasa_bc5_vit_act"
            if args.policy_kind == "vit_act"
            else "autorobobench_robocasa_bc5_history_act_flow"
            if args.policy_kind == "history_act_flow"
            else "autorobobench_robocasa_bc5_history_flow"
            if args.policy_kind == "history_flow"
            else "autorobobench_robocasa_bc5_sequence_flow"
            if args.policy_kind == "sequence_flow"
            else "autorobobench_robocasa_bc5_temporal_chunk"
        ),
        "chunk_horizon": int(args.chunk_horizon),
        "action_dim": int(train_data.actions.shape[-1]),
        "proprio_dim": int(train_data.proprio.shape[-1]),
        "task_count": task_count,
        "width": int(args.width),
        "dropout": float(args.dropout),
        "policy_kind": str(args.policy_kind),
        "flow_steps": int(args.flow_steps),
        "flow_sigma": float(args.flow_sigma),
        "flow_source": str(args.flow_source),
        "flow_eval_start": str(args.flow_eval_start),
        "flow_inference_start": str(args.flow_eval_start),
        "flow_residual_scale": float(args.flow_residual_scale),
        "flow_time_sampling": str(args.flow_time_sampling),
        "bc_aux_weight": float(args.bc_aux_weight),
        "chunk_decay": float(args.chunk_decay),
        "transformer_depth": int(args.transformer_depth),
        "action_depth": int(args.action_depth),
        "heads": int(args.heads),
        "patch_size": 8 if args.policy_kind == "vit_act" else None,
        "vlm_encoder_name": str(args.vlm_encoder_name),
        "r3m_encoder_name": str(args.r3m_encoder_name),
        "task_texts": task_texts,
        "history_stride": int(args.history_stride),
        "progress_conditioning": bool(args.progress_conditioning),
        "progress_scale": float(args.progress_scale),
        "progress_feature_dim": 4 if args.progress_conditioning else 0,
        "task_action_normalization": bool(args.task_action_normalization),
        "task_action_mean": task_action_mean,
        "task_action_std": task_action_std,
        "eval_commit_steps": int(args.eval_commit_steps),
        "raw_proprio_dim": raw_proprio_dim,
        "condition_on_robocasa_task_index": False,
        "init_checkpoint": str(args.init_checkpoint),
        "init_info": init_info,
        "freeze_non_flow": bool(args.freeze_non_flow),
        "freeze_info": freeze_info,
        "pidm_pretrain": pidm_metrics,
        "video_pretrain_objective": str(args.video_pretrain_objective),
        "video_latent_dynamics_weight": float(args.video_latent_dynamics_weight),
        "video_pretrain": video_pretrain_metrics,
        "clip_feature_cache": clip_cache_metrics,
        "vpt_pseudo_labels": vpt_metrics,
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "manifest": str(Path(args.manifest)),
        "split": str(Path(args.split)),
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    torch.save(checkpoint, out_dir / "policy.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_action_mse_normalized"] = float(best_val_loss)
    torch.save(best_checkpoint, out_dir / "policy_best.pt")

    metrics = {
        "checkpoint": str(out_dir / "policy.pt"),
        "best_checkpoint": str(out_dir / "policy_best.pt"),
        "best_step": int(best_step),
        "best_val_action_mse_normalized": float(best_val_loss),
        "final_val_action_mse_normalized": float(final_val_loss),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "split_summary": split_summary,
        "chunk_horizon": int(args.chunk_horizon),
        "frame_stride": int(args.frame_stride),
        "train_episodes_per_task": int(args.train_episodes_per_task),
        "val_episodes_per_task": int(args.val_episodes_per_task),
        "steps_completed": int(history[-1]["step"] if history else 0),
        "train_seconds": float(time.monotonic() - start_time),
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
        "flow_source": str(args.flow_source),
        "flow_eval_start": str(args.flow_eval_start),
        "flow_inference_start": str(args.flow_eval_start),
        "flow_residual_scale": float(args.flow_residual_scale),
        "flow_time_sampling": str(args.flow_time_sampling),
        "bc_aux_weight": float(args.bc_aux_weight),
        "chunk_decay": float(args.chunk_decay),
        "transformer_depth": int(args.transformer_depth),
        "action_depth": int(args.action_depth),
        "heads": int(args.heads),
        "r3m_encoder_name": str(args.r3m_encoder_name),
        "history_stride": int(args.history_stride),
        "progress_conditioning": bool(args.progress_conditioning),
        "progress_scale": float(args.progress_scale),
        "progress_feature_dim": 4 if args.progress_conditioning else 0,
        "eval_commit_steps": int(args.eval_commit_steps),
        "balanced_sampling": bool(args.balanced_sampling),
        "seed": int(args.seed),
        "init_checkpoint": str(args.init_checkpoint),
        "init_info": init_info,
        "freeze_non_flow": bool(args.freeze_non_flow),
        "freeze_info": freeze_info,
        "pidm_pretrain": pidm_metrics,
        "video_pretrain_objective": str(args.video_pretrain_objective),
        "video_latent_dynamics_weight": float(args.video_latent_dynamics_weight),
        "video_pretrain": video_pretrain_metrics,
        "vpt_pseudo_labels": vpt_metrics,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


class _MiniInverseDynamics(nn.Module):
    def __init__(self, *, proprio_dim: int, action_dim: int, task_count: int, width: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.image = nn.Sequential(
            nn.Conv2d(12, 32, 4, stride=2, padding=1),
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
        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.task = nn.Embedding(task_count, 32)
        self.head = nn.Sequential(
            nn.Linear(2 * width + 32, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, action_dim),
        )

    def forward(
        self,
        agent_t: torch.Tensor,
        wrist_t: torch.Tensor,
        agent_tp1: torch.Tensor,
        wrist_tp1: torch.Tensor,
        proprio_t: torch.Tensor,
        proprio_tp1: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent_t.max() > 1.5:
            agent_t = agent_t / 255.0
            wrist_t = wrist_t / 255.0
            agent_tp1 = agent_tp1 / 255.0
            wrist_tp1 = wrist_tp1 / 255.0
        image = self.image(torch.cat([agent_t, wrist_t, agent_tp1, wrist_tp1], dim=1))
        prop = self.proprio(torch.cat([proprio_t, proprio_tp1, proprio_tp1 - proprio_t], dim=-1))
        return self.head(torch.cat([image, prop, self.task(task_id)], dim=-1))


def _maybe_add_vpt_pseudo_labels(
    *,
    train_data: TemporalChunkData,
    manifest: dict,
    split: dict,
    task_aliases: set[str],
    idm_seconds: float,
    pseudo_episodes_per_task: int,
    batch_size: int,
    lr: float,
    pseudo_weight: float,
    chunk_horizon: int,
    frame_stride: int,
    seed: int,
    device: torch.device,
) -> dict:
    if idm_seconds <= 0 or pseudo_episodes_per_task <= 0:
        return {"enabled": False, "pseudo_samples": 0}
    idm_data, idm_summary = _load_idm_supervised_samples(manifest, split, task_aliases=task_aliases)
    if len(idm_data["actions"]) == 0:
        return {"enabled": False, "pseudo_samples": 0, "reason": "no idm samples"}
    task_count = max(1, int(max(idm_data["task_id"].max(initial=0), train_data.task_id.max(initial=0)) + 1))
    idm = _MiniInverseDynamics(
        proprio_dim=int(idm_data["proprio_t"].shape[-1]),
        action_dim=int(idm_data["actions"].shape[-1]),
        task_count=task_count,
        width=256,
    ).to(device)
    opt = torch.optim.AdamW(idm.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 29)
    history: list[dict] = []
    start_time = time.monotonic()
    idm.train()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(idm_seconds):
            break
        step += 1
        idx = rng.integers(0, len(idm_data["actions"]), size=batch_size)
        batch = _idm_batch(idm_data, idx, device)
        pred = idm(**{key: value for key, value in batch.items() if key != "actions"})
        loss = F.mse_loss(pred, batch["actions"])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(idm.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            row = {"step": step, "idm_action_mse": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time}
            history.append(row)
            print(f"vpt_idm step={step} action_mse={row['idm_action_mse']:.6f}", flush=True)

    pseudo, pseudo_summary = _pseudo_label_video_only(
        idm,
        manifest,
        split,
        task_aliases=task_aliases,
        episodes_per_task=pseudo_episodes_per_task,
        chunk_horizon=chunk_horizon,
        frame_stride=frame_stride,
        device=device,
    )
    if len(pseudo) > 0:
        pseudo.mask = (pseudo.mask * max(0.0, float(pseudo_weight))).astype(np.float32)
        _append_temporal_data_(train_data, pseudo)
    return {
        "enabled": True,
        "method": "mini_vpt_inverse_dynamics_pseudo_labels",
        "idm_seconds": float(idm_seconds),
        "idm_steps_completed": int(step),
        "idm_samples": int(len(idm_data["actions"])),
        "idm_summary": idm_summary,
        "idm_history": history,
        "pseudo_samples": int(len(pseudo)),
        "pseudo_weight": float(pseudo_weight),
        "pseudo_summary": pseudo_summary,
        "seconds": float(time.monotonic() - start_time),
    }


def _maybe_pidm_pretrain(
    *,
    model: nn.Module,
    manifest: dict,
    split: dict,
    video_pool_path: Path | None,
    task_aliases: set[str],
    max_train_seconds: float,
    video_episodes_per_task: int,
    batch_size: int,
    gap: int,
    lr: float,
    action_weight: float,
    latent_weight: float,
    seed: int,
    device: torch.device,
) -> dict:
    if max_train_seconds <= 0:
        return {"enabled": False, "steps_completed": 0}
    if not hasattr(model, "vision") and not hasattr(model, "image"):
        return {"enabled": False, "steps_completed": 0, "reason": "model has no image encoder"}
    in_channels = _first_conv_in_channels(_image_encoder(model))
    if in_channels != 6:
        return {
            "enabled": False,
            "steps_completed": 0,
            "reason": f"PIDM pretrain expects 6-channel encoder, got {in_channels}",
        }

    idm_data, idm_summary = _load_idm_supervised_samples(manifest, split, task_aliases=task_aliases)
    if len(idm_data["actions"]) == 0:
        return {"enabled": False, "steps_completed": 0, "reason": "no paired inverse-dynamics samples"}
    video_samples, video_summary = _load_video_transition_samples(
        manifest,
        split,
        video_pool_path=video_pool_path,
        task_aliases=task_aliases,
        episodes_per_task=video_episodes_per_task,
        gap=max(1, gap),
    )
    if len(video_samples["agent_t"]) == 0:
        return {"enabled": False, "steps_completed": 0, "reason": "no video transitions"}

    width = _encoder_width(model)
    action_dim = int(idm_data["actions"].shape[-1])
    task_count = max(
        1,
        int(
            max(
                idm_data["task_id"].max(initial=0),
                video_samples["task_id"].max(initial=0),
                max(int(task["task_id"]) for task in split["tasks"]),
            )
            + 1
        ),
    )
    task_emb = nn.Embedding(task_count, 32).to(device)
    action_head = nn.Sequential(
        nn.Linear(2 * width + 32, 2 * width),
        nn.SiLU(),
        nn.Linear(2 * width, action_dim),
    ).to(device)
    latent_predictor = nn.Sequential(
        nn.Linear(width + 32, width),
        nn.SiLU(),
        nn.Linear(width, width),
    ).to(device)
    params = (
        list(_image_encoder(model).parameters())
        + list(task_emb.parameters())
        + list(action_head.parameters())
        + list(latent_predictor.parameters())
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 43)
    history: list[dict] = []
    start_time = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(max_train_seconds):
            break
        step += 1
        idm_idx = rng.integers(0, len(idm_data["actions"]), size=batch_size)
        idm_batch = _idm_batch(idm_data, idm_idx, device)
        z_t = _encode_video_pair(
            model,
            idm_batch["agent_t"] / 255.0,
            idm_batch["wrist_t"] / 255.0,
        )
        z_tp1 = _encode_video_pair(
            model,
            idm_batch["agent_tp1"] / 255.0,
            idm_batch["wrist_tp1"] / 255.0,
        )
        task = task_emb(idm_batch["task_id"].clamp_max(task_count - 1))
        pred_action = action_head(torch.cat([z_t, z_tp1, task], dim=-1))
        action_loss = F.mse_loss(pred_action, idm_batch["actions"])

        video_idx = rng.integers(0, len(video_samples["agent_t"]), size=batch_size)
        agent_t = torch.as_tensor(video_samples["agent_t"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_t = torch.as_tensor(video_samples["wrist_t"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        agent_tp1 = torch.as_tensor(video_samples["agent_tp1"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_tp1 = torch.as_tensor(video_samples["wrist_tp1"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        z_video_t = _encode_video_pair(model, agent_t, wrist_t)
        with torch.no_grad():
            z_video_tp1 = F.normalize(_encode_video_pair(model, agent_tp1, wrist_tp1), dim=-1)
        video_task_id = torch.as_tensor(
            video_samples["task_id"][video_idx],
            dtype=torch.long,
            device=device,
        ).clamp_max(task_count - 1)
        pred_latent = F.normalize(
            latent_predictor(torch.cat([z_video_t, task_emb(video_task_id)], dim=-1)),
            dim=-1,
        )
        latent_loss = F.mse_loss(pred_latent, z_video_tp1)
        loss = float(action_weight) * action_loss + float(latent_weight) * latent_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            row = {
                "step": int(step),
                "pidm_loss": float(loss.detach().cpu()),
                "pidm_action_mse": float(action_loss.detach().cpu()),
                "pidm_latent_mse": float(latent_loss.detach().cpu()),
                "elapsed_seconds": float(time.monotonic() - start_time),
            }
            history.append(row)
            print(
                "pidm step={step} loss={loss:.6f} action={action:.6f} latent={latent:.6f}".format(
                    step=step,
                    loss=row["pidm_loss"],
                    action=row["pidm_action_mse"],
                    latent=row["pidm_latent_mse"],
                ),
                flush=True,
            )
    return {
        "enabled": True,
        "method": "future_visual_latent_plus_inverse_dynamics",
        "max_train_seconds": float(max_train_seconds),
        "steps_completed": int(step),
        "paired_samples": int(len(idm_data["actions"])),
        "video_samples": int(len(video_samples["agent_t"])),
        "video_episodes_per_task": int(video_episodes_per_task),
        "batch_size": int(batch_size),
        "gap": int(gap),
        "lr": float(lr),
        "action_weight": float(action_weight),
        "latent_weight": float(latent_weight),
        "idm_summary": idm_summary,
        "video_summary": video_summary,
        "history": history,
        "seconds": float(time.monotonic() - start_time),
    }


def _load_idm_supervised_samples(
    manifest: dict,
    split: dict,
    *,
    task_aliases: set[str],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    parts: dict[str, list[np.ndarray]] = {
        "agent_t": [],
        "wrist_t": [],
        "agent_tp1": [],
        "wrist_tp1": [],
        "proprio_t": [],
        "proprio_tp1": [],
        "task_id": [],
        "actions": [],
    }
    summary: list[dict] = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        episode_ids = [int(x) for x in split_task.get("paired_train_episode_ids", split_task.get("train_episode_ids", []))]
        count = 0
        for episode_id in episode_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
            frame = pd.read_parquet(episode_path)
            agent = _read_video64(dataset_root, episode_id, "robot0_agentview_left")
            wrist = _read_video64(dataset_root, episode_id, "robot0_agentview_right")
            proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
            actions = LU.get_episode_actions(dataset_root, episode_id).astype(np.float32)
            n = min(len(agent), len(wrist), len(proprio), len(actions))
            if n <= 1:
                continue
            rows = np.arange(0, n - 1, dtype=np.int32)
            parts["agent_t"].append(agent[rows])
            parts["wrist_t"].append(wrist[rows])
            parts["agent_tp1"].append(agent[rows + 1])
            parts["wrist_tp1"].append(wrist[rows + 1])
            parts["proprio_t"].append(proprio[rows])
            parts["proprio_tp1"].append(proprio[rows + 1])
            parts["actions"].append(actions[rows])
            parts["task_id"].append(np.full((len(rows),), int(split_task["task_id"]), dtype=np.int64))
            count += len(rows)
        summary.append({"alias": alias, "paired_episode_ids": episode_ids, "idm_samples": count})
        print(f"loaded idm paired {alias}: episodes={episode_ids} samples={count}", flush=True)
    return _concat_idm_parts(parts), summary


def _concat_idm_parts(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, values in parts.items():
        if values:
            out[key] = np.concatenate(values, axis=0)
        elif key == "task_id":
            out[key] = np.zeros((0,), dtype=np.int64)
        elif key == "actions":
            out[key] = np.zeros((0, 7), dtype=np.float32)
        else:
            out[key] = np.zeros((0, 64, 64, 3), dtype=np.uint8)
    return out


def _idm_batch(data: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent_t": torch.as_tensor(data["agent_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist_t": torch.as_tensor(data["wrist_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "agent_tp1": torch.as_tensor(data["agent_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist_tp1": torch.as_tensor(data["wrist_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "proprio_t": torch.as_tensor(data["proprio_t"][idx], dtype=torch.float32, device=device),
        "proprio_tp1": torch.as_tensor(data["proprio_tp1"][idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data["task_id"][idx], dtype=torch.long, device=device),
        "actions": torch.as_tensor(data["actions"][idx], dtype=torch.float32, device=device),
    }


def _pseudo_label_video_only(
    idm: _MiniInverseDynamics,
    manifest: dict,
    split: dict,
    *,
    task_aliases: set[str],
    episodes_per_task: int,
    chunk_horizon: int,
    frame_stride: int,
    device: torch.device,
) -> tuple[TemporalChunkData, list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    parts: list[dict[str, np.ndarray]] = []
    summary: list[dict] = []
    idm.eval()
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        video_ids = _video_only_ids(split_task)[:episodes_per_task]
        sample_count = 0
        for episode_id in video_ids:
            part = _pseudo_episode_samples(
                idm,
                dataset_root,
                int(episode_id),
                int(split_task["task_id"]),
                chunk_horizon,
                frame_stride,
                device,
            )
            parts.append(part)
            sample_count += int(part["agent"].shape[0])
        summary.append({"alias": alias, "video_episode_ids": [int(x) for x in video_ids], "pseudo_samples": sample_count})
        print(f"pseudo-labeled {alias}: episodes={video_ids} samples={sample_count}", flush=True)
    return _concat_parts(parts), summary


def _pseudo_episode_samples(
    idm: _MiniInverseDynamics,
    dataset_root: Path,
    episode_idx: int,
    task_id: int,
    chunk_horizon: int,
    frame_stride: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    frame = pd.read_parquet(episode_path)
    agent = _read_video64(dataset_root, episode_idx, "robot0_agentview_left")
    wrist = _read_video64(dataset_root, episode_idx, "robot0_agentview_right")
    proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    n = min(len(agent), len(wrist), len(proprio))
    pred_actions = np.zeros((max(0, n - 1), int(idm.action_dim)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, max(0, n - 1), 256):
            rows = np.arange(start, min(n - 1, start + 256), dtype=np.int32)
            batch = {
                "agent_t": torch.as_tensor(agent[rows], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "wrist_t": torch.as_tensor(wrist[rows], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "agent_tp1": torch.as_tensor(agent[rows + 1], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "wrist_tp1": torch.as_tensor(wrist[rows + 1], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "proprio_t": torch.as_tensor(proprio[rows], dtype=torch.float32, device=device),
                "proprio_tp1": torch.as_tensor(proprio[rows + 1], dtype=torch.float32, device=device),
                "task_id": torch.full((len(rows),), task_id, dtype=torch.long, device=device),
            }
            pred_actions[rows] = idm(**batch).detach().cpu().numpy().astype(np.float32)
    starts = np.arange(0, max(0, n - 1), max(1, frame_stride), dtype=np.int32)
    out_actions = np.zeros((len(starts), chunk_horizon, pred_actions.shape[-1]), dtype=np.float32)
    mask = np.zeros((len(starts), chunk_horizon), dtype=np.float32)
    for row_idx, start in enumerate(starts):
        end = min(len(pred_actions), int(start) + chunk_horizon)
        length = end - int(start)
        out_actions[row_idx, :length] = pred_actions[int(start) : end]
        mask[row_idx, :length] = 1.0
    return {
        "agent": agent[starts],
        "wrist": wrist[starts],
        "proprio": proprio[starts],
        "actions": out_actions,
        "mask": mask,
        "task_id": np.full((len(starts),), task_id, dtype=np.int64),
        "episode_idx": np.full((len(starts),), episode_idx, dtype=np.int32),
        "frame_idx": starts.astype(np.int32),
    }


def _append_temporal_data_(base: TemporalChunkData, extra: TemporalChunkData) -> None:
    if len(extra) == 0:
        return
    base.agent = np.concatenate([base.agent, extra.agent], axis=0)
    base.wrist = np.concatenate([base.wrist, extra.wrist], axis=0)
    base.proprio = np.concatenate([base.proprio, extra.proprio], axis=0)
    base.actions = np.concatenate([base.actions, extra.actions], axis=0)
    base.mask = np.concatenate([base.mask, extra.mask], axis=0)
    base.task_id = np.concatenate([base.task_id, extra.task_id], axis=0)
    base.episode_idx = np.concatenate([base.episode_idx, extra.episode_idx], axis=0)
    base.frame_idx = np.concatenate([base.frame_idx, extra.frame_idx], axis=0)


def _weighted_masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float32)
    weights = mask.reshape(-1).astype(np.float32)
    keep = weights > 0
    if not np.any(keep):
        return _masked_mean_std(values, mask)
    flat = flat[keep]
    weights = weights[keep]
    denom = max(float(weights.sum()), 1e-6)
    mean = ((flat * weights[:, None]).sum(axis=0) / denom).astype(np.float32)
    var = (((flat - mean[None]) ** 2) * weights[:, None]).sum(axis=0) / denom
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def _per_task_action_stats(data: TemporalChunkData) -> tuple[np.ndarray, np.ndarray]:
    task_count = int(data.task_id.max(initial=0)) + 1
    global_mean, global_std = _weighted_masked_mean_std(data.actions, data.mask)
    means = np.repeat(global_mean[None], task_count, axis=0).astype(np.float32)
    stds = np.repeat(global_std[None], task_count, axis=0).astype(np.float32)
    for task_id in range(task_count):
        keep = data.task_id == task_id
        if not np.any(keep):
            continue
        means[task_id], stds[task_id] = _weighted_masked_mean_std(data.actions[keep], data.mask[keep])
    return means.astype(np.float32), np.maximum(stds, 1e-6).astype(np.float32)


def _normalize_actions_by_task(
    actions: np.ndarray,
    task_id: np.ndarray,
    task_action_mean: np.ndarray,
    task_action_std: np.ndarray,
) -> np.ndarray:
    mean = task_action_mean[np.asarray(task_id, dtype=np.int64)]
    std = task_action_std[np.asarray(task_id, dtype=np.int64)]
    return ((actions - mean[:, None, :]) / std[:, None, :]).astype(np.float32)


def _maybe_video_pretrain(
    *,
    model: nn.Module,
    manifest: dict,
    split: dict,
    video_pool_path: Path | None,
    task_aliases: set[str],
    max_train_seconds: float,
    episodes_per_task: int,
    batch_size: int,
    gap: int,
    lr: float,
    temperature: float,
    objective: str,
    latent_dynamics_weight: float,
    seed: int,
    device: torch.device,
) -> dict:
    if max_train_seconds <= 0 or episodes_per_task <= 0:
        return {"enabled": False, "steps_completed": 0, "samples": 0}
    if not hasattr(model, "vision") and not hasattr(model, "image"):
        return {"enabled": False, "steps_completed": 0, "samples": 0, "reason": "model has no image encoder"}
    in_channels = _first_conv_in_channels(_image_encoder(model))
    if in_channels != 6:
        return {
            "enabled": False,
            "steps_completed": 0,
            "samples": 0,
            "reason": f"video pretrain expects 6-channel encoder, got {in_channels}",
        }

    samples, summary = _load_video_transition_samples(
        manifest,
        split,
        video_pool_path=video_pool_path,
        task_aliases=task_aliases,
        episodes_per_task=episodes_per_task,
        gap=max(1, gap),
    )
    if len(samples["agent_t"]) == 0:
        return {"enabled": False, "steps_completed": 0, "samples": 0, "reason": "no video transitions"}

    width = _encoder_width(model)
    predictor = nn.Sequential(
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, width),
    ).to(device)
    params = list(_image_encoder(model).parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 17)
    history: list[dict] = []
    start_time = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(max_train_seconds):
            break
        step += 1
        idx = rng.integers(0, len(samples["agent_t"]), size=batch_size)
        agent_t = torch.as_tensor(samples["agent_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_t = torch.as_tensor(samples["wrist_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        agent_tp1 = torch.as_tensor(samples["agent_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_tp1 = torch.as_tensor(samples["wrist_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        z_t = _encode_video_pair(model, agent_t, wrist_t)
        z_tp1 = _encode_video_pair(model, agent_tp1, wrist_tp1)
        q_t = F.normalize(predictor(z_t), dim=-1)
        q_tp1 = F.normalize(predictor(z_tp1), dim=-1)
        k_t = F.normalize(z_t, dim=-1)
        k_tp1 = F.normalize(z_tp1, dim=-1)
        labels = torch.arange(q_t.shape[0], dtype=torch.long, device=device)
        logits_fwd = q_t @ k_tp1.T / max(temperature, 1e-6)
        logits_bwd = q_tp1 @ k_t.T / max(temperature, 1e-6)
        nce_loss = 0.5 * (F.cross_entropy(logits_fwd, labels) + F.cross_entropy(logits_bwd, labels))
        dynamics_loss = 0.5 * (
            F.mse_loss(q_t, k_tp1.detach()) + F.mse_loss(q_tp1, k_t.detach())
        )
        if objective == "latent_dynamics":
            loss = float(latent_dynamics_weight) * dynamics_loss
        elif objective == "hybrid":
            loss = nce_loss + float(latent_dynamics_weight) * dynamics_loss
        else:
            loss = nce_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            acc = 0.5 * (
                (logits_fwd.argmax(dim=-1) == labels).float().mean()
                + (logits_bwd.argmax(dim=-1) == labels).float().mean()
            )
            row = {
                "step": step,
                "video_loss": float(loss.detach().cpu()),
                "video_nce_loss": float(nce_loss.detach().cpu()),
                "video_latent_dynamics_loss": float(dynamics_loss.detach().cpu()),
                "video_nce_acc": float(acc.detach().cpu()),
                "elapsed_seconds": time.monotonic() - start_time,
            }
            history.append(row)
            print(
                "video_pretrain step={step} loss={loss:.6f} nce={nce:.6f} "
                "latent={latent:.6f} acc={acc:.3f}".format(
                    step=step,
                    loss=row["video_loss"],
                    nce=row["video_nce_loss"],
                    latent=row["video_latent_dynamics_loss"],
                    acc=row["video_nce_acc"],
                ),
                flush=True,
            )
    return {
        "enabled": True,
        "objective": str(objective),
        "latent_dynamics_weight": float(latent_dynamics_weight),
        "max_train_seconds": float(max_train_seconds),
        "steps_completed": int(step),
        "samples": int(len(samples["agent_t"])),
        "episodes_per_task": int(episodes_per_task),
        "gap": int(gap),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "temperature": float(temperature),
        "summary": summary,
        "history": history,
        "seconds": float(time.monotonic() - start_time),
    }


def _load_video_transition_samples(
    manifest: dict,
    split: dict,
    *,
    video_pool_path: Path | None,
    task_aliases: set[str],
    episodes_per_task: int,
    gap: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    video_pool_tasks = None
    if video_pool_path is None and split.get("video_pool"):
        video_pool_path = Path(str(split["video_pool"]))
    if video_pool_path is not None:
        video_pool = json.loads(video_pool_path.read_text())
        if video_pool.get("contains_actions") is not False or video_pool.get("contains_proprio") is not False:
            raise ValueError(f"video pool must be RGB-only/action-free: {video_pool_path}")
        video_pool_tasks = {task["alias"]: task for task in video_pool["tasks"]}
    parts: dict[str, list[np.ndarray]] = {
        "agent_t": [],
        "wrist_t": [],
        "agent_tp1": [],
        "wrist_tp1": [],
        "task_id": [],
    }
    summary: list[dict] = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        if video_pool_tasks is not None:
            continue
        else:
            task = manifest_tasks[alias]
            dataset_root = Path(task["dataset_path"])
            video_ids = _video_only_ids(split_task)
            video_source = "split"
        if episodes_per_task > 0:
            video_ids = video_ids[:episodes_per_task]
        transition_count = 0
        for episode_id in video_ids:
            agent = _read_video64(dataset_root, int(episode_id), "robot0_agentview_left")
            wrist = _read_video64(dataset_root, int(episode_id), "robot0_agentview_right")
            n = min(len(agent), len(wrist))
            if n <= gap:
                continue
            starts = np.arange(0, n - gap, max(1, gap), dtype=np.int32)
            parts["agent_t"].append(agent[starts])
            parts["wrist_t"].append(wrist[starts])
            parts["agent_tp1"].append(agent[starts + gap])
            parts["wrist_tp1"].append(wrist[starts + gap])
            parts["task_id"].append(np.full((len(starts),), int(split_task["task_id"]), dtype=np.int64))
            transition_count += len(starts)
        summary.append(
            {
                "alias": alias,
                "video_episode_ids": [int(x) for x in video_ids],
                "transitions": transition_count,
                "source": video_source,
                "contains_actions": False,
            }
        )
        print(f"loaded video-only {alias}: episodes={video_ids} transitions={transition_count}", flush=True)
    if video_pool_tasks is not None:
        for alias, task in video_pool_tasks.items():
            if task_aliases and alias not in task_aliases:
                continue
            dataset_root = Path(task["dataset_path"])
            if not dataset_root.is_absolute():
                dataset_root = Path.cwd() / dataset_root
            video_ids = _expand_video_range(task["video_episode_range"])
            if episodes_per_task > 0:
                video_ids = video_ids[:episodes_per_task]
            transition_count = 0
            for episode_id in video_ids:
                agent = _read_video64(dataset_root, int(episode_id), "robot0_agentview_left")
                wrist = _read_video64(dataset_root, int(episode_id), "robot0_agentview_right")
                n = min(len(agent), len(wrist))
                if n <= gap:
                    continue
                starts = np.arange(0, n - gap, max(1, gap), dtype=np.int32)
                parts["agent_t"].append(agent[starts])
                parts["wrist_t"].append(wrist[starts])
                parts["agent_tp1"].append(agent[starts + gap])
                parts["wrist_tp1"].append(wrist[starts + gap])
                parts["task_id"].append(np.full((len(starts),), int(task.get("task_id", 0)), dtype=np.int64))
                transition_count += len(starts)
            summary.append(
                {
                    "alias": alias,
                    "video_episode_ids": [int(x) for x in video_ids],
                    "transitions": transition_count,
                    "source": str(video_pool_path),
                    "contains_actions": False,
                }
            )
            print(f"loaded video-only {alias}: episodes={video_ids} transitions={transition_count}", flush=True)
    out = {}
    for key, value in parts.items():
        if value:
            out[key] = np.concatenate(value, axis=0)
        elif key == "task_id":
            out[key] = np.zeros((0,), dtype=np.int64)
        else:
            out[key] = np.zeros((0, 64, 64, 3), dtype=np.uint8)
    return out, summary


def _video_only_ids(split_task: dict) -> list[int]:
    if "video_only_episode_ids" in split_task:
        return [int(x) for x in split_task["video_only_episode_ids"]]
    if "video_only_episode_range" in split_task:
        start, end = split_task["video_only_episode_range"]
        return list(range(int(start), int(end) + 1))
    paired = set(int(x) for x in split_task.get("paired_train_episode_ids", split_task.get("train_episode_ids", [])))
    return [int(x) for x in split_task.get("train_episode_ids", []) if int(x) not in paired]


def _expand_video_range(bounds: list[int]) -> list[int]:
    if len(bounds) != 2:
        raise ValueError(f"expected [start, end] range, got {bounds!r}")
    return list(range(int(bounds[0]), int(bounds[1]) + 1))


def _image_encoder(model: nn.Module) -> nn.Module:
    if hasattr(model, "vision"):
        return model.vision
    return model.image


def _encoder_width(model: nn.Module) -> int:
    if hasattr(model, "width"):
        return int(model.width)
    if hasattr(model, "head") and isinstance(model.head[0], nn.Linear):
        return int(model.head[0].in_features - model.proprio[-2].out_features - model.task.embedding_dim)
    raise ValueError("could not infer image encoder width")


def _encode_video_pair(model: nn.Module, agent: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    encoder = _image_encoder(model)
    z = encoder(torch.cat([agent, wrist], dim=1))
    if z.ndim == 4:
        z = z.mean(dim=(2, 3))
    return z


def _first_conv_in_channels(module: nn.Module) -> int | None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    return None


def load_split_data(
    manifest: dict,
    split: dict,
    *,
    task_aliases: set[str],
    train_episodes_per_task: int,
    val_episodes_per_task: int,
    chunk_horizon: int,
    frame_stride: int,
) -> tuple[TemporalChunkData, TemporalChunkData, list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    train_parts: list[dict[str, np.ndarray]] = []
    val_parts: list[dict[str, np.ndarray]] = []
    summary: list[dict] = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        task = manifest_tasks[alias]
        task_id = int(split_task["task_id"])
        dataset_root = Path(task["dataset_path"])
        train_ids = list(split_task["train_episode_ids"])
        val_ids = list(split_task["val_episode_ids"])
        if train_episodes_per_task > 0:
            train_ids = train_ids[:train_episodes_per_task]
        if val_episodes_per_task > 0:
            val_ids = val_ids[:val_episodes_per_task]
        for episode_id in train_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{int(episode_id):06d}.parquet"
            train_parts.append(
                _episode_samples(
                    dataset_root,
                    episode_path,
                    int(episode_id),
                    task_id,
                    chunk_horizon,
                    frame_stride,
                    False,
                )
            )
        for episode_id in val_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{int(episode_id):06d}.parquet"
            val_parts.append(
                _episode_samples(
                    dataset_root,
                    episode_path,
                    int(episode_id),
                    task_id,
                    chunk_horizon,
                    frame_stride,
                    False,
                )
            )
        summary.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str(dataset_root),
                "train_episode_ids": [int(x) for x in train_ids],
                "val_episode_ids": [int(x) for x in val_ids],
            }
        )
        print(f"loaded {alias}: train={train_ids} val={val_ids}", flush=True)
    return _concat_parts(train_parts), _concat_parts(val_parts), summary


def _task_texts_for_split(manifest: dict, split: dict, task_aliases: set[str]) -> list[str]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    rows = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        task = manifest_tasks[alias]
        text = str(task.get("description") or task.get("robocasa_task") or alias)
        rows.append((int(split_task["task_id"]), text))
    if not rows:
        return []
    task_count = max(task_id for task_id, _ in rows) + 1
    texts = [f"robot task {idx}" for idx in range(task_count)]
    for task_id, text in rows:
        texts[task_id] = text
    return texts


def _sample_indices(
    data: TemporalChunkData,
    batch_size: int,
    rng: np.random.Generator,
    *,
    balanced: bool,
) -> np.ndarray:
    if not balanced:
        return rng.integers(0, len(data), size=batch_size)
    task_ids = np.unique(data.task_id)
    if len(task_ids) == 0:
        return rng.integers(0, len(data), size=batch_size)
    per_task = int(math.ceil(batch_size / len(task_ids)))
    parts: list[np.ndarray] = []
    for task_id in task_ids:
        pool = np.flatnonzero(data.task_id == int(task_id))
        if len(pool) == 0:
            continue
        parts.append(rng.choice(pool, size=per_task, replace=len(pool) < per_task))
    if not parts:
        return rng.integers(0, len(data), size=batch_size)
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    if len(idx) < batch_size:
        extra = rng.integers(0, len(data), size=batch_size - len(idx))
        idx = np.concatenate([idx, extra])
    return idx[:batch_size]


def _append_progress_features(data: TemporalChunkData, progress_scale: float) -> None:
    progress = _progress_features(data.frame_idx.astype(np.float32), progress_scale)
    data.proprio = np.concatenate([data.proprio, progress], axis=-1).astype(np.float32)


def _progress_features(frame_idx: np.ndarray, progress_scale: float) -> np.ndarray:
    progress = np.clip(frame_idx.astype(np.float32) / max(float(progress_scale), 1.0), 0.0, 1.5)
    return np.stack(
        [
            progress,
            progress * progress,
            np.sin(np.pi * progress),
            np.cos(np.pi * progress),
        ],
        axis=-1,
    ).astype(np.float32)


def _attach_history(data: TemporalChunkData, history_stride: int) -> None:
    history_stride = max(0, int(history_stride))
    prev_idx = np.arange(len(data), dtype=np.int64)
    for episode_id in np.unique(data.episode_idx):
        rows = np.flatnonzero(data.episode_idx == int(episode_id))
        if len(rows) == 0:
            continue
        order = rows[np.argsort(data.frame_idx[rows])]
        frames = data.frame_idx[order]
        if history_stride <= 0:
            prev_idx[order] = order
            continue
        targets = frames - history_stride
        positions = np.searchsorted(frames, targets, side="right") - 1
        positions = np.maximum(positions, 0)
        prev_idx[order] = order[positions]
    data.prev_agent = data.agent[prev_idx]
    data.prev_wrist = data.wrist[prev_idx]
    data.prev_proprio = data.proprio[prev_idx]


def _history_batch(data: TemporalChunkData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "prev_agent": torch.as_tensor(data.prev_agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "prev_wrist": torch.as_tensor(data.prev_wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "agent": torch.as_tensor(data.agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(data.wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "prev_proprio": torch.as_tensor(data.prev_proprio[idx], dtype=torch.float32, device=device),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data.actions[idx], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(data.mask[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


@dataclass
class FrozenCLIPFeatureData:
    image_features: np.ndarray
    prev_proprio: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    task_id: np.ndarray
    episode_idx: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.image_features.shape[0])


def _feature_cache_path(
    cache_dir: Path | None,
    *,
    label: str,
    data: TemporalChunkData,
    policy_kind: str,
    encoder_name: str,
    feature_dim: int,
    manifest_path: str,
    split_path: str,
    chunk_horizon: int,
    frame_stride: int,
    history_stride: int,
    task_aliases: list[str],
    train_episodes_per_task: int,
    val_episodes_per_task: int,
) -> Path | None:
    if cache_dir is None:
        return None
    identity = hashlib.sha256()
    for array in (
        np.asarray(data.task_id, dtype=np.int64),
        np.asarray(data.episode_idx, dtype=np.int64),
        np.asarray(data.frame_idx, dtype=np.int64),
    ):
        identity.update(np.ascontiguousarray(array).view(np.uint8))
    payload = {
        "version": 1,
        "label": label,
        "policy_kind": policy_kind,
        "encoder_name": encoder_name,
        "feature_dim": int(feature_dim),
        "manifest_path": str(Path(manifest_path)),
        "split_path": str(Path(split_path)),
        "chunk_horizon": int(chunk_horizon),
        "frame_stride": int(frame_stride),
        "history_stride": int(history_stride),
        "task_aliases": list(task_aliases),
        "train_episodes_per_task": int(train_episodes_per_task),
        "val_episodes_per_task": int(val_episodes_per_task),
        "sample_count": int(len(data)),
        "sample_identity": identity.hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_encoder = encoder_name.replace("/", "_").replace(":", "_")
    return cache_dir / policy_kind / safe_encoder / f"{label}_{digest}.npz"


def _cache_clip_features(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    data: TemporalChunkData,
    *,
    device: torch.device,
    batch_size: int,
    label: str,
    cache_path: Path | None = None,
) -> FrozenCLIPFeatureData:
    if not hasattr(data, "prev_agent"):
        raise ValueError("frozen feature policy requires _attach_history before feature caching")
    batch_size = max(1, int(batch_size))
    expected_shape = (len(data), 4, int(model.feature_dim))
    features: np.ndarray | None = None
    if cache_path is not None and cache_path.exists():
        start_time = time.monotonic()
        with np.load(cache_path) as cached:
            loaded = np.asarray(cached["image_features"], dtype=np.float16)
        if tuple(loaded.shape) == expected_shape:
            features = loaded
            print(f"feature_cache {label}: loaded {cache_path} in {time.monotonic() - start_time:.1f}s", flush=True)
        else:
            print(f"feature_cache {label}: ignoring shape mismatch at {cache_path}: {loaded.shape} != {expected_shape}", flush=True)
    if features is None:
        features = np.empty(expected_shape, dtype=np.float16)
        model.eval()
        start_time = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                end = min(len(data), start + batch_size)
                idx = np.arange(start, end)
                images = np.concatenate(
                    [
                        data.prev_agent[idx],
                        data.prev_wrist[idx],
                        data.agent[idx],
                        data.wrist[idx],
                    ],
                    axis=0,
                )
                images_t = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                encoded = model.encode_images(images_t).detach().cpu().reshape(4, end - start, -1).transpose(0, 1)
                features[start:end] = encoded.numpy().astype(np.float16)
                if start == 0 or end == len(data) or (end // max(1, batch_size * 20)) != (start // max(1, batch_size * 20)):
                    print(f"feature_cache {label}: {end}/{len(data)}", flush=True)
        print(f"feature_cache {label}: done in {time.monotonic() - start_time:.1f}s", flush=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, image_features=features)
            print(f"feature_cache {label}: saved {cache_path}", flush=True)
    return FrozenCLIPFeatureData(
        image_features=features,
        prev_proprio=np.asarray(data.prev_proprio, dtype=np.float32),
        proprio=np.asarray(data.proprio, dtype=np.float32),
        actions=np.asarray(data.actions, dtype=np.float32),
        mask=np.asarray(data.mask, dtype=np.float32),
        task_id=np.asarray(data.task_id, dtype=np.int64),
        episode_idx=np.asarray(data.episode_idx, dtype=np.int32),
        frame_idx=np.asarray(data.frame_idx, dtype=np.int32),
    )


def _clip_feature_batch(data: FrozenCLIPFeatureData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image_features": torch.as_tensor(data.image_features[idx], dtype=torch.float32, device=device),
        "prev_proprio": torch.as_tensor(data.prev_proprio[idx], dtype=torch.float32, device=device),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data.actions[idx], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(data.mask[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


def _augment_clip_features(batch: dict[str, torch.Tensor], proprio_noise: float) -> dict[str, torch.Tensor]:
    if proprio_noise > 0:
        batch["prev_proprio"] = batch["prev_proprio"] + torch.randn_like(batch["prev_proprio"]) * proprio_noise
        batch["proprio"] = batch["proprio"] + torch.randn_like(batch["proprio"]) * proprio_noise
    return batch


def _augment_history(batch: dict[str, torch.Tensor], image_noise: float, proprio_noise: float) -> dict[str, torch.Tensor]:
    batch = _augment(batch, image_noise, proprio_noise)
    if image_noise > 0:
        scale = 255.0 * image_noise
        batch["prev_agent"] = (batch["prev_agent"] + torch.randn_like(batch["prev_agent"]) * scale).clamp(0.0, 255.0)
        batch["prev_wrist"] = (batch["prev_wrist"] + torch.randn_like(batch["prev_wrist"]) * scale).clamp(0.0, 255.0)
    if proprio_noise > 0:
        batch["prev_proprio"] = batch["prev_proprio"] + torch.randn_like(batch["prev_proprio"]) * proprio_noise
    return batch


def _sequence_flow_matching_loss(
    model: RoboCasaSequenceFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_aux_weight: float,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.encode_obs(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
    if flow_source == "bc":
        source = model.bc_action(context).detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    if bc_aux_weight > 0:
        pred_bc = model.bc_action(context)
        loss = loss + float(bc_aux_weight) * _masked_chunk_loss(
            pred_bc,
            actions,
            batch["mask"],
            chunk_decay=chunk_decay,
        )
    return loss


def _history_flow_matching_loss(
    model: RoboCasaHistoryFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_aux_weight: float,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.encode_obs(
        batch["prev_agent"],
        batch["prev_wrist"],
        batch["agent"],
        batch["wrist"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    if flow_source == "bc":
        source = model.bc_action(context).detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    if bc_aux_weight > 0:
        pred_bc = model.bc_action(context)
        loss = loss + float(bc_aux_weight) * _masked_chunk_loss(
            pred_bc,
            actions,
            batch["mask"],
            chunk_decay=chunk_decay,
        )
    return loss


def _history_act_flow_matching_loss(
    model: RoboCasaHistoryACTFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_weight: float,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.encode_obs(
        batch["prev_agent"],
        batch["prev_wrist"],
        batch["agent"],
        batch["wrist"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    pred_bc = model.bc_action(context)
    bc_loss = _masked_chunk_loss(
        pred_bc,
        actions,
        batch["mask"],
        chunk_decay=chunk_decay,
    )
    if flow_source == "bc":
        source = pred_bc.detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    flow_loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    return flow_loss + float(bc_weight) * bc_loss


def _frozen_clip_flow_matching_loss(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_weight: float,
    time_sampling: str,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.context_from_features(
        batch["image_features"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    pred_bc = model.bc_action(context)
    bc_loss = _masked_chunk_loss(pred_bc, actions, batch["mask"], chunk_decay=chunk_decay)
    if flow_source == "bc":
        source = pred_bc.detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = _sample_flow_time(actions.shape[0], actions.dtype, actions.device, time_sampling)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    flow_loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    return flow_loss + float(bc_weight) * bc_loss


def _mini_pi0_flow_matching_loss(
    model: RoboCasaMiniPi0Policy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    chunk_decay: float,
    time_sampling: str,
) -> torch.Tensor:
    actions = batch["actions"]
    obs_tokens = model.encode_obs_tokens(
        batch["prev_agent"],
        batch["prev_wrist"],
        batch["agent"],
        batch["wrist"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    source = torch.randn_like(actions) * sigma
    t = _sample_flow_time(actions.shape[0], actions.dtype, actions.device, time_sampling)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(obs_tokens, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    return (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)


def _sample_flow_time(
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    if mode == "beta_low_noise":
        return torch.rand((batch_size,), dtype=dtype, device=device).square().clamp(1e-4, 1.0 - 1e-4)
    return torch.rand((batch_size,), dtype=dtype, device=device)


def _eval_policy_loss(
    model: nn.Module,
    data: TemporalChunkData,
    device: torch.device,
    batch_size: int,
    *,
    policy_kind: str,
    flow_steps: int,
    flow_eval_start: str,
    flow_residual_scale: float,
) -> float:
    if policy_kind in {"history_flow", "history_act_flow", "mini_pi0", "mini_pi0_resnet"}:
        model.eval()
        total = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                idx = np.arange(start, min(len(data), start + batch_size))
                batch = _history_batch(data, idx, device)
                pred = model.sample_flow(
                    batch["prev_agent"],
                    batch["prev_wrist"],
                    batch["agent"],
                    batch["wrist"],
                    batch["prev_proprio"],
                    batch["proprio"],
                    batch["task_id"],
                    steps=flow_steps,
                    start=flow_eval_start,
                    residual_scale=flow_residual_scale,
                )
                per_step = (pred - batch["actions"]).square().mean(dim=-1)
                total = total + (per_step * batch["mask"]).sum()
                denom = denom + batch["mask"].sum()
        model.train()
        return float((total / denom.clamp_min(1.0)).detach().cpu())
    if policy_kind in {"history_act", "mini_pi0_act", "mini_pi0_act_resnet", "vit_act"}:
        model.eval()
        total = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                idx = np.arange(start, min(len(data), start + batch_size))
                batch = _history_batch(data, idx, device)
                pred = model(
                    batch["prev_agent"],
                    batch["prev_wrist"],
                    batch["agent"],
                    batch["wrist"],
                    batch["prev_proprio"],
                    batch["proprio"],
                    batch["task_id"],
                )
                per_step = (pred - batch["actions"]).square().mean(dim=-1)
                total = total + (per_step * batch["mask"]).sum()
                denom = denom + batch["mask"].sum()
        model.train()
        return float((total / denom.clamp_min(1.0)).detach().cpu())
    if policy_kind != "sequence_flow":
        return _eval_loss(
            model,
            data,
            device,
            batch_size=batch_size,
            policy_kind=policy_kind,
            flow_steps=flow_steps,
        )
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            pred = model.sample_flow(
                batch["agent"],
                batch["wrist"],
                batch["proprio"],
                batch["task_id"],
                steps=flow_steps,
                start=flow_eval_start,
            )
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _eval_clip_feature_loss(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    data: FrozenCLIPFeatureData,
    device: torch.device,
    batch_size: int,
    *,
    flow_steps: int,
    flow_eval_start: str,
    flow_residual_scale: float,
) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _clip_feature_batch(data, idx, device)
            context = model.context_from_features(
                batch["image_features"],
                batch["prev_proprio"],
                batch["proprio"],
                batch["task_id"],
            )
            pred = _sample_clip_flow_from_context(
                model,
                context,
                horizon=batch["actions"].shape[1],
                steps=flow_steps,
                start=flow_eval_start,
                residual_scale=flow_residual_scale,
            )
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _sample_clip_flow_from_context(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    context: torch.Tensor,
    *,
    horizon: int,
    steps: int,
    start: str,
    residual_scale: float,
) -> torch.Tensor:
    shape = (context.shape[0], int(horizon), int(model.action_dim))
    if start == "noise":
        action = torch.randn(shape, dtype=context.dtype, device=context.device)
    elif start == "zero":
        action = torch.zeros(shape, dtype=context.dtype, device=context.device)
    else:
        action = model.bc_action(context)
    steps = int(steps)
    if steps <= 0:
        return action
    dt = 1.0 / steps
    scale = float(residual_scale)
    for idx in range(steps):
        t = torch.full((context.shape[0],), (idx + 0.5) * dt, dtype=context.dtype, device=context.device)
        action = action + scale * dt * model.flow_velocity(context, action, t)
    return action


def _checkpoint_state_dict(model: nn.Module, policy_kind: str) -> dict[str, torch.Tensor]:
    if policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"} and hasattr(model, "head_state_dict"):
        state = model.head_state_dict()
    else:
        state = model.state_dict()
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _load_init_checkpoint(model: nn.Module, checkpoint: str, device: torch.device) -> dict:
    if not checkpoint:
        return {"loaded": 0, "skipped": 0, "path": ""}
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    source_state = payload.get("state_dict", payload)
    target_state = model.state_dict()
    compatible = {}
    for key, value in source_state.items():
        candidate_keys = [key]
        if key.startswith("context_blocks."):
            candidate_keys.append("obs_blocks." + key.removeprefix("context_blocks."))
        for candidate_key in candidate_keys:
            if candidate_key in target_state and tuple(target_state[candidate_key].shape) == tuple(value.shape):
                compatible[candidate_key] = value
                break
    missing_or_mismatch = len(target_state) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    return {
        "path": str(checkpoint),
        "loaded": int(len(compatible)),
        "skipped": int(missing_or_mismatch),
        "source_policy_type": str(payload.get("policy_type", "")) if isinstance(payload, dict) else "",
    }


def _freeze_non_flow(model: nn.Module) -> dict:
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("flow_"):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return {"frozen": int(frozen), "trainable": int(trainable)}


def _parameter_trainability(model: nn.Module) -> dict:
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"frozen": int(frozen), "trainable": int(trainable)}


if __name__ == "__main__":
    main()
