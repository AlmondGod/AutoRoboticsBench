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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_POOL,
    TransitionData,
    load_video_frame,
    load_video_frames,
    load_transition_data,
    load_video_only_pool,
    make_stats,
    normalize_data,
    save_json,
    summarize_video_only_pool,
)
from tasks.robocasa_world_model.inverse_dynamics import load_inverse_dynamics
from tasks.robocasa_world_model.model import RoboCasaWorldModel
from tasks.robocasa_world_model.video_repr import load_video_encoder
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
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
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
    progress = batch["progress"]
    if progress.ndim == 1:
        progress = progress[:, None]
    h = torch.cat([z, batch["action"], model.task(batch["task_id"].long()), progress.float()], dim=-1)
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
