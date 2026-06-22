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

from tasks.robocasa_visual_world_model.model import VisualRoboCasaWorldModel
from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    TransitionData,
    load_transition_data,
    load_video_frames,
    make_stats,
    normalize_data,
    save_json,
)
from tasks.robocasa_world_model.inverse_dynamics import load_inverse_dynamics
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
    parser = argparse.ArgumentParser(description="Train a visually grounded RoboCasa world model.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_visual_world_model/base")
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--view", default="robot0_agentview_right")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--visual-latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.25)
    parser.add_argument("--success-weight", type=float, default=0.25)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--image-vae-weight", type=float, default=0.25)
    parser.add_argument("--visual-flow-weight", type=float, default=0.5)
    parser.add_argument("--state-flow-weight", type=float, default=0.25)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--visual-kl-weight", type=float, default=1e-5)
    parser.add_argument("--inverse-dynamics-checkpoint", default="")
    parser.add_argument("--inverse-align-weight", type=float, default=0.0)
    parser.add_argument("--inverse-align-image-size", type=int, default=64)
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
        raise ValueError("need both train and val transitions for visual world-model training")
    stats = make_stats(train_raw)
    train = normalize_data(train_raw, stats)
    val = normalize_data(val_raw, stats)
    task_count = int(max(train.task_id.max(initial=0), val.task_id.max(initial=0)) + 1)
    model = VisualRoboCasaWorldModel(
        state_dim=int(train.state.shape[-1]),
        action_dim=int(train.action.shape[-1]),
        task_count=task_count,
        image_size=int(args.image_size),
        width=int(args.width),
        depth=int(args.depth),
        task_dim=int(args.task_dim),
        latent_dim=int(args.latent_dim),
        visual_latent_dim=int(args.visual_latent_dim),
        dropout=float(args.dropout),
    ).to(device)

    print("precomputing_rgb_targets", flush=True)
    train_rgb, train_next_rgb = _precompute_rgb_targets(train, summary, str(args.view), int(args.image_size))
    val_rgb, val_next_rgb = _precompute_rgb_targets(val, summary, str(args.view), int(args.image_size))
    inverse_align, inverse_head = _build_inverse_alignment(args, device, width=int(args.width))
    if inverse_align is not None and float(inverse_align["weight"]) > 0:
        print("precomputing_inverse_targets", flush=True)
        inverse_align["train_targets"] = _precompute_inverse_targets(train, summary, inverse_align, device)

    params = list(model.parameters())
    if inverse_head is not None:
        params.extend(inverse_head.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
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
        batch = _batch(train, train_rgb, train_next_rgb, idx, device)
        loss, metrics = model.loss(
            batch,
            state_weight=float(args.state_weight),
            progress_weight=float(args.progress_weight),
            success_weight=float(args.success_weight),
            visual_weight=float(args.visual_weight),
            image_vae_weight=float(args.image_vae_weight),
            visual_flow_weight=float(args.visual_flow_weight),
            state_flow_weight=float(args.state_flow_weight),
            kl_weight=float(args.kl_weight),
            visual_kl_weight=float(args.visual_kl_weight),
        )
        if inverse_align is not None and float(inverse_align["weight"]) > 0:
            inverse_loss = _inverse_alignment_loss(model, batch, idx, inverse_align, device)
            loss = loss + float(inverse_align["weight"]) * inverse_loss
            metrics["inverse_align_loss"] = inverse_loss.detach()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            val_metrics = _eval(model, val, val_rgb, val_next_rgb, int(args.batch_size), device)
            row = {
                "step": int(step),
                "elapsed_seconds": time.monotonic() - start_time,
                **{key: float(value.detach().cpu()) for key, value in metrics.items()},
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["val_visual_score_loss"] < best_val:
                best_val = row["val_visual_score_loss"]
                _save_checkpoint(out_dir / "policy_best.pt", model, stats, args, summary, history, step, inverse_align)

    final_metrics = _eval(model, val, val_rgb, val_next_rgb, int(args.batch_size), device)
    _save_checkpoint(out_dir / "policy_last.pt", model, stats, args, summary, history, len(history), inverse_align)
    payload = {
        "task": "robocasa_visual_world_model",
        "checkpoint": str(out_dir / "policy_best.pt"),
        "last_checkpoint": str(out_dir / "policy_last.pt"),
        "train_transitions": len(train),
        "val_transitions": len(val),
        "image_size": int(args.image_size),
        "view": str(args.view),
        "inverse_alignment": _inverse_alignment_summary(args, inverse_align),
        "flow_matching": _flow_matching_summary(args),
        "summary": summary,
        "final_val": final_metrics,
        "best_val_visual_score_loss": best_val,
        "history": history,
        "seconds": time.monotonic() - start_time,
    }
    save_json(out_dir / "train_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _batch(
    data: TransitionData,
    rgb: np.ndarray,
    next_rgb: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "state": torch.as_tensor(data.state[idx], dtype=torch.float32, device=device),
        "action": torch.as_tensor(data.action[idx], dtype=torch.float32, device=device),
        "next_state": torch.as_tensor(data.next_state[idx], dtype=torch.float32, device=device),
        "progress": torch.as_tensor(data.progress[idx], dtype=torch.float32, device=device),
        "next_progress": torch.as_tensor(data.next_progress[idx], dtype=torch.float32, device=device),
        "success": torch.as_tensor(data.success[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
        "rgb": torch.as_tensor(rgb[idx], dtype=torch.float32, device=device),
        "next_rgb": torch.as_tensor(next_rgb[idx], dtype=torch.float32, device=device),
    }


def _precompute_rgb_targets(
    data: TransitionData,
    summary: list[dict],
    view: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.empty((len(data), 3, int(image_size), int(image_size)), dtype=np.float32)
    next_rgb = np.empty_like(rgb)
    dataset_by_task = {int(row["task_id"]): Path(row["dataset_path"]) for row in summary}
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        groups.setdefault((int(task_id), int(episode_id)), []).append(index)
    for (task_id, episode_id), indices in sorted(groups.items()):
        root = dataset_by_task[int(task_id)]
        video = root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{int(episode_id):06d}.mp4"
        frames = load_video_frames(video)
        for index in indices:
            frame_idx = int(np.clip(data.frame_idx[index], 0, max(0, len(frames) - 1)))
            next_idx = min(frame_idx + 1, max(0, len(frames) - 1))
            rgb[index] = _preprocess_frame(frames[frame_idx], image_size)
            next_rgb[index] = _preprocess_frame(frames[next_idx], image_size)
    return rgb, next_rgb


def _preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((int(image_size), int(image_size))))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


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
            "view": str(args.view),
            "image_size": int(args.inverse_align_image_size),
            "checkpoint": str(args.inverse_dynamics_checkpoint),
        },
        head,
    )


@torch.no_grad()
def _precompute_inverse_targets(
    data: TransitionData,
    summary: list[dict],
    inverse_align: dict,
    device: torch.device,
) -> torch.Tensor:
    inverse_model = inverse_align["model"]
    inverse_model.eval()
    targets = torch.empty((len(data), int(inverse_align["feature_dim"])), dtype=torch.float32, device=device)
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


def _inverse_alignment_loss(
    model: VisualRoboCasaWorldModel,
    batch: dict[str, torch.Tensor],
    idx: np.ndarray,
    inverse_align: dict,
    device: torch.device,
) -> torch.Tensor:
    target = inverse_align["train_targets"][torch.as_tensor(idx, dtype=torch.long, device=device)]
    hidden, _, _, _ = model.transition_hidden(
        batch["state"],
        batch["action"],
        batch["task_id"],
        batch["progress"],
        sample_latent=False,
    )
    pred = F.normalize(inverse_align["head"](hidden), dim=-1)
    return F.mse_loss(pred, target)


@torch.no_grad()
def _eval(
    model: VisualRoboCasaWorldModel,
    data: TransitionData,
    rgb: np.ndarray,
    next_rgb: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {
        "state_mse": 0.0,
        "progress_mse": 0.0,
        "success_bce": 0.0,
        "rgb_mse": 0.0,
        "visual_score_loss": 0.0,
    }
    count = 0
    for start in range(0, len(data), batch_size):
        idx = np.arange(start, min(len(data), start + int(batch_size)))
        batch = _batch(data, rgb, next_rgb, idx, device)
        _, metrics = model.loss(batch)
        n = len(idx)
        for key in sums:
            if key == "visual_score_loss":
                value = metrics["rgb_mse"] + 0.25 * metrics["state_mse"]
            else:
                value = metrics[key]
            sums[key] += float(value.detach().cpu()) * n
        count += n
    return {key: value / max(1, count) for key, value in sums.items()}


def _save_checkpoint(
    path: Path,
    model: VisualRoboCasaWorldModel,
    stats: dict[str, np.ndarray],
    args: argparse.Namespace,
    summary: list[dict],
    history: list[dict],
    step: int,
    inverse_align: dict | None,
) -> None:
    cfg = {
        "state_dim": int(model.state_dim),
        "action_dim": int(model.action_dim),
        "task_count": int(model.task_count),
        "image_size": int(model.image_size),
        "width": int(args.width),
        "depth": int(args.depth),
        "task_dim": int(args.task_dim),
        "latent_dim": int(args.latent_dim),
        "visual_latent_dim": int(args.visual_latent_dim),
        "dropout": float(args.dropout),
        "view": str(args.view),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "stats": stats,
            "summary": summary,
            "history": history,
            "step": int(step),
            "inverse_alignment": _inverse_alignment_summary(args, inverse_align),
            "flow_matching": _flow_matching_summary(args),
            "task": "robocasa_visual_world_model",
        },
        path,
    )


def _inverse_alignment_summary(args: argparse.Namespace, inverse_align: dict | None) -> dict:
    return {
        "enabled": inverse_align is not None and float(args.inverse_align_weight) > 0,
        "checkpoint": str(args.inverse_dynamics_checkpoint),
        "weight": float(args.inverse_align_weight),
        "view": str(args.view),
        "image_size": int(args.inverse_align_image_size),
        "target": "frozen_inverse_dynamics_pair_encoder",
    }


def _flow_matching_summary(args: argparse.Namespace) -> dict:
    return {
        "image_vae_enabled": True,
        "visual_latent_dim": int(args.visual_latent_dim),
        "image_vae_weight": float(args.image_vae_weight),
        "visual_flow_weight": float(args.visual_flow_weight),
        "state_flow_weight": float(args.state_flow_weight),
        "visual_kl_weight": float(args.visual_kl_weight),
        "targets": ["next_visual_latent", "state_delta"],
    }


if __name__ == "__main__":
    main()
