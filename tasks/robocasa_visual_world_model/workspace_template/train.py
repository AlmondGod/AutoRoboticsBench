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

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

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


def _load_pretrained_autoencoder_payload(path: str) -> dict | None:
    if not path:
        return None
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "image_vae" not in payload or "config" not in payload:
        raise ValueError(f"{path} is not a frame autoencoder checkpoint")
    return payload


def _apply_pretrained_autoencoder_config(args: argparse.Namespace, payload: dict) -> None:
    cfg = payload["config"]
    args.visual_architecture = str(cfg.get("architecture", args.visual_architecture))
    args.image_size = int(cfg.get("image_size", args.image_size))
    args.visual_latent_dim = int(cfg.get("visual_latent_dim", args.visual_latent_dim))
    args.visual_encoder_pool_size = int(cfg.get("visual_encoder_pool_size", args.visual_encoder_pool_size))
    args.visual_decoder_width = int(cfg.get("visual_decoder_width", args.visual_decoder_width))
    args.visual_decoder_depth = int(cfg.get("visual_decoder_depth", args.visual_decoder_depth))
    args.spatial_latent_channels = int(cfg.get("spatial_latent_channels", args.spatial_latent_channels))
    args.spatial_width = int(cfg.get("spatial_width", args.spatial_width))
    args.spatial_depth = int(cfg.get("spatial_depth", args.spatial_depth))
    args.spatial_downsample_blocks = int(cfg.get("spatial_downsample_blocks", args.spatial_downsample_blocks))
    if args.visual_architecture == "spatial":
        spatial_hw = int(args.image_size) // (2 ** int(args.spatial_downsample_blocks))
        args.visual_latent_dim = int(args.spatial_latent_channels) * spatial_hw * spatial_hw



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a visually grounded RoboCasa world model.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_visual_world_model/base")
    parser.add_argument("--train-episodes-per-task", type=int, default=0)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--view", default="robot0_agentview_right")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-train-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--visual-latent-dim", type=int, default=128)
    parser.add_argument("--visual-encoder-pool-size", type=int, default=1)
    parser.add_argument("--visual-decoder-width", type=int, default=512)
    parser.add_argument("--visual-decoder-depth", type=int, default=3)
    parser.add_argument("--visual-architecture", choices=["vae", "spatial"], default="vae")
    parser.add_argument("--spatial-latent-channels", type=int, default=128)
    parser.add_argument("--spatial-width", type=int, default=128)
    parser.add_argument("--spatial-depth", type=int, default=2)
    parser.add_argument("--spatial-downsample-blocks", type=int, default=2)
    parser.add_argument("--spatial-dynamics-type", choices=["mlp", "conv"], default="conv")
    parser.add_argument("--spatial-dynamics-depth", type=int, default=4)
    parser.add_argument("--spatial-dynamics-hidden-channels", type=int, default=0)
    parser.add_argument("--pretrained-image-autoencoder", default="")
    parser.add_argument("--freeze-image-autoencoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual-delta-prediction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--visual-lr-scale", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.25)
    parser.add_argument("--success-weight", type=float, default=0.25)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--image-vae-weight", type=float, default=0.25)
    parser.add_argument("--visual-latent-weight", type=float, default=0.5)
    parser.add_argument("--visual-l1-weight", type=float, default=0.25)
    parser.add_argument("--visual-grad-weight", type=float, default=0.10)
    parser.add_argument("--image-vae-l1-weight", type=float, default=1.0)
    parser.add_argument("--image-vae-mse-weight", type=float, default=0.25)
    parser.add_argument("--image-vae-grad-weight", type=float, default=0.25)
    parser.add_argument("--image-augment", type=float, default=0.15)
    parser.add_argument("--rollout-horizon", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--rollout-visual-weight", type=float, default=0.25)
    parser.add_argument("--rollout-state-weight", type=float, default=0.10)
    parser.add_argument("--rollout-progress-weight", type=float, default=0.05)
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
    if float(args.max_train_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = device_from_arg(str(args.device))
    autoencoder_payload = _load_pretrained_autoencoder_payload(str(args.pretrained_image_autoencoder))
    if autoencoder_payload is not None:
        _apply_pretrained_autoencoder_config(args, autoencoder_payload)
    if args.freeze_image_autoencoder is None:
        args.freeze_image_autoencoder = autoencoder_payload is not None
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
        visual_encoder_pool_size=int(args.visual_encoder_pool_size),
        visual_decoder_width=int(args.visual_decoder_width) if int(args.visual_decoder_width) > 0 else None,
        visual_decoder_depth=int(args.visual_decoder_depth),
        visual_decoder_type="conv",
        visual_architecture=str(args.visual_architecture),
        spatial_latent_channels=int(args.spatial_latent_channels),
        spatial_width=int(args.spatial_width),
        spatial_depth=int(args.spatial_depth),
        spatial_downsample_blocks=int(args.spatial_downsample_blocks),
        spatial_dynamics_type=str(args.spatial_dynamics_type),
        spatial_dynamics_depth=int(args.spatial_dynamics_depth),
        spatial_dynamics_hidden_channels=int(args.spatial_dynamics_hidden_channels),
        current_rgb_conditioned=True,
        visual_delta_prediction=bool(args.visual_delta_prediction),
        dropout=float(args.dropout),
    ).to(device)
    if autoencoder_payload is not None:
        model.image_vae.load_state_dict(autoencoder_payload["image_vae"])
        print(
            json.dumps(
                {
                    "loaded_pretrained_image_autoencoder": str(args.pretrained_image_autoencoder),
                    "architecture": str(args.visual_architecture),
                    "frozen": bool(args.freeze_image_autoencoder),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if bool(args.freeze_image_autoencoder):
        for param in model.image_vae.parameters():
            param.requires_grad_(False)
        model.image_vae.eval()

    print("precomputing_rgb_targets", flush=True)
    train_rgb, train_next_rgb = _precompute_rgb_targets(train, summary, str(args.view), int(args.image_size))
    val_rgb, val_next_rgb = _precompute_rgb_targets(val, summary, str(args.view), int(args.image_size))
    rollout_starts = _sequence_starts(train, horizon=int(args.rollout_horizon), frame_stride=int(args.frame_stride))
    inverse_align, inverse_head = _build_inverse_alignment(args, device, width=int(args.width))
    if inverse_align is not None and float(inverse_align["weight"]) > 0:
        print("precomputing_inverse_targets", flush=True)
        inverse_align["train_targets"] = _precompute_inverse_targets(train, summary, inverse_align, device)

    params = _optimizer_param_groups(model, lr=float(args.lr), visual_lr_scale=float(args.visual_lr_scale))
    trainable_params = [param for group in params for param in group["params"]]
    if inverse_head is not None:
        inverse_params = list(inverse_head.parameters())
        params.append({"params": inverse_params, "lr": float(args.lr)})
        trainable_params.extend(inverse_params)
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
        if bool(args.freeze_image_autoencoder):
            model.image_vae.eval()
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, train_rgb, train_next_rgb, idx, device)
        if float(args.image_augment) > 0:
            batch = _augment_batch_images(batch, strength=float(args.image_augment))
        loss, metrics = model.loss(
            batch,
            state_weight=float(args.state_weight),
            progress_weight=float(args.progress_weight),
            success_weight=float(args.success_weight),
            visual_weight=float(args.visual_weight),
            image_vae_weight=float(args.image_vae_weight),
            visual_latent_weight=float(args.visual_latent_weight),
            visual_l1_weight=float(args.visual_l1_weight),
            visual_grad_weight=float(args.visual_grad_weight),
            image_vae_l1_weight=float(args.image_vae_l1_weight),
            image_vae_mse_weight=float(args.image_vae_mse_weight),
            image_vae_grad_weight=float(args.image_vae_grad_weight),
            kl_weight=float(args.kl_weight),
            visual_kl_weight=float(args.visual_kl_weight),
        )
        if inverse_align is not None and float(inverse_align["weight"]) > 0:
            inverse_loss = _inverse_alignment_loss(model, batch, idx, inverse_align, device)
            loss = loss + float(inverse_align["weight"]) * inverse_loss
            metrics["inverse_align_loss"] = inverse_loss.detach()
        if int(args.rollout_horizon) > 1 and float(args.rollout_visual_weight) > 0 and len(rollout_starts) > 0:
            rollout_idx = rng.choice(rollout_starts, size=min(int(args.rollout_batch_size), len(rollout_starts)), replace=len(rollout_starts) < int(args.rollout_batch_size))
            rollout_batch = _sequence_batch(train, train_rgb, train_next_rgb, rollout_idx, int(args.rollout_horizon), device)
            rollout_loss, rollout_metrics = _rollout_loss(
                model,
                rollout_batch,
                visual_weight=float(args.rollout_visual_weight),
                state_weight=float(args.rollout_state_weight),
                progress_weight=float(args.rollout_progress_weight),
            )
            loss = loss + rollout_loss
            metrics.update(rollout_metrics)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
        "visual_latent_prediction": _visual_latent_summary(args),
        "multi_step_rollout": _rollout_summary(args, rollout_starts),
        "image_augmentation": _augmentation_summary(args),
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


def _optimizer_param_groups(model: VisualRoboCasaWorldModel, *, lr: float, visual_lr_scale: float) -> list[dict]:
    visual_prefixes = ("image_vae.", "next_visual_latent.")
    visual_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(visual_prefixes):
            visual_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": float(lr)})
    if visual_params:
        groups.append({"params": visual_params, "lr": float(lr) * float(visual_lr_scale)})
    return groups


def _augment_batch_images(batch: dict[str, torch.Tensor], *, strength: float) -> dict[str, torch.Tensor]:
    strength = float(strength)
    if strength <= 0:
        return batch
    rgb = batch["rgb"]
    next_rgb = batch["next_rgb"]
    b = rgb.shape[0]
    device = rgb.device
    dtype = rgb.dtype
    brightness = 1.0 + (torch.rand((b, 1, 1, 1), dtype=dtype, device=device) * 2.0 - 1.0) * (0.35 * strength)
    contrast = 1.0 + (torch.rand((b, 1, 1, 1), dtype=dtype, device=device) * 2.0 - 1.0) * (0.35 * strength)
    noise = torch.randn_like(rgb) * (0.03 * strength)
    rgb_aug = _apply_color_aug(rgb, brightness, contrast) + noise
    next_aug = _apply_color_aug(next_rgb, brightness, contrast) + noise
    max_shift = int(round(max(rgb.shape[-1], rgb.shape[-2]) * 0.04 * strength))
    if max_shift > 0:
        shifts_y = torch.randint(-max_shift, max_shift + 1, (b,), device=device)
        shifts_x = torch.randint(-max_shift, max_shift + 1, (b,), device=device)
        rgb_aug = _shift_images(rgb_aug, shifts_y, shifts_x)
        next_aug = _shift_images(next_aug, shifts_y, shifts_x)
    out = dict(batch)
    out["rgb"] = rgb_aug.clamp(0.0, 1.0)
    out["next_rgb"] = next_aug.clamp(0.0, 1.0)
    return out


def _apply_color_aug(image: torch.Tensor, brightness: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    mean = image.mean(dim=(2, 3), keepdim=True)
    return (image - mean) * contrast + mean * brightness


def _shift_images(image: torch.Tensor, shifts_y: torch.Tensor, shifts_x: torch.Tensor) -> torch.Tensor:
    shifted = []
    for item, dy, dx in zip(image, shifts_y.tolist(), shifts_x.tolist()):
        shifted.append(torch.roll(item, shifts=(int(dy), int(dx)), dims=(-2, -1)))
    return torch.stack(shifted, dim=0)


def _sequence_starts(data: TransitionData, *, horizon: int, frame_stride: int) -> np.ndarray:
    horizon = int(horizon)
    if horizon <= 1 or len(data) < horizon:
        return np.zeros((0,), dtype=np.int64)
    starts = []
    stride = max(1, int(frame_stride))
    for start in range(0, len(data) - horizon + 1):
        end = start + horizon
        if not np.all(data.task_id[start:end] == data.task_id[start]):
            continue
        if not np.all(data.episode_id[start:end] == data.episode_id[start]):
            continue
        expected = data.frame_idx[start] + stride * np.arange(horizon, dtype=np.int32)
        if not np.array_equal(data.frame_idx[start:end], expected):
            continue
        starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def _sequence_batch(
    data: TransitionData,
    rgb: np.ndarray,
    next_rgb: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    offsets = np.arange(int(horizon), dtype=np.int64)
    indices = np.asarray(starts, dtype=np.int64)[:, None] + offsets[None, :]
    return {
        "state": torch.as_tensor(data.state[indices], dtype=torch.float32, device=device),
        "action": torch.as_tensor(data.action[indices], dtype=torch.float32, device=device),
        "next_state": torch.as_tensor(data.next_state[indices], dtype=torch.float32, device=device),
        "progress": torch.as_tensor(data.progress[indices], dtype=torch.float32, device=device),
        "next_progress": torch.as_tensor(data.next_progress[indices], dtype=torch.float32, device=device),
        "success": torch.as_tensor(data.success[indices], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[indices], dtype=torch.long, device=device),
        "rgb": torch.as_tensor(rgb[indices], dtype=torch.float32, device=device),
        "next_rgb": torch.as_tensor(next_rgb[indices], dtype=torch.float32, device=device),
    }


def _rollout_loss(
    model: VisualRoboCasaWorldModel,
    batch: dict[str, torch.Tensor],
    *,
    visual_weight: float,
    state_weight: float,
    progress_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state = batch["state"][:, 0]
    current_rgb = batch["rgb"][:, 0]
    rgb_losses = []
    state_losses = []
    progress_losses = []
    for offset in range(batch["action"].shape[1]):
        out = model(
            state,
            batch["action"][:, offset],
            task_id=batch["task_id"][:, offset],
            progress=batch["progress"][:, offset],
            current_rgb=current_rgb,
            sample_latent=False,
        )
        rgb_losses.append(F.mse_loss(out["next_rgb"], batch["next_rgb"][:, offset]))
        state_losses.append(F.mse_loss(out["next_state"], batch["next_state"][:, offset]))
        progress_losses.append(F.mse_loss(out["next_progress"], batch["next_progress"][:, offset]))
        state = out["next_state"]
        current_rgb = out["next_rgb"]
    rgb_loss = torch.stack(rgb_losses).mean()
    state_loss = torch.stack(state_losses).mean()
    progress_loss = torch.stack(progress_losses).mean()
    total = float(visual_weight) * rgb_loss + float(state_weight) * state_loss + float(progress_weight) * progress_loss
    return total, {
        "rollout_rgb_mse": rgb_loss.detach(),
        "rollout_state_mse": state_loss.detach(),
        "rollout_progress_mse": progress_loss.detach(),
        "rollout_loss": total.detach(),
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
        task_id=batch.get("task_id"),
        progress=batch.get("progress"),
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
        "visual_latent_dim": int(model.visual_latent_dim),
        "visual_encoder_pool_size": int(args.visual_encoder_pool_size),
        "visual_decoder_width": int(args.visual_decoder_width),
        "visual_decoder_depth": int(args.visual_decoder_depth),
        "visual_decoder_type": "conv",
        "visual_architecture": str(args.visual_architecture),
        "spatial_latent_channels": int(args.spatial_latent_channels),
        "spatial_width": int(args.spatial_width),
        "spatial_depth": int(args.spatial_depth),
        "spatial_downsample_blocks": int(args.spatial_downsample_blocks),
        "spatial_dynamics_type": str(args.spatial_dynamics_type),
        "spatial_dynamics_depth": int(args.spatial_dynamics_depth),
        "spatial_dynamics_hidden_channels": int(args.spatial_dynamics_hidden_channels),
        "pretrained_image_autoencoder": str(args.pretrained_image_autoencoder),
        "freeze_image_autoencoder": bool(args.freeze_image_autoencoder),
        "current_rgb_conditioned": True,
        "visual_delta_prediction": bool(args.visual_delta_prediction),
        "dropout": float(args.dropout),
        "visual_lr_scale": float(args.visual_lr_scale),
        "visual_l1_weight": float(args.visual_l1_weight),
        "visual_grad_weight": float(args.visual_grad_weight),
        "image_vae_l1_weight": float(args.image_vae_l1_weight),
        "image_vae_mse_weight": float(args.image_vae_mse_weight),
        "image_vae_grad_weight": float(args.image_vae_grad_weight),
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
            "visual_latent_prediction": _visual_latent_summary(args),
            "multi_step_rollout": _rollout_summary(args, []),
            "image_augmentation": _augmentation_summary(args),
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


def _visual_latent_summary(args: argparse.Namespace) -> dict:
    return {
        "image_vae_enabled": True,
        "architecture": str(args.visual_architecture),
        "visual_latent_dim": int(args.visual_latent_dim),
        "spatial_latent_channels": int(args.spatial_latent_channels),
        "spatial_downsample_blocks": int(args.spatial_downsample_blocks),
        "spatial_latent_hw": int(args.image_size) // (2 ** int(args.spatial_downsample_blocks)),
        "spatial_dynamics_type": str(args.spatial_dynamics_type),
        "spatial_dynamics_depth": int(args.spatial_dynamics_depth),
        "spatial_dynamics_hidden_channels": int(args.spatial_dynamics_hidden_channels),
        "pretrained_image_autoencoder": str(args.pretrained_image_autoencoder),
        "freeze_image_autoencoder": bool(args.freeze_image_autoencoder),
        "image_vae_weight": float(args.image_vae_weight),
        "visual_latent_weight": float(args.visual_latent_weight),
        "visual_kl_weight": float(args.visual_kl_weight),
        "visual_delta_prediction": bool(args.visual_delta_prediction),
        "target": "next_visual_delta" if bool(args.visual_delta_prediction) else "next_visual_latent",
        "prediction_rgb_loss": {
            "mse_weight": 1.0,
            "l1_weight": float(args.visual_l1_weight),
            "gradient_weight": float(args.visual_grad_weight),
        },
        "image_vae_reconstruction_loss": {
            "l1_weight": float(args.image_vae_l1_weight),
            "mse_weight": float(args.image_vae_mse_weight),
            "gradient_weight": float(args.image_vae_grad_weight),
        },
    }


def _augmentation_summary(args: argparse.Namespace) -> dict:
    return {
        "enabled": float(args.image_augment) > 0,
        "strength": float(args.image_augment),
        "transforms": ["paired_brightness", "paired_contrast", "paired_noise", "paired_translation_roll"],
    }


def _rollout_summary(args: argparse.Namespace, rollout_starts: np.ndarray | list) -> dict:
    return {
        "enabled": int(args.rollout_horizon) > 1 and float(args.rollout_visual_weight) > 0,
        "horizon": int(args.rollout_horizon),
        "batch_size": int(args.rollout_batch_size),
        "valid_sequence_starts": int(len(rollout_starts)),
        "visual_weight": float(args.rollout_visual_weight),
        "state_weight": float(args.rollout_state_weight),
        "progress_weight": float(args.rollout_progress_weight),
    }


if __name__ == "__main__":
    main()
