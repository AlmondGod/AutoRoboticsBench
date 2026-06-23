from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_visual_world_model.model import ImageVAE, SpatialImageAutoencoder
from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    TransitionData,
    load_transition_data,
    load_video_frames,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a standalone RoboCasa frame VAE.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_visual_world_model/frame_vae")
    parser.add_argument("--train-episodes-per-task", type=int, default=0)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--view", default="robot0_agentview_right")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--architecture", choices=("vae", "spatial"), default="vae")
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--encoder-pool-size", type=int, default=4)
    parser.add_argument("--decoder-width", type=int, default=1024)
    parser.add_argument("--decoder-depth", type=int, default=3)
    parser.add_argument("--spatial-latent-channels", type=int, default=128)
    parser.add_argument("--spatial-width", type=int, default=128)
    parser.add_argument("--spatial-depth", type=int, default=2)
    parser.add_argument("--spatial-downsample-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sample-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--grad-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=1e-7)
    parser.add_argument("--max-train-seconds", type=float, default=1200.0)
    parser.add_argument("--eval-batches", type=int, default=0, help="0 evaluates the full validation set.")
    parser.add_argument("--lpips-batches", type=int, default=16)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _device(str(args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        task_aliases=set(args.task_alias),
        frame_stride=int(args.frame_stride),
    )
    if len(train) == 0 or len(val) == 0:
        raise ValueError("need both train and val frames for VAE pretraining")

    print("precomputing_frame_targets", flush=True)
    train_rgb = _precompute_frames(train, summary, str(args.view), int(args.image_size))
    val_rgb = _precompute_frames(val, summary, str(args.view), int(args.image_size))

    vae = _build_model(args, device)
    opt = torch.optim.AdamW(vae.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: list[dict] = []
    best_val = float("inf")
    start_time = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        step += 1
        vae.train()
        idx = rng.integers(0, len(train_rgb), size=int(args.batch_size))
        batch = torch.as_tensor(train_rgb[idx], dtype=torch.float32, device=device)
        out = vae(batch, sample=bool(args.sample_latent))
        loss, metrics = _vae_loss(
            out["reconstruction"],
            batch,
            out["mu"],
            out["logvar"],
            l1_weight=float(args.l1_weight),
            mse_weight=float(args.mse_weight),
            grad_weight=float(args.grad_weight),
            kl_weight=float(args.kl_weight),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0:
            val_metrics = _eval_vae(
                vae,
                val_rgb,
                batch_size=int(args.batch_size),
                device=device,
                l1_weight=float(args.l1_weight),
                mse_weight=float(args.mse_weight),
                grad_weight=float(args.grad_weight),
                kl_weight=float(args.kl_weight),
                max_batches=int(args.eval_batches),
            )
            row = {
                "step": int(step),
                "elapsed_seconds": time.monotonic() - start_time,
                **{key: float(value.detach().cpu()) for key, value in metrics.items()},
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["val_loss"] < best_val:
                best_val = row["val_loss"]
                _save_checkpoint(out_dir / "frame_vae_best.pt", vae, args, summary, history, step)

    _save_checkpoint(out_dir / "frame_vae_last.pt", vae, args, summary, history, step)
    best = _load_vae(out_dir / "frame_vae_best.pt", device)
    final_metrics = _eval_vae(
        best,
        val_rgb,
        batch_size=int(args.batch_size),
        device=device,
        l1_weight=float(args.l1_weight),
        mse_weight=float(args.mse_weight),
        grad_weight=float(args.grad_weight),
        kl_weight=float(args.kl_weight),
        max_batches=0,
    )
    lpips_metrics = _eval_lpips(best, val_rgb, int(args.batch_size), device, int(args.lpips_batches))
    preview = _save_preview(out_dir / "frame_vae_recon_contact.png", best, val_rgb, int(args.preview_count), device)
    payload = {
        "task": "robocasa_frame_vae_pretraining",
        "checkpoint": str(out_dir / "frame_vae_best.pt"),
        "last_checkpoint": str(out_dir / "frame_vae_last.pt"),
        "train_frames": int(len(train_rgb)),
        "val_frames": int(len(val_rgb)),
        "architecture": str(args.architecture),
        "sample_latent": bool(args.sample_latent),
        "best_val_loss": float(best_val),
        "final_val": {key: float(value) for key, value in final_metrics.items()},
        "lpips_eval": lpips_metrics,
        "preview_png": str(preview),
        "history": history,
        "seconds": time.monotonic() - start_time,
        "summary": summary,
    }
    save_json(out_dir / "frame_vae_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if str(args.architecture) == "spatial":
        return SpatialImageAutoencoder(
            image_size=int(args.image_size),
            latent_channels=int(args.spatial_latent_channels),
            width=int(args.spatial_width),
            depth=int(args.spatial_depth),
            downsample_blocks=int(args.spatial_downsample_blocks),
            dropout=float(args.dropout),
        ).to(device)
    return ImageVAE(
        image_size=int(args.image_size),
        latent_dim=int(args.latent_dim),
        width=max(128, int(args.decoder_width) // 2),
        decoder_width=int(args.decoder_width),
        decoder_depth=int(args.decoder_depth),
        decoder_type="conv",
        encoder_pool_size=int(args.encoder_pool_size),
        dropout=float(args.dropout),
    ).to(device)


def _vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    l1_weight: float,
    mse_weight: float,
    grad_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(recon, target)
    mse = F.mse_loss(recon, target)
    grad = _gradient_loss(recon, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    loss = float(l1_weight) * l1 + float(mse_weight) * mse + float(grad_weight) * grad + float(kl_weight) * kl
    return loss, {"loss": loss.detach(), "l1": l1.detach(), "mse": mse.detach(), "grad": grad.detach(), "kl": kl.detach()}


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


@torch.no_grad()
def _eval_vae(
    vae: torch.nn.Module,
    rgb: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    l1_weight: float,
    mse_weight: float,
    grad_weight: float,
    kl_weight: float,
    max_batches: int,
) -> dict[str, float]:
    vae.eval()
    sums = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "grad": 0.0, "kl": 0.0}
    count = 0
    batches = 0
    for start in range(0, len(rgb), int(batch_size)):
        if int(max_batches) > 0 and batches >= int(max_batches):
            break
        batch = torch.as_tensor(rgb[start : start + int(batch_size)], dtype=torch.float32, device=device)
        out = vae(batch, sample=False)
        _, metrics = _vae_loss(
            out["reconstruction"],
            batch,
            out["mu"],
            out["logvar"],
            l1_weight=l1_weight,
            mse_weight=mse_weight,
            grad_weight=grad_weight,
            kl_weight=kl_weight,
        )
        n = int(batch.shape[0])
        for key in sums:
            sums[key] += float(metrics[key].detach().cpu()) * n
        count += n
        batches += 1
    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["psnr"] = float(-10.0 * math.log10(max(metrics["mse"], 1e-12)))
    return metrics


@torch.no_grad()
def _eval_lpips(vae: torch.nn.Module, rgb: np.ndarray, batch_size: int, device: torch.device, max_batches: int) -> dict:
    try:
        import lpips  # type: ignore
    except ModuleNotFoundError:
        return {"enabled": False, "reason": "lpips not installed"}
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    total = 0.0
    count = 0
    batches = 0
    for start in range(0, len(rgb), int(batch_size)):
        if int(max_batches) > 0 and batches >= int(max_batches):
            break
        batch = torch.as_tensor(rgb[start : start + int(batch_size)], dtype=torch.float32, device=device)
        recon = vae(batch, sample=False)["reconstruction"]
        total += float(model(_lpips_input(recon), _lpips_input(batch)).reshape(-1).sum().detach().cpu())
        count += int(batch.shape[0])
        batches += 1
    return {"enabled": True, "samples": int(count), "lpips": total / max(1, count)}


def _lpips_input(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0.0, 1.0) * 2.0 - 1.0


def _precompute_frames(data: TransitionData, summary: list[dict], view: str, image_size: int) -> np.ndarray:
    rgb = np.empty((len(data), 3, int(image_size), int(image_size)), dtype=np.float32)
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
            rgb[index] = _preprocess_frame(frames[frame_idx], image_size)
    return rgb


def _preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        resized = np.asarray(Image.fromarray(frame).resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


@torch.no_grad()
def _save_preview(path: Path, vae: torch.nn.Module, rgb: np.ndarray, count: int, device: torch.device) -> Path:
    vae.eval()
    count = max(1, min(int(count), len(rgb)))
    indices = np.linspace(0, len(rgb) - 1, count, dtype=np.int64)
    batch = torch.as_tensor(rgb[indices], dtype=torch.float32, device=device)
    recon = vae(batch, sample=False)["reconstruction"].detach().cpu().numpy()
    cell = 160
    label_h = 22
    canvas = Image.new("RGB", (cell * 2, count * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        y = row * (cell + label_h)
        draw.text((6, y + 4), f"val frame {int(index)} | input", fill=(0, 0, 0))
        draw.text((cell + 6, y + 4), "reconstruction", fill=(0, 0, 0))
        canvas.paste(_to_pil(rgb[index], cell), (0, y + label_h))
        canvas.paste(_to_pil(recon[row], cell), (cell, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _to_pil(chw: np.ndarray, size: int) -> Image.Image:
    hwc = np.transpose(np.clip(chw, 0.0, 1.0), (1, 2, 0))
    image = Image.fromarray((hwc * 255.0).round().astype(np.uint8))
    return image.resize((int(size), int(size)), Image.Resampling.BILINEAR)


def _save_checkpoint(path: Path, vae: torch.nn.Module, args: argparse.Namespace, summary: list[dict], history: list[dict], step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_vae": vae.state_dict(),
            "config": {
                "architecture": str(args.architecture),
                "image_size": int(args.image_size),
                "visual_latent_dim": int(args.latent_dim),
                "visual_encoder_pool_size": int(args.encoder_pool_size),
                "visual_decoder_width": int(args.decoder_width),
                "visual_decoder_depth": int(args.decoder_depth),
                "visual_decoder_type": "conv",
                "spatial_latent_channels": int(args.spatial_latent_channels),
                "spatial_width": int(args.spatial_width),
                "spatial_depth": int(args.spatial_depth),
                "spatial_downsample_blocks": int(args.spatial_downsample_blocks),
                "dropout": float(args.dropout),
                "sample_latent": bool(args.sample_latent),
            },
            "summary": summary,
            "history": history,
            "step": int(step),
            "task": "robocasa_frame_vae_pretraining",
        },
        path,
    )


def _load_vae(path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if str(cfg.get("architecture", "vae")) == "spatial":
        vae = SpatialImageAutoencoder(
            image_size=int(cfg["image_size"]),
            latent_channels=int(cfg.get("spatial_latent_channels", 128)),
            width=int(cfg.get("spatial_width", 128)),
            depth=int(cfg.get("spatial_depth", 2)),
            downsample_blocks=int(cfg.get("spatial_downsample_blocks", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
    else:
        vae = ImageVAE(
            image_size=int(cfg["image_size"]),
            latent_dim=int(cfg["visual_latent_dim"]),
            width=max(128, int(cfg["visual_decoder_width"]) // 2),
            decoder_width=int(cfg["visual_decoder_width"]),
            decoder_depth=int(cfg["visual_decoder_depth"]),
            decoder_type=str(cfg.get("visual_decoder_type", "conv")),
            encoder_pool_size=int(cfg.get("visual_encoder_pool_size", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
    vae.load_state_dict(payload["image_vae"])
    vae.eval()
    return vae


if __name__ == "__main__":
    main()
