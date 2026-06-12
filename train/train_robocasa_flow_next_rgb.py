from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.robocasa_flow_rgb import RoboCasaNextRGBFlow
from models.robocasa_tiny_evaluator import RoboCasaVAEWorldModel
from train.common import device_from_arg
from train.train_robocasa_tiny_evaluator import _batch, _filtered_manifest, _load_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa/world_evaluator/flow_next_rgb")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--condition-on-robocasa-task-index", action="store_true")
    parser.add_argument("--train-demos-per-task", type=int, default=80)
    parser.add_argument("--val-episode-id", action="append", type=int, default=[])
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--success-window", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--cond-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--flow-source", choices=["noise", "vae"], default="noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    if vae_ckpt.get("model_type") != "robocasa_vae_world_model":
        raise ValueError("expected robocasa_vae_world_model checkpoint")
    vae = RoboCasaVAEWorldModel(
        proprio_dim=int(vae_ckpt["proprio_dim"]),
        action_dim=int(vae_ckpt["action_dim"]),
        task_count=int(vae_ckpt["task_count"]),
        latent_dim=int(vae_ckpt["latent_dim"]),
        width=int(vae_ckpt.get("width", 512)),
        dropout=float(vae_ckpt.get("dropout", 0.0)),
    ).to(device)
    vae.load_state_dict(vae_ckpt["state_dict"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

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
    _apply_checkpoint_norm(train, vae_ckpt)
    _apply_checkpoint_norm(val, vae_ckpt)

    flow = RoboCasaNextRGBFlow(
        latent_dim=int(vae_ckpt["latent_dim"]),
        action_dim=int(vae_ckpt["action_dim"]),
        task_count=int(vae_ckpt["task_count"]),
        hidden=int(args.hidden),
        cond_dim=int(args.cond_dim),
    ).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val = math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        loss, parts = _flow_loss(vae, flow, batch, str(args.flow_source))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        record = {"step": step, **parts}
        history.append(record)
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_metrics = _eval(vae, flow, val, device, int(args.batch_size), int(args.sample_steps), str(args.flow_source))
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["sample_mse"] < best_val:
                best_val = float(val_metrics["sample_mse"])
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in flow.state_dict().items()}
            print(
                f"step={step} loss={parts['loss']:.6f} val_flow={val_metrics['flow_loss']:.6f} "
                f"val_sample_psnr={val_metrics['sample_psnr']:.2f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": flow.state_dict(),
        "model_type": "robocasa_next_rgb_flow",
        "vae_checkpoint": str(Path(args.vae_checkpoint)),
        "latent_dim": int(vae_ckpt["latent_dim"]),
        "action_dim": int(vae_ckpt["action_dim"]),
        "task_count": int(vae_ckpt["task_count"]),
        "hidden": int(args.hidden),
        "cond_dim": int(args.cond_dim),
        "sample_steps": int(args.sample_steps),
        "flow_source": str(args.flow_source),
        "manifest": str(Path(args.manifest)),
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "condition_on_robocasa_task_index": bool(args.condition_on_robocasa_task_index),
    }
    torch.save(checkpoint, out_dir / "next_rgb_flow.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_sample_mse"] = float(best_val)
    torch.save(best_checkpoint, out_dir / "next_rgb_flow_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "next_rgb_flow.pt"),
        "best_checkpoint": str(out_dir / "next_rgb_flow_best.pt"),
        "best_step": int(best_step),
        "best_sample_mse": float(best_val),
        "val": _eval(vae, flow, val, device, int(args.batch_size), int(args.sample_steps), str(args.flow_source)),
        "train_samples": len(train),
        "val_samples": len(val),
        "train_demos_per_task": int(args.train_demos_per_task),
        "val_episode_ids": [int(ep) for ep in args.val_episode_id],
        "robocasa_task_indices": [int(idx) for idx in args.robocasa_task_index],
        "frame_stride": int(args.frame_stride),
        "hidden": int(args.hidden),
        "cond_dim": int(args.cond_dim),
        "sample_steps": int(args.sample_steps),
        "flow_source": str(args.flow_source),
        "train_seconds": time.time() - started,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _apply_checkpoint_norm(data, checkpoint: dict) -> None:
    proprio_mean = np.asarray(checkpoint["proprio_mean"], dtype=np.float32)
    proprio_std = np.asarray(checkpoint["proprio_std"], dtype=np.float32)
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    data.proprio = ((data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    data.next_proprio = ((data.next_proprio - proprio_mean) / proprio_std).astype(np.float32)
    data.action = ((data.action - action_mean) / action_std).astype(np.float32)


def _flow_loss(
    vae: RoboCasaVAEWorldModel,
    flow: RoboCasaNextRGBFlow,
    batch: dict[str, torch.Tensor],
    flow_source: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = _next_target(batch)
    with torch.no_grad():
        latent = vae.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        next_latent, _ = vae.step(latent, batch["action"], batch["task_id"])
        prior = vae.decode(next_latent, batch["task_id"])
    noise = torch.rand_like(target) if flow_source == "noise" else prior
    t = torch.rand((target.shape[0],), device=target.device, dtype=target.dtype)
    x_t = noise.lerp(target, t[:, None, None, None])
    velocity = target - noise
    pred = flow(x_t, t, next_latent, batch["action"], batch["task_id"])
    loss = F.mse_loss(pred, velocity)
    return loss, {"loss": float(loss.detach().cpu())}


def _eval(
    vae: RoboCasaVAEWorldModel,
    flow: RoboCasaNextRGBFlow,
    data,
    device: torch.device,
    batch_size: int,
    sample_steps: int,
    flow_source: str,
) -> dict[str, float]:
    flow.eval()
    total_flow = 0.0
    total_sample_mse = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            loss, _ = _flow_loss(vae, flow, batch, flow_source)
            target = _next_target(batch)
            latent = vae.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            next_latent, _ = vae.step(latent, batch["action"], batch["task_id"])
            prior = vae.decode(next_latent, batch["task_id"]) if flow_source == "vae" else None
            pred = flow.sample(latent=next_latent, action=batch["action"], task_id=batch["task_id"], steps=sample_steps, noise=prior)
            sample_mse = F.mse_loss(pred, target)
            n = len(idx)
            total_flow += float(loss.detach().cpu()) * n
            total_sample_mse += float(sample_mse.detach().cpu()) * n
            count += n
    flow.train()
    mse = total_sample_mse / max(1, count)
    return {
        "flow_loss": total_flow / max(1, count),
        "sample_mse": mse,
        "sample_psnr": -10.0 * math.log10(max(mse, 1e-12)),
    }


def _next_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    agent = batch["next_agent"] / 255.0 if batch["next_agent"].max() > 1.5 else batch["next_agent"]
    wrist = batch["next_wrist"] / 255.0 if batch["next_wrist"].max() > 1.5 else batch["next_wrist"]
    return torch.cat([agent, wrist], dim=1).clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
