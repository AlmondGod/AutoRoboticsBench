from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_visual_world_model.model import VisualRoboCasaWorldModel


def load_world_model(checkpoint: str, device: str = "auto") -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    state = payload["model"]
    task_dim = int(cfg.get("task_dim", 0))
    has_task_condition = "dynamics.task.weight" in state and task_dim > 0
    trunk_in = int(state["dynamics.trunk.0.weight"].shape[1])
    latent_width = int(cfg["latent_dim"]) if int(cfg.get("latent_dim", 0)) > 0 else int(cfg["state_dim"])
    expected_without_progress = latent_width + int(cfg["action_dim"]) + (task_dim if has_task_condition else 0)
    has_progress_condition = trunk_in == expected_without_progress + 1
    model = VisualRoboCasaWorldModel(
        state_dim=int(cfg["state_dim"]),
        action_dim=int(cfg["action_dim"]),
        task_count=int(cfg["task_count"]),
        image_size=int(cfg["image_size"]),
        width=int(cfg["width"]),
        depth=int(cfg["depth"]),
        task_dim=task_dim,
        latent_dim=int(cfg["latent_dim"]),
        visual_latent_dim=int(cfg.get("visual_latent_dim", 64)),
        visual_encoder_pool_size=int(cfg.get("visual_encoder_pool_size", 1)),
        visual_decoder_width=int(cfg.get("visual_decoder_width", 0)) or None,
        visual_decoder_depth=int(cfg.get("visual_decoder_depth", 3)),
        visual_decoder_type=str(cfg.get("visual_decoder_type", "mlp")),
        visual_architecture=str(cfg.get("visual_architecture", "vae")),
        spatial_latent_channels=int(cfg.get("spatial_latent_channels", 128)),
        spatial_width=int(cfg.get("spatial_width", 128)),
        spatial_depth=int(cfg.get("spatial_depth", 2)),
        spatial_downsample_blocks=int(cfg.get("spatial_downsample_blocks", 2)),
        spatial_dynamics_type=str(cfg.get("spatial_dynamics_type", "mlp")),
        spatial_dynamics_depth=int(cfg.get("spatial_dynamics_depth", 4)),
        spatial_dynamics_hidden_channels=int(cfg.get("spatial_dynamics_hidden_channels", 0)),
        current_rgb_conditioned=bool(cfg.get("current_rgb_conditioned", False)),
        visual_delta_prediction=bool(cfg.get("visual_delta_prediction", False)),
        condition_on_task=has_task_condition,
        condition_on_progress=has_progress_condition,
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    stats = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in payload["stats"].items()}
    return {"model": model, "stats": stats, "config": cfg, "device": torch.device(device), "checkpoint": payload}


@torch.no_grad()
def predict_next(
    world_model: dict,
    state: np.ndarray,
    action: np.ndarray,
    task_id: int | None = None,
    progress: float | np.ndarray | None = None,
    current_rgb: np.ndarray | None = None,
) -> dict:
    device = world_model["device"]
    stats = world_model["stats"]
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
    state_n = (state_t - stats["state_mean"]) / stats["state_std"]
    action_n = (action_t - stats["action_mean"]) / stats["action_std"]
    rgb_t = None
    if current_rgb is not None:
        rgb = np.asarray(current_rgb, dtype=np.float32)
        if rgb.ndim != 3:
            raise ValueError(f"current_rgb must be HWC or CHW, got shape {rgb.shape}")
        if rgb.shape[0] == 3:
            rgb_chw = rgb
        elif rgb.shape[-1] == 3:
            rgb_chw = np.transpose(rgb, (2, 0, 1))
        else:
            raise ValueError(f"current_rgb must have 3 channels, got shape {rgb.shape}")
        if float(rgb_chw.max()) > 1.5:
            rgb_chw = rgb_chw / 255.0
        rgb_t = torch.as_tensor(rgb_chw, dtype=torch.float32, device=device).unsqueeze(0)
    out = world_model["model"](
        state_n,
        action_n,
        task_id=None if task_id is None else torch.as_tensor([int(task_id)], dtype=torch.long, device=device),
        progress=None if progress is None else torch.as_tensor(progress, dtype=torch.float32, device=device).reshape(1, 1),
        current_rgb=rgb_t,
    )
    next_state = out["next_state"] * stats["state_std"] + stats["state_mean"]
    return {
        "next_state": next_state.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "next_progress": float(out["next_progress"].squeeze().detach().cpu()),
        "success_prob": float(torch.sigmoid(out["success_logit"]).squeeze().detach().cpu()),
        "next_rgb": out["next_rgb"].squeeze(0).detach().cpu().numpy().astype(np.float32),
    }
