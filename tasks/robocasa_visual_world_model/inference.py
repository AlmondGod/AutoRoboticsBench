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
    model = VisualRoboCasaWorldModel(
        state_dim=int(cfg["state_dim"]),
        action_dim=int(cfg["action_dim"]),
        task_count=int(cfg["task_count"]),
        image_size=int(cfg["image_size"]),
        width=int(cfg["width"]),
        depth=int(cfg["depth"]),
        task_dim=int(cfg["task_dim"]),
        latent_dim=int(cfg["latent_dim"]),
        visual_latent_dim=int(cfg.get("visual_latent_dim", 64)),
        visual_decoder_width=int(cfg.get("visual_decoder_width", 0)) or None,
        visual_decoder_depth=int(cfg.get("visual_decoder_depth", 3)),
        visual_decoder_type=str(cfg.get("visual_decoder_type", "mlp")),
        current_rgb_conditioned=bool(cfg.get("current_rgb_conditioned", False)),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    stats = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in payload["stats"].items()}
    return {"model": model, "stats": stats, "config": cfg, "device": torch.device(device), "checkpoint": payload}


@torch.no_grad()
def predict_next(
    world_model: dict,
    state: np.ndarray,
    action: np.ndarray,
    task_id: int,
    progress: float,
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
        torch.tensor([int(task_id)], dtype=torch.long, device=device),
        torch.tensor([[float(progress)]], dtype=torch.float32, device=device),
        current_rgb=rgb_t,
    )
    next_state = out["next_state"] * stats["state_std"] + stats["state_mean"]
    return {
        "next_state": next_state.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "next_progress": float(out["next_progress"].squeeze().detach().cpu()),
        "success_prob": float(torch.sigmoid(out["success_logit"]).squeeze().detach().cpu()),
        "next_rgb": out["next_rgb"].squeeze(0).detach().cpu().numpy().astype(np.float32),
    }
