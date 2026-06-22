from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_world_model.model import RoboCasaWorldModel


def load_world_model(checkpoint: str, device: str = "auto") -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = RoboCasaWorldModel(
        state_dim=int(cfg["state_dim"]),
        action_dim=int(cfg["action_dim"]),
        task_count=int(cfg["task_count"]),
        width=int(cfg["width"]),
        depth=int(cfg["depth"]),
        task_dim=int(cfg["task_dim"]),
        latent_dim=int(cfg["latent_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    stats = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in payload["stats"].items()}
    return {"model": model, "stats": stats, "config": cfg, "device": torch.device(device), "checkpoint": payload}


@torch.no_grad()
def predict_next(world_model: dict, state: np.ndarray, action: np.ndarray, task_id: int, progress: float) -> dict:
    device = world_model["device"]
    stats = world_model["stats"]
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
    state_n = (state_t - stats["state_mean"]) / stats["state_std"]
    action_n = (action_t - stats["action_mean"]) / stats["action_std"]
    out = world_model["model"](
        state_n,
        action_n,
        torch.tensor([int(task_id)], dtype=torch.long, device=device),
        torch.tensor([[float(progress)]], dtype=torch.float32, device=device),
    )
    next_state = out["next_state"] * stats["state_std"] + stats["state_mean"]
    return {
        "next_state": next_state.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "next_progress": float(out["next_progress"].squeeze().detach().cpu()),
        "success_prob": float(torch.sigmoid(out["success_logit"]).squeeze().detach().cpu()),
    }


@torch.no_grad()
def score_trajectory(world_model: dict, states: np.ndarray, actions: np.ndarray, task_id: int) -> dict:
    device = world_model["device"]
    stats = world_model["stats"]
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    n = min(len(states), len(actions))
    if n <= 0:
        return {
            "predicted_success": 0.0,
            "final_success_prob": 0.0,
            "final_progress": 0.0,
            "success_trace": np.zeros((0,), dtype=np.float32),
            "progress_trace": np.zeros((0,), dtype=np.float32),
        }
    state_t = torch.as_tensor(states[:n], dtype=torch.float32, device=device)
    action_t = torch.as_tensor(actions[:n], dtype=torch.float32, device=device)
    state_n = (state_t - stats["state_mean"]) / stats["state_std"]
    action_n = (action_t - stats["action_mean"]) / stats["action_std"]
    progress = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32, device=device).reshape(-1, 1)
    out = world_model["model"](
        state_n,
        action_n,
        torch.full((n,), int(task_id), dtype=torch.long, device=device),
        progress,
    )
    success = torch.sigmoid(out["success_logit"]).reshape(-1)
    next_progress = out["next_progress"].reshape(-1).clamp(0.0, 1.0)
    return {
        "predicted_success": float(success.max().detach().cpu()),
        "final_success_prob": float(success[-1].detach().cpu()),
        "final_progress": float(next_progress[-1].detach().cpu()),
        "success_trace": success.detach().cpu().numpy().astype(np.float32),
        "progress_trace": next_progress.detach().cpu().numpy().astype(np.float32),
    }


@torch.no_grad()
def rollout_score(
    world_model: dict,
    initial_state: np.ndarray,
    actions: np.ndarray,
    task_id: int,
    *,
    initial_progress: float = 0.0,
) -> dict:
    state = np.asarray(initial_state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    successes = []
    progress = float(initial_progress)
    states = [state.copy()]
    for action in actions:
        step = predict_next(world_model, state, action, int(task_id), progress)
        state = step["next_state"]
        progress = float(np.clip(step["next_progress"], 0.0, 1.0))
        successes.append(float(step["success_prob"]))
        states.append(state.copy())
    return {
        "predicted_success": float(max(successes) if successes else 0.0),
        "final_success_prob": float(successes[-1] if successes else 0.0),
        "final_progress": float(progress),
        "states": np.asarray(states, dtype=np.float32),
        "success_trace": np.asarray(successes, dtype=np.float32),
    }
