from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Inlined from tasks/robocasa_world_model/model.py; keep this file self-contained.
class RoboCasaWorldModel(nn.Module):
    """State/action-conditioned dynamics model with optional latent VAE state."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        task_count: int,
        width: int = 512,
        depth: int = 4,
        task_dim: int = 64,
        latent_dim: int = 0,
        condition_on_task: bool = False,
        condition_on_progress: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task_dim = int(task_dim) if bool(condition_on_task) else 0
        self.latent_dim = int(latent_dim)
        self.condition_on_progress = bool(condition_on_progress)
        state_width = self.latent_dim if self.latent_dim > 0 else self.state_dim

        if self.latent_dim > 0:
            self.encoder = _mlp(self.state_dim, 2 * self.latent_dim, width, max(1, depth // 2), dropout)
            self.decoder = _mlp(self.latent_dim, self.state_dim, width, max(1, depth // 2), dropout)
        else:
            self.encoder = None
            self.decoder = None

        if self.task_dim > 0:
            self.task = nn.Embedding(max(1, self.task_count), self.task_dim)
        else:
            self.task = None

        inp = state_width + self.action_dim + self.task_dim + (1 if self.condition_on_progress else 0)
        self.trunk = _mlp(inp, width, width, depth, dropout, final_norm=True)
        self.delta = nn.Linear(width, state_width)
        self.progress = nn.Linear(width, 1)
        self.success = nn.Linear(width, 1)

    def encode_state(self, state: torch.Tensor, *, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            zero = torch.zeros((), dtype=state.dtype, device=state.device)
            return state, zero, zero
        stats = self.encoder(state)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        if sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            return latent
        return self.decoder(latent)

    def conditioned_input(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        *,
        task_id: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [latent, action]
        if self.task is not None:
            if task_id is None:
                task_id = torch.zeros((latent.shape[0],), dtype=torch.long, device=latent.device)
            task_id = task_id.reshape(-1).long().clamp(0, max(0, self.task_count - 1))
            parts.append(self.task(task_id))
        if self.condition_on_progress:
            if progress is None:
                progress = torch.zeros((latent.shape[0], 1), dtype=latent.dtype, device=latent.device)
            else:
                progress = progress.reshape(latent.shape[0], -1)[:, :1].to(dtype=latent.dtype, device=latent.device)
            parts.append(progress)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        task_id: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        z, mu, logvar = self.encode_state(state, sample=sample_latent)
        h = self.trunk(self.conditioned_input(z, action, task_id=task_id, progress=progress))
        next_z = z + self.delta(h)
        next_state = self.decode_state(next_z)
        return {
            "next_state": next_state,
            "next_latent": next_z,
            "next_progress": torch.sigmoid(self.progress(h)),
            "success_logit": self.success(h),
            "latent_mu": mu,
            "latent_logvar": logvar,
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        state_weight: float = 1.0,
        progress_weight: float = 0.25,
        success_weight: float = 0.25,
        kl_weight: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(
            batch["state"],
            batch["action"],
            task_id=batch.get("task_id"),
            progress=batch.get("progress"),
            sample_latent=True,
        )
        state_loss = F.mse_loss(out["next_state"], batch["next_state"])
        progress_loss = F.mse_loss(out["next_progress"], batch["next_progress"])
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        if self.latent_dim > 0:
            mu = out["latent_mu"]
            logvar = out["latent_logvar"]
            kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
        else:
            kl = torch.zeros((), dtype=state_loss.dtype, device=state_loss.device)
        total = (
            float(state_weight) * state_loss
            + float(progress_weight) * progress_loss
            + float(success_weight) * success_loss
            + float(kl_weight) * kl
        )
        metrics = {
            "loss": total.detach(),
            "state_mse": state_loss.detach(),
            "progress_mse": progress_loss.detach(),
            "success_bce": success_loss.detach(),
            "kl": kl.detach(),
        }
        return total, metrics


def _mlp(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    dropout: float,
    *,
    final_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(depth)):
        layers.extend(
            [
                nn.Linear(dim, int(width)),
                nn.LayerNorm(int(width)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
        )
        dim = int(width)
    if final_norm:
        layers.append(nn.LayerNorm(dim))
    layers.append(nn.Linear(dim, int(out_dim)))
    return nn.Sequential(*layers)




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
        task_dim=int(cfg.get("task_dim", 0)),
        latent_dim=int(cfg["latent_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    stats = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in payload["stats"].items()}
    return {"model": model, "stats": stats, "config": cfg, "device": torch.device(device), "checkpoint": payload}


@torch.no_grad()
def predict_next(world_model: dict, state: np.ndarray, action: np.ndarray) -> dict:
    device = world_model["device"]
    stats = world_model["stats"]
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
    state_n = (state_t - stats["state_mean"]) / stats["state_std"]
    action_n = (action_t - stats["action_mean"]) / stats["action_std"]
    out = world_model["model"](state_n, action_n)
    next_state = out["next_state"] * stats["state_std"] + stats["state_mean"]
    return {
        "next_state": next_state.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "next_progress": float(out["next_progress"].squeeze().detach().cpu()),
        "success_prob": float(torch.sigmoid(out["success_logit"]).squeeze().detach().cpu()),
    }


@torch.no_grad()
def score_trajectory(world_model: dict, states: np.ndarray, actions: np.ndarray) -> dict:
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
    out = world_model["model"](state_n, action_n)
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
) -> dict:
    state = np.asarray(initial_state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    successes = []
    states = [state.copy()]
    for action in actions:
        step = predict_next(world_model, state, action)
        state = step["next_state"]
        successes.append(float(step["success_prob"]))
        states.append(state.copy())
    return {
        "predicted_success": float(max(successes) if successes else 0.0),
        "final_success_prob": float(successes[-1] if successes else 0.0),
        "final_progress": float(step["next_progress"]) if successes else 0.0,
        "states": np.asarray(states, dtype=np.float32),
        "success_trace": np.asarray(successes, dtype=np.float32),
    }
