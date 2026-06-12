from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RoboCasaTinyEvaluator(nn.Module):
    """Small action-conditioned latent evaluator for fast RoboCasa scoring."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        action_dim: int,
        task_count: int,
        latent_dim: int = 256,
        task_dim: int = 32,
        width: int = 512,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.latent_dim = int(latent_dim)
        self.image = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
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
        self.task = nn.Embedding(task_count, task_dim)
        self.encoder = nn.Sequential(
            nn.Linear(width + proprio_dim + task_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim + task_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, latent_dim + proprio_dim),
        )
        self.progress_head = nn.Sequential(
            nn.Linear(latent_dim + task_dim, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self.success_head = nn.Sequential(
            nn.Linear(latent_dim + task_dim, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )

    def encode(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        image_feat = self.image(torch.cat([agent, wrist], dim=1))
        task_feat = self.task(task_id)
        return self.encoder(torch.cat([image_feat, proprio, task_feat], dim=-1))

    def step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_feat = self.task(task_id)
        out = self.dynamics(torch.cat([latent, action, task_feat], dim=-1))
        delta_z, next_proprio = out[..., : self.latent_dim], out[..., self.latent_dim :]
        next_latent = F.layer_norm(latent + 0.1 * torch.tanh(delta_z), (self.latent_dim,))
        return next_latent, next_proprio

    def heads(self, latent: torch.Tensor, task_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([latent, self.task(task_id)], dim=-1)
        progress = self.progress_head(h).squeeze(-1)
        success_logit = self.success_head(h).squeeze(-1)
        return progress, success_logit

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = self.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        z_target = self.encode(batch["next_agent"], batch["next_wrist"], batch["next_proprio"], batch["task_id"])
        z_next, proprio_next = self.step(z, batch["action"], batch["task_id"])
        progress, success_logit = self.heads(z, batch["task_id"])
        next_progress, next_success_logit = self.heads(z_next, batch["task_id"])
        return {
            "z": z,
            "z_target": z_target.detach(),
            "z_next": z_next,
            "proprio_next": proprio_next,
            "progress": progress,
            "success_logit": success_logit,
            "next_progress": next_progress,
            "next_success_logit": next_success_logit,
        }


def tiny_evaluator_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    latent_weight: float = 1.0,
    proprio_weight: float = 1.0,
    progress_weight: float = 0.5,
    success_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_loss = F.mse_loss(outputs["z_next"], outputs["z_target"])
    proprio_loss = F.mse_loss(outputs["proprio_next"], batch["next_proprio"])
    progress_loss = F.mse_loss(torch.sigmoid(outputs["progress"]), batch["progress"])
    next_progress_loss = F.mse_loss(torch.sigmoid(outputs["next_progress"]), batch["next_progress"])
    success_loss = F.binary_cross_entropy_with_logits(outputs["success_logit"], batch["success"])
    next_success_loss = F.binary_cross_entropy_with_logits(outputs["next_success_logit"], batch["next_success"])
    loss = (
        latent_weight * latent_loss
        + proprio_weight * proprio_loss
        + progress_weight * 0.5 * (progress_loss + next_progress_loss)
        + success_weight * 0.5 * (success_loss + next_success_loss)
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
        "proprio_loss": float(proprio_loss.detach().cpu()),
        "progress_loss": float(progress_loss.detach().cpu()),
        "success_loss": float(success_loss.detach().cpu()),
    }
    return loss, metrics
