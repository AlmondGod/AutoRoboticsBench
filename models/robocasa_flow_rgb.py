from __future__ import annotations

import torch
from torch import nn


class RoboCasaNextRGBFlow(nn.Module):
    """Small conditional rectified-flow decoder for next RGB prediction."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        task_count: int,
        task_dim: int = 32,
        cond_dim: int = 128,
        hidden: int = 96,
        cond_channels: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task = nn.Embedding(task_count, task_dim)
        self.cond = nn.Sequential(
            nn.Linear(latent_dim + action_dim + task_dim + 1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.cond_map = nn.Linear(cond_dim, cond_channels)
        in_ch = 6 + cond_channels + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 6, 3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        cond = self.cond(torch.cat([latent, action, self.task(task_id), t], dim=-1))
        cond_map = self.cond_map(cond)[:, :, None, None].expand(-1, -1, x_t.shape[-2], x_t.shape[-1])
        t_map = t[:, :, None, None].expand(-1, 1, x_t.shape[-2], x_t.shape[-1])
        return self.net(torch.cat([x_t, cond_map, t_map], dim=1))

    @torch.no_grad()
    def sample(
        self,
        *,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
        steps: int = 8,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            x = torch.rand((latent.shape[0], 6, 64, 64), device=latent.device, dtype=latent.dtype)
        else:
            x = noise.to(device=latent.device, dtype=latent.dtype)
        dt = 1.0 / max(1, int(steps))
        for idx in range(int(steps)):
            t = torch.full((latent.shape[0],), idx * dt, device=latent.device, dtype=latent.dtype)
            x = x + dt * self(x, t, latent, action, task_id)
        return x.clamp(0.0, 1.0)
