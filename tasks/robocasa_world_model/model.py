from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task_dim = int(task_dim)
        self.latent_dim = int(latent_dim)
        state_width = self.latent_dim if self.latent_dim > 0 else self.state_dim

        if self.latent_dim > 0:
            self.encoder = _mlp(self.state_dim, 2 * self.latent_dim, width, max(1, depth // 2), dropout)
            self.decoder = _mlp(self.latent_dim, self.state_dim, width, max(1, depth // 2), dropout)
        else:
            self.encoder = None
            self.decoder = None

        inp = state_width + self.action_dim
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

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        z, mu, logvar = self.encode_state(state, sample=sample_latent)
        h = torch.cat([z, action], dim=-1)
        h = self.trunk(h)
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
