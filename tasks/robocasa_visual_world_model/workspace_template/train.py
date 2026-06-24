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

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

# Inlined from tasks/robocasa_world_model/model.py and tasks/robocasa_visual_world_model/model.py; keep this file self-contained.
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


class ImageVAE(nn.Module):
    """Small RGB VAE used to make visual prediction latent-space aware."""

    def __init__(
        self,
        *,
        image_size: int,
        latent_dim: int = 64,
        width: int = 256,
        decoder_width: int | None = None,
        decoder_depth: int = 3,
        decoder_type: str = "conv",
        encoder_pool_size: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.decoder_type = str(decoder_type)
        self.encoder_pool_size = max(1, int(encoder_pool_size))
        decoder_width = int(decoder_width or width)
        decoder_depth = max(1, int(decoder_depth))
        base_size = max(4, int(image_size) // 16)
        self.decoder_channels = decoder_width
        self.decoder_base_size = base_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((self.encoder_pool_size, self.encoder_pool_size)),
            nn.Flatten(),
            nn.Linear(192 * self.encoder_pool_size * self.encoder_pool_size, 2 * self.latent_dim),
        )
        if self.decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, int(width)),
                nn.LayerNorm(int(width)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(width), int(width)),
                nn.LayerNorm(int(width)),
                nn.GELU(),
                nn.Linear(int(width), 3 * self.image_size * self.image_size),
            )
            return
        self.decoder_in = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_width * base_size * base_size),
            nn.LayerNorm(decoder_width * base_size * base_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        channels = [decoder_width, max(128, decoder_width // 2), max(64, decoder_width // 4), 32, 16]
        blocks: list[nn.Module] = []
        current = channels[0]
        upsample_blocks = max(1, int(image_size).bit_length() - int(base_size).bit_length())
        upsample_blocks = min(4, upsample_blocks)
        for idx in range(upsample_blocks):
            out_channels = channels[min(idx + 1, len(channels) - 1)]
            blocks.extend(
                [
                    nn.ConvTranspose2d(current, out_channels, 4, stride=2, padding=1),
                    nn.GroupNorm(max(1, min(8, out_channels)), out_channels),
                    nn.GELU(),
                ]
            )
            current = out_channels
            for _ in range(max(0, decoder_depth - 1)):
                blocks.extend(
                    [
                        nn.Conv2d(current, current, 3, padding=1),
                        nn.GroupNorm(max(1, min(8, current)), current),
                        nn.GELU(),
                    ]
                )
        blocks.append(nn.Conv2d(current, 3, 3, padding=1))
        self.decoder = nn.Sequential(*blocks)
        self.decoder_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def encode(self, image: torch.Tensor, *, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.encoder(image)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        if sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.decoder_type == "mlp":
            image = torch.sigmoid(self.decoder(latent))
            return image.reshape(-1, 3, self.image_size, self.image_size)
        x = self.decoder_in(latent)
        x = x.reshape(-1, self.decoder_channels, self.decoder_base_size, self.decoder_base_size)
        image = self.decoder(x)
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(image + self.decoder_refine(image))

    def forward(self, image: torch.Tensor, *, sample: bool = False) -> dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(image, sample=sample)
        return {"latent": z, "mu": mu, "logvar": logvar, "reconstruction": self.decode(z)}


class SpatialImageAutoencoder(nn.Module):
    """Frame autoencoder with a spatial latent map instead of a single vector bottleneck."""

    def __init__(
        self,
        *,
        image_size: int,
        latent_channels: int = 128,
        width: int = 128,
        depth: int = 2,
        downsample_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.downsample_blocks = max(1, int(downsample_blocks))
        width = max(32, int(width))
        depth = max(1, int(depth))
        dropout = float(dropout)

        encoder: list[nn.Module] = []
        in_channels = 3
        current = width
        for block_idx in range(self.downsample_blocks):
            out_channels = min(width * (2**block_idx), width * 4)
            encoder.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.GroupNorm(max(1, min(8, out_channels)), out_channels),
                    nn.GELU(),
                ]
            )
            for _ in range(depth):
                encoder.append(_SpatialResidualBlock(out_channels, dropout=dropout))
            in_channels = out_channels
            current = out_channels
        encoder.append(nn.Conv2d(current, self.latent_channels, 1))
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = [
            nn.Conv2d(self.latent_channels, current, 3, padding=1),
            nn.GroupNorm(max(1, min(8, current)), current),
            nn.GELU(),
        ]
        for block_idx in reversed(range(self.downsample_blocks)):
            out_channels = max(width, min(width * (2**block_idx), width * 4))
            for _ in range(depth):
                decoder.append(_SpatialResidualBlock(current, dropout=dropout))
            decoder.extend(
                [
                    nn.ConvTranspose2d(current, out_channels, 4, stride=2, padding=1),
                    nn.GroupNorm(max(1, min(8, out_channels)), out_channels),
                    nn.GELU(),
                ]
            )
            current = out_channels
        decoder.extend(
            [
                _SpatialResidualBlock(current, dropout=dropout),
                nn.Conv2d(current, 3, 3, padding=1),
            ]
        )
        self.decoder = nn.Sequential(*decoder)
        self.decoder_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def encode(self, image: torch.Tensor, *, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del sample
        latent = self.encoder(image)
        zero = torch.zeros((), dtype=image.dtype, device=image.device)
        return latent, zero, zero

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.decoder(latent)
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(image + self.decoder_refine(image))

    def forward(self, image: torch.Tensor, *, sample: bool = False) -> dict[str, torch.Tensor]:
        latent, mu, logvar = self.encode(image, sample=sample)
        return {"latent": latent, "mu": mu, "logvar": logvar, "reconstruction": self.decode(latent)}


class _SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(max(1, min(8, channels)), channels),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(max(1, min(8, channels)), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class SpatialLatentDynamicsHead(nn.Module):
    """Predict a spatial latent delta with convolutions over the current latent map."""

    def __init__(
        self,
        *,
        latent_shape: tuple[int, int, int],
        hidden_dim: int,
        hidden_channels: int | None = None,
        depth: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        latent_channels, _, _ = latent_shape
        channels = int(hidden_channels or latent_channels)
        depth = max(1, int(depth))
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.current_proj = nn.Conv2d(int(latent_channels), channels, 1)
        self.hidden_bias = nn.Linear(int(hidden_dim), channels)
        self.blocks = nn.ModuleList(
            [
                _ConditionedSpatialResidualBlock(
                    channels,
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                )
                for _ in range(depth)
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(max(1, min(8, channels)), channels),
            nn.GELU(),
            nn.Conv2d(channels, int(latent_channels), 3, padding=1),
        )

    def forward(self, hidden: torch.Tensor, current_latent: torch.Tensor) -> torch.Tensor:
        if current_latent.ndim != 4:
            raise ValueError(f"spatial latent head expected BCHW latent, got {tuple(current_latent.shape)}")
        bias = self.hidden_bias(hidden).reshape(hidden.shape[0], -1, 1, 1)
        x = self.current_proj(current_latent.float()) + bias
        for block in self.blocks:
            x = block(x, hidden)
        return self.out(x)


class _ConditionedSpatialResidualBlock(nn.Module):
    def __init__(self, channels: int, *, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.dropout = nn.Dropout2d(float(dropout))
        self.film = nn.Linear(int(hidden_dim), 2 * int(channels))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(hidden).reshape(hidden.shape[0], 2, x.shape[1], 1, 1).unbind(dim=1)
        y = self.conv1(x)
        y = self.norm1(y)
        y = y * (1.0 + scale) + bias
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class VisualRoboCasaWorldModel(nn.Module):
    """State/action dynamics model with image-VAE latent prediction."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        task_count: int,
        image_size: int = 32,
        width: int = 512,
        depth: int = 4,
        task_dim: int = 64,
        latent_dim: int = 64,
        visual_latent_dim: int = 64,
        visual_decoder_width: int | None = None,
        visual_decoder_depth: int = 3,
        visual_decoder_type: str = "conv",
        visual_encoder_pool_size: int = 1,
        visual_architecture: str = "vae",
        spatial_latent_channels: int = 128,
        spatial_width: int = 128,
        spatial_depth: int = 2,
        spatial_downsample_blocks: int = 2,
        spatial_dynamics_type: str = "mlp",
        spatial_dynamics_depth: int = 4,
        spatial_dynamics_hidden_channels: int = 0,
        current_rgb_conditioned: bool = False,
        visual_delta_prediction: bool = False,
        condition_on_task: bool = False,
        condition_on_progress: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.visual_architecture = str(visual_architecture)
        self.spatial_latent_channels = int(spatial_latent_channels)
        self.spatial_downsample_blocks = max(1, int(spatial_downsample_blocks))
        self.spatial_dynamics_type = str(spatial_dynamics_type)
        self.spatial_dynamics_depth = int(spatial_dynamics_depth)
        self.spatial_dynamics_hidden_channels = int(spatial_dynamics_hidden_channels)
        self.current_rgb_conditioned = bool(current_rgb_conditioned)
        self.visual_delta_prediction = bool(visual_delta_prediction)
        self.dynamics = RoboCasaWorldModel(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            task_count=int(task_count),
            width=int(width),
            depth=int(depth),
            task_dim=int(task_dim),
            latent_dim=int(latent_dim),
            condition_on_task=bool(condition_on_task),
            condition_on_progress=bool(condition_on_progress),
            dropout=float(dropout),
        )
        if self.visual_architecture == "spatial":
            spatial_hw = int(image_size) // (2 ** self.spatial_downsample_blocks)
            if spatial_hw <= 0:
                raise ValueError("spatial latent map would be empty; reduce --spatial-downsample-blocks")
            self.visual_latent_shape = (self.spatial_latent_channels, spatial_hw, spatial_hw)
            self.visual_latent_numel = int(self.spatial_latent_channels * spatial_hw * spatial_hw)
            self.visual_latent_dim = self.visual_latent_numel
            self.image_vae = SpatialImageAutoencoder(
                image_size=int(image_size),
                latent_channels=self.spatial_latent_channels,
                width=int(spatial_width),
                depth=int(spatial_depth),
                downsample_blocks=self.spatial_downsample_blocks,
                dropout=float(dropout),
            )
        else:
            self.visual_architecture = "vae"
            self.visual_latent_shape = None
            self.visual_latent_numel = int(visual_latent_dim)
            self.visual_latent_dim = int(visual_latent_dim)
            self.image_vae = ImageVAE(
                image_size=int(image_size),
                latent_dim=int(visual_latent_dim),
                width=max(128, int(width) // 2),
                decoder_width=int(visual_decoder_width or max(128, int(width) // 2)),
                decoder_depth=int(visual_decoder_depth),
                decoder_type=str(visual_decoder_type),
                encoder_pool_size=int(visual_encoder_pool_size),
                dropout=float(dropout),
            )
        if self.visual_architecture == "spatial" and self.spatial_dynamics_type == "conv":
            self.next_visual_latent = SpatialLatentDynamicsHead(
                latent_shape=self.visual_latent_shape,
                hidden_dim=int(width),
                hidden_channels=self.spatial_dynamics_hidden_channels or self.spatial_latent_channels,
                depth=self.spatial_dynamics_depth,
                dropout=float(dropout),
            )
        else:
            self.spatial_dynamics_type = "mlp" if self.visual_architecture == "spatial" else "vector_mlp"
            visual_condition_dim = int(width) + (self.visual_latent_numel if self.current_rgb_conditioned else 0)
            self.next_visual_latent = nn.Sequential(
                nn.Linear(visual_condition_dim, int(width)),
                nn.LayerNorm(int(width)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(width), self.visual_latent_numel),
            )
    @property
    def state_dim(self) -> int:
        return int(self.dynamics.state_dim)

    @property
    def action_dim(self) -> int:
        return int(self.dynamics.action_dim)

    @property
    def task_count(self) -> int:
        return int(self.dynamics.task_count)

    @property
    def latent_dim(self) -> int:
        return int(self.dynamics.latent_dim)

    def transition_hidden(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        task_id: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
        sample_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.dynamics.encode_state(state, sample=sample_latent)
        h = self.dynamics.conditioned_input(z, action, task_id=task_id, progress=progress)
        return self.dynamics.trunk(h), z, mu, logvar

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        task_id: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
        current_rgb: torch.Tensor | None = None,
        current_visual_latent: torch.Tensor | None = None,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden, z, mu, logvar = self.transition_hidden(
            state,
            action,
            task_id=task_id,
            progress=progress,
            sample_latent=sample_latent,
        )
        next_z = z + self.dynamics.delta(hidden)
        next_state = self.dynamics.decode_state(next_z)
        visual_condition, current_visual_latent = self.visual_condition(
            hidden,
            current_rgb=current_rgb,
            current_visual_latent=current_visual_latent,
        )
        if self.visual_architecture == "spatial" and self.spatial_dynamics_type == "conv":
            head_current = current_visual_latent
            if head_current is None:
                head_current = self._zero_visual_latent(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
            pred_visual_delta = self.next_visual_latent(hidden, head_current)
        else:
            pred_visual_delta = self._reshape_visual_latent(self.next_visual_latent(visual_condition))
        if self.visual_delta_prediction and current_visual_latent is not None:
            pred_visual_latent = current_visual_latent.float() + pred_visual_delta
        else:
            pred_visual_latent = pred_visual_delta
        next_rgb = self.image_vae.decode(pred_visual_latent)
        return {
            "next_state": next_state,
            "next_latent": next_z,
            "next_progress": torch.sigmoid(self.dynamics.progress(hidden)),
            "success_logit": self.dynamics.success(hidden),
            "next_rgb": next_rgb,
            "next_visual_latent": pred_visual_latent,
            "next_visual_delta": pred_visual_delta,
            "hidden": hidden,
            "visual_condition": visual_condition,
            "latent_mu": mu,
            "latent_logvar": logvar,
        }

    def visual_condition(
        self,
        hidden: torch.Tensor,
        *,
        current_rgb: torch.Tensor | None = None,
        current_visual_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.current_rgb_conditioned:
            return hidden, None
        if current_visual_latent is None:
            if current_rgb is not None:
                current_visual_latent, _, _ = self.image_vae.encode(current_rgb, sample=False)
            else:
                current_visual_latent = self._zero_visual_latent(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
        if self.visual_architecture == "spatial" and self.spatial_dynamics_type == "conv":
            return hidden, current_visual_latent
        return torch.cat([hidden, self._flatten_visual_latent(current_visual_latent).float()], dim=-1), current_visual_latent

    def _zero_visual_latent(self, batch_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.visual_architecture == "spatial":
            return torch.zeros((int(batch_size), *self.visual_latent_shape), dtype=dtype, device=device)
        return torch.zeros((int(batch_size), self.visual_latent_numel), dtype=dtype, device=device)

    def _flatten_visual_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim > 2:
            return latent.flatten(1)
        return latent

    def _reshape_visual_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.visual_architecture == "spatial" and latent.ndim == 2:
            return latent.reshape(latent.shape[0], *self.visual_latent_shape)
        return latent

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        state_weight: float = 1.0,
        progress_weight: float = 0.25,
        success_weight: float = 0.25,
        visual_weight: float = 1.0,
        image_vae_weight: float = 0.25,
        visual_latent_weight: float = 0.5,
        visual_l1_weight: float = 0.0,
        visual_grad_weight: float = 0.0,
        image_vae_l1_weight: float = 1.0,
        image_vae_mse_weight: float = 0.25,
        image_vae_grad_weight: float = 0.25,
        kl_weight: float = 1e-4,
        visual_kl_weight: float = 1e-5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        next_visual, next_visual_mu, next_visual_logvar = self.image_vae.encode(batch["next_rgb"], sample=False)
        current_visual, current_visual_mu, current_visual_logvar = self.image_vae.encode(batch["rgb"], sample=False)
        out = self(
            batch["state"],
            batch["action"],
            task_id=batch.get("task_id"),
            progress=batch.get("progress"),
            current_visual_latent=current_visual.detach(),
            sample_latent=True,
        )
        state_loss = F.mse_loss(out["next_state"], batch["next_state"])
        progress_loss = F.mse_loss(out["next_progress"], batch["next_progress"])
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        rgb_mse = F.mse_loss(out["next_rgb"], batch["next_rgb"])
        rgb_l1 = F.l1_loss(out["next_rgb"], batch["next_rgb"])
        rgb_grad = _gradient_loss(out["next_rgb"], batch["next_rgb"])
        rgb_loss = rgb_mse + float(visual_l1_weight) * rgb_l1 + float(visual_grad_weight) * rgb_grad
        if self.visual_delta_prediction:
            visual_latent_target = next_visual.detach() - current_visual.detach()
            visual_latent_loss = F.mse_loss(out["next_visual_delta"], visual_latent_target)
        else:
            visual_latent_loss = F.mse_loss(out["next_visual_latent"], next_visual.detach())

        current_recon = self.image_vae.decode(current_visual)
        next_recon = self.image_vae.decode(next_visual)
        current_recon_loss, current_recon_metrics = _reconstruction_loss(
            current_recon,
            batch["rgb"],
            l1_weight=float(image_vae_l1_weight),
            mse_weight=float(image_vae_mse_weight),
            grad_weight=float(image_vae_grad_weight),
        )
        next_recon_loss, next_recon_metrics = _reconstruction_loss(
            next_recon,
            batch["next_rgb"],
            l1_weight=float(image_vae_l1_weight),
            mse_weight=float(image_vae_mse_weight),
            grad_weight=float(image_vae_grad_weight),
        )
        image_vae_loss = 0.5 * (current_recon_loss + next_recon_loss)
        image_vae_mse = 0.5 * (current_recon_metrics["mse"] + next_recon_metrics["mse"])
        image_vae_l1 = 0.5 * (current_recon_metrics["l1"] + next_recon_metrics["l1"])
        image_vae_grad = 0.5 * (current_recon_metrics["grad"] + next_recon_metrics["grad"])
        visual_kl = 0.5 * (_kl(current_visual_mu, current_visual_logvar) + _kl(next_visual_mu, next_visual_logvar))

        if self.latent_dim > 0:
            kl = _kl(out["latent_mu"], out["latent_logvar"])
        else:
            kl = torch.zeros((), dtype=state_loss.dtype, device=state_loss.device)
        total = (
            float(state_weight) * state_loss
            + float(progress_weight) * progress_loss
            + float(success_weight) * success_loss
            + float(visual_weight) * rgb_loss
            + float(image_vae_weight) * image_vae_loss
            + float(visual_latent_weight) * visual_latent_loss
            + float(kl_weight) * kl
            + float(visual_kl_weight) * visual_kl
        )
        return total, {
            "loss": total.detach(),
            "state_mse": state_loss.detach(),
            "progress_mse": progress_loss.detach(),
            "success_bce": success_loss.detach(),
            "rgb_mse": rgb_mse.detach(),
            "rgb_l1": rgb_l1.detach(),
            "rgb_grad": rgb_grad.detach(),
            "visual_latent_mse": visual_latent_loss.detach(),
            "image_vae_loss": image_vae_loss.detach(),
            "image_vae_mse": image_vae_mse.detach(),
            "image_vae_l1": image_vae_l1.detach(),
            "image_vae_grad": image_vae_grad.detach(),
            "kl": kl.detach(),
            "visual_kl": visual_kl.detach(),
        }


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


def _reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float,
    mse_weight: float,
    grad_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    grad = _gradient_loss(pred, target)
    loss = float(l1_weight) * l1 + float(mse_weight) * mse + float(grad_weight) * grad
    return loss, {"l1": l1.detach(), "mse": mse.detach(), "grad": grad.detach()}


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


from dataclasses import dataclass
from typing import Any

import pandas as pd

# Inlined dataset helpers from the reward-model task; keep train.py self-contained.
def ensure_robocasa_runtime() -> None:
    import json as _json
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    for rel in ("third_party/robocasa", "third_party/robosuite", "."):
        path = str((repo / rel).resolve())
        if path not in _sys.path:
            _sys.path.insert(0, path)
    _os.environ.setdefault("PYTHONPATH", _os.pathsep.join(_sys.path))
    try:
        import lerobot.datasets.utils as _utils
    except ModuleNotFoundError:
        return
    if hasattr(_utils, "write_info"):
        return

    def write_info(info: dict, root: str | _Path) -> None:
        root_path = _Path(root)
        path = root_path if root_path.name == "info.json" else root_path / "info.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(info, indent=2, sort_keys=True) + "\n")

    _utils.write_info = write_info


DEFAULT_MANIFEST = ROOT / "data" / "robocasa5" / "manifest.json"
DEFAULT_SPLIT = ROOT / "data" / "autorobobench" / "robocasa_bc5_splits.json"
DEFAULT_POLICY_SET = ROOT / "data" / "autorobobench" / "robocasa_world_model_policy_set.json"
DEFAULT_VIDEO_POOL = ROOT / "data" / "autorobobench" / "robocasa_world_model_video_pool.json"


@dataclass
class TransitionData:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    progress: np.ndarray
    next_progress: np.ndarray
    success: np.ndarray
    task_id: np.ndarray
    episode_id: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.state.shape[0])


@dataclass(frozen=True)
class VideoOnlyEpisode:
    alias: str
    task_id: int
    split: str
    episode_id: int
    view: str
    video_path: Path


def load_transition_data(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    split_path: str | Path = DEFAULT_SPLIT,
    train_episodes_per_task: int = 20,
    val_episodes_per_task: int = 5,
    task_aliases: set[str] | None = None,
    frame_stride: int = 1,
) -> tuple[TransitionData, TransitionData, list[dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_path).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    aliases = task_aliases or set()
    train_parts = []
    val_parts = []
    summary = []
    for split_task in split["tasks"]:
        alias = str(split_task["alias"])
        if aliases and alias not in aliases:
            continue
        task_id = int(split_task["task_id"])
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        all_train_ids = [int(x) for x in split_task["train_episode_ids"]]
        all_val_ids = [int(x) for x in split_task["val_episode_ids"]]
        train_limit = int(train_episodes_per_task)
        val_limit = int(val_episodes_per_task)
        train_ids = all_train_ids if train_limit <= 0 else all_train_ids[:train_limit]
        val_ids = all_val_ids if val_limit <= 0 else all_val_ids[:val_limit]
        train_count = _append_episodes(train_parts, dataset_root, train_ids, task_id, int(frame_stride))
        val_count = _append_episodes(val_parts, dataset_root, val_ids, task_id, int(frame_stride))
        summary.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str(dataset_root),
                "train_episode_ids": train_ids,
                "val_episode_ids": val_ids,
                "train_transitions": int(train_count),
                "val_transitions": int(val_count),
            }
        )
    return _concat(train_parts), _concat(val_parts), summary


def load_video_only_pool(
    video_pool_path: str | Path = DEFAULT_VIDEO_POOL,
    *,
    max_episodes_per_task: int = 0,
    task_aliases: set[str] | None = None,
    splits: set[str] | None = None,
) -> list[VideoOnlyEpisode]:
    """Return RGB video-only records without reading action/state parquet data."""
    pool = json.loads(Path(video_pool_path).read_text())
    aliases = task_aliases or set()
    wanted_splits = splits or set()
    template = str(pool.get("video_path_template", "videos/chunk-000/observation.images.{view}/episode_{episode_id:06d}.mp4"))
    records: list[VideoOnlyEpisode] = []
    for task in pool.get("tasks", []):
        alias = str(task["alias"])
        split = str(task.get("split", ""))
        if aliases and alias not in aliases:
            continue
        if wanted_splits and split not in wanted_splits:
            continue
        start, end = [int(x) for x in task["video_episode_range"]]
        episode_ids = list(range(start, end + 1))
        if int(max_episodes_per_task) > 0:
            episode_ids = episode_ids[: int(max_episodes_per_task)]
        dataset_root = Path(str(task["dataset_path"]))
        if not dataset_root.is_absolute():
            dataset_root = ROOT / dataset_root
        for episode_id in episode_ids:
            for view in pool.get("views", []):
                rel = template.format(view=str(view), episode_id=int(episode_id))
                records.append(
                    VideoOnlyEpisode(
                        alias=alias,
                        task_id=int(task["task_id"]),
                        split=split,
                        episode_id=int(episode_id),
                        view=str(view),
                        video_path=dataset_root / rel,
                    )
                )
    return records


def summarize_video_only_pool(records: list[VideoOnlyEpisode]) -> dict[str, Any]:
    by_task: dict[tuple[str, str], set[int]] = {}
    existing_videos = 0
    for record in records:
        by_task.setdefault((record.alias, record.split), set()).add(int(record.episode_id))
        if record.video_path.exists():
            existing_videos += 1
    return {
        "video_records": len(records),
        "video_files_existing": existing_videos,
        "video_episodes": sum(len(ids) for ids in by_task.values()),
        "tasks": [
            {
                "alias": alias,
                "split": split,
                "video_episodes": len(ids),
            }
            for (alias, split), ids in sorted(by_task.items())
        ],
    }


def load_video_frames(video_path: str | Path, *, stride: int = 1, max_frames: int = 0) -> np.ndarray:
    """Load RGB frames from a video-only record for optional self-supervised methods."""
    path = Path(video_path)
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        frames = []
        index = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if index % max(1, int(stride)) == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if int(max_frames) > 0 and len(frames) >= int(max_frames):
                    break
            index += 1
        cap.release()
        return np.asarray(frames, dtype=np.uint8)
    except ModuleNotFoundError:
        import imageio.v3 as iio

        frames = []
        for index, frame in enumerate(iio.imiter(path)):
            if index % max(1, int(stride)) == 0:
                frames.append(np.asarray(frame, dtype=np.uint8))
                if int(max_frames) > 0 and len(frames) >= int(max_frames):
                    break
        return np.asarray(frames, dtype=np.uint8)


def load_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    path = Path(video_path)
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise IndexError(f"could not read frame {frame_idx} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    except ModuleNotFoundError:
        import imageio.v3 as iio

        for index, frame in enumerate(iio.imiter(path)):
            if index == int(frame_idx):
                return np.asarray(frame, dtype=np.uint8)
        raise IndexError(f"could not read frame {frame_idx} from {path}")


def _append_episodes(parts: list[dict[str, np.ndarray]], dataset_root: Path, episode_ids: list[int], task_id: int, frame_stride: int) -> int:
    count = 0
    for episode_id in episode_ids:
        part = load_episode_transitions(dataset_root, int(episode_id), int(task_id), frame_stride=max(1, frame_stride))
        if part["state"].shape[0] > 0:
            parts.append(part)
            count += int(part["state"].shape[0])
    return count


def load_episode_transitions(dataset_root: Path, episode_id: int, task_id: int, *, frame_stride: int = 1) -> dict[str, np.ndarray]:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    frame = pd.read_parquet(episode_path)
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    action = episode_actions(dataset_root, episode_id, frame).astype(np.float32)
    n = min(len(state), len(action))
    if n <= 1:
        return _empty_part(state_dim=state.shape[-1] if state.ndim == 2 else 1, action_dim=action.shape[-1] if action.ndim == 2 else 1)
    rows = np.arange(0, n - 1, max(1, frame_stride), dtype=np.int32)
    progress = rows.astype(np.float32) / max(1, n - 1)
    next_progress = (rows + 1).astype(np.float32) / max(1, n - 1)
    success = _episode_success(frame, rows, n)
    return {
        "state": state[rows].astype(np.float32),
        "action": action[rows].astype(np.float32),
        "next_state": state[rows + 1].astype(np.float32),
        "progress": progress[:, None].astype(np.float32),
        "next_progress": next_progress[:, None].astype(np.float32),
        "success": success[:, None].astype(np.float32),
        "task_id": np.full((len(rows),), int(task_id), dtype=np.int64),
        "episode_id": np.full((len(rows),), int(episode_id), dtype=np.int32),
        "frame_idx": rows.astype(np.int32),
    }


def _episode_success(frame: pd.DataFrame, rows: np.ndarray, n: int) -> np.ndarray:
    for key in ("next.success", "success", "is_success"):
        if key in frame:
            values = np.asarray(frame[key].to_numpy(), dtype=np.float32).reshape(-1)
            return values[np.minimum(rows + 1, len(values) - 1)]
    success = np.zeros((len(rows),), dtype=np.float32)
    if len(success):
        success[-1] = 1.0
    return success


def episode_actions(dataset_root: Path, episode_id: int, frame: pd.DataFrame | None = None) -> np.ndarray:
    if frame is None:
        episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        frame = pd.read_parquet(episode_path)
    if "action" in frame:
        return np.stack(frame["action"].to_numpy()).astype(np.float32)
    try:
        ensure_robocasa_runtime()
        import robocasa.utils.lerobot_utils as LU

        return LU.get_episode_actions(dataset_root, episode_id).astype(np.float32)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "episode parquet has no action column and RoboCasa is not importable for lerobot_utils fallback"
        ) from exc


def _concat(parts: list[dict[str, np.ndarray]]) -> TransitionData:
    if not parts:
        return TransitionData(
            state=np.zeros((0, 1), dtype=np.float32),
            action=np.zeros((0, 1), dtype=np.float32),
            next_state=np.zeros((0, 1), dtype=np.float32),
            progress=np.zeros((0, 1), dtype=np.float32),
            next_progress=np.zeros((0, 1), dtype=np.float32),
            success=np.zeros((0, 1), dtype=np.float32),
            task_id=np.zeros((0,), dtype=np.int64),
            episode_id=np.zeros((0,), dtype=np.int32),
            frame_idx=np.zeros((0,), dtype=np.int32),
        )
    return TransitionData(**{key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]})


def _empty_part(state_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    return {
        "state": np.zeros((0, int(state_dim)), dtype=np.float32),
        "action": np.zeros((0, int(action_dim)), dtype=np.float32),
        "next_state": np.zeros((0, int(state_dim)), dtype=np.float32),
        "progress": np.zeros((0, 1), dtype=np.float32),
        "next_progress": np.zeros((0, 1), dtype=np.float32),
        "success": np.zeros((0, 1), dtype=np.float32),
        "task_id": np.zeros((0,), dtype=np.int64),
        "episode_id": np.zeros((0,), dtype=np.int32),
        "frame_idx": np.zeros((0,), dtype=np.int32),
    }


def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def normalize_data(data: TransitionData, stats: dict[str, np.ndarray]) -> TransitionData:
    return TransitionData(
        state=((data.state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
        action=((data.action - stats["action_mean"]) / stats["action_std"]).astype(np.float32),
        next_state=((data.next_state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
        progress=data.progress.astype(np.float32),
        next_progress=data.next_progress.astype(np.float32),
        success=data.success.astype(np.float32),
        task_id=data.task_id.astype(np.int64),
        episode_id=data.episode_id.astype(np.int32),
        frame_idx=data.frame_idx.astype(np.int32),
    )


def make_stats(train: TransitionData) -> dict[str, np.ndarray]:
    state_mean, state_std = mean_std(np.concatenate([train.state, train.next_state], axis=0))
    action_mean, action_std = mean_std(train.action)
    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# Inlined inverse-dynamics loader; keep train.py self-contained.
class VideoInverseDynamics(nn.Module):
    def __init__(self, *, action_dim: int, task_count: int, task_dim: int = 32, width: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task = nn.Embedding(int(task_count), int(task_dim))
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(192 + int(task_dim) + 1, int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(action_dim)),
        )

    def encode_pair(self, image_pair: torch.Tensor) -> torch.Tensor:
        return self.encoder(image_pair)

    def forward(self, image_pair: torch.Tensor, task_id: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        if progress.ndim == 1:
            progress = progress[:, None]
        h = self.encode_pair(image_pair)
        h = torch.cat([h, self.task(task_id.long()), progress.float()], dim=-1)
        return self.head(h)


def load_inverse_dynamics(checkpoint: str | Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = VideoInverseDynamics(
        action_dim=int(cfg["action_dim"]),
        task_count=int(cfg["task_count"]),
        task_dim=int(cfg["task_dim"]),
        width=int(cfg["width"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return {
        "model": model,
        "config": cfg,
        "action_mean": torch.as_tensor(payload["action_mean"], dtype=torch.float32, device=device),
        "action_std": torch.as_tensor(payload["action_std"], dtype=torch.float32, device=device),
        "device": device,
    }
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
