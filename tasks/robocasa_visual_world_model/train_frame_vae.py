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

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

# Inlined from tasks/robocasa_visual_world_model/model.py.
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
    parser.add_argument("--max-train-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument("--eval-batches", type=int, default=0, help="0 evaluates the full validation set.")
    parser.add_argument("--lpips-batches", type=int, default=16)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")
    if float(args.max_train_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")

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
