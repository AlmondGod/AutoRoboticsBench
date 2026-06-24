from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

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
