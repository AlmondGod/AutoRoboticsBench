from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tasks.robocasa_world_model.model import RoboCasaWorldModel


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
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.decoder_type = str(decoder_type)
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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, 2 * self.latent_dim),
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
        current_rgb_conditioned: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.visual_latent_dim = int(visual_latent_dim)
        self.current_rgb_conditioned = bool(current_rgb_conditioned)
        self.dynamics = RoboCasaWorldModel(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            task_count=int(task_count),
            width=int(width),
            depth=int(depth),
            task_dim=int(task_dim),
            latent_dim=int(latent_dim),
            dropout=float(dropout),
        )
        self.image_vae = ImageVAE(
            image_size=int(image_size),
            latent_dim=int(visual_latent_dim),
            width=max(128, int(width) // 2),
            decoder_width=int(visual_decoder_width or max(128, int(width) // 2)),
            decoder_depth=int(visual_decoder_depth),
            decoder_type=str(visual_decoder_type),
            dropout=float(dropout),
        )
        visual_condition_dim = int(width) + (int(visual_latent_dim) if self.current_rgb_conditioned else 0)
        self.next_visual_latent = nn.Sequential(
            nn.Linear(visual_condition_dim, int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(visual_latent_dim)),
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
        sample_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.dynamics.encode_state(state, sample=sample_latent)
        h = torch.cat([z, action], dim=-1)
        return self.dynamics.trunk(h), z, mu, logvar

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        current_rgb: torch.Tensor | None = None,
        current_visual_latent: torch.Tensor | None = None,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden, z, mu, logvar = self.transition_hidden(
            state,
            action,
            sample_latent=sample_latent,
        )
        next_z = z + self.dynamics.delta(hidden)
        next_state = self.dynamics.decode_state(next_z)
        visual_condition = self.visual_condition(hidden, current_rgb=current_rgb, current_visual_latent=current_visual_latent)
        pred_visual_latent = self.next_visual_latent(visual_condition)
        next_rgb = self.image_vae.decode(pred_visual_latent)
        return {
            "next_state": next_state,
            "next_latent": next_z,
            "next_progress": torch.sigmoid(self.dynamics.progress(hidden)),
            "success_logit": self.dynamics.success(hidden),
            "next_rgb": next_rgb,
            "next_visual_latent": pred_visual_latent,
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
    ) -> torch.Tensor:
        if not self.current_rgb_conditioned:
            return hidden
        if current_visual_latent is None:
            if current_rgb is not None:
                current_visual_latent, _, _ = self.image_vae.encode(current_rgb, sample=False)
            else:
                current_visual_latent = torch.zeros(
                    (hidden.shape[0], self.visual_latent_dim),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
        return torch.cat([hidden, current_visual_latent.float()], dim=-1)

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
        kl_weight: float = 1e-4,
        visual_kl_weight: float = 1e-5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        next_visual, next_visual_mu, next_visual_logvar = self.image_vae.encode(batch["next_rgb"], sample=False)
        current_visual, current_visual_mu, current_visual_logvar = self.image_vae.encode(batch["rgb"], sample=False)
        out = self(
            batch["state"],
            batch["action"],
            current_visual_latent=current_visual.detach(),
            sample_latent=True,
        )
        state_loss = F.mse_loss(out["next_state"], batch["next_state"])
        progress_loss = F.mse_loss(out["next_progress"], batch["next_progress"])
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        rgb_loss = F.mse_loss(out["next_rgb"], batch["next_rgb"])
        visual_latent_loss = F.mse_loss(out["next_visual_latent"], next_visual.detach())

        current_recon = self.image_vae.decode(current_visual)
        next_recon = self.image_vae.decode(next_visual)
        image_vae_loss = 0.5 * (F.mse_loss(current_recon, batch["rgb"]) + F.mse_loss(next_recon, batch["next_rgb"]))
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
            "rgb_mse": rgb_loss.detach(),
            "visual_latent_mse": visual_latent_loss.detach(),
            "image_vae_mse": image_vae_loss.detach(),
            "kl": kl.detach(),
            "visual_kl": visual_kl.detach(),
        }


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
