# Inlined from tasks/robocasa_bc5/inference.py; keep this task inference.py self-contained.

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect

import numpy as np
import torch
from torch import nn

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

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

def device_from_arg(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)



ensure_robocasa_runtime()

# Inlined from tasks/robocasa_bc5/model.py; keep train.py/inference.py self-contained.
class RoboCasaTemporalChunkBC(nn.Module):
    """Legacy chunked BC policy kept for loading older benchmark checkpoints."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 512,
        dropout: float = 0.05,
        task_dim: int = 32,
    ) -> None:
        super().__init__()
        self.chunk_horizon = chunk_horizon
        self.action_dim = action_dim
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
        prop_width = max(128, width // 2)
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, prop_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(prop_width, prop_width),
            nn.SiLU(),
        )
        self.task = nn.Embedding(task_count, task_dim)
        self.head = nn.Sequential(
            nn.Linear(width + prop_width + task_dim, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, chunk_horizon * action_dim),
        )
        self.action_in = nn.Linear(chunk_horizon * action_dim, 2 * width)
        self.flow_time = nn.Sequential(
            nn.Linear(1, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, 2 * width),
        )
        self.flow_decoder = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, chunk_horizon * action_dim),
        )

    def encode_obs(
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
        proprio_feat = self.proprio(proprio)
        task_feat = self.task(task_id)
        return torch.cat([image_feat, proprio_feat, task_feat], dim=-1)

    def forward(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode_obs(agent, wrist, proprio, task_id)
        out = self.head(features)
        return out.reshape(agent.shape[0], self.chunk_horizon, self.action_dim)

    def flow_velocity(self, obs_h: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = action_t.shape[0]
        action_flat = action_t.reshape(batch, self.chunk_horizon * self.action_dim)
        t = t.reshape(batch, 1).to(dtype=obs_h.dtype, device=obs_h.device)
        h = self.head[0](obs_h)
        velocity = self.flow_decoder(h + self.action_in(action_flat) + self.flow_time(t))
        return velocity.reshape(batch, self.chunk_horizon, self.action_dim)

    def sample_flow(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 8,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_h = self.encode_obs(agent, wrist, proprio, task_id)
        if initial_noise is None:
            action = torch.zeros((obs_h.shape[0], self.chunk_horizon, self.action_dim), dtype=obs_h.dtype, device=obs_h.device)
        else:
            action = initial_noise.to(dtype=obs_h.dtype, device=obs_h.device)
        steps = max(1, int(steps))
        dt = 1.0 / steps
        for idx in range(steps):
            t = torch.full((obs_h.shape[0],), (idx + 0.5) * dt, dtype=obs_h.dtype, device=obs_h.device)
            action = action + dt * self.flow_velocity(obs_h, action, t)
        return action


class RoboCasaSequenceFlowPolicy(nn.Module):
    """Vision/proprio-conditioned rectified-flow action chunk policy."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 256,
        depth: int = 3,
        action_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)

        self.vision = nn.Sequential(
            nn.Conv2d(6, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
        )
        self.image_pos = nn.Parameter(torch.zeros(1, 16, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.task = nn.Embedding(task_count, width)
        self.context_norm = nn.LayerNorm(width)
        self.context_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )

        self.action_in = nn.Linear(action_dim, width)
        self.step = nn.Embedding(chunk_horizon, width)
        self.time = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_cond = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.flow_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        self.bc_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        nn.init.normal_(self.image_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def encode_obs(
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
        image = self.vision(torch.cat([agent, wrist], dim=1))
        image = image.flatten(2).transpose(1, 2) + self.image_pos
        prop = self.proprio(proprio).unsqueeze(1)
        task = self.task(task_id).unsqueeze(1)
        cls = self.cls.expand(agent.shape[0], -1, -1)
        tokens = torch.cat([cls, task, prop, image], dim=1)
        tokens = self.context_blocks(tokens)
        return self.context_norm(tokens[:, 0])

    def forward(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_obs(agent, wrist, proprio, task_id)
        return self.bc_action(context)

    def bc_action(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        action_t = torch.zeros(
            (batch, self.chunk_horizon, self.action_dim),
            dtype=context.dtype,
            device=context.device,
        )
        t = torch.ones((batch,), dtype=context.dtype, device=context.device)
        tokens = self._action_tokens(context, action_t, t)
        return self.bc_head(tokens)

    def flow_velocity(
        self,
        context: torch.Tensor,
        action_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._action_tokens(context, action_t, t)
        return self.flow_head(tokens)

    def sample_flow(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 8,
        start: str = "zero",
    ) -> torch.Tensor:
        context = self.encode_obs(agent, wrist, proprio, task_id)
        shape = (context.shape[0], self.chunk_horizon, self.action_dim)
        if start == "noise":
            action = torch.randn(shape, dtype=context.dtype, device=context.device)
        elif start == "bc":
            action = self.bc_action(context)
        else:
            action = torch.zeros(shape, dtype=context.dtype, device=context.device)
        steps = int(steps)
        if steps <= 0:
            return action
        dt = 1.0 / steps
        for idx in range(steps):
            t = torch.full((context.shape[0],), (idx + 0.5) * dt, dtype=context.dtype, device=context.device)
            action = action + dt * self.flow_velocity(context, action, t)
        return action

    def _action_tokens(self, context: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = action_t.shape
        step = torch.arange(horizon, dtype=torch.long, device=action_t.device).unsqueeze(0)
        action_tokens = self.action_in(action_t)
        action_tokens = action_tokens + self.step(step)
        action_tokens = action_tokens + self.time(t.reshape(batch, 1)).unsqueeze(1)
        cond = self.action_cond(context).unsqueeze(1)
        tokens = self.action_blocks(torch.cat([cond, action_tokens], dim=1))
        return tokens[:, 1:]


class RoboCasaHistoryACTPolicy(nn.Module):
    """ACT-style action chunk policy conditioned on previous and current observations."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 256,
        depth: int = 3,
        action_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)

        self.vision = nn.Sequential(
            nn.Conv2d(12, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
        )
        self.image_pos = nn.Parameter(torch.zeros(1, 16, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.task = nn.Embedding(task_count, width)
        self.context_norm = nn.LayerNorm(width)
        self.context_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_horizon, width))
        self.action_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        nn.init.normal_(self.image_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        queries = self.action_queries.expand(context.shape[0], -1, -1)
        tokens = self.action_blocks(torch.cat([context.unsqueeze(1), queries], dim=1))
        return self.head(tokens[:, 1:])

    def encode_obs(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        if prev_agent.max() > 1.5:
            prev_agent = prev_agent / 255.0
        if prev_wrist.max() > 1.5:
            prev_wrist = prev_wrist / 255.0
        image = self.vision(torch.cat([prev_agent, prev_wrist, agent, wrist], dim=1))
        image = image.flatten(2).transpose(1, 2) + self.image_pos
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1)).unsqueeze(1)
        task = self.task(task_id).unsqueeze(1)
        cls = self.cls.expand(agent.shape[0], -1, -1)
        tokens = torch.cat([cls, task, prop, image], dim=1)
        tokens = self.context_blocks(tokens)
        return self.context_norm(tokens[:, 0])


class RoboCasaPatchViTACTPolicy(nn.Module):
    """Patch-token ViT ACT policy with a conv patch embedding and long action queries."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 256,
        depth: int = 3,
        action_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.05,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        if 64 % patch_size != 0:
            raise ValueError(f"patch_size={patch_size} must divide 64")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)
        self.patch_size = int(patch_size)
        patch_count = (64 // int(patch_size)) ** 2

        self.patch_embed = nn.Conv2d(12, width, kernel_size=patch_size, stride=patch_size)
        self.patch_pos = nn.Parameter(torch.zeros(1, patch_count, width))
        self.task = nn.Embedding(task_count, width)
        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.obs_norm = nn.LayerNorm(width)
        self.obs_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_horizon, width))
        self.action_blocks = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        obs_tokens = self.encode_obs_tokens(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        queries = self.action_queries.expand(obs_tokens.shape[0], -1, -1)
        action_tokens = self.action_blocks(queries, obs_tokens)
        return self.head(action_tokens)

    def encode_obs_tokens(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        if prev_agent.max() > 1.5:
            prev_agent = prev_agent / 255.0
        if prev_wrist.max() > 1.5:
            prev_wrist = prev_wrist / 255.0
        image = self.patch_embed(torch.cat([prev_agent, prev_wrist, agent, wrist], dim=1))
        image = image.flatten(2).transpose(1, 2) + self.patch_pos
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1)).unsqueeze(1)
        task = self.task(task_id).unsqueeze(1)
        tokens = torch.cat([task, prop, image], dim=1)
        tokens = self.obs_blocks(tokens)
        return self.obs_norm(tokens)


class RoboCasaHistoryFlowPolicy(nn.Module):
    """History-conditioned rectified-flow action chunk policy."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 256,
        depth: int = 3,
        action_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)

        self.vision = nn.Sequential(
            nn.Conv2d(12, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
        )
        self.image_pos = nn.Parameter(torch.zeros(1, 16, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.task = nn.Embedding(task_count, width)
        self.context_norm = nn.LayerNorm(width)
        self.context_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )

        self.action_in = nn.Linear(action_dim, width)
        self.step = nn.Embedding(chunk_horizon, width)
        self.time = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_cond = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.flow_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        self.bc_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        nn.init.normal_(self.image_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        return self.bc_action(context)

    def encode_obs(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        if prev_agent.max() > 1.5:
            prev_agent = prev_agent / 255.0
        if prev_wrist.max() > 1.5:
            prev_wrist = prev_wrist / 255.0
        image = self.vision(torch.cat([prev_agent, prev_wrist, agent, wrist], dim=1))
        image = image.flatten(2).transpose(1, 2) + self.image_pos
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1)).unsqueeze(1)
        task = self.task(task_id).unsqueeze(1)
        cls = self.cls.expand(agent.shape[0], -1, -1)
        tokens = torch.cat([cls, task, prop, image], dim=1)
        tokens = self.context_blocks(tokens)
        return self.context_norm(tokens[:, 0])

    def bc_action(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        action_t = torch.zeros(
            (batch, self.chunk_horizon, self.action_dim),
            dtype=context.dtype,
            device=context.device,
        )
        t = torch.ones((batch,), dtype=context.dtype, device=context.device)
        tokens = self._action_tokens(context, action_t, t)
        return self.bc_head(tokens)

    def flow_velocity(self, context: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tokens = self._action_tokens(context, action_t, t)
        return self.flow_head(tokens)

    def sample_flow(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 8,
        start: str = "bc",
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        shape = (context.shape[0], self.chunk_horizon, self.action_dim)
        if start == "noise":
            action = torch.randn(shape, dtype=context.dtype, device=context.device)
        elif start == "zero":
            action = torch.zeros(shape, dtype=context.dtype, device=context.device)
        else:
            action = self.bc_action(context)
        steps = int(steps)
        if steps <= 0:
            return action
        dt = 1.0 / steps
        scale = float(residual_scale)
        for idx in range(steps):
            t = torch.full((context.shape[0],), (idx + 0.5) * dt, dtype=context.dtype, device=context.device)
            action = action + scale * dt * self.flow_velocity(context, action, t)
        return action

    def _action_tokens(self, context: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = action_t.shape
        step = torch.arange(horizon, dtype=torch.long, device=action_t.device).unsqueeze(0)
        action_tokens = self.action_in(action_t)
        action_tokens = action_tokens + self.step(step)
        action_tokens = action_tokens + self.time(t.reshape(batch, 1)).unsqueeze(1)
        cond = self.action_cond(context).unsqueeze(1)
        tokens = self.action_blocks(torch.cat([cond, action_tokens], dim=1))
        return tokens[:, 1:]


class RoboCasaHistoryACTFlowPolicy(RoboCasaHistoryACTPolicy):
    """ACT action-query policy with a rectified-flow residual action decoder."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 256,
        depth: int = 3,
        action_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            proprio_dim=proprio_dim,
            chunk_horizon=chunk_horizon,
            action_dim=action_dim,
            task_count=task_count,
            width=width,
            depth=depth,
            action_depth=action_depth,
            heads=heads,
            dropout=dropout,
        )
        self.flow_action_in = nn.Linear(action_dim, width)
        self.flow_step = nn.Embedding(chunk_horizon, width)
        self.flow_time = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.flow_cond = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.flow_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.flow_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

    def bc_action(self, context: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.expand(context.shape[0], -1, -1)
        tokens = self.action_blocks(torch.cat([context.unsqueeze(1), queries], dim=1))
        return self.head(tokens[:, 1:])

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        return self.bc_action(context)

    def flow_velocity(self, context: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = action_t.shape
        step = torch.arange(horizon, dtype=torch.long, device=action_t.device).unsqueeze(0)
        action_tokens = self.flow_action_in(action_t)
        action_tokens = action_tokens + self.flow_step(step)
        action_tokens = action_tokens + self.flow_time(t.reshape(batch, 1)).unsqueeze(1)
        cond = self.flow_cond(context).unsqueeze(1)
        tokens = self.flow_blocks(torch.cat([cond, action_tokens], dim=1))
        return self.flow_head(tokens[:, 1:])

    def sample_flow(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 8,
        start: str = "bc",
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        shape = (context.shape[0], self.chunk_horizon, self.action_dim)
        if start == "noise":
            action = torch.randn(shape, dtype=context.dtype, device=context.device)
        elif start == "zero":
            action = torch.zeros(shape, dtype=context.dtype, device=context.device)
        else:
            action = self.bc_action(context)
        steps = int(steps)
        if steps <= 0:
            return action
        dt = 1.0 / steps
        scale = float(residual_scale)
        for idx in range(steps):
            t = torch.full((context.shape[0],), (idx + 0.5) * dt, dtype=context.dtype, device=context.device)
            action = action + scale * dt * self.flow_velocity(context, action, t)
        return action


class RoboCasaMiniPi0Policy(nn.Module):
    """Small pi0-style policy with observation tokens and a separate flow action expert."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 256,
        depth: int = 3,
        action_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)

        self.vision = nn.Sequential(
            nn.Conv2d(12, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
        )
        self.image_pos = nn.Parameter(torch.zeros(1, 16, width))
        self.task = nn.Embedding(task_count, width)
        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.obs_norm = nn.LayerNorm(width)
        self.obs_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )

        self.action_in = nn.Linear(action_dim, width)
        self.step = nn.Embedding(chunk_horizon, width)
        self.time = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_norm = nn.LayerNorm(width)
        self.action_blocks = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.flow_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        nn.init.normal_(self.image_pos, std=0.02)

    def encode_obs_tokens(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        if prev_agent.max() > 1.5:
            prev_agent = prev_agent / 255.0
        if prev_wrist.max() > 1.5:
            prev_wrist = prev_wrist / 255.0
        image = self.vision(torch.cat([prev_agent, prev_wrist, agent, wrist], dim=1))
        image = image.flatten(2).transpose(1, 2) + self.image_pos
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1)).unsqueeze(1)
        task = self.task(task_id).unsqueeze(1)
        tokens = torch.cat([task, prop, image], dim=1)
        tokens = self.obs_blocks(tokens)
        return self.obs_norm(tokens)

    def flow_velocity(self, obs_tokens: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = action_t.shape
        step = torch.arange(horizon, dtype=torch.long, device=action_t.device).unsqueeze(0)
        action_tokens = self.action_in(action_t)
        action_tokens = action_tokens + self.step(step)
        action_tokens = action_tokens + self.time(t.reshape(batch, 1)).unsqueeze(1)
        action_tokens = self.action_norm(action_tokens)
        tokens = self.action_blocks(action_tokens, obs_tokens)
        return self.flow_head(tokens)

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample_flow(
            prev_agent,
            prev_wrist,
            agent,
            wrist,
            prev_proprio,
            proprio,
            task_id,
            steps=10,
            start="noise",
        )

    def sample_flow(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 10,
        start: str = "noise",
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        obs_tokens = self.encode_obs_tokens(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        shape = (obs_tokens.shape[0], self.chunk_horizon, self.action_dim)
        if start == "zero":
            action = torch.zeros(shape, dtype=obs_tokens.dtype, device=obs_tokens.device)
        else:
            action = torch.randn(shape, dtype=obs_tokens.dtype, device=obs_tokens.device)
        steps = int(steps)
        if steps <= 0:
            return action
        dt = 1.0 / steps
        scale = float(residual_scale)
        for idx in range(steps):
            t = torch.full((obs_tokens.shape[0],), (idx + 0.5) * dt, dtype=obs_tokens.dtype, device=obs_tokens.device)
            action = action + scale * dt * self.flow_velocity(obs_tokens, action, t)
        return action


class RoboCasaMiniPi0ResNetPolicy(RoboCasaMiniPi0Policy):
    """Mini pi0 variant with a frozen ImageNet ResNet18 visual encoder."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from torchvision.models import ResNet18_Weights, resnet18

        for param in self.vision.parameters():
            param.requires_grad = False
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        self.vision_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        self.image_proj = nn.Linear(512, self.width)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.vision_backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "vision_backbone"):
            self.vision_backbone.eval()
        return self

    def encode_obs_tokens(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        image = torch.cat(
            [
                self._image_tokens(prev_agent),
                self._image_tokens(prev_wrist),
                self._image_tokens(agent),
                self._image_tokens(wrist),
            ],
            dim=1,
        )
        image = image + self.image_pos[:, : image.shape[1]]
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1)).unsqueeze(1)
        task = self.task(task_id).unsqueeze(1)
        tokens = torch.cat([task, prop, image], dim=1)
        tokens = self.obs_blocks(tokens)
        return self.obs_norm(tokens)

    def _image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.max() > 1.5:
            image = image / 255.0
        image = (image - self.image_mean) / self.image_std
        with torch.no_grad():
            features = self.vision_backbone(image)
        tokens = features.flatten(2).transpose(1, 2)
        return self.image_proj(tokens)


class RoboCasaMiniPi0ACTPolicy(RoboCasaMiniPi0Policy):
    """Mini pi0-style deterministic action expert for short-budget BC."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.act_queries = nn.Parameter(torch.zeros(1, self.chunk_horizon, self.width))
        self.act_blocks = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.width,
                nhead=kwargs.get("heads", 4),
                dim_feedforward=4 * self.width,
                dropout=kwargs.get("dropout", 0.05),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=kwargs.get("action_depth", 3),
        )
        self.act_head = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, self.action_dim))
        nn.init.normal_(self.act_queries, std=0.02)
        self._freeze_flow_decoder()

    def _freeze_flow_decoder(self) -> None:
        for module in (self.action_in, self.step, self.time, self.action_norm, self.action_blocks, self.flow_head):
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        obs_tokens = self.encode_obs_tokens(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        queries = self.act_queries.expand(obs_tokens.shape[0], -1, -1)
        tokens = self.act_blocks(queries, obs_tokens)
        return self.act_head(tokens)


class RoboCasaMiniPi0ACTResNetPolicy(RoboCasaMiniPi0ResNetPolicy):
    """Deterministic mini pi0 action expert with frozen ImageNet ResNet18 tokens."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.act_queries = nn.Parameter(torch.zeros(1, self.chunk_horizon, self.width))
        self.act_blocks = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.width,
                nhead=kwargs.get("heads", 4),
                dim_feedforward=4 * self.width,
                dropout=kwargs.get("dropout", 0.05),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=kwargs.get("action_depth", 3),
        )
        self.act_head = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, self.action_dim))
        nn.init.normal_(self.act_queries, std=0.02)
        for module in (self.action_in, self.step, self.time, self.action_norm, self.action_blocks, self.flow_head):
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        obs_tokens = self.encode_obs_tokens(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        queries = self.act_queries.expand(obs_tokens.shape[0], -1, -1)
        tokens = self.act_blocks(queries, obs_tokens)
        return self.act_head(tokens)


class RoboCasaFrozenCLIPFlowPolicy(nn.Module):
    """Frozen CLIP image/text encoder with a small BC+flow action head."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        task_texts: list[str],
        encoder_name: str = "openai/clip-vit-base-patch32",
        width: int = 256,
        action_depth: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)
        self.encoder_name = str(encoder_name)
        self.task_count = int(task_count)
        self.task_texts = list(task_texts)

        self.clip, tokenizer = self._load_clip(self.encoder_name)
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False
        self.feature_dim = int(self.clip.config.projection_dim)
        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1))
        self.register_buffer("text_features", self._encode_task_texts(tokenizer, task_texts), persistent=False)

        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.visual = nn.Sequential(
            nn.Linear(5 * self.feature_dim, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.task = nn.Embedding(task_count, width)
        self.context = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )

        self.action_queries = nn.Parameter(torch.zeros(1, chunk_horizon, width))
        self.bc_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.bc_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        self.flow_action_in = nn.Linear(action_dim, width)
        self.flow_step = nn.Embedding(chunk_horizon, width)
        self.flow_time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_cond = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.flow_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        nn.init.normal_(self.action_queries, std=0.02)

    @staticmethod
    def _patch_transformers_sklearn() -> None:
        import transformers.utils as utils
        import transformers.utils.import_utils as import_utils

        utils.is_sklearn_available = lambda: False
        import_utils.is_sklearn_available = lambda: False

    @classmethod
    def _load_clip(cls, encoder_name: str):
        cls._patch_transformers_sklearn()
        from transformers.models.clip.modeling_clip import CLIPModel
        from transformers.models.clip.processing_clip import CLIPProcessor

        clip = CLIPModel.from_pretrained(encoder_name)
        processor = CLIPProcessor.from_pretrained(encoder_name)
        return clip, processor.tokenizer

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("clip.") and key not in {"image_mean", "image_std", "text_features"}
        }

    def load_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.max() > 1.5:
            images = images / 255.0
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.image_mean) / self.image_std
        with torch.no_grad():
            features = self.clip.get_image_features(pixel_values=images)
        features = self._feature_tensor(features)
        return F.normalize(features.float(), dim=-1)

    @staticmethod
    def _feature_tensor(features) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "image_embeds") and features.image_embeds is not None:
            return features.image_embeds
        if hasattr(features, "text_embeds") and features.text_embeds is not None:
            return features.text_embeds
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state[:, 0]
        raise TypeError(f"cannot extract feature tensor from {type(features)!r}")

    def _encode_task_texts(self, tokenizer, task_texts: list[str]) -> torch.Tensor:
        if len(task_texts) < self.task_count:
            task_texts = list(task_texts) + [f"robot task {idx}" for idx in range(len(task_texts), self.task_count)]
        encoded = tokenizer(
            task_texts[: self.task_count],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            features = self.clip.get_text_features(**encoded)
        features = self._feature_tensor(features)
        return F.normalize(features.float(), dim=-1)

    def context_from_features(
        self,
        image_features: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        batch = image_features.shape[0]
        text = self.text_features.to(device=image_features.device, dtype=image_features.dtype)[task_id]
        visual = self.visual(torch.cat([image_features.reshape(batch, -1), text], dim=-1))
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1))
        task = self.task(task_id)
        return self.context(torch.cat([visual, prop, task], dim=-1))

    def encode_obs(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        images = torch.cat([prev_agent, prev_wrist, agent, wrist], dim=0)
        features = self.encode_images(images).reshape(4, prev_agent.shape[0], -1).transpose(0, 1).contiguous()
        return self.context_from_features(features, prev_proprio, proprio, task_id)

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        return self.bc_action(context)

    def bc_action(self, context: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.expand(context.shape[0], -1, -1)
        tokens = self.bc_blocks(torch.cat([context.unsqueeze(1), queries], dim=1))
        return self.bc_head(tokens[:, 1:])

    def flow_velocity(self, context: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = action_t.shape
        step = torch.arange(horizon, dtype=torch.long, device=action_t.device).unsqueeze(0)
        action_tokens = self.flow_action_in(action_t)
        action_tokens = action_tokens + self.flow_step(step)
        action_tokens = action_tokens + self.flow_time(t.reshape(batch, 1)).unsqueeze(1)
        cond = self.flow_cond(context).unsqueeze(1)
        tokens = self.flow_blocks(torch.cat([cond, action_tokens], dim=1))
        return self.flow_head(tokens[:, 1:])

    def sample_flow(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 8,
        start: str = "bc",
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        shape = (context.shape[0], self.chunk_horizon, self.action_dim)
        if start == "noise":
            action = torch.randn(shape, dtype=context.dtype, device=context.device)
        elif start == "zero":
            action = torch.zeros(shape, dtype=context.dtype, device=context.device)
        else:
            action = self.bc_action(context)
        steps = int(steps)
        if steps <= 0:
            return action
        dt = 1.0 / steps
        scale = float(residual_scale)
        for idx in range(steps):
            t = torch.full((context.shape[0],), (idx + 0.5) * dt, dtype=context.dtype, device=context.device)
            action = action + scale * dt * self.flow_velocity(context, action, t)
        return action


class RoboCasaFrozenSmolVLMFlowPolicy(RoboCasaFrozenCLIPFlowPolicy):
    """Frozen SmolVLM2 image/text encoder with the same BC+flow action head."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        task_texts: list[str],
        encoder_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        width: int = 256,
        action_depth: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        nn.Module.__init__(self)
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)
        self.encoder_name = str(encoder_name)
        self.task_count = int(task_count)
        self.task_texts = list(task_texts)

        self.vlm, self.processor = self._load_vlm(self.encoder_name)
        self.vlm.eval()
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.feature_dim = self._infer_feature_dim(self.vlm.config)
        self.register_buffer("text_features", self._encode_task_texts(self.processor, task_texts), persistent=False)

        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.visual = nn.Sequential(
            nn.Linear(5 * self.feature_dim, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.task = nn.Embedding(task_count, width)
        self.context = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )

        self.action_queries = nn.Parameter(torch.zeros(1, chunk_horizon, width))
        self.bc_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.bc_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        self.flow_action_in = nn.Linear(action_dim, width)
        self.flow_step = nn.Embedding(chunk_horizon, width)
        self.flow_time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_cond = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.flow_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        nn.init.normal_(self.action_queries, std=0.02)

    @classmethod
    def _load_vlm(cls, encoder_name: str):
        cls._patch_transformers_sklearn()
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForMultimodalLM

            model_cls = AutoModelForMultimodalLM
        processor = AutoProcessor.from_pretrained(encoder_name)
        model = model_cls.from_pretrained(encoder_name, torch_dtype="auto")
        return model, processor

    @staticmethod
    def _infer_feature_dim(config) -> int:
        for path in (
            ("vision_config", "hidden_size"),
            ("text_config", "hidden_size"),
            ("hidden_size",),
        ):
            node = config
            for key in path:
                node = getattr(node, key, None)
                if node is None:
                    break
            if isinstance(node, int):
                return int(node)
        raise ValueError(f"could not infer SmolVLM hidden size from config={config!r}")

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.vlm.eval()
        return self

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.state_dict().items() if not key.startswith("vlm.")}

    def load_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.max() > 1.5:
            images = images / 255.0
        device = next(self.vlm.parameters()).device
        dtype = next(self.vlm.parameters()).dtype
        image_processor = getattr(self.processor, "image_processor", None)
        mean = torch.as_tensor(
            getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]),
            dtype=images.dtype,
            device=images.device,
        ).view(1, 3, 1, 1)
        std = torch.as_tensor(
            getattr(image_processor, "image_std", [0.5, 0.5, 0.5]),
            dtype=images.dtype,
            device=images.device,
        ).view(1, 3, 1, 1)
        pixel_values = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        pixel_values = ((pixel_values - mean) / std).to(device=device, dtype=dtype)
        with torch.no_grad():
            outputs = self.vlm.model.vision_model(pixel_values, patch_attention_mask=None, return_dict=True)
        features = outputs.last_hidden_state.to(dtype=torch.float32).mean(dim=1)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"SmolVLM feature dim changed: got {features.shape[-1]}, expected {self.feature_dim}")
        return F.normalize(features.float(), dim=-1)

    def _encode_task_texts(self, processor, task_texts: list[str]) -> torch.Tensor:
        if len(task_texts) < self.task_count:
            task_texts = list(task_texts) + [f"robot task {idx}" for idx in range(len(task_texts), self.task_count)]
        task_texts = task_texts[: self.task_count]

        tokenizer = getattr(processor, "tokenizer", processor)
        encoded = tokenizer(
            task_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.vlm.parameters()).device
        encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}

        with torch.no_grad():
            text_features = self._encode_text_with_vlm(encoded)
        text_features = self._match_feature_dim(text_features.float())
        return F.normalize(text_features, dim=-1)

    def _encode_text_with_vlm(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.vlm, "get_text_features"):
            return self.vlm.get_text_features(**encoded)

        text_model = getattr(getattr(self.vlm, "model", self.vlm), "text_model", None)
        if text_model is not None:
            outputs = text_model(
                input_ids=encoded.get("input_ids"),
                attention_mask=encoded.get("attention_mask"),
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            mask = encoded.get("attention_mask")
            if mask is None:
                return hidden.mean(dim=1)
            mask = mask.to(dtype=hidden.dtype).unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        embeddings = self.vlm.get_input_embeddings()(encoded["input_ids"])
        mask = encoded.get("attention_mask")
        if mask is None:
            return embeddings.mean(dim=1)
        mask = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _match_feature_dim(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] == self.feature_dim:
            return features
        if features.shape[-1] > self.feature_dim:
            return features[..., : self.feature_dim]
        pad = self.feature_dim - features.shape[-1]
        return F.pad(features, (0, pad))

    def context_from_features(
        self,
        image_features: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        batch = image_features.shape[0]
        text = self.text_features.to(device=image_features.device, dtype=image_features.dtype)[task_id]
        visual = self.visual(torch.cat([image_features.reshape(batch, -1), text], dim=-1))
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1))
        task = torch.zeros_like(prop)
        return self.context(torch.cat([visual, prop, task], dim=-1))

    @staticmethod
    def _tensor_to_pil(images: torch.Tensor):
        from PIL import Image

        images = images.detach().cpu().clamp(0.0, 1.0)
        arrays = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype("uint8")
        return [Image.fromarray(array) for array in arrays]

    def _image_prompt(self, text: str) -> str:
        token = getattr(self.processor, "image_token", "<image>")
        return f"{token}{text}"

    def _encode_processor_batch(self, prompts: list[str], images) -> torch.Tensor:
        device = next(self.vlm.parameters()).device
        dtype = next(self.vlm.parameters()).dtype
        kwargs = {
            "text": prompts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "do_image_splitting": False,
        }
        if images is not None:
            kwargs["images"] = [[image] for image in images]
        inputs = self.processor(**kwargs)
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        pixel_values = inputs["pixel_values"]
        patch_mask = None
        batch_size = pixel_values.shape[0]
        tile_count = 1
        if pixel_values.ndim == 5:
            batch_size, tile_count = pixel_values.shape[:2]
            pixel_values = pixel_values.reshape(batch_size * tile_count, *pixel_values.shape[2:])
        with torch.no_grad():
            outputs = self.vlm.model.vision_model(
                pixel_values,
                patch_attention_mask=patch_mask,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state
        hidden = hidden.to(dtype=torch.float32)
        pooled = hidden.mean(dim=1)
        if tile_count > 1:
            pooled = pooled.reshape(batch_size, tile_count, -1)
            pooled = pooled.mean(dim=1)
        if pooled.shape[-1] != self.feature_dim:
            raise ValueError(f"SmolVLM feature dim changed: got {pooled.shape[-1]}, expected {self.feature_dim}")
        return pooled.to(dtype=dtype)


class RoboCasaFrozenR3MFlowPolicy(nn.Module):
    """Frozen R3M visual encoder with the same small BC+flow action head."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        task_count: int,
        task_texts: list[str] | None = None,
        encoder_name: str = "resnet50",
        width: int = 256,
        action_depth: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.width = int(width)
        self.encoder_name = str(encoder_name)
        self.task_count = int(task_count)
        self.task_texts = list(task_texts or [])

        self.r3m = self._load_r3m(self.encoder_name)
        self.r3m.eval()
        for param in self.r3m.parameters():
            param.requires_grad = False
        self.feature_dim = self._infer_feature_dim()

        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.visual = nn.Sequential(
            nn.Linear(4 * self.feature_dim, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.task = nn.Embedding(task_count, width)
        self.context = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )

        self.action_queries = nn.Parameter(torch.zeros(1, chunk_horizon, width))
        self.bc_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.bc_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

        self.flow_action_in = nn.Linear(action_dim, width)
        self.flow_step = nn.Embedding(chunk_horizon, width)
        self.flow_time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_cond = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=action_depth,
        )
        self.flow_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        nn.init.normal_(self.action_queries, std=0.02)

    @staticmethod
    def _load_r3m(encoder_name: str) -> nn.Module:
        from r3m import load_r3m

        return load_r3m(encoder_name)

    def _infer_feature_dim(self) -> int:
        outdim = getattr(getattr(self.r3m, "module", self.r3m), "outdim", None)
        if outdim is not None:
            return int(outdim)
        device = next(self.r3m.parameters()).device
        with torch.no_grad():
            features = self.r3m(torch.zeros(1, 3, 224, 224, device=device))
        return int(features.shape[-1])

    def train(self, mode: bool = True):
        super().train(mode)
        self.r3m.eval()
        return self

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.state_dict().items() if not key.startswith("r3m.")}

    def load_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.max() <= 1.5:
            images = images * 255.0
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = self.r3m(images)
        return features.float()

    def context_from_features(
        self,
        image_features: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        batch = image_features.shape[0]
        visual = self.visual(image_features.reshape(batch, -1))
        prop = self.proprio(torch.cat([prev_proprio, proprio, proprio - prev_proprio], dim=-1))
        task = self.task(task_id)
        return self.context(torch.cat([visual, prop, task], dim=-1))

    def encode_obs(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        images = torch.cat([prev_agent, prev_wrist, agent, wrist], dim=0)
        features = self.encode_images(images).reshape(4, prev_agent.shape[0], -1).transpose(0, 1).contiguous()
        return self.context_from_features(features, prev_proprio, proprio, task_id)

    def forward(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        return self.bc_action(context)

    def bc_action(self, context: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.expand(context.shape[0], -1, -1)
        tokens = self.bc_blocks(torch.cat([context.unsqueeze(1), queries], dim=1))
        return self.bc_head(tokens[:, 1:])

    def flow_velocity(self, context: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = action_t.shape
        step = torch.arange(horizon, dtype=torch.long, device=action_t.device).unsqueeze(0)
        action_tokens = self.flow_action_in(action_t)
        action_tokens = action_tokens + self.flow_step(step)
        action_tokens = action_tokens + self.flow_time(t.reshape(batch, 1)).unsqueeze(1)
        cond = self.flow_cond(context).unsqueeze(1)
        tokens = self.flow_blocks(torch.cat([cond, action_tokens], dim=1))
        return self.flow_head(tokens[:, 1:])

    def sample_flow(
        self,
        prev_agent: torch.Tensor,
        prev_wrist: torch.Tensor,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        prev_proprio: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        *,
        steps: int = 8,
        start: str = "bc",
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        context = self.encode_obs(prev_agent, prev_wrist, agent, wrist, prev_proprio, proprio, task_id)
        shape = (context.shape[0], self.chunk_horizon, self.action_dim)
        if start == "noise":
            action = torch.randn(shape, dtype=context.dtype, device=context.device)
        elif start == "zero":
            action = torch.zeros(shape, dtype=context.dtype, device=context.device)
        else:
            action = self.bc_action(context)
        steps = int(steps)
        if steps <= 0:
            return action
        dt = 1.0 / steps
        scale = float(residual_scale)
        for idx in range(steps):
            t = torch.full((context.shape[0],), (idx + 0.5) * dt, dtype=context.dtype, device=context.device)
            action = action + scale * dt * self.flow_velocity(context, action, t)
        return action



FORBIDDEN_POLICY_TYPES = {"robocasa_bc5_trajectory_bank"}
FORBIDDEN_REPLAY_KEYS = {"actions", "lengths", "task_ids", "episode_ids", "embeddings"}


@dataclass
class Policy:
    model: Any
    checkpoint: dict
    device: torch.device
    action_mean: torch.Tensor
    action_std: torch.Tensor
    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor
    mode: str = "chunk"
    cursor: int = 0
    selected_bank: int | None = None
    selected_task_id: int | None = None
    selected_episode_id: int | None = None
    last_proprio: np.ndarray | None = None
    prev_agent: np.ndarray | None = None
    prev_wrist: np.ndarray | None = None
    prev_proprio: np.ndarray | None = None
    history_task_id: int | None = None
    history_episode_id: int | None = None
    history_step_idx: int = 0


def load_policy(checkpoint: str, device: str = "auto") -> Policy:
    """Load exactly one policy checkpoint for use across all BC-5 tasks."""
    torch_device = device_from_arg(device)
    payload = torch.load(Path(checkpoint), map_location=torch_device, weights_only=False)
    policy_type = str(payload.get("policy_type", ""))
    replay_keys = sorted(FORBIDDEN_REPLAY_KEYS.intersection(payload))
    if policy_type in FORBIDDEN_POLICY_TYPES or replay_keys:
        detail = f"policy_type={policy_type!r}, replay_keys={replay_keys}"
        raise ValueError(
            "BC5 inference forbids trajectory replay or stored per-episode "
            f"action/data banks at eval time ({detail}). Submit learned model "
            "weights/statistics only."
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_sequence_flow":
        model = RoboCasaSequenceFlowPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="sequence_flow",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_history_act":
        model = RoboCasaHistoryACTPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="history_act",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_history_act_flow":
        model = RoboCasaHistoryACTFlowPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="history_act_flow",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_frozen_clip_flow":
        model = RoboCasaFrozenCLIPFlowPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            task_texts=list(payload.get("task_texts", [])),
            encoder_name=str(payload.get("vlm_encoder_name", "openai/clip-vit-base-patch32")),
            width=int(payload.get("width", 256)),
            action_depth=int(payload.get("action_depth", 2)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_head_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="frozen_clip_flow",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_frozen_smolvlm_flow":
        model = RoboCasaFrozenSmolVLMFlowPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            task_texts=list(payload.get("task_texts", [])),
            encoder_name=str(payload.get("vlm_encoder_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")),
            width=int(payload.get("width", 256)),
            action_depth=int(payload.get("action_depth", 2)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_head_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="frozen_smolvlm_flow",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_frozen_r3m_flow":
        model = RoboCasaFrozenR3MFlowPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            task_texts=list(payload.get("task_texts", [])),
            encoder_name=str(payload.get("r3m_encoder_name", "resnet50")),
            width=int(payload.get("width", 256)),
            action_depth=int(payload.get("action_depth", 2)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_head_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="frozen_r3m_flow",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_mini_pi0":
        model = RoboCasaMiniPi0Policy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="mini_pi0",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_mini_pi0_act":
        model = RoboCasaMiniPi0ACTPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="mini_pi0_act",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_mini_pi0_act_resnet":
        model = RoboCasaMiniPi0ACTResNetPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="mini_pi0_act_resnet",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_mini_pi0_resnet":
        model = RoboCasaMiniPi0ResNetPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="mini_pi0_resnet",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_vit_act":
        model = RoboCasaPatchViTACTPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
            patch_size=int(payload.get("patch_size", 8) or 8),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="vit_act",
        )
    if payload.get("policy_type") == "autorobobench_robocasa_bc5_history_flow":
        model = RoboCasaHistoryFlowPolicy(
            proprio_dim=int(payload["proprio_dim"]),
            chunk_horizon=int(payload["chunk_horizon"]),
            action_dim=int(payload["action_dim"]),
            task_count=int(payload["task_count"]),
            width=int(payload.get("width", 256)),
            depth=int(payload.get("transformer_depth", 3)),
            action_depth=int(payload.get("action_depth", 3)),
            heads=int(payload.get("heads", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return Policy(
            model=model,
            checkpoint=payload,
            device=torch_device,
            action_mean=_tensor(payload, "action_mean", torch_device),
            action_std=_tensor(payload, "action_std", torch_device),
            proprio_mean=_tensor(payload, "proprio_mean", torch_device),
            proprio_std=_tensor(payload, "proprio_std", torch_device),
            mode="history_flow",
        )
    model = RoboCasaTemporalChunkBC(
        proprio_dim=int(payload["proprio_dim"]),
        chunk_horizon=int(payload["chunk_horizon"]),
        action_dim=int(payload["action_dim"]),
        task_count=int(payload["task_count"]),
        width=int(payload.get("width", 512)),
        dropout=float(payload.get("dropout", 0.0)),
    ).to(torch_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return Policy(
        model=model,
        checkpoint=payload,
        device=torch_device,
        action_mean=_tensor(payload, "action_mean", torch_device),
        action_std=_tensor(payload, "action_std", torch_device),
        proprio_mean=_tensor(payload, "proprio_mean", torch_device),
        proprio_std=_tensor(payload, "proprio_std", torch_device),
    )


def act(policy: Policy, obs: dict, task: dict) -> np.ndarray:
    """Return a chunk of actions for the current observation and task.

    `obs` contains raw `agent` and `wrist` RGB uint8 images plus raw proprio.
    `task` contains the frozen BC-5 task id, alias, and language text.
    """
    device = policy.device
    task_id = int(task["task_id"])
    if task_id < 0 or task_id >= int(policy.checkpoint["task_count"]):
        raise ValueError(f"task_id={task_id} outside loaded policy task_count={policy.checkpoint['task_count']}")
    if policy.mode in {
        "history_act",
        "history_flow",
        "history_act_flow",
        "frozen_clip_flow",
        "frozen_r3m_flow",
        "frozen_smolvlm_flow",
        "mini_pi0_act",
        "mini_pi0_act_resnet",
        "mini_pi0",
        "mini_pi0_resnet",
        "vit_act",
    }:
        return _act_history(policy, obs, task_id)

    with torch.no_grad():
        agent_t = torch.as_tensor(np.asarray(obs["agent"])[None].copy(), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        wrist_t = torch.as_tensor(np.asarray(obs["wrist"])[None].copy(), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        progress = _non_history_step_idx(policy)
        proprio = _maybe_append_progress(policy.checkpoint, np.asarray(obs["proprio"], dtype=np.float32), progress)
        proprio_t = torch.as_tensor(proprio[None], dtype=torch.float32, device=device)
        proprio_t = (proprio_t - policy.proprio_mean) / policy.proprio_std
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        if str(policy.checkpoint.get("policy_kind", "bc")) == "flow":
            pred_norm = policy.model.sample_flow(
                agent_t,
                wrist_t,
                proprio_t,
                task_t,
                steps=int(policy.checkpoint.get("flow_steps", 8)),
            )[0]
        elif policy.mode == "sequence_flow":
            pred_norm = policy.model.sample_flow(
                agent_t,
                wrist_t,
                proprio_t,
                task_t,
                steps=int(policy.checkpoint.get("flow_steps", 8)),
                start=_flow_inference_start(policy.checkpoint),
            )[0]
        else:
            pred_norm = policy.model(agent_t, wrist_t, proprio_t, task_t)[0]
        pred = _denormalize_action(policy, pred_norm, task_id)
    out = _slice_return_horizon(policy, pred.detach().cpu().numpy().astype(np.float32), task_id)
    policy.history_step_idx = int(policy.history_step_idx) + int(out.shape[0])
    return out


def commit_steps(
    policy: Policy,
    *,
    task: dict | None = None,
    action_chunk: np.ndarray | None = None,
    default_commit_steps: int = 16,
) -> int:
    checkpoint = policy.checkpoint
    task_id = int(task["task_id"]) if task is not None and "task_id" in task else None
    by_task = checkpoint.get("eval_commit_steps_by_task")
    if by_task is not None and task_id is not None:
        try:
            return int(by_task[task_id])
        except (IndexError, KeyError, TypeError):
            pass
    if checkpoint.get("eval_commit_steps") is not None:
        return int(checkpoint["eval_commit_steps"])
    if checkpoint.get("return_horizon_by_task") is not None and task_id is not None:
        try:
            return int(checkpoint["return_horizon_by_task"][task_id])
        except (IndexError, KeyError, TypeError):
            pass
    if checkpoint.get("return_horizon") is not None:
        return int(checkpoint["return_horizon"])
    if action_chunk is not None:
        return int(min(default_commit_steps, action_chunk.shape[0]))
    return int(default_commit_steps)


def _act_history(policy: Policy, obs: dict, task_id: int) -> np.ndarray:
    episode_id = _current_eval_episode_id()
    agent = np.asarray(obs["agent"], dtype=np.uint8).copy()
    wrist = np.asarray(obs["wrist"], dtype=np.uint8).copy()
    proprio = np.asarray(obs["proprio"], dtype=np.float32).copy()
    reset = (
        policy.prev_agent is None
        or policy.prev_wrist is None
        or policy.prev_proprio is None
        or policy.history_task_id != task_id
        or (episode_id is not None and policy.history_episode_id != int(episode_id))
    )
    if reset:
        policy.prev_agent = agent
        policy.prev_wrist = wrist
        policy.prev_proprio = proprio
        policy.history_task_id = task_id
        policy.history_episode_id = int(episode_id) if episode_id is not None else None
        policy.history_step_idx = 0

    device = policy.device
    with torch.no_grad():
        prev_progress = max(0, int(policy.history_step_idx) - int(policy.checkpoint.get("eval_commit_steps", 16)))
        curr_progress = int(policy.history_step_idx)
        prev_proprio = _maybe_append_progress(policy.checkpoint, policy.prev_proprio, prev_progress)
        curr_proprio = _maybe_append_progress(policy.checkpoint, proprio, curr_progress)
        prev_agent_t = torch.as_tensor(policy.prev_agent[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        prev_wrist_t = torch.as_tensor(policy.prev_wrist[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        agent_t = torch.as_tensor(agent[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        wrist_t = torch.as_tensor(wrist[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        prev_proprio_t = torch.as_tensor(prev_proprio[None], dtype=torch.float32, device=device)
        proprio_t = torch.as_tensor(curr_proprio[None], dtype=torch.float32, device=device)
        prev_proprio_t = (prev_proprio_t - policy.proprio_mean) / policy.proprio_std
        proprio_t = (proprio_t - policy.proprio_mean) / policy.proprio_std
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        if policy.mode in {"history_flow", "history_act_flow", "frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow", "mini_pi0", "mini_pi0_resnet"}:
            pred_norm = policy.model.sample_flow(
                prev_agent_t,
                prev_wrist_t,
                agent_t,
                wrist_t,
                prev_proprio_t,
                proprio_t,
                task_t,
                steps=int(policy.checkpoint.get("flow_steps", 8)),
                start=_flow_inference_start(policy.checkpoint),
                residual_scale=float(policy.checkpoint.get("flow_residual_scale", 1.0)),
            )[0]
        else:
            pred_norm = policy.model(
                prev_agent_t,
                prev_wrist_t,
                agent_t,
                wrist_t,
                prev_proprio_t,
                proprio_t,
                task_t,
            )[0]
        pred = _denormalize_action(policy, pred_norm, task_id)

    policy.prev_agent = agent
    policy.prev_wrist = wrist
    policy.prev_proprio = proprio
    policy.history_task_id = task_id
    policy.history_episode_id = int(episode_id) if episode_id is not None else None
    out = _slice_return_horizon(policy, pred.detach().cpu().numpy().astype(np.float32), task_id)
    policy.history_step_idx = int(policy.history_step_idx) + int(out.shape[0])
    return out


def _flow_inference_start(checkpoint: dict) -> str:
    explicit = checkpoint.get("flow_inference_start")
    if explicit is not None:
        return str(explicit)
    if str(checkpoint.get("flow_source", "")) == "noise":
        return "noise"
    return str(checkpoint.get("flow_eval_start", "bc"))


def _maybe_append_progress(checkpoint: dict, proprio: np.ndarray, frame_idx: int) -> np.ndarray:
    proprio = np.asarray(proprio, dtype=np.float32)
    if not checkpoint.get("progress_conditioning"):
        return proprio
    scale = float(checkpoint.get("progress_scale", 260.0))
    progress = np.clip(float(frame_idx) / max(scale, 1.0), 0.0, 1.5)
    features = np.asarray(
        [
            progress,
            progress * progress,
            np.sin(np.pi * progress),
            np.cos(np.pi * progress),
        ],
        dtype=np.float32,
    )
    return np.concatenate([proprio, features], axis=-1).astype(np.float32)


def _denormalize_action(policy: Policy, pred_norm: torch.Tensor, task_id: int) -> torch.Tensor:
    if policy.checkpoint.get("task_action_normalization"):
        means = np.asarray(policy.checkpoint["task_action_mean"], dtype=np.float32)
        stds = np.asarray(policy.checkpoint["task_action_std"], dtype=np.float32)
        mean = torch.as_tensor(means[int(task_id)], dtype=pred_norm.dtype, device=policy.device)
        std = torch.as_tensor(stds[int(task_id)], dtype=pred_norm.dtype, device=policy.device)
        return pred_norm * std + mean
    return pred_norm * policy.action_std + policy.action_mean


def _slice_return_horizon(policy: Policy, actions: np.ndarray, task_id: int | None = None) -> np.ndarray:
    horizon_by_task = policy.checkpoint.get("return_horizon_by_task")
    default_horizon = policy.checkpoint.get("return_horizon")
    if default_horizon is None:
        default_horizon = policy.checkpoint.get("eval_commit_steps", actions.shape[0])
    if horizon_by_task is not None and task_id is not None:
        if isinstance(horizon_by_task, dict):
            horizon = int(horizon_by_task.get(str(int(task_id)), horizon_by_task.get(int(task_id), default_horizon)))
        else:
            horizon = int(horizon_by_task[int(task_id)])
    else:
        horizon = int(default_horizon)
    horizon = max(1, min(horizon, int(actions.shape[0])))
    return actions[:horizon].astype(np.float32)


def _non_history_step_idx(policy: Policy) -> int:
    episode_id = _current_eval_episode_id()
    if policy.history_episode_id is None or (episode_id is not None and policy.history_episode_id != int(episode_id)):
        policy.history_episode_id = int(episode_id) if episode_id is not None else None
        policy.history_step_idx = 0
    return int(policy.history_step_idx)


def _tensor(checkpoint: dict, key: str, device: torch.device) -> torch.Tensor:
    value = checkpoint[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _current_eval_episode_id() -> int | None:
    frame = inspect.currentframe()
    while frame is not None:
        if "episode_idx" in frame.f_locals:
            try:
                return int(frame.f_locals["episode_idx"])
            except (TypeError, ValueError):
                return None
        frame = frame.f_back
    return None
