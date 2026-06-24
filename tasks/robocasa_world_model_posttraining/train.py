from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

# Inlined from tasks/robocasa_bc5/inference.py; keep this train.py self-contained.

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect

import numpy as np
import torch

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


# Inlined from tasks/robocasa_bc5/train.py; keep this train.py self-contained.

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

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


class TemporalChunkData:
    agent: np.ndarray
    wrist: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    task_id: np.ndarray
    episode_idx: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.agent.shape[0])


def _episode_samples(
    dataset_root: Path,
    episode_path: Path,
    episode_idx: int,
    task_id: int,
    chunk_horizon: int,
    frame_stride: int,
    condition_on_robocasa_task_index: bool,
) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(episode_path)
    robocasa_task_index = int(frame["task_index"].iloc[0])
    sample_task_id = robocasa_task_index if condition_on_robocasa_task_index else task_id
    agent = _read_video64(dataset_root, episode_idx, "robot0_agentview_left")
    wrist = _read_video64(dataset_root, episode_idx, "robot0_agentview_right")
    proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    actions = LU.get_episode_actions(dataset_root, episode_idx).astype(np.float32)
    n = min(len(agent), len(wrist), len(proprio), len(actions))
    starts = np.arange(0, n, max(1, frame_stride), dtype=np.int32)

    out_actions = np.zeros((len(starts), chunk_horizon, actions.shape[-1]), dtype=np.float32)
    mask = np.zeros((len(starts), chunk_horizon), dtype=np.float32)
    for row_idx, start in enumerate(starts):
        end = min(n, int(start) + chunk_horizon)
        length = end - int(start)
        out_actions[row_idx, :length] = actions[int(start) : end]
        mask[row_idx, :length] = 1.0

    return {
        "agent": agent[starts],
        "wrist": wrist[starts],
        "proprio": proprio[starts],
        "actions": out_actions,
        "mask": mask,
        "task_id": np.full((len(starts),), sample_task_id, dtype=np.int64),
        "episode_idx": np.full((len(starts),), episode_idx, dtype=np.int32),
        "frame_idx": starts.astype(np.int32),
    }


def _read_video64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frames = [_resize64(np.asarray(frame, dtype=np.uint8)) for frame in iio.imiter(video_path)]
    return np.stack(frames).astype(np.uint8)


def _resize64(image: np.ndarray) -> np.ndarray:
    if image.shape[0] == 64 and image.shape[1] == 64:
        return image[..., :3]
    return np.asarray(Image.fromarray(image[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def _concat_parts(parts: list[dict[str, np.ndarray]]) -> TemporalChunkData:
    if not parts:
        return TemporalChunkData(
            agent=np.zeros((0, 64, 64, 3), dtype=np.uint8),
            wrist=np.zeros((0, 64, 64, 3), dtype=np.uint8),
            proprio=np.zeros((0, 16), dtype=np.float32),
            actions=np.zeros((0, 1, 12), dtype=np.float32),
            mask=np.zeros((0, 1), dtype=np.float32),
            task_id=np.zeros((0,), dtype=np.int64),
            episode_idx=np.zeros((0,), dtype=np.int32),
            frame_idx=np.zeros((0,), dtype=np.int32),
        )
    return TemporalChunkData(
        agent=np.concatenate([part["agent"] for part in parts], axis=0),
        wrist=np.concatenate([part["wrist"] for part in parts], axis=0),
        proprio=np.concatenate([part["proprio"] for part in parts], axis=0),
        actions=np.concatenate([part["actions"] for part in parts], axis=0),
        mask=np.concatenate([part["mask"] for part in parts], axis=0),
        task_id=np.concatenate([part["task_id"] for part in parts], axis=0),
        episode_idx=np.concatenate([part["episode_idx"] for part in parts], axis=0),
        frame_idx=np.concatenate([part["frame_idx"] for part in parts], axis=0),
    )


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def _masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1])
    keep = mask.reshape(-1) > 0
    valid = flat[keep]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def _batch(data: TemporalChunkData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent": torch.as_tensor(data.agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(data.wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data.actions[idx], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(data.mask[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


def _augment(batch: dict[str, torch.Tensor], image_noise: float, proprio_noise: float) -> dict[str, torch.Tensor]:
    if image_noise > 0:
        scale = 255.0 * image_noise
        batch["agent"] = (batch["agent"] + torch.randn_like(batch["agent"]) * scale).clamp(0.0, 255.0)
        batch["wrist"] = (batch["wrist"] + torch.randn_like(batch["wrist"]) * scale).clamp(0.0, 255.0)
    if proprio_noise > 0:
        batch["proprio"] = batch["proprio"] + torch.randn_like(batch["proprio"]) * proprio_noise
    return batch


def _eval_loss(
    model: RoboCasaTemporalChunkBC,
    data: TemporalChunkData,
    device: torch.device,
    batch_size: int,
    *,
    policy_kind: str = "bc",
    flow_steps: int = 8,
) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            if policy_kind == "flow":
                pred = model.sample_flow(
                    batch["agent"],
                    batch["wrist"],
                    batch["proprio"],
                    batch["task_id"],
                    steps=flow_steps,
                )
            else:
                pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _masked_chunk_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, chunk_decay: float = 1.0) -> torch.Tensor:
    per_step = (pred - target).square().mean(dim=-1)
    weights = _chunk_weights(pred.shape[1], chunk_decay, pred.device, pred.dtype)
    return (per_step * mask * weights).sum() / (mask * weights).sum().clamp_min(1.0)


def _flow_matching_loss(
    model: RoboCasaTemporalChunkBC,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    chunk_decay: float,
) -> torch.Tensor:
    actions = batch["actions"]
    noise = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    view_t = t.reshape(-1, 1, 1)
    action_t = (1.0 - view_t) * noise + view_t * actions
    target_velocity = actions - noise
    obs_h = model.encode_obs(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
    pred_velocity = model.flow_velocity(obs_h, action_t, t)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    return (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)


def _chunk_weights(horizon: int, decay: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.ones((horizon,), dtype=dtype, device=device)
    if decay != 1.0:
        idx = torch.arange(horizon, dtype=dtype, device=device)
        weights = decay**idx
    return weights.reshape(1, horizon) / weights.mean().clamp_min(1e-6)


def _shared_bc5_train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the AutoroboBench RoboCasa BC-5 baseline policy.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--video-pool", default="")
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_bc5/baseline")
    parser.add_argument("--train-episodes-per-task", type=int, default=4)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-noise", type=float, default=0.01)
    parser.add_argument("--proprio-noise", type=float, default=0.01)
    parser.add_argument("--action-smooth", type=float, default=0.001)
    parser.add_argument(
        "--policy-kind",
        choices=[
            "bc",
            "flow",
            "sequence_flow",
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
        ],
        default="bc",
    )
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-sigma", type=float, default=1.0)
    parser.add_argument("--flow-source", choices=["noise", "bc"], default="noise")
    parser.add_argument("--flow-eval-start", choices=["zero", "noise", "bc"], default="noise")
    parser.add_argument("--flow-residual-scale", type=float, default=1.0)
    parser.add_argument("--flow-time-sampling", choices=["uniform", "beta_low_noise"], default="uniform")
    parser.add_argument("--bc-aux-weight", type=float, default=0.1)
    parser.add_argument("--vlm-encoder-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--r3m-encoder-name", default="resnet50")
    parser.add_argument("--vlm-cache-batch-size", type=int, default=32)
    parser.add_argument("--frozen-feature-cache-dir", default="data/autorobobench/feature_cache")
    parser.add_argument("--chunk-decay", type=float, default=1.0)
    parser.add_argument("--transformer-depth", type=int, default=3)
    parser.add_argument("--action-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--history-stride", type=int, default=16)
    parser.add_argument("--progress-conditioning", action="store_true")
    parser.add_argument("--progress-scale", type=float, default=260.0)
    parser.add_argument("--task-action-normalization", action="store_true")
    parser.add_argument("--eval-commit-steps", type=int, default=16)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--freeze-non-flow", action="store_true")
    parser.add_argument("--video-pretrain-seconds", type=float, default=0.0)
    parser.add_argument("--video-pretrain-episodes-per-task", type=int, default=0)
    parser.add_argument("--video-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--video-pretrain-gap", type=int, default=8)
    parser.add_argument("--video-pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--video-pretrain-temperature", type=float, default=0.1)
    parser.add_argument(
        "--video-pretrain-objective",
        choices=["temporal_infonce", "latent_dynamics", "hybrid"],
        default="temporal_infonce",
    )
    parser.add_argument("--video-latent-dynamics-weight", type=float, default=1.0)
    parser.add_argument("--pidm-pretrain-seconds", type=float, default=0.0)
    parser.add_argument("--pidm-video-episodes-per-task", type=int, default=0)
    parser.add_argument("--pidm-batch-size", type=int, default=128)
    parser.add_argument("--pidm-lr", type=float, default=3e-4)
    parser.add_argument("--pidm-gap", type=int, default=4)
    parser.add_argument("--pidm-action-weight", type=float, default=1.0)
    parser.add_argument("--pidm-latent-weight", type=float, default=1.0)
    parser.add_argument("--vpt-idm-seconds", type=float, default=0.0)
    parser.add_argument("--vpt-pseudo-episodes-per-task", type=int, default=0)
    parser.add_argument("--vpt-idm-batch-size", type=int, default=128)
    parser.add_argument("--vpt-idm-lr", type=float, default=3e-4)
    parser.add_argument("--vpt-pseudo-weight", type=float, default=1.0)
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")
    if float(args.max_train_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")
    if float(args.pidm_pretrain_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--pidm-pretrain-seconds is fixed at 300 for scored runs and cannot be overwritten")
    if float(args.video_pretrain_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--video-pretrain-seconds is fixed at 300 for scored runs and cannot be overwritten")

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task_aliases = set(args.task_alias)
    train_data, val_data, split_summary = load_split_data(
        manifest,
        split,
        task_aliases=task_aliases,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("need both train and val samples for RoboCasa BC-5")

    vpt_metrics = _maybe_add_vpt_pseudo_labels(
        train_data=train_data,
        manifest=manifest,
        split=split,
        task_aliases=task_aliases,
        idm_seconds=float(args.vpt_idm_seconds),
        pseudo_episodes_per_task=int(args.vpt_pseudo_episodes_per_task),
        batch_size=int(args.vpt_idm_batch_size),
        lr=float(args.vpt_idm_lr),
        pseudo_weight=float(args.vpt_pseudo_weight),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
        seed=int(args.seed),
        device=device_from_arg(args.device),
    )

    raw_proprio_dim = int(train_data.proprio.shape[-1])
    if args.progress_conditioning:
        _append_progress_features(train_data, float(args.progress_scale))
        _append_progress_features(val_data, float(args.progress_scale))

    proprio_mean, proprio_std = _mean_std(train_data.proprio)
    action_mean, action_std = _weighted_masked_mean_std(train_data.actions, train_data.mask)
    task_action_mean = None
    task_action_std = None
    if args.task_action_normalization:
        task_action_mean, task_action_std = _per_task_action_stats(train_data)
    train_data.proprio = ((train_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val_data.proprio = ((val_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    if args.task_action_normalization:
        train_data.actions = _normalize_actions_by_task(
            train_data.actions,
            train_data.task_id,
            task_action_mean,
            task_action_std,
        )
        val_data.actions = _normalize_actions_by_task(
            val_data.actions,
            val_data.task_id,
            task_action_mean,
            task_action_std,
        )
    else:
        train_data.actions = ((train_data.actions - action_mean) / action_std).astype(np.float32)
        val_data.actions = ((val_data.actions - action_mean) / action_std).astype(np.float32)
    if args.policy_kind in {
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
        _attach_history(train_data, int(args.history_stride))
        _attach_history(val_data, int(args.history_stride))

    device = device_from_arg(args.device)
    task_count = max(1, int(max(train_data.task_id.max(initial=0), val_data.task_id.max(initial=0)) + 1))
    task_texts = _task_texts_for_split(manifest, split, task_aliases)
    if args.policy_kind == "frozen_clip_flow":
        model = RoboCasaFrozenCLIPFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            task_texts=task_texts,
            encoder_name=str(args.vlm_encoder_name),
            width=int(args.width),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "frozen_smolvlm_flow":
        model = RoboCasaFrozenSmolVLMFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            task_texts=task_texts,
            encoder_name=str(args.vlm_encoder_name),
            width=int(args.width),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "frozen_r3m_flow":
        model = RoboCasaFrozenR3MFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            task_texts=task_texts,
            encoder_name=str(args.r3m_encoder_name),
            width=int(args.width),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0_act_resnet":
        model = RoboCasaMiniPi0ACTResNetPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0_act":
        model = RoboCasaMiniPi0ACTPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0_resnet":
        model = RoboCasaMiniPi0ResNetPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "mini_pi0":
        model = RoboCasaMiniPi0Policy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "vit_act":
        model = RoboCasaPatchViTACTPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "history_act_flow":
        model = RoboCasaHistoryACTFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "history_flow":
        model = RoboCasaHistoryFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "history_act":
        model = RoboCasaHistoryACTPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "sequence_flow":
        model = RoboCasaSequenceFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    else:
        model = RoboCasaTemporalChunkBC(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            dropout=float(args.dropout),
        ).to(device)
    init_info = _load_init_checkpoint(model, str(args.init_checkpoint), device)
    freeze_info = _freeze_non_flow(model) if args.freeze_non_flow else _parameter_trainability(model)
    training_budget_start = time.monotonic()
    pidm_metrics = _maybe_pidm_pretrain(
        model=model,
        manifest=manifest,
        split=split,
        video_pool_path=Path(args.video_pool) if args.video_pool else None,
        task_aliases=task_aliases,
        max_train_seconds=float(args.pidm_pretrain_seconds),
        video_episodes_per_task=int(args.pidm_video_episodes_per_task),
        batch_size=int(args.pidm_batch_size),
        gap=int(args.pidm_gap),
        lr=float(args.pidm_lr),
        action_weight=float(args.pidm_action_weight),
        latent_weight=float(args.pidm_latent_weight),
        seed=int(args.seed),
        device=device,
    )
    video_pretrain_metrics = _maybe_video_pretrain(
        model=model,
        manifest=manifest,
        split=split,
        video_pool_path=Path(args.video_pool) if args.video_pool else None,
        task_aliases=task_aliases,
        max_train_seconds=float(args.video_pretrain_seconds),
        episodes_per_task=int(args.video_pretrain_episodes_per_task),
        batch_size=int(args.video_pretrain_batch_size),
        gap=int(args.video_pretrain_gap),
        lr=float(args.video_pretrain_lr),
        temperature=float(args.video_pretrain_temperature),
        objective=str(args.video_pretrain_objective),
        latent_dynamics_weight=float(args.video_latent_dynamics_weight),
        seed=int(args.seed),
        device=device,
    )
    clip_train_data = None
    clip_val_data = None
    clip_cache_metrics = {"enabled": False}
    if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
        cache_start = time.monotonic()
        feature_cache_dir = Path(args.frozen_feature_cache_dir) if args.frozen_feature_cache_dir else None
        encoder_name = str(
            args.r3m_encoder_name
            if args.policy_kind == "frozen_r3m_flow"
            else args.vlm_encoder_name
        )
        clip_train_data = _cache_clip_features(
            model,
            train_data,
            device=device,
            batch_size=int(args.vlm_cache_batch_size),
            label="train",
            cache_path=_feature_cache_path(
                feature_cache_dir,
                label="train",
                data=train_data,
                policy_kind=str(args.policy_kind),
                encoder_name=encoder_name,
                feature_dim=int(model.feature_dim),
                manifest_path=str(args.manifest),
                split_path=str(args.split),
                chunk_horizon=int(args.chunk_horizon),
                frame_stride=int(args.frame_stride),
                history_stride=int(args.history_stride),
                task_aliases=sorted(task_aliases),
                train_episodes_per_task=int(args.train_episodes_per_task),
                val_episodes_per_task=int(args.val_episodes_per_task),
            ),
        )
        clip_val_data = _cache_clip_features(
            model,
            val_data,
            device=device,
            batch_size=int(args.vlm_cache_batch_size),
            label="val",
            cache_path=_feature_cache_path(
                feature_cache_dir,
                label="val",
                data=val_data,
                policy_kind=str(args.policy_kind),
                encoder_name=encoder_name,
                feature_dim=int(model.feature_dim),
                manifest_path=str(args.manifest),
                split_path=str(args.split),
                chunk_horizon=int(args.chunk_horizon),
                frame_stride=int(args.frame_stride),
                history_stride=int(args.history_stride),
                task_aliases=sorted(task_aliases),
                train_episodes_per_task=int(args.train_episodes_per_task),
                val_episodes_per_task=int(args.val_episodes_per_task),
            ),
        )
        clip_cache_metrics = {
            "enabled": True,
            "encoder_name": encoder_name,
            "cache_dir": str(feature_cache_dir or ""),
            "policy_kind": str(args.policy_kind),
            "feature_dim": int(model.feature_dim),
            "train_samples": int(len(clip_train_data)),
            "val_samples": int(len(clip_val_data)),
            "seconds": float(time.monotonic() - cache_start),
        }
    opt = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val_loss = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = training_budget_start

    step = 0
    while True:
        if time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        step += 1
        sample_data = (
            clip_train_data
            if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"} and clip_train_data is not None
            else train_data
        )
        idx = _sample_indices(sample_data, int(args.batch_size), rng, balanced=bool(args.balanced_sampling))
        if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
            batch = _clip_feature_batch(clip_train_data, idx, device)
            batch = _augment_clip_features(batch, float(args.proprio_noise))
            loss = _frozen_clip_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_weight=float(args.bc_aux_weight),
                time_sampling=str(args.flow_time_sampling),
            )
        elif args.policy_kind in {"mini_pi0", "mini_pi0_resnet"}:
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _mini_pi0_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                chunk_decay=float(args.chunk_decay),
                time_sampling=str(args.flow_time_sampling),
            )
        elif args.policy_kind in {"mini_pi0_act", "mini_pi0_act_resnet", "vit_act"}:
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(
                batch["prev_agent"],
                batch["prev_wrist"],
                batch["agent"],
                batch["wrist"],
                batch["prev_proprio"],
                batch["proprio"],
                batch["task_id"],
            )
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        elif args.policy_kind == "history_act_flow":
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _history_act_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_weight=float(args.bc_aux_weight),
            )
        elif args.policy_kind == "history_flow":
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _history_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_aux_weight=float(args.bc_aux_weight),
            )
        elif args.policy_kind == "history_act":
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(
                batch["prev_agent"],
                batch["prev_wrist"],
                batch["agent"],
                batch["wrist"],
                batch["prev_proprio"],
                batch["proprio"],
                batch["task_id"],
            )
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        elif args.policy_kind == "sequence_flow":
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _sequence_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_aux_weight=float(args.bc_aux_weight),
            )
        elif args.policy_kind == "flow":
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _flow_matching_loss(model, batch, sigma=float(args.flow_sigma), chunk_decay=float(args.chunk_decay))
        else:
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        row = {"step": step, "train_loss": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time}
        if step == 1 or step % int(args.log_interval) == 0:
            if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
                val_loss = _eval_clip_feature_loss(
                    model,
                    clip_val_data,
                    device,
                    batch_size=max(64, int(args.batch_size)),
                    flow_steps=int(args.flow_steps),
                    flow_eval_start=str(args.flow_eval_start),
                    flow_residual_scale=float(args.flow_residual_scale),
                )
            else:
                val_loss = _eval_policy_loss(
                    model,
                    val_data,
                    device,
                    batch_size=max(64, int(args.batch_size)),
                    policy_kind=str(args.policy_kind),
                    flow_steps=int(args.flow_steps),
                    flow_eval_start=str(args.flow_eval_start),
                    flow_residual_scale=float(args.flow_residual_scale),
                )
            row["val_loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = _checkpoint_state_dict(model, str(args.policy_kind))
            print(f"step={step} train_loss={row['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
        history.append(row)

    if args.policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"}:
        final_val_loss = _eval_clip_feature_loss(
            model,
            clip_val_data,
            device,
            batch_size=max(64, int(args.batch_size)),
            flow_steps=int(args.flow_steps),
            flow_eval_start=str(args.flow_eval_start),
            flow_residual_scale=float(args.flow_residual_scale),
        )
    else:
        final_val_loss = _eval_policy_loss(
            model,
            val_data,
            device,
            batch_size=max(64, int(args.batch_size)),
            policy_kind=str(args.policy_kind),
            flow_steps=int(args.flow_steps),
            flow_eval_start=str(args.flow_eval_start),
            flow_residual_scale=float(args.flow_residual_scale),
        )
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_step = int(history[-1]["step"] if history else 0)
        best_state = _checkpoint_state_dict(model, str(args.policy_kind))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": _checkpoint_state_dict(model, str(args.policy_kind)),
        "policy_type": (
            "autorobobench_robocasa_bc5_frozen_clip_flow"
            if args.policy_kind == "frozen_clip_flow"
            else "autorobobench_robocasa_bc5_frozen_smolvlm_flow"
            if args.policy_kind == "frozen_smolvlm_flow"
            else "autorobobench_robocasa_bc5_frozen_r3m_flow"
            if args.policy_kind == "frozen_r3m_flow"
            else "autorobobench_robocasa_bc5_history_act"
            if args.policy_kind == "history_act"
            else "autorobobench_robocasa_bc5_mini_pi0_act_resnet"
            if args.policy_kind == "mini_pi0_act_resnet"
            else "autorobobench_robocasa_bc5_mini_pi0_act"
            if args.policy_kind == "mini_pi0_act"
            else "autorobobench_robocasa_bc5_mini_pi0_resnet"
            if args.policy_kind == "mini_pi0_resnet"
            else "autorobobench_robocasa_bc5_mini_pi0"
            if args.policy_kind == "mini_pi0"
            else "autorobobench_robocasa_bc5_vit_act"
            if args.policy_kind == "vit_act"
            else "autorobobench_robocasa_bc5_history_act_flow"
            if args.policy_kind == "history_act_flow"
            else "autorobobench_robocasa_bc5_history_flow"
            if args.policy_kind == "history_flow"
            else "autorobobench_robocasa_bc5_sequence_flow"
            if args.policy_kind == "sequence_flow"
            else "autorobobench_robocasa_bc5_temporal_chunk"
        ),
        "chunk_horizon": int(args.chunk_horizon),
        "action_dim": int(train_data.actions.shape[-1]),
        "proprio_dim": int(train_data.proprio.shape[-1]),
        "task_count": task_count,
        "width": int(args.width),
        "dropout": float(args.dropout),
        "policy_kind": str(args.policy_kind),
        "flow_steps": int(args.flow_steps),
        "flow_sigma": float(args.flow_sigma),
        "flow_source": str(args.flow_source),
        "flow_eval_start": str(args.flow_eval_start),
        "flow_inference_start": str(args.flow_eval_start),
        "flow_residual_scale": float(args.flow_residual_scale),
        "flow_time_sampling": str(args.flow_time_sampling),
        "bc_aux_weight": float(args.bc_aux_weight),
        "chunk_decay": float(args.chunk_decay),
        "transformer_depth": int(args.transformer_depth),
        "action_depth": int(args.action_depth),
        "heads": int(args.heads),
        "patch_size": 8 if args.policy_kind == "vit_act" else None,
        "vlm_encoder_name": str(args.vlm_encoder_name),
        "r3m_encoder_name": str(args.r3m_encoder_name),
        "task_texts": task_texts,
        "history_stride": int(args.history_stride),
        "progress_conditioning": bool(args.progress_conditioning),
        "progress_scale": float(args.progress_scale),
        "progress_feature_dim": 4 if args.progress_conditioning else 0,
        "task_action_normalization": bool(args.task_action_normalization),
        "task_action_mean": task_action_mean,
        "task_action_std": task_action_std,
        "eval_commit_steps": int(args.eval_commit_steps),
        "raw_proprio_dim": raw_proprio_dim,
        "condition_on_robocasa_task_index": False,
        "init_checkpoint": str(args.init_checkpoint),
        "init_info": init_info,
        "freeze_non_flow": bool(args.freeze_non_flow),
        "freeze_info": freeze_info,
        "pidm_pretrain": pidm_metrics,
        "video_pretrain_objective": str(args.video_pretrain_objective),
        "video_latent_dynamics_weight": float(args.video_latent_dynamics_weight),
        "video_pretrain": video_pretrain_metrics,
        "clip_feature_cache": clip_cache_metrics,
        "vpt_pseudo_labels": vpt_metrics,
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "manifest": str(Path(args.manifest)),
        "split": str(Path(args.split)),
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    torch.save(checkpoint, out_dir / "policy.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_action_mse_normalized"] = float(best_val_loss)
    torch.save(best_checkpoint, out_dir / "policy_best.pt")

    metrics = {
        "checkpoint": str(out_dir / "policy.pt"),
        "best_checkpoint": str(out_dir / "policy_best.pt"),
        "best_step": int(best_step),
        "best_val_action_mse_normalized": float(best_val_loss),
        "final_val_action_mse_normalized": float(final_val_loss),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "split_summary": split_summary,
        "chunk_horizon": int(args.chunk_horizon),
        "frame_stride": int(args.frame_stride),
        "train_episodes_per_task": int(args.train_episodes_per_task),
        "val_episodes_per_task": int(args.val_episodes_per_task),
        "steps_completed": int(history[-1]["step"] if history else 0),
        "train_seconds": float(time.monotonic() - start_time),
        "width": int(args.width),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "image_noise": float(args.image_noise),
        "proprio_noise": float(args.proprio_noise),
        "action_smooth": float(args.action_smooth),
        "policy_kind": str(args.policy_kind),
        "flow_steps": int(args.flow_steps),
        "flow_sigma": float(args.flow_sigma),
        "flow_source": str(args.flow_source),
        "flow_eval_start": str(args.flow_eval_start),
        "flow_inference_start": str(args.flow_eval_start),
        "flow_residual_scale": float(args.flow_residual_scale),
        "flow_time_sampling": str(args.flow_time_sampling),
        "bc_aux_weight": float(args.bc_aux_weight),
        "chunk_decay": float(args.chunk_decay),
        "transformer_depth": int(args.transformer_depth),
        "action_depth": int(args.action_depth),
        "heads": int(args.heads),
        "r3m_encoder_name": str(args.r3m_encoder_name),
        "history_stride": int(args.history_stride),
        "progress_conditioning": bool(args.progress_conditioning),
        "progress_scale": float(args.progress_scale),
        "progress_feature_dim": 4 if args.progress_conditioning else 0,
        "eval_commit_steps": int(args.eval_commit_steps),
        "balanced_sampling": bool(args.balanced_sampling),
        "seed": int(args.seed),
        "init_checkpoint": str(args.init_checkpoint),
        "init_info": init_info,
        "freeze_non_flow": bool(args.freeze_non_flow),
        "freeze_info": freeze_info,
        "pidm_pretrain": pidm_metrics,
        "video_pretrain_objective": str(args.video_pretrain_objective),
        "video_latent_dynamics_weight": float(args.video_latent_dynamics_weight),
        "video_pretrain": video_pretrain_metrics,
        "vpt_pseudo_labels": vpt_metrics,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


class _MiniInverseDynamics(nn.Module):
    def __init__(self, *, proprio_dim: int, action_dim: int, task_count: int, width: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.image = nn.Sequential(
            nn.Conv2d(12, 32, 4, stride=2, padding=1),
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
        self.proprio = nn.Sequential(
            nn.Linear(3 * proprio_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.task = nn.Embedding(task_count, 32)
        self.head = nn.Sequential(
            nn.Linear(2 * width + 32, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, action_dim),
        )

    def forward(
        self,
        agent_t: torch.Tensor,
        wrist_t: torch.Tensor,
        agent_tp1: torch.Tensor,
        wrist_tp1: torch.Tensor,
        proprio_t: torch.Tensor,
        proprio_tp1: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent_t.max() > 1.5:
            agent_t = agent_t / 255.0
            wrist_t = wrist_t / 255.0
            agent_tp1 = agent_tp1 / 255.0
            wrist_tp1 = wrist_tp1 / 255.0
        image = self.image(torch.cat([agent_t, wrist_t, agent_tp1, wrist_tp1], dim=1))
        prop = self.proprio(torch.cat([proprio_t, proprio_tp1, proprio_tp1 - proprio_t], dim=-1))
        return self.head(torch.cat([image, prop, self.task(task_id)], dim=-1))


def _maybe_add_vpt_pseudo_labels(
    *,
    train_data: TemporalChunkData,
    manifest: dict,
    split: dict,
    task_aliases: set[str],
    idm_seconds: float,
    pseudo_episodes_per_task: int,
    batch_size: int,
    lr: float,
    pseudo_weight: float,
    chunk_horizon: int,
    frame_stride: int,
    seed: int,
    device: torch.device,
) -> dict:
    if idm_seconds <= 0 or pseudo_episodes_per_task <= 0:
        return {"enabled": False, "pseudo_samples": 0}
    idm_data, idm_summary = _load_idm_supervised_samples(manifest, split, task_aliases=task_aliases)
    if len(idm_data["actions"]) == 0:
        return {"enabled": False, "pseudo_samples": 0, "reason": "no idm samples"}
    task_count = max(1, int(max(idm_data["task_id"].max(initial=0), train_data.task_id.max(initial=0)) + 1))
    idm = _MiniInverseDynamics(
        proprio_dim=int(idm_data["proprio_t"].shape[-1]),
        action_dim=int(idm_data["actions"].shape[-1]),
        task_count=task_count,
        width=256,
    ).to(device)
    opt = torch.optim.AdamW(idm.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 29)
    history: list[dict] = []
    start_time = time.monotonic()
    idm.train()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(idm_seconds):
            break
        step += 1
        idx = rng.integers(0, len(idm_data["actions"]), size=batch_size)
        batch = _idm_batch(idm_data, idx, device)
        pred = idm(**{key: value for key, value in batch.items() if key != "actions"})
        loss = F.mse_loss(pred, batch["actions"])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(idm.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            row = {"step": step, "idm_action_mse": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time}
            history.append(row)
            print(f"vpt_idm step={step} action_mse={row['idm_action_mse']:.6f}", flush=True)

    pseudo, pseudo_summary = _pseudo_label_video_only(
        idm,
        manifest,
        split,
        task_aliases=task_aliases,
        episodes_per_task=pseudo_episodes_per_task,
        chunk_horizon=chunk_horizon,
        frame_stride=frame_stride,
        device=device,
    )
    if len(pseudo) > 0:
        pseudo.mask = (pseudo.mask * max(0.0, float(pseudo_weight))).astype(np.float32)
        _append_temporal_data_(train_data, pseudo)
    return {
        "enabled": True,
        "method": "mini_vpt_inverse_dynamics_pseudo_labels",
        "idm_seconds": float(idm_seconds),
        "idm_steps_completed": int(step),
        "idm_samples": int(len(idm_data["actions"])),
        "idm_summary": idm_summary,
        "idm_history": history,
        "pseudo_samples": int(len(pseudo)),
        "pseudo_weight": float(pseudo_weight),
        "pseudo_summary": pseudo_summary,
        "seconds": float(time.monotonic() - start_time),
    }


def _maybe_pidm_pretrain(
    *,
    model: nn.Module,
    manifest: dict,
    split: dict,
    video_pool_path: Path | None,
    task_aliases: set[str],
    max_train_seconds: float,
    video_episodes_per_task: int,
    batch_size: int,
    gap: int,
    lr: float,
    action_weight: float,
    latent_weight: float,
    seed: int,
    device: torch.device,
) -> dict:
    if max_train_seconds <= 0:
        return {"enabled": False, "steps_completed": 0}
    if not hasattr(model, "vision") and not hasattr(model, "image"):
        return {"enabled": False, "steps_completed": 0, "reason": "model has no image encoder"}
    in_channels = _first_conv_in_channels(_image_encoder(model))
    if in_channels != 6:
        return {
            "enabled": False,
            "steps_completed": 0,
            "reason": f"PIDM pretrain expects 6-channel encoder, got {in_channels}",
        }

    idm_data, idm_summary = _load_idm_supervised_samples(manifest, split, task_aliases=task_aliases)
    if len(idm_data["actions"]) == 0:
        return {"enabled": False, "steps_completed": 0, "reason": "no paired inverse-dynamics samples"}
    video_samples, video_summary = _load_video_transition_samples(
        manifest,
        split,
        video_pool_path=video_pool_path,
        task_aliases=task_aliases,
        episodes_per_task=video_episodes_per_task,
        gap=max(1, gap),
    )
    if len(video_samples["agent_t"]) == 0:
        return {"enabled": False, "steps_completed": 0, "reason": "no video transitions"}

    width = _encoder_width(model)
    action_dim = int(idm_data["actions"].shape[-1])
    task_count = max(
        1,
        int(
            max(
                idm_data["task_id"].max(initial=0),
                video_samples["task_id"].max(initial=0),
                max(int(task["task_id"]) for task in split["tasks"]),
            )
            + 1
        ),
    )
    task_emb = nn.Embedding(task_count, 32).to(device)
    action_head = nn.Sequential(
        nn.Linear(2 * width + 32, 2 * width),
        nn.SiLU(),
        nn.Linear(2 * width, action_dim),
    ).to(device)
    latent_predictor = nn.Sequential(
        nn.Linear(width + 32, width),
        nn.SiLU(),
        nn.Linear(width, width),
    ).to(device)
    params = (
        list(_image_encoder(model).parameters())
        + list(task_emb.parameters())
        + list(action_head.parameters())
        + list(latent_predictor.parameters())
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 43)
    history: list[dict] = []
    start_time = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(max_train_seconds):
            break
        step += 1
        idm_idx = rng.integers(0, len(idm_data["actions"]), size=batch_size)
        idm_batch = _idm_batch(idm_data, idm_idx, device)
        z_t = _encode_video_pair(
            model,
            idm_batch["agent_t"] / 255.0,
            idm_batch["wrist_t"] / 255.0,
        )
        z_tp1 = _encode_video_pair(
            model,
            idm_batch["agent_tp1"] / 255.0,
            idm_batch["wrist_tp1"] / 255.0,
        )
        task = task_emb(idm_batch["task_id"].clamp_max(task_count - 1))
        pred_action = action_head(torch.cat([z_t, z_tp1, task], dim=-1))
        action_loss = F.mse_loss(pred_action, idm_batch["actions"])

        video_idx = rng.integers(0, len(video_samples["agent_t"]), size=batch_size)
        agent_t = torch.as_tensor(video_samples["agent_t"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_t = torch.as_tensor(video_samples["wrist_t"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        agent_tp1 = torch.as_tensor(video_samples["agent_tp1"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_tp1 = torch.as_tensor(video_samples["wrist_tp1"][video_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        z_video_t = _encode_video_pair(model, agent_t, wrist_t)
        with torch.no_grad():
            z_video_tp1 = F.normalize(_encode_video_pair(model, agent_tp1, wrist_tp1), dim=-1)
        video_task_id = torch.as_tensor(
            video_samples["task_id"][video_idx],
            dtype=torch.long,
            device=device,
        ).clamp_max(task_count - 1)
        pred_latent = F.normalize(
            latent_predictor(torch.cat([z_video_t, task_emb(video_task_id)], dim=-1)),
            dim=-1,
        )
        latent_loss = F.mse_loss(pred_latent, z_video_tp1)
        loss = float(action_weight) * action_loss + float(latent_weight) * latent_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            row = {
                "step": int(step),
                "pidm_loss": float(loss.detach().cpu()),
                "pidm_action_mse": float(action_loss.detach().cpu()),
                "pidm_latent_mse": float(latent_loss.detach().cpu()),
                "elapsed_seconds": float(time.monotonic() - start_time),
            }
            history.append(row)
            print(
                "pidm step={step} loss={loss:.6f} action={action:.6f} latent={latent:.6f}".format(
                    step=step,
                    loss=row["pidm_loss"],
                    action=row["pidm_action_mse"],
                    latent=row["pidm_latent_mse"],
                ),
                flush=True,
            )
    return {
        "enabled": True,
        "method": "future_visual_latent_plus_inverse_dynamics",
        "max_train_seconds": float(max_train_seconds),
        "steps_completed": int(step),
        "paired_samples": int(len(idm_data["actions"])),
        "video_samples": int(len(video_samples["agent_t"])),
        "video_episodes_per_task": int(video_episodes_per_task),
        "batch_size": int(batch_size),
        "gap": int(gap),
        "lr": float(lr),
        "action_weight": float(action_weight),
        "latent_weight": float(latent_weight),
        "idm_summary": idm_summary,
        "video_summary": video_summary,
        "history": history,
        "seconds": float(time.monotonic() - start_time),
    }


def _load_idm_supervised_samples(
    manifest: dict,
    split: dict,
    *,
    task_aliases: set[str],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    parts: dict[str, list[np.ndarray]] = {
        "agent_t": [],
        "wrist_t": [],
        "agent_tp1": [],
        "wrist_tp1": [],
        "proprio_t": [],
        "proprio_tp1": [],
        "task_id": [],
        "actions": [],
    }
    summary: list[dict] = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        episode_ids = [int(x) for x in split_task.get("paired_train_episode_ids", split_task.get("train_episode_ids", []))]
        count = 0
        for episode_id in episode_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
            frame = pd.read_parquet(episode_path)
            agent = _read_video64(dataset_root, episode_id, "robot0_agentview_left")
            wrist = _read_video64(dataset_root, episode_id, "robot0_agentview_right")
            proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
            actions = LU.get_episode_actions(dataset_root, episode_id).astype(np.float32)
            n = min(len(agent), len(wrist), len(proprio), len(actions))
            if n <= 1:
                continue
            rows = np.arange(0, n - 1, dtype=np.int32)
            parts["agent_t"].append(agent[rows])
            parts["wrist_t"].append(wrist[rows])
            parts["agent_tp1"].append(agent[rows + 1])
            parts["wrist_tp1"].append(wrist[rows + 1])
            parts["proprio_t"].append(proprio[rows])
            parts["proprio_tp1"].append(proprio[rows + 1])
            parts["actions"].append(actions[rows])
            parts["task_id"].append(np.full((len(rows),), int(split_task["task_id"]), dtype=np.int64))
            count += len(rows)
        summary.append({"alias": alias, "paired_episode_ids": episode_ids, "idm_samples": count})
        print(f"loaded idm paired {alias}: episodes={episode_ids} samples={count}", flush=True)
    return _concat_idm_parts(parts), summary


def _concat_idm_parts(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, values in parts.items():
        if values:
            out[key] = np.concatenate(values, axis=0)
        elif key == "task_id":
            out[key] = np.zeros((0,), dtype=np.int64)
        elif key == "actions":
            out[key] = np.zeros((0, 7), dtype=np.float32)
        else:
            out[key] = np.zeros((0, 64, 64, 3), dtype=np.uint8)
    return out


def _idm_batch(data: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent_t": torch.as_tensor(data["agent_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist_t": torch.as_tensor(data["wrist_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "agent_tp1": torch.as_tensor(data["agent_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist_tp1": torch.as_tensor(data["wrist_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "proprio_t": torch.as_tensor(data["proprio_t"][idx], dtype=torch.float32, device=device),
        "proprio_tp1": torch.as_tensor(data["proprio_tp1"][idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data["task_id"][idx], dtype=torch.long, device=device),
        "actions": torch.as_tensor(data["actions"][idx], dtype=torch.float32, device=device),
    }


def _pseudo_label_video_only(
    idm: _MiniInverseDynamics,
    manifest: dict,
    split: dict,
    *,
    task_aliases: set[str],
    episodes_per_task: int,
    chunk_horizon: int,
    frame_stride: int,
    device: torch.device,
) -> tuple[TemporalChunkData, list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    parts: list[dict[str, np.ndarray]] = []
    summary: list[dict] = []
    idm.eval()
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        video_ids = _video_only_ids(split_task)[:episodes_per_task]
        sample_count = 0
        for episode_id in video_ids:
            part = _pseudo_episode_samples(
                idm,
                dataset_root,
                int(episode_id),
                int(split_task["task_id"]),
                chunk_horizon,
                frame_stride,
                device,
            )
            parts.append(part)
            sample_count += int(part["agent"].shape[0])
        summary.append({"alias": alias, "video_episode_ids": [int(x) for x in video_ids], "pseudo_samples": sample_count})
        print(f"pseudo-labeled {alias}: episodes={video_ids} samples={sample_count}", flush=True)
    return _concat_parts(parts), summary


def _pseudo_episode_samples(
    idm: _MiniInverseDynamics,
    dataset_root: Path,
    episode_idx: int,
    task_id: int,
    chunk_horizon: int,
    frame_stride: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    frame = pd.read_parquet(episode_path)
    agent = _read_video64(dataset_root, episode_idx, "robot0_agentview_left")
    wrist = _read_video64(dataset_root, episode_idx, "robot0_agentview_right")
    proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    n = min(len(agent), len(wrist), len(proprio))
    pred_actions = np.zeros((max(0, n - 1), int(idm.action_dim)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, max(0, n - 1), 256):
            rows = np.arange(start, min(n - 1, start + 256), dtype=np.int32)
            batch = {
                "agent_t": torch.as_tensor(agent[rows], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "wrist_t": torch.as_tensor(wrist[rows], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "agent_tp1": torch.as_tensor(agent[rows + 1], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "wrist_tp1": torch.as_tensor(wrist[rows + 1], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                "proprio_t": torch.as_tensor(proprio[rows], dtype=torch.float32, device=device),
                "proprio_tp1": torch.as_tensor(proprio[rows + 1], dtype=torch.float32, device=device),
                "task_id": torch.full((len(rows),), task_id, dtype=torch.long, device=device),
            }
            pred_actions[rows] = idm(**batch).detach().cpu().numpy().astype(np.float32)
    starts = np.arange(0, max(0, n - 1), max(1, frame_stride), dtype=np.int32)
    out_actions = np.zeros((len(starts), chunk_horizon, pred_actions.shape[-1]), dtype=np.float32)
    mask = np.zeros((len(starts), chunk_horizon), dtype=np.float32)
    for row_idx, start in enumerate(starts):
        end = min(len(pred_actions), int(start) + chunk_horizon)
        length = end - int(start)
        out_actions[row_idx, :length] = pred_actions[int(start) : end]
        mask[row_idx, :length] = 1.0
    return {
        "agent": agent[starts],
        "wrist": wrist[starts],
        "proprio": proprio[starts],
        "actions": out_actions,
        "mask": mask,
        "task_id": np.full((len(starts),), task_id, dtype=np.int64),
        "episode_idx": np.full((len(starts),), episode_idx, dtype=np.int32),
        "frame_idx": starts.astype(np.int32),
    }


def _append_temporal_data_(base: TemporalChunkData, extra: TemporalChunkData) -> None:
    if len(extra) == 0:
        return
    base.agent = np.concatenate([base.agent, extra.agent], axis=0)
    base.wrist = np.concatenate([base.wrist, extra.wrist], axis=0)
    base.proprio = np.concatenate([base.proprio, extra.proprio], axis=0)
    base.actions = np.concatenate([base.actions, extra.actions], axis=0)
    base.mask = np.concatenate([base.mask, extra.mask], axis=0)
    base.task_id = np.concatenate([base.task_id, extra.task_id], axis=0)
    base.episode_idx = np.concatenate([base.episode_idx, extra.episode_idx], axis=0)
    base.frame_idx = np.concatenate([base.frame_idx, extra.frame_idx], axis=0)


def _weighted_masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float32)
    weights = mask.reshape(-1).astype(np.float32)
    keep = weights > 0
    if not np.any(keep):
        return _masked_mean_std(values, mask)
    flat = flat[keep]
    weights = weights[keep]
    denom = max(float(weights.sum()), 1e-6)
    mean = ((flat * weights[:, None]).sum(axis=0) / denom).astype(np.float32)
    var = (((flat - mean[None]) ** 2) * weights[:, None]).sum(axis=0) / denom
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def _per_task_action_stats(data: TemporalChunkData) -> tuple[np.ndarray, np.ndarray]:
    task_count = int(data.task_id.max(initial=0)) + 1
    global_mean, global_std = _weighted_masked_mean_std(data.actions, data.mask)
    means = np.repeat(global_mean[None], task_count, axis=0).astype(np.float32)
    stds = np.repeat(global_std[None], task_count, axis=0).astype(np.float32)
    for task_id in range(task_count):
        keep = data.task_id == task_id
        if not np.any(keep):
            continue
        means[task_id], stds[task_id] = _weighted_masked_mean_std(data.actions[keep], data.mask[keep])
    return means.astype(np.float32), np.maximum(stds, 1e-6).astype(np.float32)


def _normalize_actions_by_task(
    actions: np.ndarray,
    task_id: np.ndarray,
    task_action_mean: np.ndarray,
    task_action_std: np.ndarray,
) -> np.ndarray:
    mean = task_action_mean[np.asarray(task_id, dtype=np.int64)]
    std = task_action_std[np.asarray(task_id, dtype=np.int64)]
    return ((actions - mean[:, None, :]) / std[:, None, :]).astype(np.float32)


def _maybe_video_pretrain(
    *,
    model: nn.Module,
    manifest: dict,
    split: dict,
    video_pool_path: Path | None,
    task_aliases: set[str],
    max_train_seconds: float,
    episodes_per_task: int,
    batch_size: int,
    gap: int,
    lr: float,
    temperature: float,
    objective: str,
    latent_dynamics_weight: float,
    seed: int,
    device: torch.device,
) -> dict:
    if max_train_seconds <= 0 or episodes_per_task <= 0:
        return {"enabled": False, "steps_completed": 0, "samples": 0}
    if not hasattr(model, "vision") and not hasattr(model, "image"):
        return {"enabled": False, "steps_completed": 0, "samples": 0, "reason": "model has no image encoder"}
    in_channels = _first_conv_in_channels(_image_encoder(model))
    if in_channels != 6:
        return {
            "enabled": False,
            "steps_completed": 0,
            "samples": 0,
            "reason": f"video pretrain expects 6-channel encoder, got {in_channels}",
        }

    samples, summary = _load_video_transition_samples(
        manifest,
        split,
        video_pool_path=video_pool_path,
        task_aliases=task_aliases,
        episodes_per_task=episodes_per_task,
        gap=max(1, gap),
    )
    if len(samples["agent_t"]) == 0:
        return {"enabled": False, "steps_completed": 0, "samples": 0, "reason": "no video transitions"}

    width = _encoder_width(model)
    predictor = nn.Sequential(
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, width),
    ).to(device)
    params = list(_image_encoder(model).parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 17)
    history: list[dict] = []
    start_time = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(max_train_seconds):
            break
        step += 1
        idx = rng.integers(0, len(samples["agent_t"]), size=batch_size)
        agent_t = torch.as_tensor(samples["agent_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_t = torch.as_tensor(samples["wrist_t"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        agent_tp1 = torch.as_tensor(samples["agent_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        wrist_tp1 = torch.as_tensor(samples["wrist_tp1"][idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        z_t = _encode_video_pair(model, agent_t, wrist_t)
        z_tp1 = _encode_video_pair(model, agent_tp1, wrist_tp1)
        q_t = F.normalize(predictor(z_t), dim=-1)
        q_tp1 = F.normalize(predictor(z_tp1), dim=-1)
        k_t = F.normalize(z_t, dim=-1)
        k_tp1 = F.normalize(z_tp1, dim=-1)
        labels = torch.arange(q_t.shape[0], dtype=torch.long, device=device)
        logits_fwd = q_t @ k_tp1.T / max(temperature, 1e-6)
        logits_bwd = q_tp1 @ k_t.T / max(temperature, 1e-6)
        nce_loss = 0.5 * (F.cross_entropy(logits_fwd, labels) + F.cross_entropy(logits_bwd, labels))
        dynamics_loss = 0.5 * (
            F.mse_loss(q_t, k_tp1.detach()) + F.mse_loss(q_tp1, k_t.detach())
        )
        if objective == "latent_dynamics":
            loss = float(latent_dynamics_weight) * dynamics_loss
        elif objective == "hybrid":
            loss = nce_loss + float(latent_dynamics_weight) * dynamics_loss
        else:
            loss = nce_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            acc = 0.5 * (
                (logits_fwd.argmax(dim=-1) == labels).float().mean()
                + (logits_bwd.argmax(dim=-1) == labels).float().mean()
            )
            row = {
                "step": step,
                "video_loss": float(loss.detach().cpu()),
                "video_nce_loss": float(nce_loss.detach().cpu()),
                "video_latent_dynamics_loss": float(dynamics_loss.detach().cpu()),
                "video_nce_acc": float(acc.detach().cpu()),
                "elapsed_seconds": time.monotonic() - start_time,
            }
            history.append(row)
            print(
                "video_pretrain step={step} loss={loss:.6f} nce={nce:.6f} "
                "latent={latent:.6f} acc={acc:.3f}".format(
                    step=step,
                    loss=row["video_loss"],
                    nce=row["video_nce_loss"],
                    latent=row["video_latent_dynamics_loss"],
                    acc=row["video_nce_acc"],
                ),
                flush=True,
            )
    return {
        "enabled": True,
        "objective": str(objective),
        "latent_dynamics_weight": float(latent_dynamics_weight),
        "max_train_seconds": float(max_train_seconds),
        "steps_completed": int(step),
        "samples": int(len(samples["agent_t"])),
        "episodes_per_task": int(episodes_per_task),
        "gap": int(gap),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "temperature": float(temperature),
        "summary": summary,
        "history": history,
        "seconds": float(time.monotonic() - start_time),
    }


def _load_video_transition_samples(
    manifest: dict,
    split: dict,
    *,
    video_pool_path: Path | None,
    task_aliases: set[str],
    episodes_per_task: int,
    gap: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    video_pool_tasks = None
    if video_pool_path is None and split.get("video_pool"):
        video_pool_path = Path(str(split["video_pool"]))
    if video_pool_path is not None:
        video_pool = json.loads(video_pool_path.read_text())
        if video_pool.get("contains_actions") is not False or video_pool.get("contains_proprio") is not False:
            raise ValueError(f"video pool must be RGB-only/action-free: {video_pool_path}")
        video_pool_tasks = {task["alias"]: task for task in video_pool["tasks"]}
    parts: dict[str, list[np.ndarray]] = {
        "agent_t": [],
        "wrist_t": [],
        "agent_tp1": [],
        "wrist_tp1": [],
        "task_id": [],
    }
    summary: list[dict] = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        if video_pool_tasks is not None:
            continue
        else:
            task = manifest_tasks[alias]
            dataset_root = Path(task["dataset_path"])
            video_ids = _video_only_ids(split_task)
            video_source = "split"
        if episodes_per_task > 0:
            video_ids = video_ids[:episodes_per_task]
        transition_count = 0
        for episode_id in video_ids:
            agent = _read_video64(dataset_root, int(episode_id), "robot0_agentview_left")
            wrist = _read_video64(dataset_root, int(episode_id), "robot0_agentview_right")
            n = min(len(agent), len(wrist))
            if n <= gap:
                continue
            starts = np.arange(0, n - gap, max(1, gap), dtype=np.int32)
            parts["agent_t"].append(agent[starts])
            parts["wrist_t"].append(wrist[starts])
            parts["agent_tp1"].append(agent[starts + gap])
            parts["wrist_tp1"].append(wrist[starts + gap])
            parts["task_id"].append(np.full((len(starts),), int(split_task["task_id"]), dtype=np.int64))
            transition_count += len(starts)
        summary.append(
            {
                "alias": alias,
                "video_episode_ids": [int(x) for x in video_ids],
                "transitions": transition_count,
                "source": video_source,
                "contains_actions": False,
            }
        )
        print(f"loaded video-only {alias}: episodes={video_ids} transitions={transition_count}", flush=True)
    if video_pool_tasks is not None:
        for alias, task in video_pool_tasks.items():
            if task_aliases and alias not in task_aliases:
                continue
            dataset_root = Path(task["dataset_path"])
            if not dataset_root.is_absolute():
                dataset_root = Path.cwd() / dataset_root
            video_ids = _expand_video_range(task["video_episode_range"])
            if episodes_per_task > 0:
                video_ids = video_ids[:episodes_per_task]
            transition_count = 0
            for episode_id in video_ids:
                agent = _read_video64(dataset_root, int(episode_id), "robot0_agentview_left")
                wrist = _read_video64(dataset_root, int(episode_id), "robot0_agentview_right")
                n = min(len(agent), len(wrist))
                if n <= gap:
                    continue
                starts = np.arange(0, n - gap, max(1, gap), dtype=np.int32)
                parts["agent_t"].append(agent[starts])
                parts["wrist_t"].append(wrist[starts])
                parts["agent_tp1"].append(agent[starts + gap])
                parts["wrist_tp1"].append(wrist[starts + gap])
                parts["task_id"].append(np.full((len(starts),), int(task.get("task_id", 0)), dtype=np.int64))
                transition_count += len(starts)
            summary.append(
                {
                    "alias": alias,
                    "video_episode_ids": [int(x) for x in video_ids],
                    "transitions": transition_count,
                    "source": str(video_pool_path),
                    "contains_actions": False,
                }
            )
            print(f"loaded video-only {alias}: episodes={video_ids} transitions={transition_count}", flush=True)
    out = {}
    for key, value in parts.items():
        if value:
            out[key] = np.concatenate(value, axis=0)
        elif key == "task_id":
            out[key] = np.zeros((0,), dtype=np.int64)
        else:
            out[key] = np.zeros((0, 64, 64, 3), dtype=np.uint8)
    return out, summary


def _video_only_ids(split_task: dict) -> list[int]:
    if "video_only_episode_ids" in split_task:
        return [int(x) for x in split_task["video_only_episode_ids"]]
    if "video_only_episode_range" in split_task:
        start, end = split_task["video_only_episode_range"]
        return list(range(int(start), int(end) + 1))
    paired = set(int(x) for x in split_task.get("paired_train_episode_ids", split_task.get("train_episode_ids", [])))
    return [int(x) for x in split_task.get("train_episode_ids", []) if int(x) not in paired]


def _expand_video_range(bounds: list[int]) -> list[int]:
    if len(bounds) != 2:
        raise ValueError(f"expected [start, end] range, got {bounds!r}")
    return list(range(int(bounds[0]), int(bounds[1]) + 1))


def _image_encoder(model: nn.Module) -> nn.Module:
    if hasattr(model, "vision"):
        return model.vision
    return model.image


def _encoder_width(model: nn.Module) -> int:
    if hasattr(model, "width"):
        return int(model.width)
    if hasattr(model, "head") and isinstance(model.head[0], nn.Linear):
        return int(model.head[0].in_features - model.proprio[-2].out_features - model.task.embedding_dim)
    raise ValueError("could not infer image encoder width")


def _encode_video_pair(model: nn.Module, agent: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    encoder = _image_encoder(model)
    z = encoder(torch.cat([agent, wrist], dim=1))
    if z.ndim == 4:
        z = z.mean(dim=(2, 3))
    return z


def _first_conv_in_channels(module: nn.Module) -> int | None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    return None


def load_split_data(
    manifest: dict,
    split: dict,
    *,
    task_aliases: set[str],
    train_episodes_per_task: int,
    val_episodes_per_task: int,
    chunk_horizon: int,
    frame_stride: int,
) -> tuple[TemporalChunkData, TemporalChunkData, list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    train_parts: list[dict[str, np.ndarray]] = []
    val_parts: list[dict[str, np.ndarray]] = []
    summary: list[dict] = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        task = manifest_tasks[alias]
        task_id = int(split_task["task_id"])
        dataset_root = Path(task["dataset_path"])
        train_ids = list(split_task["train_episode_ids"])
        val_ids = list(split_task["val_episode_ids"])
        if train_episodes_per_task > 0:
            train_ids = train_ids[:train_episodes_per_task]
        if val_episodes_per_task > 0:
            val_ids = val_ids[:val_episodes_per_task]
        for episode_id in train_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{int(episode_id):06d}.parquet"
            train_parts.append(
                _episode_samples(
                    dataset_root,
                    episode_path,
                    int(episode_id),
                    task_id,
                    chunk_horizon,
                    frame_stride,
                    False,
                )
            )
        for episode_id in val_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{int(episode_id):06d}.parquet"
            val_parts.append(
                _episode_samples(
                    dataset_root,
                    episode_path,
                    int(episode_id),
                    task_id,
                    chunk_horizon,
                    frame_stride,
                    False,
                )
            )
        summary.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str(dataset_root),
                "train_episode_ids": [int(x) for x in train_ids],
                "val_episode_ids": [int(x) for x in val_ids],
            }
        )
        print(f"loaded {alias}: train={train_ids} val={val_ids}", flush=True)
    return _concat_parts(train_parts), _concat_parts(val_parts), summary


def _task_texts_for_split(manifest: dict, split: dict, task_aliases: set[str]) -> list[str]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    rows = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        task = manifest_tasks[alias]
        text = str(task.get("description") or task.get("robocasa_task") or alias)
        rows.append((int(split_task["task_id"]), text))
    if not rows:
        return []
    task_count = max(task_id for task_id, _ in rows) + 1
    texts = [f"robot task {idx}" for idx in range(task_count)]
    for task_id, text in rows:
        texts[task_id] = text
    return texts


def _sample_indices(
    data: TemporalChunkData,
    batch_size: int,
    rng: np.random.Generator,
    *,
    balanced: bool,
) -> np.ndarray:
    if not balanced:
        return rng.integers(0, len(data), size=batch_size)
    task_ids = np.unique(data.task_id)
    if len(task_ids) == 0:
        return rng.integers(0, len(data), size=batch_size)
    per_task = int(math.ceil(batch_size / len(task_ids)))
    parts: list[np.ndarray] = []
    for task_id in task_ids:
        pool = np.flatnonzero(data.task_id == int(task_id))
        if len(pool) == 0:
            continue
        parts.append(rng.choice(pool, size=per_task, replace=len(pool) < per_task))
    if not parts:
        return rng.integers(0, len(data), size=batch_size)
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    if len(idx) < batch_size:
        extra = rng.integers(0, len(data), size=batch_size - len(idx))
        idx = np.concatenate([idx, extra])
    return idx[:batch_size]


def _append_progress_features(data: TemporalChunkData, progress_scale: float) -> None:
    progress = _progress_features(data.frame_idx.astype(np.float32), progress_scale)
    data.proprio = np.concatenate([data.proprio, progress], axis=-1).astype(np.float32)


def _progress_features(frame_idx: np.ndarray, progress_scale: float) -> np.ndarray:
    progress = np.clip(frame_idx.astype(np.float32) / max(float(progress_scale), 1.0), 0.0, 1.5)
    return np.stack(
        [
            progress,
            progress * progress,
            np.sin(np.pi * progress),
            np.cos(np.pi * progress),
        ],
        axis=-1,
    ).astype(np.float32)


def _attach_history(data: TemporalChunkData, history_stride: int) -> None:
    history_stride = max(0, int(history_stride))
    prev_idx = np.arange(len(data), dtype=np.int64)
    for episode_id in np.unique(data.episode_idx):
        rows = np.flatnonzero(data.episode_idx == int(episode_id))
        if len(rows) == 0:
            continue
        order = rows[np.argsort(data.frame_idx[rows])]
        frames = data.frame_idx[order]
        if history_stride <= 0:
            prev_idx[order] = order
            continue
        targets = frames - history_stride
        positions = np.searchsorted(frames, targets, side="right") - 1
        positions = np.maximum(positions, 0)
        prev_idx[order] = order[positions]
    data.prev_agent = data.agent[prev_idx]
    data.prev_wrist = data.wrist[prev_idx]
    data.prev_proprio = data.proprio[prev_idx]


def _history_batch(data: TemporalChunkData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "prev_agent": torch.as_tensor(data.prev_agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "prev_wrist": torch.as_tensor(data.prev_wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "agent": torch.as_tensor(data.agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(data.wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "prev_proprio": torch.as_tensor(data.prev_proprio[idx], dtype=torch.float32, device=device),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data.actions[idx], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(data.mask[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


@dataclass
class FrozenCLIPFeatureData:
    image_features: np.ndarray
    prev_proprio: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    task_id: np.ndarray
    episode_idx: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.image_features.shape[0])


def _feature_cache_path(
    cache_dir: Path | None,
    *,
    label: str,
    data: TemporalChunkData,
    policy_kind: str,
    encoder_name: str,
    feature_dim: int,
    manifest_path: str,
    split_path: str,
    chunk_horizon: int,
    frame_stride: int,
    history_stride: int,
    task_aliases: list[str],
    train_episodes_per_task: int,
    val_episodes_per_task: int,
) -> Path | None:
    if cache_dir is None:
        return None
    identity = hashlib.sha256()
    for array in (
        np.asarray(data.task_id, dtype=np.int64),
        np.asarray(data.episode_idx, dtype=np.int64),
        np.asarray(data.frame_idx, dtype=np.int64),
    ):
        identity.update(np.ascontiguousarray(array).view(np.uint8))
    payload = {
        "version": 1,
        "label": label,
        "policy_kind": policy_kind,
        "encoder_name": encoder_name,
        "feature_dim": int(feature_dim),
        "manifest_path": str(Path(manifest_path)),
        "split_path": str(Path(split_path)),
        "chunk_horizon": int(chunk_horizon),
        "frame_stride": int(frame_stride),
        "history_stride": int(history_stride),
        "task_aliases": list(task_aliases),
        "train_episodes_per_task": int(train_episodes_per_task),
        "val_episodes_per_task": int(val_episodes_per_task),
        "sample_count": int(len(data)),
        "sample_identity": identity.hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_encoder = encoder_name.replace("/", "_").replace(":", "_")
    return cache_dir / policy_kind / safe_encoder / f"{label}_{digest}.npz"


def _cache_clip_features(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    data: TemporalChunkData,
    *,
    device: torch.device,
    batch_size: int,
    label: str,
    cache_path: Path | None = None,
) -> FrozenCLIPFeatureData:
    if not hasattr(data, "prev_agent"):
        raise ValueError("frozen feature policy requires _attach_history before feature caching")
    batch_size = max(1, int(batch_size))
    expected_shape = (len(data), 4, int(model.feature_dim))
    features: np.ndarray | None = None
    if cache_path is not None and cache_path.exists():
        start_time = time.monotonic()
        with np.load(cache_path) as cached:
            loaded = np.asarray(cached["image_features"], dtype=np.float16)
        if tuple(loaded.shape) == expected_shape:
            features = loaded
            print(f"feature_cache {label}: loaded {cache_path} in {time.monotonic() - start_time:.1f}s", flush=True)
        else:
            print(f"feature_cache {label}: ignoring shape mismatch at {cache_path}: {loaded.shape} != {expected_shape}", flush=True)
    if features is None:
        features = np.empty(expected_shape, dtype=np.float16)
        model.eval()
        start_time = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                end = min(len(data), start + batch_size)
                idx = np.arange(start, end)
                images = np.concatenate(
                    [
                        data.prev_agent[idx],
                        data.prev_wrist[idx],
                        data.agent[idx],
                        data.wrist[idx],
                    ],
                    axis=0,
                )
                images_t = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                encoded = model.encode_images(images_t).detach().cpu().reshape(4, end - start, -1).transpose(0, 1)
                features[start:end] = encoded.numpy().astype(np.float16)
                if start == 0 or end == len(data) or (end // max(1, batch_size * 20)) != (start // max(1, batch_size * 20)):
                    print(f"feature_cache {label}: {end}/{len(data)}", flush=True)
        print(f"feature_cache {label}: done in {time.monotonic() - start_time:.1f}s", flush=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, image_features=features)
            print(f"feature_cache {label}: saved {cache_path}", flush=True)
    return FrozenCLIPFeatureData(
        image_features=features,
        prev_proprio=np.asarray(data.prev_proprio, dtype=np.float32),
        proprio=np.asarray(data.proprio, dtype=np.float32),
        actions=np.asarray(data.actions, dtype=np.float32),
        mask=np.asarray(data.mask, dtype=np.float32),
        task_id=np.asarray(data.task_id, dtype=np.int64),
        episode_idx=np.asarray(data.episode_idx, dtype=np.int32),
        frame_idx=np.asarray(data.frame_idx, dtype=np.int32),
    )


def _clip_feature_batch(data: FrozenCLIPFeatureData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image_features": torch.as_tensor(data.image_features[idx], dtype=torch.float32, device=device),
        "prev_proprio": torch.as_tensor(data.prev_proprio[idx], dtype=torch.float32, device=device),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data.actions[idx], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(data.mask[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


def _augment_clip_features(batch: dict[str, torch.Tensor], proprio_noise: float) -> dict[str, torch.Tensor]:
    if proprio_noise > 0:
        batch["prev_proprio"] = batch["prev_proprio"] + torch.randn_like(batch["prev_proprio"]) * proprio_noise
        batch["proprio"] = batch["proprio"] + torch.randn_like(batch["proprio"]) * proprio_noise
    return batch


def _augment_history(batch: dict[str, torch.Tensor], image_noise: float, proprio_noise: float) -> dict[str, torch.Tensor]:
    batch = _augment(batch, image_noise, proprio_noise)
    if image_noise > 0:
        scale = 255.0 * image_noise
        batch["prev_agent"] = (batch["prev_agent"] + torch.randn_like(batch["prev_agent"]) * scale).clamp(0.0, 255.0)
        batch["prev_wrist"] = (batch["prev_wrist"] + torch.randn_like(batch["prev_wrist"]) * scale).clamp(0.0, 255.0)
    if proprio_noise > 0:
        batch["prev_proprio"] = batch["prev_proprio"] + torch.randn_like(batch["prev_proprio"]) * proprio_noise
    return batch


def _sequence_flow_matching_loss(
    model: RoboCasaSequenceFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_aux_weight: float,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.encode_obs(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
    if flow_source == "bc":
        source = model.bc_action(context).detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    if bc_aux_weight > 0:
        pred_bc = model.bc_action(context)
        loss = loss + float(bc_aux_weight) * _masked_chunk_loss(
            pred_bc,
            actions,
            batch["mask"],
            chunk_decay=chunk_decay,
        )
    return loss


def _history_flow_matching_loss(
    model: RoboCasaHistoryFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_aux_weight: float,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.encode_obs(
        batch["prev_agent"],
        batch["prev_wrist"],
        batch["agent"],
        batch["wrist"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    if flow_source == "bc":
        source = model.bc_action(context).detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    if bc_aux_weight > 0:
        pred_bc = model.bc_action(context)
        loss = loss + float(bc_aux_weight) * _masked_chunk_loss(
            pred_bc,
            actions,
            batch["mask"],
            chunk_decay=chunk_decay,
        )
    return loss


def _history_act_flow_matching_loss(
    model: RoboCasaHistoryACTFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_weight: float,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.encode_obs(
        batch["prev_agent"],
        batch["prev_wrist"],
        batch["agent"],
        batch["wrist"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    pred_bc = model.bc_action(context)
    bc_loss = _masked_chunk_loss(
        pred_bc,
        actions,
        batch["mask"],
        chunk_decay=chunk_decay,
    )
    if flow_source == "bc":
        source = pred_bc.detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    flow_loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    return flow_loss + float(bc_weight) * bc_loss


def _frozen_clip_flow_matching_loss(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_weight: float,
    time_sampling: str,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.context_from_features(
        batch["image_features"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    pred_bc = model.bc_action(context)
    bc_loss = _masked_chunk_loss(pred_bc, actions, batch["mask"], chunk_decay=chunk_decay)
    if flow_source == "bc":
        source = pred_bc.detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = _sample_flow_time(actions.shape[0], actions.dtype, actions.device, time_sampling)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    flow_loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    return flow_loss + float(bc_weight) * bc_loss


def _mini_pi0_flow_matching_loss(
    model: RoboCasaMiniPi0Policy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    chunk_decay: float,
    time_sampling: str,
) -> torch.Tensor:
    actions = batch["actions"]
    obs_tokens = model.encode_obs_tokens(
        batch["prev_agent"],
        batch["prev_wrist"],
        batch["agent"],
        batch["wrist"],
        batch["prev_proprio"],
        batch["proprio"],
        batch["task_id"],
    )
    source = torch.randn_like(actions) * sigma
    t = _sample_flow_time(actions.shape[0], actions.dtype, actions.device, time_sampling)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(obs_tokens, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    return (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)


def _sample_flow_time(
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    if mode == "beta_low_noise":
        return torch.rand((batch_size,), dtype=dtype, device=device).square().clamp(1e-4, 1.0 - 1e-4)
    return torch.rand((batch_size,), dtype=dtype, device=device)


def _eval_policy_loss(
    model: nn.Module,
    data: TemporalChunkData,
    device: torch.device,
    batch_size: int,
    *,
    policy_kind: str,
    flow_steps: int,
    flow_eval_start: str,
    flow_residual_scale: float,
) -> float:
    if policy_kind in {"history_flow", "history_act_flow", "mini_pi0", "mini_pi0_resnet"}:
        model.eval()
        total = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                idx = np.arange(start, min(len(data), start + batch_size))
                batch = _history_batch(data, idx, device)
                pred = model.sample_flow(
                    batch["prev_agent"],
                    batch["prev_wrist"],
                    batch["agent"],
                    batch["wrist"],
                    batch["prev_proprio"],
                    batch["proprio"],
                    batch["task_id"],
                    steps=flow_steps,
                    start=flow_eval_start,
                    residual_scale=flow_residual_scale,
                )
                per_step = (pred - batch["actions"]).square().mean(dim=-1)
                total = total + (per_step * batch["mask"]).sum()
                denom = denom + batch["mask"].sum()
        model.train()
        return float((total / denom.clamp_min(1.0)).detach().cpu())
    if policy_kind in {"history_act", "mini_pi0_act", "mini_pi0_act_resnet", "vit_act"}:
        model.eval()
        total = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                idx = np.arange(start, min(len(data), start + batch_size))
                batch = _history_batch(data, idx, device)
                pred = model(
                    batch["prev_agent"],
                    batch["prev_wrist"],
                    batch["agent"],
                    batch["wrist"],
                    batch["prev_proprio"],
                    batch["proprio"],
                    batch["task_id"],
                )
                per_step = (pred - batch["actions"]).square().mean(dim=-1)
                total = total + (per_step * batch["mask"]).sum()
                denom = denom + batch["mask"].sum()
        model.train()
        return float((total / denom.clamp_min(1.0)).detach().cpu())
    if policy_kind != "sequence_flow":
        return _eval_loss(
            model,
            data,
            device,
            batch_size=batch_size,
            policy_kind=policy_kind,
            flow_steps=flow_steps,
        )
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            pred = model.sample_flow(
                batch["agent"],
                batch["wrist"],
                batch["proprio"],
                batch["task_id"],
                steps=flow_steps,
                start=flow_eval_start,
            )
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _eval_clip_feature_loss(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    data: FrozenCLIPFeatureData,
    device: torch.device,
    batch_size: int,
    *,
    flow_steps: int,
    flow_eval_start: str,
    flow_residual_scale: float,
) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _clip_feature_batch(data, idx, device)
            context = model.context_from_features(
                batch["image_features"],
                batch["prev_proprio"],
                batch["proprio"],
                batch["task_id"],
            )
            pred = _sample_clip_flow_from_context(
                model,
                context,
                horizon=batch["actions"].shape[1],
                steps=flow_steps,
                start=flow_eval_start,
                residual_scale=flow_residual_scale,
            )
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _sample_clip_flow_from_context(
    model: RoboCasaFrozenCLIPFlowPolicy | RoboCasaFrozenR3MFlowPolicy | RoboCasaFrozenSmolVLMFlowPolicy,
    context: torch.Tensor,
    *,
    horizon: int,
    steps: int,
    start: str,
    residual_scale: float,
) -> torch.Tensor:
    shape = (context.shape[0], int(horizon), int(model.action_dim))
    if start == "noise":
        action = torch.randn(shape, dtype=context.dtype, device=context.device)
    elif start == "zero":
        action = torch.zeros(shape, dtype=context.dtype, device=context.device)
    else:
        action = model.bc_action(context)
    steps = int(steps)
    if steps <= 0:
        return action
    dt = 1.0 / steps
    scale = float(residual_scale)
    for idx in range(steps):
        t = torch.full((context.shape[0],), (idx + 0.5) * dt, dtype=context.dtype, device=context.device)
        action = action + scale * dt * model.flow_velocity(context, action, t)
    return action


def _checkpoint_state_dict(model: nn.Module, policy_kind: str) -> dict[str, torch.Tensor]:
    if policy_kind in {"frozen_clip_flow", "frozen_r3m_flow", "frozen_smolvlm_flow"} and hasattr(model, "head_state_dict"):
        state = model.head_state_dict()
    else:
        state = model.state_dict()
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _load_init_checkpoint(model: nn.Module, checkpoint: str, device: torch.device) -> dict:
    if not checkpoint:
        return {"loaded": 0, "skipped": 0, "path": ""}
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    source_state = payload.get("state_dict", payload)
    target_state = model.state_dict()
    compatible = {}
    for key, value in source_state.items():
        candidate_keys = [key]
        if key.startswith("context_blocks."):
            candidate_keys.append("obs_blocks." + key.removeprefix("context_blocks."))
        for candidate_key in candidate_keys:
            if candidate_key in target_state and tuple(target_state[candidate_key].shape) == tuple(value.shape):
                compatible[candidate_key] = value
                break
    missing_or_mismatch = len(target_state) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    return {
        "path": str(checkpoint),
        "loaded": int(len(compatible)),
        "skipped": int(missing_or_mismatch),
        "source_policy_type": str(payload.get("policy_type", "")) if isinstance(payload, dict) else "",
    }


def _freeze_non_flow(model: nn.Module) -> dict:
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("flow_"):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return {"frozen": int(frozen), "trainable": int(trainable)}


def _parameter_trainability(model: nn.Module) -> dict:
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"frozen": int(frozen), "trainable": int(trainable)}





SUPPORTED_MODES = {"chunk", "sequence_flow"}
SUPPORTED_KINDS = {"bc", "flow", "sequence_flow"}
DEFAULT_MANIFEST = "data/autorobobench/robocasa_stand_mixer_peak_manifest.json"
DEFAULT_SPLIT = "data/autorobobench/robocasa_stand_mixer_peak_splits.json"
DEFAULT_POLICY_CHECKPOINT = "runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt"
DEFAULT_WORLD_MODEL_CHECKPOINT = (
    "data/autorobobench/pretrained_world_models/robocasa_visual_world_model_spatial_conv_11task_20min.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Posttrain a RoboCasa policy with a frozen world model.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--policy-checkpoint", default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--world-model-checkpoint", default=DEFAULT_WORLD_MODEL_CHECKPOINT)
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_world_model_posttraining/default")
    parser.add_argument("--train-episodes-per-task", type=int, default=4)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--chunk-horizon", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--wm-rollout-horizon", type=int, default=4)
    parser.add_argument("--wm-success-weight", type=float, default=1.0)
    parser.add_argument("--wm-progress-weight", type=float, default=0.4)
    parser.add_argument("--bc-weight", type=float, default=0.5)
    parser.add_argument("--init-anchor-weight", type=float, default=0.25)
    parser.add_argument("--action-l2-weight", type=float, default=0.01)
    parser.add_argument("--chunk-decay", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-start", choices=["zero", "noise", "bc"], default="bc")
    parser.add_argument("--wm-progress-scale", type=float, default=260.0)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")
    if float(args.max_train_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")

    device = device_from_arg(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task_aliases = set(args.task_alias)
    policy = load_policy(str(args.policy_checkpoint), device=str(device))
    init_policy = load_policy(str(args.policy_checkpoint), device=str(device))
    policy_kind = str(policy.checkpoint.get("policy_kind", "bc"))
    if policy.mode not in SUPPORTED_MODES or policy_kind not in SUPPORTED_KINDS:
        raise ValueError(
            "robocasa_world_model_posttraining v0 supports only direct BC/flow/sequence_flow policies; "
            f"got mode={policy.mode!r} policy_kind={policy_kind!r}"
        )
    model = policy.model
    init_model = init_policy.model
    if model is None or init_model is None:
        raise ValueError("trajectory-bank policies are not differentiable and cannot be improved with this trainer")
    model.train()
    init_model.eval()
    for param in init_model.parameters():
        param.requires_grad_(False)

    chunk_horizon = int(args.chunk_horizon) if int(args.chunk_horizon) > 0 else int(policy.checkpoint["chunk_horizon"])
    if chunk_horizon != int(policy.checkpoint["chunk_horizon"]):
        raise ValueError(
            f"--chunk-horizon={chunk_horizon} must match policy checkpoint chunk_horizon={policy.checkpoint['chunk_horizon']}"
        )
    train_data, val_data, split_summary = load_split_data(
        manifest,
        split,
        task_aliases=task_aliases,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        chunk_horizon=chunk_horizon,
        frame_stride=int(args.frame_stride),
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("need both train and val samples")

    wm = _load_world_model(str(args.world_model_checkpoint), device)
    wm_model = wm["model"]
    wm_model.eval()
    for param in wm_model.parameters():
        param.requires_grad_(False)
    wm_state_dim = int(wm["config"]["state_dim"])
    raw_train_state = train_data.proprio[:, :wm_state_dim].copy()
    raw_val_state = val_data.proprio[:, :wm_state_dim].copy()

    raw_proprio_dim = int(policy.checkpoint.get("raw_proprio_dim", train_data.proprio.shape[-1]))
    if bool(policy.checkpoint.get("progress_conditioning", False)):
        _append_progress_features(train_data, float(policy.checkpoint.get("progress_scale", 260.0)))
        _append_progress_features(val_data, float(policy.checkpoint.get("progress_scale", 260.0)))
    _normalize_policy_data(train_data, policy.checkpoint)
    _normalize_policy_data(val_data, policy.checkpoint)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_score = -math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.monotonic()

    step = 0
    while True:
        if time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        step += 1
        idx = _sample_indices(train_data, int(args.batch_size), rng, balanced=bool(args.balanced_sampling))
        batch = _batch(train_data, idx, device)
        raw_state = torch.as_tensor(raw_train_state[idx], dtype=torch.float32, device=device)
        progress = _progress_from_frame_idx(train_data.frame_idx[idx], float(args.wm_progress_scale), device)
        pred_norm = _policy_actions(
            model,
            batch,
            policy_kind=policy_kind,
            flow_steps=int(args.flow_steps),
            flow_start=str(args.flow_start),
        )
        with torch.no_grad():
            init_norm = _policy_actions(
                init_model,
                batch,
                policy_kind=policy_kind,
                flow_steps=int(args.flow_steps),
                flow_start=str(args.flow_start),
            )
        actions_raw = _denormalize_actions(pred_norm, policy.checkpoint, batch["task_id"], device)
        wm_metrics = _wm_rollout_objective(
            wm,
            raw_state,
            actions_raw,
            batch["task_id"],
            progress,
            horizon=min(int(args.wm_rollout_horizon), pred_norm.shape[1]),
            success_weight=float(args.wm_success_weight),
            progress_weight=float(args.wm_progress_weight),
        )
        bc_loss = _masked_chunk_loss(pred_norm, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
        init_anchor = _masked_chunk_loss(pred_norm, init_norm, batch["mask"], chunk_decay=float(args.chunk_decay))
        action_l2 = actions_raw.square().mean()
        loss = (
            -wm_metrics["objective"]
            + float(args.bc_weight) * bc_loss
            + float(args.init_anchor_weight) * init_anchor
            + float(args.action_l2_weight) * action_l2
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "wm_objective": float(wm_metrics["objective"].detach().cpu()),
            "wm_success": float(wm_metrics["success"].detach().cpu()),
            "wm_progress_gain": float(wm_metrics["progress_gain"].detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "init_anchor_mse": float(init_anchor.detach().cpu()),
            "action_l2": float(action_l2.detach().cpu()),
            "elapsed_seconds": float(time.monotonic() - start_time),
        }
        if step == 1 or step % int(args.log_interval) == 0:
            val_metrics = _eval_improvement(
                model,
                init_model,
                val_data,
                raw_val_state,
                wm,
                policy.checkpoint,
                policy_kind=policy_kind,
                batch_size=max(64, int(args.batch_size)),
                flow_steps=int(args.flow_steps),
                flow_start=str(args.flow_start),
                wm_rollout_horizon=int(args.wm_rollout_horizon),
                wm_progress_scale=float(args.wm_progress_scale),
                chunk_decay=float(args.chunk_decay),
                success_weight=float(args.wm_success_weight),
                progress_weight=float(args.wm_progress_weight),
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = float(val_metrics["policy_improvement_score"])
            if score > best_score:
                best_score = score
                best_step = int(step)
                best_state = _checkpoint_state_dict(model, policy_kind)
            print(
                "step={step} loss={loss:.6f} wm_obj={wm:.6f} val_score={val:.6f} val_action_mse={mse:.6f}".format(
                    step=step,
                    loss=row["loss"],
                    wm=row["wm_objective"],
                    val=row.get("val_policy_improvement_score", float("nan")),
                    mse=row.get("val_action_mse_normalized", float("nan")),
                ),
                flush=True,
            )
        history.append(row)

    final_metrics = _eval_improvement(
        model,
        init_model,
        val_data,
        raw_val_state,
        wm,
        policy.checkpoint,
        policy_kind=policy_kind,
        batch_size=max(64, int(args.batch_size)),
        flow_steps=int(args.flow_steps),
        flow_start=str(args.flow_start),
        wm_rollout_horizon=int(args.wm_rollout_horizon),
        wm_progress_scale=float(args.wm_progress_scale),
        chunk_decay=float(args.chunk_decay),
        success_weight=float(args.wm_success_weight),
        progress_weight=float(args.wm_progress_weight),
    )
    if float(final_metrics["policy_improvement_score"]) > best_score:
        best_score = float(final_metrics["policy_improvement_score"])
        best_step = int(history[-1]["step"] if history else 0)
        best_state = _checkpoint_state_dict(model, policy_kind)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _policy_checkpoint_payload(
        policy.checkpoint,
        model,
        args,
        policy_kind=policy_kind,
        raw_proprio_dim=raw_proprio_dim,
        split_summary=split_summary,
        history=history,
        best_step=best_step,
        best_score=best_score,
    )
    torch.save(checkpoint, out_dir / "policy.pt")
    best_checkpoint = copy.deepcopy(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_policy_improvement_score"] = float(best_score)
    torch.save(best_checkpoint, out_dir / "policy_best.pt")

    metrics = {
        "task": "robocasa_world_model_posttraining",
        "checkpoint": str(out_dir / "policy.pt"),
        "best_checkpoint": str(out_dir / "policy_best.pt"),
        "policy_checkpoint": str(args.policy_checkpoint),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "policy_mode": str(policy.mode),
        "policy_kind": policy_kind,
        "steps_completed": int(history[-1]["step"] if history else 0),
        "train_seconds": float(time.monotonic() - start_time),
        "train_samples": int(len(train_data)),
        "val_samples": int(len(val_data)),
        "split_summary": split_summary,
        "best_step": int(best_step),
        "best_val_policy_improvement_score": float(best_score),
        "final_val": final_metrics,
        "world_model": {
            "type": str(wm["type"]),
            "state_dim": int(wm["config"]["state_dim"]),
            "action_dim": int(wm["config"]["action_dim"]),
            "task_count": int(wm["config"]["task_count"]),
            "has_visual_prediction": bool(wm["has_visual_prediction"]),
        },
        "objective": {
            "wm_rollout_horizon": int(args.wm_rollout_horizon),
            "wm_success_weight": float(args.wm_success_weight),
            "wm_progress_weight": float(args.wm_progress_weight),
            "bc_weight": float(args.bc_weight),
            "init_anchor_weight": float(args.init_anchor_weight),
            "action_l2_weight": float(args.action_l2_weight),
            "wm_progress_scale": float(args.wm_progress_scale),
        },
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_world_model(checkpoint: str, device: torch.device) -> dict:
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    cfg = payload["config"]
    state = payload["model"]
    task_dim = int(cfg.get("task_dim", 0))
    has_task_condition = "dynamics.task.weight" in state and task_dim > 0
    trunk_in = int(state["dynamics.trunk.0.weight"].shape[1])
    latent_width = int(cfg["latent_dim"]) if int(cfg.get("latent_dim", 0)) > 0 else int(cfg["state_dim"])
    expected_without_progress = latent_width + int(cfg["action_dim"]) + (task_dim if has_task_condition else 0)
    has_progress_condition = trunk_in == expected_without_progress + 1
    if "image_size" in cfg or payload.get("task") == "robocasa_visual_world_model":
        model = VisualRoboCasaWorldModel(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            task_count=int(cfg["task_count"]),
            image_size=int(cfg.get("image_size", 32)),
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
        wm_type = "visual"
        has_visual = True
    else:
        model = RoboCasaWorldModel(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            task_count=int(cfg["task_count"]),
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            task_dim=task_dim,
            latent_dim=int(cfg["latent_dim"]),
            condition_on_task=has_task_condition,
            condition_on_progress=has_progress_condition,
            dropout=float(cfg["dropout"]),
        ).to(device)
        wm_type = "state"
        has_visual = False
    model.load_state_dict(state, strict=False)
    stats = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in payload["stats"].items()}
    return {
        "model": model,
        "stats": stats,
        "config": cfg,
        "checkpoint": payload,
        "type": wm_type,
        "has_visual_prediction": has_visual,
    }


def _normalize_policy_data(data, checkpoint: dict) -> None:
    proprio_mean = np.asarray(_cpu_array(checkpoint["proprio_mean"]), dtype=np.float32)
    proprio_std = np.asarray(_cpu_array(checkpoint["proprio_std"]), dtype=np.float32)
    data.proprio = ((data.proprio - proprio_mean) / np.maximum(proprio_std, 1e-6)).astype(np.float32)
    if checkpoint.get("task_action_normalization"):
        means = np.asarray(checkpoint["task_action_mean"], dtype=np.float32)
        stds = np.asarray(checkpoint["task_action_std"], dtype=np.float32)
        out = np.empty_like(data.actions, dtype=np.float32)
        for task_id in np.unique(data.task_id):
            mask = data.task_id == int(task_id)
            out[mask] = (data.actions[mask] - means[int(task_id)]) / np.maximum(stds[int(task_id)], 1e-6)
        data.actions = out.astype(np.float32)
    else:
        action_mean = np.asarray(_cpu_array(checkpoint["action_mean"]), dtype=np.float32)
        action_std = np.asarray(_cpu_array(checkpoint["action_std"]), dtype=np.float32)
        data.actions = ((data.actions - action_mean) / np.maximum(action_std, 1e-6)).astype(np.float32)


def _policy_actions(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    policy_kind: str,
    flow_steps: int,
    flow_start: str,
) -> torch.Tensor:
    if policy_kind == "flow":
        return model.sample_flow(
            batch["agent"],
            batch["wrist"],
            batch["proprio"],
            batch["task_id"],
            steps=int(flow_steps),
        )
    if policy_kind == "sequence_flow":
        return model.sample_flow(
            batch["agent"],
            batch["wrist"],
            batch["proprio"],
            batch["task_id"],
            steps=int(flow_steps),
            start=str(flow_start),
        )
    return model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])


def _denormalize_actions(
    pred_norm: torch.Tensor,
    checkpoint: dict,
    task_id: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if checkpoint.get("task_action_normalization"):
        means = torch.as_tensor(checkpoint["task_action_mean"], dtype=pred_norm.dtype, device=device)
        stds = torch.as_tensor(checkpoint["task_action_std"], dtype=pred_norm.dtype, device=device).clamp_min(1e-6)
        return pred_norm * stds[task_id.long()].unsqueeze(1) + means[task_id.long()].unsqueeze(1)
    mean = _tensor_from_checkpoint(checkpoint, "action_mean", device, pred_norm.dtype)
    std = _tensor_from_checkpoint(checkpoint, "action_std", device, pred_norm.dtype).clamp_min(1e-6)
    return pred_norm * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _wm_rollout_objective(
    wm: dict,
    raw_state: torch.Tensor,
    actions_raw: torch.Tensor,
    task_id: torch.Tensor,
    progress: torch.Tensor,
    *,
    horizon: int,
    success_weight: float,
    progress_weight: float,
) -> dict[str, torch.Tensor]:
    stats = wm["stats"]
    model = wm["model"]
    state = (raw_state - stats["state_mean"]) / stats["state_std"].clamp_min(1e-6)
    objective = torch.zeros((), dtype=state.dtype, device=state.device)
    success_terms = []
    progress_terms = []
    steps = max(1, min(int(horizon), int(actions_raw.shape[1])))
    for step in range(steps):
        action = (actions_raw[:, step] - stats["action_mean"]) / stats["action_std"].clamp_min(1e-6)
        out = model(state, action, task_id=task_id, progress=progress)
        success_prob = torch.sigmoid(out["success_logit"])
        next_progress = out["next_progress"].clamp(0.0, 1.0)
        objective = objective + (
            float(success_weight) * success_prob.mean()
            + float(progress_weight) * next_progress.mean()
        )
        success_terms.append(success_prob.mean())
        progress_terms.append(next_progress.mean())
        state = out["next_state"]
    scale = 1.0 / float(steps)
    return {
        "objective": objective * scale,
        "success": torch.stack(success_terms).mean(),
        "progress_gain": torch.stack(progress_terms).mean(),
    }


@torch.no_grad()
def _eval_improvement(
    model: nn.Module,
    init_model: nn.Module,
    data,
    raw_state_np: np.ndarray,
    wm: dict,
    checkpoint: dict,
    *,
    policy_kind: str,
    batch_size: int,
    flow_steps: int,
    flow_start: str,
    wm_rollout_horizon: int,
    wm_progress_scale: float,
    chunk_decay: float,
    success_weight: float,
    progress_weight: float,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums = {
        "wm_objective": 0.0,
        "wm_success": 0.0,
        "wm_progress_gain": 0.0,
        "action_mse_normalized": 0.0,
        "init_anchor_mse": 0.0,
    }
    count = 0
    device = next(model.parameters()).device
    for start in range(0, len(data), int(batch_size)):
        idx = np.arange(start, min(len(data), start + int(batch_size)))
        batch = _batch(data, idx, device)
        pred_norm = _policy_actions(model, batch, policy_kind=policy_kind, flow_steps=flow_steps, flow_start=flow_start)
        init_norm = _policy_actions(init_model, batch, policy_kind=policy_kind, flow_steps=flow_steps, flow_start=flow_start)
        actions_raw = _denormalize_actions(pred_norm, checkpoint, batch["task_id"], device)
        raw_state = torch.as_tensor(raw_state_np[idx], dtype=torch.float32, device=device)
        progress = _progress_from_frame_idx(data.frame_idx[idx], wm_progress_scale, device)
        wm_metrics = _wm_rollout_objective(
            wm,
            raw_state,
            actions_raw,
            batch["task_id"],
            progress,
            horizon=wm_rollout_horizon,
            success_weight=success_weight,
            progress_weight=progress_weight,
        )
        action_mse = _masked_chunk_loss(pred_norm, batch["actions"], batch["mask"], chunk_decay=chunk_decay)
        init_anchor = _masked_chunk_loss(pred_norm, init_norm, batch["mask"], chunk_decay=chunk_decay)
        n = len(idx)
        sums["wm_objective"] += float(wm_metrics["objective"].detach().cpu()) * n
        sums["wm_success"] += float(wm_metrics["success"].detach().cpu()) * n
        sums["wm_progress_gain"] += float(wm_metrics["progress_gain"].detach().cpu()) * n
        sums["action_mse_normalized"] += float(action_mse.detach().cpu()) * n
        sums["init_anchor_mse"] += float(init_anchor.detach().cpu()) * n
        count += n
    if was_training:
        model.train()
    values = {key: value / max(1, count) for key, value in sums.items()}
    values["policy_improvement_score"] = (
        values["wm_objective"]
        - 0.25 * values["action_mse_normalized"]
        - 0.10 * values["init_anchor_mse"]
    )
    return values


def _masked_chunk_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, chunk_decay: float) -> torch.Tensor:
    per_step = (pred - target).square().mean(dim=-1)
    weights = torch.ones((pred.shape[1],), dtype=pred.dtype, device=pred.device)
    if float(chunk_decay) != 1.0:
        idx = torch.arange(pred.shape[1], dtype=pred.dtype, device=pred.device)
        weights = float(chunk_decay) ** idx
    weights = weights.reshape(1, -1) / weights.mean().clamp_min(1e-6)
    return (per_step * mask * weights).sum() / (mask * weights).sum().clamp_min(1.0)


def _progress_from_frame_idx(frame_idx: np.ndarray, scale: float, device: torch.device) -> torch.Tensor:
    progress = np.clip(np.asarray(frame_idx, dtype=np.float32) / max(float(scale), 1.0), 0.0, 1.0)
    return torch.as_tensor(progress, dtype=torch.float32, device=device)


def _policy_checkpoint_payload(
    source: dict,
    model: nn.Module,
    args: argparse.Namespace,
    *,
    policy_kind: str,
    raw_proprio_dim: int,
    split_summary: list[dict],
    history: list[dict],
    best_step: int,
    best_score: float,
) -> dict:
    checkpoint = copy.deepcopy(source)
    checkpoint["state_dict"] = _checkpoint_state_dict(model, policy_kind)
    checkpoint["task"] = "robocasa_world_model_posttraining"
    checkpoint["world_model_posttraining"] = {
        "source_policy_checkpoint": str(args.policy_checkpoint),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "best_step": int(best_step),
        "best_val_policy_improvement_score": float(best_score),
        "wm_rollout_horizon": int(args.wm_rollout_horizon),
        "wm_success_weight": float(args.wm_success_weight),
        "wm_progress_weight": float(args.wm_progress_weight),
        "bc_weight": float(args.bc_weight),
        "init_anchor_weight": float(args.init_anchor_weight),
        "action_l2_weight": float(args.action_l2_weight),
        "wm_progress_scale": float(args.wm_progress_scale),
        "train_episodes_per_task": int(args.train_episodes_per_task),
        "val_episodes_per_task": int(args.val_episodes_per_task),
        "frame_stride": int(args.frame_stride),
        "split_summary": split_summary,
        "history_tail": history[-10:],
    }
    checkpoint["init_checkpoint"] = str(args.policy_checkpoint)
    checkpoint["raw_proprio_dim"] = int(raw_proprio_dim)
    checkpoint["flow_steps"] = int(args.flow_steps)
    checkpoint["flow_eval_start"] = str(args.flow_start)
    checkpoint["flow_inference_start"] = str(args.flow_start)
    return checkpoint


def _tensor_from_checkpoint(checkpoint: dict, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(_cpu_array(checkpoint[key]), dtype=dtype, device=device)


def _cpu_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    main()
