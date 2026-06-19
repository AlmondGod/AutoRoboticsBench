from __future__ import annotations

import torch
from torch import nn


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
