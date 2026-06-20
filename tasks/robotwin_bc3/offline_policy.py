from __future__ import annotations

import torch
import torch.nn as nn


class StateChunkPolicy(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 14,
        action_dim: int = 14,
        num_tasks: int = 3,
        horizon: int = 16,
        width: int = 512,
        depth: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.task_emb = nn.Embedding(int(num_tasks), 64)
        layers: list[nn.Module] = []
        in_dim = int(state_dim) + 64 + 1
        for idx in range(int(depth)):
            layers.append(nn.Linear(in_dim if idx == 0 else int(width), int(width)))
            layers.append(nn.LayerNorm(int(width)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(int(width), int(horizon) * int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, task_id: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        task = self.task_emb(task_id.long())
        x = torch.cat([state, task, progress.float()], dim=-1)
        out = self.net(x)
        return out.view(state.shape[0], self.horizon, self.action_dim)
