from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainBudget:
    seconds: float
    max_iterations: int
    rollouts_per_iteration: int
    eval_episodes: int

