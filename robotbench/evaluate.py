from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import make_env
from robotbench.metrics import aggregate_episodes


PolicyFn = Callable[[np.ndarray], np.ndarray]


def rollout(
    task: TaskConfig,
    world: str,
    seed: int,
    policy_fn: PolicyFn,
    backend: str = "toy",
) -> dict[str, Any]:
    env = make_env(task=task, world=world, seed=seed, backend=backend)
    obs = env.reset()
    total_reward = 0.0
    energy = 0.0
    jerk = 0.0
    final_info: dict[str, Any] = {}
    for _ in range(task.horizon):
        result = env.step(policy_fn(obs))
        obs = result.obs
        total_reward += result.reward
        energy += float(result.info.get("energy", 0.0))
        jerk += float(result.info.get("jerk", 0.0))
        final_info = result.info
        if result.terminated or result.truncated:
            break

    return {
        "return": total_reward,
        "success": bool(final_info.get("success", False)),
        "distance": float(final_info.get("distance", 999.0)),
        "energy": energy,
        "jerk": jerk,
        "joint_limit_violations": int(final_info.get("joint_limit_violations", 0)),
        "torque_limit_violations": int(final_info.get("torque_limit_violations", 0)),
        "catastrophe": bool(final_info.get("catastrophe", False)),
    }


def evaluate_policy(
    task: TaskConfig,
    world: str,
    seeds: list[int],
    episodes_per_seed: int,
    policy_fn: PolicyFn,
    backend: str = "toy",
) -> dict[str, Any]:
    episodes = []
    for seed in seeds:
        for episode_idx in range(episodes_per_seed):
            episodes.append(
                rollout(
                    task=task,
                    world=world,
                    seed=seed * 10_000 + episode_idx,
                    policy_fn=policy_fn,
                    backend=backend,
                )
            )
    return aggregate_episodes(episodes)
