from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robotbench.config import TaskConfig
from robotbench.safety import SafetyTracker


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class PlanarRobotEnv:
    """Small deterministic robotics-style benchmark with shifted dynamics."""

    obs_dim = 8
    act_dim = 2

    def __init__(self, task: TaskConfig, world: str, seed: int):
        if world not in {"train", "eval"}:
            raise ValueError("world must be 'train' or 'eval'")
        self.task = task
        self.world = world
        self.params = getattr(task, world)
        self.rng = np.random.default_rng(seed)
        self.safety = SafetyTracker(action_limit=task.action_limit)
        self.t = 0
        self.ee = np.zeros(2)
        self.obj = np.zeros(2)
        self.target = np.zeros(2)
        self.prev_action = np.zeros(2)
        self.action_queue: list[np.ndarray] = []

    def reset(self) -> np.ndarray:
        self.t = 0
        self.safety.reset()
        self.prev_action = np.zeros(2)
        latency_steps = int(self.params.get("latency_steps", 0))
        self.action_queue = [np.zeros(2) for _ in range(latency_steps)]

        self.ee = self.rng.uniform(-0.35, 0.35, size=2)
        self.target = self.rng.uniform(0.35, 0.75, size=2)
        if self.task.name == "push":
            self.obj = self.rng.uniform(-0.2, 0.25, size=2)
        else:
            self.obj = np.zeros(2)
        return self._obs()

    def step(self, action: np.ndarray) -> StepResult:
        self.t += 1
        raw_action = np.asarray(action, dtype=np.float64)
        action = np.clip(raw_action, -self.task.action_limit, self.task.action_limit)
        self.safety.observe(raw_action=raw_action, clipped_action=action)

        if self.action_queue:
            self.action_queue.append(action)
            applied = self.action_queue.pop(0)
        else:
            applied = action

        damping = float(self.params.get("damping", 0.02))
        control_scale = float(self.params.get("control_scale", 0.05))
        drift = np.asarray(self.params.get("drift", [0.0, 0.0]), dtype=np.float64)
        self.ee = self.ee + control_scale * applied - damping * self.ee + drift
        self.ee = np.clip(self.ee, -1.2, 1.2)

        if self.task.name == "push":
            self._push_object(applied)

        dist = self._distance_to_goal()
        success = dist <= self.task.success_tolerance
        action_cost = 0.01 * float(np.sum(np.square(applied)))
        jerk_cost = 0.005 * float(np.sum(np.square(applied - self.prev_action)))
        jerk = float(np.sum(np.square(applied - self.prev_action)))
        reward = -dist - action_cost - jerk_cost + (1.0 if success else 0.0)

        terminated = bool(success)
        truncated = self.t >= self.task.horizon
        info = {
            "success": success,
            "distance": dist,
            "energy": float(np.sum(np.square(applied))),
            "jerk": jerk,
            **self.safety.snapshot(),
        }
        self.prev_action = applied
        return StepResult(self._obs(), float(reward), terminated, truncated, info)

    def _push_object(self, applied: np.ndarray) -> None:
        radius = float(self.params.get("push_radius", 0.18))
        friction = float(self.params.get("friction", 1.0))
        object_mass = float(self.params.get("object_mass", 1.0))
        delta = self.obj - self.ee
        dist = float(np.linalg.norm(delta))
        if dist < radius:
            direction = applied / (np.linalg.norm(applied) + 1e-8)
            impulse = (radius - dist) / radius
            self.obj += 0.04 * friction * impulse * direction / object_mass
        self.obj = np.clip(self.obj, -1.2, 1.2)

    def _distance_to_goal(self) -> float:
        subject = self.obj if self.task.name == "push" else self.ee
        return float(np.linalg.norm(subject - self.target))

    def _obs(self) -> np.ndarray:
        noise = float(self.params.get("observation_noise", 0.0))
        obs = np.concatenate(
            [
                self.ee,
                self.obj,
                self.target,
                self.target - (self.obj if self.task.name == "push" else self.ee),
            ]
        ).astype(np.float64)
        if noise:
            obs = obs + self.rng.normal(0.0, noise, size=obs.shape)
        return obs


def make_env(task: TaskConfig, world: str, seed: int, backend: str = "toy"):
    if backend == "toy":
        return PlanarRobotEnv(task=task, world=world, seed=seed)
    if backend == "mujoco":
        from robotbench.mujoco_envs import make_mujoco_env

        return make_mujoco_env(task=task, world=world, seed=seed)
    if backend == "aloha":
        from robotbench.aloha_envs import make_aloha_env

        return make_aloha_env(task=task, world=world, seed=seed)
    if backend == "mobile_aloha_mock":
        from robotbench.aloha_envs import make_mobile_aloha_mock_env

        return make_mobile_aloha_mock_env(task=task, world=world, seed=seed)
    raise ValueError(f"unknown backend '{backend}'")
