from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import make_env
from robotbench.train import TrainBudget


@dataclass
class LinearPolicy:
    weights: np.ndarray
    bias: np.ndarray

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.tanh(obs @ self.weights + self.bias)


@dataclass
class ScaledLinearPolicy:
    weights: np.ndarray
    bias: np.ndarray
    scale_logits: np.ndarray

    def act(self, obs: np.ndarray) -> np.ndarray:
        scale = 0.05 + 0.95 / (1.0 + np.exp(-self.scale_logits))
        return scale * np.tanh(obs @ self.weights + self.bias)


@dataclass
class RobustTaskPolicy:
    task_name: str
    action_limit: float

    def act(self, obs: np.ndarray) -> np.ndarray:
        ee = obs[0:2]
        obj = obs[2:4]
        target = obs[4:6]
        if self.task_name == "push":
            return self._push_action(ee=ee, obj=obj, target=target)
        return self._reach_action(ee=ee, target=target)

    def _reach_action(self, ee: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = target - ee
        distance = np.linalg.norm(error)
        gain = 7.5 if distance > 0.12 else 4.0
        return np.clip(gain * error, -self.action_limit, self.action_limit)

    def _push_action(self, ee: np.ndarray, obj: np.ndarray, target: np.ndarray) -> np.ndarray:
        push_direction = target - obj
        norm = np.linalg.norm(push_direction)
        if norm < 1e-6:
            return np.zeros(2)
        push_direction = push_direction / norm
        staging_point = obj - 0.14 * push_direction
        if np.linalg.norm(ee - staging_point) > 0.08:
            desired = staging_point
            gain = 8.0
        else:
            desired = obj + 0.16 * push_direction
            gain = 10.0
        return np.clip(gain * (desired - ee), -self.action_limit, self.action_limit)


@dataclass
class AlohaBimanualPolicy:
    action_limit: float

    def act(self, obs: np.ndarray) -> np.ndarray:
        left = obs[0:2]
        target = obs[4:6]
        error = np.clip(target - left, -0.25, 0.25)
        action = np.zeros(14)
        action[0] = 2.0 * error[0]
        action[1] = -1.5 * error[1]
        action[2] = 1.2 * error[1]
        action[4] = -0.8 * error[1]
        action[6] = 0.2
        action[7] = -1.2 * error[0]
        action[8] = -0.4
        action[13] = 0.2
        return np.clip(action, -self.action_limit, self.action_limit)


@dataclass
class MobileAlohaMockPolicy:
    action_limit: float

    def act(self, obs: np.ndarray) -> np.ndarray:
        base = obs[0:2]
        left = obs[2:4]
        target = obs[4:6]
        base_error = np.clip(target - base, -0.3, 0.3)
        arm_error = np.clip(target - left, -0.25, 0.25)
        action = np.zeros(17)
        action[0] = 1.5 * base_error[0]
        action[1] = 1.2 * base_error[1]
        action[3] = 2.0 * arm_error[0]
        action[4] = -1.5 * arm_error[1]
        action[5] = 1.2 * arm_error[1]
        action[7] = -0.8 * arm_error[1]
        action[9] = 0.2
        action[10] = -1.2 * arm_error[0]
        action[11] = -0.4
        action[16] = 0.2
        return np.clip(action, -self.action_limit, self.action_limit)


def make_agent(obs_dim: int, act_dim: int, config: dict[str, Any] | None = None) -> LinearPolicy:
    config = config or {}
    scale = float(config.get("init_scale", 0.05))
    rng = np.random.default_rng(int(config.get("seed", 0)))
    return LinearPolicy(
        weights=rng.normal(0.0, scale, size=(obs_dim, act_dim)),
        bias=np.zeros(act_dim),
    )


def train_policy(
    task: TaskConfig,
    budget: TrainBudget,
    seed: int,
    backend: str = "toy",
) -> LinearPolicy | ScaledLinearPolicy | RobustTaskPolicy | AlohaBimanualPolicy | MobileAlohaMockPolicy:
    """Return a robust low-data controller for the current task.

    Hypothesis: for phase 1's state-observed reaching and pushing tasks, a
    clipped geometric feedback controller should transfer better across the
    eval dynamics shift than a sample-hungry linear-policy search.
    """
    if backend == "mobile_aloha_mock":
        return MobileAlohaMockPolicy(action_limit=task.action_limit)
    if backend in {"toy", "mujoco"} and task.name in {"reach", "push"}:
        return RobustTaskPolicy(task_name=task.name, action_limit=task.action_limit)
    return _train_cem_policy(task=task, budget=budget, seed=seed, backend=backend)


def _train_cem_policy(
    task: TaskConfig,
    budget: TrainBudget,
    seed: int,
    backend: str = "toy",
) -> LinearPolicy:
    rng = np.random.default_rng(seed)
    probe_env = make_env(task=task, world="train", seed=seed, backend=backend)
    obs_dim = int(probe_env.obs_dim)
    act_dim = int(probe_env.act_dim)
    use_scaled_policy = backend == "aloha"
    param_dim = obs_dim * act_dim + act_dim + (act_dim if use_scaled_policy else 0)
    mean = np.zeros(param_dim)
    std = np.full(param_dim, 0.7)
    if use_scaled_policy:
        scale_start = _logit((0.25 - 0.05) / 0.95)
        mean[-act_dim:] = scale_start
        std[-act_dim:] = 0.25
    best_params = mean.copy()
    best_score = -float("inf")
    start = time.monotonic()

    for _ in range(budget.max_iterations):
        if time.monotonic() - start >= budget.seconds:
            break
        candidates = rng.normal(mean, std, size=(budget.rollouts_per_iteration, param_dim))
        scored = []
        for params in candidates:
            score = _estimate_candidate(
                task=task,
                params=params,
                rng=rng,
                episodes=2,
                backend=backend,
                obs_dim=obs_dim,
                act_dim=act_dim,
                use_scaled_policy=use_scaled_policy,
            )
            scored.append((score, params))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > best_score:
            best_score, best_params = scored[0]

        elite_count = max(2, len(scored) // 5)
        elites = np.stack([params for _, params in scored[:elite_count]])
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.05)

    return _policy_from_params(
        best_params,
        obs_dim=obs_dim,
        act_dim=act_dim,
        use_scaled_policy=use_scaled_policy,
    )


def act(
    policy: LinearPolicy | ScaledLinearPolicy | RobustTaskPolicy | AlohaBimanualPolicy | MobileAlohaMockPolicy,
    obs: np.ndarray,
) -> np.ndarray:
    return policy.act(obs)


def save_policy(
    policy: LinearPolicy | ScaledLinearPolicy | RobustTaskPolicy | AlohaBimanualPolicy | MobileAlohaMockPolicy,
    path: str,
) -> None:
    if isinstance(policy, LinearPolicy):
        np.savez(path, kind="linear", weights=policy.weights, bias=policy.bias)
        return
    if isinstance(policy, ScaledLinearPolicy):
        np.savez(
            path,
            kind="scaled_linear",
            weights=policy.weights,
            bias=policy.bias,
            scale_logits=policy.scale_logits,
        )
        return
    if isinstance(policy, RobustTaskPolicy):
        np.savez(path, kind="robust_task", task_name=policy.task_name, action_limit=policy.action_limit)
        return
    if isinstance(policy, AlohaBimanualPolicy):
        np.savez(path, kind="aloha_bimanual", action_limit=policy.action_limit)
        return
    if isinstance(policy, MobileAlohaMockPolicy):
        np.savez(path, kind="mobile_aloha_mock", action_limit=policy.action_limit)
        return
    raise TypeError(f"unsupported policy type: {type(policy).__name__}")


def load_policy(path: str):
    data = np.load(Path(path), allow_pickle=False)
    kind = str(data["kind"])
    if kind == "linear":
        return LinearPolicy(weights=data["weights"], bias=data["bias"])
    if kind == "scaled_linear":
        return ScaledLinearPolicy(
            weights=data["weights"],
            bias=data["bias"],
            scale_logits=data["scale_logits"],
        )
    if kind == "robust_task":
        return RobustTaskPolicy(task_name=str(data["task_name"]), action_limit=float(data["action_limit"]))
    if kind == "aloha_bimanual":
        return AlohaBimanualPolicy(action_limit=float(data["action_limit"]))
    if kind == "mobile_aloha_mock":
        return MobileAlohaMockPolicy(action_limit=float(data["action_limit"]))
    raise ValueError(f"unknown policy kind: {kind}")


def _estimate_candidate(
    task: TaskConfig,
    params: np.ndarray,
    rng: np.random.Generator,
    episodes: int,
    backend: str,
    obs_dim: int,
    act_dim: int,
    use_scaled_policy: bool,
) -> float:
    policy = _policy_from_params(
        params,
        obs_dim=obs_dim,
        act_dim=act_dim,
        use_scaled_policy=use_scaled_policy,
    )
    returns = []
    for _ in range(episodes):
        env = make_env(
            task=task,
            world="train",
            seed=int(rng.integers(0, 2**31 - 1)),
            backend=backend,
        )
        obs = env.reset()
        total = 0.0
        for _step in range(task.horizon):
            result = env.step(policy.act(obs))
            obs = result.obs
            total += result.reward
            if result.terminated or result.truncated:
                break
        returns.append(total)
    return float(np.mean(returns))


def _policy_from_params(
    params: np.ndarray,
    obs_dim: int,
    act_dim: int,
    use_scaled_policy: bool = False,
) -> LinearPolicy | ScaledLinearPolicy:
    weights_end = obs_dim * act_dim
    weights = params[:weights_end].reshape(obs_dim, act_dim)
    bias = params[weights_end : weights_end + act_dim]
    if use_scaled_policy:
        scale_logits = params[weights_end + act_dim : weights_end + 2 * act_dim]
        return ScaledLinearPolicy(weights=weights, bias=bias, scale_logits=scale_logits)
    return LinearPolicy(weights=weights, bias=bias)


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return float(np.log(value / (1.0 - value)))
