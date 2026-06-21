from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_bc5 import inference as bc5_inference
from tasks.robocasa_wm_policy_improvement.train import _load_world_model
from train.common import device_from_arg


@dataclass
class SelectorPolicy:
    source: Any
    candidate: Any
    wm: dict
    checkpoint: dict
    device: torch.device
    last_choice: str = ""


def load_policy(checkpoint: str, device: str = "auto") -> SelectorPolicy:
    torch_device = device_from_arg(device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if payload.get("policy_type") != "robocasa_wm_policy_selector":
        raise ValueError(f"expected robocasa_wm_policy_selector checkpoint, got {payload.get('policy_type')!r}")
    source = bc5_inference.load_policy(str(payload["source_policy_checkpoint"]), device=str(torch_device))
    candidate = bc5_inference.load_policy(str(payload["candidate_policy_checkpoint"]), device=str(torch_device))
    wm = _load_world_model(str(payload["world_model_checkpoint"]), torch_device)
    wm["model"].eval()
    return SelectorPolicy(source=source, candidate=candidate, wm=wm, checkpoint=payload, device=torch_device)


def act(policy: SelectorPolicy, obs: dict, task: dict) -> np.ndarray:
    source_actions = np.asarray(bc5_inference.act(policy.source, obs, task), dtype=np.float32)
    candidate_actions = np.asarray(bc5_inference.act(policy.candidate, obs, task), dtype=np.float32)
    task_id = int(task["task_id"])
    source_score = _score_actions(policy, obs, source_actions, task_id, policy.source)
    candidate_score = _score_actions(policy, obs, candidate_actions, task_id, policy.candidate)
    margin = float(policy.checkpoint.get("candidate_margin", 0.0))
    if candidate_score > source_score + margin:
        policy.last_choice = "candidate"
        return candidate_actions
    policy.last_choice = "source"
    return source_actions


def commit_steps(
    policy: SelectorPolicy,
    *,
    task: dict | None = None,
    action_chunk: np.ndarray | None = None,
    default_commit_steps: int = 16,
) -> int:
    return bc5_inference.commit_steps(
        policy.source,
        task=task,
        action_chunk=action_chunk,
        default_commit_steps=default_commit_steps,
    )


@torch.no_grad()
def _score_actions(policy: SelectorPolicy, obs: dict, actions: np.ndarray, task_id: int, subpolicy: Any) -> float:
    stats = policy.wm["stats"]
    model = policy.wm["model"]
    raw_state = torch.as_tensor(np.asarray(obs["proprio"], dtype=np.float32)[None], dtype=torch.float32, device=policy.device)
    wm_state_dim = int(policy.wm["config"]["state_dim"])
    raw_state = raw_state[:, :wm_state_dim]
    state = (raw_state - stats["state_mean"]) / stats["state_std"].clamp_min(1e-6)
    task_t = torch.as_tensor([int(task_id)], dtype=torch.long, device=policy.device)
    progress_step = int(getattr(subpolicy, "history_step_idx", 0))
    progress = min(max(float(progress_step) / max(float(policy.checkpoint.get("wm_progress_scale", 260.0)), 1.0), 0.0), 1.0)
    progress_t = torch.full((1, 1), progress, dtype=torch.float32, device=policy.device)
    score = torch.zeros((), dtype=torch.float32, device=policy.device)
    horizon = min(int(policy.checkpoint.get("wm_rollout_horizon", 4)), int(actions.shape[0]))
    for step in range(max(1, horizon)):
        action = torch.as_tensor(actions[step : step + 1], dtype=torch.float32, device=policy.device)
        action_norm = (action - stats["action_mean"]) / stats["action_std"].clamp_min(1e-6)
        out = model(state, action_norm, task_t, progress_t)
        success = torch.sigmoid(out["success_logit"]).mean()
        reward = out["reward"].mean()
        next_progress = out["next_progress"].clamp(0.0, 1.0)
        progress_gain = (next_progress - progress_t).mean()
        score = score + (
            float(policy.checkpoint.get("wm_success_weight", 1.0)) * success
            + float(policy.checkpoint.get("wm_reward_weight", 0.2)) * reward
            + float(policy.checkpoint.get("wm_progress_weight", 0.4)) * progress_gain
        )
        state = out["next_state"]
        progress_t = next_progress
    score = score / float(max(1, horizon))
    return float(score.detach().cpu())
