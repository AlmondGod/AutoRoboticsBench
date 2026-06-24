from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tasks.robocasa_bc5.model import RoboCasaTemporalChunkBC
def device_from_arg(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)



@dataclass
class Policy:
    model: RoboCasaTemporalChunkBC
    checkpoint: dict
    device: torch.device
    action_mean: torch.Tensor
    action_std: torch.Tensor
    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor


def _tensor(payload: dict, key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(payload[key], dtype=torch.float32, device=device)


def load_policy(checkpoint: str, device: str = "auto") -> Policy:
    torch_device = device_from_arg(device)
    payload = torch.load(Path(checkpoint), map_location=torch_device, weights_only=False)
    policy_type = str(payload.get("policy_type", ""))
    if policy_type != "autorobobench_robocasa_offlinerl_posttraining":
        raise ValueError(f"expected robocasa_offlinerl_posttraining checkpoint, got policy_type={policy_type!r}")
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
    device = policy.device
    task_id = int(task["task_id"])
    if task_id < 0 or task_id >= int(policy.checkpoint["task_count"]):
        raise ValueError(f"task_id={task_id} outside loaded policy task_count={policy.checkpoint['task_count']}")
    advantage = float(policy.checkpoint.get("recap_eval_advantage", 1.0))
    raw_proprio = np.asarray(obs["proprio"], dtype=np.float32)
    expected_raw_dim = int(policy.checkpoint.get("raw_proprio_dim", raw_proprio.shape[-1]))
    if raw_proprio.shape[-1] != expected_raw_dim:
        raise ValueError(f"expected raw proprio dim {expected_raw_dim}, got {raw_proprio.shape[-1]}")
    proprio = np.concatenate([raw_proprio, np.asarray([advantage], dtype=np.float32)], axis=0)

    with torch.no_grad():
        agent_t = torch.as_tensor(np.asarray(obs["agent"])[None].copy(), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        wrist_t = torch.as_tensor(np.asarray(obs["wrist"])[None].copy(), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        proprio_t = torch.as_tensor(proprio[None], dtype=torch.float32, device=device)
        proprio_t = (proprio_t - policy.proprio_mean) / policy.proprio_std
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        pred_norm = policy.model(agent_t, wrist_t, proprio_t, task_t)[0]
        pred = pred_norm * policy.action_std + policy.action_mean
    return pred.detach().cpu().numpy().astype(np.float32)


def commit_steps(
    policy: Policy,
    *,
    task: dict | None = None,
    action_chunk: np.ndarray | None = None,
    default_commit_steps: int = 8,
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
    if action_chunk is not None:
        return int(min(default_commit_steps, action_chunk.shape[0]))
    return int(default_commit_steps)
