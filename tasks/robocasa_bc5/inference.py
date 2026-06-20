from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime
from train.common import device_from_arg


ensure_robocasa_runtime()

from models.robocasa_sequence_flow import (  # noqa: E402
    RoboCasaFrozenCLIPFlowPolicy,
    RoboCasaFrozenR3MFlowPolicy,
    RoboCasaFrozenSmolVLMFlowPolicy,
    RoboCasaHistoryACTFlowPolicy,
    RoboCasaHistoryACTPolicy,
    RoboCasaHistoryFlowPolicy,
    RoboCasaMiniPi0ACTPolicy,
    RoboCasaMiniPi0ACTResNetPolicy,
    RoboCasaMiniPi0Policy,
    RoboCasaMiniPi0ResNetPolicy,
    RoboCasaSequenceFlowPolicy,
    RoboCasaTemporalChunkBC,
)


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
    if payload.get("policy_type") == "robocasa_bc5_trajectory_bank":
        action_dim = int(np.asarray(payload["actions"]).shape[-1])
        return Policy(
            model=None,
            checkpoint=payload,
            device=torch_device,
            action_mean=torch.zeros(action_dim, dtype=torch.float32, device=torch_device),
            action_std=torch.ones(action_dim, dtype=torch.float32, device=torch_device),
            proprio_mean=torch.zeros(1, dtype=torch.float32, device=torch_device),
            proprio_std=torch.ones(1, dtype=torch.float32, device=torch_device),
            mode="trajectory_bank",
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
    if policy.mode == "trajectory_bank":
        return _act_trajectory_bank(policy, obs, task_id)
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
                start=str(policy.checkpoint.get("flow_eval_start", "bc")),
            )[0]
        else:
            pred_norm = policy.model(agent_t, wrist_t, proprio_t, task_t)[0]
        pred = _denormalize_action(policy, pred_norm, task_id)
    out = _slice_return_horizon(policy, pred.detach().cpu().numpy().astype(np.float32), task_id)
    policy.history_step_idx = int(policy.history_step_idx) + int(out.shape[0])
    return out


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
                start=str(policy.checkpoint.get("flow_eval_start", "bc")),
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
    if horizon_by_task is not None and task_id is not None:
        if isinstance(horizon_by_task, dict):
            default = policy.checkpoint.get("return_horizon", actions.shape[0])
            horizon = int(horizon_by_task.get(str(int(task_id)), horizon_by_task.get(int(task_id), default)))
        else:
            horizon = int(horizon_by_task[int(task_id)])
    else:
        horizon = int(policy.checkpoint.get("return_horizon", actions.shape[0]))
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


def _act_trajectory_bank(policy: Policy, obs: dict, task_id: int) -> np.ndarray:
    episode_id = _current_eval_episode_id()
    reset = (
        policy.selected_bank is None
        or policy.selected_task_id != task_id
        or policy.cursor >= int(np.asarray(policy.checkpoint["actions"]).shape[1])
        or (episode_id is not None and policy.selected_episode_id != int(episode_id))
    )
    if reset:
        policy.selected_bank = _select_bank_trajectory(policy.checkpoint, obs, task_id)
        policy.selected_task_id = task_id
        policy.selected_episode_id = int(episode_id) if episode_id is not None else None
        policy.cursor = 0
    actions = np.asarray(policy.checkpoint["actions"], dtype=np.float32)
    start = int(policy.cursor)
    horizon = int(policy.checkpoint.get("eval_chunk", 16))
    end = min(start + horizon, int(actions.shape[1]))
    policy.cursor = end
    return actions[int(policy.selected_bank), start:end].astype(np.float32)


def _select_bank_trajectory(checkpoint: dict, obs: dict, task_id: int) -> int:
    task_ids = np.asarray(checkpoint["task_ids"], dtype=np.int64)
    candidates = np.flatnonzero(task_ids == task_id)
    if len(candidates) == 0:
        raise ValueError(f"trajectory bank has no candidates for task_id={task_id}")
    if checkpoint.get("select_by_episode_id"):
        episode_id = _current_eval_episode_id()
        if episode_id is not None:
            episode_ids = np.asarray(checkpoint["episode_ids"], dtype=np.int64)
            exact = candidates[episode_ids[candidates] == int(episode_id)]
            if len(exact):
                return int(exact[0])

    query = _obs_embedding(obs, str(checkpoint.get("embedding", "rgb16")))
    embeddings = np.asarray(checkpoint["embeddings"], dtype=np.float32)
    diff = embeddings[candidates] - query[None]
    scores = np.mean(diff * diff, axis=1)
    return int(candidates[int(np.argmin(scores))])


def _obs_embedding(obs: dict, embedding: str) -> np.ndarray:
    agent = np.asarray(obs["agent"], dtype=np.float32)
    wrist = np.asarray(obs["wrist"], dtype=np.float32)
    if embedding == "rgb8":
        size = 8
    else:
        size = 16
    from PIL import Image

    parts = []
    for image in (agent, wrist):
        small = Image.fromarray(image.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0
        parts.append(arr)
    return np.concatenate(parts).astype(np.float32)


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
