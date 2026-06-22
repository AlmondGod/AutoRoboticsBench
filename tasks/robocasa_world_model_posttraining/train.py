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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_bc5.inference import load_policy
from tasks.robocasa_visual_world_model.model import VisualRoboCasaWorldModel
from tasks.robocasa_world_model.model import RoboCasaWorldModel
def device_from_arg(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)

from tasks.robocasa_bc5.train import (
    _append_progress_features,
    _batch,
    _checkpoint_state_dict,
    _eval_policy_loss,
    _sample_indices,
    load_split_data,
)


SUPPORTED_MODES = {"chunk", "sequence_flow"}
SUPPORTED_KINDS = {"bc", "flow", "sequence_flow"}
DEFAULT_MANIFEST = "data/autorobobench/robocasa_stand_mixer_peak_manifest.json"
DEFAULT_SPLIT = "data/autorobobench/robocasa_stand_mixer_peak_splits.json"
DEFAULT_POLICY_CHECKPOINT = "runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Posttrain a RoboCasa policy with a frozen world model.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--policy-checkpoint", default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_world_model_posttraining/default")
    parser.add_argument("--train-episodes-per-task", type=int, default=4)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--chunk-horizon", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
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
    if "image_size" in cfg or payload.get("task") == "robocasa_visual_world_model":
        model = VisualRoboCasaWorldModel(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            task_count=int(cfg["task_count"]),
            image_size=int(cfg.get("image_size", 32)),
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            task_dim=int(cfg["task_dim"]),
            latent_dim=int(cfg["latent_dim"]),
            visual_latent_dim=int(cfg.get("visual_latent_dim", 64)),
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
            task_dim=int(cfg["task_dim"]),
            latent_dim=int(cfg["latent_dim"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        wm_type = "state"
        has_visual = False
    model.load_state_dict(payload["model"])
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
    progress_t = progress.reshape(-1, 1).to(dtype=state.dtype, device=state.device)
    objective = torch.zeros((), dtype=state.dtype, device=state.device)
    success_terms = []
    progress_terms = []
    steps = max(1, min(int(horizon), int(actions_raw.shape[1])))
    for step in range(steps):
        action = (actions_raw[:, step] - stats["action_mean"]) / stats["action_std"].clamp_min(1e-6)
        out = model(state, action, task_id.long(), progress_t)
        success_prob = torch.sigmoid(out["success_logit"])
        next_progress = out["next_progress"].clamp(0.0, 1.0)
        progress_gain = next_progress - progress_t
        objective = objective + (
            float(success_weight) * success_prob.mean()
            + float(progress_weight) * progress_gain.mean()
        )
        success_terms.append(success_prob.mean())
        progress_terms.append(progress_gain.mean())
        state = out["next_state"]
        progress_t = next_progress
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
