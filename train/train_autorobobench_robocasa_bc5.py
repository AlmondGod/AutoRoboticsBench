from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from autorobobench.robocasa_runtime import ensure_robocasa_runtime
from models.robocasa_sequence_flow import RoboCasaHistoryACTPolicy, RoboCasaSequenceFlowPolicy
from train.common import device_from_arg


ensure_robocasa_runtime()

from eval.train_temporal_chunk_bc_robocasa import (  # noqa: E402
    RoboCasaTemporalChunkBC,
    TemporalChunkData,
    _augment,
    _batch,
    _chunk_weights,
    _concat_parts,
    _episode_samples,
    _eval_loss,
    _flow_matching_loss,
    _masked_chunk_loss,
    _masked_mean_std,
    _mean_std,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AutoroboBench RoboCasa BC-5 baseline policy.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_bc5/baseline")
    parser.add_argument("--train-episodes-per-task", type=int, default=4)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-noise", type=float, default=0.01)
    parser.add_argument("--proprio-noise", type=float, default=0.01)
    parser.add_argument("--action-smooth", type=float, default=0.001)
    parser.add_argument("--policy-kind", choices=["bc", "flow", "sequence_flow", "history_act"], default="bc")
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-sigma", type=float, default=1.0)
    parser.add_argument("--flow-source", choices=["noise", "bc"], default="noise")
    parser.add_argument("--flow-eval-start", choices=["zero", "noise", "bc"], default="bc")
    parser.add_argument("--bc-aux-weight", type=float, default=0.1)
    parser.add_argument("--chunk-decay", type=float, default=1.0)
    parser.add_argument("--transformer-depth", type=int, default=3)
    parser.add_argument("--action-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--history-stride", type=int, default=16)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task_aliases = set(args.task_alias)
    train_data, val_data, split_summary = load_split_data(
        manifest,
        split,
        task_aliases=task_aliases,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("need both train and val samples for RoboCasa BC-5")

    proprio_mean, proprio_std = _mean_std(train_data.proprio)
    action_mean, action_std = _masked_mean_std(train_data.actions, train_data.mask)
    train_data.proprio = ((train_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val_data.proprio = ((val_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    train_data.actions = ((train_data.actions - action_mean) / action_std).astype(np.float32)
    val_data.actions = ((val_data.actions - action_mean) / action_std).astype(np.float32)
    if args.policy_kind == "history_act":
        _attach_history(train_data, int(args.history_stride))
        _attach_history(val_data, int(args.history_stride))

    device = device_from_arg(args.device)
    task_count = max(1, int(max(train_data.task_id.max(initial=0), val_data.task_id.max(initial=0)) + 1))
    if args.policy_kind == "history_act":
        model = RoboCasaHistoryACTPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    elif args.policy_kind == "sequence_flow":
        model = RoboCasaSequenceFlowPolicy(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            depth=int(args.transformer_depth),
            action_depth=int(args.action_depth),
            heads=int(args.heads),
            dropout=float(args.dropout),
        ).to(device)
    else:
        model = RoboCasaTemporalChunkBC(
            proprio_dim=int(train_data.proprio.shape[-1]),
            chunk_horizon=int(args.chunk_horizon),
            action_dim=int(train_data.actions.shape[-1]),
            task_count=task_count,
            width=int(args.width),
            dropout=float(args.dropout),
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val_loss = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.monotonic()

    for step in range(1, int(args.steps) + 1):
        if args.max_train_seconds > 0 and time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        idx = _sample_indices(train_data, int(args.batch_size), rng, balanced=bool(args.balanced_sampling))
        if args.policy_kind == "history_act":
            batch = _history_batch(train_data, idx, device)
            batch = _augment_history(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(
                batch["prev_agent"],
                batch["prev_wrist"],
                batch["agent"],
                batch["wrist"],
                batch["prev_proprio"],
                batch["proprio"],
                batch["task_id"],
            )
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        elif args.policy_kind == "sequence_flow":
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _sequence_flow_matching_loss(
                model,
                batch,
                sigma=float(args.flow_sigma),
                flow_source=str(args.flow_source),
                chunk_decay=float(args.chunk_decay),
                bc_aux_weight=float(args.bc_aux_weight),
            )
        elif args.policy_kind == "flow":
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            loss = _flow_matching_loss(model, batch, sigma=float(args.flow_sigma), chunk_decay=float(args.chunk_decay))
        else:
            batch = _batch(train_data, idx, device)
            batch = _augment(batch, float(args.image_noise), float(args.proprio_noise))
            pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            loss = _masked_chunk_loss(pred, batch["actions"], batch["mask"], chunk_decay=float(args.chunk_decay))
            if args.action_smooth > 0 and pred.shape[1] > 1:
                loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        row = {"step": step, "train_loss": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time}
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_loss = _eval_policy_loss(
                model,
                val_data,
                device,
                batch_size=max(64, int(args.batch_size)),
                policy_kind=str(args.policy_kind),
                flow_steps=int(args.flow_steps),
                flow_eval_start=str(args.flow_eval_start),
            )
            row["val_loss"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"step={step} train_loss={row['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
        history.append(row)

    final_val_loss = _eval_policy_loss(
        model,
        val_data,
        device,
        batch_size=max(64, int(args.batch_size)),
        policy_kind=str(args.policy_kind),
        flow_steps=int(args.flow_steps),
        flow_eval_start=str(args.flow_eval_start),
    )
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_step = int(history[-1]["step"] if history else 0)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "policy_type": (
            "autorobobench_robocasa_bc5_history_act"
            if args.policy_kind == "history_act"
            else "autorobobench_robocasa_bc5_sequence_flow"
            if args.policy_kind == "sequence_flow"
            else "autorobobench_robocasa_bc5_temporal_chunk"
        ),
        "chunk_horizon": int(args.chunk_horizon),
        "action_dim": int(train_data.actions.shape[-1]),
        "proprio_dim": int(train_data.proprio.shape[-1]),
        "task_count": task_count,
        "width": int(args.width),
        "dropout": float(args.dropout),
        "policy_kind": str(args.policy_kind),
        "flow_steps": int(args.flow_steps),
        "flow_sigma": float(args.flow_sigma),
        "flow_source": str(args.flow_source),
        "flow_eval_start": str(args.flow_eval_start),
        "bc_aux_weight": float(args.bc_aux_weight),
        "chunk_decay": float(args.chunk_decay),
        "transformer_depth": int(args.transformer_depth),
        "action_depth": int(args.action_depth),
        "heads": int(args.heads),
        "history_stride": int(args.history_stride),
        "condition_on_robocasa_task_index": False,
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "manifest": str(Path(args.manifest)),
        "split": str(Path(args.split)),
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    torch.save(checkpoint, out_dir / "policy.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_action_mse_normalized"] = float(best_val_loss)
    torch.save(best_checkpoint, out_dir / "policy_best.pt")

    metrics = {
        "checkpoint": str(out_dir / "policy.pt"),
        "best_checkpoint": str(out_dir / "policy_best.pt"),
        "best_step": int(best_step),
        "best_val_action_mse_normalized": float(best_val_loss),
        "final_val_action_mse_normalized": float(final_val_loss),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "split_summary": split_summary,
        "chunk_horizon": int(args.chunk_horizon),
        "frame_stride": int(args.frame_stride),
        "train_episodes_per_task": int(args.train_episodes_per_task),
        "val_episodes_per_task": int(args.val_episodes_per_task),
        "steps_completed": int(history[-1]["step"] if history else 0),
        "train_seconds": float(time.monotonic() - start_time),
        "width": int(args.width),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "image_noise": float(args.image_noise),
        "proprio_noise": float(args.proprio_noise),
        "action_smooth": float(args.action_smooth),
        "policy_kind": str(args.policy_kind),
        "flow_steps": int(args.flow_steps),
        "flow_sigma": float(args.flow_sigma),
        "flow_source": str(args.flow_source),
        "flow_eval_start": str(args.flow_eval_start),
        "bc_aux_weight": float(args.bc_aux_weight),
        "chunk_decay": float(args.chunk_decay),
        "transformer_depth": int(args.transformer_depth),
        "action_depth": int(args.action_depth),
        "heads": int(args.heads),
        "history_stride": int(args.history_stride),
        "balanced_sampling": bool(args.balanced_sampling),
        "seed": int(args.seed),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def load_split_data(
    manifest: dict,
    split: dict,
    *,
    task_aliases: set[str],
    train_episodes_per_task: int,
    val_episodes_per_task: int,
    chunk_horizon: int,
    frame_stride: int,
) -> tuple[TemporalChunkData, TemporalChunkData, list[dict]]:
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    train_parts: list[dict[str, np.ndarray]] = []
    val_parts: list[dict[str, np.ndarray]] = []
    summary: list[dict] = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        task = manifest_tasks[alias]
        task_id = int(split_task["task_id"])
        dataset_root = Path(task["dataset_path"])
        train_ids = list(split_task["train_episode_ids"])
        val_ids = list(split_task["val_episode_ids"])
        if train_episodes_per_task > 0:
            train_ids = train_ids[:train_episodes_per_task]
        if val_episodes_per_task > 0:
            val_ids = val_ids[:val_episodes_per_task]
        for episode_id in train_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{int(episode_id):06d}.parquet"
            train_parts.append(
                _episode_samples(
                    dataset_root,
                    episode_path,
                    int(episode_id),
                    task_id,
                    chunk_horizon,
                    frame_stride,
                    False,
                )
            )
        for episode_id in val_ids:
            episode_path = dataset_root / "data" / "chunk-000" / f"episode_{int(episode_id):06d}.parquet"
            val_parts.append(
                _episode_samples(
                    dataset_root,
                    episode_path,
                    int(episode_id),
                    task_id,
                    chunk_horizon,
                    frame_stride,
                    False,
                )
            )
        summary.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str(dataset_root),
                "train_episode_ids": [int(x) for x in train_ids],
                "val_episode_ids": [int(x) for x in val_ids],
            }
        )
        print(f"loaded {alias}: train={train_ids} val={val_ids}", flush=True)
    return _concat_parts(train_parts), _concat_parts(val_parts), summary


def _sample_indices(
    data: TemporalChunkData,
    batch_size: int,
    rng: np.random.Generator,
    *,
    balanced: bool,
) -> np.ndarray:
    if not balanced:
        return rng.integers(0, len(data), size=batch_size)
    task_ids = np.unique(data.task_id)
    if len(task_ids) == 0:
        return rng.integers(0, len(data), size=batch_size)
    per_task = int(math.ceil(batch_size / len(task_ids)))
    parts: list[np.ndarray] = []
    for task_id in task_ids:
        pool = np.flatnonzero(data.task_id == int(task_id))
        if len(pool) == 0:
            continue
        parts.append(rng.choice(pool, size=per_task, replace=len(pool) < per_task))
    if not parts:
        return rng.integers(0, len(data), size=batch_size)
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    if len(idx) < batch_size:
        extra = rng.integers(0, len(data), size=batch_size - len(idx))
        idx = np.concatenate([idx, extra])
    return idx[:batch_size]


def _attach_history(data: TemporalChunkData, history_stride: int) -> None:
    history_stride = max(0, int(history_stride))
    prev_idx = np.arange(len(data), dtype=np.int64)
    for episode_id in np.unique(data.episode_idx):
        rows = np.flatnonzero(data.episode_idx == int(episode_id))
        if len(rows) == 0:
            continue
        order = rows[np.argsort(data.frame_idx[rows])]
        frames = data.frame_idx[order]
        if history_stride <= 0:
            prev_idx[order] = order
            continue
        targets = frames - history_stride
        positions = np.searchsorted(frames, targets, side="right") - 1
        positions = np.maximum(positions, 0)
        prev_idx[order] = order[positions]
    data.prev_agent = data.agent[prev_idx]
    data.prev_wrist = data.wrist[prev_idx]
    data.prev_proprio = data.proprio[prev_idx]


def _history_batch(data: TemporalChunkData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "prev_agent": torch.as_tensor(data.prev_agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "prev_wrist": torch.as_tensor(data.prev_wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "agent": torch.as_tensor(data.agent[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(data.wrist[idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "prev_proprio": torch.as_tensor(data.prev_proprio[idx], dtype=torch.float32, device=device),
        "proprio": torch.as_tensor(data.proprio[idx], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data.actions[idx], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(data.mask[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


def _augment_history(batch: dict[str, torch.Tensor], image_noise: float, proprio_noise: float) -> dict[str, torch.Tensor]:
    batch = _augment(batch, image_noise, proprio_noise)
    if image_noise > 0:
        scale = 255.0 * image_noise
        batch["prev_agent"] = (batch["prev_agent"] + torch.randn_like(batch["prev_agent"]) * scale).clamp(0.0, 255.0)
        batch["prev_wrist"] = (batch["prev_wrist"] + torch.randn_like(batch["prev_wrist"]) * scale).clamp(0.0, 255.0)
    if proprio_noise > 0:
        batch["prev_proprio"] = batch["prev_proprio"] + torch.randn_like(batch["prev_proprio"]) * proprio_noise
    return batch


def _sequence_flow_matching_loss(
    model: RoboCasaSequenceFlowPolicy,
    batch: dict[str, torch.Tensor],
    *,
    sigma: float,
    flow_source: str,
    chunk_decay: float,
    bc_aux_weight: float,
) -> torch.Tensor:
    actions = batch["actions"]
    context = model.encode_obs(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
    if flow_source == "bc":
        source = model.bc_action(context).detach()
    else:
        source = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    action_t = (1.0 - t.reshape(-1, 1, 1)) * source + t.reshape(-1, 1, 1) * actions
    target_velocity = actions - source
    pred_velocity = model.flow_velocity(context, action_t, t)
    weights = _chunk_weights(actions.shape[1], chunk_decay, actions.device, actions.dtype)
    per_step = (pred_velocity - target_velocity).square().mean(dim=-1)
    loss = (per_step * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
    if bc_aux_weight > 0:
        pred_bc = model.bc_action(context)
        loss = loss + float(bc_aux_weight) * _masked_chunk_loss(
            pred_bc,
            actions,
            batch["mask"],
            chunk_decay=chunk_decay,
        )
    return loss


def _eval_policy_loss(
    model: nn.Module,
    data: TemporalChunkData,
    device: torch.device,
    batch_size: int,
    *,
    policy_kind: str,
    flow_steps: int,
    flow_eval_start: str,
) -> float:
    if policy_kind == "history_act":
        model.eval()
        total = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                idx = np.arange(start, min(len(data), start + batch_size))
                batch = _history_batch(data, idx, device)
                pred = model(
                    batch["prev_agent"],
                    batch["prev_wrist"],
                    batch["agent"],
                    batch["wrist"],
                    batch["prev_proprio"],
                    batch["proprio"],
                    batch["task_id"],
                )
                per_step = (pred - batch["actions"]).square().mean(dim=-1)
                total = total + (per_step * batch["mask"]).sum()
                denom = denom + batch["mask"].sum()
        model.train()
        return float((total / denom.clamp_min(1.0)).detach().cpu())
    if policy_kind != "sequence_flow":
        return _eval_loss(
            model,
            data,
            device,
            batch_size=batch_size,
            policy_kind=policy_kind,
            flow_steps=flow_steps,
        )
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            pred = model.sample_flow(
                batch["agent"],
                batch["wrist"],
                batch["proprio"],
                batch["task_id"],
                steps=flow_steps,
                start=flow_eval_start,
            )
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


if __name__ == "__main__":
    main()
