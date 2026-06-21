from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_world_model.data import TransitionData, load_transition_data
from train.common import device_from_arg


class ProgressMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, task_count: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        self.task_emb = nn.Embedding(task_count, min(32, max(4, hidden_dim // 8)))
        in_dim = state_dim + action_dim + self.task_emb.embedding_dim
        layers: list[nn.Module] = []
        width = int(hidden_dim)
        for layer_idx in range(max(1, int(depth))):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else width, width))
            layers.append(nn.SiLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        task = self.task_emb(task_id.long())
        x = torch.cat([state, action, task], dim=-1)
        return torch.sigmoid(self.net(x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight RoboCasa task-progress predictor.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_progress_predictor/default")
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = device_from_arg(args.device)
    train_data, val_data, split_summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        task_aliases=set(args.task_alias),
        frame_stride=int(args.frame_stride),
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("need both train and val progress samples")

    stats = _stats(train_data)
    train_x = _normalize(train_data, stats)
    val_x = _normalize(val_data, stats)
    task_count = int(max(train_data.task_id.max(initial=0), val_data.task_id.max(initial=0)) + 1)
    model = ProgressMLP(
        state_dim=int(train_x["state"].shape[-1]),
        action_dim=int(train_x["action"].shape[-1]),
        task_count=task_count,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    best_r2 = -math.inf
    best_state = None
    start_time = time.monotonic()

    for step in range(1, int(args.steps) + 1):
        if float(args.max_train_seconds) > 0 and time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        idx = rng.integers(0, len(train_data), size=int(args.batch_size))
        batch = _batch(train_x, train_data, idx, device)
        pred = model(batch["state"], batch["action"], batch["task_id"])
        loss = F.mse_loss(pred, batch["target"])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            metrics = _evaluate(model, val_x, val_data, device=device, batch_size=max(512, int(args.batch_size)))
            row = {"step": int(step), "loss": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time}
            row.update({f"val_{key}": value for key, value in metrics.items()})
            history.append(row)
            if float(metrics["r2"]) > best_r2:
                best_r2 = float(metrics["r2"])
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            print(
                "step={step} loss={loss:.6f} val_mae={mae:.4f} val_rmse={rmse:.4f} val_r2={r2:.4f}".format(
                    step=step,
                    loss=float(loss.detach().cpu()),
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    r2=metrics["r2"],
                ),
                flush=True,
            )

    final = _evaluate(model, val_x, val_data, device=device, batch_size=max(512, int(args.batch_size)))
    baseline = _task_mean_baseline(train_data, val_data)
    monotonicity = _episode_monotonicity(model, val_x, val_data, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "task": "robocasa_progress_predictor",
        "state_dict": best_state or {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "state_dim": int(train_x["state"].shape[-1]),
        "action_dim": int(train_x["action"].shape[-1]),
        "task_count": int(task_count),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "stats": stats,
        "split_summary": split_summary,
    }
    torch.save(checkpoint, out_dir / "progress_predictor.pt")
    metrics = {
        "task": "robocasa_progress_predictor",
        "checkpoint": str(out_dir / "progress_predictor.pt"),
        "train_seconds": float(time.monotonic() - start_time),
        "steps_completed": int(history[-1]["step"] if history else 0),
        "train_samples": int(len(train_data)),
        "val_samples": int(len(val_data)),
        "val_mae": float(final["mae"]),
        "val_rmse": float(final["rmse"]),
        "val_r2": float(final["r2"]),
        "baseline_rmse": float(baseline["rmse"]),
        "baseline_mae": float(baseline["mae"]),
        "monotonicity": float(monotonicity),
        "history": history,
        "split_summary": split_summary,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _stats(data: TransitionData) -> dict[str, np.ndarray]:
    return {
        "state_mean": data.state.mean(axis=0).astype(np.float32),
        "state_std": np.maximum(data.state.std(axis=0), 1e-6).astype(np.float32),
        "action_mean": data.action.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(data.action.std(axis=0), 1e-6).astype(np.float32),
    }


def _normalize(data: TransitionData, stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "state": ((data.state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
        "action": ((data.action - stats["action_mean"]) / stats["action_std"]).astype(np.float32),
        "target": data.progress.astype(np.float32),
    }


def _batch(x: dict[str, np.ndarray], data: TransitionData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state": torch.as_tensor(x["state"][idx], dtype=torch.float32, device=device),
        "action": torch.as_tensor(x["action"][idx], dtype=torch.float32, device=device),
        "target": torch.as_tensor(x["target"][idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, x: dict[str, np.ndarray], data: TransitionData, *, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    preds = []
    for start in range(0, len(data), int(batch_size)):
        idx = np.arange(start, min(len(data), start + int(batch_size)))
        batch = _batch(x, data, idx, device)
        preds.append(model(batch["state"], batch["action"], batch["task_id"]).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).reshape(-1)
    target = data.progress.reshape(-1).astype(np.float32)
    err = pred - target
    mse = float(np.mean(err * err))
    var = float(np.var(target))
    model.train()
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "r2": float(1.0 - mse / max(var, 1e-8)),
    }


def _task_mean_baseline(train_data: TransitionData, val_data: TransitionData) -> dict[str, float]:
    means = {}
    global_mean = float(train_data.progress.mean())
    for task_id in np.unique(train_data.task_id):
        means[int(task_id)] = float(train_data.progress[train_data.task_id == int(task_id)].mean())
    pred = np.asarray([means.get(int(task_id), global_mean) for task_id in val_data.task_id], dtype=np.float32)
    target = val_data.progress.reshape(-1).astype(np.float32)
    err = pred - target
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(math.sqrt(float(np.mean(err * err))))}


@torch.no_grad()
def _episode_monotonicity(model: nn.Module, x: dict[str, np.ndarray], data: TransitionData, *, device: torch.device) -> float:
    model.eval()
    scores = []
    for episode_id in np.unique(data.episode_id):
        rows = np.flatnonzero(data.episode_id == int(episode_id))
        if len(rows) < 2:
            continue
        rows = rows[np.argsort(data.frame_idx[rows])]
        batch = _batch(x, data, rows, device)
        pred = model(batch["state"], batch["action"], batch["task_id"]).detach().cpu().numpy().reshape(-1)
        scores.append(float(np.mean(np.diff(pred) >= -1e-3)))
    model.train()
    if not scores:
        return 0.0
    return float(np.mean(scores))


if __name__ == "__main__":
    main()
