from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robotwin_bc3.common import DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT
from tasks.robotwin_bc3.offline_data import load_robotwin_arrays, save_json
from tasks.robotwin_bc3.offline_policy import StateChunkPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight offline RoboTwin BC-3 baseline.")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR / "offline_state_chunk_5min"))
    parser.add_argument("--train-demos-per-task", type=int, default=50)
    parser.add_argument("--val-demos-per-task", type=int, default=50)
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_robotwin_arrays(
        split_path=args.split,
        repo_id=args.repo_id or None,
        train_demos_per_task=int(args.train_demos_per_task),
        val_demos_per_task=int(args.val_demos_per_task),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
    )
    stats = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in data["stats"].items()}
    train = _to_tensors(data["train"], device)
    val = _to_tensors(data["val"], device)
    model = StateChunkPolicy(
        state_dim=train["state"].shape[-1],
        action_dim=train["action"].shape[-1],
        num_tasks=len(data["task_names"]),
        horizon=int(args.chunk_horizon),
        width=int(args.width),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best = {"val_loss": float("inf"), "step": 0}
    history = []
    start = time.monotonic()
    n = train["state"].shape[0]
    for step in range(1, int(args.steps) + 1):
        if float(args.max_train_seconds) > 0 and time.monotonic() - start >= float(args.max_train_seconds):
            break
        idx = torch.randint(0, n, (int(args.batch_size),), device=device)
        pred = model(
            _norm(train["state"][idx], stats["state_mean"], stats["state_std"]),
            train["task_id"][idx],
            train["progress"][idx],
        )
        target = _norm(train["action"][idx], stats["action_mean"], stats["action_std"])
        loss = F.smooth_l1_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % int(args.log_interval) == 0:
            metrics = _eval(model, val, stats, int(args.batch_size))
            row = {"step": step, "train_loss": float(loss.item()), **metrics, "elapsed_s": time.monotonic() - start}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["val_mse_norm"] < best["val_loss"]:
                best = {"val_loss": float(metrics["val_mse_norm"]), "step": int(step)}
                _save(out_dir / "policy_best.pt", model, data, args, history, best)

    final_metrics = _eval(model, val, stats, int(args.batch_size))
    _save(out_dir / "policy.pt", model, data, args, history, best)
    payload = {
        "task": "robotwin_bc3_offline",
        "checkpoint": str(out_dir / "policy.pt"),
        "best_checkpoint": str(out_dir / "policy_best.pt"),
        "best_step": int(best["step"]),
        "best_val_mse_norm": float(best["val_loss"]),
        "final_metrics": final_metrics,
        "train_seconds": time.monotonic() - start,
        "steps_completed": int(history[-1]["step"] if history else 0),
        "train_samples": int(train["state"].shape[0]),
        "val_samples": int(val["state"].shape[0]),
        "task_names": data["task_names"],
        "train_demos_per_task": int(args.train_demos_per_task),
        "val_demos_per_task": int(args.val_demos_per_task),
    }
    save_json(out_dir / "train_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state": torch.tensor(batch["state"], dtype=torch.float32, device=device),
        "action": torch.tensor(batch["action"], dtype=torch.float32, device=device),
        "task_id": torch.tensor(batch["task_id"], dtype=torch.long, device=device),
        "progress": torch.tensor(batch["progress"], dtype=torch.float32, device=device),
    }


def _norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


@torch.no_grad()
def _eval(model: StateChunkPolicy, val: dict[str, torch.Tensor], stats: dict[str, torch.Tensor], batch_size: int) -> dict[str, float]:
    model.eval()
    losses = []
    raw_losses = []
    for start in range(0, val["state"].shape[0], batch_size):
        end = min(start + batch_size, val["state"].shape[0])
        pred = model(
            _norm(val["state"][start:end], stats["state_mean"], stats["state_std"]),
            val["task_id"][start:end],
            val["progress"][start:end],
        )
        target = _norm(val["action"][start:end], stats["action_mean"], stats["action_std"])
        losses.append(F.mse_loss(pred, target, reduction="sum"))
        pred_raw = pred * stats["action_std"] + stats["action_mean"]
        raw_losses.append(F.mse_loss(pred_raw, val["action"][start:end], reduction="sum"))
    denom = float(np.prod(val["action"].shape))
    model.train()
    return {
        "val_mse_norm": float(torch.stack(losses).sum().item() / denom),
        "val_mse_raw": float(torch.stack(raw_losses).sum().item() / denom),
    }


def _save(path: Path, model: StateChunkPolicy, data: dict, args: argparse.Namespace, history: list[dict], best: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "stats": data["stats"],
            "task_names": data["task_names"],
            "chunk_horizon": int(args.chunk_horizon),
            "frame_stride": int(args.frame_stride),
            "width": int(args.width),
            "depth": int(args.depth),
            "dropout": float(args.dropout),
            "history": history,
            "best": best,
        },
        path,
    )


if __name__ == "__main__":
    main()
