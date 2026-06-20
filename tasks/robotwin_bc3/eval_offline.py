from __future__ import annotations

import argparse
import json
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

from tasks.robotwin_bc3.common import DEFAULT_SPLIT
from tasks.robotwin_bc3.offline_data import load_robotwin_arrays, save_json
from tasks.robotwin_bc3.offline_policy import StateChunkPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline RoboTwin BC-3 action prediction.")
    parser.add_argument("--checkpoint", "--policy", dest="checkpoint", required=True)
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--train-demos-per-task", type=int, default=50)
    parser.add_argument("--val-demos-per-task", type=int, default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.monotonic()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    data = load_robotwin_arrays(
        split_path=args.split,
        repo_id=args.repo_id or None,
        train_demos_per_task=int(args.train_demos_per_task),
        val_demos_per_task=int(args.val_demos_per_task),
        chunk_horizon=int(ckpt["chunk_horizon"]),
        frame_stride=int(ckpt["frame_stride"]),
    )
    device = torch.device(args.device)
    val = {
        "state": torch.tensor(data["val"]["state"], dtype=torch.float32, device=device),
        "action": torch.tensor(data["val"]["action"], dtype=torch.float32, device=device),
        "task_id": torch.tensor(data["val"]["task_id"], dtype=torch.long, device=device),
        "progress": torch.tensor(data["val"]["progress"], dtype=torch.float32, device=device),
        "episode_id": data["val"]["episode_id"],
    }
    stats = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in ckpt["stats"].items()}
    model = StateChunkPolicy(
        state_dim=val["state"].shape[-1],
        action_dim=val["action"].shape[-1],
        num_tasks=len(ckpt["task_names"]),
        horizon=int(ckpt["chunk_horizon"]),
        width=int(ckpt["width"]),
        depth=int(ckpt["depth"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pred = _predict(model, val, stats, int(args.batch_size))
    target = val["action"]
    norm_target = _norm(target, stats["action_mean"], stats["action_std"])
    norm_pred = _norm(pred, stats["action_mean"], stats["action_std"])
    per_task = {}
    for idx, name in enumerate(ckpt["task_names"]):
        mask = val["task_id"] == idx
        if not bool(mask.any()):
            continue
        per_task[name] = {
            "samples": int(mask.sum().item()),
            "mse_norm": float(F.mse_loss(norm_pred[mask], norm_target[mask]).item()),
            "mse_raw": float(F.mse_loss(pred[mask], target[mask]).item()),
        }
    payload = {
        "task": "robotwin_bc3_offline",
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "metric": "heldout_action_mse_norm",
        "heldout_action_mse_norm": float(F.mse_loss(norm_pred, norm_target).item()),
        "heldout_action_mse_raw": float(F.mse_loss(pred, target).item()),
        "score": float(1.0 / (1.0 + F.mse_loss(norm_pred, norm_target).item())),
        "per_task": per_task,
        "samples": int(target.shape[0]),
        "eval_seconds": time.monotonic() - start,
        "simulator_eval_available": False,
        "simulator_eval_note": "Closed-loop RoboTwin simulator eval is intentionally outside the minimal offline install.",
    }
    save_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


@torch.no_grad()
def _predict(
    model: StateChunkPolicy,
    val: dict[str, torch.Tensor | np.ndarray],
    stats: dict[str, torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    outs = []
    for start in range(0, val["state"].shape[0], batch_size):
        end = min(start + batch_size, val["state"].shape[0])
        outs.append(
            model(
                _norm(val["state"][start:end], stats["state_mean"], stats["state_std"]),
                val["task_id"][start:end],
                val["progress"][start:end],
            )
            * stats["action_std"]
            + stats["action_mean"]
        )
    return torch.cat(outs, dim=0)


def _norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


if __name__ == "__main__":
    main()
