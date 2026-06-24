from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0

from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    TransitionData,
    load_transition_data,
    load_video_frames,
    save_json,
)
def device_from_arg(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)



class VideoInverseDynamics(nn.Module):
    def __init__(self, *, action_dim: int, task_count: int, task_dim: int = 32, width: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task = nn.Embedding(int(task_count), int(task_dim))
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(192 + int(task_dim) + 1, int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(action_dim)),
        )

    def encode_pair(self, image_pair: torch.Tensor) -> torch.Tensor:
        return self.encoder(image_pair)

    def forward(self, image_pair: torch.Tensor, task_id: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        if progress.ndim == 1:
            progress = progress[:, None]
        h = self.encode_pair(image_pair)
        h = torch.cat([h, self.task(task_id.long()), progress.float()], dim=-1)
        return self.head(h)


def load_inverse_dynamics(checkpoint: str | Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = VideoInverseDynamics(
        action_dim=int(cfg["action_dim"]),
        task_count=int(cfg["task_count"]),
        task_dim=int(cfg["task_dim"]),
        width=int(cfg["width"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return {
        "model": model,
        "config": cfg,
        "action_mean": torch.as_tensor(payload["action_mean"], dtype=torch.float32, device=device),
        "action_std": torch.as_tensor(payload["action_std"], dtype=torch.float32, device=device),
        "device": device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RGB inverse dynamics on labeled RoboCasa rollouts.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_world_model/inverse_dynamics")
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--view", default="robot0_agentview_right")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-train-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--task-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")
    if float(args.max_train_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = device_from_arg(str(args.device))
    train_raw, val_raw, summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        task_aliases=set(args.task_alias),
        frame_stride=1,
    )
    action_mean = train_raw.action.mean(axis=0).astype(np.float32)
    action_std = np.maximum(train_raw.action.std(axis=0), 1e-6).astype(np.float32)
    train = _materialize_pairs(train_raw, summary, args.view, int(args.image_size), int(args.max_train_samples), rng)
    val = _materialize_pairs(val_raw, summary, args.view, int(args.image_size), int(args.max_val_samples), rng)
    train["action"] = ((train["action"] - action_mean) / action_std).astype(np.float32)
    val_action_raw = val["action"].astype(np.float32)
    val["action"] = ((val["action"] - action_mean) / action_std).astype(np.float32)

    model = VideoInverseDynamics(
        action_dim=int(train_raw.action.shape[-1]),
        task_count=int(max(train_raw.task_id.max(initial=0), val_raw.task_id.max(initial=0)) + 1),
        task_dim=int(args.task_dim),
        width=int(args.width),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    best = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        step += 1
        model.train()
        idx = rng.integers(0, len(train["action"]), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        pred = model(batch["image_pair"], batch["task_id"], batch["progress"])
        loss = F.mse_loss(pred, batch["action"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            metrics = _eval(model, val, action_mean, action_std, val_action_raw, int(args.batch_size), device)
            row = {"step": int(step), "loss": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time, **metrics}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["val_norm_mse"] < best:
                best = row["val_norm_mse"]
                _save(out_dir / "inverse_dynamics_best.pt", model, args, action_mean, action_std, history, summary)
    final = _eval(model, val, action_mean, action_std, val_action_raw, int(args.batch_size), device)
    _save(out_dir / "inverse_dynamics_last.pt", model, args, action_mean, action_std, history, summary)
    payload = {
        "task": "robocasa_world_model_inverse_dynamics",
        "checkpoint": str(out_dir / "inverse_dynamics_best.pt"),
        "last_checkpoint": str(out_dir / "inverse_dynamics_last.pt"),
        "train_samples": int(len(train["action"])),
        "val_samples": int(len(val["action"])),
        "best_val_norm_mse": float(best),
        "final_val": final,
        "history": history,
        "seconds": time.monotonic() - start_time,
    }
    save_json(out_dir / "metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _materialize_pairs(
    data: TransitionData,
    summary: list[dict],
    view: str,
    image_size: int,
    max_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    keep = np.arange(len(data), dtype=np.int64)
    if int(max_samples) > 0 and len(keep) > int(max_samples):
        keep = np.sort(rng.choice(keep, size=int(max_samples), replace=False))
    keep_set = set(int(x) for x in keep)
    keep_pos = {int(src): pos for pos, src in enumerate(keep)}
    pairs = np.empty((len(keep), 6, int(image_size), int(image_size)), dtype=np.float32)
    dataset_by_task = {int(row["task_id"]): Path(row["dataset_path"]) for row in summary}
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        if index in keep_set:
            groups.setdefault((int(task_id), int(episode_id)), []).append(index)
    for (task_id, episode_id), indices in sorted(groups.items()):
        root = dataset_by_task[int(task_id)]
        video = root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{int(episode_id):06d}.mp4"
        frames = load_video_frames(video)
        for index in indices:
            frame_idx = int(np.clip(data.frame_idx[index], 0, max(0, len(frames) - 2)))
            pair = np.concatenate([_preprocess(frames[frame_idx], image_size), _preprocess(frames[frame_idx + 1], image_size)], axis=0)
            pairs[keep_pos[index]] = pair
    return {
        "image_pair": pairs,
        "action": data.action[keep].astype(np.float32),
        "task_id": data.task_id[keep].astype(np.int64),
        "progress": data.progress[keep].astype(np.float32),
    }


def _preprocess(frame: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((int(image_size), int(image_size))))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


def _batch(data: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image_pair": torch.as_tensor(data["image_pair"][idx], dtype=torch.float32, device=device),
        "action": torch.as_tensor(data["action"][idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data["task_id"][idx], dtype=torch.long, device=device),
        "progress": torch.as_tensor(data["progress"][idx], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def _eval(
    model: VideoInverseDynamics,
    data: dict[str, np.ndarray],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    raw_action: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds = []
    losses = []
    for start in range(0, len(data["action"]), int(batch_size)):
        idx = np.arange(start, min(len(data["action"]), start + int(batch_size)))
        batch = _batch(data, idx, device)
        pred = model(batch["image_pair"], batch["task_id"], batch["progress"])
        losses.append(float(F.mse_loss(pred, batch["action"]).detach().cpu()) * len(idx))
        preds.append(pred.detach().cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0)
    pred_raw = pred_norm * action_std[None, :] + action_mean[None, :]
    raw_mse = float(np.mean((pred_raw - raw_action) ** 2))
    denom = np.linalg.norm(pred_raw, axis=-1) * np.linalg.norm(raw_action, axis=-1)
    cosine = float(np.mean(np.sum(pred_raw * raw_action, axis=-1) / np.maximum(denom, 1e-8)))
    return {
        "val_norm_mse": float(sum(losses) / max(1, len(data["action"]))),
        "val_raw_mse": raw_mse,
        "val_cosine": cosine,
    }


def _save(
    path: Path,
    model: VideoInverseDynamics,
    args: argparse.Namespace,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    history: list[dict],
    summary: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "action_dim": int(model.action_dim),
                "task_count": int(model.task_count),
                "task_dim": int(args.task_dim),
                "width": int(args.width),
                "image_size": int(args.image_size),
                "view": str(args.view),
            },
            "action_mean": action_mean,
            "action_std": action_std,
            "history": history,
            "summary": summary,
            "task": "robocasa_world_model_inverse_dynamics",
        },
        path,
    )


if __name__ == "__main__":
    main()
