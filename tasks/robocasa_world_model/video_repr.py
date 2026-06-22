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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_world_model.data import DEFAULT_VIDEO_POOL, load_video_frame, load_video_frames, load_video_only_pool
def device_from_arg(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)



class VideoProgressEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
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
            nn.Linear(192, int(embed_dim)),
            nn.LayerNorm(int(embed_dim)),
        )
        self.progress = nn.Linear(int(embed_dim), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.normalize(self.encoder(image), dim=-1)
        return {"embedding": z, "progress": torch.sigmoid(self.progress(z))}


def load_video_encoder(checkpoint: str | Path, device: torch.device) -> VideoProgressEncoder:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    model = VideoProgressEncoder(embed_dim=int(cfg.get("embed_dim", 64))).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a tiny RoboCasa video representation on action-free videos.")
    parser.add_argument("--video-pool", default=str(DEFAULT_VIDEO_POOL))
    parser.add_argument("--out", default="runs/autorobobench/robocasa_world_model/video_repr.pt")
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--video-pool-split", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames-per-video", type=int, default=128)
    parser.add_argument("--disable-frame-cache", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if float(args.max_train_seconds) <= 0:
        raise ValueError("--max-train-seconds must be > 0; training is time-budgeted only")

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = device_from_arg(str(args.device))
    records = load_video_only_pool(
        args.video_pool,
        max_episodes_per_task=int(args.max_episodes_per_task),
        splits=set(args.video_pool_split),
    )
    if not records:
        raise ValueError("video pool produced no records")
    model = VideoProgressEncoder(embed_dim=int(args.embed_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    frame_cache = None
    if not bool(args.disable_frame_cache):
        frame_cache = _FrameCache(
            image_size=int(args.image_size),
            frame_stride=int(args.frame_stride),
            max_frames_per_video=int(args.max_frames_per_video),
        )
    history = []
    start = time.monotonic()
    step = 0
    while True:
        if time.monotonic() - start >= float(args.max_train_seconds):
            break
        step += 1
        batch = _sample_batch(records, rng, int(args.batch_size), int(args.image_size), device, frame_cache)
        out = model(batch["image"])
        progress_loss = F.mse_loss(out["progress"], batch["progress"])
        order_loss = _order_loss(out["embedding"], batch["progress"])
        loss = progress_loss + 0.25 * order_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "progress_mse": float(progress_loss.detach().cpu()),
                "order_loss": float(order_loss.detach().cpu()),
                "elapsed_seconds": time.monotonic() - start,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "embed_dim": int(args.embed_dim),
                "image_size": int(args.image_size),
                "video_pool": str(args.video_pool),
                "frame_stride": int(args.frame_stride),
                "max_frames_per_video": int(args.max_frames_per_video),
                "frame_cache": not bool(args.disable_frame_cache),
            },
            "history": history,
            "record_count": len(records),
            "task": "robocasa_world_model_video_repr",
        },
        out_path,
    )
    print(json.dumps({"checkpoint": str(out_path), "record_count": len(records), "history": history[-3:]}, indent=2, sort_keys=True))


class _FrameCache:
    def __init__(self, *, image_size: int, frame_stride: int, max_frames_per_video: int) -> None:
        self.image_size = int(image_size)
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames_per_video = int(max_frames_per_video)
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, video_path: Path) -> tuple[np.ndarray, np.ndarray]:
        path = Path(video_path)
        if path not in self._cache:
            frames = load_video_frames(path, stride=self.frame_stride, max_frames=self.max_frames_per_video)
            if len(frames) == 0:
                raise ValueError(f"video has no decodable frames: {path}")
            images = np.stack([_preprocess(frame, self.image_size) for frame in frames]).astype(np.float32)
            progress = np.linspace(0.0, 1.0, num=len(images), dtype=np.float32)
            self._cache[path] = (images, progress)
        return self._cache[path]


def _sample_batch(
    records: list,
    rng: np.random.Generator,
    batch_size: int,
    image_size: int,
    device: torch.device,
    frame_cache: _FrameCache | None = None,
) -> dict[str, torch.Tensor]:
    images = []
    progress = []
    for record in rng.choice(records, size=int(batch_size), replace=True):
        if frame_cache is not None:
            cached_images, cached_progress = frame_cache.get(record.video_path)
            frame_idx = int(rng.integers(0, len(cached_images)))
            images.append(cached_images[frame_idx])
            progress.append(float(cached_progress[frame_idx]))
        else:
            frame_count = _frame_count(record.video_path)
            frame_idx = int(rng.integers(0, max(1, frame_count)))
            frame = load_video_frame(record.video_path, frame_idx)
            images.append(_preprocess(frame, image_size))
            progress.append(frame_idx / max(1, frame_count - 1))
    return {
        "image": torch.as_tensor(np.stack(images), dtype=torch.float32, device=device),
        "progress": torch.as_tensor(np.asarray(progress, dtype=np.float32)[:, None], device=device),
    }


def _frame_count(path: Path) -> int:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(1, count)
    except ModuleNotFoundError:
        import imageio.v3 as iio

        meta = iio.immeta(path)
        return max(1, int(meta.get("nframes", 1)))


def _preprocess(frame: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((int(image_size), int(image_size))))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


def _order_loss(embedding: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return torch.zeros((), dtype=embedding.dtype, device=embedding.device)
    sim = embedding @ embedding.T
    target = 1.0 - torch.cdist(progress, progress, p=1).clamp(0.0, 1.0)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return F.mse_loss(sim[~eye], target[~eye])


if __name__ == "__main__":
    main()
