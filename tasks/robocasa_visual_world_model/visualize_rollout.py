from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_visual_world_model.inference import load_world_model, predict_next
from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    load_episode_transitions,
    load_video_frames,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a RoboCasa visual world-model rollout.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--task-alias", default="")
    parser.add_argument("--episode-id", type=int, default=-1)
    parser.add_argument("--view", default="")
    parser.add_argument("--out", default="runs/autorobobench/robocasa_visual_world_model/rollout.gif")
    parser.add_argument("--mode", choices=["teacher_forced", "closed_loop"], default="closed_loop")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--panel-size", type=int, default=160)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    world = load_world_model(str(args.checkpoint), device=str(args.device))
    ckpt = world["checkpoint"]
    cfg = world["config"]
    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task = _select_task(manifest, split, ckpt, str(args.task_alias))
    episode_id = int(args.episode_id)
    if episode_id < 0:
        val_ids = task["split_task"].get("val_episode_ids", [])
        if not val_ids:
            raise ValueError(f"task {task['alias']} has no val episodes")
        episode_id = int(val_ids[0])
    view = str(args.view or cfg.get("view", "robot0_agentview_right"))
    dataset_root = Path(str(task["manifest_task"]["dataset_path"]))
    task_id = int(task["split_task"]["task_id"])
    episode = load_episode_transitions(dataset_root, episode_id, task_id, frame_stride=1)
    video = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_id:06d}.mp4"
    frames = load_video_frames(video)
    if len(frames) == 0:
        raise ValueError(f"video has no frames: {video}")

    steps = min(int(args.max_steps), len(episode["action"]))
    step_ids = np.arange(0, steps, max(1, int(args.stride)), dtype=np.int64)
    if len(step_ids) == 0:
        raise ValueError("no rollout steps selected")

    state = np.asarray(episode["state"][0], dtype=np.float32)
    progress = float(np.asarray(episode["progress"][0]).reshape(-1)[0])
    model_image_size = int(cfg["image_size"])
    current_rgb_model = _preprocess_model_rgb(frames[0], model_image_size)
    panels = []
    rows = []
    for out_index, step in enumerate(step_ids):
        action = np.asarray(episode["action"][step], dtype=np.float32)
        frame_idx = int(np.clip(episode["frame_idx"][step], 0, len(frames) - 1))
        next_idx = min(frame_idx + 1, len(frames) - 1)
        current = frames[frame_idx]
        target = frames[next_idx]
        if str(args.mode) == "teacher_forced":
            state_in = np.asarray(episode["state"][step], dtype=np.float32)
            progress_in = float(np.asarray(episode["progress"][step]).reshape(-1)[0])
            rgb_in = _preprocess_model_rgb(current, model_image_size)
        else:
            state_in = state
            progress_in = progress
            rgb_in = current_rgb_model
        pred = predict_next(
            world,
            state_in,
            action,
            task_id=int(task_id),
            progress=float(progress_in),
            current_rgb=rgb_in,
        )
        predicted = _rgb_chw_to_uint8(pred["next_rgb"])
        panels.append(
            _make_panel(
                current,
                target,
                predicted,
                panel_size=int(args.panel_size),
                title=f"{task['alias']} ep {episode_id} step {int(step)}",
                mode=str(args.mode),
            )
        )
        rows.append(
            {
                "index": int(out_index),
                "step": int(step),
                "frame_idx": int(frame_idx),
                "predicted_success_prob": float(pred["success_prob"]),
                "predicted_progress": float(pred["next_progress"]),
            }
        )
        state = np.asarray(pred["next_state"], dtype=np.float32)
        progress = float(np.clip(pred["next_progress"], 0.0, 1.0))
        current_rgb_model = np.asarray(pred["next_rgb"], dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_gif(out_path, panels, fps=float(args.fps))
    preview_path = out_path.with_suffix(".png")
    panels[0].save(preview_path)
    metrics_path = out_path.with_suffix(".json")
    save_json(
        metrics_path,
        {
            "task": "robocasa_visual_world_model_rollout_visualization",
            "checkpoint": str(args.checkpoint),
            "mode": str(args.mode),
            "task_alias": task["alias"],
            "task_id": task_id,
            "episode_id": episode_id,
            "view": view,
            "steps_visualized": len(rows),
            "gif": str(out_path),
            "preview_png": str(preview_path),
            "rows": rows,
        },
    )
    print(json.dumps({"gif": str(out_path), "preview_png": str(preview_path), "metrics": str(metrics_path)}, indent=2))


def _select_task(manifest: dict, split: dict, checkpoint: dict, task_alias: str) -> dict:
    manifest_tasks = {str(task["alias"]): task for task in manifest["tasks"]}
    split_tasks = {str(task["alias"]): task for task in split["tasks"]}
    alias = str(task_alias)
    if not alias:
        summary = checkpoint.get("summary", [])
        if summary:
            alias = str(summary[0]["alias"])
        else:
            alias = next(iter(split_tasks))
    if alias not in manifest_tasks or alias not in split_tasks:
        raise ValueError(f"unknown task alias {alias!r}")
    return {"alias": alias, "manifest_task": manifest_tasks[alias], "split_task": split_tasks[alias]}


def _rgb_chw_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected CHW RGB image, got shape {image.shape}")
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def _preprocess_model_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _resize(image: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((int(size), int(size)), Image.Resampling.BILINEAR)


def _make_panel(
    current: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    panel_size: int,
    title: str,
    mode: str,
) -> Image.Image:
    label_h = 38
    width = int(panel_size) * 3
    height = int(panel_size) + label_h
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    labels = ["current GT", "next GT", "predicted next"]
    images = [_resize(current, panel_size), _resize(target, panel_size), _resize(predicted, panel_size)]
    for index, image in enumerate(images):
        x = index * int(panel_size)
        canvas.paste(image, (x, label_h))
        draw.text((x + 6, 6), labels[index], fill=(20, 20, 20))
    draw.text((6, 22), f"{title} | {mode}", fill=(20, 20, 20))
    return canvas


def _save_gif(path: Path, frames: list[Image.Image], *, fps: float) -> None:
    duration_ms = int(round(1000.0 / max(float(fps), 1e-6)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


if __name__ == "__main__":
    main()
