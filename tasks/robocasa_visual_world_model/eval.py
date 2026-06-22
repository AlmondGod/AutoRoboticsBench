from __future__ import annotations

import argparse
import importlib
import json
import math
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

from tasks.robocasa_visual_world_model.inference import load_world_model, predict_next
from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    TransitionData,
    load_transition_data,
    load_video_frames,
    normalize_data,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate visually grounded RoboCasa world model.")
    parser.add_argument("--checkpoint", "--world-model", dest="checkpoint", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--lpips-size", type=int, default=64)
    parser.add_argument("--policy-checkpoint", default="", help="Optional policy checkpoint for generated-visual closed-loop scoring.")
    parser.add_argument("--policy-inference", default="tasks.robocasa_bc5.inference")
    parser.add_argument("--policy-eval-episodes-per-task", type=int, default=1)
    parser.add_argument("--policy-rollout-steps", type=int, default=64)
    parser.add_argument("--policy-commit-steps", type=int, default=8)
    parser.add_argument("--policy-image-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    start = time.monotonic()
    world = load_world_model(str(args.checkpoint), device=str(args.device))
    ckpt = world["checkpoint"]
    cfg = world["config"]
    _, val_raw, summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        task_aliases=set(args.task_alias),
        frame_stride=int(args.frame_stride),
    )
    val = normalize_data(val_raw, ckpt["stats"])
    rgb, next_rgb = _precompute_rgb_targets(
        val,
        summary,
        view=str(cfg.get("view", "robot0_agentview_right")),
        image_size=int(cfg["image_size"]),
    )
    lpips_model = _load_lpips_model(str(args.lpips_net), world["device"])
    metrics = _visual_transition_eval(world, val, rgb, next_rgb, int(args.batch_size), lpips_model, int(args.lpips_size))
    generated_policy = _generated_visual_policy_eval(
        world,
        val_raw,
        rgb,
        summary,
        policy_checkpoint=str(args.policy_checkpoint),
        policy_inference=str(args.policy_inference),
        episodes_per_task=int(args.policy_eval_episodes_per_task),
        rollout_steps=int(args.policy_rollout_steps),
        commit_steps=int(args.policy_commit_steps),
        policy_image_size=int(args.policy_image_size),
        view=str(cfg.get("view", "robot0_agentview_right")),
    )
    benchmark = _benchmark_score(metrics, generated_policy)
    payload = {
        "task": "robocasa_visual_world_model",
        "checkpoint": str(args.checkpoint),
        "metric": "visual_world_model_score",
        **benchmark,
        "reproducibility_integrity": 1.0,
        "visual_transition_metrics": metrics,
        "generated_visual_policy_eval": generated_policy,
        "summary": summary,
        "eval_seconds": time.monotonic() - start,
    }
    save_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


@torch.no_grad()
def _visual_transition_eval(
    world: dict,
    data: TransitionData,
    rgb: np.ndarray,
    next_rgb: np.ndarray,
    batch_size: int,
    lpips_model: torch.nn.Module,
    lpips_size: int,
) -> dict[str, float]:
    model = world["model"]
    device = world["device"]
    model.eval()
    sums = {
        "next_state_mse_norm": 0.0,
        "next_progress_mse": 0.0,
        "success_bce": 0.0,
        "next_rgb_mse": 0.0,
        "next_rgb_mae": 0.0,
        "next_rgb_lpips": 0.0,
    }
    count = 0
    for start in range(0, len(data), int(batch_size)):
        end = min(len(data), start + int(batch_size))
        batch = {
            "state": torch.as_tensor(data.state[start:end], dtype=torch.float32, device=device),
            "action": torch.as_tensor(data.action[start:end], dtype=torch.float32, device=device),
            "next_state": torch.as_tensor(data.next_state[start:end], dtype=torch.float32, device=device),
            "progress": torch.as_tensor(data.progress[start:end], dtype=torch.float32, device=device),
            "next_progress": torch.as_tensor(data.next_progress[start:end], dtype=torch.float32, device=device),
            "success": torch.as_tensor(data.success[start:end], dtype=torch.float32, device=device),
            "task_id": torch.as_tensor(data.task_id[start:end], dtype=torch.long, device=device),
            "rgb": torch.as_tensor(rgb[start:end], dtype=torch.float32, device=device),
            "next_rgb": torch.as_tensor(next_rgb[start:end], dtype=torch.float32, device=device),
        }
        out = model(batch["state"], batch["action"], batch["task_id"], batch["progress"])
        n = end - start
        sums["next_state_mse_norm"] += float((out["next_state"] - batch["next_state"]).square().mean(dim=-1).sum().detach().cpu())
        sums["next_progress_mse"] += float((out["next_progress"] - batch["next_progress"]).square().sum().detach().cpu())
        sums["success_bce"] += float(torch.nn.functional.binary_cross_entropy_with_logits(out["success_logit"], batch["success"], reduction="sum").detach().cpu())
        sums["next_rgb_mse"] += float((out["next_rgb"] - batch["next_rgb"]).square().mean(dim=(1, 2, 3)).sum().detach().cpu())
        sums["next_rgb_mae"] += float((out["next_rgb"] - batch["next_rgb"]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
        sums["next_rgb_lpips"] += float(
            lpips_model(_lpips_input(out["next_rgb"], lpips_size), _lpips_input(batch["next_rgb"], lpips_size))
            .reshape(-1)
            .sum()
            .detach()
            .cpu()
        )
        count += n
    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["samples"] = int(count)
    metrics["next_rgb_psnr"] = float(-10.0 * math.log10(max(metrics["next_rgb_mse"], 1e-12)))
    return metrics


def _generated_visual_policy_eval(
    world: dict,
    data: TransitionData,
    rgb: np.ndarray,
    summary: list[dict],
    *,
    policy_checkpoint: str,
    policy_inference: str,
    episodes_per_task: int,
    rollout_steps: int,
    commit_steps: int,
    policy_image_size: int,
    view: str,
) -> dict:
    if not policy_checkpoint:
        return {
            "enabled": False,
            "reason": "--policy-checkpoint not provided",
            "generated_visual_policy_score": 0.0,
            "episodes": 0,
        }
    if int(episodes_per_task) <= 0 or int(rollout_steps) <= 0:
        return {
            "enabled": False,
            "reason": "policy eval episodes/rollout steps disabled",
            "generated_visual_policy_score": 0.0,
            "episodes": 0,
        }
    inference = importlib.import_module(policy_inference)
    policy = inference.load_policy(policy_checkpoint, device=str(world["device"]))
    task_rows = {int(row["task_id"]): row for row in summary}
    first_index: dict[tuple[int, int], int] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        first_index.setdefault((int(task_id), int(episode_id)), int(index))

    details = []
    for task_id, row in sorted(task_rows.items()):
        episode_ids = sorted({episode for tid, episode in first_index if tid == int(task_id)})[: int(episodes_per_task)]
        task = {
            "task_id": int(task_id),
            "alias": str(row["alias"]),
            "description": str(row.get("alias", "")),
            "robocasa_task": str(row.get("alias", "")),
        }
        for episode_idx in episode_ids:
            index = first_index[(int(task_id), int(episode_idx))]
            score = _rollout_policy_on_generated_visuals(
                world,
                policy,
                inference,
                task,
                episode_idx=int(episode_idx),
                initial_state=np.asarray(data.state[index], dtype=np.float32),
                initial_progress=float(data.progress[index]),
                initial_rgb=np.asarray(rgb[index], dtype=np.float32),
                rollout_steps=int(rollout_steps),
                commit_steps=int(commit_steps),
                policy_image_size=int(policy_image_size),
            )
            details.append(
                {
                    "task_alias": task["alias"],
                    "task_id": int(task_id),
                    "episode_id": int(episode_idx),
                    **score,
                }
            )
    episode_scores = [float(row["predicted_success"]) for row in details]
    return {
        "enabled": True,
        "policy_checkpoint": str(policy_checkpoint),
        "policy_inference": str(policy_inference),
        "conditioning": "policy observes visual world-model generated RGB; the single generated view is supplied to both agent and wrist inputs",
        "generated_view": str(view),
        "episodes": int(len(details)),
        "rollout_steps": int(rollout_steps),
        "commit_steps": int(commit_steps),
        "generated_visual_policy_score": float(np.mean(episode_scores)) if episode_scores else 0.0,
        "details": details,
    }


def _rollout_policy_on_generated_visuals(
    world: dict,
    policy,
    inference,
    task: dict,
    *,
    episode_idx: int,
    initial_state: np.ndarray,
    initial_progress: float,
    initial_rgb: np.ndarray,
    rollout_steps: int,
    commit_steps: int,
    policy_image_size: int,
) -> dict[str, float | int]:
    state = np.asarray(initial_state, dtype=np.float32)
    progress = float(initial_progress)
    current_rgb = np.asarray(initial_rgb, dtype=np.float32)
    success_probs: list[float] = []
    steps = 0
    while steps < int(rollout_steps):
        image = _policy_rgb(current_rgb, int(policy_image_size))
        obs = {"agent": image, "wrist": image, "proprio": state}
        action_chunk = np.asarray(inference.act(policy, obs, task), dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"inference.act must return [horizon, action_dim], got {action_chunk.shape}")
        try:
            horizon = int(inference.commit_steps(policy, task=task, action_chunk=action_chunk, default_commit_steps=int(commit_steps)))
        except AttributeError:
            horizon = int(commit_steps)
        horizon = max(1, min(horizon, int(commit_steps), int(action_chunk.shape[0]), int(rollout_steps) - steps))
        for action in np.clip(action_chunk[:horizon], -1.0, 1.0).astype(np.float32):
            step = predict_next(world, state, action, int(task["task_id"]), progress)
            state = np.asarray(step["next_state"], dtype=np.float32)
            progress = float(np.clip(step["next_progress"], 0.0, 1.0))
            current_rgb = np.asarray(step["next_rgb"], dtype=np.float32)
            success_probs.append(float(step["success_prob"]))
            steps += 1
            if steps >= int(rollout_steps):
                break
    return {
        "steps": int(steps),
        "predicted_success": float(max(success_probs) if success_probs else 0.0),
        "final_success_prob": float(success_probs[-1] if success_probs else 0.0),
        "final_progress": float(progress),
    }


def _policy_rgb(rgb_chw: np.ndarray, image_size: int) -> np.ndarray:
    image = np.asarray(rgb_chw, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected RGB CHW image, got shape {image.shape}")
    image = np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0))
    image_u8 = (image * 255.0).round().astype(np.uint8)
    if image_u8.shape[0] == int(image_size) and image_u8.shape[1] == int(image_size):
        return image_u8
    try:
        import cv2  # type: ignore

        return cv2.resize(image_u8, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
    except ModuleNotFoundError:
        from PIL import Image

        return np.asarray(Image.fromarray(image_u8).resize((int(image_size), int(image_size))))


def _precompute_rgb_targets(
    data: TransitionData,
    summary: list[dict],
    *,
    view: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.empty((len(data), 3, int(image_size), int(image_size)), dtype=np.float32)
    next_rgb = np.empty_like(rgb)
    dataset_by_task = {int(row["task_id"]): Path(row["dataset_path"]) for row in summary}
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        groups.setdefault((int(task_id), int(episode_id)), []).append(index)
    for (task_id, episode_id), indices in sorted(groups.items()):
        root = dataset_by_task[int(task_id)]
        video = root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{int(episode_id):06d}.mp4"
        frames = load_video_frames(video)
        for index in indices:
            frame_idx = int(np.clip(data.frame_idx[index], 0, max(0, len(frames) - 1)))
            next_idx = min(frame_idx + 1, max(0, len(frames) - 1))
            rgb[index] = _preprocess_frame(frames[frame_idx], image_size)
            next_rgb[index] = _preprocess_frame(frames[next_idx], image_size)
    return rgb, next_rgb


def _preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((int(image_size), int(image_size))))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


def _load_lpips_model(net: str, device: torch.device) -> torch.nn.Module:
    try:
        import lpips  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "robocasa_visual_world_model eval requires lpips. Install with `pip install lpips` "
            "or install this repo with the robocasa extra after updating dependencies."
        ) from exc
    model = lpips.LPIPS(net=str(net), verbose=False).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _lpips_input(image: torch.Tensor, lpips_size: int) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    if image.shape[-1] < int(lpips_size) or image.shape[-2] < int(lpips_size):
        image = F.interpolate(image, size=(int(lpips_size), int(lpips_size)), mode="bilinear", align_corners=False)
    return image * 2.0 - 1.0


def _benchmark_score(metrics: dict[str, float], generated_policy: dict | None = None) -> dict[str, float | dict[str, float]]:
    visual_perceptual = _mse_like_score(metrics.get("next_rgb_lpips"), scale=0.5)
    visual_reconstruction = _mse_like_score(metrics.get("next_rgb_mse"), scale=0.08)
    next_state = _mse_like_score(metrics.get("next_state_mse_norm"), scale=0.05)
    progress = _mse_like_score(metrics.get("next_progress_mse"), scale=0.05)
    success = _mse_like_score(metrics.get("success_bce"), scale=0.1)
    generated_policy_score = float((generated_policy or {}).get("generated_visual_policy_score", 0.0))
    weights = {
        "visual_perceptual_score": 0.35,
        "visual_reconstruction_score": 0.20,
        "generated_visual_policy_score": 0.25,
        "next_state_score": 0.10,
        "progress_score": 0.05,
        "success_score": 0.05,
    }
    score = (
        weights["visual_perceptual_score"] * visual_perceptual
        + weights["visual_reconstruction_score"] * visual_reconstruction
        + weights["generated_visual_policy_score"] * generated_policy_score
        + weights["next_state_score"] * next_state
        + weights["progress_score"] * progress
        + weights["success_score"] * success
    )
    return {
        "visual_world_model_score": float(max(0.0, min(1.0, score))),
        "visual_perceptual_score": float(visual_perceptual),
        "visual_reconstruction_score": float(visual_reconstruction),
        "generated_visual_policy_score": float(max(0.0, min(1.0, generated_policy_score))),
        "next_state_score": float(next_state),
        "progress_score": float(progress),
        "success_score": float(success),
        "benchmark_score_weights": weights,
    }


def _mse_like_score(value: float | None, *, scale: float) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - float(value) / max(float(scale), 1e-12))))


if __name__ == "__main__":
    main()
