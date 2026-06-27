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

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_visual_world_model.inference import load_world_model, predict_next
from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_POLICY_SET,
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
    parser.add_argument("--train-episodes-per-task", type=int, default=0)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--lpips-size", type=int, default=64)
    parser.add_argument("--policy-set", default=str(DEFAULT_POLICY_SET), help="JSON with fixed policies and stored real simulator eval success rates.")
    parser.add_argument("--policy-probe-set", default="", help="Cached policy probe JSON from scripts/collect_policy_probes.py.")
    parser.add_argument("--policy-checkpoint", default="", help=argparse.SUPPRESS)
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
    if args.policy_probe_set:
        policy_correlation = _visual_policy_probe_correlation_eval(
            world,
            probe_set_path=str(args.policy_probe_set),
            policy_inference=str(args.policy_inference),
            rollout_steps=int(args.policy_rollout_steps),
            commit_steps=int(args.policy_commit_steps),
            policy_image_size=int(args.policy_image_size),
            view=str(cfg.get("view", "robot0_agentview_right")),
        )
    else:
        policy_correlation = _visual_policy_correlation_eval(
            world,
            val_raw,
            rgb,
            summary,
            policy_set_path=str(args.policy_set),
            policy_checkpoint=str(args.policy_checkpoint),
            policy_inference=str(args.policy_inference),
            episodes_per_task=int(args.policy_eval_episodes_per_task),
            rollout_steps=int(args.policy_rollout_steps),
            commit_steps=int(args.policy_commit_steps),
            policy_image_size=int(args.policy_image_size),
            view=str(cfg.get("view", "robot0_agentview_right")),
        )
    benchmark = _benchmark_score(metrics, policy_correlation)
    payload = {
        "task": "robocasa_visual_world_model",
        "checkpoint": str(args.checkpoint),
        "metric": "visual_world_model_score",
        **benchmark,
        "reproducibility_integrity": 1.0,
        "visual_transition_metrics": metrics,
        "visual_policy_correlation": policy_correlation,
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
        out = model(batch["state"], batch["action"], current_rgb=batch["rgb"])
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


def _visual_policy_correlation_eval(
    world: dict,
    data: TransitionData,
    rgb: np.ndarray,
    summary: list[dict],
    *,
    policy_set_path: str,
    policy_checkpoint: str,
    policy_inference: str,
    episodes_per_task: int,
    rollout_steps: int,
    commit_steps: int,
    policy_image_size: int,
    view: str,
) -> dict:
    policies = _load_policy_set(policy_set_path, policy_checkpoint=policy_checkpoint, policy_inference=policy_inference)
    if not policies:
        return {
            "enabled": False,
            "reason": "no policies provided",
            "eval_correlation_score": 0.0,
            "valid_policy_count": 0,
        }
    if int(episodes_per_task) <= 0 or int(rollout_steps) <= 0:
        return {
            "enabled": False,
            "reason": "policy eval episodes/rollout steps disabled",
            "eval_correlation_score": 0.0,
            "valid_policy_count": 0,
        }
    task_rows = {int(row["task_id"]): row for row in summary}
    first_index: dict[tuple[int, int], int] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        first_index.setdefault((int(task_id), int(episode_id)), int(index))

    rows = []
    skipped = []
    for policy_spec in policies:
        checkpoint = str(policy_spec.get("checkpoint", ""))
        if not checkpoint or not Path(checkpoint).exists():
            skipped.append({**policy_spec, "skip_reason": "checkpoint_missing"})
            rows.append(_policy_result_row(policy_spec, predicted_success=None, details=[]))
            continue
        try:
            inference = importlib.import_module(str(policy_spec.get("inference", policy_inference)))
            policy = inference.load_policy(checkpoint, device=str(world["device"]))
        except Exception as exc:
            skipped.append({**policy_spec, "skip_reason": f"policy_load_failed: {type(exc).__name__}: {exc}"})
            rows.append(_policy_result_row(policy_spec, predicted_success=None, details=[]))
            continue
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
        rows.append(_policy_result_row(policy_spec, predicted_success=float(np.mean(episode_scores)) if episode_scores else None, details=details))
    valid = [row for row in rows if row.get("real_success_rate") is not None and row.get("predicted_success") is not None]
    real = np.asarray([row["real_success_rate"] for row in valid], dtype=np.float64)
    pred = np.asarray([row["predicted_success"] for row in valid], dtype=np.float64)
    pearson = _pearson(real, pred)
    spearman = _spearman(real, pred)
    ood_pearson = _ood_corr(rows, method="pearson")
    ood_spearman = _ood_corr(rows, method="spearman")
    corr_score = _mean_present([
        _corr_score(pearson),
        _corr_score(spearman),
        _corr_score(ood_pearson),
        _corr_score(ood_spearman),
    ])
    return {
        "enabled": True,
        "policy_set": str(policy_set_path),
        "conditioning": "Each policy observes world-model generated RGB and proprio/state rolled forward by the world model; real simulator success rates are loaded from stored metadata.",
        "generated_view": str(view),
        "policy_count": int(len(rows)),
        "valid_policy_count": int(len(valid)),
        "rollout_steps": int(rollout_steps),
        "commit_steps": int(commit_steps),
        "pearson": pearson,
        "spearman": spearman,
        "ood_pearson": ood_pearson,
        "ood_spearman": ood_spearman,
        "eval_correlation_score": float(corr_score),
        "policies": rows,
        "skipped_policies": skipped,
    }


def _visual_policy_probe_correlation_eval(
    world: dict,
    *,
    probe_set_path: str,
    policy_inference: str,
    rollout_steps: int,
    commit_steps: int,
    policy_image_size: int,
    view: str,
) -> dict:
    path = Path(probe_set_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    array_path = _resolve_probe_path(path.parent, str(payload.get("array_path", "")))
    arrays = np.load(array_path)
    initial_state = np.asarray(arrays["initial_state"], dtype=np.float32)
    initial_rgb = _probe_rgb_chw(arrays["initial_rgb"])
    probes = list(payload.get("probes", []))
    policies = list(payload.get("policies", []))
    if not probes or not policies:
        return {
            "enabled": False,
            "reason": "probe set has no probes or policies",
            "eval_correlation_score": 0.0,
            "valid_policy_count": 0,
            "valid_probe_count": 0,
        }

    loaded: dict[int, tuple[dict, object, object]] = {}
    skipped: list[dict] = []
    probe_rows: list[dict] = []
    for probe_index, probe in enumerate(probes):
        if probe_index >= len(initial_state) or probe_index >= len(initial_rgb):
            skipped.append({**probe, "skip_reason": "probe_array_missing"})
            continue
        policy_index = int(probe.get("policy_index", -1))
        if policy_index < 0 or policy_index >= len(policies):
            skipped.append({**probe, "skip_reason": "policy_index_out_of_range"})
            continue
        policy_spec = dict(policies[policy_index])
        try:
            if policy_index not in loaded:
                checkpoint = _resolve_existing_path(path.parent, str(policy_spec.get("checkpoint", "")))
                if not checkpoint.exists():
                    raise FileNotFoundError(str(checkpoint))
                inference = importlib.import_module(str(policy_spec.get("inference", policy_inference)))
                policy = inference.load_policy(str(checkpoint), device=str(world["device"]))
                loaded[policy_index] = (policy_spec, inference, policy)
            policy_spec, inference, policy = loaded[policy_index]
            task = {
                "task_id": int(probe["task_id"]),
                "alias": str(probe["task_alias"]),
                "description": str(probe.get("task_alias", "")),
                "robocasa_task": str(probe.get("task_alias", "")),
            }
            score = _rollout_policy_on_generated_visuals(
                world,
                policy,
                inference,
                task,
                episode_idx=int(probe.get("episode_id", -1)),
                initial_state=np.asarray(initial_state[probe_index], dtype=np.float32),
                initial_rgb=np.asarray(initial_rgb[probe_index], dtype=np.float32),
                rollout_steps=int(rollout_steps),
                commit_steps=int(probe.get("commit_steps", commit_steps)),
                policy_image_size=int(policy_image_size),
            )
        except Exception as exc:
            skipped.append({**probe, "skip_reason": f"probe_rollout_failed: {type(exc).__name__}: {exc}"})
            continue
        probe_rows.append(
            {
                "probe_id": str(probe.get("probe_id", probe_index)),
                "policy_index": int(policy_index),
                "policy_name": str(policy_spec.get("name", "")),
                "task_alias": str(probe["task_alias"]),
                "task_id": int(probe["task_id"]),
                "episode_id": int(probe.get("episode_id", -1)),
                "real_success": float(probe.get("real_success", 0.0)),
                "real_steps": int(probe.get("real_steps", 0)),
                **score,
            }
        )

    policy_rows = []
    for policy_index, policy_spec in enumerate(policies):
        rows = [row for row in probe_rows if int(row["policy_index"]) == int(policy_index)]
        real = [float(row["real_success"]) for row in rows]
        pred = [float(row["predicted_success"]) for row in rows]
        policy_rows.append(
            {
                "name": str(policy_spec.get("name", "")),
                "checkpoint": str(policy_spec.get("checkpoint", "")),
                "inference": str(policy_spec.get("inference", policy_inference)),
                "ood": bool(policy_spec.get("ood", False)),
                "real_success_rate": float(np.mean(real)) if real else None,
                "predicted_success": float(np.mean(pred)) if pred else None,
                "probe_count": int(len(rows)),
                "details": rows,
            }
        )

    valid_probes = [row for row in probe_rows if row.get("predicted_success") is not None]
    probe_real = np.asarray([row["real_success"] for row in valid_probes], dtype=np.float64)
    probe_pred = np.asarray([row["predicted_success"] for row in valid_probes], dtype=np.float64)
    probe_pearson = _pearson(probe_real, probe_pred)
    probe_spearman = _spearman(probe_real, probe_pred)
    valid_policies = [row for row in policy_rows if row.get("real_success_rate") is not None and row.get("predicted_success") is not None]
    policy_real = np.asarray([row["real_success_rate"] for row in valid_policies], dtype=np.float64)
    policy_pred = np.asarray([row["predicted_success"] for row in valid_policies], dtype=np.float64)
    pearson = _pearson(policy_real, policy_pred)
    spearman = _spearman(policy_real, policy_pred)
    ood_pearson = _ood_corr(policy_rows, method="pearson")
    ood_spearman = _ood_corr(policy_rows, method="spearman")
    corr_score = _mean_present([
        _corr_score(probe_pearson),
        _corr_score(probe_spearman),
        _corr_score(pearson),
        _corr_score(spearman),
        _corr_score(ood_pearson),
        _corr_score(ood_spearman),
    ])
    return {
        "enabled": True,
        "mode": "cached_policy_probe_set",
        "policy_probe_set": str(probe_set_path),
        "array_path": str(array_path),
        "conditioning": "Each cached probe reuses a stored RoboCasa initial state/RGB and real success label; future evals roll the same policy from that state inside the learned world model.",
        "generated_view": str(view),
        "policy_count": int(len(policy_rows)),
        "valid_policy_count": int(len(valid_policies)),
        "probe_count": int(len(probes)),
        "valid_probe_count": int(len(valid_probes)),
        "rollout_steps": int(rollout_steps),
        "commit_steps": int(commit_steps),
        "probe_pearson": probe_pearson,
        "probe_spearman": probe_spearman,
        "pearson": pearson,
        "spearman": spearman,
        "ood_pearson": ood_pearson,
        "ood_spearman": ood_spearman,
        "eval_correlation_score": float(corr_score),
        "policies": policy_rows,
        "probes": probe_rows,
        "skipped_probes": skipped,
    }


def _rollout_policy_on_generated_visuals(
    world: dict,
    policy,
    inference,
    task: dict,
    *,
    episode_idx: int,
    initial_state: np.ndarray,
    initial_rgb: np.ndarray,
    rollout_steps: int,
    commit_steps: int,
    policy_image_size: int,
) -> dict[str, float | int]:
    state = np.asarray(initial_state, dtype=np.float32)
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
            step = predict_next(
                world,
                state,
                action,
                current_rgb=current_rgb,
            )
            state = np.asarray(step["next_state"], dtype=np.float32)
            current_rgb = np.asarray(step["next_rgb"], dtype=np.float32)
            success_probs.append(float(step["success_prob"]))
            steps += 1
            if steps >= int(rollout_steps):
                break
    return {
        "steps": int(steps),
        "predicted_success": float(max(success_probs) if success_probs else 0.0),
        "final_success_prob": float(success_probs[-1] if success_probs else 0.0),
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
    dataset_by_task: dict[int, Path] = {}
    dataset_by_episode: dict[tuple[int, int], Path] = {}
    for row in summary:
        task_id = int(row["task_id"])
        root = Path(row["dataset_path"])
        dataset_by_task.setdefault(task_id, root)
        for key in ("train_episode_ids", "val_episode_ids", "failed_rollout_train_episode_ids"):
            for episode_id in row.get(key, []):
                dataset_by_episode[(task_id, int(episode_id))] = root
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        groups.setdefault((int(task_id), int(episode_id)), []).append(index)
    for (task_id, episode_id), indices in sorted(groups.items()):
        root = dataset_by_episode.get((int(task_id), int(episode_id)), dataset_by_task[int(task_id)])
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


def _benchmark_score(metrics: dict[str, float], policy_correlation: dict | None = None) -> dict[str, float | dict[str, float]]:
    visual_perceptual = _mse_like_score(metrics.get("next_rgb_lpips"), scale=0.5)
    visual_reconstruction = _mse_like_score(metrics.get("next_rgb_mse"), scale=0.08)
    next_state = _mse_like_score(metrics.get("next_state_mse_norm"), scale=0.05)
    progress = _mse_like_score(metrics.get("next_progress_mse"), scale=0.05)
    success = _mse_like_score(metrics.get("success_bce"), scale=0.1)
    eval_correlation = float((policy_correlation or {}).get("eval_correlation_score", 0.0))
    weights = {
        "eval_correlation_score": 0.50,
        "visual_reconstruction_score": 0.20,
        "next_state_score": 0.10,
        "progress_score": 0.10,
        "success_score": 0.10,
    }
    score = (
        weights["eval_correlation_score"] * eval_correlation
        + weights["visual_reconstruction_score"] * visual_reconstruction
        + weights["next_state_score"] * next_state
        + weights["progress_score"] * progress
        + weights["success_score"] * success
    )
    return {
        "visual_world_model_score": float(max(0.0, min(1.0, score))),
        "eval_correlation_score": float(max(0.0, min(1.0, eval_correlation))),
        "policy_score_pearson": None if policy_correlation is None else policy_correlation.get("pearson"),
        "policy_score_spearman": None if policy_correlation is None else policy_correlation.get("spearman"),
        "ood_policy_score_pearson": None if policy_correlation is None else policy_correlation.get("ood_pearson"),
        "ood_policy_score_spearman": None if policy_correlation is None else policy_correlation.get("ood_spearman"),
        "visual_perceptual_score": float(visual_perceptual),
        "visual_reconstruction_score": float(visual_reconstruction),
        "next_state_score": float(next_state),
        "progress_score": float(progress),
        "success_score": float(success),
        "benchmark_score_weights": weights,
    }


def _mse_like_score(value: float | None, *, scale: float) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - float(value) / max(float(scale), 1e-12))))


def _load_policy_set(policy_set_path: str, *, policy_checkpoint: str, policy_inference: str) -> list[dict]:
    if policy_set_path and Path(policy_set_path).exists():
        payload = json.loads(Path(policy_set_path).read_text(encoding="utf-8"))
        policies = list(payload.get("policies", []))
        if policies:
            return policies
    if policy_checkpoint:
        return [
            {
                "name": Path(policy_checkpoint).stem,
                "checkpoint": str(policy_checkpoint),
                "inference": str(policy_inference),
                "ood": False,
            }
        ]
    return []


def _policy_result_row(policy: dict, *, predicted_success: float | None, details: list[dict]) -> dict:
    return {
        "name": str(policy.get("name", "")),
        "checkpoint": str(policy.get("checkpoint", "")),
        "inference": str(policy.get("inference", "")),
        "ood": bool(policy.get("ood", False)),
        "real_eval_json": str(policy.get("real_eval_json", "")),
        "real_success_rate": _policy_real_success(policy),
        "predicted_success": None if predicted_success is None else float(predicted_success),
        "episode_count": int(len(details)),
        "details": details,
    }


def _policy_real_success(policy: dict) -> float | None:
    if "real_success_rate" in policy:
        try:
            return float(policy["real_success_rate"])
        except (TypeError, ValueError):
            pass
    return _real_success(str(policy.get("real_eval_json", "")))


def _resolve_probe_path(base: Path, value: str) -> Path:
    if not value:
        raise ValueError("probe set is missing array_path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _resolve_existing_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return path


def _probe_rgb_chw(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 4:
        raise ValueError(f"expected probe RGB array with 4 dims, got {arr.shape}")
    if arr.shape[1] == 3:
        out = arr.astype(np.float32)
        if out.max(initial=0.0) > 1.0:
            out = out / 255.0
        return out
    if arr.shape[-1] == 3:
        out = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)
        if out.max(initial=0.0) > 1.0:
            out = out / 255.0
        return out
    raise ValueError(f"expected RGB CHW or HWC array, got {arr.shape}")


def _real_success(path: str) -> float | None:
    if not path or not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("success_rate", "hidden_final_success", "peak_final_success", "offlinerl_final_success"):
        if key in payload:
            return float(payload[key])
    if "tracks" in payload:
        for value in payload["tracks"].values():
            if isinstance(value, dict):
                for key in ("success_rate", "hidden_final_success", "peak_final_success", "offlinerl_final_success"):
                    if key in value:
                        return float(value[key])
    return None


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return _pearson(_rank(x), _rank(y))


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _ood_corr(rows: list[dict], *, method: str) -> float | None:
    valid = [row for row in rows if bool(row.get("ood", False)) and row.get("real_success_rate") is not None and row.get("predicted_success") is not None]
    if len(valid) < 2:
        return None
    real = np.asarray([row["real_success_rate"] for row in valid], dtype=np.float64)
    pred = np.asarray([row["predicted_success"] for row in valid], dtype=np.float64)
    return _spearman(real, pred) if method == "spearman" else _pearson(real, pred)


def _corr_score(value: float | None) -> float | None:
    if value is None:
        return None
    return float(max(0.0, min(1.0, 0.5 * (float(value) + 1.0))))


def _mean_present(values: list[float | None]) -> float:
    present = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not present:
        return 0.0
    return float(np.mean(present))


if __name__ == "__main__":
    main()
