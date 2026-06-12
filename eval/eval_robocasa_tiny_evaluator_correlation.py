from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image

from eval.eval_robocasa_policy_ensemble import _load_member
from eval.eval_world_model_ranking import compute_metrics
from eval.eval_world_model_ranking import _write_svg as write_ranking_svg
from eval.render_robocasa_chunk_policy import _ckpt_tensor, _episode_task_id
from models.robocasa_tiny_evaluator import RoboCasaTinyEvaluator
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--archive", default="research/robocasa_autoresearch_task0/archive.jsonl")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--out", default="runs/robocasa/world_evaluator/candidate_scores.jsonl")
    parser.add_argument("--metrics-out", default="runs/robocasa/world_evaluator/correlation_metrics.json")
    parser.add_argument("--plot", default="runs/robocasa/world_evaluator/correlation.svg")
    parser.add_argument("--imagined-rollouts", type=int, default=100)
    parser.add_argument("--imagined-steps", type=int, default=16)
    parser.add_argument("--action-noise", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    ckpt = torch.load(args.evaluator, map_location=device, weights_only=False)
    evaluator = RoboCasaTinyEvaluator(
        proprio_dim=int(ckpt["proprio_dim"]),
        action_dim=int(ckpt["action_dim"]),
        task_count=int(ckpt["task_count"]),
        latent_dim=int(ckpt["latent_dim"]),
        width=int(ckpt.get("width", 512)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    evaluator.load_state_dict(ckpt["state_dict"])
    evaluator.eval()

    rows = _load_archive(Path(args.archive))
    dataset_root = Path(args.dataset_root)
    out_rows: list[dict] = []
    started = time.time()
    for row in rows:
        policies = _policy_paths(row)
        if not policies:
            continue
        sim_success = float(row.get("success_rate", row.get("score", 0.0)))
        weights = _row_weights(row, len(policies))
        try:
            learned_score, learned_progress, seconds = _score_candidate(
                evaluator=evaluator,
                evaluator_ckpt=ckpt,
                policy_paths=policies,
                weights=weights,
                dataset_root=dataset_root,
                episode_ids=[int(ep) for ep in args.episode_id],
                imagined_rollouts=int(args.imagined_rollouts),
                imagined_steps=int(args.imagined_steps),
                action_noise=float(args.action_noise),
                device=device,
            )
            out_row = {
                "candidate_id": int(row.get("experiment", len(out_rows))),
                "change": row.get("change"),
                "learned_score": learned_score,
                "learned_progress": learned_progress,
                "sim_success": sim_success,
                "sim_successes": row.get("successes"),
                "sim_eval_rollouts": row.get("episodes"),
                "learned_eval_rollouts": int(args.imagined_rollouts) * len(args.episode_id),
                "learned_eval_seconds": seconds,
                "sim_eval_seconds": None,
            }
        except Exception as exc:
            out_row = {
                "candidate_id": int(row.get("experiment", len(out_rows))),
                "change": row.get("change"),
                "learned_score": None,
                "sim_success": sim_success,
                "error": repr(exc),
            }
        out_rows.append(out_row)
        print(json.dumps(out_row), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in out_rows) + "\n")
    metrics = compute_metrics(out_rows, top_k=int(args.top_k))
    metrics["total_wall_seconds"] = time.time() - started
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    plot = Path(args.plot)
    plot.parent.mkdir(parents=True, exist_ok=True)
    write_ranking_svg(out_rows, metrics, plot)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_archive(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _policy_paths(row: dict) -> list[str]:
    value = row.get("checkpoint")
    if not value:
        return []
    return [part for part in str(value).split(";") if part]


def _row_weights(row: dict, n: int) -> np.ndarray:
    change = str(row.get("change", ""))
    weights = np.ones((n,), dtype=np.float32)
    if "weights" in change:
        tail = change.split("weights", 1)[-1].strip().split()[0]
        try:
            parsed = [float(part) for part in tail.split("/") if part]
            if len(parsed) == n:
                weights = np.asarray(parsed, dtype=np.float32)
        except Exception:
            pass
    return weights / np.maximum(weights.sum(), 1e-6)


def _score_candidate(
    *,
    evaluator: RoboCasaTinyEvaluator,
    evaluator_ckpt: dict,
    policy_paths: list[str],
    weights: np.ndarray,
    dataset_root: Path,
    episode_ids: list[int],
    imagined_rollouts: int,
    imagined_steps: int,
    action_noise: float,
    device: torch.device,
) -> tuple[float, float, float]:
    members = [_load_member(Path(path), device) for path in policy_paths]
    start = time.time()
    scores: list[float] = []
    progresses: list[float] = []
    with torch.no_grad():
        for episode_id in episode_ids:
            obs = _initial_obs(dataset_root, episode_id)
            pred = _policy_chunk(members, weights, dataset_root, episode_id, obs, device)
            pred = pred[: max(1, min(imagined_steps, pred.shape[0]))]
            task_id = _evaluator_task_id(dataset_root, episode_id, evaluator_ckpt)
            task_t = torch.full((imagined_rollouts,), task_id, dtype=torch.long, device=device)
            agent = torch.as_tensor(np.repeat(obs["agent"][None], imagined_rollouts, axis=0), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            wrist = torch.as_tensor(np.repeat(obs["wrist"][None], imagined_rollouts, axis=0), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            proprio = (torch.as_tensor(np.repeat(obs["proprio"][None], imagined_rollouts, axis=0), dtype=torch.float32, device=device) - _tensor(evaluator_ckpt, "proprio_mean", device)) / _tensor(evaluator_ckpt, "proprio_std", device)
            latent = evaluator.encode(agent, wrist, proprio, task_t)
            episode_scores = []
            episode_progress = []
            for step in range(imagined_steps):
                action = pred[min(step, pred.shape[0] - 1)]
                actions = np.repeat(action[None], imagined_rollouts, axis=0).astype(np.float32)
                if action_noise > 0:
                    actions = actions + np.random.default_rng(episode_id * 1009 + step).normal(0.0, action_noise, size=actions.shape).astype(np.float32)
                action_t = (torch.as_tensor(actions, dtype=torch.float32, device=device) - _tensor(evaluator_ckpt, "action_mean", device)) / _tensor(evaluator_ckpt, "action_std", device)
                latent, _ = evaluator.step(latent, action_t, task_t)
                progress, success_logit = evaluator.heads(latent, task_t)
                episode_scores.append(torch.sigmoid(success_logit).detach().cpu().numpy())
                episode_progress.append(torch.sigmoid(progress).detach().cpu().numpy())
            scores.append(float(np.max(np.stack(episode_scores), axis=0).mean()))
            progresses.append(float(np.max(np.stack(episode_progress), axis=0).mean()))
    return float(np.mean(scores)), float(np.mean(progresses)), time.time() - start


def _policy_chunk(members, weights: np.ndarray, dataset_root: Path, episode_id: int, obs: dict, device: torch.device) -> np.ndarray:
    preds = []
    for model, checkpoint in members:
        action_mean = _ckpt_tensor(checkpoint, "action_mean", device)
        action_std = _ckpt_tensor(checkpoint, "action_std", device)
        proprio_mean = _ckpt_tensor(checkpoint, "proprio_mean", device)
        proprio_std = _ckpt_tensor(checkpoint, "proprio_std", device)
        task_id = _episode_task_id(dataset_root, episode_id, checkpoint)
        agent_t = torch.as_tensor(obs["agent"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        wrist_t = torch.as_tensor(obs["wrist"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        proprio_t = (torch.as_tensor(obs["proprio"][None], dtype=torch.float32, device=device) - proprio_mean) / proprio_std
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        if str(checkpoint.get("policy_kind", "bc")) == "flow":
            pred_norm = model.sample_flow(agent_t, wrist_t, proprio_t, task_t, steps=int(checkpoint.get("flow_steps", 8)))[0]
        else:
            pred_norm = model(agent_t, wrist_t, proprio_t, task_t)[0]
        preds.append((pred_norm * action_std + action_mean).detach().cpu().numpy())
    return np.sum(np.stack(preds) * weights.reshape(-1, 1, 1), axis=0).astype(np.float32)


def _evaluator_task_id(dataset_root: Path, episode_idx: int, checkpoint: dict) -> int:
    if not bool(checkpoint.get("condition_on_robocasa_task_index", False)):
        return 0
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet", columns=["task_index"])
    return int(frame["task_index"].iloc[0])


def _initial_obs(dataset_root: Path, episode_idx: int) -> dict:
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet", columns=["observation.state"])
    return {
        "agent": _first_frame64(dataset_root, episode_idx, "robot0_agentview_left"),
        "wrist": _first_frame64(dataset_root, episode_idx, "robot0_agentview_right"),
        "proprio": np.asarray(frame["observation.state"].iloc[0], dtype=np.float32),
    }


def _first_frame64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frame = next(iio.imiter(path))
    image = np.asarray(frame, dtype=np.uint8)[..., :3]
    if image.shape[:2] != (64, 64):
        image = np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
    return image


def _tensor(checkpoint: dict, key: str, device: torch.device) -> torch.Tensor:
    value = checkpoint[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


if __name__ == "__main__":
    main()
