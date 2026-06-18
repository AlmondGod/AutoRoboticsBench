from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from autorobobench.robocasa_runtime import ensure_robocasa_runtime
from train.common import device_from_arg


ensure_robocasa_runtime()

from eval.eval_robocasa_chunk_policy import _rollout_temporal_ensemble  # noqa: E402
from eval.render_robocasa_chunk_policy import _rollout_closed_loop, _write_mp4  # noqa: E402
from eval.train_temporal_chunk_bc_robocasa import RoboCasaTemporalChunkBC  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Immutable AutoroboBench RoboCasa BC-5 eval entrypoint.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=8)
    parser.add_argument("--eval-episodes-per-task", type=int, default=10)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--ensemble-decay", type=float, default=0.7)
    parser.add_argument("--render-dir", default="")
    parser.add_argument("--render-episodes-per-task", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task_aliases = set(args.task_alias)
    device = device_from_arg(args.device)
    model, checkpoint = _load_policy(Path(args.policy), device)
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    ffmpeg = shutil.which("ffmpeg") if args.render_dir else None

    details = []
    per_task = {}
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        episode_ids = list(split_task["eval_episode_ids"])
        if args.eval_episodes_per_task > 0:
            episode_ids = episode_ids[: int(args.eval_episodes_per_task)]
        successes = 0
        task_details = []
        for local_idx, episode_id in enumerate(episode_ids):
            rollout_kwargs = {
                "dataset_root": dataset_root,
                "episode_idx": int(episode_id),
                "model": model,
                "checkpoint": checkpoint,
                "device": device,
                "camera": str(args.camera),
                "width": int(args.render_width),
                "height": int(args.render_height),
                "max_steps": int(args.max_steps),
                "commit_steps": int(args.commit_steps),
                "clip_actions": True,
                "task_id_override": int(split_task["task_id"]),
            }
            if args.temporal_ensemble:
                frames, success, steps = _rollout_temporal_ensemble(
                    **rollout_kwargs,
                    ensemble_decay=float(args.ensemble_decay),
                )
            else:
                frames, success, steps = _rollout_closed_loop(**rollout_kwargs)
            row = {
                "task_alias": alias,
                "task_id": int(split_task["task_id"]),
                "episode_id": int(episode_id),
                "success": bool(success),
                "steps": int(steps),
            }
            if args.render_dir and ffmpeg and local_idx < int(args.render_episodes_per_task):
                out_mp4 = Path(args.render_dir) / f"{alias}_episode_{int(episode_id):06d}.mp4"
                if frames:
                    frames.extend([frames[-1].copy() for _ in range(int(args.fps))])
                _write_mp4(frames, out_mp4, int(args.fps), ffmpeg)
                row["video"] = str(out_mp4)
            details.append(row)
            task_details.append(row)
            successes += int(bool(success))
            print(json.dumps(row), flush=True)
        per_task[alias] = {
            "episodes": len(task_details),
            "successes": int(successes),
            "success_rate": successes / max(1, len(task_details)),
        }

    payload = {
        "track": "robocasa_bc5",
        "policy": str(args.policy),
        "manifest": str(args.manifest),
        "split": str(args.split),
        "episodes": len(details),
        "successes": sum(int(row["success"]) for row in details),
        "success_rate": sum(int(row["success"]) for row in details) / max(1, len(details)),
        "commit_steps": int(args.commit_steps),
        "max_steps": int(args.max_steps),
        "temporal_ensemble": bool(args.temporal_ensemble),
        "ensemble_decay": float(args.ensemble_decay),
        "per_task": per_task,
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_policy(path: Path, device: torch.device) -> tuple[RoboCasaTemporalChunkBC, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RoboCasaTemporalChunkBC(
        proprio_dim=int(checkpoint["proprio_dim"]),
        chunk_horizon=int(checkpoint["chunk_horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        task_count=int(checkpoint["task_count"]),
        width=int(checkpoint.get("width", 512)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    main()
