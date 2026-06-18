from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()

import robocasa.utils.lerobot_utils as LU  # noqa: E402
from eval.render_robocasa_openloop_rollout import _compose_frame, _write_mp4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Immutable evaluator for the RoboCasa BC-5 task.")
    parser.add_argument("--checkpoint", "--policy", dest="checkpoint", required=True)
    parser.add_argument("--inference", default="tasks.robocasa_bc5.inference")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=16)
    parser.add_argument("--eval-episodes-per-task", type=int, default=10)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--render-dir", default="")
    parser.add_argument("--render-episodes-per-task", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    inference = importlib.import_module(args.inference)
    if not hasattr(inference, "load_policy") or not hasattr(inference, "act"):
        raise AttributeError(f"{args.inference} must define load_policy(checkpoint, device) and act(policy, obs, task)")
    policy = inference.load_policy(str(args.checkpoint), device=str(args.device))

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    task_aliases = set(args.task_alias)
    ffmpeg = shutil.which("ffmpeg") if args.render_dir else None

    details = []
    per_task = {}
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        manifest_task = manifest_tasks[alias]
        dataset_root = Path(manifest_task["dataset_path"])
        episode_ids = list(split_task["eval_episode_ids"])
        if args.eval_episodes_per_task > 0:
            episode_ids = episode_ids[: int(args.eval_episodes_per_task)]
        successes = 0
        task_details = []
        task = {
            "task_id": int(split_task["task_id"]),
            "alias": alias,
            "description": manifest_task.get("description", alias),
            "robocasa_task": manifest_task.get("robocasa_task", alias),
        }
        for local_idx, episode_id in enumerate(episode_ids):
            frames, success, steps = _rollout_episode(
                dataset_root=dataset_root,
                episode_idx=int(episode_id),
                policy=policy,
                inference=inference,
                task=task,
                camera=str(args.camera),
                width=int(args.render_width),
                height=int(args.render_height),
                max_steps=int(args.max_steps),
                commit_steps=int(args.commit_steps),
            )
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
        "checkpoint": str(args.checkpoint),
        "inference": str(args.inference),
        "manifest": str(args.manifest),
        "split": str(args.split),
        "episodes": len(details),
        "successes": sum(int(row["success"]) for row in details),
        "success_rate": sum(int(row["success"]) for row in details) / max(1, len(details)),
        "commit_steps": int(args.commit_steps),
        "max_steps": int(args.max_steps),
        "per_task": per_task,
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _rollout_episode(
    *,
    dataset_root: Path,
    episode_idx: int,
    policy,
    inference,
    task: dict,
    camera: str,
    width: int,
    height: int,
    max_steps: int,
    commit_steps: int,
) -> tuple[list[np.ndarray], bool, int]:
    import robocasa  # noqa: F401
    import robosuite
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    env_meta = LU.get_env_metadata(dataset_root)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env = robosuite.make(**env_kwargs)

    reset_to(
        env,
        {
            "model": LU.get_episode_model_xml(dataset_root, episode_idx),
            "ep_meta": json.dumps(LU.get_episode_meta(dataset_root, episode_idx)),
            "states": LU.get_episode_states(dataset_root, episode_idx)[0],
        },
    )

    frames: list[np.ndarray] = []
    success = False
    step_idx = 0
    try:
        frames.append(_compose_frame(env, camera, width, height, step_idx, success=False))
        while step_idx < max_steps and not success:
            obs = {
                "agent": _render64(env, "robot0_agentview_left"),
                "wrist": _render64(env, "robot0_agentview_right"),
                "proprio": _state_from_obs(env._get_observations()),
            }
            action_chunk = np.asarray(inference.act(policy, obs, task), dtype=np.float32)
            if action_chunk.ndim != 2:
                raise ValueError(f"inference.act must return [horizon, action_dim], got shape {action_chunk.shape}")
            actions = action_chunk[: min(int(commit_steps), action_chunk.shape[0], max_steps - step_idx)]
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            for action in actions:
                _, _, _, info = env.step(action)
                step_idx += 1
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                if not success and hasattr(env, "_check_success"):
                    try:
                        success = bool(env._check_success())
                    except Exception:
                        pass
                frames.append(_compose_frame(env, camera, width, height, step_idx, success=success))
                if success or step_idx >= max_steps:
                    break
    finally:
        try:
            if getattr(env, "viewer", None) is not None:
                env.viewer.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
    return frames, success, step_idx


def _render64(env, camera_name: str) -> np.ndarray:
    image = env.sim.render(height=64, width=64, camera_name=camera_name)[::-1]
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def _state_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_base_pos"], dtype=np.float32),
            np.asarray(obs["robot0_base_quat"], dtype=np.float32),
            np.asarray(obs["robot0_base_to_eef_pos"], dtype=np.float32),
            np.asarray(obs["robot0_base_to_eef_quat"], dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    main()
