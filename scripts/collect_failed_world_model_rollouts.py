from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

bc5_eval = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect failed RoboCasa BC5 policy rollouts for visual world-model training.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--policy-set", default="data/autorobobench/robocasa_world_model_policy_set.json")
    parser.add_argument("--out-root", default="data/autorobobench/visual_world_model_failed_rollouts")
    parser.add_argument("--split-out", default="data/autorobobench/robocasa_visual_world_model_failed_rollouts.json")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--max-attempts-per-task", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=16)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-name", action="append", default=[])
    parser.add_argument("--task-alias", action="append", default=[])
    args = parser.parse_args()

    global bc5_eval
    from tasks.robocasa_bc5 import eval as _bc5_eval  # noqa: E402

    bc5_eval = _bc5_eval

    manifest = json.loads((ROOT / args.manifest).read_text())
    split = json.loads((ROOT / args.split).read_text())
    policy_set = json.loads((ROOT / args.policy_set).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    requested_tasks = set(args.task_alias)
    policies = _select_policies(policy_set.get("policies", []), set(args.policy_name))
    if not policies:
        raise ValueError("no policies selected")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ModuleNotFoundError:
            ffmpeg = None
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to write rollout videos")

    out_root = ROOT / args.out_root
    split_rows: list[dict] = []
    details: list[dict] = []
    next_episode_base = 900000
    loaded_policies: dict[tuple[str, str], tuple[object, object]] = {}

    for split_task in split["tasks"]:
        alias = str(split_task["alias"])
        if requested_tasks and alias not in requested_tasks:
            continue
        task_id = int(split_task["task_id"])
        manifest_task = manifest_tasks[alias]
        dataset_root = _resolve_dataset_root(manifest_task["dataset_path"])
        task = {
            "task_id": task_id,
            "alias": alias,
            "description": manifest_task.get("description", alias),
            "robocasa_task": manifest_task.get("robocasa_task", alias),
        }
        collected_ids: list[int] = []
        attempts = 0
        source_episodes = [int(x) for x in split_task.get("train_episode_ids", [])]
        for policy_spec in policies:
            if len(collected_ids) >= int(args.episodes_per_task):
                break
            policy, inference = _load_policy(policy_spec, loaded_policies, args.device)
            for source_episode_id in source_episodes:
                if len(collected_ids) >= int(args.episodes_per_task):
                    break
                if attempts >= int(args.max_attempts_per_task):
                    break
                attempts += 1
                bc5_eval._reset_policy_history(policy)
                reset_state_index = bc5_eval._reset_state_index_for(split_task, int(source_episode_id))
                reset_perturbation = bc5_eval._reset_perturbation_for(split_task, int(source_episode_id))
                rollout = _record_rollout(
                    dataset_root=dataset_root,
                    episode_idx=int(source_episode_id),
                    reset_state_index=reset_state_index,
                    reset_perturbation=reset_perturbation,
                    policy=policy,
                    inference=inference,
                    task=task,
                    max_steps=int(args.max_steps),
                    commit_steps=int(args.commit_steps),
                )
                row = {
                    "task_alias": alias,
                    "task_id": task_id,
                    "source_episode_id": int(source_episode_id),
                    "policy_name": str(policy_spec.get("name", "")),
                    "success": bool(rollout["success"]),
                    "steps": int(rollout["steps"]),
                }
                if rollout["success"] or int(rollout["steps"]) <= 1:
                    print(json.dumps({**row, "kept": False}), flush=True)
                    continue
                synthetic_episode_id = next_episode_base + task_id * 1000 + len(collected_ids)
                dataset_out = out_root / alias
                _write_synthetic_episode(
                    dataset_out=dataset_out,
                    episode_id=synthetic_episode_id,
                    rollout=rollout,
                    fps=int(args.fps),
                    ffmpeg=ffmpeg,
                )
                collected_ids.append(synthetic_episode_id)
                kept = {
                    **row,
                    "kept": True,
                    "synthetic_episode_id": int(synthetic_episode_id),
                    "dataset_path": str(dataset_out.relative_to(ROOT)),
                }
                details.append(kept)
                print(json.dumps(kept), flush=True)
            if attempts >= int(args.max_attempts_per_task):
                break
        split_rows.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str((out_root / alias).relative_to(ROOT)),
                "train_episode_ids": collected_ids,
                "source": "failed_policy_rollout",
            }
        )

    payload = {
        "description": "Failed BC5 policy rollouts for visual world-model training. These episodes have no success and should have zero progress labels.",
        "manifest": str(args.manifest),
        "source_split": str(args.split),
        "policy_set": str(args.policy_set),
        "episodes_per_task": int(args.episodes_per_task),
        "max_steps": int(args.max_steps),
        "commit_steps": int(args.commit_steps),
        "tasks": split_rows,
        "details": details,
    }
    split_out = ROOT / args.split_out
    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wrote": str(split_out), "failed_rollouts": len(details)}, sort_keys=True), flush=True)


def _select_policies(policies: list[dict], requested: set[str]) -> list[dict]:
    if requested:
        return [policy for policy in policies if str(policy.get("name", "")) in requested]
    return sorted(
        policies,
        key=lambda policy: (
            float(policy.get("real_success_rate", 1.0) or 0.0) > 0.0,
            float(policy.get("real_success_rate", 1.0) or 0.0),
            str(policy.get("name", "")),
        ),
    )


def _load_policy(policy_spec: dict, cache: dict[tuple[str, str], tuple[object, object]], device: str) -> tuple[object, object]:
    checkpoint = _resolve_path(policy_spec["checkpoint"])
    module_name = str(policy_spec.get("inference", "tasks.robocasa_bc5.inference"))
    key = (module_name, str(checkpoint))
    if key not in cache:
        inference = importlib.import_module(module_name)
        policy = inference.load_policy(str(checkpoint), device=str(device))
        cache[key] = (policy, inference)
    return cache[key]


def _record_rollout(
    *,
    dataset_root: Path,
    episode_idx: int,
    reset_state_index: int,
    reset_perturbation: dict,
    policy,
    inference,
    task: dict,
    max_steps: int,
    commit_steps: int,
) -> dict:
    import robocasa  # noqa: F401
    import robosuite
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    env_meta = bc5_eval.LU.get_env_metadata(dataset_root)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env = robosuite.make(**env_kwargs)

    states = bc5_eval.LU.get_episode_states(dataset_root, episode_idx)
    reset_idx = int(np.clip(int(reset_state_index), 0, max(0, len(states) - 1)))
    reset_to(
        env,
        {
            "model": bc5_eval.LU.get_episode_model_xml(dataset_root, episode_idx),
            "ep_meta": json.dumps(bc5_eval._episode_meta_for_reset(dataset_root, episode_idx)),
        },
    )
    state = np.asarray(states[reset_idx], dtype=np.float64).copy()
    if reset_perturbation:
        state = bc5_eval._apply_state_perturbation(env, state, reset_perturbation, episode_idx)
    reset_to(env, {"states": state})

    state_frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    success_frames: list[float] = []
    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []
    success = False
    step_idx = 0
    try:
        obs_raw = env._get_observations()
        state_frames.append(bc5_eval._state_from_obs(obs_raw))
        success_frames.append(0.0)
        left_frames.append(bc5_eval._render64(env, "robot0_agentview_left"))
        right_frames.append(bc5_eval._render64(env, "robot0_agentview_right"))
        while step_idx < int(max_steps) and not success:
            obs = {
                "agent": left_frames[-1],
                "wrist": right_frames[-1],
                "proprio": state_frames[-1],
            }
            action_chunk = np.asarray(inference.act(policy, obs, task), dtype=np.float32)
            if action_chunk.ndim != 2:
                raise ValueError(f"inference.act must return [horizon, action_dim], got {action_chunk.shape}")
            resolved_commit_steps = bc5_eval._resolve_commit_steps(
                policy=policy,
                inference=inference,
                task=task,
                action_chunk=action_chunk,
                default_commit_steps=int(commit_steps),
            )
            chunk = action_chunk[: min(resolved_commit_steps, action_chunk.shape[0], int(max_steps) - step_idx)]
            for action in np.clip(chunk, -1.0, 1.0).astype(np.float32):
                obs_raw, _, _, info = env.step(action)
                step_idx += 1
                actions.append(np.asarray(action, dtype=np.float32).copy())
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                if not success:
                    success = bc5_eval._check_env_success(env)
                state_frames.append(bc5_eval._state_from_obs(obs_raw))
                success_frames.append(float(success))
                left_frames.append(bc5_eval._render64(env, "robot0_agentview_left"))
                right_frames.append(bc5_eval._render64(env, "robot0_agentview_right"))
                if success or step_idx >= int(max_steps):
                    break
    finally:
        bc5_eval._close_env(env)
    return {
        "states": np.asarray(state_frames, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "success_frames": np.asarray(success_frames, dtype=np.float32),
        "left_frames": np.asarray(left_frames, dtype=np.uint8),
        "right_frames": np.asarray(right_frames, dtype=np.uint8),
        "success": bool(success),
        "steps": int(step_idx),
    }


def _write_synthetic_episode(
    *,
    dataset_out: Path,
    episode_id: int,
    rollout: dict,
    fps: int,
    ffmpeg: str,
) -> None:
    states = np.asarray(rollout["states"], dtype=np.float32)
    actions = np.asarray(rollout["actions"], dtype=np.float32)
    if len(actions) == 0:
        raise ValueError("cannot write empty rollout")
    if len(actions) < len(states):
        pad = np.zeros((len(states) - len(actions), actions.shape[-1]), dtype=np.float32)
        actions = np.concatenate([actions, pad], axis=0)
    success = np.asarray(rollout["success_frames"], dtype=np.float32)
    if len(success) < len(states):
        success = np.pad(success, (0, len(states) - len(success)))
    data_dir = dataset_out / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "observation.state": [row.astype(np.float32) for row in states],
            "action": [row.astype(np.float32) for row in actions[: len(states)]],
            "success": success[: len(states)].astype(np.float32),
        }
    )
    frame.to_parquet(data_dir / f"episode_{int(episode_id):06d}.parquet", index=False)
    for view, frames in (
        ("robot0_agentview_left", rollout["left_frames"]),
        ("robot0_agentview_right", rollout["right_frames"]),
    ):
        video_dir = dataset_out / "videos" / "chunk-000" / f"observation.images.{view}"
        video_dir.mkdir(parents=True, exist_ok=True)
        bc5_eval._write_mp4(
            [np.asarray(frame, dtype=np.uint8) for frame in frames],
            video_dir / f"episode_{int(episode_id):06d}.mp4",
            int(fps),
            ffmpeg,
        )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _resolve_dataset_root(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    parts = path.parts
    marker = ("third_party", "robocasa", "datasets")
    for idx in range(0, max(0, len(parts) - len(marker) + 1)):
        if tuple(parts[idx : idx + len(marker)]) == marker:
            return ROOT.joinpath(*parts[idx:])
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
