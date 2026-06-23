#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_bc5 import eval as bc5_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect fixed RoboCasa policy probes for fast visual world-model eval.")
    parser.add_argument("--policy-set", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--array-name", default="probe_arrays.npz")
    parser.add_argument("--probes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=8)
    parser.add_argument("--view", default="robot0_agentview_right")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.monotonic()
    policy_set_path = Path(args.policy_set)
    policy_set = json.loads(policy_set_path.read_text(encoding="utf-8"))
    policies = list(policy_set.get("policies", []))
    if not policies:
        raise ValueError(f"{policy_set_path} has no policies")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    task_aliases = set(args.task_alias)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    array_path = out_path.parent / str(args.array_name)

    policy_cache: dict[int, tuple[dict, object, object]] = {}
    probes: list[dict] = []
    initial_states: list[np.ndarray] = []
    initial_rgbs: list[np.ndarray] = []
    failures: list[dict] = []

    selected_tasks = [row for row in split["tasks"] if not task_aliases or str(row["alias"]) in task_aliases]
    probe_index = 0
    for task_offset, split_task in enumerate(selected_tasks):
        alias = str(split_task["alias"])
        manifest_task = manifest_tasks[alias]
        dataset_root = Path(manifest_task["dataset_path"])
        episode_ids = [int(x) for x in split_task.get("eval_episode_ids", [])]
        if not episode_ids:
            raise ValueError(f"{alias} has no eval_episode_ids")
        task = {
            "task_id": int(split_task["task_id"]),
            "alias": alias,
            "description": manifest_task.get("description", alias),
            "robocasa_task": manifest_task.get("robocasa_task", alias),
        }
        for local_idx in range(int(args.probes_per_task)):
            policy_index = (task_offset * int(args.probes_per_task) + local_idx) % len(policies)
            policy_spec, inference, policy = _load_policy(policy_index, policies, policy_cache, device=str(args.device))
            episode_id = int(episode_ids[local_idx % len(episode_ids)])
            reset_state_index = bc5_eval._reset_state_index_for(split_task, episode_id)
            reset_perturbation = bc5_eval._reset_perturbation_for(split_task, episode_id)
            probe_id = f"{alias}_{local_idx:03d}_{policy_spec.get('name', policy_index)}"
            try:
                result = _rollout_probe(
                    dataset_root=dataset_root,
                    episode_idx=episode_id,
                    reset_state_index=reset_state_index,
                    reset_perturbation=reset_perturbation,
                    policy=policy,
                    inference=inference,
                    task=task,
                    view=str(args.view),
                    image_size=int(args.image_size),
                    max_steps=int(args.max_steps),
                    commit_steps=int(args.commit_steps),
                )
            except Exception as exc:
                failures.append(
                    {
                        "probe_id": probe_id,
                        "policy_index": int(policy_index),
                        "policy_name": str(policy_spec.get("name", "")),
                        "task_alias": alias,
                        "episode_id": int(episode_id),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(json.dumps(failures[-1]), flush=True)
                continue
            initial_states.append(np.asarray(result["initial_state"], dtype=np.float32))
            initial_rgbs.append(np.asarray(result["initial_rgb"], dtype=np.uint8))
            row = {
                "probe_id": probe_id,
                "probe_index": int(probe_index),
                "policy_index": int(policy_index),
                "policy_name": str(policy_spec.get("name", "")),
                "checkpoint": str(policy_spec.get("checkpoint", "")),
                "inference": str(policy_spec.get("inference", "")),
                "ood": bool(policy_spec.get("ood", False)),
                "task_alias": alias,
                "task_id": int(task["task_id"]),
                "episode_id": int(episode_id),
                "reset_state_index": int(reset_state_index),
                "reset_perturbation": bc5_eval._summarize_perturbation(reset_perturbation) if reset_perturbation else {},
                "real_success": float(result["success"]),
                "real_steps": int(result["steps"]),
                "commit_steps": int(args.commit_steps),
            }
            probes.append(row)
            probe_index += 1
            print(json.dumps(row), flush=True)

    if not probes:
        raise RuntimeError("no probes collected")
    np.savez_compressed(
        array_path,
        initial_state=np.stack(initial_states).astype(np.float32),
        initial_rgb=np.stack(initial_rgbs).astype(np.uint8),
    )
    payload = {
        "task": "robocasa_visual_world_model_policy_probe_set",
        "description": "Fixed RoboCasa simulator rollouts used as cached labels for fast visual world-model policy correlation.",
        "source_policy_set": str(policy_set_path),
        "manifest": str(args.manifest),
        "split": str(args.split),
        "array_path": array_path.name,
        "view": str(args.view),
        "image_size": int(args.image_size),
        "max_steps": int(args.max_steps),
        "commit_steps": int(args.commit_steps),
        "probes_per_task": int(args.probes_per_task),
        "policy_count": int(len(policies)),
        "probe_count": int(len(probes)),
        "failure_count": int(len(failures)),
        "real_success_rate": float(np.mean([row["real_success"] for row in probes])),
        "policies": policies,
        "probes": probes,
        "failures": failures,
        "seconds": float(time.monotonic() - started),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_policy(policy_index: int, policies: list[dict], cache: dict[int, tuple[dict, object, object]], *, device: str):
    if policy_index in cache:
        return cache[policy_index]
    spec = dict(policies[policy_index])
    inference = importlib.import_module(str(spec.get("inference", "tasks.robocasa_bc5.inference")))
    checkpoint = Path(str(spec["checkpoint"]))
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    policy = inference.load_policy(str(checkpoint), device=device)
    cache[policy_index] = (spec, inference, policy)
    return cache[policy_index]


def _rollout_probe(
    *,
    dataset_root: Path,
    episode_idx: int,
    reset_state_index: int,
    reset_perturbation: dict,
    policy,
    inference,
    task: dict,
    view: str,
    image_size: int,
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
            "ep_meta": json.dumps(bc5_eval.LU.get_episode_meta(dataset_root, episode_idx)),
        },
    )
    state = np.asarray(states[reset_idx], dtype=np.float64).copy()
    if reset_perturbation:
        state = bc5_eval._apply_state_perturbation(env, state, reset_perturbation, episode_idx)
    reset_to(env, {"states": state})

    success = False
    step_idx = 0
    try:
        initial_obs = env._get_observations()
        initial_state = bc5_eval._state_from_obs(initial_obs)
        initial_rgb = bc5_eval._render64(env, view)
        if int(image_size) != 64:
            from PIL import Image

            initial_rgb = np.asarray(Image.fromarray(initial_rgb).resize((int(image_size), int(image_size))), dtype=np.uint8)
        while step_idx < int(max_steps) and not success:
            obs = {
                "agent": bc5_eval._render64(env, "robot0_agentview_left"),
                "wrist": bc5_eval._render64(env, "robot0_agentview_right"),
                "proprio": bc5_eval._state_from_obs(env._get_observations()),
            }
            action_chunk = np.asarray(inference.act(policy, obs, task), dtype=np.float32)
            if action_chunk.ndim != 2:
                raise ValueError(f"inference.act must return [horizon, action_dim], got shape {action_chunk.shape}")
            resolved_commit_steps = bc5_eval._resolve_commit_steps(
                policy=policy,
                inference=inference,
                task=task,
                action_chunk=action_chunk,
                default_commit_steps=int(commit_steps),
            )
            actions = action_chunk[: min(resolved_commit_steps, action_chunk.shape[0], int(max_steps) - step_idx)]
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            for action in actions:
                _, _, _, info = env.step(action)
                step_idx += 1
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                if not success:
                    success = bc5_eval._check_env_success(env)
                if success or step_idx >= int(max_steps):
                    break
    finally:
        bc5_eval._close_env(env)
    return {
        "initial_state": np.asarray(initial_state, dtype=np.float32),
        "initial_rgb": np.asarray(initial_rgb, dtype=np.uint8),
        "success": bool(success),
        "steps": int(step_idx),
    }


if __name__ == "__main__":
    main()
