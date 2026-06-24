from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def ensure_robocasa_runtime() -> None:
    import json as _json
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    for rel in ("third_party/robocasa", "third_party/robosuite", "."):
        path = str((repo / rel).resolve())
        if path not in _sys.path:
            _sys.path.insert(0, path)
    _os.environ.setdefault("PYTHONPATH", _os.pathsep.join(_sys.path))
    try:
        import lerobot.datasets.utils as _utils
    except ModuleNotFoundError:
        return
    if hasattr(_utils, "write_info"):
        return

    def write_info(info: dict, root: str | _Path) -> None:
        root_path = _Path(root)
        path = root_path if root_path.name == "info.json" else root_path / "info.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(info, indent=2, sort_keys=True) + "\n")

    _utils.write_info = write_info


def generate_policy_traces(
    *,
    policies: list[dict[str, Any]],
    manifest_path: str | Path,
    split_path: str | Path,
    trace_root: str | Path,
    episodes_per_task: int = 1,
    max_steps: int = 260,
    commit_steps: int = 16,
    device: str = "auto",
    source: str = "auto",
) -> list[dict[str, Any]]:
    """Generate policy-conditioned traces.

    `source=sim` runs real RoboCasa rollouts. `source=offline` runs the policy
    on heldout dataset observations and records the resulting action sequence.
    `source=auto` uses simulator when importable and otherwise falls back to
    offline traces, which still condition on the actual policy checkpoints and
    real heldout observations.
    """
    source = _resolve_source(source)
    manifest = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_path).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    out_policies = []
    for policy_row in policies:
        trace_dir = Path(trace_root) / _safe_name(str(policy_row["name"]))
        trace_dir.mkdir(parents=True, exist_ok=True)
        inference = importlib.import_module(str(policy_row.get("inference", "tasks.robocasa_bc5.inference")))
        policy = inference.load_policy(str(policy_row["checkpoint"]), device=str(device))
        trace_paths = []
        details = []
        for split_task in split["tasks"]:
            alias = str(split_task["alias"])
            manifest_task = manifest_tasks[alias]
            dataset_root = Path(manifest_task["dataset_path"])
            episode_ids = [int(x) for x in split_task["eval_episode_ids"][: int(episodes_per_task)]]
            task = {
                "task_id": int(split_task["task_id"]),
                "alias": alias,
                "description": manifest_task.get("description", alias),
                "robocasa_task": manifest_task.get("robocasa_task", alias),
            }
            for episode_id in episode_ids:
                trace = rollout_trace(
                    source=source,
                    dataset_root=dataset_root,
                    episode_idx=int(episode_id),
                    policy=policy,
                    inference=inference,
                    task=task,
                    max_steps=int(max_steps),
                    commit_steps=int(commit_steps),
                )
                out_path = trace_dir / alias / f"episode_{episode_id:06d}.npz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_path, **trace)
                trace_paths.append(str(out_path))
                details.append(
                    {
                        "task_alias": alias,
                        "episode_id": int(episode_id),
                        "success": bool(trace["final_success"][0]),
                        "steps": int(trace["steps"][0]),
                        "trace_path": str(out_path),
                    }
                )
                print(json.dumps({"policy": policy_row["name"], **details[-1]}), flush=True)
        updated = dict(policy_row)
        updated["trace_npz_paths"] = trace_paths
        updated["trace_dir"] = str(trace_dir)
        updated["trace_source"] = source
        updated["trace_details"] = details
        out_policies.append(updated)
    return out_policies


def rollout_trace(
    *,
    source: str,
    dataset_root: Path,
    episode_idx: int,
    policy,
    inference,
    task: dict,
    max_steps: int,
    commit_steps: int,
) -> dict[str, np.ndarray]:
    if source == "offline":
        return rollout_trace_offline(
            dataset_root=dataset_root,
            episode_idx=episode_idx,
            policy=policy,
            inference=inference,
            task=task,
            max_steps=max_steps,
            commit_steps=commit_steps,
        )
    return rollout_trace_sim(
        dataset_root=dataset_root,
        episode_idx=episode_idx,
        policy=policy,
        inference=inference,
        task=task,
        max_steps=max_steps,
        commit_steps=commit_steps,
    )


def rollout_trace_sim(
    *,
    dataset_root: Path,
    episode_idx: int,
    policy,
    inference,
    task: dict,
    max_steps: int,
    commit_steps: int,
) -> dict[str, np.ndarray]:
    ensure_robocasa_runtime()
    import robocasa  # noqa: F401
    import robocasa.utils.lerobot_utils as LU
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

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    success_trace: list[float] = []
    step_idx = 0
    success = False
    try:
        while step_idx < max_steps and not success:
            raw_obs = env._get_observations()
            state = _state_from_obs(raw_obs)
            obs = {
                "agent": _render64(env, "robot0_agentview_left"),
                "wrist": _render64(env, "robot0_agentview_right"),
                "proprio": state,
            }
            action_chunk = np.asarray(inference.act(policy, obs, task), dtype=np.float32)
            if action_chunk.ndim != 2:
                raise ValueError(f"inference.act must return [horizon, action_dim], got {action_chunk.shape}")
            chunk = action_chunk[: min(int(commit_steps), action_chunk.shape[0], max_steps - step_idx)]
            for action in np.clip(chunk, -1.0, 1.0).astype(np.float32):
                states.append(state.copy())
                actions.append(action.copy())
                _, _, _, info = env.step(action)
                step_idx += 1
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                if not success and hasattr(env, "_check_success"):
                    try:
                        success = bool(env._check_success())
                    except Exception:
                        pass
                success_trace.append(float(success))
                if success or step_idx >= max_steps:
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "task_id": np.asarray([int(task["task_id"])], dtype=np.int64),
        "episode_id": np.asarray([int(episode_idx)], dtype=np.int64),
        "success": np.asarray(success_trace, dtype=np.float32),
        "final_success": np.asarray([float(success)], dtype=np.float32),
        "steps": np.asarray([int(step_idx)], dtype=np.int64),
    }


def rollout_trace_offline(
    *,
    dataset_root: Path,
    episode_idx: int,
    policy,
    inference,
    task: dict,
    max_steps: int,
    commit_steps: int,
) -> dict[str, np.ndarray]:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    frame = pd.read_parquet(episode_path)
    states_all = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    agent = _read_video64(dataset_root, episode_idx, "robot0_agentview_left")
    wrist = _read_video64(dataset_root, episode_idx, "robot0_agentview_right")
    n = min(len(states_all), len(agent), len(wrist), int(max_steps))
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    success_trace: list[float] = []
    step_idx = 0
    while step_idx < n:
        obs_idx = min(step_idx, n - 1)
        obs = {
            "agent": agent[obs_idx],
            "wrist": wrist[obs_idx],
            "proprio": states_all[obs_idx],
        }
        action_chunk = np.asarray(inference.act(policy, obs, task), dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"inference.act must return [horizon, action_dim], got {action_chunk.shape}")
        chunk = action_chunk[: min(int(commit_steps), action_chunk.shape[0], n - step_idx)]
        for action in np.clip(chunk, -1.0, 1.0).astype(np.float32):
            states.append(states_all[min(step_idx, n - 1)].copy())
            actions.append(action.copy())
            step_idx += 1
            success_trace.append(0.0)
            if step_idx >= n:
                break
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "task_id": np.asarray([int(task["task_id"])], dtype=np.int64),
        "episode_id": np.asarray([int(episode_idx)], dtype=np.int64),
        "success": np.asarray(success_trace, dtype=np.float32),
        "final_success": np.asarray([0.0], dtype=np.float32),
        "steps": np.asarray([int(step_idx)], dtype=np.int64),
    }


def _render64(env, camera_name: str) -> np.ndarray:
    from PIL import Image

    image = env.sim.render(height=64, width=64, camera_name=camera_name)[::-1]
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def _read_video64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frames = [_resize64(np.asarray(frame, dtype=np.uint8)) for frame in iio.imiter(video_path)]
    return np.stack(frames).astype(np.uint8)


def _resize64(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    if image.shape[0] == 64 and image.shape[1] == 64:
        return image[..., :3]
    return np.asarray(Image.fromarray(image[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


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


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


def _resolve_source(source: str) -> str:
    if source not in {"auto", "sim", "offline"}:
        raise ValueError(f"trace source must be auto, sim, or offline, got {source!r}")
    if source in {"sim", "offline"}:
        return source
    try:
        import importlib.util

        if importlib.util.find_spec("robocasa") and importlib.util.find_spec("robosuite"):
            return "sim"
    except Exception:
        pass
    return "offline"
