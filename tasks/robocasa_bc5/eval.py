from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

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



ensure_robocasa_runtime()

import robocasa.utils.lerobot_utils as LU  # noqa: E402


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
    parser.add_argument("--eval-episodes-per-task", type=int, default=20)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--render-dir", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--render-episodes-per-task", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-imitation-episodes-per-task", type=int, default=2)
    parser.add_argument("--val-imitation-frames-per-episode", type=int, default=8)
    parser.add_argument("--val-imitation-score-weight", type=float, default=0.05)
    parser.add_argument("--skip-val-imitation", action="store_true")
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
    available_eval_episodes_per_task = {}
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        manifest_task = manifest_tasks[alias]
        dataset_root = Path(manifest_task["dataset_path"])
        raw_episode_ids = list(split_task["eval_episode_ids"])
        available_eval_episodes_per_task[alias] = len(raw_episode_ids)
        episode_ids = _select_eval_episode_ids(raw_episode_ids, int(args.eval_episodes_per_task))
        successes = 0
        task_details = []
        task = {
            "task_id": int(split_task["task_id"]),
            "alias": alias,
            "description": manifest_task.get("description", alias),
            "robocasa_task": manifest_task.get("robocasa_task", alias),
        }
        for local_idx, episode_id in enumerate(episode_ids):
            reset_state_index = _reset_state_index_for(split_task, int(episode_id))
            reset_perturbation = _reset_perturbation_for(split_task, int(episode_id))
            frames, success, steps, actions, success_trace = _rollout_episode(
                dataset_root=dataset_root,
                episode_idx=int(episode_id),
                reset_state_index=reset_state_index,
                reset_perturbation=reset_perturbation,
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
                "eval_local_idx": int(local_idx),
                "success": bool(success),
                "steps": int(steps),
            }
            if reset_state_index:
                row["reset_state_index"] = int(reset_state_index)
            if reset_perturbation:
                row["reset_perturbation"] = _summarize_perturbation(reset_perturbation)
            if args.trace_dir:
                trace_path = (
                    Path(args.trace_dir)
                    / alias
                    / f"episode_{int(episode_id):06d}.npz"
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    trace_path,
                    task_alias=np.asarray([alias]),
                    task_id=np.asarray([int(split_task["task_id"])], dtype=np.int64),
                    episode_id=np.asarray([int(episode_id)], dtype=np.int64),
                    reset_state_index=np.asarray([int(reset_state_index)], dtype=np.int64),
                    reset_perturbation=np.asarray([json.dumps(_summarize_perturbation(reset_perturbation))]),
                    actions=np.asarray(actions, dtype=np.float32),
                    success=np.asarray(success_trace, dtype=np.float32),
                    final_success=np.asarray([float(success)], dtype=np.float32),
                    steps=np.asarray([int(steps)], dtype=np.int64),
                )
                row["trace_path"] = str(trace_path)
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

    success_rate = sum(int(row["success"]) for row in details) / max(1, len(details))
    payload = {
        "track": "robocasa_bc5",
        "checkpoint": str(args.checkpoint),
        "inference": str(args.inference),
        "manifest": str(args.manifest),
        "split": str(args.split),
        "episodes": len(details),
        "successes": sum(int(row["success"]) for row in details),
        "success_rate": success_rate,
        "commit_steps": int(args.commit_steps),
        "max_steps": int(args.max_steps),
        "requested_eval_episodes_per_task": int(args.eval_episodes_per_task),
        "available_eval_episodes_per_task": available_eval_episodes_per_task,
        "per_task": per_task,
        "details": details,
    }
    if not args.skip_val_imitation:
        payload.update(
            _validation_imitation_metrics(
                manifest=manifest,
                split=split,
                policy=policy,
                inference=inference,
                task_aliases=task_aliases,
                episodes_per_task=int(args.val_imitation_episodes_per_task),
                frames_per_episode=int(args.val_imitation_frames_per_episode),
            )
        )
    _apply_auxiliary_validation_score(
        payload,
        base_key="success_rate",
        out_key="bc5_success_val_mse_score",
        weight=float(args.val_imitation_score_weight),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _select_eval_episode_ids(episode_ids: list[int], requested: int) -> list[int]:
    if requested <= 0:
        return episode_ids
    if requested <= len(episode_ids):
        return episode_ids[:requested]
    if not episode_ids:
        return []
    return [int(episode_ids[idx % len(episode_ids)]) for idx in range(requested)]


def _validation_imitation_metrics(
    *,
    manifest: dict,
    split: dict,
    policy,
    inference,
    task_aliases: set[str],
    episodes_per_task: int,
    frames_per_episode: int,
) -> dict:
    if episodes_per_task <= 0 or frames_per_episode <= 0:
        return {
            "val_action_mse": None,
            "val_action_rmse": None,
            "val_action_mse_score": 0.0,
            "val_imitation_samples": 0,
            "val_imitation_episodes": 0,
        }
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    total = 0.0
    denom = 0
    episodes_used = 0
    per_task: dict[str, dict] = {}
    for split_task in split.get("tasks", []):
        alias = str(split_task["alias"])
        if task_aliases and alias not in task_aliases:
            continue
        if alias not in manifest_tasks:
            continue
        manifest_task = manifest_tasks[alias]
        dataset_root = Path(manifest_task["dataset_path"])
        episode_ids = [int(x) for x in split_task.get("val_episode_ids", [])[: max(0, episodes_per_task)]]
        task_total = 0.0
        task_denom = 0
        task = {
            "task_id": int(split_task["task_id"]),
            "alias": alias,
            "description": manifest_task.get("description", alias),
            "robocasa_task": manifest_task.get("robocasa_task", alias),
        }
        for episode_id in episode_ids:
            _reset_policy_history(policy)
            episode_metrics = _validation_episode_action_mse(
                dataset_root=dataset_root,
                episode_idx=int(episode_id),
                policy=policy,
                inference=inference,
                task=task,
                frames_per_episode=int(frames_per_episode),
            )
            task_total += float(episode_metrics["sse"])
            task_denom += int(episode_metrics["count"])
            episodes_used += int(episode_metrics["used"])
        total += task_total
        denom += task_denom
        task_mse = task_total / task_denom if task_denom > 0 else None
        per_task[alias] = {
            "episodes": len(episode_ids),
            "samples": int(task_denom),
            "val_action_mse": task_mse,
            "val_action_mse_score": _mse_to_score(task_mse),
        }
    mse = total / denom if denom > 0 else None
    return {
        "val_action_mse": mse,
        "val_action_rmse": None if mse is None else float(np.sqrt(max(0.0, mse))),
        "val_action_mse_score": _mse_to_score(mse),
        "val_imitation_samples": int(denom),
        "val_imitation_episodes": int(episodes_used),
        "val_imitation_episodes_per_task": int(episodes_per_task),
        "val_imitation_frames_per_episode": int(frames_per_episode),
        "val_imitation_per_task": per_task,
    }


def _validation_episode_action_mse(
    *,
    dataset_root: Path,
    episode_idx: int,
    policy,
    inference,
    task: dict,
    frames_per_episode: int,
) -> dict:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    frame = pd.read_parquet(episode_path)
    agent = _read_demo_video64(dataset_root, episode_idx, "robot0_agentview_left")
    wrist = _read_demo_video64(dataset_root, episode_idx, "robot0_agentview_right")
    proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    actions = LU.get_episode_actions(dataset_root, episode_idx).astype(np.float32)
    n = min(len(agent), len(wrist), len(proprio), len(actions))
    if n <= 0:
        return {"sse": 0.0, "count": 0, "used": 0}
    sample_count = min(int(frames_per_episode), n)
    indices = np.linspace(0, n - 1, num=sample_count, dtype=np.int32)
    sse = 0.0
    count = 0
    for frame_idx in indices:
        obs = {
            "agent": agent[int(frame_idx)],
            "wrist": wrist[int(frame_idx)],
            "proprio": proprio[int(frame_idx)],
        }
        pred = np.asarray(inference.act(policy, obs, task), dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        horizon = min(pred.shape[0], n - int(frame_idx))
        action_dim = min(pred.shape[-1], actions.shape[-1])
        if horizon <= 0 or action_dim <= 0:
            continue
        target = actions[int(frame_idx) : int(frame_idx) + horizon, :action_dim]
        diff = pred[:horizon, :action_dim] - target
        sse += float(np.square(diff).sum())
        count += int(diff.size)
    return {"sse": sse, "count": count, "used": int(count > 0)}


def _reset_policy_history(policy) -> None:
    for name, value in (
        ("history_task_id", None),
        ("history_episode_id", None),
        ("history_step_idx", 0),
        ("prev_agent", None),
        ("prev_wrist", None),
        ("prev_proprio", None),
    ):
        if hasattr(policy, name):
            try:
                setattr(policy, name, value)
            except Exception:
                pass


def _resize64(image: np.ndarray) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(image, dtype=np.uint8)[..., :3]).resize((64, 64), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _read_demo_video64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frames = [_resize64(np.asarray(frame, dtype=np.uint8)) for frame in iio.imiter(video_path)]
    return np.stack(frames).astype(np.uint8)


def _mse_to_score(mse: float | None) -> float:
    if mse is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(mse)))


def _apply_auxiliary_validation_score(payload: dict, *, base_key: str, out_key: str, weight: float) -> None:
    base = float(payload.get(base_key, 0.0) or 0.0)
    aux = float(payload.get("val_action_mse_score", 0.0) or 0.0)
    w = max(0.0, min(1.0, float(weight)))
    payload["val_imitation_score_weight"] = w
    payload[out_key] = max(0.0, min(1.0, (1.0 - w) * base + w * aux))
    payload["metric"] = out_key


def _rollout_episode(
    *,
    dataset_root: Path,
    episode_idx: int,
    reset_state_index: int,
    reset_perturbation: dict,
    policy,
    inference,
    task: dict,
    camera: str,
    width: int,
    height: int,
    max_steps: int,
    commit_steps: int,
) -> tuple[list[np.ndarray], bool, int, list[np.ndarray], list[float]]:
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

    states = LU.get_episode_states(dataset_root, episode_idx)
    reset_idx = int(np.clip(int(reset_state_index), 0, max(0, len(states) - 1)))
    reset_to(
        env,
        {
            "model": LU.get_episode_model_xml(dataset_root, episode_idx),
            "ep_meta": json.dumps(LU.get_episode_meta(dataset_root, episode_idx)),
        },
    )
    state = np.asarray(states[reset_idx], dtype=np.float64).copy()
    if reset_perturbation:
        state = _apply_state_perturbation(env, state, reset_perturbation, episode_idx)
    reset_to(
        env,
        {
            "states": state,
        },
    )

    frames: list[np.ndarray] = []
    actions_applied: list[np.ndarray] = []
    success_trace: list[float] = []
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
            resolved_commit_steps = _resolve_commit_steps(
                policy=policy,
                inference=inference,
                task=task,
                action_chunk=action_chunk,
                default_commit_steps=int(commit_steps),
            )
            actions = action_chunk[: min(resolved_commit_steps, action_chunk.shape[0], max_steps - step_idx)]
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            for action in actions:
                _, _, _, info = env.step(action)
                step_idx += 1
                actions_applied.append(np.asarray(action, dtype=np.float32).copy())
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                if not success:
                    success = _check_env_success(env)
                success_trace.append(float(success))
                frames.append(_compose_frame(env, camera, width, height, step_idx, success=success))
                if success or step_idx >= max_steps:
                    break
    finally:
        _close_env(env)
    return frames, success, step_idx, actions_applied, success_trace


def _check_env_success(env) -> bool:
    if not hasattr(env, "_check_success"):
        return False
    try:
        return bool(env._check_success())
    except Exception:
        return False


def _close_env(env) -> None:
    try:
        viewer = getattr(env, "viewer", None)
        if viewer is not None:
            viewer.close()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass


def _reset_state_index_for(split_task: dict, episode_id: int) -> int:
    overrides = split_task.get("eval_reset_state_indices", {})
    if not overrides:
        return 0
    if isinstance(overrides, dict):
        value = overrides.get(str(int(episode_id)), overrides.get(int(episode_id), 0))
        return max(0, int(value or 0))
    for row in overrides:
        if int(row.get("episode_id", -1)) == int(episode_id):
            return max(0, int(row.get("reset_state_index", row.get("state_index", 0)) or 0))
    return 0


def _reset_perturbation_for(split_task: dict, episode_id: int) -> dict:
    overrides = split_task.get("eval_reset_perturbations", {})
    if not overrides:
        return {}
    if isinstance(overrides, dict):
        value = overrides.get(str(int(episode_id)), overrides.get(int(episode_id), {}))
        return dict(value or {})
    for row in overrides:
        if int(row.get("episode_id", -1)) == int(episode_id):
            return dict(row)
    return {}


def _apply_state_perturbation(env, state: np.ndarray, spec: dict, episode_idx: int) -> np.ndarray:
    out = np.asarray(state, dtype=np.float64).copy()
    qpos_indices = [int(x) for x in spec.get("qpos_indices", [])]
    qpos_noise_std = float(spec.get("qpos_noise_std", 0.0) or 0.0)
    qpos_noise_clip = float(spec.get("qpos_noise_clip", 0.0) or 0.0)
    if qpos_indices and qpos_noise_std > 0:
        nq = int(env.sim.model.nq)
        seed = int(spec.get("seed", 0)) + int(episode_idx) * 1009
        rng = np.random.default_rng(seed)
        delta = rng.normal(loc=0.0, scale=qpos_noise_std, size=len(qpos_indices))
        if qpos_noise_clip > 0:
            delta = np.clip(delta, -qpos_noise_clip, qpos_noise_clip)
        for local_idx, qpos_idx in enumerate(qpos_indices):
            if 0 <= qpos_idx < nq:
                out[1 + qpos_idx] += float(delta[local_idx])
    return out


def _summarize_perturbation(spec: dict) -> dict:
    out = {
        "type": spec.get("type", "qpos_noise"),
        "seed": int(spec.get("seed", 0)),
        "qpos_indices": [int(x) for x in spec.get("qpos_indices", [])],
        "qpos_noise_std": float(spec.get("qpos_noise_std", 0.0) or 0.0),
        "qpos_noise_clip": float(spec.get("qpos_noise_clip", 0.0) or 0.0),
    }
    if "description" in spec:
        out["description"] = str(spec["description"])
    return out


def _resolve_commit_steps(
    *,
    policy,
    inference,
    task: dict,
    action_chunk: np.ndarray,
    default_commit_steps: int,
) -> int:
    value = None
    if hasattr(inference, "commit_steps"):
        fn = getattr(inference, "commit_steps")
        for kwargs in (
            {
                "task": task,
                "action_chunk": action_chunk,
                "default_commit_steps": int(default_commit_steps),
            },
            {"task": task, "default_commit_steps": int(default_commit_steps)},
            {"default_commit_steps": int(default_commit_steps)},
            {},
        ):
            try:
                value = fn(policy, **kwargs)
                break
            except TypeError:
                continue
    if value is None:
        checkpoint = getattr(policy, "checkpoint", None)
        if isinstance(checkpoint, dict):
            by_task = checkpoint.get("eval_commit_steps_by_task")
            if by_task is not None:
                task_id = int(task["task_id"])
                try:
                    value = int(by_task[task_id])
                except (IndexError, KeyError, TypeError):
                    value = None
            if value is None and checkpoint.get("eval_commit_steps") is not None:
                value = checkpoint.get("eval_commit_steps")
    if value is None:
        value = default_commit_steps
    value = int(value)
    if value <= 0:
        raise ValueError(f"commit_steps must be positive, got {value}")
    return value


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


def _compose_frame(env, camera: str, width: int, height: int, step_idx: int, success: bool) -> np.ndarray:
    image = env.sim.render(height=height, width=width, camera_name=camera)[::-1]
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(pil)
    bar_color = (38, 150, 78) if success else (190, 55, 45)
    draw.rectangle([0, 0, pil.width, 10], fill=bar_color)
    draw.rectangle([10, 16, 250, 46], fill=(255, 255, 255))
    draw.text((18, 24), f"step {step_idx:03d} | success {int(success)}", fill=(20, 20, 20))
    return np.asarray(pil, dtype=np.uint8)


def _write_mp4(frames: list[np.ndarray], out: Path, fps: int, ffmpeg: str) -> None:
    if not frames:
        raise ValueError("no frames to render")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, frame in enumerate(frames):
            ppm = tmp_path / f"frame_{idx:04d}.ppm"
            with ppm.open("wb") as handle:
                h, w, _ = frame.shape
                handle.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
                handle.write(frame.tobytes())
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "frame_%04d.ppm"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    main()
