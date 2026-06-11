from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from eval.eval_robocasa_policy_ensemble import _load_member
from eval.render_robocasa_chunk_policy import _compose_frame, _ckpt_tensor, _episode_task_id, _render64, _state_from_obs, _write_mp4
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float, default=[])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=16)
    parser.add_argument("--white-scene", action="store_true")
    parser.add_argument("--white-studio", action="store_true")
    parser.add_argument("--camera-back-offset", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MP4 video")

    device = device_from_arg(args.device)
    members = [_load_member(Path(path), device) for path in args.policy]
    weights = np.asarray(args.weight if args.weight else [1.0] * len(members), dtype=np.float32)
    if len(weights) != len(members):
        raise ValueError("--weight count must match --policy count")
    weights = weights / np.maximum(weights.sum(), 1e-6)

    frames, success, steps = _rollout_video(
        members=members,
        weights=weights,
        dataset_root=Path(args.dataset_root),
        episode_idx=int(args.episode_id),
        device=device,
        camera=str(args.camera),
        width=int(args.width),
        height=int(args.height),
        max_steps=int(args.max_steps),
        commit_steps=int(args.commit_steps),
        white_scene=bool(args.white_scene),
        white_studio=bool(args.white_studio),
        camera_back_offset=float(args.camera_back_offset),
    )
    if frames:
        frames.extend([frames[-1].copy() for _ in range(args.fps)])
    _write_mp4(frames, Path(args.out), int(args.fps), ffmpeg)
    print(json.dumps({"out": args.out, "success": bool(success), "steps": int(steps), "episode_id": int(args.episode_id)}, indent=2))


def _rollout_video(
    *,
    members,
    weights: np.ndarray,
    dataset_root: Path,
    episode_idx: int,
    device: torch.device,
    camera: str,
    width: int,
    height: int,
    max_steps: int,
    commit_steps: int,
    white_scene: bool,
    white_studio: bool,
    camera_back_offset: float,
):
    import robocasa  # noqa: F401
    import robosuite
    import robocasa.utils.lerobot_utils as LU
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
    if camera_back_offset:
        _move_camera_back(env, camera, camera_back_offset)

    frames = []
    success = False
    step_idx = 0
    try:
        frames.append(_compose_visual_frame(env, camera, width, height, step_idx, success=False, white_scene=white_scene, white_studio=white_studio))
        while step_idx < max_steps and not success:
            agent = _render64(env, "robot0_agentview_left")
            wrist = _render64(env, "robot0_agentview_right")
            proprio = _state_from_obs(env._get_observations())
            preds = []
            with torch.no_grad():
                for model, checkpoint in members:
                    action_mean = _ckpt_tensor(checkpoint, "action_mean", device)
                    action_std = _ckpt_tensor(checkpoint, "action_std", device)
                    proprio_mean = _ckpt_tensor(checkpoint, "proprio_mean", device)
                    proprio_std = _ckpt_tensor(checkpoint, "proprio_std", device)
                    task_id = _episode_task_id(dataset_root, episode_idx, checkpoint)
                    agent_t = torch.as_tensor(agent[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                    wrist_t = torch.as_tensor(wrist[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                    proprio_t = (torch.as_tensor(proprio[None], dtype=torch.float32, device=device) - proprio_mean) / proprio_std
                    task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
                    pred_norm = model(agent_t, wrist_t, proprio_t, task_t)[0]
                    preds.append((pred_norm * action_std + action_mean).detach().cpu().numpy())
            pred = np.sum(np.stack(preds) * weights.reshape(-1, 1, 1), axis=0)
            actions = np.clip(pred[: min(commit_steps, pred.shape[0], max_steps - step_idx)].astype(np.float32), -1.0, 1.0)
            for action in actions:
                _, _, _, info = env.step(action)
                step_idx += 1
                success = bool(info.get("success", False)) if isinstance(info, dict) else False
                if not success and hasattr(env, "_check_success"):
                    try:
                        success = bool(env._check_success())
                    except Exception:
                        pass
                frames.append(_compose_visual_frame(env, camera, width, height, step_idx, success=success, white_scene=white_scene, white_studio=white_studio))
                if success or step_idx >= max_steps:
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return frames, success, step_idx


def _compose_visual_frame(env, camera: str, width: int, height: int, step_idx: int, success: bool, white_scene: bool, white_studio: bool) -> np.ndarray:
    if not white_scene and not white_studio:
        return _compose_frame(env, camera, width, height, step_idx, success)
    model = env.sim.model
    geom_rgba = model.geom_rgba.copy() if hasattr(model, "geom_rgba") else None
    mat_rgba = model.mat_rgba.copy() if hasattr(model, "mat_rgba") else None
    site_rgba = model.site_rgba.copy() if hasattr(model, "site_rgba") else None
    light_diffuse = model.light_diffuse.copy() if hasattr(model, "light_diffuse") else None
    light_ambient = model.light_ambient.copy() if hasattr(model, "light_ambient") else None
    try:
        if white_studio:
            _make_white_studio(env)
        else:
            _make_scene_white(env)
        return _compose_frame(env, camera, width, height, step_idx, success)
    finally:
        if geom_rgba is not None:
            model.geom_rgba[:] = geom_rgba
        if mat_rgba is not None:
            model.mat_rgba[:] = mat_rgba
        if site_rgba is not None:
            model.site_rgba[:] = site_rgba
        if light_diffuse is not None:
            model.light_diffuse[:] = light_diffuse
        if light_ambient is not None:
            model.light_ambient[:] = light_ambient


def _make_scene_white(env) -> None:
    model = env.sim.model
    if hasattr(model, "geom_rgba"):
        model.geom_rgba[:, :3] = 1.0
        model.geom_rgba[:, 3] = np.maximum(model.geom_rgba[:, 3], 0.82)
    if hasattr(model, "mat_rgba"):
        model.mat_rgba[:, :3] = 1.0
        model.mat_rgba[:, 3] = np.maximum(model.mat_rgba[:, 3], 0.82)
    if hasattr(model, "site_rgba"):
        model.site_rgba[:, :3] = 1.0
    if hasattr(model, "light_diffuse"):
        model.light_diffuse[:, :3] = 1.0
    if hasattr(model, "light_ambient"):
        model.light_ambient[:, :3] = np.maximum(model.light_ambient[:, :3], 0.65)


def _make_white_studio(env) -> None:
    model = env.sim.model
    if hasattr(model, "geom_rgba"):
        names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
        for idx, name in enumerate(names):
            lower = name.lower()
            if "robot" in lower or "gripper" in lower or "eef" in lower:
                color = (0.72, 0.72, 0.72)
            elif "handle" in lower or "knob" in lower:
                color = (0.18, 0.18, 0.18)
            elif "drawer" in lower:
                color = (0.92, 0.92, 0.88)
            else:
                color = (0.98, 0.98, 0.95)
            model.geom_rgba[idx, :3] = color
            model.geom_rgba[idx, 3] = np.maximum(model.geom_rgba[idx, 3], 0.9)
    if hasattr(model, "mat_rgba"):
        model.mat_rgba[:, :3] = (0.96, 0.96, 0.93)
        model.mat_rgba[:, 3] = np.maximum(model.mat_rgba[:, 3], 0.9)
    if hasattr(model, "light_diffuse"):
        model.light_diffuse[:, :3] = 1.0
    if hasattr(model, "light_ambient"):
        model.light_ambient[:, :3] = np.maximum(model.light_ambient[:, :3], 0.45)


def _move_camera_back(env, camera_name: str, offset: float) -> None:
    model = env.sim.model
    data = env.sim.data
    cam_id = model.camera_name2id(camera_name)
    # MuJoCo cameras look along local -Z, so +Z in camera frame moves backward.
    back_dir = data.cam_xmat[cam_id].reshape(3, 3)[:, 2].copy()
    model.cam_pos[cam_id] = model.cam_pos[cam_id] + back_dir * float(offset)
    env.sim.forward()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
