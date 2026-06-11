from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import robocasa.utils.lerobot_utils as LU
from eval.render_robocasa_openloop_rollout import _compose_frame, _write_mp4
from eval.train_temporal_chunk_bc_robocasa import RoboCasaTemporalChunkBC
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", type=int, default=90)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=8)
    parser.add_argument("--clip-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MP4 video")

    dataset_root = Path(args.dataset_root)
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.policy, map_location=device, weights_only=False)
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

    frames, success, steps = _rollout_closed_loop(
        dataset_root=dataset_root,
        episode_idx=int(args.episode_id),
        model=model,
        checkpoint=checkpoint,
        device=device,
        camera=str(args.camera),
        width=int(args.width),
        height=int(args.height),
        max_steps=int(args.max_steps),
        commit_steps=int(args.commit_steps),
        clip_actions=bool(args.clip_actions),
    )
    if frames:
        frames.extend([frames[-1].copy() for _ in range(args.fps)])
    _write_mp4(frames, Path(args.out), int(args.fps), ffmpeg)
    print(
        json.dumps(
            {
                "out": args.out,
                "frames": len(frames),
                "episode_id": int(args.episode_id),
                "success": bool(success),
                "steps": int(steps),
                "commit_steps": int(args.commit_steps),
                "camera": args.camera,
            },
            indent=2,
        )
    )


def _rollout_closed_loop(
    *,
    dataset_root: Path,
    episode_idx: int,
    model: RoboCasaTemporalChunkBC,
    checkpoint: dict,
    device: torch.device,
    camera: str,
    width: int,
    height: int,
    max_steps: int,
    commit_steps: int,
    clip_actions: bool,
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

    model_xml = LU.get_episode_model_xml(dataset_root, episode_idx)
    states = LU.get_episode_states(dataset_root, episode_idx)
    ep_meta = LU.get_episode_meta(dataset_root, episode_idx)
    reset_to(env, {"model": model_xml, "ep_meta": json.dumps(ep_meta), "states": states[0]})

    action_mean = _ckpt_tensor(checkpoint, "action_mean", device)
    action_std = _ckpt_tensor(checkpoint, "action_std", device)
    proprio_mean = _ckpt_tensor(checkpoint, "proprio_mean", device)
    proprio_std = _ckpt_tensor(checkpoint, "proprio_std", device)
    task_id = _episode_task_id(dataset_root, episode_idx, checkpoint)

    frames: list[np.ndarray] = []
    success = False
    step_idx = 0
    try:
        frames.append(_compose_frame(env, camera, width, height, step_idx, success=False))
        while step_idx < max_steps and not success:
            agent = _render64(env, "robot0_agentview_left")
            wrist = _render64(env, "robot0_agentview_right")
            proprio = _state_from_obs(env._get_observations())
            with torch.no_grad():
                agent_t = torch.as_tensor(agent[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                wrist_t = torch.as_tensor(wrist[None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                proprio_t = (torch.as_tensor(proprio[None], dtype=torch.float32, device=device) - proprio_mean) / proprio_std
                task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
                if str(checkpoint.get("policy_kind", "bc")) == "flow":
                    pred_norm = model.sample_flow(
                        agent_t,
                        wrist_t,
                        proprio_t,
                        task_t,
                        steps=int(checkpoint.get("flow_steps", 8)),
                    )[0]
                else:
                    pred_norm = model(agent_t, wrist_t, proprio_t, task_t)[0]
                pred = (pred_norm * action_std + action_mean).detach().cpu().numpy()
            actions = pred[: min(commit_steps, pred.shape[0], max_steps - step_idx)].astype(np.float32)
            if clip_actions:
                actions = np.clip(actions, -1.0, 1.0)
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


def _ckpt_tensor(checkpoint: dict, key: str, device: torch.device) -> torch.Tensor:
    value = checkpoint[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _episode_task_id(dataset_root: Path, episode_idx: int, checkpoint: dict) -> int:
    if not bool(checkpoint.get("condition_on_robocasa_task_index", False)):
        return 0
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet", columns=["task_index"])
    return int(frame["task_index"].iloc[0])


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
