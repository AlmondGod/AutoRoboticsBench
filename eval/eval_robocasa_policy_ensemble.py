from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from eval.render_robocasa_chunk_policy import _ckpt_tensor, _episode_task_id, _render64, _state_from_obs
from eval.train_temporal_chunk_bc_robocasa import RoboCasaTemporalChunkBC
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float, default=[])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    members = [_load_member(Path(path), device) for path in args.policy]
    weights = np.asarray(args.weight if args.weight else [1.0] * len(members), dtype=np.float32)
    if len(weights) != len(members):
        raise ValueError("--weight count must match --policy count")
    weights = weights / np.maximum(weights.sum(), 1e-6)

    details = []
    for episode_id in args.episode_id:
        success, steps = _rollout(
            members=members,
            weights=weights,
            dataset_root=Path(args.dataset_root),
            episode_idx=int(episode_id),
            device=device,
            max_steps=int(args.max_steps),
            commit_steps=int(args.commit_steps),
        )
        details.append({"episode_id": int(episode_id), "success": bool(success), "steps": int(steps)})
        print(json.dumps(details[-1]), flush=True)

    successes = sum(int(row["success"]) for row in details)
    payload = {
        "policies": args.policy,
        "weights": weights.tolist(),
        "episodes": len(details),
        "successes": successes,
        "success_rate": successes / max(1, len(details)),
        "commit_steps": int(args.commit_steps),
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_member(path: Path, device: torch.device) -> tuple[RoboCasaTemporalChunkBC, dict]:
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


def _rollout(
    *,
    members: list[tuple[RoboCasaTemporalChunkBC, dict]],
    weights: np.ndarray,
    dataset_root: Path,
    episode_idx: int,
    device: torch.device,
    max_steps: int,
    commit_steps: int,
) -> tuple[bool, int]:
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

    success = False
    step_idx = 0
    try:
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
                if success or step_idx >= max_steps:
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return success, step_idx


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
