from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from eval.render_robocasa_chunk_policy import _ckpt_tensor, _episode_task_id, _render64, _rollout_closed_loop, _state_from_obs
from eval.train_temporal_chunk_bc_robocasa import RoboCasaTemporalChunkBC
from train.common import device_from_arg

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=1)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--ensemble-decay", type=float, default=0.7)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

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

    details = []
    for episode_id in args.episode_id:
        kwargs = dict(
            dataset_root=Path(args.dataset_root),
            episode_idx=int(episode_id),
            model=model,
            checkpoint=checkpoint,
            device=device,
            camera=str(args.camera),
            width=64,
            height=64,
            max_steps=int(args.max_steps),
            commit_steps=int(args.commit_steps),
            clip_actions=True,
        )
        if args.temporal_ensemble:
            _, success, steps = _rollout_temporal_ensemble(**kwargs, ensemble_decay=float(args.ensemble_decay))
        else:
            _, success, steps = _rollout_closed_loop(**kwargs)
        details.append({"episode_id": int(episode_id), "success": bool(success), "steps": int(steps)})
        print(json.dumps(details[-1]), flush=True)

    payload = {
        "policy": args.policy,
        "episodes": len(details),
        "successes": sum(int(row["success"]) for row in details),
        "success_rate": sum(int(row["success"]) for row in details) / max(1, len(details)),
        "commit_steps": int(args.commit_steps),
        "temporal_ensemble": bool(args.temporal_ensemble),
        "ensemble_decay": float(args.ensemble_decay),
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


def _rollout_temporal_ensemble(
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
    ensemble_decay: float,
    task_id_override: int | None = None,
) -> tuple[list[np.ndarray], bool, int]:
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

    action_mean = _ckpt_tensor(checkpoint, "action_mean", device)
    action_std = _ckpt_tensor(checkpoint, "action_std", device)
    proprio_mean = _ckpt_tensor(checkpoint, "proprio_mean", device)
    proprio_std = _ckpt_tensor(checkpoint, "proprio_std", device)
    task_id = int(task_id_override) if task_id_override is not None else _episode_task_id(dataset_root, episode_idx, checkpoint)
    queued: dict[int, list[np.ndarray]] = {}

    frames: list[np.ndarray] = []
    success = False
    step_idx = 0
    try:
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
                    pred_norm = model.sample_flow(agent_t, wrist_t, proprio_t, task_t, steps=int(checkpoint.get("flow_steps", 8)))[0]
                else:
                    pred_norm = model(agent_t, wrist_t, proprio_t, task_t)[0]
                pred = (pred_norm * action_std + action_mean).detach().cpu().numpy()
            horizon = min(pred.shape[0], max_steps - step_idx)
            for offset in range(horizon):
                queued.setdefault(step_idx + offset, []).append(pred[offset].astype(np.float32))
            for _ in range(max(1, int(commit_steps))):
                candidates = queued.pop(step_idx, [])
                if not candidates:
                    action = pred[min(step_idx, pred.shape[0] - 1)].astype(np.float32)
                else:
                    weights = np.asarray([ensemble_decay ** i for i in range(len(candidates) - 1, -1, -1)], dtype=np.float32)
                    weights = weights / np.maximum(weights.sum(), 1e-6)
                    action = np.sum(np.stack(candidates) * weights[:, None], axis=0).astype(np.float32)
                if clip_actions:
                    action = np.clip(action, -1.0, 1.0)
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
    return frames, success, step_idx


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
