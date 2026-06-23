from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
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

from tasks.robocasa_bc5.eval import _rollout_episode, _write_mp4


ensure_robocasa_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(description="Immutable evaluator for the long-horizon sequential RoboCasa task.")
    parser.add_argument("--checkpoint", "--policy", dest="checkpoint", required=True)
    parser.add_argument("--inference", default="tasks.robocasa_long_horizon.inference")
    parser.add_argument("--manifest", default="data/autorobobench/robocasa_long_horizon_manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_long_horizon_splits.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--commit-steps", type=int, default=8)
    parser.add_argument("--eval-episodes-per-task", type=int, default=100)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--render-dir", default="")
    parser.add_argument("--trace-dir", default="")
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
    split_tasks = {task["alias"]: task for task in split["tasks"]}
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
        episode_ids = _select_eval_episode_ids(
            list(split_task["eval_episode_ids"]),
            int(args.eval_episodes_per_task),
        )
        successes = 0
        subgoal_progress = 0.0
        task_details = []
        task = {
            "task_id": int(split_task["task_id"]),
            "alias": alias,
            "description": manifest_task.get("description", alias),
            "robocasa_task": manifest_task.get("robocasa_task", alias),
            "subgoals": list(split_task.get("subgoals", [])),
        }
        for local_idx, episode_id in enumerate(episode_ids):
            os.environ["AUTOROBOBENCH_EVAL_EPISODE_ID"] = str(int(episode_id))
            frames, success, steps, actions, success_trace = _rollout_episode(
                dataset_root=dataset_root,
                episode_idx=int(episode_id),
                reset_state_index=0,
                reset_perturbation={},
                policy=policy,
                inference=inference,
                task=task,
                camera=str(args.camera),
                width=int(args.render_width),
                height=int(args.render_height),
                max_steps=int(args.max_steps),
                commit_steps=int(args.commit_steps),
            )
            progress = _episode_progress(success=bool(success), steps=int(steps), max_steps=int(args.max_steps))
            row = {
                "task_alias": alias,
                "task_id": int(split_task["task_id"]),
                "episode_id": int(episode_id),
                "eval_local_idx": int(local_idx),
                "success": bool(success),
                "steps": int(steps),
                "subgoal_progress": float(progress),
            }
            if args.trace_dir:
                trace_path = Path(args.trace_dir) / alias / f"episode_{int(episode_id):06d}.npz"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    trace_path,
                    task_alias=np.asarray([alias]),
                    task_id=np.asarray([int(split_task["task_id"])], dtype=np.int64),
                    episode_id=np.asarray([int(episode_id)], dtype=np.int64),
                    actions=np.asarray(actions, dtype=np.float32),
                    success=np.asarray(success_trace, dtype=np.float32),
                    final_success=np.asarray([float(success)], dtype=np.float32),
                    subgoal_progress=np.asarray([float(progress)], dtype=np.float32),
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
            subgoal_progress += float(progress)
            print(json.dumps(row), flush=True)
        per_task[alias] = {
            "episodes": len(task_details),
            "successes": int(successes),
            "success_rate": successes / max(1, len(task_details)),
            "subgoal_progress": subgoal_progress / max(1, len(task_details)),
            "subgoals": list(split_tasks[alias].get("subgoals", [])),
        }

    success_count = sum(int(row["success"]) for row in details)
    payload = {
        "track": "robocasa_long_horizon",
        "checkpoint": str(args.checkpoint),
        "inference": str(args.inference),
        "manifest": str(args.manifest),
        "split": str(args.split),
        "episodes": len(details),
        "successes": success_count,
        "success_rate": success_count / max(1, len(details)),
        "hidden_final_success": success_count / max(1, len(details)),
        "subgoal_progress": sum(float(row["subgoal_progress"]) for row in details) / max(1, len(details)),
        "commit_steps": int(args.commit_steps),
        "max_steps": int(args.max_steps),
        "per_task": per_task,
        "details": details,
        "notes": [
            "Public subgoal_progress is a coarse progress proxy based on full success and episode budget use.",
            "Hidden scoring may replace this field with simulator task-specific subgoal predicates."
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _episode_progress(*, success: bool, steps: int, max_steps: int) -> float:
    if success:
        return 1.0
    # Without public per-task subgoal predicates, expose a bounded partial-credit
    # proxy so result JSONs carry the suite's expected metric key.
    return 0.25 * max(0.0, 1.0 - float(steps) / max(1.0, float(max_steps)))


def _select_eval_episode_ids(episode_ids: list[int], requested: int) -> list[int]:
    if requested <= 0:
        return episode_ids
    if requested <= len(episode_ids):
        return episode_ids[:requested]
    if not episode_ids:
        return []
    return [int(episode_ids[idx % len(episode_ids)]) for idx in range(requested)]


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    main()
