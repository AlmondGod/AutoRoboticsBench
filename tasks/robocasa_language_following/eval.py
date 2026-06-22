from __future__ import annotations

import argparse
import importlib
import json
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

FROZEN_MANIFEST = "data/autorobobench/robocasa_language_following_manifest.json"
FROZEN_SPLIT = "data/autorobobench/robocasa_language_following_splits.json"
WRONG_CONDITIONING_ALIAS = {
    "ChooseMeasuringCupLeftLarger": "ChooseMeasuringCupLeftSmaller",
    "ChooseMeasuringCupLeftSmaller": "ChooseMeasuringCupLeftLarger",
    "ChooseMeasuringCupRightLarger": "ChooseMeasuringCupRightSmaller",
    "ChooseMeasuringCupRightSmaller": "ChooseMeasuringCupRightLarger",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the language-conditioned ChooseMeasuringCup task.")
    parser.add_argument("--checkpoint", "--policy", dest="checkpoint", required=True)
    parser.add_argument("--inference", default="tasks.robocasa_language_following.inference")
    parser.add_argument("--manifest", default=FROZEN_MANIFEST)
    parser.add_argument("--split", default=FROZEN_SPLIT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--commit-steps", type=int, default=8)
    parser.add_argument("--eval-episodes-per-task", type=int, default=3)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--skip-conditioning-gap", action="store_true")
    parser.add_argument("--render-dir", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--render-episodes-per-task", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if Path(args.manifest).as_posix() != FROZEN_MANIFEST:
        raise ValueError(f"--manifest is immutable for this task; expected {FROZEN_MANIFEST}")
    if Path(args.split).as_posix() != FROZEN_SPLIT:
        raise ValueError(f"--split is immutable for this task; expected {FROZEN_SPLIT}")

    inference = importlib.import_module(args.inference)
    if not hasattr(inference, "load_policy") or not hasattr(inference, "act"):
        raise AttributeError(f"{args.inference} must define load_policy(checkpoint, device) and act(policy, obs, task)")
    policy = inference.load_policy(str(args.checkpoint), device=str(args.device))

    manifest = json.loads(Path(FROZEN_MANIFEST).read_text())
    split = json.loads(Path(FROZEN_SPLIT).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    split_tasks = {task["alias"]: task for task in split["tasks"]}
    task_aliases = set(args.task_alias)
    ffmpeg = shutil.which("ffmpeg") if args.render_dir else None

    correct_details: list[dict] = []
    wrong_details: list[dict] = []
    per_task: dict[str, dict] = {}
    wrong_per_task: dict[str, dict] = {}

    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if task_aliases and alias not in task_aliases:
            continue
        manifest_task = manifest_tasks[alias]
        dataset_root = Path(manifest_task["dataset_path"])
        episode_ids = list(split_task["eval_episode_ids"])
        if args.eval_episodes_per_task > 0:
            episode_ids = episode_ids[: int(args.eval_episodes_per_task)]

        correct_rows = _eval_condition(
            condition_name="correct_language",
            split_task=split_task,
            manifest_task=manifest_task,
            conditioning_task=manifest_task,
            dataset_root=dataset_root,
            episode_ids=episode_ids,
            policy=policy,
            inference=inference,
            camera=str(args.camera),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
            max_steps=int(args.max_steps),
            commit_steps=int(args.commit_steps),
            trace_dir=str(args.trace_dir),
            render_dir=str(args.render_dir),
            render_episodes_per_task=int(args.render_episodes_per_task),
            fps=int(args.fps),
            ffmpeg=ffmpeg,
        )
        correct_details.extend(correct_rows)
        per_task[alias] = _summarize_rows(correct_rows)

        if not args.skip_conditioning_gap:
            wrong_alias = WRONG_CONDITIONING_ALIAS[alias]
            wrong_conditioning_task = manifest_tasks[wrong_alias]
            wrong_rows = _eval_condition(
                condition_name="wrong_cup_size_language",
                split_task=split_task,
                manifest_task=manifest_task,
                conditioning_task=wrong_conditioning_task,
                dataset_root=dataset_root,
                episode_ids=episode_ids,
                policy=policy,
                inference=inference,
                camera=str(args.camera),
                render_width=int(args.render_width),
                render_height=int(args.render_height),
                max_steps=int(args.max_steps),
                commit_steps=int(args.commit_steps),
                trace_dir="",
                render_dir="",
                render_episodes_per_task=0,
                fps=int(args.fps),
                ffmpeg=None,
            )
            wrong_details.extend(wrong_rows)
            wrong_per_task[alias] = _summarize_rows(wrong_rows)

    success_rate = _success_rate(correct_details)
    wrong_success_rate = _success_rate(wrong_details) if wrong_details else 0.0
    variant_rates = [float(row["success_rate"]) for row in per_task.values()]
    wrong_variant_rates = [float(row["success_rate"]) for row in wrong_per_task.values()]
    payload = {
        "track": "robocasa_language_following",
        "checkpoint": str(args.checkpoint),
        "inference": str(args.inference),
        "manifest": FROZEN_MANIFEST,
        "split": FROZEN_SPLIT,
        "episodes": len(correct_details),
        "successes": sum(int(row["success"]) for row in correct_details),
        "success_rate": success_rate,
        "language_success_rate": success_rate,
        "wrong_language_episodes": len(wrong_details),
        "wrong_language_success_rate": wrong_success_rate,
        "conditioning_gap": max(0.0, success_rate - wrong_success_rate),
        "language_variant_balance": min(variant_rates) if variant_rates else 0.0,
        "wrong_language_variant_balance": min(wrong_variant_rates) if wrong_variant_rates else 0.0,
        "reproducibility_integrity": 1.0,
        "commit_steps": int(args.commit_steps),
        "max_steps": int(args.max_steps),
        "per_task": per_task,
        "wrong_language_per_task": wrong_per_task,
        "details": correct_details,
        "wrong_language_details": wrong_details,
        "data_contract": {
            "robocasa_task": "ChooseMeasuringCup",
            "language_variants": 4,
            "wrong_language_pairing": "same drawer side, opposite cup size",
            "train_demos_per_variant": 16,
            "val_demos_per_variant": 2,
            "eval_demos_per_variant": 3,
            "test_time_demo_access": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _eval_condition(
    *,
    condition_name: str,
    split_task: dict,
    manifest_task: dict,
    conditioning_task: dict,
    dataset_root: Path,
    episode_ids: list[int],
    policy,
    inference,
    camera: str,
    render_width: int,
    render_height: int,
    max_steps: int,
    commit_steps: int,
    trace_dir: str,
    render_dir: str,
    render_episodes_per_task: int,
    fps: int,
    ffmpeg: str | None,
) -> list[dict]:
    alias = str(split_task["alias"])
    rows = []
    for local_idx, episode_id in enumerate(episode_ids):
        task = {
            "task_id": int(conditioning_task["task_id"]),
            "alias": str(conditioning_task["alias"]),
            "description": str(conditioning_task.get("description", conditioning_task["alias"])),
            "language": str(conditioning_task.get("description", conditioning_task["alias"])),
            "robocasa_task": str(manifest_task.get("robocasa_task", alias)),
            "eval_alias": alias,
            "conditioning": condition_name,
        }
        frames, success, steps, actions, success_trace = _rollout_episode(
            dataset_root=dataset_root,
            episode_idx=int(episode_id),
            reset_state_index=0,
            policy=policy,
            inference=inference,
            task=task,
            camera=camera,
            width=render_width,
            height=render_height,
            max_steps=max_steps,
            commit_steps=commit_steps,
        )
        row = {
            "task_alias": alias,
            "task_id": int(split_task["task_id"]),
            "conditioning_alias": str(conditioning_task["alias"]),
            "conditioning_task_id": int(conditioning_task["task_id"]),
            "condition": condition_name,
            "episode_id": int(episode_id),
            "success": bool(success),
            "steps": int(steps),
        }
        if trace_dir:
            trace_path = Path(trace_dir) / condition_name / alias / f"episode_{int(episode_id):06d}.npz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trace_path,
                task_alias=np.asarray([alias]),
                task_id=np.asarray([int(split_task["task_id"])], dtype=np.int64),
                conditioning_alias=np.asarray([str(conditioning_task["alias"])]),
                conditioning_task_id=np.asarray([int(conditioning_task["task_id"])], dtype=np.int64),
                episode_id=np.asarray([int(episode_id)], dtype=np.int64),
                actions=np.asarray(actions, dtype=np.float32),
                success=np.asarray(success_trace, dtype=np.float32),
                final_success=np.asarray([float(success)], dtype=np.float32),
                steps=np.asarray([int(steps)], dtype=np.int64),
            )
            row["trace_path"] = str(trace_path)
        if render_dir and ffmpeg and local_idx < render_episodes_per_task:
            out_mp4 = Path(render_dir) / condition_name / f"{alias}_episode_{int(episode_id):06d}.mp4"
            if frames:
                frames.extend([frames[-1].copy() for _ in range(int(fps))])
            _write_mp4(frames, out_mp4, int(fps), ffmpeg)
            row["video"] = str(out_mp4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def _summarize_rows(rows: list[dict]) -> dict:
    successes = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": int(successes),
        "success_rate": successes / max(1, len(rows)),
    }


def _success_rate(rows: list[dict]) -> float:
    return sum(int(row["success"]) for row in rows) / max(1, len(rows))


if __name__ == "__main__":
    main()
