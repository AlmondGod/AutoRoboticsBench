from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "robocasa5" / "manifest.json"
DEFAULT_SPLIT = ROOT / "data" / "autorobobench" / "robocasa_bc5_splits.json"
DEFAULT_POLICY_SET = ROOT / "data" / "autorobobench" / "robocasa_world_model_policy_set.json"
DEFAULT_VIDEO_POOL = ROOT / "data" / "autorobobench" / "robocasa_world_model_video_pool.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup verifier for RoboCasa reward-model eval.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--policy-set", default=str(DEFAULT_POLICY_SET))
    parser.add_argument("--video-pool", default=str(DEFAULT_VIDEO_POOL))
    parser.add_argument("--write-policy-template", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    split_path = Path(args.split)
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"missing split: {split_path}")
    video_pool_path = Path(args.video_pool)
    if not video_pool_path.exists():
        raise FileNotFoundError(f"missing video pool: {video_pool_path}")
    manifest = json.loads(manifest_path.read_text())
    split = json.loads(split_path.read_text())
    video_pool = json.loads(video_pool_path.read_text())
    if video_pool.get("contains_actions", True):
        raise ValueError("reward-model video pool must not expose actions")
    if video_pool.get("contains_proprio", True):
        raise ValueError("reward-model video pool must not expose proprio")
    if video_pool.get("contains_state", True):
        raise ValueError("reward-model video pool must not expose state")
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    tasks = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if alias not in manifest_tasks:
            raise ValueError(f"split task {alias!r} missing from manifest")
        tasks.append(
            {
                "alias": alias,
                "task_id": int(split_task["task_id"]),
                "train_episodes": len(split_task.get("train_episode_ids", [])),
                "val_episodes": len(split_task.get("val_episode_ids", [])),
                "eval_episodes": len(split_task.get("eval_episode_ids", [])),
                "dataset_path": manifest_tasks[alias].get("dataset_path", ""),
            }
        )

    policy_set = Path(args.policy_set)
    wrote_template = False
    if args.write_policy_template and not policy_set.exists():
        _save_json(policy_set, policy_template())
        wrote_template = True

    payload = {
        "task": "robocasa_reward_model",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "policy_set": str(policy_set),
        "policy_set_exists": policy_set.exists(),
        "video_pool": str(video_pool_path),
        "video_pool_exists": video_pool_path.exists(),
        "video_pool_contains_actions": bool(video_pool.get("contains_actions", True)),
        "video_pool_contains_proprio": bool(video_pool.get("contains_proprio", True)),
        "video_only_demo_count": int(video_pool.get("video_only_demo_count", 0)),
        "video_pool_task_count": len(video_pool.get("tasks", [])),
        "wrote_policy_template": wrote_template,
        "tasks": tasks,
        "task_count": len(tasks),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def policy_template() -> dict:
    return {
        "task": "robocasa_reward_model_policy_set",
        "description": "Optional policy set for reward-model score correlation against real RoboCasa evals.",
        "policies": [
            {
                "name": "bc5_baseline",
                "ood": False,
                "real_eval_json": "runs/autorobobench/robocasa_bc5/baseline/eval_success.json",
                "trace_dir": "runs/autorobobench/robocasa_bc5/baseline/world_model_traces",
                "notes": "Trace npz files must contain states, actions, and task_id.",
            },
            {
                "name": "ood_policy_example",
                "ood": True,
                "real_eval_json": "",
                "trace_npz_paths": [],
                "notes": "Fill with traces from an OOD policy family or checkpoint.",
            },
        ],
    }


def _save_json(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
