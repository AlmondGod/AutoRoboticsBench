from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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



ensure_robocasa_runtime()


DEFAULT_MANIFEST = Path("data/autorobobench/robocasa_bc1_manifest.json")
DEFAULT_SPLIT = Path("data/autorobobench/robocasa_bc1_splits.json")
DEFAULT_VIDEO_POOL = Path("data/autorobobench/robocasa_bc1_video_pool.json")
DEFAULT_VIEWS = ("robot0_agentview_left", "robot0_agentview_right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the RoboCasa faucet peak task assets.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--video-pool", default=str(DEFAULT_VIDEO_POOL))
    parser.add_argument("--verify", action="store_true", help="Verify dataset files exist, not just metadata.")
    args = parser.parse_args()

    payload = verify_assets(
        Path(args.manifest),
        Path(args.split),
        Path(args.video_pool),
        verify_files=bool(args.verify),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def verify_assets(manifest_path: Path, split_path: Path, video_pool_path: Path, *, verify_files: bool) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"missing frozen split: {split_path}")
    if not video_pool_path.exists():
        raise FileNotFoundError(f"missing action-free video pool: {video_pool_path}")

    manifest = json.loads(manifest_path.read_text())
    split = json.loads(split_path.read_text())
    video_pool = json.loads(video_pool_path.read_text())
    if video_pool.get("contains_actions") is not False:
        raise ValueError(f"{video_pool_path} must declare contains_actions=false")
    if video_pool.get("contains_proprio") is not False:
        raise ValueError(f"{video_pool_path} must declare contains_proprio=false")
    if video_pool.get("target_task_included") is not False:
        raise ValueError(f"{video_pool_path} must exclude the target faucet task")

    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    if set(manifest_tasks) != {"TurnOnSinkFaucet"}:
        raise ValueError("faucet peak manifest must contain only TurnOnSinkFaucet")

    summary = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if alias not in manifest_tasks:
            raise ValueError(f"split task {alias!r} missing from manifest")
        dataset_root = _resolve_path(manifest_tasks[alias]["dataset_path"])
        train_ids = [int(x) for x in split_task["train_episode_ids"]]
        val_ids = [int(x) for x in split_task["val_episode_ids"]]
        eval_ids = [int(x) for x in split_task["eval_episode_ids"]]
        overlaps = {
            "train_val": sorted(set(train_ids) & set(val_ids)),
            "train_eval": sorted(set(train_ids) & set(eval_ids)),
            "val_eval": sorted(set(val_ids) & set(eval_ids)),
        }
        bad = {key: value for key, value in overlaps.items() if value}
        if bad and not bool(split.get("intentional_train_eval_overlap")):
            raise ValueError(f"split overlap for {alias}: {bad}")
        missing = []
        if verify_files:
            for episode_id in train_ids + val_ids + eval_ids:
                parquet = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
                if not parquet.exists():
                    missing.append(str(parquet))
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} action files for {alias}, first={missing[0]}")
        summary.append(
            {
                "alias": alias,
                "dataset_path": str(dataset_root),
                "train_episodes": len(train_ids),
                "val_episodes": len(val_ids),
                "eval_episodes": len(eval_ids),
                "overlaps": overlaps,
                "exists": dataset_root.exists(),
            }
        )

    pool_summary = []
    for task in video_pool["tasks"]:
        alias = str(task["alias"])
        if alias == "TurnOnSinkFaucet":
            raise ValueError("generic video pool must not include the target task")
        dataset_root = _resolve_path(task["dataset_path"])
        video_ids = _expand_range(task["video_episode_range"])
        missing = []
        if verify_files:
            for episode_id in video_ids:
                for view in DEFAULT_VIEWS:
                    video = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_id:06d}.mp4"
                    if not video.exists():
                        missing.append(str(video))
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} generic video files for {alias}, first={missing[0]}")
        pool_summary.append(
            {
                "alias": alias,
                "dataset_path": str(dataset_root),
                "video_only_episodes": len(video_ids),
                "exists": dataset_root.exists(),
            }
        )

    return {
        "task": "robocasa_bc1",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "video_pool": str(video_pool_path),
        "same_sink_protocol": split.get("same_sink_protocol"),
        "visual_eval_protocol": split.get("visual_eval_protocol"),
        "anti_replay_eval_protocol": split.get("anti_replay_eval_protocol"),
        "task_count": len(summary),
        "task_specific_action_train_episodes": sum(row["train_episodes"] for row in summary),
        "all_target_trajectory_data_available": bool(split.get("all_target_trajectory_data_available")),
        "intentional_train_eval_overlap": bool(split.get("intentional_train_eval_overlap")),
        "generic_video_pool_tasks": len(pool_summary),
        "generic_video_only_episodes": sum(row["video_only_episodes"] for row in pool_summary),
        "video_pool_contains_actions": bool(video_pool.get("contains_actions")),
        "video_pool_contains_proprio": bool(video_pool.get("contains_proprio")),
        "target_task_in_video_pool": bool(video_pool.get("target_task_included")),
        "tasks": summary,
        "video_pool_tasks": pool_summary,
        "verified_files": bool(verify_files),
    }


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _expand_range(bounds: list[int]) -> list[int]:
    if len(bounds) != 2:
        raise ValueError(f"expected [start, end] range, got {bounds!r}")
    start, end = int(bounds[0]), int(bounds[1])
    if end < start:
        raise ValueError(f"invalid range {bounds!r}")
    return list(range(start, end + 1))


if __name__ == "__main__":
    main()
