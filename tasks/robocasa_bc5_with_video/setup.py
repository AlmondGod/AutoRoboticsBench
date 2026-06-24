from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio

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


DEFAULT_MANIFEST = Path("data/robocasa5/manifest.json")
DEFAULT_SPLIT = Path("data/autorobobench/robocasa_bc5_with_video_splits.json")
DEFAULT_VIDEO_POOL = Path("data/autorobobench/robocasa_bc5_with_video_video_pool.json")
DEFAULT_VIEWS = ("robot0_agentview_left", "robot0_agentview_right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Video Policy Transfer task assets.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--video-pool", default=str(DEFAULT_VIDEO_POOL))
    parser.add_argument("--verify", action="store_true", help="Verify dataset files exist, not just metadata.")
    args = parser.parse_args()

    payload = verify_assets(Path(args.manifest), Path(args.split), Path(args.video_pool), verify_files=bool(args.verify))
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
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    video_tasks = {task["alias"]: task for task in video_pool["tasks"]}
    summary = []
    paired_frames = 0
    split_video_frames = 0
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if alias not in manifest_tasks:
            raise ValueError(f"split task {alias!r} missing from manifest")
        if alias not in video_tasks:
            raise ValueError(f"split task {alias!r} missing from action-free video pool")
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        paired_ids = [int(x) for x in split_task["paired_train_episode_ids"]]
        video_ids = _expand_range(split_task["video_only_episode_range"])
        pool_ids = _expand_range(video_tasks[alias]["video_episode_range"])
        if video_ids != pool_ids:
            raise ValueError(f"video pool ids for {alias} do not match split ids")
        val_ids = [int(x) for x in split_task["val_episode_ids"]]
        eval_ids = [int(x) for x in split_task["eval_episode_ids"]]

        overlaps = {
            "paired_video": sorted(set(paired_ids) & set(video_ids)),
            "train_val": sorted((set(paired_ids) | set(video_ids)) & set(val_ids)),
            "train_eval": sorted((set(paired_ids) | set(video_ids)) & set(eval_ids)),
            "val_eval": sorted(set(val_ids) & set(eval_ids)),
        }
        bad = {key: value for key, value in overlaps.items() if value}
        if bad:
            raise ValueError(f"split overlap for {alias}: {bad}")

        missing: list[str] = []
        if verify_files:
            for episode_id in paired_ids + val_ids + eval_ids:
                parquet = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
                if not parquet.exists():
                    missing.append(str(parquet))
            for episode_id in video_ids:
                for view in DEFAULT_VIEWS:
                    video = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_id:06d}.mp4"
                    if not video.exists():
                        missing.append(str(video))
        if verify_files and dataset_root.exists():
            paired_frames += sum(_frame_count(dataset_root, episode_id, DEFAULT_VIEWS[0]) for episode_id in paired_ids)
            split_video_frames += sum(_frame_count(dataset_root, episode_id, DEFAULT_VIEWS[0]) for episode_id in video_ids)
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} files for {alias}, first={missing[0]}")

        summary.append(
            {
                "alias": alias,
                "task_id": int(split_task["task_id"]),
                "dataset_path": str(dataset_root),
                "paired_action_train_episodes": len(paired_ids),
                "video_only_train_episodes": len(video_ids),
                "val_episodes": len(val_ids),
                "eval_episodes": len(eval_ids),
                "video_to_paired_ratio": len(video_ids) / max(1, len(paired_ids)),
                "video_interface": "rgb_only_action_free_manifest",
                "exists": dataset_root.exists(),
            }
        )

    pool_summary = []
    pool_video_frames = 0
    for task in video_pool["tasks"]:
        alias = str(task["alias"])
        dataset_root = Path(task["dataset_path"])
        if not dataset_root.is_absolute():
            dataset_root = Path.cwd() / dataset_root
        video_ids = _expand_range(task["video_episode_range"])
        missing = []
        if verify_files:
            for episode_id in video_ids:
                for view in DEFAULT_VIEWS:
                    video = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_id:06d}.mp4"
                    if not video.exists():
                        missing.append(str(video))
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} video-pool files for {alias}, first={missing[0]}")
        frames = 0
        if verify_files and dataset_root.exists():
            frames = sum(_frame_count(dataset_root, episode_id, DEFAULT_VIEWS[0]) for episode_id in video_ids)
        pool_video_frames += frames
        pool_summary.append(
            {
                "alias": alias,
                "dataset_path": str(dataset_root),
                "video_only_episodes": len(video_ids),
                "video_frames": int(frames),
                "exists": dataset_root.exists(),
            }
        )

    return {
        "task": "robocasa_bc5_with_video",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "video_pool": str(video_pool_path),
        "contains_actions_in_video_pool": bool(video_pool.get("contains_actions")),
        "contains_proprio_in_video_pool": bool(video_pool.get("contains_proprio")),
        "task_count": len(summary),
        "paired_action_train_episodes": sum(row["paired_action_train_episodes"] for row in summary),
        "video_only_train_episodes": sum(row["video_only_train_episodes"] for row in summary),
        "video_pool_episodes": sum(row["video_only_episodes"] for row in pool_summary),
        "video_to_paired_ratio": (
            sum(row["video_only_train_episodes"] for row in summary)
            / max(1, sum(row["paired_action_train_episodes"] for row in summary))
        ),
        "video_pool_to_paired_demo_ratio": (
            sum(row["video_only_episodes"] for row in pool_summary)
            / max(1, sum(row["paired_action_train_episodes"] for row in summary))
        ),
        "paired_action_frames": int(paired_frames if verify_files else video_pool.get("paired_action_frame_count", 0)),
        "split_video_only_frames": int(split_video_frames),
        "video_pool_frames": int(pool_video_frames if verify_files else video_pool.get("video_only_frame_count", 0)),
        "video_to_paired_frame_ratio": (
            pool_video_frames / max(1, paired_frames)
            if verify_files
            else float(video_pool.get("video_to_paired_frame_ratio", 0.0))
        ),
        "video_pool_tasks": pool_summary,
        "tasks": summary,
        "verified_files": bool(verify_files),
    }


def _expand_range(bounds: list[int]) -> list[int]:
    if len(bounds) != 2:
        raise ValueError(f"expected [start, end] range, got {bounds!r}")
    start, end = int(bounds[0]), int(bounds[1])
    if end < start:
        raise ValueError(f"invalid range {bounds!r}")
    return list(range(start, end + 1))


def _frame_count(dataset_root: Path, episode_id: int, view: str) -> int:
    video = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_id:06d}.mp4"
    if not video.exists():
        return 0
    meta = iio.immeta(video)
    frames = meta.get("nframes") or meta.get("n_frames")
    if frames and frames != float("inf"):
        return int(frames)
    return sum(1 for _ in iio.imiter(video))


if __name__ == "__main__":
    main()
