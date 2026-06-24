from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "robocasa5" / "manifest.json"
DEFAULT_SPLIT = ROOT / "data" / "autorobobench" / "robocasa_bc5_splits.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup verifier for RoboCasa visual world-model eval.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--view", default="robot0_agentview_right")
    parser.add_argument("--skip-lpips-check", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    split_path = Path(args.split)
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"missing split: {split_path}")
    lpips_available = _lpips_available()
    if not lpips_available and not bool(args.skip_lpips_check):
        raise ModuleNotFoundError("robocasa_visual_world_model eval requires lpips. Install with `pip install lpips`.")
    manifest = json.loads(manifest_path.read_text())
    split = json.loads(split_path.read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    tasks = []
    missing_videos = []
    for split_task in split["tasks"]:
        alias = str(split_task["alias"])
        if alias not in manifest_tasks:
            raise ValueError(f"split task {alias!r} missing from manifest")
        dataset_root = Path(str(manifest_tasks[alias]["dataset_path"]))
        train_ids = [int(x) for x in split_task.get("train_episode_ids", [])]
        val_ids = [int(x) for x in split_task.get("val_episode_ids", [])]
        checked = 0
        for episode_id in train_ids[:1] + val_ids[:1]:
            video = dataset_root / "videos" / "chunk-000" / f"observation.images.{args.view}" / f"episode_{episode_id:06d}.mp4"
            checked += 1
            if not video.exists():
                missing_videos.append(str(video))
        tasks.append(
            {
                "alias": alias,
                "task_id": int(split_task["task_id"]),
                "train_episodes": len(train_ids),
                "val_episodes": len(val_ids),
                "checked_video_files": checked,
            }
        )
    if missing_videos:
        raise FileNotFoundError(f"missing required video files: {missing_videos[:5]}")
    print(
        json.dumps(
            {
                "task": "robocasa_visual_world_model",
                "manifest": str(manifest_path),
                "split": str(split_path),
                "view": str(args.view),
                "lpips_available": bool(lpips_available),
                "task_count": len(tasks),
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
    )

def _lpips_available() -> bool:
    try:
        import lpips  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


if __name__ == "__main__":
    main()
