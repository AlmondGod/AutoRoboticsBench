from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()

from data.robocasa_dataset import DEFAULT_VIEWS, build_manifest  # noqa: E402


DEFAULT_SPLIT = Path("data/autorobobench/robocasa_bc5_splits.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time setup verifier for the RoboCasa BC-5 task.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--verify", action="store_true", help="Verify required local files and datasets exist.")
    parser.add_argument("--make-manifest", action="store_true", help="Rebuild data/robocasa5/manifest.json from local RoboCasa registry.")
    parser.add_argument("--source", default="human", choices=["human", "mg", "mg_5x5", "mg_5x1"])
    parser.add_argument("--data-split", default="pretrain", choices=["pretrain", "target", "real"])
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if args.make_manifest:
        build_manifest(
            manifest_path.parent,
            split=str(args.data_split),
            source=str(args.source),
            policy_demos_per_task=50,
            views=list(DEFAULT_VIEWS),
            verify_exists=bool(args.verify),
        )

    split_path = Path(args.split)
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"missing frozen split: {split_path}")

    manifest = json.loads(manifest_path.read_text())
    split = json.loads(split_path.read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    summary = []
    for split_task in split["tasks"]:
        alias = split_task["alias"]
        if alias not in manifest_tasks:
            raise ValueError(f"split task {alias!r} missing from manifest")
        dataset_path = Path(manifest_tasks[alias]["dataset_path"])
        if args.verify and not dataset_path.exists():
            raise FileNotFoundError(f"missing dataset for {alias}: {dataset_path}")
        summary.append(
            {
                "alias": alias,
                "dataset_path": str(dataset_path),
                "train_episodes": len(split_task["train_episode_ids"]),
                "val_episodes": len(split_task["val_episode_ids"]),
                "eval_episodes": len(split_task["eval_episode_ids"]),
                "exists": dataset_path.exists(),
            }
        )

    payload = {
        "task": "robocasa_bc5",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "task_count": len(summary),
        "tasks": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
