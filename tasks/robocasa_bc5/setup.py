from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


DEFAULT_SPLIT = Path("data/autorobobench/robocasa_bc5_splits.json")
DEFAULT_VIEWS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
]
DEFAULT_TASKS = [
    ("OpenCabinet", "Open a kitchen cabinet."),
    ("CloseDrawer", "Close a kitchen drawer."),
    ("CloseFridge", "Close a fridge door."),
    ("TurnOffStove", "Turn a stove knob off."),
    ("PickPlaceCounterToCabinet", "Pick an object from the counter and place it into a cabinet region."),
]


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


def build_manifest(
    out_dir: Path,
    *,
    split: str,
    source: str,
    policy_demos_per_task: int,
    views: list[str],
    verify_exists: bool,
) -> dict:
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    tasks = []
    for task_id, (alias, description) in enumerate(DEFAULT_TASKS):
        meta = get_ds_meta(task=alias, split=split, source=source)
        if meta is None:
            raise ValueError(f"no RoboCasa dataset metadata for {alias}")
        dataset_path = Path(meta["path"])
        stats = _inspect_lerobot_dataset(dataset_path, views)
        registered_demos = _registered_demo_count(meta["filter_key"])
        if verify_exists and not stats["exists"]:
            raise FileNotFoundError(f"missing dataset for {alias}: {dataset_path}")
        tasks.append(
            {
                "task_id": task_id,
                "alias": alias,
                "robocasa_task": alias,
                "description": description,
                "dataset_path": str(dataset_path),
                "split": split,
                "source": source,
                "horizon": meta["horizon"],
                "filter_key": meta["filter_key"],
                "exists": stats["exists"],
                "registered_demos": registered_demos,
                "available_demos": stats["num_episodes"],
                "selected_demos": min(policy_demos_per_task, registered_demos or stats["num_episodes"] or policy_demos_per_task),
                "available_views": stats["available_views"],
            }
        )

    manifest = {
        "benchmark": "robocasa5",
        "suite": "robocasa",
        "split": split,
        "source": source,
        "policy_demos_per_task": policy_demos_per_task,
        "views": views,
        "action_dim": 7,
        "tasks": tasks,
        "task_count": len(tasks),
        "total_registered_demos": sum(int(task["registered_demos"]) for task in tasks),
        "total_available_demos": sum(int(task["available_demos"]) for task in tasks),
        "total_selected_demos": sum(int(task["selected_demos"]) for task in tasks),
        "notes": [
            "Built from the RoboCasa registry via robocasa.utils.dataset_registry_utils.get_ds_meta.",
            "Datasets are expected in LeRobot format under the local RoboCasa dataset base path.",
            "selected_demos may be smaller than available_demos for policy training.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _inspect_lerobot_dataset(dataset_path: Path, expected_views: list[str]) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    exists = dataset_path.exists()
    num_episodes = 0
    features = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            num_episodes = int(info.get("total_episodes") or info.get("num_episodes") or 0)
            features = info.get("features") or {}
        except (TypeError, ValueError):
            pass
    if num_episodes == 0 and episodes_path.exists():
        num_episodes = sum(1 for line in episodes_path.read_text().splitlines() if line.strip())

    available_views = [view for view in expected_views if f"observation.images.{view}" in features]
    if not available_views:
        available_views = [
            key.removeprefix("observation.images.")
            for key in sorted(features)
            if key.startswith("observation.images.")
        ]
    return {"exists": exists, "num_episodes": num_episodes, "available_views": available_views}


def _registered_demo_count(filter_key: str) -> int:
    try:
        return int(str(filter_key).split("_", 1)[0])
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
