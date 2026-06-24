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


DEFAULT_SPLIT = Path("data/autorobobench/robocasa_long_horizon_splits.json")
DEFAULT_MANIFEST = Path("data/autorobobench/robocasa_long_horizon_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time setup verifier for the long-horizon sequential RoboCasa task.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--verify", action="store_true", help="Verify required local files and datasets exist.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
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
        manifest_task = manifest_tasks[alias]
        dataset_path = Path(manifest_task["dataset_path"])
        if args.verify and not dataset_path.exists():
            raise FileNotFoundError(f"missing dataset for {alias}: {dataset_path}")
        horizon = int(manifest_task.get("horizon", 0))
        if horizon < 600:
            raise ValueError(f"long-horizon task {alias!r} has short manifest horizon={horizon}")
        summary.append(
            {
                "alias": alias,
                "dataset_path": str(dataset_path),
                "horizon": horizon,
                "subgoals": list(split_task.get("subgoals", [])),
                "train_episodes": len(split_task["train_episode_ids"]),
                "val_episodes": len(split_task["val_episode_ids"]),
                "eval_episodes": len(split_task["eval_episode_ids"]),
                "exists": dataset_path.exists(),
            }
        )

    payload = {
        "task": "robocasa_long_horizon",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "task_count": len(summary),
        "tasks": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
