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


FROZEN_MANIFEST = "data/autorobobench/robocasa_stand_mixer_peak_manifest.json"
FROZEN_SPLIT = "data/autorobobench/robocasa_stand_mixer_peak_splits.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup verifier for the RoboCasa offline-RL posttraining task.")
    parser.add_argument("--manifest", default=FROZEN_MANIFEST)
    parser.add_argument("--split", default=FROZEN_SPLIT)
    parser.add_argument("--verify", action="store_true", help="Verify required local dataset paths exist.")
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
        "task": "robocasa_offlinerl_posttraining",
        "manifest": str(manifest_path),
        "split": str(split_path),
        "task_count": len(summary),
        "target_task": "PickPlaceCounterToStandMixer",
        "offline_experience_contract": {
            "demo_advantage": 1.0,
            "bad_rollout_advantage": -1.0,
            "correction_advantage": 1.0,
        },
        "tasks": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
