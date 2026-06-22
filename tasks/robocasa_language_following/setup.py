from __future__ import annotations

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

FROZEN_MANIFEST = ROOT / "data/autorobobench/robocasa_language_following_manifest.json"
FROZEN_SPLIT = ROOT / "data/autorobobench/robocasa_language_following_splits.json"


def main() -> None:
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    split = json.loads(FROZEN_SPLIT.read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    missing = []
    for split_task in split["tasks"]:
        task = manifest_tasks[split_task["alias"]]
        dataset = ROOT / task["dataset_path"]
        if not dataset.exists():
            missing.append(str(dataset))
            continue
        _check_episode_language(dataset, split_task)
    payload = {
        "task": "robocasa_language_following",
        "manifest": str(FROZEN_MANIFEST),
        "split": str(FROZEN_SPLIT),
        "language_variants": len(split["tasks"]),
        "missing_datasets": missing,
        "ok": not missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise FileNotFoundError("missing ChooseMeasuringCup dataset; run RoboCasa downloader for ChooseMeasuringCup pretrain human")


def _check_episode_language(dataset: Path, split_task: dict) -> None:
    rows = {}
    for line in (dataset / "meta/episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["episode_index"])] = str(row.get("tasks", [""])[0])
    expected = str(split_task["language"])
    ids = (
        list(split_task["train_episode_ids"])
        + list(split_task["val_episode_ids"])
        + list(split_task["eval_episode_ids"])
    )
    bad = [episode_id for episode_id in ids if rows.get(int(episode_id)) != expected]
    if bad:
        raise ValueError(f"{split_task['alias']} has episode language mismatches: {bad[:8]}")


if __name__ == "__main__":
    main()
