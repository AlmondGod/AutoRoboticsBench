from __future__ import annotations

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


FROZEN_MANIFEST = "data/autorobobench/robocasa_bc1_manifest.json"
FROZEN_SPLIT = "data/autorobobench/robocasa_bc1_splits.json"
FROZEN_VIDEO_POOL = "data/autorobobench/robocasa_bc1_video_pool.json"


def main() -> None:
    _default("--manifest", FROZEN_MANIFEST)
    _default("--split", FROZEN_SPLIT)
    _default("--video-pool", FROZEN_VIDEO_POOL)
    _default("--out-dir", "runs/autorobobench/robocasa_bc1/bc_base")
    _default("--train-episodes-per-task", "23")
    _default("--val-episodes-per-task", "10")
    _default("--chunk-horizon", "16")
    _default("--frame-stride", "1")
    _default("--max-train-seconds", "300")
    _default("--batch-size", "128")
    _default("--width", "256")
    _default("--dropout", "0.03")
    _default("--image-noise", "0.004")
    _default("--proprio-noise", "0.004")
    _default("--chunk-decay", "0.82")
    _default("--action-smooth", "0.0005")
    _default("--progress-scale", "750")
    _default("--eval-commit-steps", "8")

    from tasks.robocasa_bc5.train import main as train_main

    train_main()


def _default(flag: str, value: str | None) -> None:
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    main()
