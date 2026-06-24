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

from tasks.robocasa_bc5.train import main  # noqa: E402

# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0


def _default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def _fixed_train_cap() -> None:
    value = _arg_value("--max-train-seconds")
    if value is None:
        sys.argv.extend(["--max-train-seconds", str(int(BENCHMARK_TRAIN_SECONDS_CAP))])
        return
    if float(value) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-train-seconds is fixed at 300 for scored runs and cannot be overwritten")


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


if __name__ == "__main__":
    _default("--split", "data/autorobobench/robocasa_bc5_with_video_splits.json")
    _default("--video-pool", "data/autorobobench/robocasa_bc5_with_video_video_pool.json")
    _default("--out-dir", "runs/autorobobench/robocasa_bc5_with_video/scarce_paired_bc")
    _default("--train-episodes-per-task", "5")
    _default("--val-episodes-per-task", "10")
    _default("--chunk-horizon", "16")
    _default("--frame-stride", "1")
    _fixed_train_cap()
    _default("--batch-size", "128")
    _default("--width", "256")
    _default("--dropout", "0.05")
    _default("--lr", "2e-4")
    _default("--image-noise", "0.01")
    _default("--proprio-noise", "0.01")
    _default("--action-smooth", "0.0005")
    _default("--chunk-decay", "0.8")
    _default("--pidm-pretrain-seconds", "120")
    _default("--pidm-video-episodes-per-task", "16")
    _default("--pidm-batch-size", "128")
    _default("--pidm-gap", "4")
    _default("--pidm-action-weight", "1.0")
    _default("--pidm-latent-weight", "1.0")
    _default("--video-pretrain-seconds", "0")
    _default("--video-pretrain-episodes-per-task", "16")
    _default("--video-pretrain-batch-size", "128")
    _default("--video-pretrain-gap", "8")
    _default("--video-pretrain-objective", "hybrid")
    _default("--video-latent-dynamics-weight", "1.0")
    _default("--vpt-idm-seconds", "0")
    _default("--vpt-pseudo-episodes-per-task", "0")
    main()
