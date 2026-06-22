from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_bc5.train import main  # noqa: E402


DEFAULT_ARGS = {
    "--manifest": "data/autorobobench/robocasa_long_horizon_manifest.json",
    "--split": "data/autorobobench/robocasa_long_horizon_splits.json",
    "--out-dir": "runs/autorobobench/robocasa_long_horizon/baseline",
    "--train-episodes-per-task": "80",
    "--val-episodes-per-task": "10",
    "--chunk-horizon": "16",
    "--frame-stride": "1",
    "--max-train-seconds": "300",
    "--batch-size": "128",
    "--width": "256",
    "--dropout": "0.03",
    "--image-noise": "0.004",
    "--proprio-noise": "0.004",
    "--chunk-decay": "0.82",
    "--action-smooth": "0.0005",
    "--progress-scale": "750",
    "--eval-commit-steps": "8",
}


def _insert_default_args(argv: list[str]) -> list[str]:
    updated = list(argv)
    present = {_arg_key(item) for item in updated[1:] if item.startswith("--")}
    for key, value in reversed(list(DEFAULT_ARGS.items())):
        if key not in present:
            updated[1:1] = [key, value]
    return updated


def _arg_key(item: str) -> str:
    return item.split("=", 1)[0]


if __name__ == "__main__":
    sys.argv = _insert_default_args(sys.argv)
    main()
