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


FROZEN_MANIFEST = "data/autorobobench/robocasa_language_following_manifest.json"
FROZEN_SPLIT = "data/autorobobench/robocasa_language_following_splits.json"
# Benchmark rule: scored training has a fixed 5 minute loop cap. Do not overwrite or raise this.
BENCHMARK_TRAIN_SECONDS_CAP = 300.0


def main() -> None:
    _default("--manifest", FROZEN_MANIFEST)
    _default("--split", FROZEN_SPLIT)
    _default("--out-dir", "runs/autorobobench/robocasa_language_following/smolvlm_flow")
    _default("--train-episodes-per-task", "16")
    _default("--val-episodes-per-task", "2")
    _default("--policy-kind", "frozen_smolvlm_flow")
    _default("--vlm-encoder-name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    _default("--chunk-horizon", "16")
    _default("--frame-stride", "2")
    _fixed_train_cap()
    _default("--batch-size", "64")
    _default("--width", "256")
    _default("--dropout", "0.03")
    _default("--lr", "2e-4")
    _default("--weight-decay", "1e-4")
    _default("--image-noise", "0.004")
    _default("--proprio-noise", "0.004")
    _default("--chunk-decay", "0.85")
    _default("--bc-aux-weight", "0.1")
    _default("--flow-steps", "8")
    _default("--flow-source", "noise")
    _default("--flow-eval-start", "bc")
    _default("--action-depth", "2")
    _default("--heads", "4")
    _default("--history-stride", "16")
    _default("--eval-commit-steps", "8")
    _default("--balanced-sampling", None)

    from tasks.robocasa_bc5.train import main as train_main

    train_main()


def _default(flag: str, value: str | None) -> None:
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


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
    main()
