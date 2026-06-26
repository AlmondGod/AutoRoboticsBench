from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = "data/autorobobench/robocasa_stand_mixer_peak_manifest.json"
DEFAULT_SPLIT = "data/autorobobench/robocasa_stand_mixer_peak_splits.json"


def main() -> None:
    out_path = _arg_value("--out")
    _default("--manifest", DEFAULT_MANIFEST)
    _default("--split", DEFAULT_SPLIT)
    _default("--inference", "tasks.robocasa_world_model_posttraining.inference")
    _default("--max-steps", "750")
    _default("--commit-steps", "8")
    _default("--eval-episodes-per-task", "100")

    from tasks.robocasa_bc5.eval import main as robocasa_eval_main

    robocasa_eval_main()
    if out_path:
        _rewrite_result(Path(out_path))


def _default(flag: str, value: str) -> None:
    if not any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv):
        sys.argv.extend([flag, value])


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _rewrite_result(out: Path) -> dict | None:
    if not out.exists():
        return None
    payload = json.loads(out.read_text())
    payload["track"] = "robocasa_world_model_posttraining"
    payload["manifest"] = DEFAULT_MANIFEST
    payload["split"] = DEFAULT_SPLIT
    payload["target_task"] = "PickPlaceCounterToStandMixer"
    payload["metric"] = "success_rate"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    main()
