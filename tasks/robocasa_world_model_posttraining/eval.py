from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = "data/autorobobench/robocasa_stand_mixer_peak_manifest.json"
DEFAULT_SPLIT = "data/autorobobench/robocasa_stand_mixer_peak_splits.json"


def main() -> None:
    _default("--manifest", DEFAULT_MANIFEST)
    _default("--split", DEFAULT_SPLIT)
    _default("--inference", "tasks.robocasa_world_model_posttraining.inference")
    _default("--max-steps", "750")
    _default("--commit-steps", "8")
    _default("--eval-episodes-per-task", "100")

    from tasks.robocasa_bc5.eval import main as robocasa_eval_main

    robocasa_eval_main()


def _default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    main()
