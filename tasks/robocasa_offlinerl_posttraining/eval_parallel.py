from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_offlinerl_posttraining.eval import FROZEN_MANIFEST, FROZEN_SPLIT


def main() -> None:
    out_path = _arg_value("--out")
    _default("--manifest", FROZEN_MANIFEST)
    _default("--split", FROZEN_SPLIT)
    _default("--max-steps", "750")
    _default("--commit-steps", "8")
    _default("--inference", "tasks.robocasa_offlinerl_posttraining.inference")

    from tasks.robocasa_bc5.eval_parallel import main as robocasa_eval_parallel_main

    robocasa_eval_parallel_main()
    if out_path:
        out = Path(out_path)
        if out.exists():
            payload = json.loads(out.read_text())
            payload["track"] = "robocasa_offlinerl_posttraining"
            payload["manifest"] = FROZEN_MANIFEST
            payload["split"] = FROZEN_SPLIT
            payload["target_task"] = "PickPlaceCounterToStandMixer"
            payload["offlinerl_final_success"] = float(payload.get("success_rate", 0.0))
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(json.dumps(payload, indent=2, sort_keys=True))


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


if __name__ == "__main__":
    main()
