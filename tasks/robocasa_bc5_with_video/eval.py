from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

FROZEN_SPLIT = "data/autorobobench/robocasa_bc5_with_video_splits.json"


def main() -> None:
    _force_arg("--split", FROZEN_SPLIT)
    _default("--inference", "tasks.robocasa_bc5_with_video.inference")

    from tasks.robocasa_bc5.eval import main as robocasa_eval_main

    out_path = _arg_value("--out")
    robocasa_eval_main()
    if out_path:
        payload = _rewrite_result(Path(out_path))
        if payload is not None:
            print(json.dumps(payload, indent=2, sort_keys=True))


def _default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def _force_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])
        return
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    if sys.argv[idx + 1] != value:
        raise ValueError(f"{flag} is immutable for this task; expected {value}")


def _arg_value(flag: str) -> str | None:
    if flag not in sys.argv:
        return None
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


def _rewrite_result(out: Path) -> dict | None:
    if not out.exists():
        return None
    payload = json.loads(out.read_text())
    success_rate = float(payload.get("success_rate", 0.0))
    payload["track"] = "robocasa_bc5_with_video"
    payload["split"] = FROZEN_SPLIT
    payload["video_transfer_success"] = success_rate
    payload["paired_action_efficiency"] = success_rate
    payload["data_budget_integrity"] = 1.0
    payload["reproducibility_integrity"] = 1.0
    payload["data_contract"] = {
        "paired_action_demos_per_task": 5,
        "in_task_video_only_demos_per_task": 75,
        "total_video_only_demos": 909,
        "video_to_paired_demo_ratio": 36.36,
        "video_to_paired_frame_ratio": 38.781372702999846,
        "video_pool_contains_actions": False,
        "video_pool_contains_proprio": False,
        "test_time_demo_access": False
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    main()
