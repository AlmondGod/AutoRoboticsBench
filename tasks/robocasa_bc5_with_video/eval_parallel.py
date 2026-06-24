from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__import__("os").environ.get("ROBOAUTORESEARCH_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_bc5_with_video.eval import FROZEN_SPLIT, _rewrite_result


def main() -> None:
    _force_arg("--split", FROZEN_SPLIT)
    _default("--inference", "tasks.robocasa_bc5_with_video.inference")

    from tasks.robocasa_bc5.eval_parallel import main as robocasa_eval_parallel_main

    out_path = _arg_value("--out")
    robocasa_eval_parallel_main()
    if out_path:
        payload = _rewrite_result(Path(out_path))
        if payload is not None:
            print(json.dumps(payload, indent=2, sort_keys=True))


def _default(flag: str, value: str) -> None:
    if not any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv):
        sys.argv.extend([flag, value])


def _force_arg(flag: str, value: str) -> None:
    if not any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv):
        sys.argv.extend([flag, value])
        return
    for idx, arg in enumerate(sys.argv):
        if arg == flag:
            if idx + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires a value")
            actual = sys.argv[idx + 1]
            break
        if arg.startswith(f"{flag}="):
            actual = arg.split("=", 1)[1]
            break
    else:
        return
    if actual != value:
        raise ValueError(f"{flag} is immutable for this task; expected {value}")


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


if __name__ == "__main__":
    main()
