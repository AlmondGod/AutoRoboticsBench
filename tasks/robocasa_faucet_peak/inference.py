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

from tasks.robocasa_bc5 import inference as _bc5_inference  # noqa: E402


FORBIDDEN_POLICY_TYPES = {"robocasa_bc5_trajectory_bank"}
FORBIDDEN_REPLAY_KEYS = {"actions", "lengths", "episode_ids", "embeddings"}


def load_policy(checkpoint: str, device: str = "auto"):
    """Load a learned Faucet Peak policy.

    Test-time trajectory replay is banned for this track. The checkpoint and
    inference code may use learned weights/statistics only; they may not carry
    or read demonstration action trajectories, trajectory banks, manifest/split
    files, datasets, or video pools during eval.
    """
    policy = _bc5_inference.load_policy(checkpoint, device)
    policy_type = str(policy.checkpoint.get("policy_type", ""))
    replay_keys = FORBIDDEN_REPLAY_KEYS.intersection(policy.checkpoint)
    if policy_type in FORBIDDEN_POLICY_TYPES or getattr(policy, "mode", "") == "trajectory_bank" or replay_keys:
        details = f" policy_type={policy_type!r}"
        if replay_keys:
            details += f" replay_keys={sorted(replay_keys)}"
        raise ValueError(
            "robocasa_faucet_peak forbids test-time trajectory replay and trajectory-bank checkpoints;"
            + details
        )
    return policy


act = _bc5_inference.act


__all__ = ["act", "load_policy"]
