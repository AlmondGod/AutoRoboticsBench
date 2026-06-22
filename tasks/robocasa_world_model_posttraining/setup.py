from __future__ import annotations

import importlib.util
import json
from pathlib import Path

DEFAULT_MANIFEST = Path("data/autorobobench/robocasa_stand_mixer_peak_manifest.json")
DEFAULT_SPLIT = Path("data/autorobobench/robocasa_stand_mixer_peak_splits.json")
DEFAULT_POLICY_CHECKPOINT = Path("runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt")


def main() -> None:
    checks = {
        "task": "robocasa_world_model_posttraining",
        "torch_available": importlib.util.find_spec("torch") is not None,
        "robocasa_bc5_inference_available": importlib.util.find_spec("tasks.robocasa_bc5.inference") is not None,
        "world_model_available": importlib.util.find_spec("tasks.robocasa_world_model.model") is not None,
        "visual_world_model_available": importlib.util.find_spec("tasks.robocasa_visual_world_model.model") is not None,
        "manifest_exists": DEFAULT_MANIFEST.exists(),
        "split_exists": DEFAULT_SPLIT.exists(),
        "default_policy_checkpoint_exists": DEFAULT_POLICY_CHECKPOINT.exists(),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    optional = {"default_policy_checkpoint_exists"}
    missing = [key for key, ok in checks.items() if key != "task" and key not in optional and not ok]
    if missing:
        raise SystemExit(f"missing requirements: {', '.join(missing)}")


if __name__ == "__main__":
    main()
