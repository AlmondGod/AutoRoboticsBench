from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = Path("data/autorobobench/robocasa_stand_mixer_peak_manifest.json")
DEFAULT_SPLIT = Path("data/autorobobench/robocasa_stand_mixer_peak_splits.json")
DEFAULT_POLICY_CHECKPOINT = Path("runs/autorobobench/robocasa_stand_mixer_base/nonzero_base/policy_best.pt")
DEFAULT_WORLD_MODEL_CHECKPOINT = Path(
    "data/autorobobench/pretrained_world_models/robocasa_visual_world_model_spatial_conv_11task_20min.pt"
)


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
        "default_world_model_checkpoint_exists": DEFAULT_WORLD_MODEL_CHECKPOINT.exists(),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    optional = {"default_policy_checkpoint_exists"}
    missing = [key for key, ok in checks.items() if key != "task" and key not in optional and not ok]
    if missing:
        raise SystemExit(f"missing requirements: {', '.join(missing)}")


if __name__ == "__main__":
    main()
