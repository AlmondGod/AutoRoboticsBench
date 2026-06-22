from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

FROZEN_MANIFEST = "data/autorobobench/robocasa_bc1_manifest.json"
FROZEN_SPLIT = "data/autorobobench/robocasa_bc1_splits.json"
SPEED_TIEBREAKER_SUCCESSES = 0.25


def main() -> None:
    _force_arg("--manifest", FROZEN_MANIFEST)
    _force_arg("--split", FROZEN_SPLIT)
    _default("--inference", "tasks.robocasa_bc1.inference")
    _default("--max-steps", "750")
    _default("--commit-steps", "8")

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
    split = json.loads((ROOT / FROZEN_SPLIT).read_text())
    split_task = next(task for task in split["tasks"] if task["alias"] == "TurnOnSinkFaucet")
    visual_eval_protocol = split.get("visual_eval_protocol")
    anti_replay_eval_protocol = split.get("anti_replay_eval_protocol")
    same_sink_protocol = split.get("same_sink_protocol")
    success_rate = float(payload.get("success_rate", 0.0))
    speed_metrics = _successful_speed_metrics(payload)
    payload["track"] = "robocasa_bc1"
    payload["manifest"] = FROZEN_MANIFEST
    payload["split"] = FROZEN_SPLIT
    if visual_eval_protocol is not None:
        payload["visual_eval_protocol"] = visual_eval_protocol
    if anti_replay_eval_protocol is not None:
        payload["anti_replay_eval_protocol"] = anti_replay_eval_protocol
    if same_sink_protocol is not None:
        payload["same_sink_protocol"] = same_sink_protocol
    payload["peak_final_success"] = success_rate
    payload.update(speed_metrics)
    payload["reliability_stability"] = success_rate
    payload["data_budget_integrity"] = 1.0
    payload["reproducibility_integrity"] = 1.0
    payload["data_contract"] = {
        "target_task": "TurnOnSinkFaucet",
        "task_specific_action_demos": len(split_task["train_episode_ids"]),
        "all_target_trajectory_data_available": True,
        "training_time_eval_trajectory_access": True,
        "generic_video_pool_tasks": 9,
        "generic_video_pool_contains_target_task": False,
        "video_pool_contains_actions": False,
        "video_pool_contains_proprio": False,
        "test_time_demo_access": False,
        "test_time_trajectory_replay": False,
        "inference_time_dataset_access": False,
        "inference_time_manifest_split_access": False,
        "inference_time_video_pool_access": False,
    }
    if visual_eval_protocol is not None:
        payload["data_contract"]["visual_eval_protocol"] = visual_eval_protocol["name"]
    if anti_replay_eval_protocol is not None:
        payload["data_contract"]["anti_replay_eval_protocol"] = anti_replay_eval_protocol["name"]
    if same_sink_protocol is not None:
        payload["data_contract"]["same_sink_protocol"] = same_sink_protocol["name"]
        payload["data_contract"]["sink_fixture_key"] = same_sink_protocol["sink_fixture_key"]
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _successful_speed_metrics(payload: dict) -> dict:
    details = payload.get("details") if isinstance(payload.get("details"), list) else []
    episodes = int(payload.get("episodes") or len(details) or 0)
    successes = int(payload.get("successes") or sum(1 for row in details if row.get("success")))
    max_steps = max(1, int(payload.get("max_steps") or 750))
    successful_steps = [int(row.get("steps", max_steps)) for row in details if row.get("success")]
    speed_values = [
        max(0.0, min(1.0, 1.0 - float(steps) / float(max_steps)))
        for steps in successful_steps
    ]
    successful_eval_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0
    score = 0.0
    if episodes > 0:
        score = (float(successes) + SPEED_TIEBREAKER_SUCCESSES * successful_eval_speed) / (
            float(episodes) + SPEED_TIEBREAKER_SUCCESSES
        )
    return {
        "bc1_reliability_speed_score": max(0.0, min(1.0, score)),
        "successful_eval_speed": successful_eval_speed,
        "successful_eval_steps_mean": (
            sum(successful_steps) / len(successful_steps) if successful_steps else None
        ),
        "successful_eval_steps_min": min(successful_steps) if successful_steps else None,
        "successful_eval_steps_max": max(successful_steps) if successful_steps else None,
        "speed_tiebreaker_successes": SPEED_TIEBREAKER_SUCCESSES,
    }


if __name__ == "__main__":
    main()
