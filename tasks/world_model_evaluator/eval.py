from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from eval.eval_robocasa_tiny_evaluator_correlation import (  # noqa: E402
    _load_archive,
    _load_evaluator,
    _score_trace_candidate,
    _trace_paths,
)
from eval.eval_world_model_ranking import _write_svg, compute_metrics  # noqa: E402
from train.common import device_from_arg  # noqa: E402


DEFAULT_ARCHIVE = "runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Immutable evaluator for the World Model Evaluator task.")
    parser.add_argument("--checkpoint", "--evaluator", dest="checkpoint", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--task-alias", default="OpenDrawer")
    parser.add_argument("--out", default="runs/autorobobench/world_model_evaluator/eval_result.json")
    parser.add_argument("--scores-out", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--imagined-rollouts", type=int, default=1)
    parser.add_argument("--imagined-steps", type=int, default=260)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--invert-learned-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sim-seconds-per-rollout", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_path = Path(args.out)
    score_path = Path(args.scores_out) if args.scores_out else out_path.with_name("candidate_scores.jsonl")
    plot_path = Path(args.plot) if args.plot else out_path.with_suffix(".svg")

    device = device_from_arg(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    evaluator = _load_evaluator(checkpoint, device)
    evaluator.load_state_dict(checkpoint["state_dict"])
    evaluator.eval()

    dataset_root = _dataset_root(Path(args.manifest), str(args.task_alias))
    rows = _load_archive(Path(args.archive))
    scored_rows: list[dict] = []
    for idx, row in enumerate(rows):
        trace_paths = _trace_paths(row)
        sim_success = float(row.get("success_rate", row.get("score", 0.0)))
        if trace_paths:
            raw_learned_score, learned_progress, seconds = _score_trace_candidate(
                evaluator=evaluator,
                evaluator_ckpt=checkpoint,
                dataset_root=dataset_root,
                trace_paths=trace_paths,
                imagined_rollouts=int(args.imagined_rollouts),
                imagined_steps=int(args.imagined_steps),
                action_noise=float(args.action_noise),
                device=device,
            )
            learned_score = 1.0 - raw_learned_score if args.invert_learned_score else raw_learned_score
            error = None
        else:
            raw_learned_score = None
            learned_score = None
            learned_progress = None
            seconds = None
            error = "no trace paths"
        scored_rows.append(
            {
                "candidate_id": int(row.get("experiment", idx)),
                "change": row.get("change"),
                "learned_score": learned_score,
                "raw_learned_score": raw_learned_score,
                "learned_score_transform": "1-raw" if args.invert_learned_score else "raw",
                "learned_progress": learned_progress,
                "sim_success": sim_success,
                "sim_successes": row.get("successes"),
                "sim_eval_rollouts": row.get("episodes"),
                "sim_eval_seconds": float(row.get("episodes") or 0.0) * float(args.sim_seconds_per_rollout),
                "learned_eval_rollouts": int(args.imagined_rollouts) * len(trace_paths),
                "learned_eval_seconds": seconds,
                "learned_mode": "action_trace",
                "error": error,
            }
        )

    metrics = compute_metrics(scored_rows, top_k=int(args.top_k))
    calibration = _calibration_score(scored_rows)
    speedup = metrics.get("speedup_ratio")
    result = {
        "track": "world_model_evaluator",
        "checkpoint": str(args.checkpoint),
        "archive": str(args.archive),
        "task_alias": str(args.task_alias),
        "candidate_count": int(metrics["n"]),
        "wm_spearman": float(metrics["spearman_rho"]),
        "pearson_r": float(metrics["pearson_r"]),
        "checkpoint_ranking_accuracy": float(metrics["top_k_hit_rate"]),
        "wm_speedup_score": _speedup_score(speedup),
        "speedup_ratio": speedup,
        "calibration_score": calibration,
        "reproducibility_integrity": 1.0,
        "ranking_metrics": metrics,
        "imagined_rollouts": int(args.imagined_rollouts),
        "imagined_steps": int(args.imagined_steps),
        "action_noise": float(args.action_noise),
        "learned_score_transform": "1-raw" if args.invert_learned_score else "raw",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    score_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in scored_rows) + "\n")
    _write_svg(scored_rows, metrics, plot_path)
    print(json.dumps(result, indent=2, sort_keys=True))


def _dataset_root(manifest_path: Path, alias: str) -> Path:
    manifest = json.loads(manifest_path.read_text())
    for task in manifest["tasks"]:
        if task["alias"] == alias:
            return Path(task["dataset_path"])
    raise ValueError(f"{alias!r} not found in {manifest_path}")


def _calibration_score(rows: list[dict]) -> float:
    usable = [
        row
        for row in rows
        if row.get("learned_score") is not None and row.get("sim_success") is not None
    ]
    if not usable:
        return 0.0
    errors = [abs(float(row["learned_score"]) - float(row["sim_success"])) for row in usable]
    return max(0.0, min(1.0, 1.0 - float(np.mean(errors))))


def _speedup_score(speedup: float | None) -> float:
    if speedup is None or not math.isfinite(float(speedup)) or speedup <= 0:
        return 0.0
    return max(0.0, min(1.0, float(speedup) / 10.0))


if __name__ == "__main__":
    main()
