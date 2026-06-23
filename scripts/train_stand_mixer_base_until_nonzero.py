#!/usr/bin/env python3
"""Train a PickPlaceCounterToStandMixer base policy until eval is nonzero."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "data/autorobobench/robocasa_stand_mixer_peak_manifest.json"
DEFAULT_SPLIT = "data/autorobobench/robocasa_stand_mixer_peak_splits.json"
DEFAULT_OUT_ROOT = "runs/autorobobench/robocasa_stand_mixer_base"
DEFAULT_PROMOTED = "runs/autorobobench/robocasa_stand_mixer_base/nonzero_base"
BENCHMARK_TRAIN_SECONDS_CAP = 300.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--promoted-dir", default=DEFAULT_PROMOTED)
    parser.add_argument("--attempt-seconds", type=float, default=BENCHMARK_TRAIN_SECONDS_CAP)
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=BENCHMARK_TRAIN_SECONDS_CAP,
        help="Total helper budget; capped at 300 seconds for benchmark consistency.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means unlimited attempts.")
    parser.add_argument("--policy-kind", choices=["bc", "flow", "sequence_flow"], default="bc")
    parser.add_argument("--chunk-horizon", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=3)
    parser.add_argument("--action-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image-noise", type=float, default=0.004)
    parser.add_argument("--proprio-noise", type=float, default=0.004)
    parser.add_argument("--chunk-decay", type=float, default=0.82)
    parser.add_argument("--action-smooth", type=float, default=0.0005)
    parser.add_argument("--progress-scale", type=float, default=750.0)
    parser.add_argument("--task-action-normalization", action="store_true")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--eval-episodes-per-task", type=int, default=100)
    parser.add_argument("--eval-workers", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--commit-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if float(args.attempt_seconds) <= 0 or float(args.attempt_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--attempt-seconds is fixed at 300 for benchmark runs and cannot be overwritten")
    if float(args.max_total_seconds) <= 0 or float(args.max_total_seconds) > BENCHMARK_TRAIN_SECONDS_CAP:
        raise ValueError("--max-total-seconds is fixed at 300 for benchmark runs and cannot be overwritten")

    started = time.monotonic()
    attempts = 0
    out_root = ROOT / args.out_root
    promoted = ROOT / args.promoted_dir
    out_root.mkdir(parents=True, exist_ok=True)

    while True:
        if args.max_attempts and attempts >= int(args.max_attempts):
            return 1
        if args.max_total_seconds > 0 and time.monotonic() - started >= float(args.max_total_seconds):
            return 1
        attempts += 1
        seed = int(args.seed) + attempts - 1
        run_dir = out_root / f"{args.policy_kind}_seed{seed}_{int(args.attempt_seconds)}s_attempt{attempts}"
        train_cmd = [
            sys.executable,
            "tasks/robocasa_bc5/train.py",
            "--manifest",
            str(args.manifest),
            "--split",
            str(args.split),
            "--out-dir",
            str(run_dir),
            "--train-episodes-per-task",
            "80",
            "--val-episodes-per-task",
            "10",
            "--task-alias",
            "PickPlaceCounterToStandMixer",
            "--policy-kind",
            str(args.policy_kind),
            "--chunk-horizon",
            str(args.chunk_horizon),
            "--frame-stride",
            str(args.frame_stride),
            "--max-train-seconds",
            str(args.attempt_seconds),
            "--batch-size",
            str(args.batch_size),
            "--width",
            str(args.width),
            "--transformer-depth",
            str(args.transformer_depth),
            "--action-depth",
            str(args.action_depth),
            "--heads",
            str(args.heads),
            "--dropout",
            str(args.dropout),
            "--lr",
            str(args.lr),
            "--image-noise",
            str(args.image_noise),
            "--proprio-noise",
            str(args.proprio_noise),
            "--chunk-decay",
            str(args.chunk_decay),
            "--action-smooth",
            str(args.action_smooth),
            "--progress-conditioning",
            "--progress-scale",
            str(args.progress_scale),
            "--eval-commit-steps",
            str(args.commit_steps),
            "--seed",
            str(seed),
            "--device",
            str(args.device),
        ]
        if args.task_action_normalization:
            train_cmd.append("--task-action-normalization")
        if args.balanced_sampling:
            train_cmd.append("--balanced-sampling")
        if str(args.policy_kind) in {"flow", "sequence_flow"}:
            train_cmd.extend(["--flow-eval-start", "bc", "--flow-source", "bc"])
        eval_out = run_dir / f"eval_{int(args.eval_episodes_per_task)}.json"
        eval_cmd = [
            sys.executable,
            "tasks/robocasa_bc5/eval_parallel.py",
            "--manifest",
            str(args.manifest),
            "--split",
            str(args.split),
            "--inference",
            "tasks.robocasa_bc5.inference",
            "--checkpoint",
            str(run_dir / "policy_best.pt"),
            "--out",
            str(eval_out),
            "--eval-episodes-per-task",
            str(args.eval_episodes_per_task),
            "--task-alias",
            "PickPlaceCounterToStandMixer",
            "--max-steps",
            str(args.max_steps),
            "--commit-steps",
            str(args.commit_steps),
            "--workers",
            str(args.eval_workers),
            "--device",
            str(args.device),
        ]
        _run(train_cmd, dry_run=bool(args.dry_run))
        _run(eval_cmd, dry_run=bool(args.dry_run))
        if args.dry_run:
            return 0
        result = json.loads(eval_out.read_text())
        successes = int(result.get("successes", 0))
        success_rate = float(result.get("success_rate", 0.0))
        attempt_summary = {
            "attempt": attempts,
            "run_dir": str(run_dir),
            "policy": str(run_dir / "policy_best.pt"),
            "eval": str(eval_out),
            "successes": successes,
            "success_rate": success_rate,
            "episodes": int(result.get("episodes", 0)),
            "seed": seed,
        }
        (run_dir / "base_attempt_summary.json").write_text(json.dumps(attempt_summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(attempt_summary, sort_keys=True), flush=True)
        if successes > 0 or success_rate > 0:
            _promote(run_dir, promoted, attempt_summary)
            return 0


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def _promote(run_dir: Path, promoted: Path, summary: dict) -> None:
    promoted.mkdir(parents=True, exist_ok=True)
    for name in ("policy_best.pt", "policy.pt", "train_summary.json", "base_attempt_summary.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, promoted / name)
    eval_src = Path(str(summary["eval"]))
    if eval_src.exists():
        shutil.copy2(eval_src, promoted / eval_src.name)
    (promoted / "selected_base.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"promoted_dir": str(promoted), **summary}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
