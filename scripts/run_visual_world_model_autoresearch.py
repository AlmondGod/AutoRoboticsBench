#!/usr/bin/env python3
"""Run a timed visual-world-model experiment sweep and record run summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK = "robocasa_visual_world_model"


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "baseline_32",
        "summary": "Baseline 32px visual world model with default loss weights.",
        "args": ["--image-size", "32", "--width", "512", "--depth", "4"],
    },
    {
        "name": "visual_heavy_32",
        "summary": "Increase visual reconstruction and flow weights at 32px.",
        "args": [
            "--image-size",
            "32",
            "--width",
            "512",
            "--depth",
            "4",
            "--visual-weight",
            "1.5",
            "--visual-flow-weight",
            "0.8",
            "--image-vae-weight",
            "0.35",
        ],
    },
    {
        "name": "state_light_32",
        "summary": "Reduce state/progress pressure to prioritize visual prediction.",
        "args": [
            "--image-size",
            "32",
            "--width",
            "512",
            "--depth",
            "4",
            "--state-weight",
            "0.5",
            "--progress-weight",
            "0.15",
            "--success-weight",
            "0.15",
            "--visual-weight",
            "1.5",
        ],
    },
    {
        "name": "larger_latent_32",
        "summary": "Use larger latent and visual latent dimensions.",
        "args": [
            "--image-size",
            "32",
            "--width",
            "640",
            "--depth",
            "4",
            "--latent-dim",
            "96",
            "--visual-latent-dim",
            "96",
        ],
    },
    {
        "name": "image_48",
        "summary": "Train at 48px to improve visual fidelity.",
        "args": [
            "--image-size",
            "48",
            "--width",
            "512",
            "--depth",
            "4",
            "--batch-size",
            "384",
        ],
    },
    {
        "name": "image_48_visual_heavy",
        "summary": "48px model with stronger visual reconstruction/flow weighting.",
        "args": [
            "--image-size",
            "48",
            "--width",
            "512",
            "--depth",
            "4",
            "--batch-size",
            "384",
            "--visual-weight",
            "1.5",
            "--visual-flow-weight",
            "0.8",
        ],
    },
    {
        "name": "deep_32",
        "summary": "Deeper 32px model with moderate dropout.",
        "args": [
            "--image-size",
            "32",
            "--width",
            "512",
            "--depth",
            "6",
            "--dropout",
            "0.08",
        ],
    },
    {
        "name": "low_kl_32",
        "summary": "Reduce KL pressure to favor deterministic visual reconstruction.",
        "args": [
            "--image-size",
            "32",
            "--width",
            "512",
            "--depth",
            "4",
            "--kl-weight",
            "0.00003",
            "--visual-kl-weight",
            "0.000003",
        ],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-budget-hours", type=float, default=9.0)
    parser.add_argument("--experiment-seconds", type=float, default=2700.0)
    parser.add_argument("--min-remaining-seconds", type=float, default=900.0)
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--scaffold", default="local_autoresearch")
    parser.add_argument("--base", default="visual_wm_sweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs-prefix", default="visual_wm_autoresearch")
    parser.add_argument("--task-run-root", default="runs/autorobobench/robocasa_visual_world_model")
    parser.add_argument("--analysis-dir", default="analysis/visual_world_model_autoresearch")
    parser.add_argument("--policy-checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    deadline = started + float(args.time_budget_hours) * 3600.0
    analysis_dir = repo / args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    exp_index = 0
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining < float(args.min_remaining_seconds):
            break
        spec = EXPERIMENTS[exp_index % len(EXPERIMENTS)]
        cycle = exp_index // len(EXPERIMENTS)
        seed = int(args.seed) + exp_index
        train_seconds = min(float(args.experiment_seconds), max(60.0, remaining - float(args.min_remaining_seconds)))
        run_id = f"{args.runs_prefix}_exp{exp_index + 1:03d}_{spec['name']}_seed{seed}"
        run_dir = repo / "runs" / run_id
        task_run_dir = repo / args.task_run_root / run_id
        if run_dir.exists() and (run_dir / "run_summary.json").exists():
            results.append(_load_json(run_dir / "run_summary.json"))
            exp_index += 1
            continue

        train_cmd = [
            sys.executable,
            "tasks/robocasa_visual_world_model/train.py",
            "--out-dir",
            str(task_run_dir),
            "--max-train-seconds",
            str(train_seconds),
            "--seed",
            str(seed),
            "--device",
            str(args.device),
            *[str(x) for x in spec["args"]],
        ]
        eval_cmd = [
            sys.executable,
            "tasks/robocasa_visual_world_model/eval.py",
            "--checkpoint",
            str(task_run_dir / "policy_best.pt"),
            "--out",
            str(task_run_dir / "eval_lpips.json"),
            "--device",
            str(args.device),
        ]
        if args.policy_checkpoint:
            eval_cmd.extend(["--policy-checkpoint", str(args.policy_checkpoint)])

        result = run_experiment(
            repo=repo,
            run_id=run_id,
            run_dir=run_dir,
            task_run_dir=task_run_dir,
            experiment_number=exp_index + 1,
            cycle=cycle,
            spec=spec,
            seed=seed,
            train_seconds=train_seconds,
            metadata={
                "agent": args.agent,
                "model": args.model,
                "scaffold": args.scaffold,
                "base": args.base,
            },
            train_cmd=train_cmd,
            eval_cmd=eval_cmd,
            dry_run=bool(args.dry_run),
        )
        results.append(result)
        write_plots(results, analysis_dir)
        if args.dry_run:
            break
        exp_index += 1

    write_results_csv(results, analysis_dir / "visual_world_model_autoresearch.csv")
    write_plots(results, analysis_dir)
    print(json.dumps({"experiments": len(results), "analysis_dir": str(analysis_dir)}, indent=2, sort_keys=True))
    return 0


def run_experiment(
    *,
    repo: Path,
    run_id: str,
    run_dir: Path,
    task_run_dir: Path,
    experiment_number: int,
    cycle: int,
    spec: dict[str, Any],
    seed: int,
    train_seconds: float,
    metadata: dict[str, str],
    train_cmd: list[str],
    eval_cmd: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    task_run_dir.mkdir(parents=True, exist_ok=True)
    started_epoch = time.time()
    (run_dir / "started_epoch.txt").write_text(f"{started_epoch}\n", encoding="utf-8")
    run_metadata = {
        "run_id": run_id,
        "agent": metadata["agent"],
        "model": metadata["model"],
        "scaffold": metadata["scaffold"],
        "task": TASK,
        "base": metadata["base"],
        "seed": int(seed),
        "experiment_number": int(experiment_number),
        "cycle": int(cycle),
        "change_summary": str(spec["summary"]),
        "task_run_dir": str(task_run_dir),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "experiment.json").write_text(
        json.dumps(
            {
                **run_metadata,
                "train_seconds": float(train_seconds),
                "train_command": train_cmd,
                "eval_command": eval_cmd,
                "dry_run": bool(dry_run),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    commands = []
    train_step = run_step(repo, "train", train_cmd, dry_run=dry_run)
    commands.append(train_step)
    eval_step = run_step(repo, "eval", eval_cmd, dry_run=dry_run)
    commands.append(eval_step)
    (run_dir / "commands.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in commands),
        encoding="utf-8",
    )

    eval_payload = _load_json(task_run_dir / "eval_lpips.json")
    (run_dir / "eval").mkdir(exist_ok=True)
    eval_results = {
        "score": _score(eval_payload),
        "success_rate": eval_payload.get("generated_visual_policy_score"),
        "num_episodes": eval_payload.get("generated_visual_policy_eval", {}).get("episodes"),
        "valid": all(int(step.get("returncode", 1)) == 0 for step in commands),
        "raw_eval": eval_payload,
        "change_summary": str(spec["summary"]),
    }
    (run_dir / "eval" / "results.json").write_text(json.dumps(eval_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    finished_epoch = time.time()
    wall_seconds = finished_epoch - started_epoch
    container_seconds = sum(float(row.get("duration_seconds") or 0.0) for row in commands)
    fatal = any(int(step.get("returncode", 0)) != 0 for step in commands)
    summary = {
        "run_id": run_id,
        "agent": metadata["agent"],
        "model": metadata["model"],
        "scaffold": metadata["scaffold"],
        "task": TASK,
        "base": metadata["base"],
        "seed": int(seed),
        "status": "failed" if fatal else "complete",
        "valid": not fatal,
        "wall_seconds": wall_seconds,
        "container_seconds": container_seconds,
        "gpu_seconds": container_seconds,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "estimated_usd": None,
        "score": eval_results["score"],
        "success_rate": eval_results["success_rate"],
        "baseline_score": None,
        "improvement": None,
        "normalized_score": eval_results["score"],
        "num_eval_episodes": eval_results["num_episodes"],
        "num_flags": 0 if not fatal else 1,
        "fatal": fatal,
        "experiment_number": int(experiment_number),
        "change_summary": str(spec["summary"]),
        "task_run_dir": str(task_run_dir),
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "timing.jsonl").write_text(
        json.dumps(
            {
                "experiment_number": int(experiment_number),
                "wall_seconds": wall_seconds,
                "container_seconds": container_seconds,
                "score": summary["score"],
                "normalized_score": summary["normalized_score"],
                "success_rate": summary["success_rate"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def run_step(repo: Path, name: str, cmd: list[str], *, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    if dry_run:
        return {
            "name": name,
            "command": cmd,
            "returncode": 0,
            "duration_seconds": 0.0,
            "stdout": "dry run",
            "stderr": "",
        }
    completed = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
    return {
        "name": name,
        "command": cmd,
        "returncode": int(completed.returncode),
        "duration_seconds": time.time() - started,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _score(eval_payload: dict[str, Any]) -> float | None:
    for key in ("visual_world_model_score", "score", "normalized_score"):
        value = eval_payload.get(key)
        if value is not None:
            return float(value)
    return None


def write_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment_number",
        "run_id",
        "change_summary",
        "score",
        "normalized_score",
        "success_rate",
        "wall_seconds",
        "total_tokens",
    ]
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def write_plots(rows: list[dict[str, Any]], analysis_dir: Path) -> None:
    scored = [row for row in rows if row.get("score") is not None and row.get("experiment_number") is not None]
    if not scored:
        return
    import matplotlib.pyplot as plt

    scored = sorted(scored, key=lambda row: int(row["experiment_number"]))
    xs = [int(row["experiment_number"]) for row in scored]
    ys = [float(row["score"]) for row in scored]
    labels = [str(row.get("run_id", "")) for row in scored]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Eval score")
    ax.grid(True, alpha=0.25)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label.split("_exp")[-1].split("_seed")[0], (x, y), textcoords="offset points", xytext=(4, 5), fontsize=7)
    fig.tight_layout()
    fig.savefig(analysis_dir / "score_by_experiment.png", dpi=160)
    plt.close(fig)

    token_rows = [row for row in scored if row.get("total_tokens") not in (None, "")]
    if token_rows:
        tx = [float(row["total_tokens"]) for row in token_rows]
        ty = [float(row["score"]) for row in token_rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(tx, ty)
        ax.set_xlabel("Total tokens")
        ax.set_ylabel("Eval score")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(analysis_dir / "score_by_tokens.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
