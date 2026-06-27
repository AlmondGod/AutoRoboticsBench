#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK = "robocasa_visual_world_model"
DEFAULT_OUT_ROOT = ROOT / "runs" / "autorobobench" / TASK / "a100_autoresearch"
DEFAULT_ANALYSIS_DIR = ROOT / "analysis" / "visual_world_model_autoresearch_a100"
TRAIN = ROOT / "tasks" / TASK / "train.py"
EVAL = ROOT / "tasks" / TASK / "eval.py"
POLICY_SET = ROOT / "data" / "autorobobench" / "robocasa_world_model_policy_set.json"


# Literature-derived search axes, adapted to this compact RoboCasa benchmark model:
# - Sora/Seedance: patch/latent video modeling, larger latent capacity, efficient transformer-like scaling.
# - NVIDIA Cosmos: action/past-frame-conditioned world prediction with autoregressive rollouts.
# - Genie: spatial tokenization, latent dynamics, and controllable transition prediction.
# The implementation below only uses benchmark-legal flags; every scored train call remains capped at 300 seconds.
SEED_SPECS: list[dict[str, Any]] = [
    {
        "name": "baseline_spatial_conv_64",
        "idea": "current compact spatial latent AR baseline",
        "args": {},
    },
    {
        "name": "cosmos_rollout4",
        "idea": "longer action-conditioned autoregressive rollout consistency",
        "args": {"--rollout-horizon": 4, "--rollout-visual-weight": 0.15, "--rollout-batch-size": 24},
    },
    {
        "name": "cosmos_rollout6_low_weight",
        "idea": "longer rollout with lower visual weight to reduce blur accumulation",
        "args": {"--rollout-horizon": 6, "--rollout-visual-weight": 0.08, "--rollout-state-weight": 0.15, "--rollout-batch-size": 16},
    },
    {
        "name": "genie_more_spatial_tokens",
        "idea": "larger spatial token map and dynamics capacity",
        "args": {"--spatial-latent-channels": 32, "--spatial-dynamics-hidden-channels": 64, "--spatial-dynamics-depth": 3},
    },
    {
        "name": "genie_finer_tokens",
        "idea": "higher-resolution spatial latent grid",
        "args": {"--spatial-downsample-blocks": 3, "--spatial-latent-channels": 16, "--spatial-width": 96, "--spatial-dynamics-hidden-channels": 64},
    },
    {
        "name": "sora_wide_latent_map",
        "idea": "wider latent/decoder capacity for spatial-patch reconstruction",
        "args": {"--spatial-latent-channels": 40, "--spatial-width": 96, "--spatial-depth": 2, "--spatial-dynamics-hidden-channels": 80},
    },
    {
        "name": "seedance_efficient_wide_batch",
        "idea": "A100-efficient larger batch for steadier 5-minute optimization",
        "args": {"--batch-size": 1024, "--rollout-batch-size": 32},
    },
    {
        "name": "seedance_large_batch_low_lr",
        "idea": "large batch with lower LR for stable high-throughput training",
        "args": {"--batch-size": 1536, "--lr": 2e-4, "--weight-decay": 5e-5},
    },
    {
        "name": "ar_state_success_heavy",
        "idea": "align generated policy success with progress/success heads",
        "args": {"--success-weight": 2.0, "--progress-weight": 0.3, "--state-weight": 1.25},
    },
    {
        "name": "visual_quality_heavy",
        "idea": "improve generated visual fidelity before policy conditioning",
        "args": {"--visual-weight": 1.75, "--image-vae-weight": 0.55, "--visual-l1-weight": 0.4, "--visual-grad-weight": 0.15},
    },
    {
        "name": "latent_prediction_heavy",
        "idea": "make visual-token transition prediction sharper",
        "args": {"--visual-latent-weight": 0.9, "--visual-weight": 1.1, "--image-vae-weight": 0.35},
    },
    {
        "name": "less_aug_clean_rollout",
        "idea": "reduce augmentation noise for correlation-sensitive rollouts",
        "args": {"--image-augment": 0.05, "--rollout-horizon": 4, "--rollout-visual-weight": 0.12},
    },
    {
        "name": "more_aug_robust_policy_views",
        "idea": "augment RGB for policy-conditioned robustness",
        "args": {"--image-augment": 0.25, "--visual-l1-weight": 0.35, "--image-vae-grad-weight": 0.35},
    },
    {
        "name": "mlp_spatial_dynamics",
        "idea": "compare token-flattened dynamics against conv-local dynamics",
        "args": {"--spatial-dynamics-type": "mlp", "--spatial-dynamics-depth": 4, "--spatial-latent-channels": 20},
    },
    {
        "name": "vector_vae_latent",
        "idea": "Sora-like compressed latent bottleneck instead of spatial map",
        "args": {"--visual-architecture": "vae", "--visual-latent-dim": 256, "--visual-decoder-width": 512, "--visual-decoder-depth": 3},
    },
    {
        "name": "vector_vae_wide",
        "idea": "larger vector latent decoder capacity",
        "args": {"--visual-architecture": "vae", "--visual-latent-dim": 512, "--visual-decoder-width": 768, "--visual-decoder-depth": 3},
    },
    {
        "name": "state_mlp_wider",
        "idea": "wider action/state trunk for more faithful reward/progress dynamics",
        "args": {"--width": 768, "--depth": 5, "--latent-dim": 96, "--batch-size": 768},
    },
    {
        "name": "state_mlp_deeper",
        "idea": "deeper action/state trunk with moderate dropout",
        "args": {"--width": 512, "--depth": 6, "--dropout": 0.08},
    },
    {
        "name": "low_dropout_sharp",
        "idea": "less dropout for sharper deterministic short-horizon prediction",
        "args": {"--dropout": 0.02, "--image-augment": 0.08},
    },
    {
        "name": "no_delta_absolute_visual",
        "idea": "predict next visual latent directly instead of latent delta",
        "args": {"--no-visual-delta-prediction": True, "--visual-latent-weight": 0.75},
    },
    {
        "name": "faster_visual_head",
        "idea": "higher visual LR to fit the RGB/token path inside 5 minutes",
        "args": {"--visual-lr-scale": 1.5, "--image-vae-weight": 0.45, "--visual-weight": 1.4},
    },
    {
        "name": "slower_visual_head",
        "idea": "lower visual LR to avoid rollout drift from overfitted RGB path",
        "args": {"--visual-lr-scale": 0.6, "--image-vae-weight": 0.35},
    },
    {
        "name": "high_recon_grad",
        "idea": "edge/gradient regularization for visual coherence",
        "args": {"--image-vae-grad-weight": 0.5, "--visual-grad-weight": 0.25, "--visual-l1-weight": 0.4},
    },
    {
        "name": "rollout_progress_focus",
        "idea": "make multi-step rollout carry task progress for policy ranking",
        "args": {"--rollout-horizon": 5, "--rollout-progress-weight": 0.15, "--rollout-state-weight": 0.2, "--rollout-visual-weight": 0.08},
    },
]


SEARCH_KEYS = [
    ("--batch-size", [512, 768, 1024, 1536]),
    ("--lr", [1.5e-4, 2e-4, 3e-4, 5e-4]),
    ("--weight-decay", [0.0, 5e-5, 1e-4, 3e-4]),
    ("--dropout", [0.0, 0.02, 0.05, 0.08, 0.12]),
    ("--spatial-latent-channels", [16, 20, 24, 32, 40, 48]),
    ("--spatial-width", [48, 64, 96, 128]),
    ("--spatial-depth", [1, 2, 3]),
    ("--spatial-downsample-blocks", [3, 4]),
    ("--spatial-dynamics-depth", [2, 3, 4]),
    ("--spatial-dynamics-hidden-channels", [32, 48, 64, 80, 96]),
    ("--rollout-horizon", [1, 2, 3, 4, 5, 6]),
    ("--rollout-visual-weight", [0.0, 0.05, 0.08, 0.1, 0.15, 0.2]),
    ("--rollout-state-weight", [0.05, 0.1, 0.15, 0.2]),
    ("--rollout-progress-weight", [0.02, 0.05, 0.1, 0.15]),
    ("--visual-weight", [0.9, 1.1, 1.25, 1.5, 1.75, 2.0]),
    ("--image-vae-weight", [0.25, 0.35, 0.4, 0.5, 0.65]),
    ("--visual-latent-weight", [0.25, 0.5, 0.75, 1.0]),
    ("--success-weight", [0.75, 1.0, 1.5, 2.0]),
    ("--progress-weight", [0.05, 0.15, 0.25, 0.35]),
    ("--image-augment", [0.0, 0.05, 0.1, 0.15, 0.25]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-by-one autoresearch experiments for RoboCasa visual world model.")
    parser.add_argument("--experiments", type=int, default=100)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--train-episodes-per-task", type=int, default=0)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--eval-val-episodes-per-task", type=int, default=5)
    parser.add_argument("--policy-eval-episodes-per-task", type=int, default=1)
    parser.add_argument("--policy-rollout-steps", type=int, default=64)
    parser.add_argument("--policy-commit-steps", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--commit-improvements", action="store_true")
    parser.add_argument("--push-improvements", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args()

    if args.max_train_seconds > 300.0:
        raise ValueError("visual world-model experiments must keep --max-train-seconds <= 300")
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    experiments_jsonl = args.analysis_dir / "experiments.jsonl"
    experiments_csv = args.analysis_dir / "experiments.csv"
    best_config_path = args.analysis_dir / "best_config.json"
    best_eval_path = args.analysis_dir / "best_eval.json"
    plot_path = args.analysis_dir / "score_progress.png"
    log_dir = args.analysis_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(experiments_jsonl)
    best = _best_record(existing)
    completed_indices = {int(row["experiment"]) for row in existing if "experiment" in row}
    rows = list(existing)

    for exp_idx in range(int(args.start_index), int(args.experiments) + 1):
        if exp_idx in completed_indices:
            continue
        spec = _experiment_spec(exp_idx, best, rng)
        exp_name = f"exp{exp_idx:03d}_{_slug(spec['name'])}"
        out_dir = args.out_root / exp_name
        eval_path = out_dir / "eval.json"
        train_log = log_dir / f"{exp_name}_train.log"
        eval_log = log_dir / f"{exp_name}_eval.log"
        command_args = _merge_args(_base_train_args(args, out_dir, exp_idx), spec["args"])
        start = time.time()
        status = "ok"
        fatal = ""
        train_rc = eval_rc = None
        train_metrics: dict[str, Any] = {}
        eval_metrics: dict[str, Any] = {}

        print(f"\n=== experiment {exp_idx}/{args.experiments}: {spec['name']} ===", flush=True)
        print(spec["idea"], flush=True)
        try:
            train_rc = _run([sys.executable, str(TRAIN), *command_args], train_log)
            if train_rc != 0:
                raise RuntimeError(f"train exited {train_rc}")
            train_metrics = _read_json(out_dir / "train_metrics.json")
            eval_cmd = [
                sys.executable,
                str(EVAL),
                "--checkpoint",
                str(out_dir / "policy_best.pt"),
                "--out",
                str(eval_path),
                "--val-episodes-per-task",
                str(args.eval_val_episodes_per_task),
                "--batch-size",
                str(args.eval_batch_size),
                "--policy-set",
                str(POLICY_SET),
                "--policy-eval-episodes-per-task",
                str(args.policy_eval_episodes_per_task),
                "--policy-rollout-steps",
                str(args.policy_rollout_steps),
                "--policy-commit-steps",
                str(args.policy_commit_steps),
                "--device",
                str(args.device),
            ]
            eval_rc = _run(eval_cmd, eval_log)
            if eval_rc != 0:
                raise RuntimeError(f"eval exited {eval_rc}")
            eval_metrics = _read_json(eval_path)
        except Exception as exc:  # noqa: BLE001 - records failures without stopping the run.
            status = "failed"
            fatal = f"{type(exc).__name__}: {exc}"
            print(f"experiment {exp_idx} failed: {fatal}", flush=True)

        elapsed = time.time() - start
        record = _record(
            exp_idx=exp_idx,
            status=status,
            fatal=fatal,
            spec=spec,
            out_dir=out_dir,
            train_log=train_log,
            eval_log=eval_log,
            elapsed=elapsed,
            train_rc=train_rc,
            eval_rc=eval_rc,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
        )
        keep = _is_better(record, best)
        record["kept"] = bool(keep)
        if keep:
            best = record
            _write_json(best_config_path, {"best": best, "config_args": command_args})
            if eval_metrics:
                _write_json(best_eval_path, eval_metrics)
            best_ckpt = args.analysis_dir / "best_policy_best.pt"
            src_ckpt = out_dir / "policy_best.pt"
            if src_ckpt.exists():
                shutil.copy2(src_ckpt, best_ckpt)
            print(
                "kept improvement: "
                f"corr={record.get('eval_correlation_score')} "
                f"score={record.get('visual_world_model_score')}",
                flush=True,
            )
        rows.append(record)
        _append_jsonl(experiments_jsonl, record)
        _write_csv(experiments_csv, rows)
        _plot(rows, plot_path)
        if keep and args.commit_improvements:
            _commit_improvement(args.analysis_dir, exp_idx, record, push=bool(args.push_improvements))

    return 0


def _base_train_args(args: argparse.Namespace, out_dir: Path, exp_idx: int) -> dict[str, Any]:
    return {
        "--out-dir": str(out_dir),
        "--max-train-seconds": float(args.max_train_seconds),
        "--train-episodes-per-task": int(args.train_episodes_per_task),
        "--val-episodes-per-task": int(args.val_episodes_per_task),
        "--device": str(args.device),
        "--seed": int(args.seed) + int(exp_idx),
    }


def _experiment_spec(exp_idx: int, best: dict[str, Any] | None, rng: random.Random) -> dict[str, Any]:
    if exp_idx <= len(SEED_SPECS):
        return dict(SEED_SPECS[exp_idx - 1])
    base_args: dict[str, Any] = {}
    if best and isinstance(best.get("args"), dict):
        base_args.update(best["args"])
    mutations = rng.randint(2, 5)
    for key, values in rng.sample(SEARCH_KEYS, k=mutations):
        base_args[key] = rng.choice(values)
    if int(base_args.get("--rollout-horizon", 2)) <= 1:
        base_args["--rollout-visual-weight"] = 0.0
    if base_args.get("--spatial-downsample-blocks") == 3 and base_args.get("--spatial-latent-channels", 24) > 32:
        base_args["--spatial-latent-channels"] = 32
    return {
        "name": f"adaptive_from_best_{exp_idx:03d}",
        "idea": "one-by-one adaptive mutation of the current best correlation/score configuration",
        "args": base_args,
    }


def _merge_args(base: dict[str, Any], override: dict[str, Any]) -> list[str]:
    merged = dict(base)
    merged.update(override)
    out: list[str] = []
    for key, value in merged.items():
        if isinstance(value, bool):
            if value:
                out.append(str(key))
            continue
        out.extend([str(key), str(value)])
    return out


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ROBOAUTORESEARCH_REPO_ROOT"] = str(ROOT)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return int(proc.wait())


def _record(
    *,
    exp_idx: int,
    status: str,
    fatal: str,
    spec: dict[str, Any],
    out_dir: Path,
    train_log: Path,
    eval_log: Path,
    elapsed: float,
    train_rc: int | None,
    eval_rc: int | None,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
) -> dict[str, Any]:
    benchmark = {k: eval_metrics.get(k) for k in [
        "visual_world_model_score",
        "eval_correlation_score",
        "policy_score_pearson",
        "policy_score_spearman",
        "ood_policy_score_pearson",
        "ood_policy_score_spearman",
        "visual_reconstruction_score",
        "visual_perceptual_score",
        "next_state_score",
        "progress_score",
        "success_score",
    ]}
    transition = eval_metrics.get("visual_transition_metrics") or {}
    policy_corr = eval_metrics.get("visual_policy_correlation") or {}
    final_val = train_metrics.get("final_val") or {}
    return {
        "experiment": int(exp_idx),
        "name": spec.get("name"),
        "idea": spec.get("idea"),
        "status": status,
        "fatal": fatal,
        "args": spec.get("args", {}),
        "out_dir": str(out_dir),
        "train_log": str(train_log),
        "eval_log": str(eval_log),
        "elapsed_seconds": float(elapsed),
        "train_returncode": train_rc,
        "eval_returncode": eval_rc,
        "train_seconds": train_metrics.get("seconds"),
        "best_val_visual_score_loss": train_metrics.get("best_val_visual_score_loss"),
        "train_steps": len(train_metrics.get("history") or []),
        "eval_seconds": eval_metrics.get("eval_seconds"),
        "policy_count": policy_corr.get("policy_count"),
        "valid_policy_count": policy_corr.get("valid_policy_count"),
        **benchmark,
        "next_rgb_mse": transition.get("next_rgb_mse"),
        "next_rgb_lpips": transition.get("next_rgb_lpips"),
        "next_state_mse_norm": transition.get("next_state_mse_norm"),
        "next_progress_mse": transition.get("next_progress_mse"),
        "success_bce": transition.get("success_bce"),
        "val_rgb_mse": final_val.get("rgb_mse"),
        "val_state_mse": final_val.get("state_mse"),
    }


def _is_better(record: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if record.get("status") != "ok":
        return False
    if best is None:
        return True
    return _objective_tuple(record) > _objective_tuple(best)


def _objective_tuple(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _float(record.get("eval_correlation_score")),
        _float(record.get("visual_world_model_score")),
        -_float(record.get("next_rgb_mse"), default=1e9),
    )


def _best_record(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return None
    return max(valid, key=_objective_tuple)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "experiment",
        "kept",
        "status",
        "name",
        "eval_correlation_score",
        "visual_world_model_score",
        "policy_score_pearson",
        "policy_score_spearman",
        "ood_policy_score_pearson",
        "ood_policy_score_spearman",
        "next_rgb_mse",
        "next_rgb_lpips",
        "next_state_mse_norm",
        "success_bce",
        "train_seconds",
        "eval_seconds",
        "elapsed_seconds",
        "valid_policy_count",
        "fatal",
        "idea",
        "args",
        "out_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    valid = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    if not rows:
        return
    kept = []
    running_best = []
    best: dict[str, Any] | None = None
    for row in rows:
        if row.get("status") == "ok" and _is_better(row, best):
            best = row
            kept.append(row)
        if best is not None:
            running_best.append((row["experiment"], _float(best.get("eval_correlation_score"))))
    plt.figure(figsize=(12, 5))
    if failed:
        plt.scatter([r["experiment"] for r in failed], [0 for _ in failed], c="#bdbdbd", s=16, label="failed", alpha=0.7)
    discarded = [r for r in valid if r not in kept]
    if discarded:
        plt.scatter([r["experiment"] for r in discarded], [_float(r.get("eval_correlation_score")) for r in discarded], c="#cfcfcf", s=18, label="discarded", alpha=0.65)
    if kept:
        plt.scatter([r["experiment"] for r in kept], [_float(r.get("eval_correlation_score")) for r in kept], c="#2ecc71", edgecolors="#145a32", s=46, label="kept")
    if running_best:
        plt.step([x for x, _ in running_best], [y for _, y in running_best], where="post", c="#58c98b", linewidth=2, label="running best")
    plt.xlabel("Experiment #")
    plt.ylabel("Eval correlation score (higher is better)")
    successes = len(kept)
    total = max(int(row["experiment"]) for row in rows)
    plt.title(f"World Model Autoresearch: {total} Experiments, {successes} Kept Improvements")
    plt.grid(True, alpha=0.25)
    plt.ylim(-0.03, 1.03)
    plt.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _commit_improvement(analysis_dir: Path, exp_idx: int, record: dict[str, Any], *, push: bool) -> None:
    rel = analysis_dir.relative_to(ROOT)
    score = _float(record.get("eval_correlation_score"))
    msg = f"Record visual world model exp {exp_idx:03d} corr {score:.4f}"
    cmds = [
        ["git", "add", "-f", str(rel)],
        ["git", "commit", "-m", msg],
    ]
    if push:
        cmds.append(["git", "push", "origin", "HEAD"])
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(proc.stdout, flush=True)
            return


def _slug(text: str) -> str:
    clean = []
    for ch in str(text).lower():
        clean.append(ch if ch.isalnum() else "_")
    return "_".join("".join(clean).split("_"))[:80]


if __name__ == "__main__":
    raise SystemExit(main())
