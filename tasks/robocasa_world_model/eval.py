from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_POLICY_SET,
    DEFAULT_SPLIT,
    TransitionData,
    load_transition_data,
    normalize_data,
    save_json,
)
from tasks.robocasa_world_model.inference import load_world_model, rollout_score
from tasks.robocasa_world_model.policy_set import discover_policy_runs
from tasks.robocasa_world_model.trace import generate_policy_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoboCasa world-model transition fit and policy-score correlation.")
    parser.add_argument("--checkpoint", "--world-model", dest="checkpoint", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-set", default="", help="Optional JSON listing real evals and world-model rollout traces.")
    parser.add_argument("--policy-runs-root", default="runs/autorobobench/robocasa_bc5")
    parser.add_argument("--trace-root", default="runs/autorobobench/robocasa_world_model/policy_traces")
    parser.add_argument("--trace-episodes-per-task", type=int, default=1)
    parser.add_argument("--trace-max-steps", type=int, default=260)
    parser.add_argument("--trace-commit-steps", type=int, default=16)
    parser.add_argument("--trace-source", choices=["auto", "sim", "offline"], default="auto")
    parser.add_argument("--generate-missing-traces", action="store_true", help="Generate missing policy traces before scoring.")
    parser.add_argument("--no-generate-missing-traces", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transition-only", action="store_true")
    args = parser.parse_args()

    start = time.monotonic()
    world = load_world_model(str(args.checkpoint), device=str(args.device))
    ckpt = world["checkpoint"]
    _, val_raw, summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        frame_stride=int(args.frame_stride),
    )
    val = normalize_data(val_raw, ckpt["stats"])
    transition_metrics = _transition_eval(world, val, int(args.batch_size))
    correlation = (
        {"enabled": False, "reason": "transition-only requested"}
        if args.transition_only
        else _policy_correlation(
            world,
            policy_set_path=str(args.policy_set) if args.policy_set else "",
            policy_runs_root=str(args.policy_runs_root),
            manifest_path=str(args.manifest),
            split_path=str(args.split),
            trace_root=str(args.trace_root),
            episodes_per_task=int(args.trace_episodes_per_task),
            max_steps=int(args.trace_max_steps),
            commit_steps=int(args.trace_commit_steps),
            device=str(args.device),
            trace_source=str(args.trace_source),
            generate_missing=bool(args.generate_missing_traces) and not bool(args.no_generate_missing_traces),
        )
    )
    benchmark = _benchmark_score(correlation, transition_metrics)
    payload = {
        "task": "robocasa_world_model",
        "checkpoint": str(args.checkpoint),
        "metric": "world_model_benchmark_score",
        **benchmark,
        "reproducibility_integrity": 1.0,
        "transition_metrics": transition_metrics,
        "policy_correlation": correlation,
        "summary": summary,
        "eval_seconds": time.monotonic() - start,
    }
    save_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


@torch.no_grad()
def _transition_eval(world: dict, data: TransitionData, batch_size: int) -> dict[str, float]:
    model = world["model"]
    device = world["device"]
    model.eval()
    state_sse = 0.0
    progress_sse = 0.0
    success_loss = 0.0
    count = 0
    for start in range(0, len(data), batch_size):
        end = min(len(data), start + batch_size)
        batch = {
            "state": torch.as_tensor(data.state[start:end], dtype=torch.float32, device=device),
            "action": torch.as_tensor(data.action[start:end], dtype=torch.float32, device=device),
            "next_state": torch.as_tensor(data.next_state[start:end], dtype=torch.float32, device=device),
            "progress": torch.as_tensor(data.progress[start:end], dtype=torch.float32, device=device),
            "next_progress": torch.as_tensor(data.next_progress[start:end], dtype=torch.float32, device=device),
            "success": torch.as_tensor(data.success[start:end], dtype=torch.float32, device=device),
            "task_id": torch.as_tensor(data.task_id[start:end], dtype=torch.long, device=device),
        }
        out = model(batch["state"], batch["action"], batch["task_id"], batch["progress"])
        n = end - start
        state_sse += float((out["next_state"] - batch["next_state"]).square().mean(dim=-1).sum().detach().cpu())
        progress_sse += float((out["next_progress"] - batch["next_progress"]).square().sum().detach().cpu())
        success_loss += float(torch.nn.functional.binary_cross_entropy_with_logits(out["success_logit"], batch["success"], reduction="sum").detach().cpu())
        count += n
    return {
        "samples": int(count),
        "next_state_mse_norm": state_sse / max(1, count),
        "next_progress_mse": progress_sse / max(1, count),
        "success_bce": success_loss / max(1, count),
    }


def _policy_correlation(
    world: dict,
    *,
    policy_set_path: str,
    policy_runs_root: str,
    manifest_path: str,
    split_path: str,
    trace_root: str,
    episodes_per_task: int,
    max_steps: int,
    commit_steps: int,
    device: str,
    trace_source: str,
    generate_missing: bool,
) -> dict:
    materialized = Path(trace_root) / "policy_set_materialized.json"
    if not policy_set_path and materialized.exists():
        policy_set_path = str(materialized)
    policies, skipped = _supported_policies(_load_or_discover_policies(policy_set_path, policy_runs_root))
    if generate_missing:
        missing = [p for p in policies if not _trace_paths(p)]
        if missing:
            generated = generate_policy_traces(
                policies=missing,
                manifest_path=manifest_path,
                split_path=split_path,
                trace_root=trace_root,
                episodes_per_task=episodes_per_task,
                max_steps=max_steps,
                commit_steps=commit_steps,
                device=device,
                source=trace_source,
            )
            generated_by_name = {row["name"]: row for row in generated}
            policies = [generated_by_name.get(row["name"], row) for row in policies]
            materialized_path = Path(trace_root) / "policy_set_materialized.json"
            save_json(materialized_path, {"task": "robocasa_world_model_policy_set", "policies": policies})
            policy_set_path = str(materialized_path)
    rows = []
    for policy in policies:
        row = _score_policy(world, policy)
        rows.append(row)
    valid = [row for row in rows if row.get("real_success_rate") is not None and row.get("predicted_success") is not None]
    real = np.asarray([row["real_success_rate"] for row in valid], dtype=np.float64)
    pred = np.asarray([row["predicted_success"] for row in valid], dtype=np.float64)
    return {
        "enabled": True,
        "policy_set": str(policy_set_path or policy_runs_root),
        "policies": rows,
        "skipped_policies": skipped,
        "valid_policy_count": int(len(valid)),
        "pearson": _pearson(real, pred),
        "spearman": _spearman(real, pred),
        "calibration_rmse": _rmse(real, pred),
        "calibration_mae": _mae(real, pred),
        "ood_pearson": _ood_corr(rows, method="pearson"),
        "ood_spearman": _ood_corr(rows, method="spearman"),
        "contains_ood": any(bool(row.get("ood", False)) for row in rows),
        "notes": [
            "Each policy entry may provide real_eval_json and trace_npz_paths or trace_dir.",
            "Trace npz files must contain states [T,state_dim], actions [T,action_dim], and task_id.",
        ],
    }


def _benchmark_score(correlation: dict, transition: dict) -> dict:
    if not correlation.get("enabled"):
        ranking = 0.0
        calibration = 0.0
        ood = 0.0
        pearson = None
        spearman = None
        calibration_rmse = None
        calibration_mae = None
    else:
        pearson = correlation.get("pearson")
        spearman = correlation.get("spearman")
        ranking = _mean_present([_corr_score(pearson), _corr_score(spearman)])
        calibration_rmse = correlation.get("calibration_rmse")
        calibration_mae = correlation.get("calibration_mae")
        calibration = _mse_like_score(calibration_rmse, scale=0.25)
        ood = _mean_present([
            _corr_score(correlation.get("ood_pearson")),
            _corr_score(correlation.get("ood_spearman")),
        ])
        if ood == 0.0:
            ood = ranking
    progress = _mse_like_score(transition.get("next_progress_mse"), scale=0.05)
    next_state = _mse_like_score(transition.get("next_state_mse_norm"), scale=0.05)
    weights = {
        "policy_and_ood_ranking_score": 0.50,
        "success_calibration_score": 0.20,
        "progress_score": 0.15,
        "next_state_score": 0.10,
    }
    score = (
        weights["policy_and_ood_ranking_score"] * (ranking + ood)
        + weights["success_calibration_score"] * calibration
        + weights["progress_score"] * progress
        + weights["next_state_score"] * next_state
    )
    return {
        "world_model_benchmark_score": float(max(0.0, min(1.0, score))),
        "policy_ranking_score": float(ranking),
        "success_calibration_score": float(calibration),
        "progress_score": float(progress),
        "next_state_score": float(next_state),
        "ood_ranking_score": float(ood),
        "policy_score_pearson": None if pearson is None else float(pearson),
        "policy_score_spearman": None if spearman is None else float(spearman),
        "success_calibration_rmse": None if calibration_rmse is None else float(calibration_rmse),
        "success_calibration_mae": None if calibration_mae is None else float(calibration_mae),
        "benchmark_score_weights": weights,
    }


def _supported_policies(policies: list[dict]) -> tuple[list[dict], list[dict]]:
    supported = []
    skipped = []
    unsupported_types = {"autorobobench_robocasa_bc5_policy_ensemble"}
    for policy in policies:
        checkpoint = policy.get("checkpoint", "")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            policy_type = str(payload.get("policy_type", ""))
        except Exception as exc:
            skipped.append({**policy, "skip_reason": f"checkpoint_load_failed: {type(exc).__name__}: {exc}"})
            continue
        if policy_type in unsupported_types:
            skipped.append({**policy, "policy_type": policy_type, "skip_reason": "not loadable by tasks.robocasa_bc5.inference"})
            continue
        supported.append({**policy, "policy_type": policy_type})
    return supported, skipped


def _load_or_discover_policies(policy_set_path: str, policy_runs_root: str) -> list[dict]:
    if policy_set_path:
        payload = json.loads(Path(policy_set_path).read_text())
        policies = list(payload.get("policies", []))
        if policies:
            return policies
    if DEFAULT_POLICY_SET.exists() and not policy_set_path:
        payload = json.loads(DEFAULT_POLICY_SET.read_text())
        policies = list(payload.get("policies", []))
        if policies:
            return policies
    policies = discover_policy_runs(policy_runs_root)
    if not policies:
        raise FileNotFoundError(
            f"no policies discovered under {policy_runs_root}; expected */eval_10_per_task_local.json plus policy_best.pt"
        )
    return policies


def _score_policy(world: dict, policy: dict) -> dict:
    traces = _trace_paths(policy)
    trace_scores = []
    for path in traces:
        data = np.load(path)
        if "states" not in data or "actions" not in data:
            continue
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if len(states) == 0 or len(actions) == 0:
            continue
        task_id = int(np.asarray(data["task_id"]).reshape(-1)[0]) if "task_id" in data else int(policy.get("task_id", 0))
        score = rollout_score(world, states[0], actions, task_id)
        trace_scores.append(float(score["predicted_success"]))
    real = _real_success(policy.get("real_eval_json", ""))
    return {
        "name": str(policy.get("name", "")),
        "ood": bool(policy.get("ood", False)),
        "real_success_rate": real,
        "predicted_success": float(np.mean(trace_scores)) if trace_scores else None,
        "trace_count": len(trace_scores),
        "trace_sources": [str(path) for path in traces],
    }


def _trace_paths(policy: dict) -> list[Path]:
    paths = [Path(str(path)) for path in policy.get("trace_npz_paths", [])]
    if policy.get("trace_dir"):
        paths.extend(sorted(Path(str(policy["trace_dir"])).rglob("*.npz")))
    out = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _real_success(path: str) -> float | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    for key in ("success_rate", "hidden_final_success", "peak_final_success"):
        if key in payload:
            return float(payload[key])
    if "tracks" in payload:
        for value in payload["tracks"].values():
            if isinstance(value, dict):
                for key in ("success_rate", "hidden_final_success", "peak_final_success"):
                    if key in value:
                        return float(value[key])
    return None


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _rmse(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0:
        return None
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _mae(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0:
        return None
    return float(np.mean(np.abs(x - y)))


def _ood_corr(rows: list[dict], *, method: str) -> float | None:
    valid = [
        row
        for row in rows
        if row.get("ood") and row.get("real_success_rate") is not None and row.get("predicted_success") is not None
    ]
    if len(valid) < 2:
        return None
    real = np.asarray([row["real_success_rate"] for row in valid], dtype=np.float64)
    pred = np.asarray([row["predicted_success"] for row in valid], dtype=np.float64)
    if method == "spearman":
        return _spearman(real, pred)
    return _pearson(real, pred)


def _corr_score(value: float | None) -> float | None:
    if value is None:
        return None
    return float(max(0.0, min(1.0, (float(value) + 1.0) / 2.0)))


def _mse_like_score(value: float | None, *, scale: float) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - float(value) / max(float(scale), 1e-12))))


def _mean_present(values: list[float | None]) -> float:
    present = [float(value) for value in values if value is not None]
    if not present:
        return 0.0
    return float(np.mean(present))


if __name__ == "__main__":
    main()
