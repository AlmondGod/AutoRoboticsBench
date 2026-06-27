#!/usr/bin/env python3
"""Fail fast when benchmark reference artifacts are missing or miswired."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


COUNTED_SUITE = "autorobobench_v0"
POLICY_SET_TASKS = {"robocasa_visual_world_model", "robocasa_world_model"}
POSTTRAINING_TASKS = {"robocasa_world_model_posttraining"}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise AssertionError(f"missing JSON: {path}") from None
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON: {path}: {exc}") from None


def repo_path(repo_root: Path, value: str, *, relative_to: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if (repo_root / path).exists():
        return repo_root / path
    if relative_to is not None and (relative_to / path).exists():
        return relative_to / path
    return repo_root / path


def real_success_from_eval(path: Path) -> float | None:
    payload = load_json(path)
    for key in ("success_rate", "hidden_final_success", "peak_final_success", "offlinerl_final_success"):
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return None
    return None


def is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:128]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def require_artifact(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} missing: {path}")
    elif path.suffix == ".pt" and is_lfs_pointer(path):
        errors.append(f"{label} is an unresolved Git LFS pointer, run git lfs pull: {path}")


def suite_tasks(repo_root: Path, suite: str) -> list[str]:
    payload = load_json(repo_root / "benchmark.json")
    tracks = payload.get("suites", {}).get(suite, {}).get("tracks", [])
    tasks = [str(track.get("id", "")) for track in tracks if track.get("id")]
    if not tasks:
        raise AssertionError(f"benchmark suite has no tasks: {suite}")
    return tasks


def check_task_template(repo_root: Path, task: str, errors: list[str]) -> None:
    task_dir = repo_root / "tasks" / task
    template = task_dir / "workspace_template"
    if not task_dir.exists():
        errors.append(f"{task}: missing task directory: {task_dir}")
        return
    if not template.exists():
        errors.append(f"{task}: missing workspace template: {template}")
        return
    for name in ("train.py", "inference.py"):
        if not (template / name).exists():
            errors.append(f"{task}: workspace template missing {name}")
    if not (template / "task.md").exists() and not (task_dir / "task.md").exists() and not (task_dir / "INSTRUCTIONS.md").exists():
        errors.append(f"{task}: missing task guidance; expected workspace_template/task.md, task.md, or INSTRUCTIONS.md")


def check_robocasa5_metadata(repo_root: Path, errors: list[str]) -> None:
    manifest_path = repo_root / "data" / "robocasa5" / "manifest.json"
    split_path = repo_root / "data" / "autorobobench" / "robocasa_bc5_splits.json"
    try:
        manifest = load_json(manifest_path)
    except AssertionError as exc:
        errors.append(str(exc))
        manifest = {}
    try:
        split = load_json(split_path)
    except AssertionError as exc:
        errors.append(str(exc))
        split = {}
    if manifest and not manifest.get("tasks"):
        errors.append(f"{manifest_path}: missing tasks")
    if split and not split.get("tasks"):
        errors.append(f"{split_path}: missing tasks")


def check_policy_set(repo_root: Path, errors: list[str]) -> None:
    path = repo_root / "data" / "autorobobench" / "robocasa_world_model_policy_set.json"
    payload = load_json(path)
    policies = payload.get("policies")
    if not isinstance(policies, list) or len(policies) < 7:
        errors.append(f"{path}: expected at least seven policy entries")
        return
    valid_real = 0
    real_labels: list[float] = []
    for index, policy in enumerate(policies):
        prefix = f"{path}: policies[{index}] {policy.get('name', '<unnamed>')!r}"
        checkpoint_value = str(policy.get("checkpoint", ""))
        if not checkpoint_value:
            errors.append(f"{prefix}: missing checkpoint")
        else:
            checkpoint = repo_path(repo_root, checkpoint_value, relative_to=path.parent)
            require_artifact(checkpoint, f"{prefix}: checkpoint", errors)
        inference = str(policy.get("inference", ""))
        if not inference:
            errors.append(f"{prefix}: missing inference module")
        if "real_success_rate" in policy:
            try:
                real_labels.append(float(policy["real_success_rate"]))
                valid_real += 1
            except (TypeError, ValueError):
                errors.append(f"{prefix}: invalid real_success_rate")
        else:
            eval_value = str(policy.get("real_eval_json", ""))
            if not eval_value:
                errors.append(f"{prefix}: missing real_success_rate or real_eval_json")
            else:
                eval_path = repo_path(repo_root, eval_value, relative_to=path.parent)
                if not eval_path.exists():
                    errors.append(f"{prefix}: real_eval_json missing: {eval_value}")
                elif real_success_from_eval(eval_path) is None:
                    errors.append(f"{prefix}: real_eval_json has no success metric: {eval_value}")
                else:
                    real_labels.append(float(real_success_from_eval(eval_path)))
                    valid_real += 1
    if valid_real < 2:
        errors.append(f"{path}: fewer than two policies have usable real success labels")
    if len({round(value, 6) for value in real_labels}) < 2:
        errors.append(f"{path}: policy real success labels have no variance; visual world-model correlation would be capped")


def check_world_model_posttraining(repo_root: Path, errors: list[str]) -> None:
    spec_path = repo_root / "tasks" / "robocasa_world_model_posttraining" / "task.json"
    spec = load_json(spec_path)
    required = {
        "default_policy_checkpoint": spec.get("default_policy_checkpoint"),
        "default_world_model_checkpoint": spec.get("default_world_model_checkpoint"),
    }
    for key, value in required.items():
        if not value:
            errors.append(f"{spec_path}: missing {key}")
            continue
        artifact = repo_path(repo_root, str(value))
        require_artifact(artifact, f"{spec_path}: {key}", errors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate required RoboAutoresearch benchmark artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--suite", default=COUNTED_SUITE)
    parser.add_argument("--task", action="append", default=[], help="Task id to check; defaults to every task in --suite.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    tasks = args.task or suite_tasks(repo_root, str(args.suite))
    errors: list[str] = []
    for task in tasks:
        check_task_template(repo_root, task, errors)
    if any(task in POLICY_SET_TASKS for task in tasks):
        check_robocasa5_metadata(repo_root, errors)
        check_policy_set(repo_root, errors)
    if any(task in POSTTRAINING_TASKS for task in tasks):
        check_world_model_posttraining(repo_root, errors)

    payload = {"ok": not errors, "suite": args.suite, "tasks": tasks, "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        print("Benchmark preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print(f"Benchmark preflight passed for {', '.join(tasks)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
