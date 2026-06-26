#!/usr/bin/env python3
"""Run or prepare a RunPod dockerless benchmark suite.

The single-task RunPod launcher is intentionally task-scoped. This script adds
the missing suite layer: read benchmark.json, prepare each counted track, apply
the same hard timeout per task, optionally run an external agent command, then
finalize and aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any


def load_suite(repo_root: Path, suite_name: str) -> list[dict[str, Any]]:
    payload = json.loads((repo_root / "benchmark.json").read_text(encoding="utf-8"))
    suites = payload.get("suites", {})
    suite = suites.get(suite_name)
    if not isinstance(suite, dict):
        raise SystemExit(f"Suite not found in benchmark.json: {suite_name}")
    tracks = suite.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise SystemExit(f"Suite has no tracks: {suite_name}")
    return tracks


def run_capture(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)


def run_stream(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def git_current_branch(repo_root: Path) -> str:
    completed = run_capture(["git", "branch", "--show-current"], cwd=repo_root)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or "git branch failed")
    return completed.stdout.strip()


def git_switch(repo_root: Path, branch: str) -> None:
    completed = run_stream(["git", "switch", branch], cwd=repo_root)
    if completed.returncode != 0:
        raise SystemExit(f"Failed to switch back to {branch}")


def parse_run_id(output: str) -> str:
    run_id = ""
    for line in output.splitlines():
        if line.startswith("run_id="):
            run_id = line.split("=", 1)[1].strip()
    if not run_id:
        raise SystemExit("Could not parse run_id from runpod_prepare_run.sh output")
    return run_id


def prepare_run(
    repo_root: Path,
    *,
    agent: str,
    model: str,
    scaffold: str,
    task: str,
    base: str,
    seed: int,
    timeout_hours: float,
    start_branch: str,
    workspace_root: str,
    skip_setup: bool,
    no_git_branch: bool,
) -> str:
    command = [
        str(repo_root / "scripts" / "runpod_prepare_run.sh"),
        "--agent",
        agent,
        "--model",
        model,
        "--scaffold",
        scaffold,
        "--task",
        task,
        "--base",
        base,
        "--seed",
        str(seed),
        "--timeout-hours",
        str(timeout_hours),
        "--workspace-root",
        workspace_root,
        "--start-branch",
        start_branch,
    ]
    if skip_setup:
        command.append("--skip-setup")
    if no_git_branch:
        command.append("--no-git-branch")
    completed = run_capture(command, cwd=repo_root)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"Failed to prepare task {task}")
    return parse_run_id(completed.stdout)


def render_agent_command(template: str, values: dict[str, str]) -> str:
    rendered = Template(template).safe_substitute(values)
    return rendered.format(**values)


def run_agent_command(command: str, *, cwd: Path, env: dict[str, str], timeout_seconds: int) -> int:
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124


def run_agent_until_timeout(command: str, *, cwd: Path, env: dict[str, str], timeout_seconds: int) -> list[dict[str, Any]]:
    started = datetime.now(timezone.utc).timestamp()
    deadline = started + timeout_seconds
    invocations: list[dict[str, Any]] = []
    index = 0

    while True:
        remaining = int(deadline - datetime.now(timezone.utc).timestamp())
        if remaining <= 0:
            break
        index += 1
        invocation_env = dict(env)
        invocation_env["INVOCATION_INDEX"] = str(index)
        invocation_env["REMAINING_SECONDS"] = str(remaining)
        invocation_started = datetime.now(timezone.utc)
        returncode = run_agent_command(command, cwd=cwd, env=invocation_env, timeout_seconds=remaining)
        invocation_finished = datetime.now(timezone.utc)
        invocations.append(
            {
                "index": index,
                "returncode": returncode,
                "started_at": invocation_started.isoformat().replace("+00:00", "Z"),
                "finished_at": invocation_finished.isoformat().replace("+00:00", "Z"),
                "remaining_seconds_at_start": remaining,
            }
        )
        if returncode == 124:
            break
    return invocations


def finalize_run(repo_root: Path, run_id: str, task: str) -> int:
    completed = run_stream(
        [
            sys.executable,
            str(repo_root / "scripts" / "finalize_run.py"),
            "--run-id",
            run_id,
            "--task",
            task,
            "--mode",
            "runpod",
        ],
        cwd=repo_root,
    )
    return completed.returncode


def write_suite_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or run every task in a RunPod suite.")
    parser.add_argument("--suite", default="autorobobench_v0")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--scaffold", default="")
    parser.add_argument("--base", default="dummy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-hours", type=float, default=2.0)
    parser.add_argument("--start-branch", default="main")
    parser.add_argument("--workspace-root", default="/workspace")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--no-git-branch", action="store_true")
    parser.add_argument("--agent-command", default="", help="Shell command run for each task. Supports {run_id}, {task}, {prompt}, {run_dir}, {timeout_seconds}.")
    parser.add_argument("--repeat-agent-command-until-timeout", action="store_true", help="Re-run --agent-command until the per-task timeout expires.")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.timeout_hours <= 0:
        raise SystemExit("--timeout-hours must be > 0")

    repo_root = Path(__file__).resolve().parents[1]
    tracks = load_suite(repo_root, args.suite)
    timeout_seconds = int(float(args.timeout_hours) * 3600)
    suite_run_id = f"{args.suite}_{args.agent}_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    suite_state_path = repo_root / "runs" / "suites" / f"{suite_run_id}.json"
    suite_state: dict[str, Any] = {
        "suite_run_id": suite_run_id,
        "suite": args.suite,
        "agent": args.agent,
        "model": args.model,
        "scaffold": args.scaffold,
        "base": args.base,
        "seed": int(args.seed),
        "timeout_hours_per_task": float(args.timeout_hours),
        "timeout_seconds_per_task": timeout_seconds,
        "start_branch": args.start_branch,
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs": [],
    }

    for index, track in enumerate(tracks, start=1):
        task = str(track["id"])
        record: dict[str, Any] = {"task": task, "index": index, "weight": track.get("weight")}
        try:
            if not args.no_git_branch and git_current_branch(repo_root) != args.start_branch:
                git_switch(repo_root, args.start_branch)
            run_id = prepare_run(
                repo_root,
                agent=args.agent,
                model=args.model,
                scaffold=args.scaffold,
                task=task,
                base=args.base,
                seed=int(args.seed),
                timeout_hours=float(args.timeout_hours),
                start_branch=args.start_branch,
                workspace_root=args.workspace_root,
                skip_setup=bool(args.skip_setup),
                no_git_branch=bool(args.no_git_branch),
            )
            run_dir = repo_root / "runs" / run_id
            prompt = run_dir / "prompt.txt"
            record.update({"run_id": run_id, "run_dir": str(run_dir), "prompt": str(prompt), "status": "prepared"})

            if args.agent_command:
                env = os.environ.copy()
                env.update(
                    {
                        "RUN_ID": run_id,
                        "TASK": task,
                        "PROMPT": str(prompt),
                        "RUN_DIR": str(run_dir),
                        "TIMEOUT_SECONDS": str(timeout_seconds),
                    }
                )
                command = render_agent_command(
                    args.agent_command,
                    {
                        "run_id": run_id,
                        "task": task,
                        "prompt": str(prompt),
                        "run_dir": str(run_dir),
                        "timeout_seconds": str(timeout_seconds),
                    },
                )
                record["agent_command"] = command
                if args.repeat_agent_command_until_timeout:
                    record["agent_invocations"] = run_agent_until_timeout(
                        command,
                        cwd=repo_root,
                        env=env,
                        timeout_seconds=timeout_seconds,
                    )
                    last_returncode = None
                    if record["agent_invocations"]:
                        last_returncode = record["agent_invocations"][-1].get("returncode")
                    record["agent_returncode"] = last_returncode
                    record["status"] = "agent_timeout" if last_returncode == 124 else "agent_budget_exhausted"
                else:
                    record["agent_returncode"] = run_agent_command(command, cwd=repo_root, env=env, timeout_seconds=timeout_seconds)
                    record["status"] = "agent_timeout" if record["agent_returncode"] == 124 else "agent_complete"

            if not args.no_finalize and args.agent_command:
                record["finalize_returncode"] = finalize_run(repo_root, run_id, task)
                if record["finalize_returncode"] == 0:
                    record["status"] = "finalized"

            if not args.no_git_branch and git_current_branch(repo_root) != args.start_branch:
                git_switch(repo_root, args.start_branch)
        except SystemExit as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            suite_state["runs"].append(record)
            write_suite_state(suite_state_path, suite_state)
            if not args.continue_on_error:
                raise
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = repr(exc)
            suite_state["runs"].append(record)
            write_suite_state(suite_state_path, suite_state)
            if not args.continue_on_error:
                raise
        else:
            suite_state["runs"].append(record)
            write_suite_state(suite_state_path, suite_state)

    suite_state["finished_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    write_suite_state(suite_state_path, suite_state)
    print(f"Wrote suite state to {suite_state_path}")

    if any(run.get("status") == "finalized" for run in suite_state["runs"]):
        run_stream([sys.executable, str(repo_root / "scripts" / "aggregate_runs.py")], cwd=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
