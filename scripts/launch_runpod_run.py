#!/usr/bin/env python3
"""Create a dockerless RunPod benchmark run.

This mode is intended for GPU pods that are already containers and cannot start
nested Docker containers. It preserves the /workspace/task and /workspace/output
paths by linking them to a run directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


def git_output(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def create_run_branch(repo_root: Path, run_id: str, start_branch: str, enabled: bool) -> str | None:
    if not enabled:
        return None

    try:
        current_branch = git_output(repo_root, ["branch", "--show-current"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: git is unavailable; skipping branch creation.")
        return None

    if current_branch != start_branch:
        raise SystemExit(
            f"Refusing to create an agent branch from {current_branch!r}; "
            f"expected {start_branch!r}. Use --no-git-branch to skip this guard."
        )

    status = git_output(repo_root, ["status", "--porcelain"])
    if status:
        raise SystemExit(
            "Refusing to create an agent branch with uncommitted changes. "
            "Commit/stash them first, or use --no-git-branch."
        )

    branch_name = f"codex/{run_id}"
    subprocess.run(["git", "switch", "-c", branch_name], cwd=repo_root, check=True)
    return branch_name


def link_workspace_path(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir() and not any(link_path.iterdir()):
            link_path.rmdir()
        else:
            raise SystemExit(
                f"Refusing to replace non-empty workspace path: {link_path}. "
                "Move it aside or pass a different --workspace-root."
            )
    link_path.symlink_to(target, target_is_directory=True)


def build_prompt(
    run_id: str,
    task: str,
    agent: str,
    model: str,
    scaffold: str,
    base: str,
    seed: int,
    timeout_hours: float,
    branch_name: str | None,
) -> str:
    branch_text = branch_name or "not created"
    return f"""RoboAutoresearch Bench external-agent run

Run ID: {run_id}
Agent: {agent}
Model: {model or "unspecified"}
Scaffold: {scaffold or "unspecified"}
Task: {task}
Base: {base}
Seed: {seed}
Timeout: {timeout_hours} hours
Mode: RunPod dockerless development mode
Git branch: {branch_text}

You are operating inside the outer RunPod GPU container through the generated wrapper:

  runs/{run_id}/run.sh "<command>"

Instructions:
- Read /workspace/task/task.md before making changes.
- Run all commands through runs/{run_id}/run.sh "<command>" from the repo root.
- Modify only /workspace/task and /workspace/output.
- You may read /workspace/readonly, but do not modify it.
- Put the final submission in /workspace/output/final_submission.
- Do not access held-out eval data or modify eval code.
- This mode is not clean Docker isolation; use it for RunPod development runs.
- If you make tracked repo source changes that improve eval score, commit them on this branch with:
  python scripts/commit_improvement.py --run-id {run_id} --task {task}
- If token/cost usage is available, write it to runs/{run_id}/run_usage.json with keys:
  input_tokens, output_tokens, reasoning_tokens, total_tokens, estimated_usd
- Do not manually edit run_summary.json; finalization writes it.
- Do not commit runs/, output artifacts, secrets, or dataset files.
- Never merge this branch to main.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--scaffold", default="")
    parser.add_argument("--task", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-hours", type=float, default=10)
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--start-branch", default="main")
    parser.add_argument("--no-git-branch", action="store_true")
    args = parser.parse_args()

    if args.workspace_root != Path("/workspace"):
        print(
            "Warning: most task scripts use absolute /workspace paths. "
            "Use a custom --workspace-root only for launcher path tests, not for real runs."
        )

    repo_root = Path(__file__).resolve().parents[1]
    task_src = repo_root / "tasks" / args.task
    template_src = task_src / "workspace_template"
    if not template_src.exists():
        raise SystemExit(f"Task workspace template not found: {template_src}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{slug(args.agent)}_{slug(args.task)}_{slug(args.base)}_seed{args.seed}_{timestamp}"
    branch_name = create_run_branch(
        repo_root=repo_root,
        run_id=run_id,
        start_branch=args.start_branch,
        enabled=not args.no_git_branch,
    )
    run_dir = repo_root / "runs" / run_id
    task_dir = run_dir / "task"
    output_dir = run_dir / "output"
    logs_dir = run_dir / "logs"

    task_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    shutil.copytree(template_src, task_dir, dirs_exist_ok=True)
    if (task_src / "task.md").exists():
        shutil.copy2(task_src / "task.md", task_dir / "task.md")

    workspace_root = args.workspace_root
    link_workspace_path(workspace_root / "task", task_dir)
    link_workspace_path(workspace_root / "output", output_dir)
    link_workspace_path(workspace_root / "logs", logs_dir)
    link_workspace_path(workspace_root / "readonly", repo_root / "readonly")

    prompt = build_prompt(
        run_id,
        args.task,
        args.agent,
        args.model,
        args.scaffold,
        args.base,
        args.seed,
        args.timeout_hours,
        branch_name,
    )
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "mode.txt").write_text("runpod_no_docker\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "agent": args.agent,
                "model": args.model,
                "scaffold": args.scaffold,
                "task": args.task,
                "base": args.base,
                "seed": int(args.seed),
                "timeout_hours": float(args.timeout_hours),
                "mode": "runpod",
                "branch": branch_name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if branch_name:
        (run_dir / "branch_name.txt").write_text(branch_name + "\n", encoding="utf-8")

    run_sh = run_dir / "run.sh"
    run_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ $# -lt 1 ]]; then\n"
        "  echo 'Usage: run.sh \"command\"' >&2\n"
        "  exit 2\n"
        "fi\n"
        f"cd {sh_quote(str(repo_root))}\n"
        f"./scripts/run_runpod_command.sh {sh_quote(run_id)} \"$*\"\n",
        encoding="utf-8",
    )
    run_sh.chmod(0o755)

    print(prompt)
    print(f"run_id={run_id}")
    if branch_name:
        print(f"branch={branch_name}")
    print(f"workspace_root={workspace_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
