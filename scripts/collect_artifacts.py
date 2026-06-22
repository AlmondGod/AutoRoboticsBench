#!/usr/bin/env python3
"""Collect simple run metadata and tarball snapshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def run_command(command: list[str]) -> dict:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}


def tar_directory(source: Path, out_path: Path) -> None:
    with tarfile.open(out_path, "w:gz") as tar:
        if source.exists():
            tar.add(source, arcname=source.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / args.run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    container_file = run_dir / "container_name.txt"
    container_name = container_file.read_text(encoding="utf-8").strip() if container_file.exists() else None

    if container_name:
        nvidia_smi = run_command(["docker", "exec", container_name, "bash", "-lc", "nvidia-smi"])
        pip_freeze = run_command(["docker", "exec", container_name, "bash", "-lc", "python -m pip freeze"])
    else:
        nvidia_smi = run_command(["nvidia-smi"])
        pip_freeze = run_command(["python", "-m", "pip", "freeze"])

    system_info = {
        "run_id": args.run_id,
        "collected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "container_name": container_name,
        "final_submission_exists": (run_dir / "output" / "final_submission").exists(),
        "nvidia_smi": nvidia_smi,
        "pip_freeze": pip_freeze,
    }
    (artifacts_dir / "system_info.json").write_text(
        json.dumps(system_info, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tar_directory(run_dir / "task", artifacts_dir / "task_snapshot.tar.gz")
    tar_directory(run_dir / "output", artifacts_dir / "output_snapshot.tar.gz")
    print(f"Wrote artifacts to {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
