#!/usr/bin/env python3
"""Tiny SSH wrapper for running harness commands on a remote GPU host."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def load_config(repo_root: Path) -> dict:
    config_path = repo_root / "configs" / "compute.yaml"
    if not config_path.exists():
        raise SystemExit(
            "configs/compute.yaml not found. Copy configs/compute.yaml.example "
            "and fill in your remote GPU host settings."
        )

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read configs/compute.yaml") from exc

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command_parts = args.command
    if command_parts and command_parts[0] == "--":
        command_parts = command_parts[1:]
    if not command_parts:
        parser.error("expected a command after --")

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root)

    gpu_host = config.get("gpu_host")
    remote_repo_root = config.get("repo_root")
    ssh_config = config.get("ssh", {}) or {}
    port = str(ssh_config.get("port", 22))
    key_path = ssh_config.get("key_path")

    if not gpu_host:
        raise SystemExit("configs/compute.yaml must define gpu_host")
    if not remote_repo_root:
        raise SystemExit("configs/compute.yaml must define repo_root")

    remote_command = " ".join(command_parts).replace("{repo_root}", shlex.quote(str(remote_repo_root)))
    ssh_cmd = ["ssh", "-p", port]
    if key_path:
        ssh_cmd.extend(["-i", os.path.expanduser(str(key_path))])
    ssh_cmd.extend([str(gpu_host), remote_command])

    return subprocess.call(ssh_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
