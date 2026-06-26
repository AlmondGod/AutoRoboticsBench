#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/output /workspace/task /workspace/logs
mkdir -p /workspace/.cache/pip /workspace/.cache/torch

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export ROBOAUTORESEARCH_REPO_ROOT="${ROBOAUTORESEARCH_REPO_ROOT:-/workspace/repo}"
export PYTHONPATH="${PYTHONPATH:-/workspace/task:/workspace/repo:/workspace/read_only}"

exec "$@"
