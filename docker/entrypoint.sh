#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/output /workspace/task /workspace/logs
mkdir -p /workspace/.cache/pip /workspace/.cache/torch

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"

exec "$@"
