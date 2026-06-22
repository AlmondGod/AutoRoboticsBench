#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== RoboAutoresearch RunPod environment =="
echo "repo_root=${REPO_ROOT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; GPU is not visible from this container" >&2
fi

python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}"
python -m pip install pyyaml numpy

if [[ "${INSTALL_ROBOCASA_RUNTIME:-1}" == "1" ]]; then
  "${REPO_ROOT}/scripts/install_robocasa_runtime.sh"
fi

mkdir -p /workspace/cache /workspace/task /workspace/output /workspace/logs

echo
echo "RunPod environment is ready."
echo "Start a dockerless run with:"
echo "  python scripts/launch_runpod_run.py --agent codex --task toy_pickplace --base dummy --seed 0"
