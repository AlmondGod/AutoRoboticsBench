#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RUN_ID \"command\"" >&2
  exit 2
fi

RUN_ID="$1"
shift
COMMAND="$*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
CONTAINER_FILE="${RUN_DIR}/container_name.txt"

if [[ ! -f "${CONTAINER_FILE}" ]]; then
  echo "Container name file not found: ${CONTAINER_FILE}" >&2
  exit 1
fi

CONTAINER_NAME="$(cat "${CONTAINER_FILE}")"

if [[ "${REMOTE_GPU:-0}" == "1" ]]; then
  python "${REPO_ROOT}/scripts/remote_exec.py" -- \
    "cd {repo_root} && docker exec ${CONTAINER_NAME} bash -lc $(printf '%q' "${COMMAND}")"
else
  docker exec "${CONTAINER_NAME}" bash -lc "${COMMAND}"
fi
