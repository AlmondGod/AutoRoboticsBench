#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID is required" >&2
  exit 2
fi

if [[ -z "${TASK:-}" ]]; then
  echo "TASK is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-roboautoresearch:latest}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
TASK_SRC="${REPO_ROOT}/tasks/${TASK}"
TEMPLATE_SRC="${TASK_SRC}/workspace_template"
CONTAINER_NAME="robo_train_${RUN_ID}"

if [[ ! -d "${TEMPLATE_SRC}" ]]; then
  echo "Task workspace template not found: ${TEMPLATE_SRC}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/task" "${RUN_DIR}/output" "${RUN_DIR}/logs"
rm -rf "${RUN_DIR}/task"
mkdir -p "${RUN_DIR}/task"
cp -R "${TEMPLATE_SRC}/." "${RUN_DIR}/task/"
if [[ -f "${TASK_SRC}/task.md" ]]; then
  cp "${TASK_SRC}/task.md" "${RUN_DIR}/task/task.md"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  "${GPU_ARGS[@]}" \
  -v "${REPO_ROOT}/benchmark_harness:/workspace/read_only:ro" \
  -v "${RUN_DIR}/task:/workspace/task" \
  -v "${RUN_DIR}/output:/workspace/output" \
  -v "${RUN_DIR}/logs:/workspace/logs" \
  "${IMAGE_NAME}" >/dev/null

printf "%s\n" "${CONTAINER_NAME}" > "${RUN_DIR}/container_name.txt"
printf "Started %s for run %s\n" "${CONTAINER_NAME}" "${RUN_ID}"
