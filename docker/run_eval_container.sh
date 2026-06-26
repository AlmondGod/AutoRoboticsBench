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
SUBMISSION_DIR="${RUN_DIR}/output/final_submission"
EVAL_DIR="${RUN_DIR}/eval"
CONTAINER_NAME="robo_eval_${RUN_ID}"

if [[ ! -d "${SUBMISSION_DIR}" ]]; then
  echo "Final submission directory not found: ${SUBMISSION_DIR}" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

docker run --rm \
  --name "${CONTAINER_NAME}" \
  "${GPU_ARGS[@]}" \
  --network none \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${SUBMISSION_DIR}:/workspace/final_submission:ro" \
  -v "${EVAL_DIR}:/workspace/eval" \
  -w /workspace/repo \
  "${IMAGE_NAME}" \
  python /workspace/repo/scripts/run_eval_submission.py \
    --task "${TASK}" \
    --submission /workspace/final_submission \
    --out /workspace/eval/results.json \
    --repo-root /workspace/repo
